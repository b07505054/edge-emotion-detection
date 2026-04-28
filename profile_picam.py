import cv2
import numpy as np
import time
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

IMG_SIZE = 96
MODEL_PATH = "models/emotion_int8.tflite"
N_RUNS = 100 

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)

capture_times = []
preprocess_times = []
inference_times = []
total_times = []

def preprocess_full_frame(frame):
    if frame.shape[-1] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    t_pre0 = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32)
    face = np.stack([face, face, face], axis=-1)
    face = (face / 127.5) - 1.0
    face = np.expand_dims(face, axis=0)

    if input_details[0]["dtype"] == np.uint8:
        scale, zero_point = input_details[0]["quantization"]
        face = face / scale + zero_point
        face = np.clip(face, 0, 255).astype(np.uint8)
    else:
        face = face.astype(input_details[0]["dtype"])

    t_pre1 = time.time()
    return face, (t_pre1 - t_pre0) * 1000

for i in range(N_RUNS):
    t0 = time.time()

    frame = picam2.capture_array()
    t1 = time.time()

    inp, preprocess_ms = preprocess_full_frame(frame)
    t2 = time.time()

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]["index"])
    t3 = time.time()

    capture_ms = (t1 - t0) * 1000
    infer_ms = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000

    capture_times.append(capture_ms)
    preprocess_times.append(preprocess_ms)
    inference_times.append(infer_ms)
    total_times.append(total_ms)

    print(
        f"[{i:03d}] "
        f"capture: {capture_ms:.2f} ms | "
        f"preprocess: {preprocess_ms:.2f} ms | "
        f"infer: {infer_ms:.2f} ms | "
        f"total: {total_ms:.2f} ms"
    )

picam2.stop()

def summary(name, values):
    arr = np.array(values)
    print(f"{name}: avg={arr.mean():.2f} ms | p50={np.percentile(arr, 50):.2f} ms | p95={np.percentile(arr, 95):.2f} ms")

print("\n=== Latency Summary ===")
summary("capture", capture_times)
summary("preprocess", preprocess_times)
summary("inference", inference_times)
summary("total", total_times)

avg_total = np.mean(total_times)
fps = 1000.0 / avg_total
print(f"Estimated FPS: {fps:.2f}")
