import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

IMG_SIZE = 96
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = "models/emotion_fp32.tflite"

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("cascade loaded:", not face_cascade.empty())

def preprocess_face(face_gray):
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
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

    return face

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    t0 = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    t1 = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    pred_label = "no-face"
    conf = 0.0
    inf_ms = 0.0

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        face_gray = gray[y:y+h, x:x+w]
        inp = preprocess_face(face_gray)

        t2 = time.time()
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        t3 = time.time()

        if output_details[0]["dtype"] == np.uint8:
            scale, zero_point = output_details[0]["quantization"]
            output = scale * (output.astype(np.float32) - zero_point)

        pred_idx = int(np.argmax(output[0]))
        pred_label = CLASS_NAMES[pred_idx]
        conf = float(np.max(output[0]))
        inf_ms = (t3 - t2) * 1000.0

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{pred_label} {conf:.2f}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    t4 = time.time()
    total_ms = (t4 - t0) * 1000.0
    fps = 1.0 / (t4 - t0) if (t4 - t0) > 0 else 0.0
    capture_ms = (t1 - t0) * 1000.0

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"capture: {capture_ms:.1f} ms", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"infer: {inf_ms:.1f} ms", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"total: {total_ms:.1f} ms", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Edge Emotion Demo", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()