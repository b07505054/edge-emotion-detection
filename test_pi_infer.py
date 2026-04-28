import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

IMG_SIZE = 96
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

MODEL_PATH = "models/emotion_fp32.tflite"
IMAGE_PATH = "test.jpg"

print("RUNNING NEW SCRIPT")
print("MODEL_PATH =", MODEL_PATH)
print("IMAGE_PATH =", IMAGE_PATH)

def preprocess(img_path):
    img = Image.open(img_path)
    print("original mode:", img.mode)
    img = img.convert("L")
    print("after convert:", img.mode)

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32)

    print("min/max before stack:", img.min(), img.max())

    img = np.stack([img, img, img], axis=-1)
    img = (img / 127.5) - 1.0

    print("shape:", img.shape)
    print("min/max after normalize:", img.min(), img.max())

    return np.expand_dims(img, axis=0)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

x = preprocess(IMAGE_PATH)

print("input dtype:", input_details[0]["dtype"])
print("output dtype:", output_details[0]["dtype"])

if input_details[0]["dtype"] == np.uint8:
    scale, zero_point = input_details[0]["quantization"]
    print("input quant:", scale, zero_point)
    x = x / scale + zero_point
    x = np.clip(x, 0, 255).astype(np.uint8)

interpreter.set_tensor(input_details[0]["index"], x)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])
print("raw output:", output)

if output_details[0]["dtype"] == np.uint8:
    scale, zero_point = output_details[0]["quantization"]
    print("output quant:", scale, zero_point)
    output = scale * (output.astype(np.float32) - zero_point)

print("final output:", output)

pred = int(np.argmax(output[0]))
conf = float(np.max(output[0]))

top3_idx = np.argsort(output[0])[::-1][:3]
print("top3:")
for i in top3_idx:
    print(CLASS_NAMES[int(i)], float(output[0][i]))

print("Prediction:", CLASS_NAMES[pred])
print("Confidence:", round(conf, 4))
