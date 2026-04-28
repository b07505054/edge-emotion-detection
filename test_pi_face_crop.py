import cv2
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

IMG_SIZE = 96
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

MODEL_PATH = "models/emotion_fp32.tflite"
IMAGE_PATH = "test.jpg"

print("MODEL_PATH =", MODEL_PATH)
print("IMAGE_PATH =", IMAGE_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        raise RuntimeError("No face detected")

    # choose largest face
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]
    return face, (x, y, w, h), img_bgr

def preprocess_face(face_gray):
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32)

    # expand grayscale to 3 channels
    face = np.stack([face, face, face], axis=-1)
    face = (face / 127.5) - 1.0
    return np.expand_dims(face, axis=0)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

face_gray, bbox, img_bgr = detect_and_crop_face(IMAGE_PATH)
x, y, w, h = bbox

inp = preprocess_face(face_gray)

if input_details[0]["dtype"] == np.uint8:
    scale, zero_point = input_details[0]["quantization"]
    inp = inp / scale + zero_point
    inp = np.clip(inp, 0, 255).astype(np.uint8)

interpreter.set_tensor(input_details[0]["index"], inp)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])

if output_details[0]["dtype"] == np.uint8:
    scale, zero_point = output_details[0]["quantization"]
    output = scale * (output.astype(np.float32) - zero_point)

pred = int(np.argmax(output[0]))
conf = float(np.max(output[0]))

top3_idx = np.argsort(output[0])[::-1][:3]
print("top3:")
for i in top3_idx:
    print(CLASS_NAMES[int(i)], float(output[0][i]))

print("Prediction:", CLASS_NAMES[pred])
print("Confidence:", round(conf, 4))

# save debug image with bbox
debug = img_bgr.copy()
cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.putText(
    debug,
    f"{CLASS_NAMES[pred]} {conf:.2f}",
    (x, max(y - 10, 20)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)
cv2.imwrite("debug_face_result.jpg", debug)
print("Saved debug_face_result.jpg")