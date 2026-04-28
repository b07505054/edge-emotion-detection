t0 = time.time()

frame = picam2.capture_array()
t1 = time.time()

# preprocess
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
t2 = time.time()

# inference
interpreter.set_tensor(input_details[0]["index"], inp)
interpreter.invoke()
t3 = time.time()

capture_ms = (t1 - t0) * 1000
preprocess_ms = (t2 - t1) * 1000
infer_ms = (t3 - t2) * 1000
total_ms = (t3 - t0) * 1000
print(f"capture: {capture_ms:.2f} ms | preprocess: {preprocess_ms:.2f} ms | infer: {infer_ms:.2f} ms | total: {total_ms:.2f} ms")
