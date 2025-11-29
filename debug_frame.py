import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print("ret:", ret)
if ret:
    print("shape:", frame.shape, "min/max/mean:", frame.min(), frame.max(), frame.mean())
    cv2.imwrite("debug_frame.jpg", frame)
    print("Saved debug_frame.jpg")
else:
    print("No frame read. Try other camera indexes (1,2).")
cap.release()