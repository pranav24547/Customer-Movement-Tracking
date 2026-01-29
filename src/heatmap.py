import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolov8m.pt")



cap = cv2.VideoCapture("data/store2.mp4")

if not cap.isOpened():
    print("❌ Error: Could not open video file!")
    exit()


ret, frame = cap.read()
if not ret:
    print("❌ Couldn't read first frame!")
    exit()

height, width = frame.shape[:2]
heatmap = np.zeros((height, width), np.float32)


HEATMAP_DECAY = 0.02
GAUSS_KERNEL = (51, 51)
BLUR_SIGMA = 10


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("outputs/heatmap_output.mp4", fourcc,
                      cap.get(cv2.CAP_PROP_FPS), (width, height))

print("🔥 Generating customer movement heatmap... Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete.")
        break


    heatmap = np.maximum(heatmap - HEATMAP_DECAY, 0)


    results = model(frame)
    temp = np.zeros_like(heatmap)

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:  # person only
            x1, y1, x2, y2 = box.xyxy[0]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(temp, (cx, cy), 15, 1.0, -1)


    blurred = cv2.GaussianBlur(temp, GAUSS_KERNEL, BLUR_SIGMA)
    heatmap = np.clip(heatmap + blurred, 0, 10)

   
    norm_map = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    norm_map = norm_map.astype(np.uint8)
    colored = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

   
    output = cv2.addWeighted(frame, 0.6, colored, 0.4, 0)

    cv2.imshow("CPI - Heatmap", output)
    out.write(output)

    if cv2.waitKey(1) & 0xFF == 27:
        print("🛑 ESC pressed. Exiting...")
        break


cv2.imwrite("outputs/final_heatmap.jpg", colored)
cap.release()
out.release()
cv2.destroyAllWindows()
print("📸 Heatmap saved to: outputs/final_heatmap.jpg")