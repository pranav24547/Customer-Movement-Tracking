from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


model = YOLO("yolov8m.pt")  


tracker = DeepSort(max_age=70)

cap = cv2.VideoCapture("data/store2.mp4")

if not cap.isOpened():
    print("❌ Error: Could not open video file. Check path!")
    exit()


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("outputs/tracked_output.mp4", fourcc, fps, (width, height))


track_history = {}
MAX_TRAIL_LENGTH = 50
COLORS = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)

print("🚀 Starting person detection and tracking... Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing completed.")
        break

    
    results = model(frame, classes=0) 
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf[0])
        cls = int(box.cls[0]) 

        if conf > 0.5: 
            detections.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], conf))

 
    tracks = tracker.update_tracks(detections, frame=frame)

   
    for t in tracks:
        if not t.is_confirmed():
            continue

        track_id = t.track_id
        try:
            l, t_, r, b = map(int, t.to_ltrb())
        except Exception:
            continue

       
        l, t_, r, b = max(l, 0), max(t_, 0), min(r, width - 1), min(b, height - 1)
        cx, cy = (l + r) // 2, (t_ + b) // 2

        color_index = int(track_id) % len(COLORS)
        color = tuple(int(c) for c in COLORS[color_index])

      
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > MAX_TRAIL_LENGTH:
            track_history[track_id].pop(0)

       
        for i in range(1, len(track_history[track_id])):
            cv2.line(frame, track_history[track_id][i - 1],
                     track_history[track_id][i], color, 2)
            cv2.circle(frame, track_history[track_id][i], 3, color, -1)

    
        cv2.rectangle(frame, (l, t_), (r, b), color, 2)
        cv2.putText(frame, f"ID {track_id}", (l, t_ - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    
    cv2.imshow("CPI - Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("🛑 ESC pressed. Exiting...")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("🎥 Output video saved to: outputs/tracked_output.mp4")
