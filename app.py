import cv2
import time
from ultralytics import YOLO
import winsound  # for alert (Windows)

# Load model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# FPS
prev_time = 0

# Counting line position
line_y = 300

# Store counted IDs
counted_ids = set()
total_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Detection + Tracking
    results = model.track(frame, persist=True)

    annotated_frame = results[0].plot()

    # Draw counting line
    cv2.line(annotated_frame, (0, line_y), (640, line_y), (0, 255, 255), 2)

    # Get boxes + IDs
    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:
        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)

            # Object center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw center
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

            # Check line crossing
            if cy > line_y and track_id not in counted_ids:
                counted_ids.add(track_id)
                total_count += 1

                # 🔔 Alert sound
                winsound.Beep(1000, 300)

                print(f"Object Counted! ID: {track_id}")

    # Show count
    cv2.putText(annotated_frame, f'Count: {total_count}', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save video
    out.write(annotated_frame)

    # Show output
    cv2.imshow("AI Object Detection + Tracking + Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()