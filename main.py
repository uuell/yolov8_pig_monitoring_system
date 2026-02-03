import cv2
import time
import math
import pandas as pd
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================
# MODEL_PATH = "best.pt"
MODEL_PATH = "best_4_classes.pt"         # your detection model (with growth classes)
# VIDEO_PATH = "test_video.mp4"  # or 0 for webcam
VIDEO_PATH = "test_vid_skin-disease.mp4"  # or 0 for webcam

CONF = 0.3
IOU = 0.4

OUTPUT_CSV = "pig_activity_log_2s.csv"
LOG_INTERVAL = 2.0  # log every 2 seconds

# =========================
# MOVEMENT SETTINGS
# =========================
MOVE_THRESHOLD = 10  # pixels change to count as moved

# =========================
# HELPERS
# =========================
def get_center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# STORAGE
# =========================
pig_state = {}
logs = []
last_log_time = 0

# pig_state format:
# {
#   track_id: {
#       "last_pos": (cx, cy),
#       "last_move_time": timestamp
#   }
# }

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: Could not open video source.")
    exit()

print("✅ Pig Tracking + Growth Label + Activity Timer Running...")
print("Press Q to quit.")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video.")
        break

    current_time = time.time()

    results = model.track(
        frame,
        conf=CONF,
        iou=IOU,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    annotated_frame = frame.copy()

    if results[0].boxes is None:
        cv2.imshow("Pig Monitoring", annotated_frame)
        if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
            break
        continue

    boxes = results[0].boxes
    do_log = (current_time - last_log_time) >= LOG_INTERVAL

    for box in boxes:
        if box.id is None:
            continue

        track_id = int(box.id.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Get detected class (growth label)
        class_id = int(box.cls.item())
        growth_label = model.names[class_id]

        cx, cy = get_center(x1, y1, x2, y2)

        # Init pig state
        if track_id not in pig_state:
            pig_state[track_id] = {
                "last_pos": (cx, cy),
                "last_move_time": current_time
            }

        # Movement check
        last_pos = pig_state[track_id]["last_pos"]
        moved_distance = distance((cx, cy), last_pos)

        if moved_distance > MOVE_THRESHOLD:
            pig_state[track_id]["last_move_time"] = current_time

        # Always update last position
        pig_state[track_id]["last_pos"] = (cx, cy)

        # Time since last movement
        time_since_move = current_time - pig_state[track_id]["last_move_time"]

        # Status (always healthy for now)
        status = "HEALTHY"

        # Logging every 2 seconds
        if do_log:
            logs.append({
                "time": current_time,
                "track_id": track_id,
                "growth_label": growth_label,
                "time_since_move_sec": round(time_since_move, 1),
                "status": status
            })

        # Draw box
        color = (0, 255, 0)
        cv2.rectangle(
            annotated_frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2
        )

        # Draw text
        label1 = f"ID:{track_id} | {growth_label}"
        label2 = f"Inactive: {time_since_move:.0f}s"
        label3 = status

        cv2.putText(annotated_frame, label1, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(annotated_frame, label2, (int(x1), int(y1) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(annotated_frame, label3, (int(x1), int(y1) + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if do_log:
        last_log_time = current_time

    cv2.imshow("Pig Monitoring", annotated_frame)

    if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
        break

cap.release()
cv2.destroyAllWindows()

# Save CSV
if logs:
    df = pd.DataFrame(logs)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved log to: {OUTPUT_CSV}")
    print(f"✅ Total rows saved: {len(df)}")
else:
    print("⚠️ No logs saved.")
