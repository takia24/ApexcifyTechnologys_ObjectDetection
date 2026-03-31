import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Detection System", layout="wide")

st.title("🚀 AI Object Detection + Tracking + Counting System")

# ---------------- STYLE (BIG CAMERA) ----------------
st.markdown("""
<style>
.stImage img {
    border-radius: 12px;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

source = st.sidebar.selectbox("Select Input", ["Image", "Video"])  # Webcam removed for deployment

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

class_option = st.sidebar.selectbox("Filter Class", ["All", "Person", "Car"])

st.sidebar.warning("⚠️ Webcam works only in local environment")

# ---------------- CLASS MAP ----------------
CLASS_MAP = {
    "Person": 0,
    "Car": 2
}

def get_classes():
    if class_option == "All":
        return None
    return [CLASS_MAP[class_option]]

# ---------------- LOAD MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- VARIABLES ----------------
LINE_Y = 300
track_history = {}
count_ids = set()
total_count = 0

prev_time = 0
last_alert_time = 0

# ---------------- ALERT ----------------
def alert():
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time > 3:
        st.toast("🚨 Object Crossed the Line!", icon="🚨")
        last_alert_time = current_time

# ---------------- FRAME (CENTER BIG VIEW) ----------------
col1, col2, col3 = st.columns([1,3,1])
with col2:
    FRAME_WINDOW = st.empty()

# ---------------- IMAGE ----------------
if source == "Image":
    file = st.file_uploader("📸 Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        results = model(frame, conf=confidence, classes=get_classes())
        annotated = results[0].plot()

        FRAME_WINDOW.image(annotated, channels="BGR", use_container_width=True)

# ---------------- VIDEO ----------------
elif source == "Video":
    file = st.file_uploader("🎥 Upload Video", type=["mp4", "avi", "mov"])

    if file:
        with open("temp.mp4", "wb") as f:
            f.write(file.read())

        cap = cv2.VideoCapture("temp.mp4")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-5)
            prev_time = curr_time

            results = model.track(
                frame,
                persist=True,
                conf=confidence,
                classes=get_classes()
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if track_id not in track_history:
                        track_history[track_id] = []

                    track_history[track_id].append((cx, cy))

                    # Line crossing detection
                    if len(track_history[track_id]) > 1:
                        if track_history[track_id][-2][1] < LINE_Y and cy > LINE_Y:
                            if track_id not in count_ids:
                                count_ids.add(track_id)
                                total_count += 1
                                alert()

            # Draw counting line
            cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0,255,255), 2)

            annotated = results[0].plot()

            # Overlay text
            cv2.putText(annotated, f"FPS: {int(fps)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(annotated, f"Count: {total_count}", (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            FRAME_WINDOW.image(annotated, channels="BGR", use_container_width=True)

        cap.release()

# ---------------- DASHBOARD ----------------
st.markdown("---")

col1, col2 = st.columns(2)

col1.metric("📊 Total Objects Counted", total_count)
col2.metric("⚡ System Status", "Running")

# ---------------- FOOTER ----------------
st.caption("🔥 Built with ❤️ using YOLOv8 + Streamlit")