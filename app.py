
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np
import time

model = YOLO("best.pt")

st.title("YOLOv8 Detection App")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

mode = st.sidebar.radio("Select Mode", ["Webcam", "Upload Video", "Use Sample Video"])

if mode == "Webcam":
    st.subheader("Live Webcam Detection")

    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(img, conf=conf_threshold, verbose=False)
            annotated = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif mode in ["Upload Video", "Use Sample Video"]:
    st.subheader("Video Detection")

    if mode == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video is None:
            st.stop()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

    else:  # Use Sample Video
        video_path = "sample_video.mp4"  # Ensure this exists in the same directory

    if st.button("▶️ Start Detection"):
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=conf_threshold, verbose=False)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR", use_container_width=True)

            time.sleep(1 / 30.0)  # simulate ~30 FPS

        cap.release()
        st.success("Finished processing video.")
