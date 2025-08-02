import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

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
elif mode == "Upload Video" or mode == "Use Sample Video":
    st.subheader("Upload a Video for Detection")

    if mode == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
        else:
            st.stop()
    elif mode == "Use Sample Video":
        video_path = "sample_video.mp4"  # Make sure this file exists in the same directory
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        stframe = st.empty()
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=conf_threshold, verbose=False)
            annotated = results[0].plot()

            stframe.image(annotated, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Finished processing video.")

elif mode == "Use Sample Video":
    st.subheader("Using Sample Video for Detection")

    # Replace this with your actual sample video path
    sample_video_path = "sample_video.mp4"

    stframe = st.empty()
    cap = cv2.VideoCapture(sample_video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()

        stframe.image(annotated, channels="BGR", use_container_width=True)

    cap.release()
    st.success("Finished processing sample video.")

