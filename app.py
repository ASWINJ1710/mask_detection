"""Streamlit dashboard for real-time face mask detection.

Run from the project root:
    streamlit run app.py
"""
import time

import streamlit as st
from streamlit_webrtc import webrtc_streamer

from src.video_processor import MaskVideoProcessor, get_models

st.set_page_config(page_title="Face Mask Detection", layout="wide")
st.title("Real-Time Face Mask Detection")
st.caption("Stage 1: Haar Cascade finds faces. Stage 2: MobileNetV2 classifies each face.")

with st.spinner("Loading models (first start takes a few seconds)..."):
    get_models()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold", 0.10, 0.90, 0.50, 0.05,
        help="A face is labelled No Mask when the model's no-mask probability "
             "is at or above this value. Lower = flags more faces as No Mask.",
    )
    st.markdown("**Green box** = mask  \n**Red box** = no mask")
    st.caption("Frontal, well-lit faces work best. Haar can miss turned or covered faces.")

left, right = st.columns([3, 1])
with left:
    ctx = webrtc_streamer(
        key="face-mask",
        video_processor_factory=MaskVideoProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        async_processing=True,
    )
with right:
    st.subheader("Live stats")
    ph_faces = st.empty()
    ph_mask = st.empty()
    ph_nomask = st.empty()
    ph_fps = st.empty()


def show(stats):
    ph_faces.metric("Faces in frame", stats["faces"])
    ph_mask.metric("With mask", stats["with_mask"])
    ph_nomask.metric("Without mask", stats["without_mask"])
    ph_fps.metric("FPS", stats["fps"])


show({"faces": 0, "with_mask": 0, "without_mask": 0, "fps": 0.0})

while ctx.state.playing:
    if ctx.video_processor:
        ctx.video_processor.set_threshold(threshold)
        show(ctx.video_processor.get_stats())
    time.sleep(0.5)

