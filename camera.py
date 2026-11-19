from io import BytesIO
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


def decode_image(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_photo_input(input_mode, class_name, confidence, analyze_and_save):
    result = None
    if input_mode == "Take Photo":
        image_file = st.camera_input("Take Photo", key="take_photo_input")
        if image_file and st.button("Analyze & Save Photo", key="analyze_live"):
            try:
                result = analyze_and_save(
                    decode_image(image_file.getvalue()),
                    class_name,
                    "Live",
                    confidence=confidence,
                )
            except RuntimeError as error:
                st.error(str(error))
            except (UnidentifiedImageError, OSError, ValueError) as error:
                st.error(f"The photo could not be processed: {error}")
    elif input_mode == "Photo Upload":
        image_file = st.file_uploader(
            "Upload a class photo",
            type=["jpg", "jpeg", "png"],
            key="photo_upload",
        )
        if image_file and st.button("Analyze & Save Photo", key="analyze_photo"):
            try:
                result = analyze_and_save(
                    decode_image(image_file.getvalue()),
                    class_name,
                    "Photo",
                    confidence=confidence,
                )
            except RuntimeError as error:
                st.error(str(error))
            except (UnidentifiedImageError, OSError, ValueError) as error:
                st.error(f"The photo could not be processed: {error}")
    else:
        video_file = st.file_uploader(
            "Upload a video (MP4, MOV, AVI)",
            type=["mp4", "mov", "avi"],
            key="video_upload",
        )
        if video_file:
            st.video(video_file)
        if video_file and st.button("Analyze first frame & Save", key="analyze_video"):
            with tempfile.TemporaryDirectory() as temporary_directory:
                video_path = (
                    Path(temporary_directory)
                    / f"upload{Path(video_file.name).suffix}"
                )
                video_path.write_bytes(video_file.getvalue())
                capture = cv2.VideoCapture(str(video_path))
                try:
                    if not capture.isOpened():
                        st.error("The uploaded video could not be opened.")
                    else:
                        ok, frame = capture.read()
                        if not ok:
                            st.error("The first frame of the video could not be read.")
                        else:
                            try:
                                result = analyze_and_save(
                                    frame,
                                    class_name,
                                    "Video",
                                    confidence=confidence,
                                )
                            except RuntimeError as error:
                                st.error(str(error))
                finally:
                    capture.release()
    return result
