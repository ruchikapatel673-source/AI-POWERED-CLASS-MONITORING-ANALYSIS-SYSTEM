
import streamlit as st
from PIL import UnidentifiedImageError

from analysis import render_dashboard, render_reports, render_settings
from camera import decode_image
from classroom import render_class_sidebar
from database import initialize_database
from face import (
    YUNET_MODEL_PATH,
    HaarCascadeFaceDetector,
    load_face_detector,
    registered_face_photo_counts,
    save_additional_face,
    save_known_face,
    save_re_registered_faces,
)
from history import render_analytics
from voice import render_voice_sidebar

st.set_page_config(page_title="ClassMonitor AI", layout="wide")














initialize_database()
with st.sidebar:
    if not YUNET_MODEL_PATH.is_file() or YUNET_MODEL_PATH.stat().st_size < 100_000:
        with st.spinner("Downloading face model, please wait..."):
            face_detector = load_face_detector()
    else:
        face_detector = load_face_detector()
    if isinstance(face_detector, HaarCascadeFaceDetector):
        st.info("YuNet is unavailable; using OpenCV Haar Cascade for face detection.")
    elif face_detector is None:
        st.warning(
            "Face detection is unavailable; Save Face will store the uploaded photo without detection."
        )

    st.markdown("## ClassMonitor AI")
    render_voice_sidebar()

    with st.expander("Register Face"):
        face_photo = st.file_uploader(
            "Upload one face photo",
            type=["jpg", "jpeg", "png"],
            key="register_face_photo",
        )
        face_name = st.text_input("Name", key="register_face_name")
        st.caption("Face photos remain on this computer in the known_faces folder.")
        if st.button("Save Face", key="save_face"):
            if face_photo is None:
                st.warning("Upload a face photo before saving.")
            elif not face_name.strip():
                st.warning("Enter a name before saving the face.")
            else:
                try:
                    with st.spinner("Preparing face detector..."):
                        saved_name = save_known_face(
                            face_name,
                            decode_image(face_photo.getvalue()),
                        )
                except (RuntimeError, ValueError, UnidentifiedImageError, OSError) as error:
                    st.error(f"Could not save the face: {error}")
                else:
                    st.success(f"Face saved for {saved_name}.")
        face_photo_counts = registered_face_photo_counts()
        st.caption(f"Registered faces: {len(face_photo_counts)}")
        for registered_name, photo_count in face_photo_counts:
            st.caption(
                f"{registered_name} - {photo_count} "
                f"{'photo' if photo_count == 1 else 'photos'}"
            )

    with st.expander("Add More Photo for Same Student"):
        additional_face_name = st.text_input(
            "Student name",
            key="additional_face_name",
        )
        additional_face_photo = st.file_uploader(
            "Upload one additional face photo",
            type=["jpg", "jpeg", "png"],
            key="additional_face_photo",
        )
        if st.button(
            "Add More Photo for Same Student",
            key="add_additional_face",
        ):
            if not additional_face_name.strip():
                st.warning("Enter the student's registered name.")
            elif additional_face_photo is None:
                st.warning("Upload a face photo before adding it.")
            else:
                try:
                    with st.spinner("Checking and adding face photo..."):
                        saved_name = save_additional_face(
                            additional_face_name,
                            decode_image(additional_face_photo.getvalue()),
                        )
                except (
                    RuntimeError,
                    ValueError,
                    UnidentifiedImageError,
                    OSError,
                ) as error:
                    st.error(f"Could not add the face photo: {error}")
                else:
                    st.success(
                        f"Added a face photo for {saved_name}; existing photos were kept."
                    )

    with st.expander("Re-Register Face"):
        st.caption(
            "Upload three clear, front-facing photos of one student. Include photos "
            "with glasses and without glasses when possible."
        )
        re_register_name = st.text_input(
            "Student name",
            key="re_register_face_name",
        )
        re_register_photos = [
            st.file_uploader(
                f"Front photo {index}",
                type=["jpg", "jpeg", "png"],
                key=f"re_register_face_photo_{index}",
            )
            for index in range(1, 4)
        ]
        if st.button("Re-Register Face", key="re_register_face"):
            if not re_register_name.strip():
                st.warning("Enter the student's registered name.")
            elif any(photo is None for photo in re_register_photos):
                st.warning("Upload all three photos before re-registering.")
            else:
                try:
                    with st.spinner("Checking and saving face photos..."):
                        saved_name = save_re_registered_faces(
                            re_register_name,
                            [
                                decode_image(photo.getvalue())
                                for photo in re_register_photos
                            ],
                        )
                except (
                    RuntimeError,
                    ValueError,
                    UnidentifiedImageError,
                    OSError,
                ) as error:
                    st.error(f"Could not re-register the face: {error}")
                else:
                    st.success(
                        f"Added three face photos for {saved_name}; existing photos were kept."
                    )

    page = st.radio("Page", ["Dashboard", "Analytics", "Reports", "Settings"])
    selected_class_name, selected_class = render_class_sidebar()
    st.caption("AI: YOLOv8 detection for people, phones, books, and laptops")

if page == "Dashboard":
    render_dashboard(selected_class, selected_class_name)
elif page == "Analytics":
    render_analytics()
elif page == "Reports":
    render_reports(selected_class_name)
else:
    render_settings(selected_class_name)
