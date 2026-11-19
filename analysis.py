from datetime import date
from io import BytesIO
from pathlib import Path

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ultralytics import YOLO

from camera import render_photo_input
from classroom import render_class_details
from database import save_attendance, saved_attendance
from face import (
    confirm_sleeping_for_two_frames,
    detection_box_iou,
    load_face_detector,
    match_people_to_known_faces,
    person_detection_box,
    prepare_person_face_candidates,
    sleeping_posture_flags,
    split_people_by_faces,
)
from history import read_history, render_history, save_analysis
from voice import activity_voice_messages, queue_voice_alerts

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "yolov8m.pt"
DEFAULT_DETECTION_CONFIDENCE = 0.35
MIN_PERSON_DETECTION_CONFIDENCE = 0.35
ROSTER_PATH = APP_DIR / "attendance.xlsx"
ATTENDANCE_STATUSES = ["Unmarked", "Present", "Absent"]

@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)

def load_roster(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Attendance roster not found: {path}")
    workbook = pd.read_excel(
        path,
        dtype=str,
        keep_default_na=False,
        header=None,
    )
    if workbook.empty or workbook.shape[1] < 2:
        raise ValueError("The roster must contain at least two columns.")

    header_lookup = {
        "".join(character for character in str(value).lower() if character.isalnum()): index
        for index, value in enumerate(workbook.iloc[0])
    }
    roll_position = next(
        (
            header_lookup[key]
            for key in ("roll", "rollno", "rollnumber", "studentid")
            if key in header_lookup
        ),
        None,
    )
    name_position = next(
        (
            header_lookup[key]
            for key in ("name", "studentname")
            if key in header_lookup
        ),
        None,
    )
    has_header = roll_position is not None or name_position is not None
    if has_header:
        roll_position = 0 if roll_position is None else roll_position
        if name_position is None or name_position == roll_position:
            name_position = 1 if roll_position != 1 else 0
        workbook = workbook.iloc[1:]
    else:
        roll_position, name_position = 0, 1

    roster = workbook.iloc[:, [roll_position, name_position]].copy()
    roster.columns = ["Roll", "Name"]
    roster["Roll"] = roster["Roll"].astype(str).str.strip()
    roster["Name"] = roster["Name"].astype(str).str.strip()
    roster = roster[(roster["Roll"] != "") & (roster["Name"] != "")]
    roster = roster.drop_duplicates(subset="Roll", keep="first").reset_index(drop=True)
    return roster

def fallback_roster() -> pd.DataFrame:
    rolls = [str(roll) for roll in range(1, 61)]
    return pd.DataFrame(
        {
            "Roll": rolls,
            "Name": [f"Student {roll}" for roll in rolls],
        }
    )

def make_attendance_excel(attendance: pd.DataFrame, class_name: str, attendance_date: date) -> bytes:
    report = attendance.rename(columns={"Roll": "Roll No"}).copy()
    report.insert(0, "Date", attendance_date.isoformat())
    report.insert(1, "Class", class_name)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        report.to_excel(writer, index=False, sheet_name="Attendance")
    return output.getvalue()

def people_are_close(first_box, second_box) -> bool:
    first_height = max(first_box[3] - first_box[1], 1)
    second_height = max(second_box[3] - second_box[1], 1)
    vertical_overlap = max(
        0,
        min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]),
    )
    overlap_ratio = vertical_overlap / min(first_height, second_height)
    horizontal_gap = max(
        first_box[0] - second_box[2],
        second_box[0] - first_box[2],
        0,
    )
    return (
        overlap_ratio >= 0.5
        and horizontal_gap <= max(12, min(first_height, second_height) * 0.15)
    )

def nearest_phone_user_indices(people, phone_boxes):
    phone_users = set()
    for phone_box in phone_boxes:
        phone_center_x = (phone_box[0] + phone_box[2]) / 2
        phone_center_y = (phone_box[1] + phone_box[3]) / 2
        nearest_person = min(
            range(len(people)),
            key=lambda index: (
                (people[index]["box"][0] + people[index]["box"][2]) / 2
                - phone_center_x
            )
            ** 2
            + (
                (people[index]["box"][1] + people[index]["box"][3]) / 2
                - phone_center_y
            )
            ** 2,
        )
        phone_users.add(nearest_person)
    return phone_users

def suppress_duplicate_people(people, iou_threshold: float = 0.5):
    ranked_people = sorted(
        enumerate(people),
        key=lambda item: item[1]["confidence"],
        reverse=True,
    )
    kept_indices = set()
    while ranked_people:
        original_index, person = ranked_people.pop(0)
        kept_indices.add(original_index)
        remaining = []
        for candidate_index, candidate in ranked_people:
            overlap = detection_box_iou(
                person_detection_box(person),
                person_detection_box(candidate),
            )
            if overlap <= iou_threshold:
                remaining.append((candidate_index, candidate))
        ranked_people = remaining
    return [people[index] for index in sorted(kept_indices)]

def deduplicate_matched_people(people):
    best_by_name = {}
    unknown_people = []
    ranked_people = sorted(
        enumerate(people),
        key=lambda item: item[1]["confidence"],
        reverse=True,
    )
    for index, person in ranked_people:
        name = str(person.get("name") or "").strip()
        if not name or name.casefold() == "unknown student":
            unknown_people.append(person)
            continue
        key = name.casefold()
        candidates = best_by_name.setdefault(key, [])
        if any(
            detection_box_iou(
                person_detection_box(person),
                person_detection_box(existing_person),
            )
            > 0.3
            for _, existing_person in candidates
        ):
            continue
        candidates.append((index, person))

    kept_unknown = suppress_duplicate_people(unknown_people, iou_threshold=0.3)
    original_indices = {id(person): index for index, person in enumerate(people)}
    retained = [
        person
        for candidates in best_by_name.values()
        for _, person in candidates
    ]
    retained.extend(kept_unknown)
    return sorted(retained, key=lambda person: original_indices[id(person)])

def analyze_image(
    image_bgr,
    confidence: float = DEFAULT_DETECTION_CONFIDENCE,
    sleeping_tracks=None,
):
    model = load_model(str(MODEL_PATH))
    prediction = model.predict(
        image_bgr,
        conf=max(float(confidence), MIN_PERSON_DETECTION_CONFIDENCE),
        iou=0.45,
        imgsz=1280,
        classes=[0, 63, 67, 73],
        verbose=False,
    )[0]
    people = []
    detected_objects = {
        "cell phone": [],
        "book": [],
        "laptop": [],
    }
    for box in prediction.boxes or []:
        class_id = int(box.cls[0].item())
        class_name = model.names.get(class_id, str(class_id))
        coordinates = [int(value) for value in box.xyxy[0].tolist()]
        if class_name == "person":
            people.append(
                {
                    "box": coordinates,
                    "confidence": float(box.conf[0].item()),
                    "activity": None,
                }
            )
        elif class_name in detected_objects:
            detected_objects[class_name].append(coordinates)

    if not people:
        if sleeping_tracks is not None:
            sleeping_tracks.clear()
        return {
            "annotated": image_bgr.copy(),
            "total_detected": 0,
            "attentive": 0,
            "sleeping": 0,
            "phone_use": 0,
            "talking": 0,
            "students": [],
        }

    face_detector = load_face_detector()
    if face_detector is not None:
        prepare_person_face_candidates(image_bgr, people, face_detector)
    else:
        for person in people:
            person["face_candidates"] = []
    people = split_people_by_faces(people)
    people = suppress_duplicate_people(people)
    match_people_to_known_faces(people)
    people = deduplicate_matched_people(people)
    has_sleep_blocking_object = any(
        detected_objects[object_name]
        for object_name in ("cell phone", "laptop", "book")
    )
    sleeping_candidates = (
        [False] * len(people)
        if has_sleep_blocking_object
        else sleeping_posture_flags(image_bgr, people)
    )
    sleeping_flags = confirm_sleeping_for_two_frames(
        people,
        sleeping_candidates,
        sleeping_tracks,
    )
    phone_users = nearest_phone_user_indices(
        people,
        detected_objects["cell phone"],
    )

    talking_people = set()
    for first_index, first_person in enumerate(people):
        for second_index in range(first_index + 1, len(people)):
            if people_are_close(first_person["box"], people[second_index]["box"]):
                talking_people.update((first_index, second_index))

    for index, (person, sleeping) in enumerate(zip(people, sleeping_flags)):
        if index in phone_users:
            person["activity"] = "Using Phone"
        elif sleeping:
            person["activity"] = "Sleeping"
        elif index in talking_people:
            person["activity"] = "Talking"
        else:
            person["activity"] = "Attentive / Listening"

    annotated = image_bgr.copy()
    activity_colors = {
        "Attentive / Listening": (0, 255, 0),
        "Sleeping": (0, 0, 255),
        "Using Phone": (0, 255, 255),
        "Talking": (255, 0, 0),
    }
    drawn_person_boxes = set()
    for person in people:
        x1, y1, x2, y2 = person["box"]
        color = activity_colors[person["activity"]]
        person_box = tuple(person["box"])
        if person_box not in drawn_person_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            drawn_person_boxes.add(person_box)
        display_name = person["name"]
        if person["match_confidence"] is not None:
            display_name = (
                f"{display_name} ({person['match_confidence']:.0%})"
            )
        if person.get("face_box"):
            face_x1, face_y1, face_x2, face_y2 = person["face_box"]
            cv2.rectangle(
                annotated,
                (face_x1, face_y1),
                (face_x2, face_y2),
                (0, 255, 0),
                2,
            )
            label_x = face_x1
            label_y = max(face_y1 - 8, 18)
        else:
            label_x = x1
            label_y = max(y1 - 8, 18)
        cv2.putText(
            annotated,
            f"{display_name} - {person['activity']}",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    student_rows = [
        {
            "Student No": person_number,
            "Student Name": (
                f"{person['name']} ({person['match_confidence']:.0%})"
                if person["match_confidence"] is not None
                else person["name"]
            ),
            "Match Scores": person.get("match_scores", []),
            "Match Metric": person.get("match_metric"),
            "Match Threshold": person.get("match_threshold"),
            "Activity": person["activity"],
            "Confidence": f"{person['confidence']:.1%}",
        }
        for person_number, person in enumerate(people, start=1)
    ]
    return {
        "annotated": annotated,
        "total_detected": len(people),
        "attentive": sum(
            person["activity"] == "Attentive / Listening" for person in people
        ),
        "sleeping": sum(person["activity"] == "Sleeping" for person in people),
        "phone_use": sum(person["activity"] == "Using Phone" for person in people),
        "talking": sum(person["activity"] == "Talking" for person in people),
        "students": student_rows,
    }

def analyze_and_save(
    image_bgr,
    class_name: str,
    mode: str,
    confidence: float = DEFAULT_DETECTION_CONFIDENCE,
):
    result = analyze_image(image_bgr, confidence)
    st.session_state["last_image"] = image_bgr.copy()
    st.session_state["last_result"] = result
    st.session_state["last_analysis"] = result
    st.session_state["last_analysis_class"] = class_name
    st.session_state["last_analysis_mode"] = mode
    save_analysis(class_name, mode, result)
    queue_voice_alerts(activity_voice_messages(result), browser=True)
    return result

def show_analysis(result: dict):
    if result["total_detected"] == 0:
        st.warning(
            "No person detected in this image. Please upload a photo with students."
        )
        return

    st.caption(
        "Activities are heuristic estimates based on YOLOv8 detections and bounding-box geometry. "
        "Talking is estimated from nearby person boxes; facing direction cannot be verified."
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric("Total Detected", result["total_detected"])
    metric_columns[1].metric("Attentive Count", result["attentive"])
    metric_columns[2].metric("Sleeping Count", result["sleeping"])
    metric_columns[3].metric("Phone Use Count", result["phone_use"])
    metric_columns[4].metric("Talking Count", result["talking"])
    st.image(
        cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB),
        caption="Detected people and estimated activities",
        use_container_width=True,
    )
    st.caption(
        "Student numbers are assigned in detection order and do not identify roster records."
    )
    status_dots = {
        "Attentive / Listening": "🟢",
        "Sleeping": "🔴",
        "Using Phone": "🟡",
        "Talking": "🔵",
    }
    activity_table = pd.DataFrame(
        result["students"],
        columns=["Student No", "Student Name", "Activity", "Confidence"],
    )
    activity_table["Status (color dot)"] = activity_table["Activity"].map(status_dots)
    st.subheader("Student Activity")
    st.dataframe(
        activity_table,
        use_container_width=True,
        hide_index=True,
    )
    unknown_students = [
        student
        for student in result["students"]
        if student["Student Name"] == "Unknown Student"
    ]
    with st.sidebar.expander("Face match debug", expanded=True):
        if not unknown_students:
            st.caption("No unknown faces in this analysis.")
        for student in unknown_students:
            ranked_scores = student.get("Match Scores", [])
            if ranked_scores:
                closest = ", ".join(
                    f"{item['name']} {item['score']:.2f}"
                    for item in ranked_scores
                )
                metric = student.get("Match Metric") or "face similarity"
                threshold = student.get("Match Threshold")
                threshold_text = (
                    f", threshold {threshold:.2f}"
                    if threshold is not None
                    else ""
                )
                st.write(
                    f"Unknown face {student['Student No']} "
                    f"({metric}{threshold_text}): Closest: {closest}"
                )
            else:
                st.write(
                    f"Unknown face {student['Student No']}: "
                    "No registered-face scores available."
                )
    st.info(
        "The analysis summary has been saved to history. No photos or videos are stored in the database."
    )

def render_dashboard(selected_class, class_name: str):
    st.title("Dashboard")
    st.subheader("Class Information")
    recent = read_history(class_name=class_name, limit=1)
    latest = None if recent.empty else recent.iloc[0].to_dict()
    present_count = int(latest["total_detected"]) if latest is not None else 0
    total_students = int(selected_class["total_students"])
    absent_count = total_students - present_count
    class_attendance_percent = (
        present_count / total_students * 100 if total_students else 0
    )
    class_info_columns = st.columns(5)
    class_info_columns[0].metric("Class Name", class_name)
    class_info_columns[1].metric("Total Students", total_students)
    present_metric = class_info_columns[2].empty()
    present_metric.metric("Present", present_count)
    absent_metric = class_info_columns[3].empty()
    absent_metric.metric("Absent", absent_count)
    attendance_metric = class_info_columns[4].empty()
    attendance_metric.metric(
        "Attendance %",
        f"{class_attendance_percent:.1f}%",
    )
    st.caption(
        f"Teacher: {selected_class['teacher_name']} · Room: {selected_class['room_no']} · "
        f"Subject: {selected_class['subject']}"
    )
    st.caption(f"Class: {class_name}")
    summary_placeholder = st.empty()
    charts_placeholder = st.empty()
    detection_confidence = max(
        MIN_PERSON_DETECTION_CONFIDENCE,
        float(
            st.session_state.get(
                "detection_confidence", DEFAULT_DETECTION_CONFIDENCE
            )
        ),
    )
    input_mode = st.radio(
        "Input source",
        ["Take Photo", "Photo Upload", "Video Upload"],
        horizontal=True,
    )
    expected_analysis_mode = {
        "Take Photo": "Live",
        "Photo Upload": "Photo",
        "Video Upload": "Video",
    }.get(input_mode)
    result = None
    if (
        expected_analysis_mode is not None
        and st.session_state.get("last_analysis_class") == class_name
        and st.session_state.get("last_analysis_mode") == expected_analysis_mode
    ):
        result = st.session_state.get(
            "last_analysis",
            st.session_state.get("last_result"),
        )
    new_result = render_photo_input(
        input_mode,
        class_name,
        detection_confidence,
        analyze_and_save,
    )
    if new_result is not None:
        result = new_result

    if result is None:
        if latest is None:
            summary = {
                "total_detected": 0,
                "attentive": 0,
                "sleeping": 0,
                "phone_use": 0,
                "talking": 0,
            }
        else:
            summary = latest
    else:
        summary = result

    total_detected = int(summary["total_detected"])
    present_count = total_detected
    absent_count = total_students - present_count
    class_attendance_percent = (
        present_count / total_students * 100 if total_students else 0
    )
    present_metric.metric("Present", present_count)
    absent_metric.metric("Absent", absent_count)
    attendance_metric.metric(
        "Attendance %",
        f"{class_attendance_percent:.1f}%",
    )
    attentive_count = int(summary["attentive"])
    sleeping_count = int(summary["sleeping"])
    phone_use_count = int(summary["phone_use"])
    talking_count = int(summary["talking"])
    attendance_percent = (
        total_detected / total_students * 100 if total_students else 0
    )
    attentive_percent = (
        attentive_count / total_detected * 100 if total_detected else 0
    )
    alert_count = sleeping_count + phone_use_count

    with summary_placeholder.container():
        metric_columns = st.columns(4)
        metric_columns[0].metric("👥 Total Detected", total_detected)
        metric_columns[1].metric("📊 Attendance %", f"{attendance_percent:.1f}%")
        metric_columns[2].metric("✅ Attentive %", f"{attentive_percent:.1f}%")
        metric_columns[3].metric("🚨 Alert Count", alert_count)
        if phone_use_count > 0 or sleeping_count > 0:
            st.error(
                f"Attention Needed: {phone_use_count} students using phone, "
                f"{sleeping_count} students sleeping"
            )
        if latest is not None:
            st.caption(
                f"Showing latest saved analysis: {latest['created_at']} · {latest['mode']}"
            )

    activity_labels = ["Attentive", "Sleeping", "Phone Use", "Talking"]
    activity_values = [
        attentive_count,
        sleeping_count,
        phone_use_count,
        talking_count,
    ]
    activity_colors = ["#22c55e", "#ef4444", "#facc15", "#3b82f6"]
    with charts_placeholder.container():
        chart_columns = st.columns(2)
        pie_chart = go.Figure(
            data=[
                go.Pie(
                    labels=activity_labels,
                    values=activity_values,
                    hole=0.55,
                    marker={"colors": activity_colors},
                    textinfo="label+value",
                    sort=False,
                )
            ]
        )
        pie_chart.update_layout(
            title="Activity Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=340,
            margin={"t": 55, "b": 10, "l": 10, "r": 10},
            showlegend=True,
        )
        if sum(activity_values) == 0:
            pie_chart.add_annotation(
                text="No activity data yet",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"color": "#b7c5e5"},
            )
        chart_columns[0].plotly_chart(pie_chart, use_container_width=True)

        bar_chart = go.Figure(
            data=[
                go.Bar(
                    x=activity_labels,
                    y=activity_values,
                    marker_color=activity_colors,
                    text=activity_values,
                    textposition="outside",
                    cliponaxis=False,
                )
            ]
        )
        bar_chart.update_layout(
            title="Activity Counts",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=340,
            margin={"t": 55, "b": 50, "l": 10, "r": 10},
            xaxis_title="Activity",
            yaxis_title="Students",
            yaxis={"rangemode": "tozero"},
            showlegend=False,
        )
        chart_columns[1].plotly_chart(bar_chart, use_container_width=True)

    if result is not None:
        show_analysis(result)
    elif latest is None:
        st.info(
            "There is no history for this class yet. Capture or upload a photo to save an analysis."
        )

    render_history("Recent analysis history", class_name=class_name, limit=10)

def render_reports(class_name: str):
    st.title("Reports")
    st.write(
        "The attendance roster is read from `attendance.xlsx`. Confirm each student's status manually."
    )
    st.warning(
        "YOLO and OpenCV do not identify a student's name or roll number. "
        "Attendance is not inferred from photo detections."
    )
    attendance_date = st.date_input("Attendance date", value=date.today())
    try:
        roster = load_roster(ROSTER_PATH)
    except (FileNotFoundError, ValueError, OSError, ImportError) as error:
        st.info(
            f"The roster could not be read ({error}). Showing placeholder roll numbers 1–60."
        )
        roster = fallback_roster()
    if roster.empty:
        st.info("No student rows were found. Showing placeholder roll numbers 1–60.")
        roster = fallback_roster()

    if roster is not None:
        saved = saved_attendance(class_name, attendance_date)
        attendance = roster.merge(saved, on="Roll", how="left")
        attendance["Status"] = attendance["Status"].fillna("Unmarked")
        with st.form("attendance_report_form"):
            edited_attendance = st.data_editor(
                attendance,
                column_config={
                    "Roll": st.column_config.TextColumn("Roll No", disabled=True),
                    "Name": st.column_config.TextColumn("Name", disabled=True),
                    "Status": st.column_config.SelectboxColumn(
                        "Status",
                        options=ATTENDANCE_STATUSES,
                        help="Manually mark each student as Present or Absent.",
                    ),
                },
                disabled=["Roll", "Name"],
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
            )
            submitted = st.form_submit_button("Save attendance")
        if submitted:
            if edited_attendance["Status"].isin(["Present", "Absent"]).all():
                save_attendance(class_name, attendance_date, edited_attendance)
                st.success("Attendance saved.")
            else:
                st.warning(
                    "Each student must be marked Present or Absent before you can download the Excel report."
                )

        saved = saved_attendance(class_name, attendance_date)
        complete_report = roster.merge(saved, on="Roll", how="left")
        complete_report["Status"] = complete_report["Status"].fillna("Unmarked")
        if not complete_report.empty and complete_report["Status"].isin(["Present", "Absent"]).all():
            excel_data = make_attendance_excel(complete_report, class_name, attendance_date)
            st.download_button(
                "Download Excel report",
                data=excel_data,
                file_name=f"attendance_{attendance_date.isoformat()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info(
                "Save the attendance to download a complete Present/Absent Excel report."
            )

    render_history("Analysis history for this class", class_name=class_name, limit=20)

def render_settings(class_name: str):
    st.title("Settings")
    st.write("Add classes using the **New Class** form in the sidebar.")
    if (
        "detection_confidence_slider" in st.session_state
        and st.session_state["detection_confidence_slider"]
        < MIN_PERSON_DETECTION_CONFIDENCE
    ):
        st.session_state["detection_confidence_slider"] = (
            MIN_PERSON_DETECTION_CONFIDENCE
        )
    detection_confidence = st.slider(
        "Detection Confidence",
        min_value=MIN_PERSON_DETECTION_CONFIDENCE,
        max_value=0.8,
        value=max(
            MIN_PERSON_DETECTION_CONFIDENCE,
            float(
                st.session_state.get(
                    "detection_confidence", DEFAULT_DETECTION_CONFIDENCE
                )
            )
        ),
        step=0.05,
        key="detection_confidence_slider",
        help="Lower values may detect more people but can also increase false detections.",
    )
    st.session_state["detection_confidence"] = detection_confidence
    render_class_details()
    render_history("Shared analysis history", class_name=class_name, limit=20)
