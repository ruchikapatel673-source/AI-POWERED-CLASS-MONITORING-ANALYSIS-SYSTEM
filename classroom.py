import sqlite3

import pandas as pd
import streamlit as st

from database import database_connection

def read_classes():
    with database_connection() as connection:
        return connection.execute(
            """
            SELECT id, name, teacher_name, room_no, subject, total_students
            FROM classes
            ORDER BY name COLLATE NOCASE
            """
        ).fetchall()

def add_class(name: str, teacher: str, room: str, subject: str, students: int):
    with database_connection() as connection:
        connection.execute(
            """
            INSERT INTO classes (name, teacher_name, room_no, subject, total_students)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, teacher, room, subject, students),
        )

def delete_class(class_id: int):
    with database_connection() as connection:
        connection.execute("DELETE FROM classes WHERE id = ?", (class_id,))

def render_class_details():
    classes = read_classes()
    class_details = pd.DataFrame(
        [dict(row) for row in classes],
        columns=["id", "name", "teacher_name", "room_no", "subject", "total_students"],
    ).rename(
        columns={
            "id": "ID",
            "name": "Class Name",
            "teacher_name": "Teacher Name",
            "room_no": "Room No",
            "subject": "Subject",
            "total_students": "Total Students",
        }
    )
    st.subheader("Class details")
    st.dataframe(class_details, use_container_width=True, hide_index=True)
    if len(classes) > 1:
        class_to_delete = st.selectbox(
            "Delete class",
            [row["name"] for row in classes],
            key="class_to_delete",
        )
        if st.button("Delete selected class", type="secondary"):
            selected = next(row for row in classes if row["name"] == class_to_delete)
            delete_class(selected["id"])
            st.success(
                "The class was deleted. Existing analysis history and attendance records were preserved."
            )
            st.rerun()
    else:
        st.info(
            "At least one class is required; the last remaining class cannot be deleted."
        )


def render_class_sidebar():
    class_rows = read_classes()
    class_names = [row["name"] for row in class_rows]
    selected_class_name = st.selectbox(
        "Class",
        class_names,
        key="selected_class_name",
    )
    selected_class = next(row for row in class_rows if row["name"] == selected_class_name)

    with st.expander("➕ New Class"):
        with st.form("new_class_form", clear_on_submit=True):
            new_class_name = st.text_input("Class Name")
            teacher_name = st.text_input("Teacher Name")
            room_no = st.text_input("Room No")
            subject = st.text_input("Subject")
            total_students = st.number_input(
                "Total Students", min_value=0, value=0, step=1
            )
            add_class_submitted = st.form_submit_button("Add Class")
        if add_class_submitted:
            if not all(
                value.strip()
                for value in (new_class_name, teacher_name, room_no, subject)
            ):
                st.error("Class Name, Teacher Name, Room No, and Subject are required.")
            else:
                try:
                    add_class(
                        new_class_name.strip(),
                        teacher_name.strip(),
                        room_no.strip(),
                        subject.strip(),
                        int(total_students),
                    )
                except sqlite3.IntegrityError:
                    st.error("A class with this name already exists.")
                else:
                    st.success("Class added.")
                    st.rerun()

    return selected_class_name, selected_class
