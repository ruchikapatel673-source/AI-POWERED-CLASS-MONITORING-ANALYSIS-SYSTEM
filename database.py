import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pandas as pd

DATABASE_PATH = Path(__file__).resolve().parent / "classmonitor.sqlite3"

@contextmanager
def database_connection():
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

def initialize_database():
    with database_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                teacher_name TEXT NOT NULL,
                room_no TEXT NOT NULL,
                subject TEXT NOT NULL,
                total_students INTEGER NOT NULL CHECK (total_students >= 0)
            );
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                class_name TEXT NOT NULL,
                mode TEXT NOT NULL,
                total_detected INTEGER NOT NULL,
                sleeping INTEGER NOT NULL,
                attentive INTEGER NOT NULL,
                phone_use INTEGER NOT NULL DEFAULT 0,
                talking INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS attendance_status (
                attendance_date TEXT NOT NULL,
                class_name TEXT NOT NULL,
                roll TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('Present', 'Absent')),
                PRIMARY KEY (attendance_date, class_name, roll)
            );
            """
        )
        history_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(analysis_history)")
        }
        if "phone_use" not in history_columns:
            connection.execute(
                "ALTER TABLE analysis_history ADD COLUMN phone_use INTEGER NOT NULL DEFAULT 0"
            )
        if "talking" not in history_columns:
            connection.execute(
                "ALTER TABLE analysis_history ADD COLUMN talking INTEGER NOT NULL DEFAULT 0"
            )
        class_count = connection.execute(
            "SELECT COUNT(*) FROM classes"
        ).fetchone()[0]
        if class_count == 0:
            connection.executemany(
                """
                INSERT INTO classes (name, teacher_name, room_no, subject, total_students)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    ("CSE-A (5th Sem)", "Not set", "Not set", "Not set", 0),
                    ("CSE-B (5th Sem)", "Not set", "Not set", "Not set", 0),
                ],
            )

def saved_attendance(class_name: str, attendance_date: date) -> pd.DataFrame:
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT roll AS Roll, status AS Status
            FROM attendance_status
            WHERE class_name = ? AND attendance_date = ?
            """,
            (class_name, attendance_date.isoformat()),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows], columns=["Roll", "Status"])

def save_attendance(class_name: str, attendance_date: date, attendance: pd.DataFrame):
    with database_connection() as connection:
        connection.executemany(
            """
            INSERT INTO attendance_status (attendance_date, class_name, roll, name, status)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (attendance_date, class_name, roll)
            DO UPDATE SET name = excluded.name, status = excluded.status
            """,
            [
                (
                    attendance_date.isoformat(),
                    class_name,
                    row.Roll,
                    row.Name,
                    row.Status,
                )
                for row in attendance.itertuples(index=False)
            ],
        )
