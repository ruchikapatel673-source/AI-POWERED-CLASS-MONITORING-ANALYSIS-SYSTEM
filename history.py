from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import database_connection

HISTORY_COLUMNS = [
    "id",
    "created_at",
    "class_name",
    "mode",
    "total_detected",
    "sleeping",
    "attentive",
    "phone_use",
    "talking",
]

def read_history(class_name=None, limit=None):
    query = (
        "SELECT id, created_at, class_name, mode, total_detected, sleeping, attentive, "
        "phone_use, talking FROM analysis_history"
    )
    parameters = []
    if class_name:
        query += " WHERE class_name = ?"
        parameters.append(class_name)
    query += " ORDER BY created_at DESC, id DESC"
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)
    with database_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return pd.DataFrame([dict(row) for row in rows], columns=HISTORY_COLUMNS)

def save_analysis(class_name: str, mode: str, result: dict):
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with database_connection() as connection:
        connection.execute(
            """
            INSERT INTO analysis_history
                (created_at, class_name, mode, total_detected, sleeping, attentive, phone_use, talking)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                class_name,
                mode,
                result["total_detected"],
                result["sleeping"],
                result["attentive"],
                result["phone_use"],
                result["talking"],
            ),
        )
    result["created_at"] = created_at

def history_for_display(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Time",
                "Class",
                "Mode",
                "Total detected",
                "Sleeping",
                "Attentive",
                "Phone use",
                "Talking",
            ]
        )
    display = history.copy()
    timestamps = pd.to_datetime(display["created_at"])
    display["Date"] = timestamps.dt.strftime("%Y-%m-%d")
    display["Time"] = timestamps.dt.strftime("%H:%M:%S")
    return display.rename(
        columns={
            "class_name": "Class",
            "mode": "Mode",
            "total_detected": "Total detected",
            "sleeping": "Sleeping",
            "attentive": "Attentive",
            "phone_use": "Phone use",
            "talking": "Talking",
        }
    )[
        [
            "Date",
            "Time",
            "Class",
            "Mode",
            "Total detected",
            "Sleeping",
            "Attentive",
            "Phone use",
            "Talking",
        ]
    ]

def render_history(title: str, class_name=None, limit=10):
    st.subheader(title)
    history = read_history(class_name=class_name, limit=limit)
    if history.empty:
        st.info(
            "No photo analyses have been saved yet. Take or upload a photo from the Dashboard."
        )
    st.dataframe(history_for_display(history), use_container_width=True, hide_index=True)


def render_analytics():
    st.title("Analytics")
    st.write(
        "Live photo, photo upload, and video analyses share one persistent history."
    )
    history = read_history()
    if history.empty:
        st.info(
            "There is no analysis history to chart yet. Analyze and save a photo on the Dashboard."
        )
        line_chart = go.Figure()
        line_chart.add_annotation(
            text="Attentive and sleeping trends will appear here after an analysis is saved.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        line_chart.update_layout(title="Attention & sleeping over time", xaxis_title="Time")
        st.plotly_chart(line_chart, use_container_width=True)
        bar_chart = go.Figure()
        bar_chart.add_annotation(
            text="Class totals will appear here after an analysis is saved.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        bar_chart.update_layout(title="Class-wise student counts")
        st.plotly_chart(bar_chart, use_container_width=True)
        st.dataframe(history_for_display(history), use_container_width=True, hide_index=True)
        return

    history["created_at"] = pd.to_datetime(history["created_at"])
    summary = st.columns(3)
    summary[0].metric("Saved analyses", len(history))
    summary[1].metric("People detected", int(history["total_detected"].sum()))
    summary[2].metric("Sleeping estimates", int(history["sleeping"].sum()))

    class_filter = st.selectbox(
        "Analytics class filter",
        ["All classes"] + sorted(history["class_name"].unique().tolist()),
    )
    filtered = history if class_filter == "All classes" else history[
        history["class_name"] == class_filter
    ]
    line_chart = go.Figure()
    for column, label, color in (
        ("attentive", "Attentive", "#22d3ee"),
        ("sleeping", "Sleeping estimate", "#c084fc"),
    ):
        line_chart.add_trace(
            go.Scatter(
                x=filtered["created_at"],
                y=filtered[column],
                mode="lines+markers",
                name=label,
                line={"color": color},
            )
        )
    line_chart.update_layout(
        title="Attentive and sleeping estimates over time",
        xaxis_title="Analysis time",
        yaxis_title="People",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    st.plotly_chart(line_chart, use_container_width=True)

    class_counts = (
        filtered.groupby("class_name", as_index=False)[
            ["total_detected", "attentive", "sleeping"]
        ]
        .sum()
        .melt(id_vars="class_name", var_name="Measure", value_name="Count")
    )
    bar_chart = go.Figure()
    for label, color in (
        ("total_detected", "#3b82f6"),
        ("attentive", "#22c55e"),
        ("sleeping", "#c084fc"),
    ):
        values = class_counts[class_counts["Measure"] == label]
        bar_chart.add_bar(
            x=values["class_name"],
            y=values["Count"],
            name=label.replace("_", " ").title(),
            marker_color=color,
        )
    bar_chart.update_layout(
        title="Class-wise counts across saved analyses",
        barmode="group",
        xaxis_title="Class",
        yaxis_title="People",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    st.plotly_chart(bar_chart, use_container_width=True)
    st.subheader("Complete analysis history")
    st.dataframe(
        history_for_display(filtered.sort_values("created_at", ascending=False)),
        use_container_width=True,
        hide_index=True,
    )
