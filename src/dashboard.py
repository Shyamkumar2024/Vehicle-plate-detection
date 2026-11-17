import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def apply_dashboard_styles():
    st.markdown("""
        <style>
        .kpi-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.4);
        }
        .kpi-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 4px;
            font-weight: 600;
        }
        .kpi-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #f8fafc;
            line-height: 1.2;
        }
        .kpi-subtext {
            font-size: 0.75rem;
            color: #38bdf8;
            margin-top: 4px;
        }
        </style>
    """, unsafe_allow_html=True)


def show_dashboard(df):
    apply_dashboard_styles()

    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 1rem;">
            <div>
                <h2 style="margin: 0; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    📊 Intelligence & Fleet Analytics
                </h2>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                    Real-time operational metrics, vehicle distribution, and ALPR quality statistics
                </p>
            </div>
            <div style="text-align: right;">
                <span style="display: inline-flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.3);">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: #34d399; display: inline-block;"></span> Live Database Active
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if df is None or df.empty:
        st.info("ℹ️ No vehicle activity records found yet. Upload images or run video tracking to generate telemetry.")
        return

    df = df.copy()

    # Pre-process fields safely
    if "vehicle_type" in df.columns:
        df["vehicle_type"] = df["vehicle_type"].fillna("unknown").astype(str).str.lower()
    else:
        df["vehicle_type"] = "unknown"

    if "event_type" not in df.columns:
        df["event_type"] = "detected"
    df["event_type"] = df["event_type"].fillna("detected").astype(str).str.lower()

    if "review_status" not in df.columns:
        df["review_status"] = "pending"
    df["review_status"] = df["review_status"].fillna("pending").astype(str).str.lower()

    if "plate_confidence" in df.columns:
        df["plate_confidence"] = pd.to_numeric(df["plate_confidence"], errors="coerce").fillna(0.0)
    else:
        df["plate_confidence"] = 0.0

    time_col = next((c for c in ["event_time", "timestamp", "created_at"] if c in df.columns), None)
    if time_col:
        df["parsed_time"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        df["parsed_time"] = pd.NaT

    # ----------------------------------------------------
    # FILTER CONTROLS (EXPANDABLE)
    # ----------------------------------------------------
    with st.expander("🔍 Interactive Filters & Date Scope", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            all_types = sorted(df["vehicle_type"].unique().tolist())
            selected_types = st.multiselect("Vehicle Types", all_types, default=all_types)
        with fcol2:
            all_events = sorted(df["event_type"].unique().tolist())
            selected_events = st.multiselect("Event Types", all_events, default=all_events)
        with fcol3:
            all_statuses = sorted(df["review_status"].unique().tolist())
            selected_statuses = st.multiselect("Review Status", all_statuses, default=all_statuses)

    filtered_df = df[
        (df["vehicle_type"].isin(selected_types)) &
        (df["event_type"].isin(selected_events)) &
        (df["review_status"].isin(selected_statuses))
    ]

    if filtered_df.empty:
        st.warning("⚠️ No records match the selected filter criteria.")
        return

    # ----------------------------------------------------
    # KPI METRIC CARDS
    # ----------------------------------------------------
    total_events = len(filtered_df)
    entries_count = len(filtered_df[filtered_df["event_type"] == "entry"])
    exits_count = len(filtered_df[filtered_df["event_type"] == "exit"])
    currently_inside = max(0, entries_count - exits_count)
    has_plate = len(filtered_df[filtered_df["plate_text"].astype(str).str.strip().ne("")])
    plate_rate = (has_plate / total_events * 100) if total_events > 0 else 0
    pending_count = len(filtered_df[filtered_df["review_status"] == "pending"])

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Logged Events</div>
                <div class="kpi-value">{total_events}</div>
                <div class="kpi-subtext">Across all sources</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Estimated in Lot</div>
                <div class="kpi-value" style="color: #38bdf8;">{currently_inside}</div>
                <div class="kpi-subtext">Entries ({entries_count}) - Exits ({exits_count})</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Plates Captured</div>
                <div class="kpi-value" style="color: #34d399;">{has_plate} <span style="font-size: 1rem; color: #94a3b8;">({plate_rate:.0f}%)</span></div>
                <div class="kpi-subtext">Identified plates</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Pending Verification</div>
                <div class="kpi-value" style="color: {'#f59e0b' if pending_count > 0 else '#94a3b8'};">{pending_count}</div>
                <div class="kpi-subtext">Needs human review</div>
            </div>
        """, unsafe_allow_html=True)
    with m5:
        verified_count = len(filtered_df[filtered_df["review_status"] == "verified"])
        verified_rate = (verified_count / total_events * 100) if total_events > 0 else 0
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Verification Rate</div>
                <div class="kpi-value" style="color: #818cf8;">{verified_rate:.0f}%</div>
                <div class="kpi-subtext">{verified_count} verified events</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ----------------------------------------------------
    # CHARTS ROW 1: Distribution & Classification
    # ----------------------------------------------------
    chart_col1, chart_col2 = st.columns(2)

    palette = ["#38bdf8", "#818cf8", "#34d399", "#f59e0b", "#ec4899", "#a855f7"]

    with chart_col1:
        type_counts = filtered_df["vehicle_type"].value_counts().reset_index()
        type_counts.columns = ["Vehicle Type", "Count"]
        fig_donut = px.pie(
            type_counts,
            names="Vehicle Type",
            values="Count",
            hole=0.55,
            color_discrete_sequence=palette,
            title="Vehicle Fleet Composition"
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, width='stretch')

    with chart_col2:
        fig_bar = px.bar(
            type_counts,
            x="Vehicle Type",
            y="Count",
            color="Vehicle Type",
            color_discrete_sequence=palette,
            text="Count",
            title="Vehicle Count by Classification"
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=False
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, width='stretch')

    # ----------------------------------------------------
    # CHARTS ROW 2: Timeline & Quality Metrics
    # ----------------------------------------------------
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # Time-series activity if time column is available
        valid_times = filtered_df.dropna(subset=["parsed_time"]).copy()
        if not valid_times.empty and len(valid_times) > 1:
            valid_times["Hour"] = valid_times["parsed_time"].dt.floor("h")
            time_series = valid_times.groupby(["Hour", "event_type"]).size().reset_index(name="Vehicles")
            fig_time = px.line(
                time_series,
                x="Hour",
                y="Vehicles",
                color="event_type",
                markers=True,
                title="Activity Timeline (Hourly Breakdown)",
                color_discrete_sequence=["#38bdf8", "#34d399", "#f59e0b"]
            )
            fig_time.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_time, width='stretch')
        else:
            # Fallback to camera ID distribution
            cam_counts = filtered_df["camera_id"].value_counts().reset_index()
            cam_counts.columns = ["Camera ID", "Events"]
            fig_cam = px.bar(
                cam_counts,
                x="Camera ID",
                y="Events",
                title="Event Volume by Camera / Source",
                color="Camera ID",
                color_discrete_sequence=palette
            )
            fig_cam.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                margin=dict(t=40, b=20, l=20, r=20),
                showlegend=False
            )
            st.plotly_chart(fig_cam, width='stretch')

    with chart_col4:
        # Confidence distribution
        conf_df = filtered_df[filtered_df["plate_confidence"] > 0]
        if not conf_df.empty:
            fig_conf = px.histogram(
                conf_df,
                x="plate_confidence",
                nbins=15,
                title="Plate Detection Confidence Distribution",
                color_discrete_sequence=["#818cf8"]
            )
            fig_conf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(title="Confidence Score (0.0 - 1.0)", showgrid=False),
                yaxis=dict(title="Detections", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_conf, width='stretch')
        else:
            status_counts = filtered_df["review_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_status = px.bar(
                status_counts,
                x="Status",
                y="Count",
                color="Status",
                title="Event Verification Pipeline",
                color_discrete_map={"verified": "#10b981", "pending": "#f59e0b", "rejected": "#ef4444"}
            )
            fig_status.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                margin=dict(t=40, b=20, l=20, r=20),
                showlegend=False
            )
            st.plotly_chart(fig_status, width='stretch')

    # ----------------------------------------------------
    # DATA EXPORT & TABLE
    # ----------------------------------------------------
    st.write("")
    with st.expander("📄 Export Telemetry & Raw Records", expanded=False):
        col_exp1, col_exp2 = st.columns([3, 1])
        with col_exp1:
            st.caption(f"Showing {len(filtered_df)} filtered records from SQLite database.")
        with col_exp2:
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export CSV",
                data=csv_data,
                file_name="vehicle_telemetry_report.csv",
                mime="text/csv",
                width='stretch'
            )
        display_cols = [c for c in ["id", "plate_text", "vehicle_type", "vehicle_color", "event_type", "event_time", "camera_id", "review_status", "plate_confidence"] if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols], width='stretch')
