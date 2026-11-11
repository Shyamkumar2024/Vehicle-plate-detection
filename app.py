import streamlit as st
import pandas as pd
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from src.dashboard import show_dashboard
from src.vehicle_detector import VehicleDetector
from src.tracker import VehicleTracker
from src.quality import assess_image
from src.enhancement import (
    enhance_low_light,
    enhance_plate,
    unsharp_mask,
    deblur_motion,
    denoise_bilateral,
    morphological_contrast,
    deskew_plate,
    super_clean_plate
)
from src.attributes import estimate_color
from src.database import (
    initialize_database,
    get_all_events,
    add_event,
    update_event_status,
    update_event_plate,
    delete_event,
    get_event_counts
)
from src.plate_detector import PlateDetector
from src.ocr import PlateOCR
from src.utils import ROOT

# ==========================================
# 1. PAGE SETUP & DESIGN SYSTEM
# ==========================================

st.set_page_config(
    page_title="Vehicle Intelligence Suite",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_database()

# Inject Modern Cyber-Intelligence Styling
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Navigation Banner */
    .top-hero {
        background: radial-gradient(120% 120% at 50% 0%, rgba(30, 58, 138, 0.45) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 60%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
    }

    /* System Status Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        backdrop-filter: blur(8px);
    }
    .badge-active {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-neutral {
        background: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    /* Inspection & Result Cards */
    .inspection-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    .inspection-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.12);
    }

    /* Realistic Monospace License Plate Badge */
    .license-plate-tag {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        font-size: 1.15rem;
        letter-spacing: 0.12em;
        background: linear-gradient(180deg, #ffffff 0%, #e2e8f0 100%);
        color: #0f172a;
        padding: 4px 14px;
        border-radius: 6px;
        border: 2px solid #334155;
        box-shadow: inset 0 1px 2px rgba(255,255,255,0.8), 0 2px 8px rgba(0,0,0,0.3);
        display: inline-block;
        text-transform: uppercase;
    }
    .license-plate-tag.unreadable {
        background: #1e293b;
        color: #64748b;
        border: 1px dashed #475569;
        font-size: 0.9rem;
    }

    /* Metric Meter Box */
    .meter-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
    }
    .meter-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .meter-lbl {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. SIDEBAR CONFIGURATION & DIAGNOSTICS
# ==========================================

with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
            <span style="font-size: 1.8rem;">⚙️</span>
            <div>
                <h3 style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Control Console</h3>
                <span style="font-size: 0.75rem; color: #38bdf8;">Autonomous ALPR Settings</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # System Status Monitor
    st.markdown("**Core Model Health**")
    veh_model_exists = Path("models/vehicle_detector.pt").exists() or Path("yolo26n.pt").exists()
    plate_model_exists = Path("models/plate_detector.pt").exists()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if veh_model_exists:
            st.markdown('<span class="badge-pill badge-active">🟢 YOLOv8 Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-pill badge-warn">🟡 Fallback Model</span>', unsafe_allow_html=True)
    with col_s2:
        if plate_model_exists:
            st.markdown('<span class="badge-pill badge-active">🟢 ALPR Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-pill badge-warn">🔴 ALPR Missing</span>', unsafe_allow_html=True)

    st.divider()

    st.markdown("**Detection Sensitivities**")
    conf_vehicle = st.slider(
        "Vehicle Confidence Threshold",
        min_value=0.15,
        max_value=0.85,
        value=0.35,
        step=0.05,
        help="Higher values reduce false positives; lower values detect distant vehicles."
    )

    conf_plate = st.slider(
        "License Plate Threshold",
        min_value=0.15,
        max_value=0.85,
        value=0.30,
        step=0.05,
        help="Sensitivity for identifying license plate bounding rectangles."
    )

    enable_clahe = st.checkbox(
        "Low-Light Auto-Enhancement",
        value=True,
        help="Applies adaptive contrast enhancement (CLAHE) on dark or foggy scenes."
    )

    auto_log = st.checkbox(
        "Auto-Save to Inventory DB",
        value=True,
        help="Automatically records every detected vehicle into SQLite event ledger."
    )

    default_cam = st.text_input(
        "Default Ingestion Camera ID",
        value="CAM-01-ENTRY"
    )

    st.divider()
    counts = get_event_counts()
    st.markdown(f"""
        <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px;">
            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Ledger Quick Stats</div>
            <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                <span>Total Events:</span> <strong>{counts['total']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; color: #34d399;">
                <span>Verified:</span> <strong>{counts['verified']}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; color: #fbbf24;">
                <span>Pending Review:</span> <strong>{counts['pending']}</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# 3. TOP HERO BANNER
# ==========================================

st.markdown("""
    <div class="top-hero">
        <div>
            <h1 class="hero-title">🚘 Vehicle Intelligence // AI Operations Suite</h1>
            <div class="hero-subtitle">
                High-throughput vehicle detection, autonomous license plate localization, optical character recognition & live inventory analytics.
            </div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end;">
            <span class="badge-pill badge-active">● Edge Inference</span>
            <span class="badge-pill badge-neutral">SQLite Ledger</span>
            <span class="badge-pill badge-neutral">EasyOCR v1.7</span>
        </div>
    </div>
""", unsafe_allow_html=True)


# ==========================================
# 4. NAVIGATION TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Image Inspection Station",
    "🎥 Video Telemetry Stream",
    "📋 Verification & Inventory Station",
    "📊 Fleet Analytics Dashboard",
    "🔬 Plate Deblur & Restoration Lab"
])


# ==========================================
# 5. TAB 1: IMAGE INSPECTION STATION
# ==========================================

with tab1:
    st.markdown("### 🔍 Single & Multi-Vehicle Inspection")
    st.caption("Upload a surveillance still or vehicle photo to run image quality diagnostics, vehicle localization, plate extraction, and OCR.")

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload Vehicle Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_uploader",
            label_visibility="collapsed"
        )
    with up_col2:
        run_alpr = st.toggle("Enable License Plate Localization + OCR", value=True)
        enhance_toggle = st.toggle("Apply Pre-enhancement Filter", value=enable_clahe)

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
        raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if raw_bgr is None:
            st.error("❌ Unable to decode image buffer. Please ensure the file is an undamaged image.")
        else:
            # Optional pre-enhancement
            active_image = enhance_low_light(raw_bgr) if enhance_toggle else raw_bgr

            # ----------------------------------------------------
            # Visual Quality Assessment Dashboard
            # ----------------------------------------------------
            q = assess_image(active_image)
            sharp_status = "Crystal Sharp" if q["blur_score"] > 150 else ("Fair Clarity" if q["blur_score"] > 60 else "Motion Blur Warning")
            sharp_color = "#34d399" if q["blur_score"] > 100 else ("#fbbf24" if q["blur_score"] > 60 else "#f87171")

            light_status = "Well Balanced" if not q["is_dark"] and q["brightness"] < 210 else ("Low-Light Alert" if q["is_dark"] else "High Glare Alert")
            light_color = "#34d399" if not q["is_dark"] and q["brightness"] < 210 else "#fbbf24"

            st.write("")
            qc1, qc2, qc3, qc4 = st.columns(4)
            with qc1:
                st.markdown(f"""
                    <div class="meter-box">
                        <div class="meter-val" style="color: {sharp_color};">{q['blur_score']:.1f}</div>
                        <div class="meter-lbl">Clarity Score ({sharp_status})</div>
                    </div>
                """, unsafe_allow_html=True)
            with qc2:
                st.markdown(f"""
                    <div class="meter-box">
                        <div class="meter-val" style="color: {light_color};">{q['brightness']:.1f}</div>
                        <div class="meter-lbl">Mean Luminance ({light_status})</div>
                    </div>
                """, unsafe_allow_html=True)
            with qc3:
                st.markdown(f"""
                    <div class="meter-box">
                        <div class="meter-val">{q['width']} × {q['height']}</div>
                        <div class="meter-lbl">Sensor Resolution ({q['width']*q['height']/1e6:.1f} MP)</div>
                    </div>
                """, unsafe_allow_html=True)
            with qc4:
                passed = not q["is_potentially_blurry"]
                status_txt = "✅ Passed for ALPR" if passed else "⚠️ Marginal Focus"
                status_col = "#34d399" if passed else "#fbbf24"
                st.markdown(f"""
                    <div class="meter-box">
                        <div class="meter-val" style="color: {status_col};">{status_txt}</div>
                        <div class="meter-lbl">ALPR Compliance Verdict</div>
                    </div>
                """, unsafe_allow_html=True)

            st.write("")

            # Action trigger
            if st.button("🚀 Run Vehicle Detection & ALPR Pipeline", type="primary", use_container_width=True):
                with st.spinner("Executing neural inference across models..."):
                    try:
                        # 1. Vehicle Detection
                        veh_detector = VehicleDetector(
                            "models/vehicle_detector.pt" if Path("models/vehicle_detector.pt").exists() else "yolo26n.pt"
                        )
                        detections = veh_detector.detect(active_image, confidence=conf_vehicle)

                        # 2. Plate Detection & OCR setup
                        pd_model = None
                        ocr_engine = None
                        if run_alpr and Path("models/plate_detector.pt").exists():
                            pd_model = PlateDetector("models/plate_detector.pt")
                            ocr_engine = PlateOCR(gpu=False)

                        # Annotate image
                        annotated = active_image.copy()
                        color_map = {
                            "car": (248, 189, 56),      # Cyan-ish in BGR
                            "truck": (56, 165, 245),    # Amber/Orange in BGR
                            "bus": (247, 85, 168),      # Purple/Magenta in BGR
                            "motorcycle": (129, 212, 52) # Green in BGR
                        }

                        results_payload = []
                        saved_events = 0

                        for idx, d in enumerate(detections):
                            x1, y1, x2, y2 = d["bbox"]
                            cls_name = d["class"]
                            conf = d["confidence"]
                            box_col = color_map.get(cls_name, (255, 255, 255))

                            # Crop vehicle
                            vh, vw = active_image.shape[:2]
                            vx1, vy1 = max(0, x1), max(0, y1)
                            vx2, vy2 = min(vw, x2), min(vh, y2)
                            vehicle_crop = active_image[vy1:vy2, vx1:vx2]

                            # Estimate color
                            est_color = estimate_color(vehicle_crop)

                            # Plate detection
                            detected_plate_text = ""
                            plate_conf = 0.0
                            ocr_conf = 0.0
                            needs_review = True
                            plate_crop = None

                            if pd_model and vehicle_crop.size > 0:
                                try:
                                    plate_candidates = pd_model.detect(vehicle_crop, confidence=conf_plate)
                                    if plate_candidates:
                                        best_plate = max(plate_candidates, key=lambda x: x["confidence"])
                                        px1, py1, px2, py2 = best_plate["bbox"]
                                        plate_conf = float(best_plate["confidence"])

                                        # Global plate bbox for annotation
                                        gx1, gy1 = vx1 + px1, vy1 + py1
                                        gx2, gy2 = vx1 + px2, vy1 + py2
                                        cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 235, 255), 2)

                                        # Crop plate for OCR
                                        p_crop = vehicle_crop[max(0, py1):max(0, py2), max(0, px1):max(0, px2)]
                                        if p_crop.size > 0 and ocr_engine:
                                            plate_crop = p_crop
                                            ocr_res = ocr_engine.read(p_crop)
                                            detected_plate_text = ocr_res.get("text", "").strip()
                                            ocr_conf = float(ocr_res.get("confidence", 0.0))
                                            needs_review = bool(ocr_res.get("needs_review", True))
                                except Exception as err:
                                    st.toast(f"Plate localization error: {err}", icon="⚠️")

                            # Draw vehicle bounding box & label
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_col, 2)
                            label_str = f"{cls_name.upper()} {conf:.2f}"
                            if detected_plate_text:
                                label_str += f" | {detected_plate_text}"
                            cv2.putText(
                                annotated,
                                label_str,
                                (x1, max(24, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                box_col,
                                2
                            )

                            # Auto save to SQLite ledger
                            if auto_log:
                                try:
                                    add_event(
                                        plate_text=detected_plate_text,
                                        plate_confidence=plate_conf,
                                        vehicle_type=cls_name,
                                        vehicle_color=est_color,
                                        event_type="detected",
                                        event_time=datetime.now().astimezone().isoformat(timespec="seconds"),
                                        camera_id=default_cam,
                                        review_status="pending" if needs_review or not detected_plate_text else "verified"
                                    )
                                    saved_events += 1
                                except Exception as dbe:
                                    pass

                            results_payload.append({
                                "id": idx + 1,
                                "class": cls_name,
                                "confidence": conf,
                                "color": est_color,
                                "plate_text": detected_plate_text,
                                "plate_conf": plate_conf,
                                "ocr_conf": ocr_conf,
                                "needs_review": needs_review,
                                "vehicle_crop": vehicle_crop,
                                "plate_crop": plate_crop
                            })

                        st.session_state["last_image_results"] = results_payload
                        st.session_state["last_annotated_img"] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.session_state["last_raw_img"] = cv2.cvtColor(active_image, cv2.COLOR_BGR2RGB)
                        st.session_state["saved_events_count"] = saved_events

                        st.toast(f"Found {len(detections)} vehicles! {saved_events} logged to database.", icon="🎯")

                    except Exception as pipeline_err:
                        st.error(f"❌ Inference pipeline failure: {pipeline_err}")

            # ----------------------------------------------------
            # Display Annotated Visuals & Per-Vehicle Gallery
            # ----------------------------------------------------
            if "last_annotated_img" in st.session_state:
                st.write("")
                img_tab_a, img_tab_b = st.tabs(["🎯 Annotated Detections", "🖼️ Original Image"])
                with img_tab_a:
                    st.image(st.session_state["last_annotated_img"], width='stretch')
                with img_tab_b:
                    st.image(st.session_state["last_raw_img"], width='stretch')

                results_list = st.session_state.get("last_image_results", [])
                st.write("")
                st.markdown(f"#### 🚘 Vehicle Inspection Cards ({len(results_list)} Identified)")

                if not results_list:
                    st.warning("No vehicles were recognized within the configured threshold.")
                else:
                    for item in results_list:
                        with st.container():
                            st.markdown(f"""
                                <div class="inspection-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                        <div style="display: flex; align-items: center; gap: 8px;">
                                            <span style="font-weight: 700; font-size: 1.1rem; color: #f8fafc;">Vehicle #{item['id']}</span>
                                            <span class="badge-pill badge-neutral" style="text-transform: capitalize;">{item['class']}</span>
                                            <span class="badge-pill badge-neutral" style="text-transform: capitalize;">🎨 {item['color']}</span>
                                        </div>
                                        <div>
                                            <span style="font-size: 0.85rem; color: #38bdf8; font-weight: 600;">Conf: {item['confidence']:.1%}</span>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                            col_vimg, col_pimg, col_acts = st.columns([1.5, 1.5, 2])
                            with col_vimg:
                                if item["vehicle_crop"] is not None and item["vehicle_crop"].size > 0:
                                    st.image(cv2.cvtColor(item["vehicle_crop"], cv2.COLOR_BGR2RGB), caption="Vehicle Crop", width='stretch')
                            with col_pimg:
                                if item["plate_crop"] is not None and item["plate_crop"].size > 0:
                                    st.image(cv2.cvtColor(item["plate_crop"], cv2.COLOR_BGR2RGB), caption="Plate Localization", width='stretch')
                                else:
                                    st.info("No plate detected for this vehicle.")
                            with col_acts:
                                plate_val = item["plate_text"]
                                if plate_val:
                                    st.markdown(f'<div class="license-plate-tag">{plate_val}</div>', unsafe_allow_html=True)
                                    st.caption(f"OCR Confidence: {item['ocr_conf']:.1%} | Localization: {item['plate_conf']:.1%}")
                                else:
                                    st.markdown('<div class="license-plate-tag unreadable">NO PLATE OCR</div>', unsafe_allow_html=True)

                                st.write("")
                                q_act1, q_act2 = st.columns(2)
                                with q_act1:
                                    if st.button(f"Log Entry 🚗➡️", key=f"btn_entry_{item['id']}"):
                                        add_event(
                                            plate_text=plate_val or "UNVERIFIED",
                                            plate_confidence=item["plate_conf"],
                                            vehicle_type=item["class"],
                                            vehicle_color=item["color"],
                                            event_type="entry",
                                            event_time=datetime.now().astimezone().isoformat(timespec="seconds"),
                                            camera_id=default_cam,
                                            review_status="verified" if plate_val else "pending"
                                        )
                                        st.toast(f"Logged Vehicle #{item['id']} as ENTRY!", icon="✅")
                                with q_act2:
                                    if st.button(f"Log Exit ⬅️🚗", key=f"btn_exit_{item['id']}"):
                                        add_event(
                                            plate_text=plate_val or "UNVERIFIED",
                                            plate_confidence=item["plate_conf"],
                                            vehicle_type=item["class"],
                                            vehicle_color=item["color"],
                                            event_type="exit",
                                            event_time=datetime.now().astimezone().isoformat(timespec="seconds"),
                                            camera_id=default_cam,
                                            review_status="verified" if plate_val else "pending"
                                        )
                                        st.toast(f"Logged Vehicle #{item['id']} as EXIT!", icon="🚪")

                                if item["plate_crop"] is not None and item["plate_crop"].size > 0:
                                    if st.button(f"🔬 Send to Deblur Lab", key=f"btn_send_lab_{item['id']}", use_container_width=True):
                                        st.session_state["deblur_lab_img"] = item["plate_crop"]
                                        st.toast(f"Vehicle #{item['id']} plate sent to Deblur Lab! Switch to Tab 5.", icon="🔬")


# ==========================================
# 6. TAB 2: VIDEO TELEMETRY STREAM
# ==========================================

with tab2:
    st.markdown("### 🎥 Video Surveillance & Flow Analysis")
    st.caption("Perform frame-by-frame vehicle tracking, autonomous license plate localization, optical character recognition & throughput analytics.")

    v_col1, v_col2 = st.columns([2, 1])
    with v_col1:
        video_file = st.file_uploader(
            "Upload Traffic Footage (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            key="video_uploader",
            label_visibility="collapsed"
        )
    with v_col2:
        enable_video_alpr = st.toggle("🔍 Detect & Read License Plates (ALPR)", value=True, help="Detects and reads license plates on every tracked vehicle in the video.")
        frame_sample_rate = st.select_slider(
            "Processing Skip Rate",
            options=[1, 2, 3, 5, 10],
            value=2,
            help="Analyzes every Nth frame. Skip rate of 2 gives high accuracy while maintaining speed."
        )

    if video_file:
        temp_video_path = ROOT / "outputs" / "temp_input_stream.mp4"
        temp_video_path.parent.mkdir(parents=True, exist_ok=True)
        temp_video_path.write_bytes(video_file.getvalue())

        col_vprev, col_vctl = st.columns([3, 1])
        with col_vctl:
            st.write("")
            start_video_btn = st.button("▶️ Start Video ALPR & Telemetry", type="primary", use_container_width=True)

        if start_video_btn:
            cap = cv2.VideoCapture(str(temp_video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps_source = cap.get(cv2.CAP_PROP_FPS) or 25

            # Initialize ByteTrack vehicle tracker
            tracker = VehicleTracker(
                "models/vehicle_detector.pt" if Path("models/vehicle_detector.pt").exists() else "yolo26n.pt"
            )

            # Initialize Plate localization and OCR if enabled
            plate_model = None
            ocr_engine = None
            if enable_video_alpr and Path("models/plate_detector.pt").exists():
                plate_model = PlateDetector("models/plate_detector.pt")
                ocr_engine = PlateOCR(gpu=False)

            # UI Live Monitors
            st.write("")
            hud1, hud2, hud3, hud4 = st.columns(4)
            with hud1:
                frame_metric = st.empty()
            with hud2:
                count_metric = st.empty()
            with hud3:
                unique_metric = st.empty()
            with hud4:
                plate_metric = st.empty()

            progress_bar = st.progress(0)
            video_frame_placeholder = st.empty()

            frame_idx = 0
            processed_frames = 0
            telemetry_records = []
            tracked_vehicles = {}
            start_time = datetime.now()

            while cap.isOpened() and frame_idx < 350:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_sample_rate == 0:
                    vh, vw = frame.shape[:2]
                    tracks = tracker.track_frame(frame, confidence=conf_vehicle)
                    curr_count = len(tracks)

                    for t in tracks:
                        tid = t["track_id"]
                        cls_name = t["class"]
                        conf = t["confidence"]
                        x1, y1, x2, y2 = t["bbox"]

                        vx1, vy1 = max(0, x1), max(0, y1)
                        vx2, vy2 = min(vw, x2), min(vh, y2)
                        vehicle_crop = frame[vy1:vy2, vx1:vx2]

                        if tid not in tracked_vehicles:
                            est_col = estimate_color(vehicle_crop) if vehicle_crop.size > 0 else "unknown"
                            tracked_vehicles[tid] = {
                                "track_id": tid,
                                "class": cls_name,
                                "color": est_col,
                                "confidence": conf,
                                "plate_text": "",
                                "plate_conf": 0.0,
                                "ocr_conf": 0.0,
                                "vehicle_crop": vehicle_crop.copy() if vehicle_crop.size > 0 else None,
                                "plate_crop": None,
                                "first_seen_sec": round(frame_idx / fps_source, 2),
                                "logged": False
                            }

                        # Plate Detection & Recognition trigger
                        should_check_plate = (
                            plate_model is not None and
                            vehicle_crop.size > 0 and
                            (tracked_vehicles[tid]["plate_text"] == "" or frame_idx % (frame_sample_rate * 3) == 0)
                        )

                        if should_check_plate:
                            try:
                                plate_candidates = plate_model.detect(vehicle_crop, confidence=conf_plate)
                                if plate_candidates:
                                    best_p = max(plate_candidates, key=lambda x: x["confidence"])
                                    px1, py1, px2, py2 = best_p["bbox"]
                                    p_conf = float(best_p["confidence"])

                                    gx1, gy1 = vx1 + px1, vy1 + py1
                                    gx2, gy2 = vx1 + px2, vy1 + py2
                                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 240, 255), 2)

                                    p_crop = vehicle_crop[max(0, py1):max(0, py2), max(0, px1):max(0, px2)]
                                    if p_crop.size > 0 and ocr_engine:
                                        ocr_res = ocr_engine.read(p_crop)
                                        rec_text = ocr_res.get("text", "").strip()
                                        rec_conf = float(ocr_res.get("confidence", 0.0))

                                        if rec_text and rec_conf > tracked_vehicles[tid]["ocr_conf"]:
                                            tracked_vehicles[tid]["plate_text"] = rec_text
                                            tracked_vehicles[tid]["plate_conf"] = p_conf
                                            tracked_vehicles[tid]["ocr_conf"] = rec_conf
                                            tracked_vehicles[tid]["plate_crop"] = p_crop.copy()
                                            if vehicle_crop.size > 0:
                                                tracked_vehicles[tid]["vehicle_crop"] = vehicle_crop.copy()

                                            # Auto save to SQLite ledger once identified
                                            if auto_log and not tracked_vehicles[tid]["logged"]:
                                                try:
                                                    add_event(
                                                        plate_text=rec_text,
                                                        plate_confidence=p_conf,
                                                        vehicle_type=cls_name,
                                                        vehicle_color=tracked_vehicles[tid]["color"],
                                                        event_type="detected",
                                                        event_time=datetime.now().astimezone().isoformat(timespec="seconds"),
                                                        camera_id=default_cam,
                                                        track_id=tid,
                                                        review_status="verified" if rec_conf >= 0.80 else "pending"
                                                    )
                                                    tracked_vehicles[tid]["logged"] = True
                                                except Exception:
                                                    pass
                            except Exception:
                                pass

                        # Visual Bounding Box & Label
                        p_info = tracked_vehicles[tid]["plate_text"]
                        if p_info:
                            box_color = (0, 230, 255) # Bright Gold/Cyan
                            label_str = f"{cls_name.upper()} #{tid} | {p_info} ({tracked_vehicles[tid]['ocr_conf']:.0%})"
                        else:
                            box_color = (56, 189, 248) # Sky blue
                            label_str = f"{cls_name.upper()} #{tid}"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Filled banner for high contrast readability
                        (t_w, t_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + t_w + 8, max(22, y1)), (15, 23, 42), -1)
                        cv2.putText(
                            frame,
                            label_str,
                            (x1 + 4, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            box_color,
                            2
                        )

                    # Update Live HUD
                    elapsed = max(0.001, (datetime.now() - start_time).total_seconds())
                    live_fps = processed_frames / elapsed
                    plates_found = sum(1 for v in tracked_vehicles.values() if v["plate_text"])

                    frame_metric.metric("Frame", f"{frame_idx} / {total_frames}")
                    count_metric.metric("Active in Frame", curr_count)
                    unique_metric.metric("Unique Vehicles", len(tracked_vehicles))
                    plate_metric.metric("Plates Identified", f"{plates_found} Plates", delta=f"{plates_found} captured")

                    video_frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width='stretch')

                    telemetry_records.append({
                        "frame": frame_idx,
                        "timestamp_sec": round(frame_idx / fps_source, 2),
                        "vehicles_detected": curr_count,
                        "unique_vehicles": len(tracked_vehicles),
                        "plates_captured": plates_found
                    })
                    processed_frames += 1

                frame_idx += 1
                progress_bar.progress(min(1.0, frame_idx / min(total_frames, 350)))

            cap.release()
            plates_total = sum(1 for v in tracked_vehicles.values() if v["plate_text"])
            st.success(f"✅ Telemetry completed! Tracked {len(tracked_vehicles)} unique vehicles and identified {plates_total} license plates across {processed_frames} frames.")

            # Store in session state for persistence
            st.session_state["last_video_vehicles"] = list(tracked_vehicles.values())
            st.session_state["last_video_telemetry"] = telemetry_records

        # Display Gallery of Identified Vehicles & Plates
        video_vehicles = st.session_state.get("last_video_vehicles", [])
        if video_vehicles:
            st.write("")
            st.markdown(f"#### 🚘 Captured Video Vehicles & License Plates ({len(video_vehicles)} Vehicles)")

            for veh in video_vehicles:
                v_id = veh["track_id"]
                plate_str = veh["plate_text"]
                c_cls = veh["class"]
                c_col = veh["color"]

                with st.container():
                    st.markdown(f"""
                        <div class="inspection-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <div>
                                    <span style="font-weight: 700; font-size: 1.05rem; color: #f8fafc;">Vehicle Track #{v_id}</span>
                                    <span style="margin: 0 8px; color: #64748b;">|</span>
                                    <span class="badge-pill badge-neutral" style="text-transform: capitalize;">{c_cls}</span>
                                    <span class="badge-pill badge-neutral" style="text-transform: capitalize;">🎨 {c_col}</span>
                                    <span style="margin: 0 8px; color: #64748b;">|</span>
                                    <span style="color: #94a3b8; font-size: 0.85rem;">First Seen: {veh['first_seen_sec']}s</span>
                                </div>
                                <div>
                                    <span class="badge-pill {'badge-active' if plate_str else 'badge-warn'}">
                                        {'● Plate Verified' if plate_str else '○ Searching Plate'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    c_v1, c_v2, c_v3 = st.columns([1.5, 1.5, 2])
                    with c_v1:
                        if veh["vehicle_crop"] is not None and veh["vehicle_crop"].size > 0:
                            st.image(cv2.cvtColor(veh["vehicle_crop"], cv2.COLOR_BGR2RGB), caption=f"Vehicle #{v_id} Crop", width='stretch')
                    with c_v2:
                        if veh["plate_crop"] is not None and veh["plate_crop"].size > 0:
                            st.image(cv2.cvtColor(veh["plate_crop"], cv2.COLOR_BGR2RGB), caption="Plate Localization", width='stretch')
                        else:
                            st.info("Plate not visible or occluded in this track angle.")
                    with c_v3:
                        if plate_str:
                            st.markdown(f'<div class="license-plate-tag">{plate_str}</div>', unsafe_allow_html=True)
                            st.caption(f"OCR Confidence: {veh['ocr_conf']:.1%} | Localization: {veh['plate_conf']:.1%}")
                        else:
                            st.markdown('<div class="license-plate-tag unreadable">NO PLATE DETECTED</div>', unsafe_allow_html=True)

                        st.write("")
                        b_col1, b_col2 = st.columns(2)
                        with b_col1:
                            if st.button(f"Log Entry 🚗➡️", key=f"v_ent_{v_id}"):
                                add_event(
                                    plate_text=plate_str or "UNVERIFIED",
                                    plate_confidence=veh["plate_conf"],
                                    vehicle_type=c_cls,
                                    vehicle_color=c_col,
                                    event_type="entry",
                                    event_time=datetime.now().astimezone().isoformat(timespec="seconds"),
                                    camera_id=default_cam,
                                    track_id=v_id,
                                    review_status="verified" if plate_str else "pending"
                                )
                                st.toast(f"Logged Vehicle #{v_id} as ENTRY!", icon="✅")
                        with b_col2:
                            if st.button(f"Log Exit ⬅️🚗", key=f"v_ext_{v_id}"):
                                add_event(
                                    plate_text=plate_str or "UNVERIFIED",
                                    plate_confidence=veh["plate_conf"],
                                    vehicle_type=c_cls,
                                    vehicle_color=c_col,
                                    event_type="exit",
                                    event_time=datetime.now().astimezone().isoformat(timespec="seconds"),
                                    camera_id=default_cam,
                                    track_id=v_id,
                                    review_status="verified" if plate_str else "pending"
                                )
                                st.toast(f"Logged Vehicle #{v_id} as EXIT!", icon="🚪")

                        if veh["plate_crop"] is not None and veh["plate_crop"].size > 0:
                            if st.button(f"🔬 Send Plate to Deblur Lab", key=f"v_lab_plate_{v_id}", use_container_width=True):
                                st.session_state["deblur_lab_img"] = veh["plate_crop"].copy()
                                st.session_state["deblur_lab_source_info"] = f"Video Track #{v_id} ({c_cls})"
                                st.toast(f"Sent Plate from Track #{v_id} to Deblur Lab! Switch to Tab 5 to enhance & clear.", icon="🔬")
                        elif veh["vehicle_crop"] is not None and veh["vehicle_crop"].size > 0:
                            if st.button(f"🔬 Send Vehicle Crop to Deblur Lab", key=f"v_lab_veh_{v_id}", use_container_width=True):
                                st.session_state["deblur_lab_img"] = veh["vehicle_crop"].copy()
                                st.session_state["deblur_lab_source_info"] = f"Video Track #{v_id} ({c_cls} vehicle crop)"
                                st.toast(f"Sent Vehicle #{v_id} crop to Deblur Lab! Switch to Tab 5 to isolate & clear.", icon="🔬")

            # Table of video detections
            st.write("")
            table_records = [
                {
                    "Track ID": v["track_id"],
                    "Vehicle Type": v["class"].upper(),
                    "Color": v["color"],
                    "License Plate": v["plate_text"] or "UNREADABLE",
                    "OCR Confidence": f"{v['ocr_conf']:.1%}",
                    "Timestamp (s)": v["first_seen_sec"]
                }
                for v in video_vehicles
            ]
            st.dataframe(pd.DataFrame(table_records), width='stretch')

        # Telemetry line chart
        video_telemetry = st.session_state.get("last_video_telemetry", [])
        if video_telemetry:
            tel_df = pd.DataFrame(video_telemetry)
            st.markdown("#### 📈 Video Traffic Density & Plate Capture Timeline")
            st.line_chart(tel_df.set_index("timestamp_sec")[["vehicles_detected", "plates_captured"]], width='stretch')


# ==========================================
# 7. TAB 3: VERIFICATION & INVENTORY STATION
# ==========================================

with tab3:
    st.markdown("### 📋 Ledger & Human-in-the-Loop Verification")
    st.caption("Review autonomous vehicle events, correct OCR plate readings, and register manual gate crossings.")

    # Live ledger counts
    kpis = get_event_counts()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Events", kpis["total"])
    with c2:
        st.metric("Verified Events", kpis["verified"])
    with c3:
        st.metric("Pending Review", kpis["pending"], delta=None if kpis["pending"] == 0 else f"{kpis['pending']} require attention")
    with c4:
        st.metric("Rejected / False Alarms", kpis["rejected"])

    st.write("")

    inv_tab_queue, inv_tab_manual = st.tabs(["🔍 Interactive Review Station", "✍️ Manual Event Registration"])

    with inv_tab_queue:
        # Filter buttons
        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            status_filter = st.selectbox(
                "Filter Ledger by Status",
                ["All", "Pending Review Only", "Verified Only", "Rejected Only"]
            )
        with filter_col2:
            search_query = st.text_input(
                "Search by License Plate Number",
                placeholder="Type plate number to filter...",
                label_visibility="visible"
            )

        all_events = get_all_events(limit=500)
        events_df = pd.DataFrame([dict(r) for r in all_events])

        if events_df.empty:
            st.info("No recorded events currently exist in the database.")
        else:
            if status_filter == "Pending Review Only":
                events_df = events_df[events_df["review_status"] == "pending"]
            elif status_filter == "Verified Only":
                events_df = events_df[events_df["review_status"] == "verified"]
            elif status_filter == "Rejected Only":
                events_df = events_df[events_df["review_status"] == "rejected"]

            if search_query:
                events_df = events_df[
                    events_df["plate_text"].fillna("").astype(str).str.contains(search_query, case=False, regex=False)
                ]

            st.caption(f"Showing {len(events_df)} records matching filter criteria.")

            # Interactive verification cards for pending records
            for idx, row in events_df.head(20).iterrows():
                row_id = row["id"]
                p_text = row["plate_text"] or ""
                v_type = row["vehicle_type"] or "car"
                v_color = row["vehicle_color"] or "unknown"
                ev_type = row["event_type"] or "detected"
                r_status = row["review_status"] or "pending"
                time_str = row["event_time"] or ""
                conf_val = float(row["plate_confidence"] or 0.0)

                with st.container():
                    st.markdown(f"""
                        <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-weight: 700; color: #f8fafc; font-size: 1rem;">Record #{row_id}</span>
                                    <span style="margin: 0 8px; color: #64748b;">|</span>
                                    <span class="license-plate-tag {'unreadable' if not p_text else ''}">{p_text or 'UNREADABLE'}</span>
                                    <span style="margin: 0 8px; color: #64748b;">|</span>
                                    <span style="color: #94a3b8; font-size: 0.85rem;">{v_type.upper()} ({v_color})</span>
                                    <span style="margin: 0 8px; color: #64748b;">|</span>
                                    <span style="color: #38bdf8; font-size: 0.85rem;">{ev_type.upper()}</span>
                                </div>
                                <div>
                                    <span class="badge-pill {'badge-active' if r_status == 'verified' else ('badge-warn' if r_status == 'pending' else 'badge-neutral')}">
                                        {r_status.upper()}
                                    </span>
                                </div>
                            </div>
                            <div style="font-size: 0.78rem; color: #64748b; margin-top: 6px;">
                                Timestamp: {time_str} | Camera: {row['camera_id']} | Conf: {conf_val:.1%}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 3])
                    with btn_c1:
                        if r_status != "verified" and st.button("✅ Approve", key=f"app_{row_id}"):
                            update_event_status(row_id, "verified")
                            st.toast(f"Record #{row_id} marked as VERIFIED!", icon="✅")
                            st.rerun()
                    with btn_c2:
                        if r_status != "rejected" and st.button("❌ Reject", key=f"rej_{row_id}"):
                            update_event_status(row_id, "rejected")
                            st.toast(f"Record #{row_id} marked as REJECTED!", icon="🗑️")
                            st.rerun()
                    with btn_c3:
                        with st.popover("✏️ Edit Plate"):
                            new_plate = st.text_input("Correct License Plate", value=p_text, key=f"edit_inp_{row_id}")
                            if st.button("Save Correction", key=f"save_edit_{row_id}"):
                                if new_plate.strip():
                                    update_event_plate(row_id, new_plate.strip(), "verified")
                                    st.toast(f"Updated Record #{row_id} to '{new_plate.strip().upper()}'!", icon="💾")
                                    st.rerun()

    with inv_tab_manual:
        st.markdown("#### ✍️ Register Manual Gate Event")
        with st.form("manual_event_form"):
            mf_c1, mf_c2 = st.columns(2)
            with mf_c1:
                man_plate = st.text_input("License Plate Number", placeholder="e.g. MH12AB1234").strip().upper()
                man_type = st.selectbox("Vehicle Category", ["car", "truck", "bus", "motorcycle", "van"])
                man_color = st.selectbox("Vehicle Color", ["white", "black", "silver", "gray", "red", "blue", "yellow", "green", "other"])
            with mf_c2:
                man_event = st.selectbox("Event Action", ["entry", "exit", "detected"])
                man_cam = st.text_input("Sensor / Gate Camera ID", value="GATE-MANUAL-01")
                man_status = st.selectbox("Initial Review Status", ["verified", "pending"])

            submit_manual = st.form_submit_button("💾 Save Verified Gate Record", type="primary", use_container_width=True)
            if submit_manual:
                if not man_plate:
                    st.error("Please enter a valid license plate string.")
                else:
                    try:
                        add_event(
                            plate_text=man_plate,
                            plate_confidence=1.0,
                            vehicle_type=man_type,
                            vehicle_color=man_color,
                            event_type=man_event,
                            event_time=datetime.now().astimezone().isoformat(timespec="seconds"),
                            camera_id=man_cam,
                            review_status=man_status
                        )
                        st.success(f"Successfully recorded event for vehicle {man_plate}!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Database error: {err}")


# ==========================================
# 8. TAB 4: FLEET ANALYTICS DASHBOARD
# ==========================================

with tab4:
    dashboard_records = get_all_events(limit=5000)
    db_df = pd.DataFrame([dict(r) for r in dashboard_records])
    show_dashboard(db_df)


# ==========================================
# 9. TAB 5: PLATE DEBLUR & RESTORATION LAB
# ==========================================

with tab5:
    st.markdown("### 🔬 Plate Deblurring & Super-Resolution Lab")
    st.caption("Mathematical deconvolution, high-boost unsharp filtering, orientation deskewing & morphological character extraction to read degraded, blurry, or low-resolution license plates.")

    source_opt = st.radio(
        "Plate Source Selection",
        ["Use Stored Crop (from Image Inspection or Video Telemetry)", "Upload Blurry Plate / Vehicle Photo"],
        horizontal=True
    )

    test_plate_bgr = None

    if source_opt == "Use Stored Crop (from Image Inspection or Video Telemetry)":
        if "deblur_lab_img" in st.session_state and st.session_state["deblur_lab_img"] is not None:
            test_plate_bgr = st.session_state["deblur_lab_img"]
            source_info = st.session_state.get("deblur_lab_source_info", "Active Inspection")
            st.info(f"Loaded crop from: **{source_info}**")
        else:
            st.warning("No crop sent from Image Inspection or Video Telemetry yet. Send a plate/crop from Tab 1 or Tab 2, or choose 'Upload Blurry Plate / Vehicle Photo' above.")
    else:
        deblur_file = st.file_uploader(
            "Upload Blurry Plate Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="deblur_uploader"
        )
        if deblur_file is not None:
            raw_bytes = np.frombuffer(deblur_file.getvalue(), np.uint8)
            test_plate_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    if test_plate_bgr is not None and test_plate_bgr.size > 0:
        # Check if crop might be a full vehicle or wide shot, and offer auto-isolate plate
        h_orig, w_orig = test_plate_bgr.shape[:2]
        if w_orig > 180 and h_orig > 120:
            iso_col1, iso_col2 = st.columns([3, 1])
            with iso_col1:
                st.caption(f"Current Crop Dimensions: {w_orig}x{h_orig} px. If this is a vehicle crop, you can auto-isolate the plate below:")
            with iso_col2:
                if st.button("🎯 Auto-Extract Plate ROI", help="Runs plate detection model to tightly crop the license plate out of this vehicle image."):
                    try:
                        p_detector = PlateDetector("models/plate_detector.pt")
                        p_cands = p_detector.detect(test_plate_bgr, confidence=0.25)
                        if p_cands:
                            best_p = max(p_cands, key=lambda x: x["confidence"])
                            px1, py1, px2, py2 = best_p["bbox"]
                            cropped_p = test_plate_bgr[max(0, py1):max(0, py2), max(0, px1):max(0, px2)]
                            if cropped_p.size > 0:
                                st.session_state["deblur_lab_img"] = cropped_p
                                test_plate_bgr = cropped_p
                                st.toast("Successfully isolated license plate ROI!", icon="🎯")
                                st.rerun()
                        else:
                            st.warning("Could not auto-localize plate with confidence > 0.25. Proceeding with current crop.")
                    except Exception as err:
                        st.error(f"Error localizing plate: {err}")

        st.markdown("**Optimization Presets:**")
        preset_cols = st.columns(3)
        with preset_cols[0]:
            p_smart = st.button("🎯 Balanced / Sharp (Default)", use_container_width=True, help="Safe non-destructive enhancement. Preserves solid fonts and high confidence.")
        with preset_cols[1]:
            p_motion = st.button("🌊 Fast Vehicle Motion Deblur", use_container_width=True, help="Applies Richardson-Lucy deconvolution for directional motion streaks.")
        with preset_cols[2]:
            p_lowlight = st.button("💡 Low-Light / Glare Booster", use_container_width=True, help="High CLAHE equalization for dim or overexposed scenes.")

        if p_motion:
            st.session_state["tune_motion"] = True
            st.session_state["tune_sharp"] = 0.8
        elif p_lowlight:
            st.session_state["tune_motion"] = False
            st.session_state["tune_sharp"] = 0.6
        elif p_smart:
            st.session_state["tune_motion"] = False
            st.session_state["tune_sharp"] = 0.6

        with st.expander("🛠️ Interactive Restoration Controls & Mathematical Tuners", expanded=True):
            tune_c1, tune_c2, tune_c3 = st.columns(3)
            with tune_c1:
                st.markdown("**1. Upscaling & Deskewing**")
                scale_factor = st.slider("Bicubic Super-Res Scale", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
                enable_deskew = st.checkbox("Auto-Deskew Tilt Angle", value=True, help="Detects rotation and levels text horizontally.")
                enable_bilateral = st.checkbox("Bilateral Denoising", value=True, help="Removes sensor noise while preserving character contours.")

            with tune_c2:
                st.markdown("**2. Motion Deblurring (Richardson-Lucy)**")
                def_motion = st.session_state.get("tune_motion", False)
                enable_motion_deblur = st.checkbox(
                    "Enable Motion Deconvolution",
                    value=def_motion,
                    help="⚠️ Only enable if vehicle has linear motion smear. Applying to in-focus plates creates ringing/honeycomb artifacts."
                )
                motion_len = st.slider("Motion Kernel Size (px)", min_value=3, max_value=21, value=9, step=2)
                motion_ang = st.slider("Motion Angle (degrees)", min_value=0, max_value=180, value=0, help="0° = horizontal motion; 90° = vertical.")
                rl_iters = st.slider("Deconvolution Iterations", min_value=5, max_value=25, value=8)

            with tune_c3:
                st.markdown("**3. Edge Boost & Contrast**")
                enable_sharpen = st.checkbox("High-Boost Unsharp Mask", value=True)
                def_sharp = st.session_state.get("tune_sharp", 0.6)
                sharp_strength = st.slider("Sharpening Strength", min_value=0.2, max_value=2.5, value=def_sharp, step=0.1)
                enable_morph = st.checkbox("Morphological Contrast Leveling", value=True, help="Smooths uneven background illumination without eroding text.")
                binarize_mode = st.selectbox("Final Representation", ["Grayscale CLAHE (Recommended)", "Otsu Dynamic Threshold", "Adaptive Gaussian Threshold"])

        if st.button("⚡ Execute Deblur & Compare OCR", type="primary", use_container_width=True):
            with st.spinner("Processing multi-stage image restoration & running dual OCR analysis..."):
                ocr_engine = PlateOCR(gpu=False)
                # Baseline: run OCR on the original unmodified plate (uses internal enhancement)
                baseline_res = ocr_engine.read(test_plate_bgr)
                base_text = baseline_res.get("text", "").strip() or "UNREADABLE"
                base_conf = float(baseline_res.get("confidence", 0.0))

                # Step 1: Rescale — upscale for better resolution
                h, w = test_plate_bgr.shape[:2]
                scaled = cv2.resize(test_plate_bgr, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)

                # Step 2: Convert to grayscale early for all subsequent steps
                gray_plate = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY) if len(scaled.shape) == 3 else scaled.copy()

                # Step 3: Bilateral denoise first — cleaner image gives more reliable deskew angle
                if enable_bilateral:
                    denoised = denoise_bilateral(gray_plate, d=7, sigma_color=40, sigma_space=40)
                else:
                    denoised = gray_plate

                # Step 4: Deskew — only acts when meaningful tilt detected (>1.5°)
                tilt_detected = 0.0
                if enable_deskew:
                    denoised, tilt_detected = deskew_plate(denoised)

                # Step 5: Motion deblur (only when user explicitly enables for blurry plates)
                if enable_motion_deblur:
                    deblurred = deblur_motion(denoised, kernel_size=motion_len, angle=motion_ang, iterations=rl_iters)
                else:
                    deblurred = denoised

                # Step 6: High-boost unsharp mask to restore edge crispness
                if enable_sharpen:
                    sharpened = unsharp_mask(deblurred, sigma=1.0, strength=sharp_strength)
                else:
                    sharpened = deblurred

                # Step 7: Morphological contrast leveling (top-hat only — safe for characters)
                if enable_morph:
                    morphed = morphological_contrast(sharpened)
                else:
                    morphed = sharpened

                # Step 8: CLAHE final contrast boost
                clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
                enhanced_clahe = clahe.apply(morphed)

                # Step 9: Optional binarization for display
                if binarize_mode == "Otsu Dynamic Threshold":
                    _, final_restored = cv2.threshold(enhanced_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif binarize_mode == "Adaptive Gaussian Threshold":
                    final_restored = cv2.adaptiveThreshold(enhanced_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
                else:
                    final_restored = enhanced_clahe

                # OCR on the restored grayscale directly — skip internal re-enhancement to avoid double-processing
                restored_res = ocr_engine.read_preenhanced(final_restored)
                rest_text = restored_res.get("text", "").strip() or "UNREADABLE"
                rest_conf = float(restored_res.get("confidence", 0.0))

                st.session_state["deblur_results"] = {
                    "raw": test_plate_bgr,
                    "restored": final_restored,
                    "base_text": base_text,
                    "base_conf": base_conf,
                    "rest_text": rest_text,
                    "rest_conf": rest_conf,
                    "tilt": tilt_detected
                }

        if "deblur_results" in st.session_state:
            res = st.session_state["deblur_results"]
            st.write("")
            st.markdown("#### 🔬 Comparative Optical Restoration Analysis")

            comp_c1, comp_c2 = st.columns(2)
            with comp_c1:
                st.markdown("""
                    <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 14px; text-align: center;">
                        <h4 style="margin: 0; color: #f87171;">Original Degraded Plate</h4>
                    </div>
                """, unsafe_allow_html=True)
                st.write("")
                st.image(cv2.cvtColor(res["raw"], cv2.COLOR_BGR2RGB), width='stretch', caption="Input Image (Blur / Noise / Tilt)")
                st.markdown(f'<div class="license-plate-tag {"unreadable" if res["base_text"] == "UNREADABLE" else ""}">{res["base_text"]}</div>', unsafe_allow_html=True)
                st.caption(f"Raw OCR Confidence: {res['base_conf']:.1%}")

            with comp_c2:
                st.markdown("""
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 14px; text-align: center;">
                        <h4 style="margin: 0; color: #34d399;">Restored & Deblurred Output</h4>
                    </div>
                """, unsafe_allow_html=True)
                st.write("")
                st.image(res["restored"], width='stretch', caption="Cleaned + Deconvolved + Sharpened + Deskewed")
                st.markdown(f'<div class="license-plate-tag">{res["rest_text"]}</div>', unsafe_allow_html=True)
                st.caption(f"Restored OCR Confidence: {res['rest_conf']:.1%} | Tilt Corrected: {res['tilt']:.1f}°")

            st.write("")
            conf_diff = res["rest_conf"] - res["base_conf"]
            diff_color = "#34d399" if conf_diff >= 0 else "#f87171"
            st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 16px 20px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Recovery Delta</span>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-top: 2px;">
                            OCR: <span style="color: #f87171;">{res['base_text']}</span> ➔ <span style="color: #34d399;">{res['rest_text']}</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.85rem; color: #94a3b8;">Confidence Gain</span>
                        <div style="font-size: 1.4rem; font-weight: 800; color: {diff_color};">
                            {'+' if conf_diff >= 0 else ''}{conf_diff:.1%}
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.write("")
            if st.button("💾 Save Restored Plate Record to Inventory Ledger", type="primary", use_container_width=True):
                add_event(
                    plate_text=res["rest_text"] if res["rest_text"] != "UNREADABLE" else "RESTORED-PENDING",
                    plate_confidence=res["rest_conf"],
                    vehicle_type="car",
                    vehicle_color="restored_sample",
                    event_type="detected",
                    event_time=datetime.now().astimezone().isoformat(timespec="seconds"),
                    camera_id="DEBLUR-LAB-01",
                    review_status="verified" if res["rest_text"] != "UNREADABLE" else "pending"
                )
                st.toast(f"Successfully recorded restored plate '{res['rest_text']}' into SQLite ledger!", icon="✅")

