# Render Deployment Guide — Vehicle Intelligence Suite

This guide explains how to deploy this project to Render.

---

## 🛠️ Deployment Configuration (Automatic with render.yaml or Manual)

### If using Manual Web Service Setup:

1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** ➔ **Web Service**.
3. Connect your GitHub repository: `Shyamkumar2024/Vehicle-plate-detection`.
4. Enter the following settings:
   - **Name:** `vehicle-intelligence`
   - **Runtime:** `Python 3`
   - **Branch:** `main`
   - **Region:** Any (e.g. Frankfurt, Oregon, Singapore)
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
   - **Instance Type:** `Free` (or `Standard` 2GB for faster OCR processing)

5. Under **Environment Variables**, add:
   - `PYTHONPATH`: `.`
   - `PYTHON_VERSION`: `3.10.12`

6. Click **Deploy Web Service**.

---

## 🔍 Key Fixes Applied for Render
- **`opencv-python-headless`**: Fixes the missing `libGL.so.1` system library error on Linux.
- **Lightweight CPU PyTorch**: `--extra-index-url https://download.pytorch.org/whl/cpu` prevents out-of-disk and build timeout errors.
- **`0.0.0.0` Binding**: Ensures Render's proxy can route external web traffic to Streamlit.
- **Included Model Weights**: `models/plate_detector.pt` and `yolo26n.pt` are included directly in the repo.
