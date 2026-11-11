# Render Deployment Guide — Vehicle Intelligence Suite

## ⚠️ RAM Requirements

| Render Plan | RAM | Status |
|-------------|-----|--------|
| Free / Starter | 512MB | ❌ Too small (EasyOCR+PyTorch needs ~2GB) |
| Standard | 2GB | ✅ Minimum viable |
| Pro | 4GB | ✅ Recommended |

---

## Step 1 — Prepare for Render

### 1a. Replace `opencv-python` with headless version
Render Linux servers don't have a display. Edit `requirements.txt`:
```
# Change this:
opencv-python>=4.8.0
# To this:
opencv-python-headless>=4.8.0
```

### 1b. Create `.streamlit/config.toml`
```toml
[server]
headless = true
port = 10000
enableCORS = false
enableXsrfProtection = false

[theme]
base = "dark"
```

---

## Step 2 — Handle Model Weights

Model `.pt` files are too large for git. Options:

**Option A — Render Persistent Disk (Recommended)**
1. Add a Persistent Disk in Render dashboard (mount at `/data`)
2. SSH into instance and upload `.pt` files to `/data/models/`
3. Update `src/utils.py` ROOT or set env var `MODELS_DIR=/data/models`

**Option B — Download at startup**
Create `startup.sh`:
```bash
#!/bin/bash
mkdir -p models
# Download your model from a cloud bucket (S3, GCS, etc.)
# wget -q "https://your-bucket.s3.amazonaws.com/plate_detector.pt" -O models/plate_detector.pt
streamlit run app.py --server.port $PORT --server.headless true
```

---

## Step 3 — Create `render.yaml`

```yaml
services:
  - type: web
    name: vehicle-intelligence
    env: python
    plan: standard        # Minimum 2GB RAM
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.headless true --browser.gatherUsageStats false
    envVars:
      - key: PYTHONPATH
        value: .
    disk:
      name: vehicle-data
      mountPath: /data
      sizeGB: 5
```

---

## Step 4 — Deploy on Render

1. Push this repo to GitHub (already done ✅)
2. Go to [render.com](https://render.com) → **New +** → **Web Service**
3. Connect your GitHub account
4. Select `ShyamThakur507/vehicle-intelligence`
5. Configure:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.headless true`
   - **Plan**: Standard (2GB) or Pro (4GB)
6. Click **Create Web Service**

---

## Step 5 — Environment Variables on Render

| Key | Value | Purpose |
|-----|-------|---------|
| `PYTHONPATH` | `.` | Allow `from src.xxx import` |
| `MODELS_DIR` | `/data/models` | (Optional) Custom models path |
| `DB_PATH` | `/data/vehicle_inventory.db` | Persistent SQLite path |

---

## Notes

- **EasyOCR models** (~100MB) download automatically on first request — set `model_storage_directory` to persistent disk path to avoid re-downloading on each restart
- **SQLite** data is lost on restart unless using Persistent Disk
- Build time is long (~10-15 min) due to PyTorch (~2GB install)

