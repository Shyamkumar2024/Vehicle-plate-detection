# Vehicle Intelligence Suite 🚗🤖

An advanced AI-powered vehicle detection, license plate recognition (ALPR), and fleet inventory management system built with **Streamlit**, **YOLOv8**, and **EasyOCR**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Tab | Feature |
|-----|---------|
| 🔍 **Image Inspection** | Upload any vehicle image — detects vehicles & license plates with YOLOv8, reads plate text with EasyOCR |
| 🎥 **Video Telemetry** | Frame-by-frame vehicle tracking on video footage |
| ✅ **Verification Station** | Review, correct, approve or reject detected plate records |
| 📊 **Fleet Analytics** | Interactive Plotly dashboards — KPI cards, activity heatmaps, fleet breakdown |
| 🔬 **Plate Deblur Lab** | Multi-stage ALPR restoration pipeline — bilateral denoising, motion deconvolution, deskewing, unsharp mask, CLAHE; side-by-side OCR comparison |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ShyamThakur507/vehicle-intelligence.git
cd vehicle-intelligence
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download model weights
The YOLOv8 model weights are **not included** in this repository (too large for git).  
Download them and place in the `models/` directory:

| File | Purpose | Download |
|------|---------|----------|
| `yolo26n.pt` | Vehicle detection (fallback) | [YOLOv8n from Ultralytics](https://github.com/ultralytics/assets/releases) |
| `models/vehicle_detector.pt` | Fine-tuned vehicle detector | Train yourself — see `training/` |
| `models/plate_detector.pt` | License plate detector | Train yourself — see `training/` |

> **EasyOCR models** are downloaded automatically on first run to `models/easyocr/`

### 5. Run the app
```bash
streamlit run app.py --server.port 8502
```

Open **http://localhost:8502** in your browser.

---

## 📁 Project Structure

```
vehicle-intelligence/
├── app.py                  # Main Streamlit app (5 tabs)
├── requirements.txt        # Python dependencies
├── src/
│   ├── vehicle_detector.py # YOLOv8 vehicle detection
│   ├── plate_detector.py   # License plate detection
│   ├── ocr.py              # EasyOCR + ALPR grammar correction
│   ├── enhancement.py      # Image enhancement pipeline
│   ├── tracker.py          # Multi-object vehicle tracking
│   ├── database.py         # SQLite CRUD operations
│   ├── dashboard.py        # Plotly analytics charts
│   ├── quality.py          # Image quality assessment
│   ├── attributes.py       # Vehicle color estimation
│   └── utils.py            # Shared utilities & paths
├── models/                 # Model weights (not in git — download separately)
├── data/                   # SQLite database (created at runtime)
├── outputs/                # Saved detections
├── notebooks/              # Jupyter exploration notebooks
├── training/               # Model training scripts
└── tests/                  # Unit tests
```

---

## 🧠 Technical Architecture

```
Input (Image/Video)
        │
        ▼
 VehicleDetector (YOLOv8)
        │
        ▼
 PlateDetector (YOLOv8)
        │
        ▼
 Enhancement Pipeline
   ├─ Bicubic Upscaling
   ├─ Bilateral Denoising
   ├─ Deskew (minAreaRect)
   ├─ Richardson-Lucy Deconvolution (optional)
   ├─ Unsharp Mask
   ├─ Morphological Top-Hat
   └─ CLAHE Contrast
        │
        ▼
 EasyOCR → correct_plate_syntax()
        │
        ▼
 SQLite Inventory Ledger
        │
        ▼
 Streamlit Dashboard
```

### ALPR Grammar Correction
Indian number plate format: `[State:2][RTO:2][Series:2][Number:4]`  
Example: `BR01PC3938` — Bihar, RTO 01, Series PC, Number 3938

The `correct_plate_syntax()` function resolves OCR character ambiguities:
- Position 0,1 (State): `0→O`, `1→I`, `8→B` etc.
- Position 2,3 (RTO): `O→0`, `I→1`, `S→5` etc.
- Position 4,5 (Series): `0→O`, `1→I` etc.
- Position 6-9 (Number): `O→0`, `I→1` etc.

---

## 🛠️ Configuration

### Streamlit config (`.streamlit/config.toml`)
```toml
[server]
headless = true
port = 8502

[theme]
base = "dark"
```

---

## 📦 Dependencies

```
streamlit>=1.32
ultralytics>=8.3.0        # YOLOv8
easyocr>=1.7.1            # OCR engine
opencv-python>=4.8.0      # Image processing
scikit-image>=0.22        # Richardson-Lucy deconvolution
numpy>=1.24
pandas>=2.0
plotly>=5.0
Pillow>=10.0
```

---

## 🧪 Running Tests
```bash
python -m pytest tests/ -v
```

---

## 📸 Screenshots

Upload an image on the **Image Inspection** tab → the system detects vehicles, crops the plate, runs the enhancement pipeline, and reads the number plate text with confidence score.

Use the **Plate Deblur Lab** tab to upload a blurry/noisy plate and compare raw vs. restored OCR results side-by-side.

---

## 🚢 Deployment (Render)

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for full Render deployment guide.

**Minimum requirements:** 2GB RAM (Standard plan) due to EasyOCR + PyTorch.

---

## 👨‍💻 Author

**Shyam Thakur**  
GitHub: [@ShyamThakur507](https://github.com/ShyamThakur507)  
Email: thakurshyam507@gmail.com

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
