# Vehicle Intelligence Dashboard Upgrade

## Before applying
Back up your current `app.py`, `src/` folder, and `requirements.txt`.

## Apply
Copy the files from this ZIP into the root of your existing `vehicle-intelligence` project, allowing the updated `app.py` and matching `src` files to replace the old versions. Merge the `tests` and `training` folders. Add the packages in `requirements-additions.txt` to your existing `requirements.txt`.

## Install and run
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-additions.txt
python -m streamlit run app.py
```

## Model notes
The dashboard uses `models/vehicle_detector.pt` if present, otherwise it falls back to the root `yolo26n.pt` file. It expects a trained plate detector at `models/plate_detector.pt` before plate localization can run. This upgrade does not include datasets or trained custom weights.

## Dataset format
Prepare `data/processed/vehicle_yolo/` and `data/processed/plate_yolo/` with `images/train`, `images/val`, `images/test`, and corresponding `labels` directories. Update YAML paths and class mappings to match your converted annotations.

## Important limits
The image inspection page detects vehicles and reports when plate weights are missing. It does not create automatic entry/exit events from a still image. Production deployment needs video tracking, camera-specific crossing rules, privacy controls, and validation.
