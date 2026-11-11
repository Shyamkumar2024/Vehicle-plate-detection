from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]


def ensure_project_dirs():
    for rel in ["data/raw/vehicles", "data/raw/license_plates", "data/raw/videos",
                "data/processed", "data/annotations", "data/splits", "models",
                "outputs/detections", "outputs/plate_crops", "outputs/annotated_videos",
                "outputs/reports"]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image
