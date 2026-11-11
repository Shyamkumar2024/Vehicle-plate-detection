
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "training" / "plate_data.yaml"
BASE_MODEL = ROOT / "yolo26n.pt"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing dataset config: {DATA_YAML}")

    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"Missing model: {BASE_MODEL}")

    model = YOLO(str(BASE_MODEL))

    model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=8,
        workers=2,
        project=str(ROOT / "runs" / "plate"),
        name="plate_detector",
    )

    print("Training complete.")
    print("Check runs/plate/plate_detector/weights/best.pt")

if __name__ == "__main__":
    main()
