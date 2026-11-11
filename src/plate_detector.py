from pathlib import Path
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path="models/plate_detector.pt"):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Plate model not found: {model_path}. Train it using training/train_detector.py "
                "and training/plate_data.yaml, or pass a valid plate detector weight path."
            )
        self.model = YOLO(model_path)

    def detect(self, image, confidence=0.30):
        results = self.model.predict(source=image, conf=confidence, verbose=False)
        out = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                out.append({"bbox": [int(v) for v in box.xyxy[0].tolist()],
                            "confidence": float(box.conf.item())})
        return out
