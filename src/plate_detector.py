from pathlib import Path
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path="models/plate_detector.pt"):
        p = Path(model_path)
        if not p.exists():
            if Path("plate_detector.pt").exists():
                model_path = "plate_detector.pt"
            elif Path("models/plate_detector.pt").exists():
                model_path = "models/plate_detector.pt"
            else:
                model_path = "yolov8n.pt"  # Fallback gracefully
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
