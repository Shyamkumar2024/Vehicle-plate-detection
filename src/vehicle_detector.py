from pathlib import Path
from ultralytics import YOLO

# COCO IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class VehicleDetector:
    def __init__(self, model_path="yolo26n.pt"):
        # If models/vehicle_detector.pt exists, use it; otherwise use pretrained weights.
        p = Path(model_path)
        if not p.exists() and model_path == "models/vehicle_detector.pt":
            model_path = "yolo26n.pt"
        self.model = YOLO(model_path)
        # Support both COCO pretrained weights and the custom 4-class YAML in this project.
        model_names = self.model.names
        self.class_names = {int(k): str(v).lower() for k, v in model_names.items()}
        self.allowed_names = {"car", "motorcycle", "motorbike", "bus", "truck", "van"}

    def detect(self, image, confidence=0.35):
        results = self.model.predict(source=image, conf=confidence, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cid = int(box.cls.item())
                name = self.class_names.get(cid, VEHICLE_CLASSES.get(cid, "unknown"))
                if name not in self.allowed_names and cid not in VEHICLE_CLASSES:
                    continue
                if name == "motorbike":
                    name = "motorcycle"
                detections.append({"class": name,
                    "class_id": cid, "confidence": float(box.conf.item()),
                    "bbox": [int(v) for v in box.xyxy[0].tolist()]})
        return detections
