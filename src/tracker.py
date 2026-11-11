from pathlib import Path
from ultralytics import YOLO
from src.vehicle_detector import VEHICLE_CLASSES

class VehicleTracker:
    def __init__(self, model_path="models/vehicle_detector.pt"):
        if not Path(model_path).exists():
            model_path = "yolo26n.pt"
        self.model = YOLO(model_path)
        self.class_names = {int(k): str(v).lower() for k, v in self.model.names.items()}

    def track_frame(self, frame, confidence=0.35):
        results = self.model.track(source=frame, persist=True, tracker="bytetrack.yaml",
                                   conf=confidence, verbose=False)
        tracks = []
        for result in results:
            if result.boxes is None:
                continue
            box_ids = result.boxes.id
            for i, box in enumerate(result.boxes):
                cid = int(box.cls.item())
                name = self.class_names.get(cid, VEHICLE_CLASSES.get(cid, "unknown"))
                if name not in {"car", "motorcycle", "motorbike", "bus", "truck", "van"} and cid not in VEHICLE_CLASSES:
                    continue
                if name == "motorbike":
                    name = "motorcycle"
                tid = int(box_ids[i].item()) if box_ids is not None else (i + 1)
                tracks.append({
                    "track_id": tid,
                    "class": name,
                    "bbox": [int(v) for v in box.xyxy[0].tolist()],
                    "confidence": float(box.conf.item())
                })
        return tracks
