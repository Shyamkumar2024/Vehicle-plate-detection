from datetime import datetime, timezone
from src.database import add_event
from src.inventory import update_vehicle_status

class EventProcessor:
    """Minimal event writer. Call only after your configured line-crossing logic fires."""
    def __init__(self, camera_id="camera_1", cooldown_seconds=20, db_path=None):
        self.camera_id = camera_id
        self.cooldown_seconds = cooldown_seconds
        self.db_path = db_path
        self.last_event = {}

    def record_crossing(self, plate_text, plate_confidence, vehicle_type, event_type,
                        track_id=None, vehicle_color=None, image_path=None,
                        review_status="pending", event_time=None):
        event_time = event_time or datetime.now(timezone.utc).isoformat()
        key = (plate_text or f"unknown-track-{track_id}", event_type)
        now = datetime.now(timezone.utc).timestamp()
        if key in self.last_event and now - self.last_event[key] < self.cooldown_seconds:
            return None
        self.last_event[key] = now
        event_id = add_event(plate_text, plate_confidence, vehicle_type, event_type,
            event_time, self.camera_id, track_id, vehicle_color, image_path,
            review_status, self.db_path)
        update_vehicle_status(plate_text, vehicle_type, vehicle_color, event_type,
                              event_time, review_status, self.db_path)
        return event_id
