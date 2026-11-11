from src.database import get_connection

def update_vehicle_status(plate_text, vehicle_type, vehicle_color, event_type, event_time,
                          review_status="verified", db_path=None):
    """Update current inventory only for a verified, non-empty plate and valid event."""
    if review_status != "verified" or not plate_text:
        return False
    event_type = event_type.lower()
    if event_type not in {"entry", "exit"}:
        raise ValueError("event_type must be 'entry' or 'exit'")
    status = "inside" if event_type == "entry" else "outside"
    with get_connection(db_path) as conn:
        conn.execute("""INSERT INTO vehicles (plate_text, vehicle_type, vehicle_color, status, last_seen, review_status)
            VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(plate_text) DO UPDATE SET
            vehicle_type=excluded.vehicle_type, vehicle_color=excluded.vehicle_color,
            status=excluded.status, last_seen=excluded.last_seen, review_status=excluded.review_status""",
            (plate_text, vehicle_type, vehicle_color, status, event_time, review_status))
    return True

def get_current_inventory(db_path=None):
    with get_connection(db_path) as conn:
        return conn.execute("SELECT * FROM vehicles WHERE status='inside' ORDER BY last_seen DESC").fetchall()
