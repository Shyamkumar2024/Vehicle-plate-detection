import sqlite3
from pathlib import Path
from src.utils import ROOT
DB_PATH = ROOT / "data/vehicle_inventory.db"

def get_connection(db_path=None):
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database(db_path=None):
    with get_connection(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS vehicle_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT, plate_confidence REAL, vehicle_type TEXT,
            vehicle_color TEXT, event_type TEXT NOT NULL, event_time TEXT NOT NULL,
            camera_id TEXT, track_id INTEGER, image_path TEXT,
            review_status TEXT NOT NULL DEFAULT 'pending')""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_time ON vehicle_events(plate_text, event_time)")
        conn.execute("""CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT UNIQUE NOT NULL, vehicle_type TEXT, vehicle_color TEXT,
            status TEXT NOT NULL DEFAULT 'outside', last_seen TEXT,
            review_status TEXT NOT NULL DEFAULT 'pending')""")

def add_event(plate_text, plate_confidence, vehicle_type, event_type, event_time,
              camera_id="camera_1", track_id=None, vehicle_color=None,
              image_path=None, review_status="pending", db_path=None):
    with get_connection(db_path) as conn:
        cur = conn.execute("""INSERT INTO vehicle_events
            (plate_text, plate_confidence, vehicle_type, vehicle_color, event_type,
             event_time, camera_id, track_id, image_path, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (plate_text, plate_confidence, vehicle_type, vehicle_color, event_type,
             event_time, camera_id, track_id, image_path, review_status))
        return cur.lastrowid

def get_all_events(limit=500, db_path=None):
    with get_connection(db_path) as conn:
        return conn.execute("SELECT * FROM vehicle_events ORDER BY event_time DESC LIMIT ?", (int(limit),)).fetchall()

def get_vehicles(db_path=None):
    with get_connection(db_path) as conn:
        return conn.execute("SELECT * FROM vehicles ORDER BY last_seen DESC").fetchall()

def update_event_status(event_id, review_status, db_path=None):
    with get_connection(db_path) as conn:
        conn.execute("UPDATE vehicle_events SET review_status = ? WHERE id = ?", (review_status, int(event_id)))

def update_event_plate(event_id, plate_text, review_status="verified", db_path=None):
    with get_connection(db_path) as conn:
        conn.execute("UPDATE vehicle_events SET plate_text = ?, review_status = ? WHERE id = ?", (plate_text.strip().upper(), review_status, int(event_id)))

def delete_event(event_id, db_path=None):
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM vehicle_events WHERE id = ?", (int(event_id),))

def get_event_counts(db_path=None):
    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM vehicle_events").fetchone()[0]
        verified = conn.execute("SELECT COUNT(*) FROM vehicle_events WHERE review_status = 'verified'").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM vehicle_events WHERE review_status = 'pending'").fetchone()[0]
        rejected = conn.execute("SELECT COUNT(*) FROM vehicle_events WHERE review_status = 'rejected'").fetchone()[0]
        return {"total": total, "verified": verified, "pending": pending, "rejected": rejected}

