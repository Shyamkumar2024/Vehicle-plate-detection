from src.database import initialize_database, add_event, get_all_events

def test_event_insert(tmp_path):
    db = tmp_path / "test.db"
    initialize_database(db)
    add_event("TEST123", 0.99, "car", "entry", "2026-01-01T10:00:00Z", db_path=db)
    rows = get_all_events(db_path=db)
    assert len(rows) == 1
    assert rows[0]["plate_text"] == "TEST123"
