from src.database import initialize_database
from src.inventory import update_vehicle_status, get_current_inventory

def test_verified_entry_and_exit(tmp_path):
    db = tmp_path / "test.db"
    initialize_database(db)
    assert update_vehicle_status("TEST123", "car", "white", "entry", "2026-01-01T10:00:00Z", db_path=db)
    assert len(get_current_inventory(db_path=db)) == 1
    update_vehicle_status("TEST123", "car", "white", "exit", "2026-01-01T11:00:00Z", db_path=db)
    assert len(get_current_inventory(db_path=db)) == 0

def test_unverified_is_not_added(tmp_path):
    db = tmp_path / "test.db"
    initialize_database(db)
    assert not update_vehicle_status("TEST123", "car", "white", "entry", "2026-01-01T10:00:00Z", review_status="pending", db_path=db)
