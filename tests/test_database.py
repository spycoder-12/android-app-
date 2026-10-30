import os
import sqlite3

import pytest

from database.db_manager import DatabaseError, DBManager


@pytest.fixture
def db(tmp_path):
    return DBManager(str(tmp_path / "h.db"))


def add(db, path=None, label="Early blight", conf=0.9, status="ok"):
    return db.add_detection(image_path=path, plant="tomato", predicted_key="early_blight",
                            predicted_label=label, confidence=conf, status=status,
                            top_k=[{"key": "early_blight", "label": label, "prob": conf}], model_version="t")


def test_insert_and_get(db, tmp_path):
    img = tmp_path / "a.jpg"; img.write_bytes(b"x")
    rid = add(db, str(img))
    r = db.get(rid)
    assert r["predicted_label"] == "Early blight" and r["confidence"] == 0.9
    assert r["top_k"][0]["key"] == "early_blight" and r["image_exists"] is True
    assert r["created_at"]


def test_list_newest_first_and_count(db):
    ids = [add(db, label=f"L{i}") for i in range(3)]
    assert db.count() == 3
    assert [r["id"] for r in db.list()] == ids[::-1]
    assert len(db.list(limit=2)) == 2


def test_missing_image_flagged_not_crashing(db, tmp_path):
    rid = add(db, str(tmp_path / "gone.jpg"))
    assert db.get(rid)["image_exists"] is False


def test_delete_removes_row_and_image(db, tmp_path):
    img = tmp_path / "a.jpg"; img.write_bytes(b"x")
    rid = add(db, str(img))
    assert db.delete(rid) is True
    assert db.get(rid) is None and not img.exists()
    assert db.delete(rid) is False        # already gone
    assert db.delete("nope") is False


def test_clear_all(db, tmp_path):
    imgs = []
    for i in range(3):
        p = tmp_path / f"{i}.jpg"; p.write_bytes(b"x"); imgs.append(p); add(db, str(p))
    db.clear_all()
    assert db.count() == 0 and not any(p.exists() for p in imgs)


def test_get_unknown_id(db):
    assert db.get("missing") is None


def test_persists_across_instances(tmp_path):
    p = str(tmp_path / "h.db")
    rid = add(DBManager(p))
    assert DBManager(p).get(rid) is not None


def test_corrupt_db_file_gives_user_message(tmp_path):
    p = tmp_path / "bad.db"; p.write_bytes(b"this is not a database" * 100)
    with pytest.raises(DatabaseError):
        DBManager(str(p)).count()


def test_unwritable_location(tmp_path):
    f = tmp_path / "file"; f.write_text("x")
    with pytest.raises(DatabaseError):
        DBManager(str(f / "sub" / "h.db"))   # parent is a file
