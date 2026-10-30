"""SQLite history storage. Short-lived connections per call (thread-safe, crash-safe)."""
import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime

SCHEMA_VERSION = 1


class DatabaseError(Exception):
    """User-presentable database failure."""


class DBManager:
    def __init__(self, db_path):
        self.db_path = db_path
        try:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            self._init()
        except (sqlite3.Error, OSError) as exc:
            raise DatabaseError(f"Could not open the history database: {exc}") from exc

    def _connect(self):
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def _init(self):
        with closing(self._connect()) as con, con:
            con.execute("""CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                image_path TEXT,
                plant TEXT NOT NULL,
                predicted_key TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                top_k TEXT,
                model_version TEXT,
                created_at TEXT NOT NULL)""")
            con.execute("CREATE INDEX IF NOT EXISTS idx_created ON detections(created_at DESC)")
            con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _row(r):
        d = dict(r)
        d["top_k"] = json.loads(d["top_k"]) if d.get("top_k") else []
        d["image_exists"] = bool(d.get("image_path")) and os.path.isfile(d["image_path"])
        return d

    def add_detection(self, *, image_path, plant, predicted_key, predicted_label,
                      confidence, status, top_k=None, model_version=""):
        rid = uuid.uuid4().hex
        try:
            with closing(self._connect()) as con, con:
                con.execute("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (rid, image_path, plant, predicted_key, predicted_label, float(confidence),
                             status, json.dumps(top_k or []), model_version,
                             datetime.now().isoformat(timespec="seconds")))
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not save this result: {exc}") from exc
        return rid

    def get(self, rid):
        try:
            with closing(self._connect()) as con:
                r = con.execute("SELECT * FROM detections WHERE id=?", (rid,)).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not read history: {exc}") from exc
        return self._row(r) if r else None

    def list(self, limit=100, offset=0):
        try:
            with closing(self._connect()) as con:
                rows = con.execute("SELECT * FROM detections ORDER BY created_at DESC, rowid DESC "
                                   "LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not read history: {exc}") from exc
        return [self._row(r) for r in rows]

    def count(self):
        try:
            with closing(self._connect()) as con:
                return con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not read history: {exc}") from exc

    def delete(self, rid, delete_image=True):
        """Delete one record (and its stored image). Returns True if a record was removed."""
        rec = self.get(rid)
        if not rec:
            return False
        try:
            with closing(self._connect()) as con, con:
                con.execute("DELETE FROM detections WHERE id=?", (rid,))
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not delete this record: {exc}") from exc
        if delete_image:
            self._remove_file(rec.get("image_path"))
        return True

    def clear_all(self, delete_images=True):
        paths = [r["image_path"] for r in self.list(limit=1_000_000)] if delete_images else []
        try:
            with closing(self._connect()) as con, con:
                con.execute("DELETE FROM detections")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not clear history: {exc}") from exc
        for p in paths:
            self._remove_file(p)

    @staticmethod
    def _remove_file(path):
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
