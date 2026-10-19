"""Bounded persistent run storage. SQLite transactions survive process restarts."""

import json
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Store:
    def __init__(self, root, quota_mb):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "runs.sqlite3"
        self.quota = quota_mb * 1024 * 1024
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, body TEXT);
                CREATE TABLE IF NOT EXISTS predictions(id TEXT PRIMARY KEY, run_id TEXT, body TEXT, snapshot BLOB);
                CREATE INDEX IF NOT EXISTS events_run ON events(run_id, seq);
                CREATE INDEX IF NOT EXISTS predictions_run ON predictions(run_id);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def check_space(self, needed=8 * 1024 * 1024):
        used = sum(p.stat().st_size for p in self.root.glob("runs.sqlite3*"))
        if (
            used + needed > self.quota
            or shutil.disk_usage(self.root).free < needed + 64 * 1024 * 1024
        ):
            raise OSError("运行日志空间不足，请归档历史数据后重启服务")

    def save_run(self, run):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO runs VALUES (?, ?)", (run["id"], dumps(run)))

    def get_run(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT body FROM runs WHERE id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def runs(self):
        with self.connect() as db:
            return [
                json.loads(r[0])
                for r in db.execute("SELECT body FROM runs ORDER BY rowid DESC LIMIT 100")
            ]

    def event(self, run_id, kind, payload):
        body = {
            "schema_version": 1,
            "run_id": run_id,
            "type": kind,
            "time_unix": time.time(),
            "payload": payload,
        }
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO events(run_id,body) VALUES (?,?)", (run_id, dumps(body))
            )
            body["event_seq"] = cursor.lastrowid
        return body

    def events(self, run_id, after=0, limit=200):
        with self.connect() as db:
            rows = db.execute(
                "SELECT seq,body FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                (run_id, after, limit),
            ).fetchall()
        return [{**json.loads(body), "event_seq": seq} for seq, body in rows]

    def prediction(self, result, snapshot):
        blob = snapshot.pack(compressed=True)
        self.check_space(len(blob) * 2 + 1024 * 1024)
        with self.connect() as db:
            db.execute(
                "INSERT INTO predictions VALUES (?,?,?,?)",
                (result["id"], result["run_id"], dumps(result), blob),
            )

    def get_prediction(self, prediction_id, with_snapshot=False):
        with self.connect() as db:
            columns = "body,snapshot" if with_snapshot else "body"
            row = db.execute(
                f"SELECT {columns} FROM predictions WHERE id=?", (prediction_id,)
            ).fetchone()
        if not row:
            return None
        return (json.loads(row[0]), row[1]) if with_snapshot else json.loads(row[0])

    def predictions(self, run_id):
        with self.connect() as db:
            return [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT body FROM predictions WHERE run_id=? ORDER BY rowid DESC LIMIT 100",
                    (run_id,),
                )
            ]

    def update_diagnostic(self, prediction_id, report):
        self.check_space(1024 * 1024)
        with self.connect() as db:
            row = db.execute("SELECT body FROM predictions WHERE id=?", (prediction_id,)).fetchone()
            if row is None:
                raise KeyError("Prediction missing for offline diagnostic")
            value = json.loads(row[0])
            value["offline_ik"] = report
            db.execute("UPDATE predictions SET body=? WHERE id=?", (dumps(value), prediction_id))

    def interrupt_pending_diagnostics(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,body FROM predictions WHERE json_extract(body, '$.offline_ik.status')='pending'"
            ).fetchall()
            for key, body in rows:
                value = json.loads(body)
                value["offline_ik"] = {
                    "status": "interrupted",
                    "passed": False,
                    "execution_verified": False,
                    "error": "服务重启，离线检查未完成",
                    "completed_unix": time.time(),
                }
                db.execute("UPDATE predictions SET body=? WHERE id=?", (dumps(value), key))
