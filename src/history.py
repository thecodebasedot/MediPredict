"""প্রেডিকশন হিস্টোরি — SQLite-এ পূর্বাভাস সংরক্ষণ ও পুনরুদ্ধার।

স্ট্যান্ডার্ড লাইব্রেরির sqlite3 ব্যবহার করে, কোনো অতিরিক্ত নির্ভরতা ছাড়াই।
প্রতিটি পূর্বাভাস টাইমস্ট্যাম্প, রোগ, সম্ভাবনা ও ইনপুট ফিচারসহ সংরক্ষিত হয়।
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from . import config

DB_PATH = config.DATA_DIR / "history.db"


def _connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """হিস্টোরি টেবিল তৈরি করে (না থাকলে)।"""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                disease TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                probability_percent REAL NOT NULL,
                risk_level TEXT NOT NULL,
                features TEXT NOT NULL
            )
            """
        )


def save_prediction(result: dict, features: dict) -> int:
    """একটি পূর্বাভাস সংরক্ষণ করে এবং নতুন রেকর্ডের id রিটার্ন করে।"""
    init_db()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (timestamp, disease, disease_name, probability_percent, risk_level, features)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                result["disease"],
                result["disease_name"],
                result["probability_percent"],
                result["risk_level"],
                json.dumps(features, ensure_ascii=False),
            ),
        )
        return int(cur.lastrowid)


def get_history(limit: int = 20) -> list[dict]:
    """সর্বশেষ পূর্বাভাসগুলো (নতুন থেকে পুরনো) রিটার্ন করে।"""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    history = []
    for r in rows:
        item = dict(r)
        item["features"] = json.loads(item["features"])
        history.append(item)
    return history


def clear_history() -> int:
    """সব হিস্টোরি মুছে ফেলে এবং কয়টি রেকর্ড মোছা হলো তা রিটার্ন করে।"""
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM predictions")
        return cur.rowcount
