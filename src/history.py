"""প্রেডিকশন হিস্টোরি — SQLite-এ পূর্বাভাস সংরক্ষণ ও পুনরুদ্ধার।

স্ট্যান্ডার্ড লাইব্রেরির sqlite3 ব্যবহার করে, কোনো অতিরিক্ত নির্ভরতা ছাড়াই।
প্রতিটি পূর্বাভাস টাইমস্ট্যাম্প, রোগী আইডি, রোগ, সম্ভাবনা ও ইনপুট ফিচারসহ
সংরক্ষিত হয় — যা রোগীভিত্তিক ঝুঁকি-ট্রেন্ড ও সমষ্টিগত অ্যানালিটিক্স সম্ভব করে।
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
    """হিস্টোরি টেবিল তৈরি করে এবং পুরনো স্কিমা মাইগ্রেট করে।"""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT NOT NULL DEFAULT '',
                disease TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                probability_percent REAL NOT NULL,
                risk_level TEXT NOT NULL,
                features TEXT NOT NULL
            )
            """
        )
        # পুরনো DB-তে patient_id কলাম না থাকলে যোগ করা হয়
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(predictions)")}
        if "patient_id" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN patient_id TEXT NOT NULL DEFAULT ''")


def save_prediction(result: dict, features: dict, patient_id: str = "") -> int:
    """একটি পূর্বাভাস সংরক্ষণ করে এবং নতুন রেকর্ডের id রিটার্ন করে।"""
    init_db()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (timestamp, patient_id, disease, disease_name,
                 probability_percent, risk_level, features)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                (patient_id or "").strip(),
                result["disease"],
                result["disease_name"],
                result["probability_percent"],
                result["risk_level"],
                json.dumps(features, ensure_ascii=False),
            ),
        )
        return int(cur.lastrowid)


def _rows_to_dicts(rows) -> list[dict]:
    out = []
    for r in rows:
        item = dict(r)
        item["features"] = json.loads(item["features"])
        out.append(item)
    return out


def get_history(limit: int = 20) -> list[dict]:
    """সর্বশেষ পূর্বাভাসগুলো (নতুন থেকে পুরনো) রিটার্ন করে।"""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return _rows_to_dicts(rows)


def get_all() -> list[dict]:
    """সব রেকর্ড রিটার্ন করে (অ্যানালিটিক্সের জন্য)।"""
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id ASC").fetchall()
    return _rows_to_dicts(rows)


def list_patients() -> list[dict]:
    """রোগীভিত্তিক সারাংশ: প্রতিটি patient_id-এর রেকর্ড সংখ্যা ও সর্বশেষ সময়।"""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT patient_id,
                   COUNT(*) AS count,
                   MAX(timestamp) AS last_seen
            FROM predictions
            WHERE patient_id != ''
            GROUP BY patient_id
            ORDER BY last_seen DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_patient_trend(patient_id: str, disease: str | None = None) -> list[dict]:
    """একজন রোগীর সময়ক্রমিক পূর্বাভাস (পুরনো থেকে নতুন) — ট্রেন্ড চার্টের জন্য।"""
    init_db()
    query = "SELECT * FROM predictions WHERE patient_id = ?"
    params: list = [patient_id]
    if disease:
        query += " AND disease = ?"
        params.append(disease)
    query += " ORDER BY id ASC"
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return _rows_to_dicts(rows)


def clear_history() -> int:
    """সব হিস্টোরি মুছে ফেলে এবং কয়টি রেকর্ড মোছা হলো তা রিটার্ন করে।"""
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM predictions")
        return cur.rowcount
