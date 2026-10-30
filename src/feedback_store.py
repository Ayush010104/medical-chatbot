"""Clinician feedback storage (SQLite) — the human-review loop for the graph.

Deliberately not another LLM call: the whole point is to improve retrieval
quality using clinician judgment instead of paying for more extraction runs.
A flagged graph fact gets reviewed by a person and deleted at the source; it
never resurfaces because it's gone from Neo4j, not just hidden in the UI.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

DB_PATH = os.environ.get("FEEDBACK_DB_PATH", "feedback.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT,
            answer TEXT,
            item_type TEXT,
            item_source TEXT,
            item_excerpt TEXT,
            item_meta TEXT,
            rating TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()


def add_feedback(
    question: str,
    rating: str,
    answer: Optional[str] = None,
    item_type: Optional[str] = None,
    item_source: Optional[str] = None,
    item_excerpt: Optional[str] = None,
    item_meta: Optional[dict] = None,
) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO feedback "
        "(created_at, question, answer, item_type, item_source, item_excerpt, item_meta, rating) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            question[:2000] if question else None,
            answer[:4000] if answer else None,
            item_type,
            item_source[:500] if item_source else None,
            item_excerpt[:2000] if item_excerpt else None,
            json.dumps(item_meta) if item_meta else None,
            rating,
        ),
    )
    conn.commit()
    conn.close()


def list_feedback(unresolved_only: bool = True) -> List[dict]:
    conn = _connect()
    query = "SELECT * FROM feedback"
    if unresolved_only:
        query += " WHERE resolved = 0"
    query += " ORDER BY created_at DESC"
    rows = [dict(r) for r in conn.execute(query).fetchall()]
    conn.close()
    for row in rows:
        row["item_meta"] = json.loads(row["item_meta"]) if row["item_meta"] else None
    return rows


def get_feedback(feedback_id: int) -> Optional[dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,)).fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    result["item_meta"] = json.loads(result["item_meta"]) if result["item_meta"] else None
    return result


def mark_resolved(feedback_id: int) -> None:
    conn = _connect()
    conn.execute("UPDATE feedback SET resolved = 1 WHERE id = ?", (feedback_id,))
    conn.commit()
    conn.close()
