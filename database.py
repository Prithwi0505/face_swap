"""
database.py
-----------
SQLite helper for tracking face-swap request records.

Schema
------
requests
    request_id  TEXT  PRIMARY KEY   -- UUID-v4 string
    timestamp   REAL                -- time spent (seconds); NULL until completed
    status      TEXT                -- "not_completed" | "in_progress" | "completed"
    request_type TEXT               -- "image" | "video"
    created_at  TEXT                -- ISO-8601 wall-clock time of request arrival
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = Path("face_swap_requests.db")

# Each thread gets its own connection so SQLite doesn't complain about
# multi-threaded access; FastAPI uses a thread-pool for sync operations.
_local = threading.local()


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def _get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, opening it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row          # dict-like row access
        conn.execute("PRAGMA journal_mode=WAL")  # better concurrent read perf
        _local.conn = conn
    return _local.conn


# ---------------------------------------------------------------------------
# Schema initialisation (called once at startup)
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create (or migrate) the requests table if it doesn't exist yet."""
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            request_id   TEXT PRIMARY KEY,
            timestamp    REAL,
            status       TEXT NOT NULL DEFAULT 'not_completed',
            request_type TEXT NOT NULL DEFAULT 'unknown',
            created_at   TEXT NOT NULL
        )
        """
    )
    conn.commit()
    print(f"[DB] SQLite database ready at '{DB_PATH}'.")


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------
def create_request(request_id: str, request_type: str) -> None:
    """
    Insert a new request row with status='not_completed'.
    Call this as soon as a request arrives.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO requests (request_id, timestamp, status, request_type, created_at)
        VALUES (?, NULL, 'not_completed', ?, ?)
        """,
        (request_id, request_type, now),
    )
    conn.commit()


def mark_in_progress(request_id: str) -> None:
    """Update a request's status to 'in_progress'."""
    conn = _get_conn()
    conn.execute(
        "UPDATE requests SET status = 'in_progress' WHERE request_id = ?",
        (request_id,),
    )
    conn.commit()


def mark_completed(request_id: str, elapsed_seconds: float) -> None:
    """
    Update a request's status to 'completed' and record how long it took.
    elapsed_seconds: wall-clock seconds from request arrival to completion.
    """
    conn = _get_conn()
    conn.execute(
        """
        UPDATE requests
        SET status    = 'completed',
            timestamp = ?
        WHERE request_id = ?
        """,
        (round(elapsed_seconds, 3), request_id),
    )
    conn.commit()


def mark_failed(request_id: str, elapsed_seconds: float) -> None:
    """Update a request as failed (treat as a special status for clarity)."""
    conn = _get_conn()
    conn.execute(
        """
        UPDATE requests
        SET status    = 'failed',
            timestamp = ?
        WHERE request_id = ?
        """,
        (round(elapsed_seconds, 3), request_id),
    )
    conn.commit()


def get_request(request_id: str) -> Optional[dict]:
    """
    Fetch a single request record by its ID.
    Returns a dict or None if not found.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM requests WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    return dict(row) if row else None


def get_all_requests() -> list[dict]:
    """Return all request records, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM requests ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]
