from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from engine.contracts import utc_now_iso


def default_state_db_path() -> Path:
    """Return the default coordinator state database path.

    Returns:
        Default SQLite database path.
    """

    return Path.home() / ".drunkenbot_ide" / "coordinator_state.sqlite3"


class JobStateStore:
    """SQLite-backed state store for coordinator jobs and workers."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Create a job state store.

        Args:
            db_path: SQLite file path. Defaults to the user's app data folder.
        """

        self.db_path = Path(db_path) if db_path else default_state_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_job(self, job_id: str, status: str, payload: dict[str, Any]) -> None:
        """Save or replace a managed job record.

        Args:
            job_id: Job identifier.
            status: Job status.
            payload: Serializable managed job payload.
        """

        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO jobs(job_id, status, payload_json, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (job_id, status, json.dumps(payload, indent=2), utc_now_iso()),
            )
            connection.commit()

    def load_jobs(self) -> list[dict[str, Any]]:
        """Load all persisted job payloads.

        Returns:
            Serialized managed job payloads.
        """

        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT payload_json FROM jobs ORDER BY updated_at").fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def save_worker(self, worker_id: str, status: str, payload: dict[str, Any]) -> None:
        """Save or replace a worker record.

        Args:
            worker_id: Worker identifier.
            status: Worker status.
            payload: Serializable worker payload.
        """

        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO workers(worker_id, status, payload_json, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (worker_id, status, json.dumps(payload, indent=2), utc_now_iso()),
            )
            connection.commit()

    def load_workers(self) -> list[dict[str, Any]]:
        """Load all persisted worker payloads.

        Returns:
            Serialized worker payloads.
        """

        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT payload_json FROM workers ORDER BY worker_id").fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def record_heartbeat(self, worker_id: str, payload: dict[str, Any]) -> None:
        """Record a worker heartbeat.

        Args:
            worker_id: Worker identifier.
            payload: Serializable heartbeat payload.
        """

        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO worker_heartbeats(worker_id, payload_json, received_at) VALUES(?, ?, ?)",
                (worker_id, json.dumps(payload, indent=2), utc_now_iso()),
            )
            connection.commit()

    def latest_heartbeats(self) -> dict[str, dict[str, Any]]:
        """Return the latest heartbeat for each worker.

        Returns:
            Mapping of worker ID to heartbeat payload.
        """

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT h.worker_id, h.payload_json
                FROM worker_heartbeats h
                JOIN (
                    SELECT worker_id, MAX(received_at) AS received_at
                    FROM worker_heartbeats
                    GROUP BY worker_id
                ) latest
                ON h.worker_id = latest.worker_id AND h.received_at = latest.received_at
                """
            ).fetchall()
        return {row["worker_id"]: json.loads(row["payload_json"]) for row in rows}

    def _initialize(self) -> None:
        """Create database tables when they do not exist."""

        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs(
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workers(
                    worker_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_heartbeats(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    worker_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    received_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_heartbeats_worker ON worker_heartbeats(worker_id)")
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection.

        Returns:
            SQLite connection.
        """

        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

