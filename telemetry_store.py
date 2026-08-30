from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


METRIC_FIELDS = (
    "epoch",
    "total_epochs",
    "total_steps",
    "train_loss",
    "val_loss",
    "learning_rate",
    "grad_norm",
    "weight_norm",
    "update_ratio",
    "tokens_per_second",
    "samples_per_second",
    "vram_allocated_gb",
    "vram_reserved_gb",
    "gpu_memory_percent",
    "system_cpu_percent",
    "system_ram_percent",
    "data_loader_workers",
    "sample_text",
)
FORCE_FLUSH_EVENT_TYPES = {
    "validation",
    "checkpoint",
    "warning",
    "stop",
    "failure",
    "completion",
}


def telemetry_db_path(model_dir: Path) -> Path:
    """Return the telemetry SQLite path for a model directory."""

    return model_dir / "training_telemetry.sqlite"


def ensure_schema(connection: sqlite3.Connection) -> None:
    """Create or migrate the worker-owned telemetry schema."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS live_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            recorded_at REAL NOT NULL,
            step INTEGER NOT NULL,
            epoch INTEGER,
            total_epochs INTEGER,
            total_steps INTEGER,
            train_loss REAL,
            val_loss REAL,
            learning_rate REAL,
            grad_norm REAL,
            weight_norm REAL,
            update_ratio REAL,
            tokens_per_second REAL,
            samples_per_second REAL,
            vram_allocated_gb REAL,
            vram_reserved_gb REAL,
            gpu_memory_percent REAL,
            system_cpu_percent REAL,
            system_ram_percent REAL,
            data_loader_workers INTEGER,
            sample_text TEXT,
            payload_json TEXT
        )
        """
    )
    metric_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(live_metrics)")
    }
    if "sample_text" not in metric_columns:
        connection.execute("ALTER TABLE live_metrics ADD COLUMN sample_text TEXT")
    if "payload_json" not in metric_columns:
        connection.execute("ALTER TABLE live_metrics ADD COLUMN payload_json TEXT")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            recorded_at REAL NOT NULL,
            event_type TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT,
            percent INTEGER,
            payload_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_live_metrics_run_id_id ON live_metrics(run_id, id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_telemetry_events_run_id_id "
        "ON telemetry_events(run_id, id)"
    )


class TelemetryWriter:
    """Persist a run through one WAL connection with bounded batching."""

    def __init__(
        self,
        db_path: Path,
        run_id: str,
        batch_size: int = 25,
        flush_interval_seconds: float = 1.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.run_id = run_id
        self.batch_size = max(1, int(batch_size))
        self.flush_interval_seconds = max(0.0, float(flush_interval_seconds))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path, timeout=15.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA busy_timeout=15000")
        ensure_schema(self._connection)
        self._connection.commit()
        self._pending_metrics: list[tuple[Any, ...]] = []
        self._pending_events: list[tuple[Any, ...]] = []
        self._last_flush_at = time.monotonic()
        self._last_metric_row_id: Optional[int] = None
        self._closed = False

    def record(self, event: dict[str, Any]) -> None:
        """Persist a progress event as a sample or durable lifecycle record."""

        event_type = str(event.get("event_type") or _infer_event_type(event))
        if event_type == "metrics" and event.get("step") is not None:
            self.record_metric(event)
            return
        self.record_event(event_type, event)

    def record_metric(self, event: dict[str, Any]) -> Optional[int]:
        """Queue one coalescible metric sample."""

        recorded_at = float(event.get("recorded_at") or time.time())
        values = [event.get(field) for field in METRIC_FIELDS]
        self._pending_metrics.append(
            (
                self.run_id,
                recorded_at,
                int(event["step"]),
                *values,
                _json_payload(event),
            )
        )
        self._flush_if_due()
        return self._last_metric_row_id if not self._pending_metrics else None

    def record_event(self, event_type: str, event: dict[str, Any]) -> None:
        """Queue one durable lifecycle, warning, checkpoint, or log event."""

        level = str(
            event.get("level")
            or ("warning" if event_type == "warning" else "info")
        )
        self._pending_events.append(
            (
                self.run_id,
                float(event.get("recorded_at") or time.time()),
                event_type,
                level,
                event.get("message"),
                event.get("percent"),
                _json_payload(event),
            )
        )
        if event_type in FORCE_FLUSH_EVENT_TYPES:
            self.flush()
        else:
            self._flush_if_due()

    def flush_due(self) -> None:
        """Flush if the configured time threshold has elapsed."""

        self._flush_if_due()

    def flush(self) -> None:
        """Commit all queued samples and events in one transaction."""

        if self._closed or not (self._pending_metrics or self._pending_events):
            return
        with self._connection:
            if self._pending_metrics:
                self._connection.executemany(
                    """
                    INSERT INTO live_metrics (
                        run_id, recorded_at, step, epoch, total_epochs, total_steps,
                        train_loss, val_loss, learning_rate, grad_norm, weight_norm,
                        update_ratio, tokens_per_second, samples_per_second,
                        vram_allocated_gb, vram_reserved_gb, gpu_memory_percent,
                        system_cpu_percent, system_ram_percent, data_loader_workers,
                        sample_text, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._pending_metrics,
                )
                row = self._connection.execute("SELECT last_insert_rowid()").fetchone()
                self._last_metric_row_id = int(row[0])
            if self._pending_events:
                self._connection.executemany(
                    """
                    INSERT INTO telemetry_events (
                        run_id, recorded_at, event_type, level, message, percent, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._pending_events,
                )
        self._pending_metrics.clear()
        self._pending_events.clear()
        self._last_flush_at = time.monotonic()

    def close(self) -> None:
        """Force a final commit and close the writer connection."""

        if self._closed:
            return
        self.flush()
        self._closed = True
        self._connection.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _flush_if_due(self) -> None:
        pending_count = len(self._pending_metrics) + len(self._pending_events)
        elapsed = time.monotonic() - self._last_flush_at
        if pending_count >= self.batch_size or elapsed >= self.flush_interval_seconds:
            self.flush()


def initialize_store(model_dir: Path) -> Path:
    """Create a WAL telemetry database for a model directory."""

    db_path = telemetry_db_path(model_dir)
    with TelemetryWriter(db_path, run_id="__initialize__"):
        pass
    return db_path


def insert_metric(db_path: Path, run_id: str, event: dict[str, Any]) -> int:
    """Compatibility helper that immediately persists one metric."""

    with TelemetryWriter(db_path, run_id, batch_size=1) as writer:
        row_id = writer.record_metric(event)
        if row_id is None:
            raise RuntimeError("Immediate telemetry insert did not flush")
        return row_id


def latest_run(db_path: Path) -> Optional[sqlite3.Row]:
    """Return the latest sampled telemetry run metadata."""

    with _read_connection(db_path) as connection:
        return connection.execute(
            """
            SELECT run_id, COUNT(*) AS sample_count, MAX(id) AS latest_id
            FROM live_metrics
            GROUP BY run_id
            ORDER BY MAX(id) DESC
            LIMIT 1
            """
        ).fetchone()


def rows_until(db_path: Path, run_id: str, sample_index: int) -> list[sqlite3.Row]:
    """Load metric rows up to a selected sample count."""

    if sample_index <= 0:
        return []
    with _read_connection(db_path) as connection:
        return list(
            connection.execute(
                "SELECT * FROM live_metrics WHERE run_id = ? ORDER BY id LIMIT ?",
                (run_id, int(sample_index)),
            )
        )


def metric_rows_after(
    db_path: Path,
    run_id: str,
    last_row_id: int = 0,
    limit: int = 1000,
) -> list[sqlite3.Row]:
    """Incrementally load metric samples after a previously observed row ID."""

    with _read_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT * FROM live_metrics
                WHERE run_id = ? AND id > ?
                ORDER BY id
                LIMIT ?
                """,
                (run_id, max(0, int(last_row_id)), max(1, int(limit))),
            )
        )


def event_rows_after(
    db_path: Path,
    run_id: str,
    last_row_id: int = 0,
    limit: int = 1000,
) -> list[sqlite3.Row]:
    """Incrementally load durable events after a previously observed row ID."""

    with _read_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT * FROM telemetry_events
                WHERE run_id = ? AND id > ?
                ORDER BY id
                LIMIT ?
                """,
                (run_id, max(0, int(last_row_id)), max(1, int(limit))),
            )
        )


rows_after = metric_rows_after


def _read_connection(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(db_path), timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _json_payload(event: dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)


def _infer_event_type(event: dict[str, Any]) -> str:
    message = str(event.get("message") or "")
    lowered = message.lower()
    if "[warn]" in lowered or event.get("level") == "warning":
        return "warning"
    if "checkpoint" in lowered or "saved" in lowered:
        return "checkpoint"
    if "validation" in lowered:
        return "validation"
    if "stopped" in lowered:
        return "stop"
    if "complete" in lowered:
        return "completion"
    return "lifecycle"
