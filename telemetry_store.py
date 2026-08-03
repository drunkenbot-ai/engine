from __future__ import annotations

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


def telemetry_db_path(model_dir: Path) -> Path:
    """Return the telemetry SQLite path for a model directory.

    Args:
        model_dir: Model output directory.

    Returns:
        Path to the telemetry SQLite database.
    """

    return model_dir / "training_telemetry.sqlite"


def ensure_schema(connection: sqlite3.Connection) -> None:
    """Create or migrate the live telemetry schema.

    Args:
        connection: Open SQLite connection.
    """

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
            sample_text TEXT
        )
        """
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(live_metrics)")}
    if "sample_text" not in columns:
        connection.execute("ALTER TABLE live_metrics ADD COLUMN sample_text TEXT")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_live_metrics_run_id_id ON live_metrics(run_id, id)")


def initialize_store(model_dir: Path) -> Path:
    """Create a telemetry database for a model directory.

    Args:
        model_dir: Model output directory.

    Returns:
        Telemetry database path.
    """

    model_dir.mkdir(parents=True, exist_ok=True)
    db_path = telemetry_db_path(model_dir)
    with sqlite3.connect(db_path) as connection:
        ensure_schema(connection)
        connection.commit()
    return db_path


def insert_metric(db_path: Path, run_id: str, event: dict[str, Any]) -> int:
    """Persist one training metric event.

    Args:
        db_path: Telemetry database path.
        run_id: Training run identifier.
        event: Progress event emitted by training.

    Returns:
        Inserted row id.
    """

    values = [event.get(field) for field in METRIC_FIELDS]
    with sqlite3.connect(db_path) as connection:
        ensure_schema(connection)
        cursor = connection.execute(
            """
            INSERT INTO live_metrics (
                run_id, recorded_at, step, epoch, total_epochs, total_steps,
                train_loss, val_loss, learning_rate, grad_norm, weight_norm, update_ratio,
                tokens_per_second, samples_per_second, vram_allocated_gb, vram_reserved_gb,
                gpu_memory_percent, system_cpu_percent, system_ram_percent, data_loader_workers,
                sample_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [run_id, time.time(), int(event["step"]), *values],
        )
        connection.commit()
        return int(cursor.lastrowid)


def latest_run(db_path: Path) -> Optional[sqlite3.Row]:
    """Return latest telemetry run metadata.

    Args:
        db_path: Telemetry database path.

    Returns:
        Row with run id, sample count, and latest row id, or None.
    """

    with sqlite3.connect(db_path) as connection:
        ensure_schema(connection)
        connection.row_factory = sqlite3.Row
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
    """Load telemetry rows up to a selected sample.

    Args:
        db_path: Telemetry database path.
        run_id: Training run identifier.
        sample_index: Maximum row count to load.

    Returns:
        Ordered telemetry rows.
    """

    if sample_index <= 0:
        return []
    with sqlite3.connect(db_path) as connection:
        ensure_schema(connection)
        connection.row_factory = sqlite3.Row
        return list(
            connection.execute(
                "SELECT * FROM live_metrics WHERE run_id = ? ORDER BY id LIMIT ?",
                (run_id, int(sample_index)),
            )
        )

