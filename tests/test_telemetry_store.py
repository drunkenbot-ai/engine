from __future__ import annotations

import sqlite3
from pathlib import Path

from engine.telemetry_store import (
    TelemetryWriter,
    event_rows_after,
    latest_run,
    metric_rows_after,
)


def _metric(step: int) -> dict:
    return {
        "event_type": "metrics",
        "step": step,
        "train_loss": 1.0 / step,
        "message": f"step {step}",
    }


def test_wal_writer_batches_and_supports_concurrent_incremental_reads(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "telemetry.sqlite"
    writer = TelemetryWriter(
        db_path,
        "run-test",
        batch_size=3,
        flush_interval_seconds=3600.0,
    )
    reader = sqlite3.connect(db_path, timeout=1.0)
    try:
        assert reader.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        writer.record(_metric(1))
        writer.record(_metric(2))
        assert metric_rows_after(db_path, "run-test") == []

        writer.record(_metric(3))
        first_page = metric_rows_after(db_path, "run-test", limit=2)
        assert [row["step"] for row in first_page] == [1, 2]
        second_page = metric_rows_after(
            db_path,
            "run-test",
            last_row_id=first_page[-1]["id"],
        )
        assert [row["step"] for row in second_page] == [3]

        writer.record(
            {
                "event_type": "validation",
                "message": "validation complete",
                "step": 3,
            }
        )
        events = event_rows_after(db_path, "run-test")
        assert [row["event_type"] for row in events] == ["validation"]
    finally:
        reader.close()
        writer.close()


def test_lifecycle_events_are_separate_from_coalescible_metrics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "telemetry.sqlite"
    with TelemetryWriter(db_path, "run-test", batch_size=100) as writer:
        writer.record(_metric(1))
        writer.record({"message": "building model", "percent": 2})
        assert [row["event_type"] for row in event_rows_after(db_path, "run-test")] == [
            "lifecycle"
        ]
        writer.record({"event_type": "completion", "message": "done", "percent": 100})

    assert len(metric_rows_after(db_path, "run-test")) == 1
    events = event_rows_after(db_path, "run-test")
    assert [row["event_type"] for row in events] == ["lifecycle", "completion"]


def test_incremental_readers_are_safe_before_worker_initialization(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "not-created.sqlite"

    assert metric_rows_after(db_path, "run-test") == []
    assert event_rows_after(db_path, "run-test") == []
    assert latest_run(db_path) is None
    assert not db_path.exists()
