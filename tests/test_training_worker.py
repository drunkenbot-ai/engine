from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engine.config import ModelConfig, TrainingConfig
from engine.contracts import TrainingJobSpec
from engine.telemetry_store import event_rows_after, metric_rows_after
from engine.training import TrainingResult
from engine.training_worker import run_worker_request
from engine.training_worker_protocol import (
    StandaloneTrainingRequest,
    create_worker_request,
    load_run_manifest,
    load_worker_request,
    manifest_is_stale,
    worker_command,
    write_stop_request,
    write_worker_request,
)


def _job(tmp_path: Path) -> TrainingJobSpec:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    return TrainingJobSpec.local(
        dataset_dir,
        ModelConfig(
            vocab_size=16,
            context_length=8,
            embedding_size=8,
            head_count=2,
            layer_count=1,
        ),
        TrainingConfig(
            output_dir=tmp_path / "output",
            device="cpu",
            use_amp=False,
            precision="fp32",
            resume=False,
        ),
    )


def _result(request: StandaloneTrainingRequest, stopped: bool = False) -> TrainingResult:
    output_dir = request.job.artifacts.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / ("stopped.pt" if stopped else "final.pt")
    summary = output_dir / "summary.json"
    checkpoint.touch()
    summary.write_text("{}", encoding="utf-8")
    return TrainingResult(checkpoint, summary, 0.5, None, stopped=stopped)


def test_request_round_trip_and_version_rejection(tmp_path: Path) -> None:
    request = create_worker_request(_job(tmp_path), run_id="run-round-trip")
    request_path = write_worker_request(tmp_path / "request.json", request)

    loaded = load_worker_request(request_path)
    assert loaded.to_jsonable() == request.to_jsonable()
    assert worker_command(request_path)[1:4] == [
        "-m",
        "engine.training_worker",
        "--request",
    ]

    data = request.to_jsonable()
    data["version"] = 999
    with pytest.raises(ValueError, match="Unsupported"):
        StandaloneTrainingRequest.from_jsonable(data)


def test_worker_completes_with_heartbeat_and_persisted_telemetry(
    tmp_path: Path,
) -> None:
    request = create_worker_request(_job(tmp_path), run_id="run-complete")
    request = StandaloneTrainingRequest(
        **{
            **request.__dict__,
            "heartbeat_interval_seconds": 0.1,
            "telemetry_batch_size": 10,
        }
    )
    request_path = write_worker_request(tmp_path / "request.json", request)
    observed: dict = {}

    def fake_run(_job, progress, _should_stop):
        initial = load_run_manifest(request.manifest_path)
        time.sleep(0.25)
        later = load_run_manifest(request.manifest_path)
        observed["status"] = initial["status"]
        observed["heartbeat_changed"] = (
            initial["heartbeat_at"] != later["heartbeat_at"]
        )
        progress(
            {
                "event_type": "metrics",
                "message": "step 1",
                "step": 1,
                "train_loss": 0.5,
            }
        )
        return _result(request)

    assert run_worker_request(request_path, run_job=fake_run) == 0

    manifest = load_run_manifest(request.manifest_path)
    assert observed == {"status": "running", "heartbeat_changed": True}
    assert manifest["status"] == "completed"
    assert manifest["exit_code"] == 0
    assert manifest["failure"] is None
    assert manifest["config_fingerprint"]
    assert "notifier" not in manifest
    assert len(metric_rows_after(request.telemetry_db_path, request.run_id)) == 1
    assert event_rows_after(request.telemetry_db_path, request.run_id)[-1][
        "event_type"
    ] == "completion"


def test_worker_honors_cooperative_stop(tmp_path: Path) -> None:
    request = create_worker_request(_job(tmp_path), run_id="run-stop")
    request_path = write_worker_request(tmp_path / "request.json", request)
    started = threading.Event()

    def fake_run(_job, _progress, should_stop):
        started.set()
        deadline = time.time() + 2.0
        while time.time() < deadline and not should_stop():
            time.sleep(0.01)
        assert should_stop()
        return _result(request, stopped=True)

    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(run_worker_request(request_path, fake_run))
    )
    thread.start()
    assert started.wait(1.0)
    write_stop_request(request.control_path, request.run_id)
    thread.join(timeout=3.0)

    assert result == [0]
    manifest = load_run_manifest(request.manifest_path)
    assert manifest["status"] == "stopped"
    assert manifest["stop_requested_at"]


def test_worker_records_failure_status_and_details(tmp_path: Path) -> None:
    request = create_worker_request(_job(tmp_path), run_id="run-failure")
    request_path = write_worker_request(tmp_path / "request.json", request)

    def fail(*_args):
        raise RuntimeError("mock training failure")

    assert run_worker_request(request_path, run_job=fail) == 1

    manifest = load_run_manifest(request.manifest_path)
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == 1
    assert manifest["failure"]["type"] == "RuntimeError"
    assert Path(manifest["failure"]["details_path"]).exists()
    assert event_rows_after(request.telemetry_db_path, request.run_id)[-1][
        "event_type"
    ] == "failure"


def test_stale_manifest_detection_uses_heartbeat_and_process_identity(
    tmp_path: Path,
) -> None:
    request = create_worker_request(_job(tmp_path), run_id="run-stale")
    request_path = write_worker_request(tmp_path / "request.json", request)

    def inspect(_job, _progress, _should_stop):
        current = load_run_manifest(request.manifest_path)
        assert not manifest_is_stale(current, heartbeat_timeout_seconds=10.0)
        wrong_pid = {**current, "pid": 999_999_999}
        assert manifest_is_stale(wrong_pid, heartbeat_timeout_seconds=10.0)
        old = {
            **current,
            "heartbeat_at": (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        }
        assert manifest_is_stale(old, heartbeat_timeout_seconds=1.0)
        return _result(request)

    assert run_worker_request(request_path, run_job=inspect) == 0
