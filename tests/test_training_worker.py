from __future__ import annotations

import sqlite3
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import engine.training_worker as worker_module
import engine.training_worker_protocol as protocol_module
from engine.config import ModelConfig, TrainingConfig
from engine.contracts import TrainingJobSpec
from engine.telemetry_store import event_rows_after, metric_rows_after
from engine.training import TrainingResult
from engine.training_worker import run_worker_request
from engine.training_worker_protocol import (
    ActiveRunError,
    RunClaim,
    StandaloneTrainingRequest,
    create_worker_request,
    load_run_manifest,
    load_worker_request,
    manifest_is_stale,
    process_identity,
    process_identity_matches,
    worker_command,
    write_stop_request,
    write_worker_request,
)


def _job(tmp_path: Path) -> TrainingJobSpec:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)
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
    second = create_worker_request(request.job, run_id="run-second")
    assert second.manifest_path != request.manifest_path
    assert second.control_path != request.control_path

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
        deadline = time.monotonic() + 2.0
        later = initial
        while (
            later["heartbeat_at"] == initial["heartbeat_at"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
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
        progress({"event_type": "completion", "message": "trainer complete"})
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
    events = event_rows_after(request.telemetry_db_path, request.run_id)
    assert events[-1]["event_type"] == "completion"
    assert sum(row["event_type"] == "completion" for row in events) == 1


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


def test_worker_setup_failure_still_writes_terminal_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = create_worker_request(_job(tmp_path), run_id="run-setup-failure")
    request_path = write_worker_request(tmp_path / "request.json", request)

    class BrokenTelemetryWriter:
        def __init__(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("cannot open telemetry")

    monkeypatch.setattr(worker_module, "TelemetryWriter", BrokenTelemetryWriter)

    assert run_worker_request(request_path, run_job=lambda *_args: None) == 1
    manifest = load_run_manifest(request.manifest_path)
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "OperationalError"


def test_concurrent_launch_is_rejected_without_overwriting_active_run(
    tmp_path: Path,
) -> None:
    request = create_worker_request(_job(tmp_path), run_id="run-exclusive")
    request_path = write_worker_request(tmp_path / "request.json", request)
    started = threading.Event()
    release = threading.Event()
    first_result: list[int] = []

    def slow_run(_job, _progress, _should_stop):
        started.set()
        release.wait(2.0)
        return _result(request)

    first = threading.Thread(
        target=lambda: first_result.append(run_worker_request(request_path, slow_run))
    )
    first.start()
    assert started.wait(1.0)

    assert run_worker_request(request_path, slow_run) == 2
    assert load_run_manifest(request.manifest_path)["status"] == "running"
    release.set()
    first.join(timeout=3.0)

    assert first_result == [0]
    assert load_run_manifest(request.manifest_path)["status"] == "completed"


def test_run_claim_normal_acquisition_is_exclusive(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    first = RunClaim(output_dir, "run-first")
    second = RunClaim(output_dir, "run-second")

    first.acquire()
    with pytest.raises(ActiveRunError, match="active run run-first"):
        second.acquire()
    assert protocol_module._load_json(first.path)["run_id"] == "run-first"

    first.release()
    second.acquire()
    assert protocol_module._load_json(second.path)["run_id"] == "run-second"
    second.release()


def test_stale_cleanup_serializes_competing_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    stale = RunClaim(output_dir, "run-stale")
    protocol_module.atomic_write_json(
        stale.path,
        {
            "schema": protocol_module.CLAIM_SCHEMA,
            "version": protocol_module.PROTOCOL_VERSION,
            "run_id": "run-stale",
            "pid": 999_999_999,
            "process_identity": {
                "kind": "psutil-create-time",
                "value": "0.000000",
            },
            "created_at": protocol_module.utc_now(),
        },
    )
    recovery = RunClaim(output_dir, "run-recovery")
    contender = RunClaim(output_dir, "run-contender")
    stale_inspected = threading.Event()
    allow_cleanup = threading.Event()
    contender_started = threading.Event()
    contender_finished = threading.Event()
    errors: list[Exception] = []
    original_read = recovery._read_existing_claim

    def pause_after_stale_read():
        value = original_read()
        stale_inspected.set()
        assert allow_cleanup.wait(2.0)
        return value

    monkeypatch.setattr(recovery, "_read_existing_claim", pause_after_stale_read)

    recovery_thread = threading.Thread(target=recovery.acquire)
    recovery_thread.start()
    assert stale_inspected.wait(1.0)

    def acquire_contender() -> None:
        contender_started.set()
        try:
            contender.acquire()
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)
        finally:
            contender_finished.set()

    contender_thread = threading.Thread(target=acquire_contender)
    contender_thread.start()
    assert contender_started.wait(1.0)
    assert not contender_finished.wait(0.1)

    allow_cleanup.set()
    recovery_thread.join(timeout=2.0)
    contender_thread.join(timeout=2.0)

    assert not recovery_thread.is_alive()
    assert not contender_thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ActiveRunError)
    assert protocol_module._load_json(recovery.path)["run_id"] == "run-recovery"
    recovery.release()


def test_run_claim_fails_closed_when_mutation_lock_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    claim = RunClaim(output_dir, "run-existing")
    protocol_module.atomic_write_json(
        claim.path,
        {
            "schema": protocol_module.CLAIM_SCHEMA,
            "version": protocol_module.PROTOCOL_VERSION,
            "run_id": "run-existing",
            "pid": 999_999_999,
            "process_identity": {
                "kind": "psutil-create-time",
                "value": "0.000000",
            },
            "created_at": protocol_module.utc_now(),
        },
    )
    monkeypatch.setattr(protocol_module, "CLAIM_MUTEX_ATTEMPTS", 1)
    monkeypatch.setattr(protocol_module, "CLAIM_MUTEX_RETRY_SECONDS", 0)

    def unsupported_lock(_handle) -> None:
        raise OSError("unsupported")

    monkeypatch.setattr(
        protocol_module._ClaimMutationLock,
        "_lock",
        staticmethod(unsupported_lock),
    )

    with pytest.raises(ActiveRunError, match="mutation lock"):
        RunClaim(output_dir, "run-contender").acquire()

    assert protocol_module._load_json(claim.path)["run_id"] == "run-existing"


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


def test_stdlib_process_identity_fallback_is_verifiable(monkeypatch) -> None:
    monkeypatch.setattr(protocol_module, "psutil", None)

    identity = process_identity(os.getpid())

    assert identity["kind"] != "unverifiable"
    assert process_identity_matches(os.getpid(), identity)
