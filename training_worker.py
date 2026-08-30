from __future__ import annotations

import argparse
import logging
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from .backends.local_backend import LocalTrainerBackend
from .contracts import TrainingJobSpec
from .notifier import NotificationManager
from .telemetry_store import TelemetryWriter
from .training import TrainingResult
from .training_worker_protocol import (
    AtomicRunManifest,
    StandaloneTrainingRequest,
    initial_run_manifest,
    load_stop_request,
    load_worker_request,
    utc_now,
)


LOGGER = logging.getLogger(__name__)
WorkerJob = Callable[
    [TrainingJobSpec, Callable[[dict[str, Any]], None], Callable[[], bool]],
    TrainingResult,
]


class _Heartbeat:
    def __init__(self, manifest: AtomicRunManifest, interval_seconds: float) -> None:
        self._manifest = manifest
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="training-worker-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._interval_seconds * 2.0))

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._manifest.heartbeat()


class _ControlMonitor:
    def __init__(
        self,
        request: StandaloneTrainingRequest,
        manifest: AtomicRunManifest,
    ) -> None:
        self._request = request
        self._manifest = manifest
        self._stopping = False

    def should_stop(self) -> bool:
        if self._stopping:
            return True
        control = load_stop_request(self._request.control_path, self._request.run_id)
        if control is None:
            return False
        if not self._stopping:
            self._stopping = True
            self._manifest.update(
                status="stopping",
                stop_requested_at=control.get("requested_at") or utc_now(),
            )
        return True


def run_worker_request(
    request_path: Path,
    run_job: Optional[WorkerJob] = None,
) -> int:
    """Execute one durable request and return a process exit code."""

    request_path = Path(request_path).resolve()
    request = load_worker_request(request_path)
    manifest = AtomicRunManifest(
        request.manifest_path,
        initial_run_manifest(request, request_path),
    )
    notifier = (
        NotificationManager(request.notifier_config_path)
        if request.notifier_config_path
        else None
    )
    heartbeat = _Heartbeat(manifest, request.heartbeat_interval_seconds)
    control = _ControlMonitor(request, manifest)
    failure_path = request.job.artifacts.output_dir / "training_failure.txt"
    request.job.artifacts.output_dir.mkdir(parents=True, exist_ok=True)

    with TelemetryWriter(
        request.telemetry_db_path,
        request.run_id,
        batch_size=request.telemetry_batch_size,
        flush_interval_seconds=request.telemetry_flush_interval_seconds,
    ) as telemetry:
        terminal_event_seen = False

        def progress(event: dict[str, Any]) -> None:
            nonlocal terminal_event_seen
            payload = dict(event)
            telemetry.record(payload)
            if notifier is None:
                return
            event_type = str(payload.get("event_type") or "")
            terminal_event_seen = terminal_event_seen or event_type in {
                "completion",
                "stop",
            }
            message = str(payload.get("message") or "Training update")
            percent = payload.get("percent")
            lines = _notification_lines(payload)
            if event_type in {"completion", "stop"}:
                notifier.notify_complete("training", message, lines)
            elif event_type == "failure":
                notifier.notify_failure("training", "Training failed", message)
            else:
                notifier.notify_progress("training", message, lines, percent)

        heartbeat.start()
        manifest.update(status="running")
        progress(
            {
                "event_type": "lifecycle",
                "message": f"Standalone worker started run {request.run_id}",
            }
        )
        try:
            execute = run_job or _run_local_job
            result = execute(request.job, progress, control.should_stop)
            status = "stopped" if result.stopped else "completed"
            event_type = "stop" if result.stopped else "completion"
            if not terminal_event_seen:
                progress(
                    {
                        "event_type": event_type,
                        "message": "Training stopped cooperatively."
                        if result.stopped
                        else "Standalone training completed.",
                        "percent": 100,
                        "checkpoint_path": str(result.checkpoint_path),
                        "summary_path": str(result.summary_path),
                        "train_loss": result.final_train_loss,
                        "val_loss": result.final_val_loss,
                    }
                )
            manifest.update(
                status=status,
                finished_at=utc_now(),
                exit_code=0,
                output_paths={
                    "output_dir": str(request.job.artifacts.output_dir),
                    "telemetry_db": str(request.telemetry_db_path),
                    "checkpoint": str(result.checkpoint_path),
                    "summary": str(result.summary_path),
                },
            )
            return 0
        except Exception as exc:
            failure_path.write_text(traceback.format_exc(), encoding="utf-8")
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "details_path": str(failure_path),
            }
            progress(
                {
                    "event_type": "failure",
                    "level": "error",
                    "message": str(exc),
                    "failure": failure,
                }
            )
            manifest.update(
                status="failed",
                finished_at=utc_now(),
                exit_code=1,
                failure=failure,
            )
            LOGGER.exception("Standalone training worker failed")
            return 1
        finally:
            heartbeat.close()
            if notifier is not None:
                notifier.close()


def _run_local_job(
    job: TrainingJobSpec,
    progress: Callable[[dict[str, Any]], None],
    should_stop: Callable[[], bool],
) -> TrainingResult:
    return LocalTrainerBackend().run(job, progress=progress, should_stop=should_stop)


def _notification_lines(event: dict[str, Any]) -> list[str]:
    fields = (
        ("step", "Step"),
        ("train_loss", "Train loss"),
        ("val_loss", "Validation loss"),
        ("tokens_per_second", "Tokens/sec"),
    )
    return [f"{label}: {event[key]}" for key, label in fields if event.get(key) is not None]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a standalone local training request")
    parser.add_argument("--request", required=True, type=Path, help="Durable request JSON path")
    args = parser.parse_args(argv)
    return run_worker_request(args.request)


if __name__ == "__main__":
    raise SystemExit(main())
