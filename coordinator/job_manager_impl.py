from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from engine.backends.base import ProgressCallback, StopCallback, TrainerBackend
from engine.backends.registry import DEFAULT_BACKEND_REGISTRY, BackendRegistry
from .job_models import *
from .job_manager_core import JobManagerCore
from .job_manager_aux import JobManagerAux

class JobManager(JobManagerCore, JobManagerAux):
    """Compatibility class combining coordinator job manager behavior."""

    pass

def run_local_job(
    job: TrainingJobSpec,
    backend: Optional[TrainerBackend] = None,
    progress: Optional[ProgressCallback] = None,
    should_stop: Optional[StopCallback] = None,
) -> TrainingResult:
    """Run a job through a temporary local manager.

    Args:
        job: Training job contract.
        backend: Optional backend override for tests or embedded use.
        progress: Optional progress callback.
        should_stop: Optional cooperative cancellation callback.

    Returns:
        Training result.
    """

    registry = BackendRegistry()
    if backend is not None:
        registry.register(job.runtime.backend, backend)
    manager = JobManager(registry=registry)
    manager.submit(job)
    return manager.run_job(job.job_id, progress=progress, should_stop=should_stop)


def _jsonable_result(result: TrainingResultSpec) -> dict[str, Any]:
    """Convert a result spec to JSON-friendly values.

    Args:
        result: Training result specification.

    Returns:
        Serializable dictionary.
    """

    output: dict[str, Any] = {}
    for key, value in result.__dict__.items():
        if isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, Enum):
            output[key] = value.value
        else:
            output[key] = value
    return output


def _managed_job_from_jsonable(data: dict[str, Any]) -> ManagedJob:
    """Create a managed job from JSON-friendly values.

    Args:
        data: Serialized managed job payload.

    Returns:
        Managed job.
    """

    result_data = data.get("result")
    return ManagedJob(
        spec=TrainingJobSpec.from_jsonable(data["spec"]),
        assigned_worker_id=data.get("assigned_worker_id"),
        result=_result_from_jsonable(result_data) if result_data else None,
        error=data.get("error"),
        latest_metrics=TrainingMetrics(**dict(data.get("latest_metrics") or {})) if data.get("latest_metrics") else None,
        updated_at=data.get("updated_at", utc_now_iso()),
    )


def _result_from_jsonable(data: dict[str, Any]) -> TrainingResultSpec:
    """Create a training result spec from JSON-friendly values.

    Args:
        data: Serialized result data.

    Returns:
        Training result specification.
    """

    checkpoint_path = data.get("checkpoint_path")
    summary_path = data.get("summary_path")
    return TrainingResultSpec(
        job_id=data["job_id"],
        status=JobStatus(data["status"]),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        summary_path=Path(summary_path) if summary_path else None,
        final_train_loss=data.get("final_train_loss"),
        final_val_loss=data.get("final_val_loss"),
        stopped=bool(data.get("stopped")),
        error=data.get("error"),
        artifact_bundle_url=data.get("artifact_bundle_url"),
    )


def _availability_to_worker_status(availability: WorkerAvailability) -> WorkerStatus:
    """Convert protocol availability to manager worker status.

    Args:
        availability: Protocol worker availability.

    Returns:
        Manager worker status.
    """

    if availability == WorkerAvailability.BUSY:
        return WorkerStatus.BUSY
    if availability == WorkerAvailability.OFFLINE:
        return WorkerStatus.OFFLINE
    return WorkerStatus.AVAILABLE


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp.

    Args:
        value: ISO timestamp.

    Returns:
        Parsed timestamp when valid.
    """

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.astimezone()
    return parsed


def _worker_total_vram_gb(worker: WorkerDescriptor) -> Optional[float]:
    """Return worker total VRAM in GB when known.

    Args:
        worker: Worker descriptor.

    Returns:
        Total VRAM in GB.
    """

    value = worker.capabilities.get("total_vram_gb")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _control_message(should_stop: bool, should_pause: bool, default: str) -> str:
    """Return a human-readable control message.

    Args:
        should_stop: Whether stop was requested.
        should_pause: Whether pause was requested.
        default: Default message.

    Returns:
        Control message.
    """

    if should_stop:
        return "stop requested"
    if should_pause:
        return "pause requested"
    return default

