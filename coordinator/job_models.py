from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from engine.backends.base import ProgressCallback, StopCallback, TrainerBackend
from engine.backends.registry import DEFAULT_BACKEND_REGISTRY, BackendRegistry
from engine.contracts import (
    BackendKind,
    ClaimJobRequest,
    ClaimJobResponse,
    CompleteJobRequest,
    CompleteJobResponse,
    FailJobRequest,
    FailJobResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    JobStatus,
    ProgressReportRequest,
    ProgressReportResponse,
    ProtocolStatus,
    RegisterWorkerRequest,
    RegisterWorkerResponse,
    TrainingMetrics,
    TrainingJobSpec,
    TrainingResultSpec,
    WorkerAvailability,
    utc_now_iso,
)
from engine.coordinator.state_store import JobStateStore
from engine.training import TrainingResult


def _jsonable_result(result: TrainingResultSpec) -> dict[str, Any]:
    """Convert a result specification to JSON-friendly values.

    Args:
        result: Training result specification.

    Returns:
        Serializable result dictionary.
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


class WorkerStatus(str, Enum):
    """Worker availability state."""

    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class WorkerDescriptor:
    """Training worker registered with the job manager.

    Attributes:
        worker_id: Stable worker identifier.
        backend: Backend kind this worker can execute.
        status: Current availability state.
        device: Device advertised by the worker.
        hostname: Optional worker host name.
        capabilities: Free-form hardware/runtime capabilities.
    """

    worker_id: str
    backend: BackendKind = BackendKind.LOCAL
    status: WorkerStatus = WorkerStatus.AVAILABLE
    device: str = "auto"
    hostname: Optional[str] = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    last_heartbeat_at: Optional[str] = None

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the worker descriptor to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            "worker_id": self.worker_id,
            "backend": self.backend.value,
            "status": self.status.value,
            "device": self.device,
            "hostname": self.hostname,
            "capabilities": self.capabilities,
            "last_heartbeat_at": self.last_heartbeat_at,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "WorkerDescriptor":
        """Create a worker descriptor from JSON-friendly values.

        Args:
            data: Serialized worker data.

        Returns:
            Worker descriptor.
        """

        return cls(
            worker_id=data["worker_id"],
            backend=BackendKind(data.get("backend", BackendKind.LOCAL.value)),
            status=WorkerStatus(data.get("status", WorkerStatus.AVAILABLE.value)),
            device=data.get("device", "auto"),
            hostname=data.get("hostname"),
            capabilities=dict(data.get("capabilities") or {}),
            last_heartbeat_at=data.get("last_heartbeat_at"),
        )


@dataclass
class WorkerHeartbeat:
    """Heartbeat reported by a training worker.

    Attributes:
        worker_id: Worker identifier.
        status: Worker availability state.
        backend: Backend kind the worker can execute.
        active_job_id: Job currently running on the worker.
        device: Worker device.
        metrics: Runtime metrics reported by the worker.
        timestamp: UTC heartbeat timestamp.
    """

    worker_id: str
    status: WorkerStatus
    backend: BackendKind = BackendKind.LOCAL
    active_job_id: Optional[str] = None
    device: str = "auto"
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the heartbeat to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "backend": self.backend.value,
            "active_job_id": self.active_job_id,
            "device": self.device,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


@dataclass
class ManagedJob:
    """Job state tracked by the manager.

    Attributes:
        spec: Training job contract.
        assigned_worker_id: Worker currently assigned to this job.
        result: Serializable result metadata when finished.
        error: Error text when failed.
    """

    spec: TrainingJobSpec
    assigned_worker_id: Optional[str] = None
    result: Optional[TrainingResultSpec] = None
    error: Optional[str] = None
    latest_metrics: Optional[TrainingMetrics] = None
    updated_at: str = field(default_factory=utc_now_iso)

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the managed job to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            "spec": self.spec.to_jsonable(),
            "assigned_worker_id": self.assigned_worker_id,
            "result": _jsonable_result(self.result) if self.result else None,
            "error": self.error,
            "latest_metrics": self.latest_metrics.__dict__ if self.latest_metrics else None,
            "updated_at": self.updated_at,
        }


