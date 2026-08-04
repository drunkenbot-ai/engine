from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from engine.contracts.jobs import BackendKind, TrainingJobSpec, TrainingMetrics, TrainingResultSpec, utc_now_iso


from .protocol_types import *
from .protocol_types import _restore_envelope
class RegisterWorkerRequest(ProtocolEnvelope):
    """Request sent by a worker to join the coordinator."""

    worker_id: str = ""
    backend: BackendKind = BackendKind.REMOTE_CLIENT
    device: str = "auto"
    capabilities: WorkerCapabilities = field(default_factory=WorkerCapabilities)
    labels: list[str] = field(default_factory=list)

    def __init__(
        self,
        worker_id: str,
        backend: BackendKind = BackendKind.REMOTE_CLIENT,
        device: str = "auto",
        capabilities: Optional[WorkerCapabilities] = None,
        labels: Optional[list[str]] = None,
    ) -> None:
        """Create a register worker request.

        Args:
            worker_id: Worker identifier.
            backend: Backend kind the worker can run.
            device: Preferred runtime device.
            capabilities: Worker hardware/runtime capabilities.
            labels: Free-form worker labels for scheduling.
        """

        super().__init__(ProtocolMessageKind.REGISTER_WORKER_REQUEST)
        self.worker_id = worker_id
        self.backend = backend
        self.device = device
        self.capabilities = capabilities or WorkerCapabilities()
        self.labels = labels or []

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "backend": self.backend.value,
            "device": self.device,
            "capabilities": self.capabilities.to_jsonable(),
            "labels": self.labels,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "RegisterWorkerRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Register worker request.
        """

        request = cls(
            worker_id=data["worker_id"],
            backend=BackendKind(data.get("backend", BackendKind.REMOTE_CLIENT.value)),
            device=data.get("device", "auto"),
            capabilities=WorkerCapabilities.from_jsonable(data.get("capabilities") or {}),
            labels=list(data.get("labels") or []),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class RegisterWorkerResponse(ProtocolEnvelope):
    """Response returned after worker registration."""

    status: ProtocolStatus = ProtocolStatus.OK
    worker_id: str = ""
    accepted: bool = True
    heartbeat_interval_seconds: int = 10
    message: str = ""

    def __init__(
        self,
        worker_id: str,
        accepted: bool = True,
        status: ProtocolStatus = ProtocolStatus.OK,
        heartbeat_interval_seconds: int = 10,
        message: str = "",
    ) -> None:
        """Create a register worker response.

        Args:
            worker_id: Worker identifier.
            accepted: Whether registration was accepted.
            status: Protocol response status.
            heartbeat_interval_seconds: Requested heartbeat interval.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.REGISTER_WORKER_RESPONSE)
        self.status = status
        self.worker_id = worker_id
        self.accepted = accepted
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "worker_id": self.worker_id,
            "accepted": self.accepted,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "message": self.message,
        }


@dataclass
class HeartbeatRequest(ProtocolEnvelope):
    """Heartbeat request sent by a worker."""

    worker_id: str = ""
    availability: WorkerAvailability = WorkerAvailability.AVAILABLE
    backend: BackendKind = BackendKind.REMOTE_CLIENT
    active_job_id: Optional[str] = None
    device: str = "auto"
    metrics: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        worker_id: str,
        availability: WorkerAvailability = WorkerAvailability.AVAILABLE,
        backend: BackendKind = BackendKind.REMOTE_CLIENT,
        active_job_id: Optional[str] = None,
        device: str = "auto",
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create a heartbeat request.

        Args:
            worker_id: Worker identifier.
            availability: Current worker availability.
            backend: Backend kind the worker can run.
            active_job_id: Active job identifier when busy.
            device: Runtime device.
            metrics: Worker metrics.
        """

        super().__init__(ProtocolMessageKind.HEARTBEAT_REQUEST)
        self.worker_id = worker_id
        self.availability = availability
        self.backend = backend
        self.active_job_id = active_job_id
        self.device = device
        self.metrics = metrics or {}

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "availability": self.availability.value,
            "backend": self.backend.value,
            "active_job_id": self.active_job_id,
            "device": self.device,
            "metrics": self.metrics,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "HeartbeatRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Heartbeat request.
        """

        request = cls(
            worker_id=data["worker_id"],
            availability=WorkerAvailability(data.get("availability", WorkerAvailability.AVAILABLE.value)),
            backend=BackendKind(data.get("backend", BackendKind.REMOTE_CLIENT.value)),
            active_job_id=data.get("active_job_id"),
            device=data.get("device", "auto"),
            metrics=dict(data.get("metrics") or {}),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class HeartbeatResponse(ProtocolEnvelope):
    """Response returned after a worker heartbeat."""

    status: ProtocolStatus = ProtocolStatus.OK
    should_stop_job: bool = False
    should_pause_job: bool = False
    message: str = ""

    def __init__(
        self,
        status: ProtocolStatus = ProtocolStatus.OK,
        should_stop_job: bool = False,
        should_pause_job: bool = False,
        message: str = "",
    ) -> None:
        """Create a heartbeat response.

        Args:
            status: Protocol response status.
            should_stop_job: Whether active job should stop.
            should_pause_job: Whether active job should pause.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.HEARTBEAT_RESPONSE)
        self.status = status
        self.should_stop_job = should_stop_job
        self.should_pause_job = should_pause_job
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "should_stop_job": self.should_stop_job,
            "should_pause_job": self.should_pause_job,
            "message": self.message,
        }


@dataclass
class ClaimJobRequest(ProtocolEnvelope):
    """Request sent by a worker asking for a job."""

    worker_id: str = ""
    backend: BackendKind = BackendKind.REMOTE_CLIENT
    capabilities: WorkerCapabilities = field(default_factory=WorkerCapabilities)

    def __init__(
        self,
        worker_id: str,
        backend: BackendKind = BackendKind.REMOTE_CLIENT,
        capabilities: Optional[WorkerCapabilities] = None,
    ) -> None:
        """Create a claim job request.

        Args:
            worker_id: Worker identifier.
            backend: Backend kind requested by the worker.
            capabilities: Latest worker capabilities.
        """

        super().__init__(ProtocolMessageKind.CLAIM_JOB_REQUEST)
        self.worker_id = worker_id
        self.backend = backend
        self.capabilities = capabilities or WorkerCapabilities()

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "backend": self.backend.value,
            "capabilities": self.capabilities.to_jsonable(),
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "ClaimJobRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Claim job request.
        """

        request = cls(
            worker_id=data["worker_id"],
            backend=BackendKind(data.get("backend", BackendKind.REMOTE_CLIENT.value)),
            capabilities=WorkerCapabilities.from_jsonable(data.get("capabilities") or {}),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class ClaimJobResponse(ProtocolEnvelope):
    """Response with an assigned job or empty assignment."""

    status: ProtocolStatus = ProtocolStatus.OK
    job: Optional[TrainingJobSpec] = None
    message: str = ""

    def __init__(
        self,
        job: Optional[TrainingJobSpec] = None,
        status: ProtocolStatus = ProtocolStatus.OK,
        message: str = "",
    ) -> None:
        """Create a claim job response.

        Args:
            job: Assigned job contract when available.
            status: Protocol response status.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.CLAIM_JOB_RESPONSE)
        self.status = status
        self.job = job
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "job": self.job.to_jsonable() if self.job else None,
            "message": self.message,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "ClaimJobResponse":
        """Create a response from JSON-friendly values.

        Args:
            data: Serialized response.

        Returns:
            Claim job response.
        """

        job_data = data.get("job")
        response = cls(
            job=TrainingJobSpec.from_jsonable(job_data) if job_data else None,
            status=ProtocolStatus(data.get("status", ProtocolStatus.OK.value)),
            message=data.get("message", ""),
        )
        _restore_envelope(response, data)
        return response


@dataclass
class ProgressReportRequest(ProtocolEnvelope):
    """Progress update sent by a worker for a running job."""

    worker_id: str = ""
    job_id: str = ""
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)

    def __init__(self, worker_id: str, job_id: str, metrics: Optional[TrainingMetrics] = None) -> None:
        """Create a progress report request.

        Args:
            worker_id: Worker identifier.
            job_id: Job identifier.
            metrics: Training metrics.
        """

        super().__init__(ProtocolMessageKind.PROGRESS_REPORT_REQUEST)
        self.worker_id = worker_id
        self.job_id = job_id
        self.metrics = metrics or TrainingMetrics()

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "job_id": self.job_id,
            "metrics": self.metrics.__dict__,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "ProgressReportRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Progress report request.
        """

        request = cls(
            worker_id=data["worker_id"],
            job_id=data["job_id"],
            metrics=TrainingMetrics(**dict(data.get("metrics") or {})),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class ProgressReportResponse(ProtocolEnvelope):
    """Response returned after a progress report."""

    status: ProtocolStatus = ProtocolStatus.OK
    should_stop_job: bool = False
    should_pause_job: bool = False
    message: str = ""

    def __init__(
        self,
        status: ProtocolStatus = ProtocolStatus.OK,
        should_stop_job: bool = False,
        should_pause_job: bool = False,
        message: str = "",
    ) -> None:
        """Create a progress report response.

        Args:
            status: Protocol response status.
            should_stop_job: Whether the worker should stop the active job.
            should_pause_job: Whether the worker should pause the active job.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.PROGRESS_REPORT_RESPONSE)
        self.status = status
        self.should_stop_job = should_stop_job
        self.should_pause_job = should_pause_job
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "should_stop_job": self.should_stop_job,
            "should_pause_job": self.should_pause_job,
            "message": self.message,
        }



