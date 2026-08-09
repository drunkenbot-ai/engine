from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from engine.contracts.jobs import BackendKind, TrainingJobSpec, TrainingMetrics, TrainingResultSpec, utc_now_iso


from .protocol_types import *
from .protocol_types import _restore_envelope
@dataclass
class CompleteJobRequest(ProtocolEnvelope):
    """Completion report sent by a worker."""

    worker_id: str = ""
    result: Optional[TrainingResultSpec] = None

    def __init__(self, worker_id: str, result: TrainingResultSpec) -> None:
        """Create a complete job request.

        Args:
            worker_id: Worker identifier.
            result: Training result specification.
        """

        super().__init__(ProtocolMessageKind.COMPLETE_JOB_REQUEST)
        self.worker_id = worker_id
        self.result = result

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "result": _result_to_jsonable(self.result) if self.result else None,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "CompleteJobRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Complete job request.
        """

        request = cls(
            worker_id=data["worker_id"],
            result=_result_from_jsonable(dict(data["result"])),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class CompleteJobResponse(ProtocolEnvelope):
    """Response returned after job completion."""

    status: ProtocolStatus = ProtocolStatus.OK
    message: str = ""

    def __init__(self, status: ProtocolStatus = ProtocolStatus.OK, message: str = "") -> None:
        """Create a complete job response.

        Args:
            status: Protocol response status.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.COMPLETE_JOB_RESPONSE)
        self.status = status
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "message": self.message,
        }


@dataclass
class FailJobRequest(ProtocolEnvelope):
    """Failure report sent by a worker."""

    worker_id: str = ""
    job_id: str = ""
    error: str = ""
    retryable: bool = False

    def __init__(self, worker_id: str, job_id: str, error: str, retryable: bool = False) -> None:
        """Create a fail job request.

        Args:
            worker_id: Worker identifier.
            job_id: Job identifier.
            error: Failure text.
            retryable: Whether the coordinator may retry the job.
        """

        super().__init__(ProtocolMessageKind.FAIL_JOB_REQUEST)
        self.worker_id = worker_id
        self.job_id = job_id
        self.error = error
        self.retryable = retryable

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "job_id": self.job_id,
            "error": self.error,
            "retryable": self.retryable,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "FailJobRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Fail job request.
        """

        request = cls(
            worker_id=data["worker_id"],
            job_id=data["job_id"],
            error=data.get("error", ""),
            retryable=bool(data.get("retryable")),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class FailJobResponse(ProtocolEnvelope):
    """Response returned after job failure report."""

    status: ProtocolStatus = ProtocolStatus.OK
    message: str = ""

    def __init__(self, status: ProtocolStatus = ProtocolStatus.OK, message: str = "") -> None:
        """Create a fail job response.

        Args:
            status: Protocol response status.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.FAIL_JOB_RESPONSE)
        self.status = status
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "message": self.message,
        }


def _result_to_jsonable(result: TrainingResultSpec) -> dict[str, Any]:
    """Convert a result spec to JSON-friendly values.

    Args:
        result: Training result specification.

    Returns:
        Serializable dictionary.
    """

    output: dict[str, Any] = {}
    for key, value in result.__dict__.items():
        if hasattr(value, "value"):
            output[key] = value.value
        elif value is None:
            output[key] = None
        else:
            output[key] = str(value) if key.endswith("_path") else value
    return output


def _result_from_jsonable(data: dict[str, Any]) -> TrainingResultSpec:
    """Create a result spec from JSON-friendly values.

    Args:
        data: Serialized result.

    Returns:
        Training result specification.
    """

    from pathlib import Path
    from engine.contracts.jobs import JobStatus

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

