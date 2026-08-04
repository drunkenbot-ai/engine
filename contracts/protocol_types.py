from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from engine.contracts.jobs import BackendKind, TrainingJobSpec, TrainingMetrics, TrainingResultSpec, utc_now_iso


class ProtocolMessageKind(str, Enum):
    """Coordinator protocol message kind."""

    REGISTER_WORKER_REQUEST = "register_worker_request"
    REGISTER_WORKER_RESPONSE = "register_worker_response"
    HEARTBEAT_REQUEST = "heartbeat_request"
    HEARTBEAT_RESPONSE = "heartbeat_response"
    CLAIM_JOB_REQUEST = "claim_job_request"
    CLAIM_JOB_RESPONSE = "claim_job_response"
    PROGRESS_REPORT_REQUEST = "progress_report_request"
    PROGRESS_REPORT_RESPONSE = "progress_report_response"
    COMPLETE_JOB_REQUEST = "complete_job_request"
    COMPLETE_JOB_RESPONSE = "complete_job_response"
    FAIL_JOB_REQUEST = "fail_job_request"
    FAIL_JOB_RESPONSE = "fail_job_response"


class ProtocolStatus(str, Enum):
    """Coordinator protocol response status."""

    OK = "ok"
    REJECTED = "rejected"
    ERROR = "error"


class WorkerAvailability(str, Enum):
    """Remote worker availability status."""

    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class ProtocolEnvelope:
    """Base metadata for coordinator protocol messages.

    Attributes:
        message_id: Unique protocol message identifier.
        kind: Protocol message kind.
        sent_at: UTC timestamp when the message was created.
        protocol_version: Protocol version string.
    """

    kind: ProtocolMessageKind
    message_id: str = field(default_factory=lambda: f"msg_{uuid4().hex}")
    sent_at: str = field(default_factory=utc_now_iso)
    protocol_version: str = "0.1"

    def envelope_json(self) -> dict[str, Any]:
        """Return JSON-friendly envelope fields.

        Returns:
            Serializable envelope dictionary.
        """

        return {
            "message_id": self.message_id,
            "kind": self.kind.value,
            "sent_at": self.sent_at,
            "protocol_version": self.protocol_version,
        }


def _restore_envelope(message: ProtocolEnvelope, data: dict[str, Any]) -> None:
    """Restore envelope fields on a protocol message.

    Args:
        message: Protocol message.
        data: Serialized message data.
    """

    message.message_id = data.get("message_id", message.message_id)
    message.sent_at = data.get("sent_at", message.sent_at)
    message.protocol_version = data.get("protocol_version", message.protocol_version)


@dataclass
class WorkerCapabilities:
    """Hardware and runtime capabilities advertised by a worker.

    Attributes:
        hostname: Optional host name.
        platform: Operating system or runtime platform label.
        cpu_count: Logical CPU count.
        system_ram_gb: System RAM in GB.
        gpu_names: GPU names visible to the worker.
        total_vram_gb: Total visible GPU VRAM in GB.
        supports_cuda: Whether CUDA is available.
        supports_bf16: Whether BF16 is available.
        supports_fp16: Whether FP16 is available.
        extra: Free-form implementation details.
    """

    hostname: Optional[str] = None
    platform: Optional[str] = None
    cpu_count: Optional[int] = None
    system_ram_gb: Optional[float] = None
    gpu_names: list[str] = field(default_factory=list)
    total_vram_gb: Optional[float] = None
    supports_cuda: bool = False
    supports_bf16: bool = False
    supports_fp16: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        """Convert capabilities to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "system_ram_gb": self.system_ram_gb,
            "gpu_names": self.gpu_names,
            "total_vram_gb": self.total_vram_gb,
            "supports_cuda": self.supports_cuda,
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
            "extra": self.extra,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "WorkerCapabilities":
        """Create capabilities from JSON-friendly values.

        Args:
            data: Serialized capabilities.

        Returns:
            Worker capabilities.
        """

        return cls(
            hostname=data.get("hostname"),
            platform=data.get("platform"),
            cpu_count=data.get("cpu_count"),
            system_ram_gb=data.get("system_ram_gb"),
            gpu_names=list(data.get("gpu_names") or []),
            total_vram_gb=data.get("total_vram_gb"),
            supports_cuda=bool(data.get("supports_cuda")),
            supports_bf16=bool(data.get("supports_bf16")),
            supports_fp16=bool(data.get("supports_fp16")),
            extra=dict(data.get("extra") or {}),
        )



