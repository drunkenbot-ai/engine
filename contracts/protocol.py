"""Compatibility facade for protocol contracts.

The concrete protocol definitions live in focused modules.
"""

from .protocol_types import (
    ProtocolEnvelope, ProtocolMessageKind, ProtocolStatus, WorkerAvailability,
    WorkerCapabilities,
)
from .protocol_messages import (
    ClaimJobRequest, ClaimJobResponse, CompleteJobRequest, CompleteJobResponse,
    FailJobRequest, FailJobResponse, HeartbeatRequest, HeartbeatResponse,
    ProgressReportRequest, ProgressReportResponse, RegisterWorkerRequest,
    RegisterWorkerResponse,
)

__all__ = [name for name in globals() if not name.startswith("_")]

