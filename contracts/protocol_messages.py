"""Compatibility facade for protocol message contracts."""

from .protocol_messages_core import (
    RegisterWorkerRequest, RegisterWorkerResponse, HeartbeatRequest,
    HeartbeatResponse, ClaimJobRequest, ClaimJobResponse, ProgressReportRequest,
    ProgressReportResponse,
)
from .protocol_messages_extended import (
    CompleteJobRequest, CompleteJobResponse, FailJobRequest, FailJobResponse,
)

__all__ = [name for name in globals() if not name.startswith("_")]

