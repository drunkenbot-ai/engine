"""Persistent chat session backed by a native MicroGPT checkpoint.

Note: Core implementation has moved to `inference.core.microgpt_chat`.
This module re-exports all symbols for backwards compatibility.
"""

from __future__ import annotations

from inference.core.microgpt_chat import (
    STOP_SEQUENCES,
    MicroGPTChatSession,
    _resolve_model_checkpoint,
    load_microgpt_chat_session,
    stream_microgpt_chat_reply,
)

__all__ = [
    "STOP_SEQUENCES",
    "MicroGPTChatSession",
    "_resolve_model_checkpoint",
    "load_microgpt_chat_session",
    "stream_microgpt_chat_reply",
]
