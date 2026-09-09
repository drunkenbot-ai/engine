"""Persistent GGUF chat session backed by llama-cpp-python.

Note: Core implementation has moved to `inference.core.llama_chat`.
This module re-exports all symbols for backwards compatibility.
"""

from __future__ import annotations

from inference.core.llama_chat import (
    LlamaChatSession,
    generate_chat_reply,
    load_llama_chat_session,
    stream_chat_reply,
)

__all__ = [
    "LlamaChatSession",
    "load_llama_chat_session",
    "generate_chat_reply",
    "stream_chat_reply",
]
