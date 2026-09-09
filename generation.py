"""Checkpoint generation utilities.

Note: Core implementation has moved to `inference.core.generation`.
This module re-exports all symbols for backwards compatibility.
"""

from __future__ import annotations

from inference.core.generation import (
    generate_text,
    load_model_from_checkpoint,
)

__all__ = [
    "load_model_from_checkpoint",
    "generate_text",
]
