"""Checkpoint benchmark evaluation utilities.

Note: Core implementation has moved to `inference.core.evaluation`.
This module re-exports all symbols for backwards compatibility.
"""

from __future__ import annotations

from inference.core.evaluation import (
    DEFAULT_BENCHMARK_PROMPTS,
    BenchmarkResult,
    evaluate_checkpoint,
    normalize_prompts,
)

__all__ = [
    "DEFAULT_BENCHMARK_PROMPTS",
    "BenchmarkResult",
    "normalize_prompts",
    "evaluate_checkpoint",
]
