from __future__ import annotations

"""Compatibility facade for split service subsystems."""

from .dataset_build import DatasetBuildResult, build_dataset, content_warning, estimate_vocab_size
from .dataset_preview import DatasetPreviewResult, ProjectHealthResult, check_project_health, scan_dataset_preview
from .training_orchestrator import train_from_dataset

__all__ = [
    "DatasetBuildResult",
    "ProjectHealthResult",
    "DatasetPreviewResult",
    "build_dataset",
    "train_from_dataset",
    "check_project_health",
    "scan_dataset_preview",
    "estimate_vocab_size",
    "content_warning",
]

