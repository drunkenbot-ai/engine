from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from .config import ModelConfig, TrainingConfig, dataclass_to_jsonable
from .data import file_sha256
from .lineage import read_json
from .training import latest_checkpoint


def _resume_checkpoint_for(training_config: TrainingConfig) -> Optional[Path]:
    """Return the checkpoint that will be used for resume, if any.

    Args:
        training_config: Training configuration.

    Returns:
        Resume checkpoint path or ``None``.
    """

    if not training_config.resume:
        return None
    if training_config.resume_from_checkpoint:
        return Path(training_config.resume_from_checkpoint)
    return latest_checkpoint(training_config.output_dir / "checkpoints")


def _compatible_model_config(model_config: ModelConfig) -> dict[str, Any]:
    """Return checkpoint compatibility fields for a model config.

    Args:
        model_config: Current model configuration.

    Returns:
        Dictionary of architecture-shape fields.
    """

    data = dataclass_to_jsonable(model_config)
    return {
        key: data.get(key)
        for key in (
            "vocab_size",
            "context_length",
            "embedding_size",
            "head_count",
            "layer_count",
            "bias",
            "norm_type",
            "position_encoding",
            "mlp_type",
            "rope_theta",
            "attention_type",
        )
    }


def _tokenizer_files_match(left: Path, right: Path) -> bool:
    """Return whether tokenizer files are byte-identical or JSON-equivalent.

    Args:
        left: First tokenizer path.
        right: Second tokenizer path.

    Returns:
        True when tokenizers are compatible.
    """

    if file_sha256(left) == file_sha256(right):
        return True
    left_json = read_json(left, default=None)
    right_json = read_json(right, default=None)
    return left_json is not None and left_json == right_json


def _validate_resume_compatibility(
    data_dir: Path,
    tokenizer_path: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> Optional[Path]:
    """Validate tokenizer and architecture before continuing training.

    Args:
        data_dir: Prepared dataset folder.
        tokenizer_path: Dataset tokenizer path.
        model_config: Current model architecture.
        training_config: Training configuration.

    Returns:
        Resume checkpoint path when one exists.

    Raises:
        ValueError: If tokenizer or architecture is incompatible.
    """

    resume_path = _resume_checkpoint_for(training_config)
    if not resume_path or not resume_path.exists() or not training_config.require_compatible_resume:
        return resume_path

    existing_tokenizer = training_config.output_dir / "tokenizer.json"
    if existing_tokenizer.exists():
        if not _tokenizer_files_match(existing_tokenizer, tokenizer_path):
            raise ValueError(
                "Resume safety check failed: the selected dataset tokenizer does not match the tokenizer "
                f"used by the existing model folder.\nExisting tokenizer: {existing_tokenizer}\n"
                f"Dataset tokenizer: {tokenizer_path}\n\nUse the same tokenizer policy for continued training, "
                "or choose a new model output folder to start a new model."
            )

    checkpoint = torch.load(resume_path, map_location="cpu")
    checkpoint_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"Resume safety check failed: checkpoint has no model_config: {resume_path}")

    current = _compatible_model_config(model_config)
    legacy_defaults = {
        "bias": True,
        "norm_type": "layernorm",
        "position_encoding": "learned",
        "mlp_type": "gelu",
        "rope_theta": 10000.0,
        "attention_type": "mha",
    }
    previous = {key: checkpoint_config.get(key, legacy_defaults.get(key)) for key in current}
    mismatches = {
        key: {"checkpoint": previous.get(key), "current": current.get(key)}
        for key in current
        if previous.get(key) != current.get(key)
    }
    if mismatches:
        mismatch_text = ", ".join(
            f"{key} checkpoint={value['checkpoint']} current={value['current']}"
            for key, value in mismatches.items()
        )
        raise ValueError(
            "Resume safety check failed: model architecture does not match the checkpoint. "
            f"{mismatch_text}. Keep architecture settings identical for continued training, "
            "or use a new model output folder."
        )

    return resume_path

__all__ = [
    "_resume_checkpoint_for",
    "_compatible_model_config",
    "_tokenizer_files_match",
    "_validate_resume_compatibility",
]

