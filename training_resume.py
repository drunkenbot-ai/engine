from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional, Union

import numpy as np
import numpy.lib.format as npy_format
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig, TrainingConfig, dataclass_to_jsonable
from .model import (
    MicroGPT,
    apply_lora_adapters,
    freeze_non_lora_parameters,
    load_lora_state_dict,
    lora_parameter_count,
    lora_state_dict,
    merge_lora_adapters,
)

try:
    import psutil
except ImportError:
    psutil = None



from .training_core import *
from .training_runtime import *
from .training_evaluation import *
from .training_evaluation import *

def latest_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    """Find the newest checkpoint in a folder.

    Args:
        checkpoints_dir: Directory containing checkpoint files.

    Returns:
        Newest checkpoint path, or ``None``.
    """

    checkpoints = sorted(
        checkpoints_dir.glob("checkpoint_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def _config_value(data: dict[str, Any], key: str, default: Any) -> Any:
    """Return a saved config value with a default for old checkpoints.

    Args:
        data: Saved configuration dictionary.
        key: Configuration key.
        default: Default value when the key is missing.

    Returns:
        Saved or default value.
    """

    return data[key] if key in data else default


def _same_config_value(left: Any, right: Any) -> bool:
    """Compare config values with tolerance for numeric fields.

    Args:
        left: First value.
        right: Second value.

    Returns:
        True when values are effectively equal.
    """

    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1e-9
        except (TypeError, ValueError):
            return False
    return left == right


def _saved_model_default(key: str) -> Any:
    """Return ModelConfig defaults for legacy checkpoints.

    Args:
        key: ModelConfig field name.

    Returns:
        Default value used by current ModelConfig.
    """

    defaults = {
        "context_length": 128,
        "embedding_size": 256,
        "head_count": 4,
        "layer_count": 4,
        "dropout": 0.1,
        "bias": True,
        "norm_type": "layernorm",
        "position_encoding": "learned",
        "mlp_type": "gelu",
        "rope_theta": 10000.0,
        "attention_type": "mha",
        "kv_head_count": 0,
        "attention_backend": "sdpa",
        "attention_window": 0,
    }
    return defaults.get(key)


def _saved_training_default(key: str) -> Any:
    """Return TrainingConfig defaults for legacy checkpoints.

    Args:
        key: TrainingConfig field name.

    Returns:
        Default value used by current TrainingConfig.
    """

    defaults = {
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_linear",
        "scheduler_min_lr_ratio": 0.1,
        "polynomial_power": 1.0,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "precision": "fp16",
        "use_amp": True,
        "training_mode": "pretrain",
        "fine_tune_from_checkpoint": None,
    }
    return defaults.get(key)


def check_resume_compatibility(
    checkpoint_path: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> ResumeCompatibilityReport:
    """Check whether a checkpoint can be safely resumed.

    Args:
        checkpoint_path: Checkpoint to inspect.
        model_config: Current model configuration.
        training_config: Current training configuration.

    Returns:
        Resume compatibility report.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_model = checkpoint.get("model_config", {})
    saved_training = checkpoint.get("training_config", {})
    if not isinstance(saved_model, dict):
        saved_model = {}
    if not isinstance(saved_training, dict):
        saved_training = {}

    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = [f"Resume checkpoint: {checkpoint_path.name}."]

    critical_model_fields = (
        ("vocab_size", model_config.vocab_size, "Tokenizer vocabulary"),
        ("context_length", model_config.context_length, "Context length"),
        ("embedding_size", model_config.embedding_size, "n_embd"),
        ("head_count", model_config.head_count, "n_head"),
        ("layer_count", model_config.layer_count, "n_layer"),
        ("bias", model_config.bias, "Bias layout"),
        ("norm_type", model_config.norm_type, "Normalization"),
        ("position_encoding", model_config.position_encoding, "Position encoding"),
        ("mlp_type", model_config.mlp_type, "MLP type"),
        ("rope_theta", model_config.rope_theta, "RoPE theta"),
        ("attention_type", model_config.attention_type, "Attention type"),
    )
    for key, current_value, label in critical_model_fields:
        saved_value = _config_value(saved_model, key, _saved_model_default(key))
        if not _same_config_value(saved_value, current_value):
            errors.append(f"{label} changed: checkpoint={saved_value}, current={current_value}.")

    saved_attention_type = _config_value(saved_model, "attention_type", "mha")
    saved_kv_heads = _config_value(saved_model, "kv_head_count", 0)
    try:
        saved_kv_effective = ModelConfig(
            vocab_size=int(_config_value(saved_model, "vocab_size", model_config.vocab_size)),
            context_length=int(_config_value(saved_model, "context_length", _saved_model_default("context_length"))),
            embedding_size=int(_config_value(saved_model, "embedding_size", _saved_model_default("embedding_size"))),
            head_count=int(_config_value(saved_model, "head_count", _saved_model_default("head_count"))),
            layer_count=int(_config_value(saved_model, "layer_count", _saved_model_default("layer_count"))),
            attention_type=str(saved_attention_type),
            kv_head_count=int(saved_kv_heads),
        ).resolved_kv_head_count()
    except Exception:
        saved_kv_effective = saved_kv_heads
    current_kv_effective = model_config.resolved_kv_head_count()
    if saved_kv_effective != current_kv_effective:
        errors.append(f"Effective KV heads changed: checkpoint={saved_kv_effective}, current={current_kv_effective}.")

    warning_model_fields = (
        ("dropout", model_config.dropout, "Dropout"),
        ("attention_backend", model_config.attention_backend, "Attention backend"),
        ("attention_window", model_config.attention_window, "Sliding attention window"),
    )
    for key, current_value, label in warning_model_fields:
        saved_value = _config_value(saved_model, key, _saved_model_default(key))
        if not _same_config_value(saved_value, current_value):
            warnings.append(f"{label} changed: checkpoint={saved_value}, current={current_value}.")

    can_load_optimizer_state = True
    can_load_scheduler_state = True
    can_load_scaler_state = True
    if "optimizer_state_dict" in checkpoint:
        saved_optimizer = _config_value(saved_training, "optimizer_name", _saved_training_default("optimizer_name"))
        if saved_optimizer != training_config.optimizer_name:
            warnings.append(
                f"Optimizer changed: checkpoint={saved_optimizer}, current={training_config.optimizer_name}. "
                "Optimizer state will not be loaded."
            )
            can_load_optimizer_state = False
    if "scheduler_state_dict" in checkpoint:
        saved_scheduler = _config_value(saved_training, "scheduler_name", _saved_training_default("scheduler_name"))
        if saved_scheduler != training_config.scheduler_name:
            warnings.append(
                f"LR scheduler changed: checkpoint={saved_scheduler}, current={training_config.scheduler_name}. "
                "Scheduler state will not be loaded."
            )
            can_load_scheduler_state = False
        for key, current_value, label in (
            ("scheduler_min_lr_ratio", training_config.scheduler_min_lr_ratio, "Scheduler min LR ratio"),
            ("polynomial_power", training_config.polynomial_power, "Polynomial power"),
        ):
            saved_value = _config_value(saved_training, key, _saved_training_default(key))
            if not _same_config_value(saved_value, current_value):
                warnings.append(f"{label} changed: checkpoint={saved_value}, current={current_value}.")
    if "scaler_state_dict" in checkpoint:
        saved_precision = _config_value(saved_training, "precision", _saved_training_default("precision"))
        if saved_precision != training_config.precision:
            warnings.append(f"Precision changed: checkpoint={saved_precision}, current={training_config.precision}.")
            can_load_scaler_state = saved_precision == "fp16" and training_config.precision == "fp16"

    for key, current_value, label in (
        ("learning_rate", training_config.learning_rate, "Learning rate"),
        ("weight_decay", training_config.weight_decay, "Weight decay"),
        ("max_grad_norm", training_config.max_grad_norm, "Gradient clipping"),
    ):
        saved_value = _config_value(saved_training, key, _saved_training_default(key))
        if not _same_config_value(saved_value, current_value):
            warnings.append(f"{label} changed: checkpoint={saved_value}, current={current_value}.")

    if not errors:
        info.append("Checkpoint architecture and tokenizer are compatible.")
    return ResumeCompatibilityReport(
        checkpoint_path=checkpoint_path,
        errors=errors,
        warnings=warnings,
        info=info,
        can_load_optimizer_state=can_load_optimizer_state,
        can_load_scheduler_state=can_load_scheduler_state,
        can_load_scaler_state=can_load_scaler_state,
    )


def _configure_cuda_allocator() -> None:
    """Configure the CUDA caching allocator to reduce VRAM over-reservation.

    Sets ``expandable_segments:True`` so PyTorch grows GPU memory in smaller
    increments rather than grabbing large contiguous blocks up front.
    """

    key = "PYTORCH_CUDA_ALLOC_CONF"
    current = os.environ.get(key, "")
    if "expandable_segments" not in current:
        new_value = "expandable_segments:True"
        if current:
            new_value = current + "," + new_value
        os.environ[key] = new_value


def _release_cuda_cache() -> None:
    """Release unused cached VRAM back to the OS."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _estimated_training_vram_bytes(model: MicroGPT, model_config: ModelConfig, training_config: TrainingConfig) -> int:
    """Return a conservative, explainable training-memory estimate.

    Parameters, gradients, and Adam-style optimizer states are generally
    fp32.  Activations vary by kernel, therefore the estimate intentionally
    includes headroom rather than pretending to be exact.
    """

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_and_optimizer = parameter_count * 16  # fp32 weights, grads, m/v
    activation_bytes_per_value = 2 if training_config.use_amp and training_config.precision != "fp32" else 4
    activation_multiplier = 3 if training_config.activation_checkpointing else 10
    activations = (
        training_config.batch_size
        * model_config.context_length
        * model_config.embedding_size
        * model_config.layer_count
        * activation_bytes_per_value
        * activation_multiplier
    )
    logits = training_config.batch_size * model_config.context_length * model_config.vocab_size * activation_bytes_per_value
    return int((parameter_and_optimizer + activations + logits) * 1.15)




