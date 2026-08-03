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
from .training_resume import *

def save_checkpoint(
    path: Path,
    model: MicroGPT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    global_step: int,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
) -> None:
    """Save a resumable training checkpoint.

    Args:
        path: Destination checkpoint path.
        model: Model being trained.
        optimizer: Optimizer state to save.
        scheduler: Learning-rate scheduler state to save.
        scaler: AMP scaler state to save.
        model_config: Model configuration.
        training_config: Training configuration.
        global_step: Current optimizer step.
        epoch: Current epoch number.
        train_loss: Most recent training loss.
        val_loss: Most recent validation loss.
    """

    payload = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "model_config": dataclass_to_jsonable(model_config),
        "training_config": dataclass_to_jsonable(training_config),
        "global_step": global_step,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    if training_config.peft_method == "lora" and path.name != "final_model.pt":
        payload["adapter_state_dict"] = lora_state_dict(model)
        payload["fine_tune_base_checkpoint"] = (
            str(training_config.fine_tune_from_checkpoint)
            if training_config.fine_tune_from_checkpoint
            else None
        )
        payload["lora_config"] = {
            "rank": training_config.lora_rank,
            "alpha": training_config.lora_alpha,
            "dropout": training_config.lora_dropout,
            "target_modules": training_config.lora_target_modules,
        }
    else:
        payload["model_state_dict"] = model.state_dict()
    torch.save(payload, path)
from .training_resume import _release_cuda_cache, _configure_cuda_allocator, _estimated_training_vram_bytes

