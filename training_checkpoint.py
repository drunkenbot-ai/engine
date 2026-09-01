from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import torch
from torch.amp import GradScaler

from .config import ModelConfig, TrainingConfig, dataclass_to_jsonable
from .model import (
    MicroGPT,
    lora_state_dict,
    merged_lora_state_dict,
)

CheckpointArtifact = Literal["resume", "adapter", "inference"]


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
    artifact_type: CheckpointArtifact = "resume",
) -> None:
    """Save a resume, adapter, or chat-loadable inference artifact.

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
        artifact_type: Explicit artifact semantics. Resume artifacts include
            optimizer state, adapter artifacts contain LoRA deltas, and
            inference artifacts contain complete plain-model weights.
    """

    if artifact_type not in {"resume", "adapter", "inference"}:
        raise ValueError(f"Unsupported checkpoint artifact type: {artifact_type}")
    if artifact_type == "adapter" and training_config.peft_method != "lora":
        raise ValueError("Adapter checkpoints require LoRA training.")

    payload = {
        "artifact_type": artifact_type,
        "model_config": dataclass_to_jsonable(model_config),
        "training_config": dataclass_to_jsonable(training_config),
        "global_step": global_step,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    if artifact_type == "resume":
        payload.update(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
        )
    if training_config.peft_method == "lora" and artifact_type in {"resume", "adapter"}:
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
    elif training_config.peft_method == "lora":
        payload["model_state_dict"] = merged_lora_state_dict(model)
    else:
        payload["model_state_dict"] = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        }
    torch.save(payload, path)
