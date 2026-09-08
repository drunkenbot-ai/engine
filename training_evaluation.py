from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from .model import MicroGPT
from .training_runtime import emit_progress, system_cpu_percent, system_ram_percent


class TrainingStopRequested(RuntimeError):
    """Signal cooperative cancellation from inside validation."""


def evaluate(
    model: MicroGPT,
    loader: DataLoader,
    device: str,
    pad_token_id: int,
    max_batches: int = 50,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    step: Optional[int] = None,
    total_steps: Optional[int] = None,
    percent: Optional[int] = None,
    use_autocast: bool = False,
    autocast_dtype: torch.dtype = torch.float32,
) -> float:
    """Evaluate validation loss.

    Args:
        model: Model to evaluate.
        loader: Validation data loader.
        device: Device used for evaluation.
        pad_token_id: Token ID ignored in loss.
        max_batches: Maximum validation batches to evaluate.
            Zero evaluates the full loader.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.
        step: Current optimizer step for progress metrics.
        total_steps: Total planned optimizer steps for progress metrics.
        percent: Current outer training progress percentage.
        use_autocast: Whether to use mixed-precision autocast.
        autocast_dtype: Datatype for autocast when enabled.

    Returns:
        Mean validation loss.

    Raises:
        TrainingStopRequested: When the user requests a stop.
    """
    pin_memory = device.startswith("cuda")
    model.eval()
    total_loss_sum = 0.0
    total_valid_tokens = 0
    batch_limit = len(loader) if max_batches <= 0 else min(len(loader), max_batches)
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader, start=1):
            if should_stop and should_stop():
                model.train()
                raise TrainingStopRequested(
                    "Training stopped by user during validation."
                )
            if batch_index > batch_limit:
                break
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)
            with autocast("cuda", enabled=use_autocast, dtype=autocast_dtype):
                logits = model(x)
                targets_flat = y.reshape(-1)
                if pad_token_id is not None and pad_token_id >= 0:
                    targets_flat = targets_flat.clone()
                    targets_flat[targets_flat == pad_token_id] = -100
                valid_mask = (targets_flat != -100)
                if valid_mask.any():
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets_flat,
                        ignore_index=-100,
                        reduction="sum",
                    )
                    total_loss_sum += float(loss_sum.item())
                    total_valid_tokens += int(valid_mask.sum().item())
            if progress and (
                batch_index == 1
                or batch_index == batch_limit
                or batch_index % 10 == 0
            ):
                emit_progress(
                    progress,
                    f"Validation running: batch {batch_index}/{batch_limit}.",
                    percent,
                    step=step,
                    total_steps=total_steps,
                    system_cpu_percent=system_cpu_percent(),
                    system_ram_percent=system_ram_percent(),
                    validation_batch=batch_index,
                    validation_batches=batch_limit,
                )
    model.train()
    if total_valid_tokens > 0:
        return total_loss_sum / total_valid_tokens
    return 0.0
