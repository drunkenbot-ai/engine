from __future__ import annotations
from .training_runtime import *
from .training_resume import _release_cuda_cache


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
) -> float:
    """Evaluate validation loss.
    Args:
        model: Model to evaluate.
        loader: Validation data loader.
        device: Device used for evaluation.
        pad_token_id: Token ID ignored in loss.
        max_batches: Maximum validation batches to evaluate. Zero evaluates the full loader.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.
        step: Current optimizer step for progress metrics.
        total_steps: Total planned optimizer steps for progress metrics.
        percent: Current outer training progress percentage.
    Returns:
        Mean validation loss.
    """
    model.eval()
    losses: list[float] = []
    batch_limit = len(loader) if max_batches <= 0 else min(len(loader), max_batches)
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader, start=1):
            if should_stop and should_stop():
                model.train()
                raise RuntimeError("Training stopped by user during validation.")
            if batch_index > batch_limit:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=pad_token_id,
            )
            losses.append(float(loss.item()))
            if progress and (batch_index == 1 or batch_index == batch_limit or batch_index % 10 == 0):
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
    _release_cuda_cache()
    return sum(losses) / max(len(losses), 1)

