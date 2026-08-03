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
class Lion(torch.optim.Optimizer):
    """Lion optimizer with decoupled weight decay.
    The implementation follows the common Lion update rule and keeps the
    optimizer self-contained so the app does not require an extra dependency.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        """Create a Lion optimizer.
        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Momentum coefficients.
            weight_decay: Decoupled weight decay.
        """
        if lr <= 0.0:
            raise ValueError("lr must be greater than 0")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        """Perform one optimization step.
        Args:
            closure: Optional closure that reevaluates the model.
        Returns:
            Closure loss when a closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                state = self.state[parameter]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                parameter.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss
class TokenDataset(Dataset):
    """Sliding-window token dataset for next-token prediction.
    Accepts a list of ints, a numpy array, or a numpy memmap.  When backed by a
    memmap the full token stream lives on disk - only individual windows are
    loaded into RAM on each ``__getitem__`` call, so datasets of any size can be
    used without exhausting system memory.
    """
    def __init__(self, tokens: Union[list[int], np.ndarray], context_length: int, stride: int = 1) -> None:
        """Create a token dataset.
        Args:
            tokens: Complete token stream (list, ndarray, or memmap).
            context_length: Number of input tokens per sample.
            stride: Token offset step between consecutive windows.
        Raises:
            ValueError: If there are not enough tokens.
        """
        if len(tokens) <= context_length:
            raise ValueError("Not enough tokens for the selected context length")
        if stride <= 0:
            raise ValueError("stride must be greater than 0")
        # Keep the backing store as-is (memmap stays on disk).
        if isinstance(tokens, np.ndarray):
            self._tokens_np: Optional[np.ndarray] = tokens
            self._tokens_tensor: Optional[torch.Tensor] = None
        else:
            self._tokens_np = None
            self._tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride
        available_windows = len(tokens) - self.context_length
        self.sample_count = (available_windows + self.stride - 1) // self.stride
    def __len__(self) -> int:
        """Return the number of sliding windows available.
        Returns:
            Dataset length.
        """
        return self.sample_count
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one input/target token window.
        Args:
            index: Starting token index.
        Returns:
            Pair of input tokens and next-token targets.
        """
        start = index * self.stride
        end = start + self.context_length + 1
        if self._tokens_np is not None:
            # Read the slice from the numpy array / memmap and convert to tensor.
            chunk = torch.from_numpy(np.array(self._tokens_np[start:end], dtype=np.int64))
        else:
            chunk = self._tokens_tensor[start:end]  # type: ignore[index]
        return chunk[:-1], chunk[1:]
@dataclass
class TrainingResult:
    """Result returned after training.
    Attributes:
        checkpoint_path: Final model checkpoint path.
        summary_path: Training summary JSON path.
        final_train_loss: Final epoch training loss.
        final_val_loss: Final validation loss when available.
        stopped: Whether training was stopped by the user.
    """
    checkpoint_path: Path
    summary_path: Path
    final_train_loss: float
    final_val_loss: Optional[float]
    stopped: bool = False
@dataclass
class ResumeCompatibilityReport:
    """Compatibility result for a checkpoint resume attempt.
    Attributes:
        checkpoint_path: Checkpoint path that was inspected.
        errors: Blocking compatibility problems.
        warnings: Non-blocking but important differences.
        info: Informational compatibility details.
        can_load_optimizer_state: Whether optimizer state can be safely loaded.
        can_load_scheduler_state: Whether scheduler state can be safely loaded.
        can_load_scaler_state: Whether AMP scaler state can be safely loaded.
    """
    checkpoint_path: Path
    errors: list[str]
    warnings: list[str]
    info: list[str]
    can_load_optimizer_state: bool = True
    can_load_scheduler_state: bool = True
    can_load_scaler_state: bool = True
def emit_progress(
    progress: Optional[Callable[[Any], None]],
    message: str,
    percent: Optional[int] = None,
    **metrics: Any,
) -> None:
    """Emit training progress if a callback is available.
    Args:
        progress: Optional callback for progress dictionaries.
        message: Human-readable status message.
        percent: Optional progress percentage.
        **metrics: Optional structured metrics for UI dashboards.
    """
    if progress:
        progress({"message": message, "percent": percent, **metrics})
def set_seed(seed: int) -> None:
    """Set random seeds for repeatable training.
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def split_tokens(
    tokens: list[int],
    validation_split: float,
    chunk_size: int = 2048,
    seed: int = 1337,
) -> tuple[list[int], list[int]]:
    """Split tokens into train and validation streams.
    The corpus is written to disk as one big concatenation of source
    documents, then tokenized into a single flat stream. A plain positional
    split would make validation depend on whichever source happened to be at
    the tail of the corpus. This chunks and deterministically shuffles the
    stream first, so validation samples are drawn from across the corpus.
    Args:
        tokens: Full token stream.
        validation_split: Fraction reserved for validation.
        chunk_size: Number of tokens per shuffle unit.
        seed: Fixed seed for reproducible train/validation assignment.
    Returns:
        Pair of training tokens and validation tokens.
    """
    total = len(tokens)
    if total <= 1 or validation_split <= 0:
        return list(tokens), []
    if validation_split >= 1:
        return [], list(tokens)
    chunk_size = max(1, chunk_size)
    chunk_ranges = [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]
    if len(chunk_ranges) <= 1:
        split_at = int(total * (1.0 - validation_split))
        split_at = max(1, min(split_at, total - 1))
        return tokens[:split_at], tokens[split_at:]
    shuffled_indices = list(range(len(chunk_ranges)))
    random.Random(seed).shuffle(shuffled_indices)
    val_chunk_count = max(1, round(len(chunk_ranges) * validation_split))
    val_chunk_count = min(val_chunk_count, len(chunk_ranges) - 1)
    val_chunk_indices = set(shuffled_indices[:val_chunk_count])
    train_tokens: list[int] = []
    val_tokens: list[int] = []
    for chunk_index, (start, end) in enumerate(chunk_ranges):
        piece = tokens[start:end]
        if chunk_index in val_chunk_indices:
            val_tokens.extend(piece)
        else:
            train_tokens.extend(piece)
    return train_tokens, val_tokens
def split_tokens_to_files(
    tokens: np.memmap,
    train_path: Path,
    val_path: Path,
    validation_split: float,
    dtype: np.dtype,
    chunk_size: int = 2048,
    seed: int = 1337,
    should_stop: Optional[Callable[[], bool]] = None,
) -> tuple[int, int]:
    """Split a token stream into train/validation ``.npy`` files on disk.
    Behaves like :func:`split_tokens` (same chunked, seeded shuffle so
    validation samples are drawn from across the corpus, not just the tail),
    but never materializes the full train or validation token stream in
    memory. ``tokens`` is expected to be a read-only memmap (or any
    ``__len__``/slice-able array) backed by disk; each chunk is read, cast to
    ``dtype``, and written straight to the appropriate output file. Peak
    memory use is therefore bounded by ``chunk_size`` regardless of corpus
    size.
    Args:
        tokens: Full token stream, typically a memory-mapped ``.npy`` array.
        train_path: Destination path for the training token ``.npy`` file.
        val_path: Destination path for the validation token ``.npy`` file.
        validation_split: Fraction of chunks reserved for validation.
        dtype: Integer dtype to store each token ID as (see
            ``tokenizer.token_dtype_for_vocab``).
        chunk_size: Number of tokens per shuffle unit.
        seed: Fixed seed for reproducible train/validation assignment.
        should_stop: Optional callback returning true when the split should
            stop early.
    Returns:
        Pair of ``(train_token_count, val_token_count)``.
    Raises:
        RuntimeError: If cancellation is requested.
    """
    total = len(tokens)
    chunk_size = max(1, chunk_size)
    validation_split = max(0.0, min(1.0, validation_split))
    if total <= 1 or validation_split <= 0:
        chunk_ranges: list[tuple[int, int]] = [(0, total)]
        val_chunk_indices: set[int] = set()
    elif validation_split >= 1:
        chunk_ranges = [(0, total)]
        val_chunk_indices = {0}
    else:
        chunk_ranges = [
            (start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)
        ]
        if len(chunk_ranges) <= 1:
            split_at = int(total * (1.0 - validation_split))
            split_at = max(1, min(split_at, total - 1))
            chunk_ranges = [(0, split_at), (split_at, total)]
            val_chunk_indices = {1}
        else:
            shuffled_indices = list(range(len(chunk_ranges)))
            random.Random(seed).shuffle(shuffled_indices)
            val_chunk_count = max(1, round(len(chunk_ranges) * validation_split))
            val_chunk_count = min(val_chunk_count, len(chunk_ranges) - 1)
            val_chunk_indices = set(shuffled_indices[:val_chunk_count])
    train_token_count = sum(
        end - start for index, (start, end) in enumerate(chunk_ranges) if index not in val_chunk_indices
    )
    val_token_count = sum(
        end - start for index, (start, end) in enumerate(chunk_ranges) if index in val_chunk_indices
    )
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train_header = {
        "descr": npy_format.dtype_to_descr(np.dtype(dtype)),
        "fortran_order": False,
        "shape": (train_token_count,),
    }
    val_header = {
        "descr": npy_format.dtype_to_descr(np.dtype(dtype)),
        "fortran_order": False,
        "shape": (val_token_count,),
    }
    with train_path.open("wb") as train_file, val_path.open("wb") as val_file:
        npy_format.write_array_header_1_0(train_file, train_header)
        npy_format.write_array_header_1_0(val_file, val_header)
        for chunk_index, (start, end) in enumerate(chunk_ranges):
            if should_stop and should_stop():
                raise RuntimeError("Dataset preparation stopped by user.")
            piece = np.asarray(tokens[start:end], dtype=dtype)
            if chunk_index in val_chunk_indices:
                piece.tofile(val_file)
            else:
                piece.tofile(train_file)
    return train_token_count, val_token_count
def make_optimizer(model: MicroGPT, training_config: TrainingConfig) -> torch.optim.Optimizer:
    """Create the configured optimizer.
    Args:
        model: Model whose parameters will be optimized.
        training_config: Training configuration.
    Returns:
        Configured optimizer.
    Raises:
        ValueError: If the optimizer is unsupported by the installed PyTorch.
    """
    name = training_config.optimizer_name
    common = {
        "lr": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
    }
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters are available for optimization")
    if name == "adamw":
        return torch.optim.AdamW(parameters, betas=(0.9, 0.95), **common)
    if name == "adam":
        return torch.optim.Adam(parameters, betas=(0.9, 0.95), **common)
    if name == "lion":
        return Lion(parameters, betas=(0.9, 0.99), **common)
    if name == "adafactor":
        adafactor = getattr(torch.optim, "Adafactor", None)
        if adafactor is None:
            raise ValueError("Adafactor requires a newer PyTorch build that includes torch.optim.Adafactor")
        return adafactor(parameters, **common)
    raise ValueError(f"Unsupported optimizer: {name}")
def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    training_config: TrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create the configured learning-rate scheduler.
    Args:
        optimizer: Optimizer to schedule.
        total_steps: Total optimizer steps.
        training_config: Training configuration.
    Returns:
        Lambda learning-rate scheduler.
    """
    warmup_steps = training_config.warmup_steps
    warmup_steps = min(warmup_steps, max(total_steps - 1, 1))
    min_ratio = training_config.scheduler_min_lr_ratio
    schedule = training_config.scheduler_name
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = max(0.0, min(progress, 1.0))
        if schedule == "cosine":
            value = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * value
        if schedule == "polynomial":
            value = (1.0 - progress) ** training_config.polynomial_power
            return min_ratio + (1.0 - min_ratio) * value
        if schedule == "one_cycle":
            if progress < 0.3:
                return min_ratio + (1.0 - min_ratio) * (progress / 0.3)
            decay_progress = (progress - 0.3) / 0.7
            value = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_ratio + (1.0 - min_ratio) * value
        return max(min_ratio, 1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
def amp_settings(training_config: TrainingConfig) -> tuple[bool, bool, torch.dtype]:
    """Return autocast and scaler settings for the selected precision.
    Args:
        training_config: Training configuration.
    Returns:
        Tuple of ``use_autocast``, ``use_scaler``, and autocast dtype.
    """
    use_cuda_amp = training_config.use_amp and training_config.device == "cuda"
    if not use_cuda_amp or training_config.precision == "fp32":
        return False, False, torch.float32
    if training_config.precision == "bf16":
        return True, False, torch.bfloat16
    return True, True, torch.float16
def system_ram_percent() -> Optional[float]:
    """Return system RAM utilization when psutil is available.
    Returns:
        RAM utilization percentage, or None when unavailable.
    """
    if psutil is None:
        return None
    return float(psutil.virtual_memory().percent)
def system_cpu_percent() -> Optional[float]:
    """Return system CPU utilization when psutil is available.
    Returns:
        CPU utilization percentage, or None when unavailable.
    """
    if psutil is None:
        return None
    return float(psutil.cpu_percent(interval=None))
