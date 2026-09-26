from __future__ import annotations

from .config import ModelConfig, TrainingConfig


def estimate_model_parameters(model_config: ModelConfig) -> int:
    """Estimate trainable parameters for a MicroGPT architecture.

    Args:
        model_config: Model architecture configuration.

    Returns:
        Approximate trainable parameter count.
    """

    vocab = model_config.vocab_size
    emb = model_config.embedding_size
    layers = model_config.layer_count
    breakdown = estimate_parameter_breakdown(model_config)
    return int(sum(breakdown.values()))


def estimate_parameter_breakdown(model_config: ModelConfig) -> dict[str, int]:
    """Estimate parameter groups for a MicroGPT architecture.

    Args:
        model_config: Model architecture configuration.

    Returns:
        Dictionary with parameter counts by major model component.
    """

    vocab = model_config.vocab_size
    emb = model_config.embedding_size
    layers = model_config.layer_count
    tied = getattr(model_config, "tie_word_embeddings", True)
    token_embedding = vocab * emb if tied else (2 * vocab * emb)
    position_embedding = model_config.context_length * emb if model_config.position_encoding == "learned" else 0
    head_size = emb // max(model_config.head_count, 1)
    kv_emb = model_config.resolved_kv_head_count() * head_size
    attention = (emb * (emb + (2 * kv_emb))) + (emb * emb)
    if model_config.bias:
        attention += emb + (2 * kv_emb) + emb
    inter = model_config.resolved_intermediate_size()
    if model_config.mlp_type == "swiglu":
        mlp = emb * inter * 3
        if model_config.bias:
            mlp += 3 * inter
    else:
        mlp = (emb * inter) + (inter * emb)
        if model_config.bias:
            mlp += inter + emb
    norms = 4 * emb
    return {
        "token_embedding": int(token_embedding),
        "position_embedding": int(position_embedding),
        "attention": int(layers * attention),
        "mlp": int(layers * mlp),
        "norms": int(layers * norms + (2 * emb)),
    }


def estimate_training_resources(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_tokens: int,
) -> dict[str, int]:
    """Estimate model size, VRAM, steps, and storage footprint.

    Args:
        model_config: Selected model architecture.
        training_config: Selected training settings.
        train_tokens: Number of training tokens.

    Returns:
        Estimate dictionary.
    """

    parameter_breakdown = estimate_parameter_breakdown(model_config)
    params = int(sum(parameter_breakdown.values()))
    mixed_precision = training_config.use_amp and training_config.device == "cuda" and training_config.precision in {"fp16", "bf16"}
    param_bytes = params * (2 if mixed_precision else 4)
    optimizer_bytes = params * 8
    activation_bytes = (
        training_config.batch_size
        * model_config.context_length
        * model_config.embedding_size
        * model_config.layer_count
        * 8
    )
    vram_bytes = param_bytes + optimizer_bytes + activation_bytes
    kv_cache_bytes = (
        training_config.batch_size
        * model_config.context_length
        * model_config.layer_count
        * model_config.resolved_kv_head_count()
        * (model_config.embedding_size // max(model_config.head_count, 1))
        * 2
        * (2 if mixed_precision else 4)
    )
    checkpoint_bytes = params * 16
    steps_per_epoch = max(
        (train_tokens - model_config.context_length)
        // max(model_config.context_length * training_config.batch_size, 1),
        1,
    )
    total_steps = max(steps_per_epoch * training_config.epochs, 1)
    checkpoint_count = max(total_steps // max(training_config.save_interval, 1), 1) + training_config.epochs + 2
    estimated_storage = checkpoint_bytes * checkpoint_count
    return {
        "parameters": params,
        "parameter_breakdown": parameter_breakdown,
        "checkpoint_bytes": checkpoint_bytes,
        "vram_bytes": vram_bytes,
        "memory_breakdown": {
            "weights": int(param_bytes),
            "optimizer": int(optimizer_bytes),
            "activations": int(activation_bytes),
            "kv_cache": int(kv_cache_bytes),
        },
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "checkpoint_count": checkpoint_count,
        "estimated_storage": estimated_storage,
    }


def format_bytes(byte_count: float) -> str:
    """Format a byte count for compact display.

    Args:
        byte_count: Number of bytes.

    Returns:
        Human-readable storage size.
    """

    value = float(byte_count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024
    return f"{value:.1f} TB"


def optimize_training_hyperparameters(
    model_config: ModelConfig,
    target_vram_gb: float,
    train_tokens: int,
    training_mode: str = "pretrain",
    cpu_count: int | None = None,
    device_type: str = "cuda",
    available_gpu_memory_gb: float | None = None,
) -> dict[str, Any]:
    """Calculate optimal training and runtime parameters for a given model, VRAM budget, and dataset.

    Considers:
    - Model parameter footprint and architecture dimensions
    - Available/target training VRAM in GB
    - Total token count in the prepared dataset
    - Training mode: Base Pre-training vs LoRA Fine-Tuning
    - Modern scaling guidelines (Chinchilla tokens/param, global batch size, cosine warmup/decay)

    Args:
        model_config: Model architecture configuration.
        target_vram_gb: Target available VRAM in gigabytes (e.g. 16.0).
        train_tokens: Total tokens available in the prepared dataset.
        training_mode: "pretrain" or "fine_tune" / specialized fine-tuning modes.
        cpu_count: Number of CPU worker cores available.
        device_type: "cuda" or "cpu".
        available_gpu_memory_gb: Optional physically detected GPU VRAM.

    Returns:
        Dictionary of recommended training and runtime hyperparameters.
    """
    params = estimate_model_parameters(model_config)
    is_fine_tune = training_mode != "pretrain"
    target_vram = max(2.0, float(target_vram_gb))
    seq_len = max(16, model_config.context_length)
    emb = model_config.embedding_size
    layers = model_config.layer_count

    # 1. Precision & AMP
    precision = "BF16" if device_type.startswith("cuda") else "FP32"
    use_amp = device_type.startswith("cuda")

    # 2. Activation Checkpointing
    # Always enabled for 400M+ models or <= 24GB VRAM to protect against OOM
    activation_checkpointing = True

    # 3. Memory per activation sample during backward pass
    # Under PyTorch AMP + activation checkpointing, recomputing one transformer block
    # plus attention logits and PyTorch caching allocator requires ~ 5.6 * seq_len * emb * layers bytes.
    # Without checkpointing, storing intermediate forward states takes ~ 14.0 * seq_len * emb * layers bytes.
    if activation_checkpointing:
        act_bytes_per_sample = max(2 * 1024 * 1024, int(5.6 * seq_len * emb * layers))
    else:
        act_bytes_per_sample = max(5 * 1024 * 1024, int(14.0 * seq_len * emb * layers))

    # 4. Weight and optimizer memory calculation
    # In mixed precision: 2 bytes/weight.
    # Optimizer: 8-bit AdamW = 2 bytes/param; 32-bit AdamW = 8 bytes/param.
    cuda_overhead_bytes = int(1.2 * (1024 ** 3))  # PyTorch CUDA context, workspace, and kernels

    if is_fine_tune:
        # LoRA fine-tuning: base weights are frozen in 16-bit (2 bytes/param) or quantized.
        # Only adapters (~0.2% of params) have optimizer states.
        weight_bytes = params * 2
        adapter_opt_bytes = int(0.15 * (1024 ** 3))
        static_memory_bytes = weight_bytes + adapter_opt_bytes
        optimizer_name = "AdamW"
    else:
        # Base Pre-training:
        # If full 32-bit AdamW (10 bytes/param total) consumes > 55% of target VRAM, switch to 8-bit AdamW
        full_opt_bytes = params * (2 + 8)
        if full_opt_bytes > (target_vram * 0.55 * (1024 ** 3)):
            optimizer_name = "AdamW (8-bit)"
            static_memory_bytes = params * (2 + 2)
        else:
            optimizer_name = "AdamW"
            static_memory_bytes = full_opt_bytes

    # Target 88% usable VRAM headroom to maximize utilization of 16GB+ hardware without OOMing
    target_usable_bytes = int(target_vram * (1024 ** 3) * 0.88)
    avail_act_bytes = max(act_bytes_per_sample, target_usable_bytes - static_memory_bytes - cuda_overhead_bytes)

    # 5. Micro-Batch Size Selection (granular descent to fill available headroom)
    candidate_batches = [
        256, 192, 160, 128, 112, 96, 80, 64, 48, 32, 24, 16, 12, 8, 4, 2, 1
    ]
    batch_size = 1
    for cand in candidate_batches:
        if cand * act_bytes_per_sample <= avail_act_bytes:
            batch_size = cand
            break

    # 6. Global Batch Size & Gradient Accumulation
    # LLM quality relies on a healthy effective token batch size:
    # Pre-training:
    #   < 500M params: ~32K - 65K tokens
    #   500M - 3B params: ~65K - 131K tokens
    #   >= 3B params: ~131K - 262K tokens
    # Fine-tuning: ~16K - 32K tokens
    if is_fine_tune:
        target_effective_tokens = 16384 if target_vram <= 16.0 else 32768
    elif params < 500_000_000:
        target_effective_tokens = 65536
    elif params < 3_000_000_000:
        target_effective_tokens = 131072
    else:
        target_effective_tokens = 262144

    tokens_per_micro_step = max(1, batch_size * seq_len)
    raw_accum = max(1, round(target_effective_tokens / tokens_per_micro_step))
    gradient_accumulation = max(1, min(128, raw_accum))
    effective_batch_tokens = batch_size * gradient_accumulation * seq_len

    # 7. Dataset-Aware Scheduling: Steps, Epochs, Warmup, Eval, Checkpoints
    tokens_available = max(train_tokens, effective_batch_tokens)
    steps_per_epoch = max(1, tokens_available // effective_batch_tokens)

    if is_fine_tune:
        epochs = 3 if tokens_available > 5_000_000 else 4
    else:
        # Pre-training Chinchilla ratio: tokens / parameters
        tokens_per_param = float(tokens_available) / float(max(1, params))
        if tokens_per_param >= 20.0:
            epochs = 1
        elif tokens_per_param >= 5.0:
            epochs = 2
        else:
            epochs = 3

    total_steps = max(1, steps_per_epoch * epochs)

    # Warmup: ~3% of total steps, bounded between 50 and 2000
    warmup_steps = min(2000, max(50, int(0.03 * total_steps)))

    # Eval interval: 4-6 checks per epoch, bounded between 50 and 500
    eval_interval = max(50, min(500, max(1, steps_per_epoch // 4)))
    max_eval_batches = 25

    # Checkpoint save interval: 2-3 saves per epoch, bounded between 100 and 1000
    save_interval = max(100, min(1000, max(1, steps_per_epoch // 2)))

    # Sample stride: context_length // 2 to capture boundary context without excessive duplicate compute
    sample_stride = max(64, seq_len // 2)

    # CPU Dataloader Workers: half of available CPU cores, bounded between 2 and 8
    workers = min(8, max(2, (cpu_count or 8) // 2))

    # 8. Learning Rate & Weight Decay
    if is_fine_tune:
        learning_rate = 0.0002
        weight_decay = 0.05
    elif params <= 300_000_000:
        learning_rate = 0.0006
        weight_decay = 0.1
    elif params <= 1_500_000_000:
        learning_rate = 0.0004
        weight_decay = 0.1
    elif params <= 8_000_000_000:
        learning_rate = 0.00025
        weight_decay = 0.1
    else:
        learning_rate = 0.00015
        weight_decay = 0.1

    scheduler_name = "Cosine decay"
    min_lr_ratio = 0.1
    polynomial_power = 1.0
    max_grad_norm = 1.0

    # 9. LoRA hyperparameters
    lora_rank = 16 if target_vram >= 12.0 else 8
    lora_alpha = float(lora_rank * 2)
    lora_dropout = 0.05

    # 10. Estimated Peak VRAM
    est_act_bytes = batch_size * act_bytes_per_sample
    est_peak_bytes = static_memory_bytes + est_act_bytes + cuda_overhead_bytes
    est_peak_gb = round(est_peak_bytes / (1024 ** 3), 2)
    vram_headroom_gb = round(target_vram - est_peak_gb, 2)

    # Format human-readable summary
    mode_str = "LoRA Fine-Tuning" if is_fine_tune else "Pre-training"
    if tokens_available >= 1_000_000:
        tok_str = f"{tokens_available / 1_000_000:.1f}M"
    else:
        tok_str = f"{tokens_available / 1_000:.0f}K"

    summary_text = (
        f"Auto-Tuned for {target_vram:.0f}GB VRAM ({mode_str} | {tok_str} tokens): "
        f"Micro-Batch {batch_size} × Accum {gradient_accumulation} "
        f"({effective_batch_tokens:,} tokens/step) | "
        f"{epochs} ep ({total_steps:,} steps) | "
        f"VRAM est {est_peak_gb}GB / {target_vram:.0f}GB ({optimizer_name})"
    )

    return {
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_tokens": effective_batch_tokens,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "eval_interval": eval_interval,
        "max_eval_batches": max_eval_batches,
        "save_interval": save_interval,
        "sample_stride": sample_stride,
        "data_loader_workers": workers,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer_name": optimizer_name,
        "scheduler_name": scheduler_name,
        "min_lr_ratio": min_lr_ratio,
        "polynomial_power": polynomial_power,
        "max_grad_norm": max_grad_norm,
        "precision": precision,
        "use_amp": use_amp,
        "activation_checkpointing": activation_checkpointing,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "estimated_vram_gb": est_peak_gb,
        "vram_headroom_gb": vram_headroom_gb,
        "summary_text": summary_text,
    }


