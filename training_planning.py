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
    token_embedding = vocab * emb
    position_embedding = model_config.context_length * emb if model_config.position_encoding == "learned" else 0
    head_size = emb // max(model_config.head_count, 1)
    kv_emb = model_config.resolved_kv_head_count() * head_size
    attention = (emb * (emb + (2 * kv_emb))) + (emb * emb)
    if model_config.bias:
        attention += emb + (2 * kv_emb) + emb
    if model_config.mlp_type == "swiglu":
        mlp = emb * 4 * emb * 3
        if model_config.bias:
            mlp += 9 * emb
    else:
        mlp = (emb * 4 * emb) + (4 * emb * emb)
        if model_config.bias:
            mlp += 5 * emb
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

