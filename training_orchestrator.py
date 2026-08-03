from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .config import ModelConfig, TrainingConfig
from .data import file_sha256
from .lineage import read_json, stable_json_hash, utc_timestamp, write_json
from .resume_checks import _validate_resume_compatibility
from .tokenizer import PAD_TOKEN, load_tokenizer, token_id, validate_training_tokenizer
from .training import TrainingResult, train_model


def _load_tokens_for_training(data_dir: Path) -> tuple[Any, Any]:
    train_npy = data_dir / "train_tokens.npy"
    val_npy = data_dir / "val_tokens.npy"
    if train_npy.exists() and val_npy.exists():
        train_tokens = np.load(train_npy, mmap_mode="r", allow_pickle=False)
        val_tokens = np.load(val_npy, mmap_mode="r", allow_pickle=False)
        return train_tokens, val_tokens
    train_json = data_dir / "train_tokens.json"
    val_json = data_dir / "val_tokens.json"
    if train_json.exists() and val_json.exists():
        train_tokens = json.loads(train_json.read_text(encoding="utf-8"))
        val_tokens = json.loads(val_json.read_text(encoding="utf-8"))
        return train_tokens, val_tokens
    raise FileNotFoundError("Prepared dataset is missing token files (expected .npy or .json train/val tokens).")


def train_from_dataset(
    data_dir: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> TrainingResult:
    """Train a model using a prepared dataset folder.

    Args:
        data_dir: Prepared dataset folder.
        model_config: Model architecture settings.
        training_config: Optimizer and checkpoint settings.
        progress: Optional callback receiving progress event dictionaries.
        should_stop: Optional callback returning true when the user requested stop.

    Returns:
        Training result with final checkpoint and summary paths.

    Raises:
        FileNotFoundError: If the tokenizer is missing.
    """

    tokenizer_path = data_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    dataset_summary = read_json(data_dir / "dataset_summary.json", default={}) or {}
    dataset_lineage = read_json(data_dir / "dataset_lineage.json", default={}) or {}
    tokenizer = load_tokenizer(tokenizer_path)
    validate_training_tokenizer(tokenizer)
    train_tokens, val_tokens = _load_tokens_for_training(data_dir)

    if model_config.vocab_size != tokenizer.get_vocab_size():
        model_config.vocab_size = tokenizer.get_vocab_size()

    training_config.output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = _validate_resume_compatibility(data_dir, tokenizer_path, model_config, training_config)
    if resume_path and progress:
        progress({"message": f"Resume safety check passed: {resume_path}", "percent": 3})
    shutil.copy2(tokenizer_path, training_config.output_dir / "tokenizer.json")
    for metadata_name in ("tokenizer_config.json", "special_tokens_map.json"):
        metadata_path = data_dir / metadata_name
        if metadata_path.exists():
            shutil.copy2(metadata_path, training_config.output_dir / metadata_name)
    result = train_model(
        model_config,
        training_config,
        train_tokens,
        val_tokens,
        pad_token_id=token_id(tokenizer, PAD_TOKEN),
        progress=progress,
        should_stop=should_stop,
        decode_preview=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
    )
    training_summary = read_json(result.summary_path, default={}) or {}
    run_id = (
        f"run_{utc_timestamp()}_"
        f"{stable_json_hash({'dataset': dataset_summary.get('dataset_version'), 'model': training_summary.get('model_config'), 'training': training_summary.get('training_config')})}"
    )
    lineage = {
        "schema": "micro_llm_model_lineage",
        "version": 1,
        "training_run_id": run_id,
        "created_at": utc_timestamp(),
        "dataset_dir": str(data_dir),
        "dataset_id": dataset_summary.get("dataset_id") or dataset_lineage.get("dataset_id"),
        "dataset_version": dataset_summary.get("dataset_version"),
        "dataset_fingerprint": (dataset_summary.get("dataset_version") or {}).get("source_fingerprint"),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "training_mode": training_config.training_mode,
        "fine_tune_from_checkpoint": (
            str(training_config.fine_tune_from_checkpoint)
            if training_config.fine_tune_from_checkpoint
            else None
        ),
        "peft_method": training_config.peft_method,
        "lora_rank": training_config.lora_rank if training_config.peft_method == "lora" else None,
        "lora_alpha": training_config.lora_alpha if training_config.peft_method == "lora" else None,
        "lora_target_modules": training_config.lora_target_modules if training_config.peft_method == "lora" else None,
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "resume_safety_required": training_config.require_compatible_resume,
        "checkpoint_path": str(result.checkpoint_path),
        "summary_path": str(result.summary_path),
        "stopped": result.stopped,
    }
    training_summary["training_run_id"] = run_id
    training_summary["model_lineage"] = lineage
    write_json(result.summary_path, training_summary)
    write_json(training_config.output_dir / "model_lineage.json", lineage)
    write_json(training_config.output_dir / "dataset_summary.json", dataset_summary)
    return result

__all__ = ["train_from_dataset"]

