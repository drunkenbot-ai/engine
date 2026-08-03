from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class DatasetConfig:
    """Configuration for building a tokenizer-ready dataset.

    Attributes:
        input_dir: Folder containing source PDFs, text, JSONL, or code files.
        output_dir: Folder where prepared dataset artifacts are written.
        vocab_size: Optional manual tokenizer vocabulary size.
        min_frequency: Minimum token frequency for BPE vocabulary entries.
        context_length: Token window length used by downstream training.
        validation_split: Fraction of tokens reserved for validation.
        lowercase: Whether to lowercase text during ingestion.
        max_workers: Number of parallel file readers.
        code_training_mode: Enables code/prose tagging and code preservation.
        include_prose: Keeps prose/explanation samples when code mode is active.
        include_source_code: Includes source-code files when code mode is active.
        extract_code_blocks: Detects code-like blocks in PDFs/text.
        preserve_indentation: Keeps code line breaks and indentation.
        generate_instruction_samples: Wraps code with simple instruction tags.
        reasoning_sample_mode: Instruction/reasoning format: none, scaffold, or detailed.
        prepare_mode: Dataset update mode: incremental, full_rebuild, or force_reprocess.
        tokenizer_strategy: Tokenizer policy: auto, train_new, reuse_dataset, or import_tokenizer.
        tokenizer_path: Optional existing tokenizer JSON used by import_tokenizer.
        dataset_stage: Intended dataset purpose: base, instruction, conversation, or code.
        conversation_datasets: Built-in Hugging Face conversation dataset IDs to include.
        conversation_sample_limit: Maximum rows to read from each selected conversation dataset. Zero means no limit.
        conversation_dataset_path: Optional local JSON/JSONL file or folder containing conversation samples.
        instruction_dataset_path: Optional local JSON/JSONL file or folder containing instruction samples.
        conversation_dataset_paths: Local JSON/JSONL files or folders containing conversation samples.
        instruction_dataset_paths: Local JSON/JSONL files or folders containing instruction samples.
        default_data_paths: Bundled starter data files selected from the Dataset Blueprint panel.
        mixture_weights: Planned dataset mixture percentages by source family.
        fast_scan_mode: Uses sampled fingerprints for faster large-corpus scans.
        fast_scan_sample_bytes: Head/tail bytes per file used for fast fingerprints.
        strict_duplicate_verification: In fast mode, fully re-hashes only suspected duplicate groups.
        tokenizer_training_max_gb: Maximum corpus size, in GiB, shown to the BPE
            tokenizer trainer (which holds a frequency table sized to whatever it
            is shown, in memory, for the whole training run). Sampling is applied
            only above this size; the full corpus is still encoded into training
            tokens regardless of this setting. 0 or a negative value disables the
            cap and trains on the entire corpus -- only safe if you have enough
            RAM to hold a frequency table for your full corpus size at once.
    """

    input_dir: Path
    output_dir: Path
    vocab_size: Optional[int] = None
    min_frequency: int = 2
    context_length: int = 512
    validation_split: float = 0.1
    lowercase: bool = False
    max_workers: int = 4
    code_training_mode: bool = False
    include_prose: bool = True
    include_source_code: bool = True
    extract_code_blocks: bool = True
    preserve_indentation: bool = True
    generate_instruction_samples: bool = True
    reasoning_sample_mode: str = "scaffold"
    prepare_mode: str = "incremental"
    tokenizer_strategy: str = "auto"
    tokenizer_path: Optional[Path] = None
    dataset_stage: str = "base"
    conversation_datasets: list[str] = field(default_factory=list)
    conversation_sample_limit: int = 20000
    conversation_dataset_path: Optional[Path] = None
    instruction_dataset_path: Optional[Path] = None
    conversation_dataset_paths: list[Path] = field(default_factory=list)
    instruction_dataset_paths: list[Path] = field(default_factory=list)
    default_data_paths: list[Path] = field(default_factory=list)
    mixture_weights: dict[str, float] = field(default_factory=dict)
    fast_scan_mode: bool = False
    fast_scan_sample_bytes: int = 64 * 1024
    strict_duplicate_verification: bool = False
    tokenizer_training_max_gb: float = 2.0


@dataclass
class ModelConfig:
    """Configuration for the GPT-style model architecture.

    Attributes:
        vocab_size: Tokenizer vocabulary size.
        context_length: Maximum tokens visible to the model at once.
        embedding_size: Width of token embeddings and transformer channels.
        head_count: Number of causal attention heads.
        layer_count: Number of transformer blocks.
        dropout: Dropout probability for regularization.
        bias: Whether linear and normalization layers include bias terms.
        norm_type: Normalization type: layernorm or rmsnorm.
        position_encoding: Position encoding type: learned or rope.
        mlp_type: Feed-forward type: gelu or swiglu.
        rope_theta: RoPE frequency base when position_encoding is rope.
        attention_type: Attention layout: mha, gqa, or mqa.
        kv_head_count: Key/value head count for grouped-query attention.
        attention_backend: Attention kernel backend: manual or sdpa.
        attention_window: Sliding-window attention size. Zero means full context.
    """

    vocab_size: int
    context_length: int = 512
    embedding_size: int = 256
    head_count: int = 4
    layer_count: int = 6
    dropout: float = 0.1
    bias: bool = True
    norm_type: str = "layernorm"
    position_encoding: str = "learned"
    mlp_type: str = "gelu"
    rope_theta: float = 10000.0
    attention_type: str = "mha"
    kv_head_count: int = 0
    attention_backend: str = "sdpa"
    attention_window: int = 0

    def validate(self) -> None:
        """Validate architecture constraints.

        Raises:
            ValueError: If dimensions are incompatible or too small.
        """

        if self.embedding_size % self.head_count != 0:
            raise ValueError("embedding_size must be divisible by head_count")
        if self.context_length < 8:
            raise ValueError("context_length must be at least 8")
        if self.vocab_size < 16:
            raise ValueError("vocab_size is too small for language modeling")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm_type must be layernorm or rmsnorm")
        if self.position_encoding not in {"learned", "rope"}:
            raise ValueError("position_encoding must be learned or rope")
        if self.mlp_type not in {"gelu", "swiglu"}:
            raise ValueError("mlp_type must be gelu or swiglu")
        if self.position_encoding == "rope":
            head_size = self.embedding_size // self.head_count
            if head_size % 2 != 0:
                raise ValueError("RoPE requires an even attention head size")
        if self.attention_type not in {"mha", "gqa", "mqa"}:
            raise ValueError("attention_type must be mha, gqa, or mqa")
        if self.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be manual or sdpa")
        kv_heads = self.resolved_kv_head_count()
        if kv_heads < 1 or kv_heads > self.head_count:
            raise ValueError("kv_head_count must be between 1 and head_count")
        if self.head_count % kv_heads != 0:
            raise ValueError("head_count must be divisible by kv_head_count")
        if self.attention_window < 0:
            raise ValueError("attention_window cannot be negative")

    def resolved_kv_head_count(self) -> int:
        """Return the effective key/value head count.

        Returns:
            Key/value head count after applying the attention type.
        """

        if self.attention_type == "mqa":
            return 1
        if self.attention_type == "gqa":
            return self.kv_head_count if self.kv_head_count > 0 else max(1, self.head_count // 2)
        return self.head_count


@dataclass
class TrainingConfig:
    """Configuration for model optimization and checkpointing.

    Attributes:
        output_dir: Folder where checkpoints and summaries are saved.
        epochs: Number of full passes over the training dataset.
        batch_size: Number of token windows per training batch.
        learning_rate: Base optimizer learning rate.
        weight_decay: Optimizer weight decay regularization.
        optimizer_name: Optimizer family: adamw, adam, lion, or adafactor.
        scheduler_name: Learning-rate schedule: warmup_linear, cosine, polynomial, one_cycle, or constant.
        scheduler_min_lr_ratio: Minimum learning-rate multiplier after decay.
        polynomial_power: Power used by polynomial decay.
        gradient_accumulation: Batches to accumulate before optimizer step.
        sample_stride: Token offset step between consecutive training windows.
        warmup_steps: Steps used to ramp up learning rate.
        eval_interval: Steps between validation loss checks.
        max_eval_batches: Maximum validation batches per interval evaluation. Zero evaluates all validation batches.
        save_interval: Steps between checkpoint writes.
        data_loader_workers: CPU worker processes used to prepare token batches.
        max_grad_norm: Gradient clipping norm.
        use_amp: Enables mixed precision on CUDA.
        precision: Numeric precision policy: fp32, fp16, or bf16.
        device: Training device, usually "cuda" or "cpu".
        seed: Random seed for repeatability.
        training_mode: Training mode: pretrain or fine_tune.
        fine_tune_from_checkpoint: Optional checkpoint used as the base model for fine-tuning.
        peft_method: Parameter-efficient fine-tuning method: none or lora.
        lora_rank: LoRA adapter rank.
        lora_alpha: LoRA scaling alpha.
        lora_dropout: Dropout applied before LoRA adapters.
        lora_target_modules: Comma-separated LoRA target groups.
        resume: Whether to resume from checkpoints.
        resume_from_checkpoint: Optional exact checkpoint path to resume from.
        require_compatible_resume: Validate tokenizer/model compatibility before resuming.
        early_stopping: Stop training when validation loss stops improving.
        early_stopping_patience: Consecutive evaluations without improvement before stopping.
    """

    output_dir: Path
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    optimizer_name: str = "adamw"
    scheduler_name: str = "warmup_linear"
    scheduler_min_lr_ratio: float = 0.1
    polynomial_power: float = 1.0
    gradient_accumulation: int = 1
    sample_stride: int = 128 #1
    warmup_steps: int = 100
    eval_interval: int = 100
    max_eval_batches: int = 50
    save_interval: int = 500
    data_loader_workers: int = 0
    max_grad_norm: float = 1.0
    activation_checkpointing: bool = False
    use_amp: bool = True
    precision: str = "fp16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    training_mode: str = "pretrain"
    fine_tune_from_checkpoint: Optional[Path] = None
    peft_method: str = "none"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: str = "attention"
    resume: bool = True
    resume_from_checkpoint: Optional[Path] = None
    require_compatible_resume: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 3

    def validate(self) -> None:
        """Validate optimizer and schedule settings.

        Raises:
            ValueError: If any optimization setting is unsupported.
        """

        if self.optimizer_name not in {"adamw", "adam", "lion", "adafactor"}:
            raise ValueError("optimizer_name must be adamw, adam, lion, or adafactor")
        if self.scheduler_name not in {"warmup_linear", "cosine", "polynomial", "one_cycle", "constant"}:
            raise ValueError("scheduler_name must be warmup_linear, cosine, polynomial, one_cycle, or constant")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.training_mode not in {"pretrain", "fine_tune"}:
            raise ValueError("training_mode must be pretrain or fine_tune")
        if self.training_mode == "fine_tune" and self.fine_tune_from_checkpoint is None:
            raise ValueError("fine_tune_from_checkpoint is required for fine_tune mode")
        if self.peft_method not in {"none", "lora"}:
            raise ValueError("peft_method must be none or lora")
        if self.peft_method == "lora":
            if self.training_mode != "fine_tune":
                raise ValueError("LoRA requires fine_tune training mode")
            if self.lora_rank <= 0:
                raise ValueError("lora_rank must be greater than 0")
            if self.lora_alpha <= 0.0:
                raise ValueError("lora_alpha must be greater than 0")
            if self.lora_dropout < 0.0 or self.lora_dropout > 0.9:
                raise ValueError("lora_dropout must be between 0 and 0.9")
        if self.scheduler_min_lr_ratio < 0.0 or self.scheduler_min_lr_ratio > 1.0:
            raise ValueError("scheduler_min_lr_ratio must be between 0 and 1")
        if self.polynomial_power <= 0.0:
            raise ValueError("polynomial_power must be greater than 0")
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be greater than 0")


def dataclass_to_jsonable(value: Any) -> dict[str, Any]:
    """Convert a dataclass into JSON-friendly values.

    Args:
        value: Dataclass instance to convert.

    Returns:
        Dictionary safe to pass to ``json.dumps``.
    """

    def convert(item: Any) -> Any:
        """Convert nested values into JSON-friendly values."""

        if isinstance(item, Path):
            return str(item)
        if isinstance(item, list):
            return [convert(child) for child in item]
        if isinstance(item, tuple):
            return [convert(child) for child in item]
        if isinstance(item, dict):
            return {str(key): convert(child) for key, child in item.items()}
        return item

    return convert(asdict(value))
