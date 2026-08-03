from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from engine.config import ModelConfig, TrainingConfig, dataclass_to_jsonable


class BackendKind(str, Enum):
    """Training backend type."""

    LOCAL = "local"
    REMOTE_CLIENT = "remote_client"
    HUGGINGFACE = "huggingface"
    CLOUD = "cloud"


class JobStatus(str, Enum):
    """Training job lifecycle state."""

    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Relative scheduler priority."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


def utc_now_iso() -> str:
    """Return the current UTC timestamp.

    Returns:
        ISO formatted UTC timestamp.
    """

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DatasetSpec:
    """Dataset artifacts required for a training job.

    Attributes:
        dataset_dir: Prepared dataset directory.
        tokenizer_path: Tokenizer path, usually inside ``dataset_dir``.
        train_tokens_path: Training token file.
        val_tokens_path: Validation token file.
        summary_path: Dataset summary path.
        dataset_id: Optional dataset identifier from lineage.
        dataset_version: Optional dataset version identifier.
    """

    dataset_dir: Path
    tokenizer_path: Optional[Path] = None
    train_tokens_path: Optional[Path] = None
    val_tokens_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    dataset_id: Optional[str] = None
    dataset_version: Optional[str] = None

    @classmethod
    def from_dataset_dir(cls, dataset_dir: Path) -> "DatasetSpec":
        """Create a dataset spec from a prepared dataset directory.

        Args:
            dataset_dir: Prepared dataset directory.

        Returns:
            Dataset specification.
        """

        dataset_dir = Path(dataset_dir)
        train_tokens_path = dataset_dir / "train_tokens.npy"
        val_tokens_path = dataset_dir / "val_tokens.npy"
        if not train_tokens_path.exists() or not val_tokens_path.exists():
            train_tokens_path = dataset_dir / "train_tokens.json"
            val_tokens_path = dataset_dir / "val_tokens.json"
        return cls(
            dataset_dir=dataset_dir,
            tokenizer_path=dataset_dir / "tokenizer.json",
            train_tokens_path=train_tokens_path,
            val_tokens_path=val_tokens_path,
            summary_path=dataset_dir / "dataset_summary.json",
        )


@dataclass
class ModelSpec:
    """Model architecture payload for a training job.

    Attributes:
        config: Model configuration.
        base_checkpoint: Optional base checkpoint used for fine-tuning.
    """

    config: ModelConfig
    base_checkpoint: Optional[Path] = None


@dataclass
class RuntimeSpec:
    """Runtime and scheduling requirements for a training job.

    Attributes:
        backend: Backend kind selected for this job.
        device: Requested training device.
        min_vram_gb: Optional minimum VRAM requirement.
        preferred_worker_id: Optional worker ID to target.
        priority: Scheduler priority.
        tags: Free-form job tags.
    """

    backend: BackendKind = BackendKind.LOCAL
    device: str = "auto"
    min_vram_gb: Optional[float] = None
    preferred_worker_id: Optional[str] = None
    priority: JobPriority = JobPriority.NORMAL
    tags: list[str] = field(default_factory=list)


@dataclass
class ArtifactSpec:
    """Artifact destinations for a training job.

    Attributes:
        output_dir: Model output directory.
        checkpoints_dir: Checkpoint directory.
        final_checkpoint: Final checkpoint path.
        summary_path: Training summary path.
        telemetry_db: Optional telemetry database path.
    """

    output_dir: Path
    checkpoints_dir: Optional[Path] = None
    final_checkpoint: Optional[Path] = None
    summary_path: Optional[Path] = None
    telemetry_db: Optional[Path] = None

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "ArtifactSpec":
        """Create artifact paths from a model output directory.

        Args:
            output_dir: Model output directory.

        Returns:
            Artifact specification.
        """

        output_dir = Path(output_dir)
        return cls(
            output_dir=output_dir,
            checkpoints_dir=output_dir / "checkpoints",
            final_checkpoint=output_dir / "final_model.pt",
            summary_path=output_dir / "training_summary.json",
        )


@dataclass
class TrainingMetrics:
    """Serializable training metrics emitted by trainers.

    Attributes:
        step: Current optimizer step.
        total_steps: Planned optimizer steps.
        epoch: Current epoch.
        total_epochs: Planned epochs.
        train_loss: Latest training loss.
        val_loss: Latest validation loss.
        learning_rate: Current learning rate.
        tokens_per_second: Token throughput.
        samples_per_second: Sample throughput.
        gpu_memory_percent: GPU memory usage percentage.
        system_ram_percent: System RAM usage percentage.
        message: Optional status message.
    """

    step: Optional[int] = None
    total_steps: Optional[int] = None
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    tokens_per_second: Optional[float] = None
    samples_per_second: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    system_ram_percent: Optional[float] = None
    message: Optional[str] = None


@dataclass
class TrainingResultSpec:
    """Serializable result returned by a training backend.

    Attributes:
        job_id: Job identifier.
        status: Final job status.
        checkpoint_path: Final or stopped checkpoint path.
        summary_path: Training summary JSON path.
        final_train_loss: Final training loss.
        final_val_loss: Final validation loss.
        stopped: Whether the job stopped by request.
        error: Optional error text.
        artifact_bundle_url: Optional coordinator URL for downloaded worker outputs.
    """

    job_id: str
    status: JobStatus
    checkpoint_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    stopped: bool = False
    error: Optional[str] = None
    artifact_bundle_url: Optional[str] = None


@dataclass
class TrainingJobSpec:
    """Complete backend-neutral training job contract.

    Attributes:
        job_id: Stable job identifier.
        created_at: UTC creation timestamp.
        dataset: Dataset artifact specification.
        model: Model architecture specification.
        training: Training configuration.
        runtime: Runtime/scheduler specification.
        artifacts: Output artifact specification.
        status: Current job status.
        metadata: Free-form metadata for UI, manager, or cloud adapters.
    """

    dataset: DatasetSpec
    model: ModelSpec
    training: TrainingConfig
    artifacts: ArtifactSpec
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    job_id: str = field(default_factory=lambda: f"job_{uuid4().hex}")
    created_at: str = field(default_factory=utc_now_iso)
    status: JobStatus = JobStatus.QUEUED
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def local(
        cls,
        dataset_dir: Path,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "TrainingJobSpec":
        """Create a local training job spec.

        Args:
            dataset_dir: Prepared dataset directory.
            model_config: Model architecture configuration.
            training_config: Training configuration.
            metadata: Optional metadata.

        Returns:
            Training job specification.
        """

        base_checkpoint = training_config.fine_tune_from_checkpoint
        return cls(
            dataset=DatasetSpec.from_dataset_dir(dataset_dir),
            model=ModelSpec(model_config, base_checkpoint=base_checkpoint),
            training=training_config,
            artifacts=ArtifactSpec.from_output_dir(training_config.output_dir),
            runtime=RuntimeSpec(backend=BackendKind.LOCAL, device=training_config.device),
            metadata=metadata or {},
        )

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the job spec to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "status": self.status.value,
            "dataset": _paths_to_strings(self.dataset.__dict__),
            "model": {
                "config": dataclass_to_jsonable(self.model.config),
                "base_checkpoint": str(self.model.base_checkpoint) if self.model.base_checkpoint else None,
            },
            "training": dataclass_to_jsonable(self.training),
            "runtime": _enum_values(_paths_to_strings(self.runtime.__dict__)),
            "artifacts": _paths_to_strings(self.artifacts.__dict__),
            "metadata": self.metadata,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "TrainingJobSpec":
        """Create a training job spec from JSON-friendly values.

        Args:
            data: Serialized training job data.

        Returns:
            Training job specification.
        """

        model_data = dict(data["model"]["config"])
        training_data = dict(data["training"])
        dataset_data = _strings_to_paths(data["dataset"], _DATASET_PATH_FIELDS)
        artifact_data = _strings_to_paths(data["artifacts"], _ARTIFACT_PATH_FIELDS)
        runtime_data = dict(data["runtime"])
        model_payload = dict(data["model"])
        for key in _TRAINING_PATH_FIELDS:
            if training_data.get(key):
                training_data[key] = Path(training_data[key])
        base_checkpoint = model_payload.get("base_checkpoint")
        return cls(
            dataset=DatasetSpec(**dataset_data),
            model=ModelSpec(
                config=ModelConfig(**model_data),
                base_checkpoint=Path(base_checkpoint) if base_checkpoint else None,
            ),
            training=TrainingConfig(**training_data),
            artifacts=ArtifactSpec(**artifact_data),
            runtime=RuntimeSpec(
                backend=BackendKind(runtime_data.get("backend", BackendKind.LOCAL.value)),
                device=runtime_data.get("device", "auto"),
                min_vram_gb=runtime_data.get("min_vram_gb"),
                preferred_worker_id=runtime_data.get("preferred_worker_id"),
                priority=JobPriority(runtime_data.get("priority", JobPriority.NORMAL.value)),
                tags=list(runtime_data.get("tags") or []),
            ),
            job_id=data["job_id"],
            created_at=data["created_at"],
            status=JobStatus(data.get("status", JobStatus.QUEUED.value)),
            metadata=dict(data.get("metadata") or {}),
        )


def _paths_to_strings(data: dict[str, Any]) -> dict[str, Any]:
    """Convert path values in a dictionary to strings.

    Args:
        data: Dictionary to convert.

    Returns:
        Converted dictionary.
    """

    output: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def _enum_values(data: dict[str, Any]) -> dict[str, Any]:
    """Convert enum values in a dictionary to their raw values.

    Args:
        data: Dictionary to convert.

    Returns:
        Converted dictionary.
    """

    output: dict[str, Any] = {}
    for key, value in data.items():
        output[key] = value.value if isinstance(value, Enum) else value
    return output


_DATASET_PATH_FIELDS = {"dataset_dir", "tokenizer_path", "train_tokens_path", "val_tokens_path", "summary_path"}
_ARTIFACT_PATH_FIELDS = {"output_dir", "checkpoints_dir", "final_checkpoint", "summary_path", "telemetry_db"}
_TRAINING_PATH_FIELDS = {"output_dir", "fine_tune_from_checkpoint", "resume_from_checkpoint"}


def _strings_to_paths(data: dict[str, Any], path_fields: set[str]) -> dict[str, Any]:
    """Convert selected string fields in a dictionary to paths.

    Args:
        data: Dictionary to convert.
        path_fields: Keys that should become paths.

    Returns:
        Converted dictionary.
    """

    output: dict[str, Any] = {}
    for key, value in data.items():
        if key in path_fields and value:
            output[key] = Path(value)
        else:
            output[key] = value
    return output

