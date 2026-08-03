from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .backends import LocalTrainerBackend, TrainerBackend
from .backends.registry import DEFAULT_BACKEND_REGISTRY, BackendRegistry
from .config import ModelConfig, TrainingConfig
from .contracts import BackendKind, TrainingJobSpec
from .coordinator import JobManager
from .training import TrainingResult


ProgressCallback = Callable[[Any], None]
StopCallback = Callable[[], bool]


@dataclass
class TrainingJobRequest:
    """Request payload for a training service job.

    Args:
        dataset_dir: Prepared dataset directory.
        model_config: Model architecture settings.
        training_config: Training runtime and optimizer settings.
    """

    dataset_dir: Path
    model_config: ModelConfig
    training_config: TrainingConfig
    backend: BackendKind = BackendKind.LOCAL
    metadata: Optional[dict[str, Any]] = None

    def to_spec(self) -> TrainingJobSpec:
        """Convert this request to a backend-neutral job spec.

        Returns:
            Training job specification.
        """

        if self.backend != BackendKind.LOCAL:
            raise ValueError(f"Unsupported backend for local service: {self.backend.value}")
        return TrainingJobSpec.local(
            self.dataset_dir,
            self.model_config,
            self.training_config,
            metadata=self.metadata,
        )


class TrainingService:
    """Training service that dispatches jobs to a backend."""

    def __init__(
        self,
        backend: Optional[TrainerBackend] = None,
        registry: Optional[BackendRegistry] = None,
    ) -> None:
        """Create a training service.

        Args:
            backend: Backend used to execute jobs.
            registry: Backend registry used when backend is not provided.
        """

        self.backend = backend
        self.registry = registry or DEFAULT_BACKEND_REGISTRY

    def run(
        self,
        request: TrainingJobRequest,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCallback] = None,
    ) -> TrainingResult:
        """Run a training request through the configured backend.

        Args:
            request: Training job request.
            progress: Optional progress callback.
            should_stop: Optional cooperative cancellation callback.

        Returns:
            Training result.
        """

        job = request.to_spec()
        registry = self.registry
        if self.backend is not None:
            registry = BackendRegistry()
            registry.register(job.runtime.backend, self.backend)
        manager = JobManager(registry=registry)
        manager.submit(job)
        return manager.run_job(job.job_id, progress=progress, should_stop=should_stop)


class LocalTrainerService(TrainingService):
    """Local trainer service used by the desktop coordinator.

    This service is intentionally small: the GUI depends on this API boundary
    instead of calling the trainer directly, which leaves room for a future
    cloud or multi-machine implementation behind the same contract.
    """

    def __init__(self) -> None:
        """Create a local trainer service."""

        super().__init__(LocalTrainerBackend())


def run_training_job(
    dataset_dir: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    progress: Optional[ProgressCallback] = None,
    should_stop: Optional[StopCallback] = None,
) -> TrainingResult:
    """Run a training job through the configured training service.

    Args:
        dataset_dir: Prepared dataset directory.
        model_config: Model architecture settings.
        training_config: Training runtime and optimizer settings.
        progress: Optional progress callback.
        should_stop: Optional cooperative cancellation callback.

    Returns:
        Training result from the active trainer service.
    """

    service = TrainingService(LocalTrainerBackend())
    return service.run(
        TrainingJobRequest(dataset_dir, model_config, training_config),
        progress=progress,
        should_stop=should_stop,
    )

