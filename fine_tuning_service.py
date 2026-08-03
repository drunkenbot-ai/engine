from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .config import ModelConfig, TrainingConfig
from .training import TrainingResult
from .training_service import LocalTrainerService, TrainingJobRequest, TrainingService


ProgressCallback = Callable[[Any], None]
StopCallback = Callable[[], bool]


@dataclass
class FineTuningJobRequest:
    """Request payload for a fine-tuning job.

    Args:
        dataset_dir: Prepared instruction/conversation/domain dataset folder.
        model_config: Compatible model architecture settings.
        training_config: Fine-tuning optimizer/runtime settings.
        stage: Fine-tuning stage label: instruction, conversation, or domain.
        metadata: Optional metadata persisted with the training job.
    """

    dataset_dir: Path
    model_config: ModelConfig
    training_config: TrainingConfig
    stage: str = "domain"
    metadata: Optional[dict[str, Any]] = None

    def to_training_request(self) -> TrainingJobRequest:
        """Convert to the generic training service request.

        Returns:
            Generic training request.
        """

        self.training_config.training_mode = "fine_tune"
        metadata = dict(self.metadata or {})
        metadata["fine_tune_stage"] = self.stage
        return TrainingJobRequest(
            dataset_dir=self.dataset_dir,
            model_config=self.model_config,
            training_config=self.training_config,
            metadata=metadata,
        )


class FineTuningService:
    """Fine-tuning API boundary used by the desktop UI.

    The service delegates to the generic training service but keeps
    fine-tuning orchestration separate from GUI code.
    """

    def __init__(self, training_service: Optional[TrainingService] = None) -> None:
        """Create the fine-tuning service.

        Args:
            training_service: Optional training service implementation.
        """

        self.training_service = training_service or LocalTrainerService()

    def run(
        self,
        request: FineTuningJobRequest,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCallback] = None,
    ) -> TrainingResult:
        """Run a fine-tuning request.

        Args:
            request: Fine-tuning request.
            progress: Optional progress callback.
            should_stop: Optional cooperative cancellation callback.

        Returns:
            Training result.
        """

        return self.training_service.run(
            request.to_training_request(),
            progress=progress,
            should_stop=should_stop,
        )


def run_fine_tuning_job(
    dataset_dir: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    stage: str = "domain",
    progress: Optional[ProgressCallback] = None,
    should_stop: Optional[StopCallback] = None,
) -> TrainingResult:
    """Run a fine-tuning job through the fine-tuning service.

    Args:
        dataset_dir: Prepared fine-tuning dataset directory.
        model_config: Compatible model architecture settings.
        training_config: Fine-tuning settings.
        stage: Fine-tuning stage label.
        progress: Optional progress callback.
        should_stop: Optional cooperative cancellation callback.

    Returns:
        Training result.
    """

    service = FineTuningService()
    return service.run(
        FineTuningJobRequest(dataset_dir, model_config, training_config, stage=stage),
        progress=progress,
        should_stop=should_stop,
    )

