from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

from engine.contracts import TrainingJobSpec
from engine.training import TrainingResult


ProgressCallback = Callable[[Any], None]
StopCallback = Callable[[], bool]


class TrainerBackend(Protocol):
    """Protocol implemented by training backends."""

    name: str

    def run(
        self,
        job: TrainingJobSpec,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCallback] = None,
    ) -> TrainingResult:
        """Run a training job.

        Args:
            job: Backend-neutral training job spec.
            progress: Optional progress callback.
            should_stop: Optional cooperative cancellation callback.

        Returns:
            Training result.
        """

