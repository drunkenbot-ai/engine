from __future__ import annotations

import json
from typing import Optional

from engine.backends.base import ProgressCallback, StopCallback
from engine.contracts import JobStatus, TrainingJobSpec
from engine.training_orchestrator import train_from_dataset
from engine.training import TrainingResult


class LocalTrainerBackend:
    """Training backend that runs jobs in the current Python process."""

    name = "local"

    def run(
        self,
        job: TrainingJobSpec,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCallback] = None,
    ) -> TrainingResult:
        """Run a local training job.

        Args:
            job: Backend-neutral training job specification.
            progress: Optional progress callback.
            should_stop: Optional cooperative cancellation callback.

        Returns:
            Training result.
        """

        job.status = JobStatus.RUNNING
        try:
            job.artifacts.output_dir.mkdir(parents=True, exist_ok=True)
            job_manifest_path = job.artifacts.output_dir / "training_job.json"
            job_manifest_path.write_text(json.dumps(job.to_jsonable(), indent=2), encoding="utf-8")
            if progress:
                progress(
                    {
                        "message": f"Local backend started job {job.job_id}",
                        "job_id": job.job_id,
                        "backend": self.name,
                    }
                )
            result = train_from_dataset(
                job.dataset.dataset_dir,
                job.model.config,
                job.training,
                progress=progress,
                should_stop=should_stop,
            )
            job.status = JobStatus.CANCELLED if result.stopped else JobStatus.COMPLETED
            return result
        except Exception:
            job.status = JobStatus.FAILED
            raise

