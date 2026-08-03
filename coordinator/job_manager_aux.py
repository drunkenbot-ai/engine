from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from engine.backends.base import ProgressCallback, StopCallback, TrainerBackend
from engine.backends.registry import DEFAULT_BACKEND_REGISTRY, BackendRegistry
from engine.contracts import (
    BackendKind,
    ClaimJobRequest,
    ClaimJobResponse,
    CompleteJobRequest,
    CompleteJobResponse,
    FailJobRequest,
    FailJobResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    JobStatus,
    ProgressReportRequest,
    ProgressReportResponse,
    ProtocolStatus,
    RegisterWorkerRequest,
    RegisterWorkerResponse,
    TrainingMetrics,
    TrainingJobSpec,
    TrainingResultSpec,
    WorkerAvailability,
    utc_now_iso,
)
from engine.coordinator.state_store import JobStateStore
from engine.coordinator.job_models import WorkerDescriptor, WorkerStatus
from engine.training import TrainingResult


from .job_manager_core import JobManagerCore
class JobManagerAux:
    def run_job(
        self,
        job_id: str,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCallback] = None,
    ) -> TrainingResult:
        """Run one job through an available backend.

        Args:
            job_id: Job identifier.
            progress: Optional progress callback.
            should_stop: Optional external stop callback.

        Returns:
            Training result.
        """

        managed = self.get_job(job_id)
        job = managed.spec
        worker = self._select_worker(job.runtime.backend)
        managed.assigned_worker_id = worker.worker_id
        worker.status = WorkerStatus.BUSY
        job.status = JobStatus.ASSIGNED
        self._persist_worker(worker)
        self._persist_job(job_id)
        backend = self.registry.get(job.runtime.backend)
        try:
            result = backend.run(
                job,
                progress=progress,
                should_stop=lambda: self._should_stop(job_id, should_stop),
            )
            managed.result = self._result_spec(job, result)
            job.status = managed.result.status
            self._persist_job(job_id)
            return result
        except Exception as exc:
            job.status = JobStatus.FAILED
            managed.error = str(exc)
            self._persist_job(job_id)
            raise
        finally:
            worker.status = WorkerStatus.AVAILABLE
            worker.last_heartbeat_at = utc_now_iso()
            self._persist_worker(worker)
            self._stop_requested.discard(job_id)

    def _select_worker(self, backend: BackendKind) -> WorkerDescriptor:
        """Select an available worker for a backend.

        Args:
            backend: Backend kind required by the job.

        Returns:
            Available worker.

        Raises:
            ValueError: If no worker is available.
        """

        for worker in self._workers.values():
            if worker.backend == backend and worker.status == WorkerStatus.AVAILABLE:
                return worker
        raise ValueError(f"No available worker for backend {backend.value}")

    def _worker_can_claim_job(self, worker: WorkerDescriptor, job: TrainingJobSpec) -> bool:
        """Return whether a remote worker can claim a job.

        Args:
            worker: Worker descriptor.
            job: Training job contract.

        Returns:
            Whether the worker can claim the job.
        """

        if job.status != JobStatus.QUEUED:
            return False
        if job.runtime.backend != worker.backend:
            return False
        if job.runtime.preferred_worker_id and job.runtime.preferred_worker_id != worker.worker_id:
            return False
        if job.runtime.min_vram_gb is not None:
            worker_vram = _worker_total_vram_gb(worker)
            if worker_vram is None or worker_vram < job.runtime.min_vram_gb:
                return False
        if job.runtime.tags:
            worker_labels = set(worker.capabilities.get("labels") or [])
            if not set(job.runtime.tags).issubset(worker_labels):
                return False
        return True

    def _should_stop(self, job_id: str, external_stop: Optional[StopCallback]) -> bool:
        """Return whether a job should stop.

        Args:
            job_id: Job identifier.
            external_stop: Optional external stop callback.

        Returns:
            Whether the job should stop.
        """

        return job_id in self._stop_requested or bool(external_stop and external_stop())

    def _should_stop_remote_job(self, job_id: str) -> bool:
        """Return whether a remote worker should stop a job.

        Args:
            job_id: Job identifier.

        Returns:
            Whether the worker should stop the job.
        """

        if job_id in self._stop_requested:
            return True
        managed = self._jobs.get(job_id)
        return bool(managed and managed.spec.status in {JobStatus.STOPPING, JobStatus.CANCELLED, JobStatus.FAILED})

    def _should_pause_remote_job(self, job_id: str) -> bool:
        """Return whether a remote worker should pause a job.

        Args:
            job_id: Job identifier.

        Returns:
            Whether the worker should pause the job.
        """

        managed = self._jobs.get(job_id)
        return bool(managed and managed.spec.status == JobStatus.PAUSED)

    def _result_spec(self, job: TrainingJobSpec, result: TrainingResult) -> TrainingResultSpec:
        """Create a serializable result contract from a training result.

        Args:
            job: Training job contract.
            result: Concrete training result.

        Returns:
            Serializable result specification.
        """

        return TrainingResultSpec(
            job_id=job.job_id,
            status=JobStatus.CANCELLED if result.stopped else JobStatus.COMPLETED,
            checkpoint_path=result.checkpoint_path,
            summary_path=result.summary_path,
            final_train_loss=result.final_train_loss,
            final_val_loss=result.final_val_loss,
            stopped=result.stopped,
        )

    def _restore_state(self) -> None:
        """Restore persisted jobs and workers from the state store."""
        from engine.coordinator.job_manager_impl import _managed_job_from_jsonable

        for worker_data in self.state_store.load_workers():
            worker = WorkerDescriptor.from_jsonable(worker_data)
            if worker.status == WorkerStatus.BUSY:
                worker.status = WorkerStatus.OFFLINE
            self._workers[worker.worker_id] = worker
        for job_data in self.state_store.load_jobs():
            managed = _managed_job_from_jsonable(job_data)
            if managed.spec.status in {JobStatus.ASSIGNED, JobStatus.RUNNING, JobStatus.STOPPING}:
                managed.spec.status = JobStatus.QUEUED
                managed.assigned_worker_id = None
                managed.error = "Recovered after app restart before job completion."
            self._jobs[managed.spec.job_id] = managed
            self._persist_job(managed.spec.job_id)

    def _persist_job(self, job_id: str) -> None:
        """Persist a managed job.

        Args:
            job_id: Job identifier.
        """

        managed = self._jobs.get(job_id)
        if managed:
            self.state_store.save_job(job_id, managed.spec.status.value, managed.to_jsonable())

    def _persist_worker(self, worker: WorkerDescriptor) -> None:
        """Persist a worker descriptor.

        Args:
            worker: Worker descriptor.
        """

        self.state_store.save_worker(worker.worker_id, worker.status.value, worker.to_jsonable())

