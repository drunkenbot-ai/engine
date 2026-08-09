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
from engine.coordinator.job_models import ManagedJob, WorkerDescriptor, WorkerHeartbeat, WorkerStatus
from engine.training import TrainingResult


def _availability_to_worker_status(availability: WorkerAvailability) -> WorkerStatus:
    """Convert the wire-level availability enum into stored worker status."""
    return {
        WorkerAvailability.AVAILABLE: WorkerStatus.AVAILABLE,
        WorkerAvailability.BUSY: WorkerStatus.BUSY,
        WorkerAvailability.OFFLINE: WorkerStatus.OFFLINE,
    }[availability]


def _control_message(should_stop: bool, should_pause: bool, default: str) -> str:
    """Return the remote worker control state as a human-readable message."""
    if should_stop:
        return "stop requested"
    if should_pause:
        return "pause requested"
    return default


class JobManagerCore:
    """Coordinates training jobs across available backends.

    The first implementation runs local jobs in-process. The API is intentionally
    shaped like a distributed coordinator so remote clients can be added without
    rewriting the desktop app's training calls.
    """

    def __init__(
        self,
        registry: Optional[BackendRegistry] = None,
        state_store: Optional[JobStateStore] = None,
    ) -> None:
        """Create a job manager.

        Args:
            registry: Backend registry used to resolve job backends.
            state_store: Optional persistent state store.
        """

        self.registry = registry or DEFAULT_BACKEND_REGISTRY
        self.state_store = state_store or JobStateStore()
        self._jobs: dict[str, ManagedJob] = {}
        self._workers: dict[str, WorkerDescriptor] = {}
        self._stop_requested: set[str] = set()
        self._restore_state()
        self.register_worker(
            WorkerDescriptor(
                worker_id="local",
                backend=BackendKind.LOCAL,
                status=WorkerStatus.AVAILABLE,
                device="auto",
                hostname="localhost",
            )
        )

    def register_worker(self, worker: WorkerDescriptor) -> None:
        """Register or update a worker.

        Args:
            worker: Worker descriptor.
        """

        self._workers[worker.worker_id] = worker
        self._persist_worker(worker)

    def register_remote_worker(self, request: RegisterWorkerRequest) -> RegisterWorkerResponse:
        """Register a remote worker from a protocol request.

        Args:
            request: Register worker request.

        Returns:
            Register worker response.
        """

        if not request.worker_id.strip():
            return RegisterWorkerResponse(
                worker_id=request.worker_id,
                accepted=False,
                status=ProtocolStatus.REJECTED,
                message="worker_id is required",
            )
        capabilities = request.capabilities.to_jsonable()
        if request.labels:
            capabilities["labels"] = request.labels
        self.register_worker(
            WorkerDescriptor(
                worker_id=request.worker_id,
                backend=request.backend,
                status=WorkerStatus.AVAILABLE,
                device=request.device,
                hostname=request.capabilities.hostname,
                capabilities=capabilities,
                last_heartbeat_at=utc_now_iso(),
            )
        )
        return RegisterWorkerResponse(
            worker_id=request.worker_id,
            accepted=True,
            status=ProtocolStatus.OK,
            heartbeat_interval_seconds=10,
            message="worker registered",
        )

    def record_heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        """Record a worker heartbeat and update worker availability.

        Args:
            heartbeat: Worker heartbeat.
        """

        worker = self._workers.get(
            heartbeat.worker_id,
            WorkerDescriptor(
                worker_id=heartbeat.worker_id,
                backend=heartbeat.backend,
                device=heartbeat.device,
            ),
        )
        worker.backend = heartbeat.backend
        worker.status = heartbeat.status
        worker.device = heartbeat.device
        worker.last_heartbeat_at = heartbeat.timestamp
        self._workers[worker.worker_id] = worker
        self._persist_worker(worker)
        self.state_store.record_heartbeat(worker.worker_id, heartbeat.to_jsonable())

    def handle_heartbeat(self, request: HeartbeatRequest) -> HeartbeatResponse:
        """Handle a protocol heartbeat request from a worker.

        Args:
            request: Heartbeat request.

        Returns:
            Heartbeat response.
        """

        if not request.worker_id.strip():
            return HeartbeatResponse(
                status=ProtocolStatus.REJECTED,
                should_stop_job=True,
                message="worker_id is required",
            )
        heartbeat = WorkerHeartbeat(
            worker_id=request.worker_id,
            status=_availability_to_worker_status(request.availability),
            backend=request.backend,
            active_job_id=request.active_job_id,
            device=request.device,
            metrics=request.metrics,
            timestamp=request.sent_at,
        )
        self.record_heartbeat(heartbeat)
        should_stop = bool(request.active_job_id and self._should_stop_remote_job(request.active_job_id))
        should_pause = bool(request.active_job_id and self._should_pause_remote_job(request.active_job_id))
        return HeartbeatResponse(
            status=ProtocolStatus.OK,
            should_stop_job=should_stop,
            should_pause_job=should_pause,
            message=_control_message(should_stop, should_pause, "heartbeat accepted"),
        )

    def handle_claim_job(self, request: ClaimJobRequest) -> ClaimJobResponse:
        """Assign a queued job to a worker when compatible work exists.

        Args:
            request: Claim job request.

        Returns:
            Claim job response containing an assigned job when available.
        """

        if not request.worker_id.strip():
            return ClaimJobResponse(status=ProtocolStatus.REJECTED, message="worker_id is required")
        worker = self._workers.get(
            request.worker_id,
            WorkerDescriptor(
                worker_id=request.worker_id,
                backend=request.backend,
                status=WorkerStatus.AVAILABLE,
                hostname=request.capabilities.hostname,
                capabilities=request.capabilities.to_jsonable(),
            ),
        )
        worker.backend = request.backend
        worker.status = WorkerStatus.AVAILABLE
        worker.hostname = request.capabilities.hostname or worker.hostname
        worker.capabilities.update(request.capabilities.to_jsonable())
        worker.last_heartbeat_at = utc_now_iso()
        self._workers[worker.worker_id] = worker
        self._persist_worker(worker)

        for managed in self._jobs.values():
            job = managed.spec
            if not self._worker_can_claim_job(worker, job):
                continue
            managed.assigned_worker_id = worker.worker_id
            worker.status = WorkerStatus.BUSY
            job.status = JobStatus.ASSIGNED
            self._persist_worker(worker)
            self._persist_job(job.job_id)
            return ClaimJobResponse(job=job, status=ProtocolStatus.OK, message="job assigned")
        return ClaimJobResponse(status=ProtocolStatus.OK, message="no compatible queued job")

    def handle_progress_report(self, request: ProgressReportRequest) -> ProgressReportResponse:
        """Handle progress metrics from a worker.

        Args:
            request: Progress report request.

        Returns:
            Progress report response.
        """

        managed = self._jobs.get(request.job_id)
        if managed is None:
            return ProgressReportResponse(
                status=ProtocolStatus.REJECTED,
                should_stop_job=True,
                message="unknown job",
            )
        if managed.assigned_worker_id and managed.assigned_worker_id != request.worker_id:
            return ProgressReportResponse(
                status=ProtocolStatus.REJECTED,
                should_stop_job=True,
                message="job is assigned to a different worker",
            )
        managed.assigned_worker_id = request.worker_id
        managed.latest_metrics = request.metrics
        managed.updated_at = request.sent_at
        if managed.spec.status == JobStatus.ASSIGNED:
            managed.spec.status = JobStatus.RUNNING
        worker = self._workers.get(request.worker_id)
        if worker:
            worker.status = WorkerStatus.BUSY
            worker.last_heartbeat_at = request.sent_at
            self._persist_worker(worker)
        self._persist_job(request.job_id)
        should_stop = self._should_stop_remote_job(request.job_id)
        should_pause = self._should_pause_remote_job(request.job_id)
        return ProgressReportResponse(
            status=ProtocolStatus.OK,
            should_stop_job=should_stop,
            should_pause_job=should_pause,
            message=_control_message(should_stop, should_pause, "progress accepted"),
        )

    def handle_complete_job(self, request: CompleteJobRequest) -> CompleteJobResponse:
        """Handle successful remote job completion.

        Args:
            request: Complete job request.

        Returns:
            Complete job response.
        """

        if request.result is None:
            return CompleteJobResponse(status=ProtocolStatus.REJECTED, message="result is required")
        managed = self._jobs.get(request.result.job_id)
        if managed is None:
            return CompleteJobResponse(status=ProtocolStatus.REJECTED, message="unknown job")
        managed.assigned_worker_id = request.worker_id
        managed.result = request.result
        managed.spec.status = request.result.status
        managed.updated_at = request.sent_at
        managed.error = request.result.error
        worker = self._workers.get(request.worker_id)
        if worker:
            worker.status = WorkerStatus.AVAILABLE
            worker.last_heartbeat_at = request.sent_at
            self._persist_worker(worker)
        self._persist_job(request.result.job_id)
        self._stop_requested.discard(request.result.job_id)
        return CompleteJobResponse(status=ProtocolStatus.OK, message="job completion accepted")

    def handle_fail_job(self, request: FailJobRequest) -> FailJobResponse:
        """Handle remote job failure.

        Args:
            request: Fail job request.

        Returns:
            Fail job response.
        """

        managed = self._jobs.get(request.job_id)
        if managed is None:
            return FailJobResponse(status=ProtocolStatus.REJECTED, message="unknown job")
        managed.assigned_worker_id = None if request.retryable else request.worker_id
        managed.spec.status = JobStatus.QUEUED if request.retryable else JobStatus.FAILED
        managed.error = request.error
        managed.updated_at = request.sent_at
        worker = self._workers.get(request.worker_id)
        if worker:
            worker.status = WorkerStatus.AVAILABLE
            worker.last_heartbeat_at = request.sent_at
            self._persist_worker(worker)
        self._persist_job(request.job_id)
        self._stop_requested.discard(request.job_id)
        return FailJobResponse(
            status=ProtocolStatus.OK,
            message="job requeued after failure" if request.retryable else "job failure accepted",
        )

    def mark_stale_workers_offline(self, timeout_seconds: int = 30) -> list[str]:
        """Mark workers offline when their last heartbeat is too old.

        Args:
            timeout_seconds: Age in seconds after which a worker is stale.

        Returns:
            Worker IDs marked offline.
        """

        marked: list[str] = []
        cutoff = datetime.now().astimezone() - timedelta(seconds=timeout_seconds)
        for worker in self._workers.values():
            if worker.worker_id == "local" or worker.status == WorkerStatus.OFFLINE:
                continue
            heartbeat_at = _parse_timestamp(worker.last_heartbeat_at)
            if heartbeat_at and heartbeat_at < cutoff:
                worker.status = WorkerStatus.OFFLINE
                self._persist_worker(worker)
                marked.append(worker.worker_id)
        return marked

    def list_workers(self) -> list[WorkerDescriptor]:
        """Return known workers.

        Returns:
            Registered workers.
        """

        return list(self._workers.values())

    def submit(self, job: TrainingJobSpec) -> str:
        """Submit a job to the manager queue.

        Args:
            job: Training job contract.

        Returns:
            Job identifier.
        """

        job.status = JobStatus.QUEUED
        self._jobs[job.job_id] = ManagedJob(spec=job)
        self._persist_job(job.job_id)
        return job.job_id

    def get_job(self, job_id: str) -> ManagedJob:
        """Return a managed job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            Managed job.

        Raises:
            KeyError: If the job is unknown.
        """

        return self._jobs[job_id]

    def list_jobs(self) -> list[ManagedJob]:
        """Return tracked jobs.

        Returns:
            Managed jobs.
        """

        return list(self._jobs.values())

    def cancel(self, job_id: str) -> None:
        """Request cooperative cancellation for a job.

        Args:
            job_id: Job identifier.
        """

        managed = self.get_job(job_id)
        managed.spec.status = JobStatus.STOPPING
        self._stop_requested.add(job_id)
        self._persist_job(job_id)

    def stop_all_jobs(self) -> int:
        """Request cooperative stop for all active jobs.

        Returns:
            Number of jobs marked for stopping.
        """

        count = 0
        for managed in self._jobs.values():
            if managed.spec.status in {JobStatus.QUEUED, JobStatus.ASSIGNED, JobStatus.RUNNING, JobStatus.PAUSED}:
                managed.spec.status = JobStatus.STOPPING
                self._stop_requested.add(managed.spec.job_id)
                self._persist_job(managed.spec.job_id)
                count += 1
        return count

    def pause_all_jobs(self) -> int:
        """Pause queued or active remote jobs.

        Returns:
            Number of jobs marked paused.
        """

        count = 0
        for managed in self._jobs.values():
            if managed.spec.status in {JobStatus.QUEUED, JobStatus.ASSIGNED, JobStatus.RUNNING}:
                managed.spec.status = JobStatus.PAUSED
                self._persist_job(managed.spec.job_id)
                count += 1
        return count

    def resume_all_jobs(self) -> int:
        """Resume paused jobs by returning them to the queue.

        Returns:
            Number of jobs resumed.
        """

        count = 0
        for managed in self._jobs.values():
            if managed.spec.status == JobStatus.PAUSED:
                managed.spec.status = JobStatus.QUEUED
                managed.assigned_worker_id = None
                self._persist_job(managed.spec.job_id)
                count += 1
        return count

    def run_next(
        self,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCallback] = None,
    ) -> TrainingResult:
        """Run the next queued job.

        Args:
            progress: Optional progress callback.
            should_stop: Optional external stop callback.

        Returns:
            Training result.

        Raises:
            ValueError: If no queued job is available.
        """

        for managed in self._jobs.values():
            if managed.spec.status == JobStatus.QUEUED:
                return self.run_job(managed.spec.job_id, progress=progress, should_stop=should_stop)
        raise ValueError("No queued training jobs are available")

