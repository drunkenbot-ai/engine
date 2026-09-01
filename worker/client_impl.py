from __future__ import annotations

import hashlib
import os
import platform
import shutil
import socket
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

import torch

from engine.coordinator.artifacts import create_result_artifact_bundle
from engine.contracts import (
    ArtifactSpec,
    ClaimJobRequest,
    ClaimJobResponse,
    CompleteJobRequest,
    DatasetSpec,
    FailJobRequest,
    HeartbeatRequest,
    ProgressReportRequest,
    ProtocolStatus,
    RegisterWorkerRequest,
    TrainingMetrics,
    TrainingResultSpec,
    WorkerAvailability,
    WorkerCapabilities,
)
from engine.contracts.jobs import JobStatus, TrainingJobSpec
from engine.gpu_discovery import discover_gpus
from engine.training_orchestrator import train_from_dataset
try:
    import psutil
except ImportError:
    psutil = None
from .client_core import CoordinatorHttpClient, WorkerClientConfig
class RemoteWorkerClient:
    """Remote worker client that talks to the coordinator API."""

    def __init__(self, config: WorkerClientConfig) -> None:
        """Create a remote worker client.

        Args:
            config: Worker client configuration.
        """

        self.config = config
        self.http = CoordinatorHttpClient(config.coordinator_url, config.api_key)
        self.stop_requested = False
        self.pause_requested = False
        self.active_job_id: Optional[str] = None

    def register(self) -> dict[str, Any]:
        """Register this worker with the coordinator.

        Returns:
            Register response payload.
        """

        request = RegisterWorkerRequest(
            worker_id=self.config.worker_id,
            backend=self.config.backend,
            device=self.config.device,
            capabilities=detect_worker_capabilities(),
            labels=self.config.labels,
        )
        response = self.http.post("/register", request.to_jsonable())
        if response.get("heartbeat_interval_seconds"):
            self.config.heartbeat_interval_seconds = int(response["heartbeat_interval_seconds"])
        return response

    def heartbeat(self, availability: WorkerAvailability, metrics: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Send a heartbeat.

        Args:
            availability: Worker availability.
            metrics: Optional runtime metrics.

        Returns:
            Heartbeat response payload.
        """

        request = HeartbeatRequest(
            worker_id=self.config.worker_id,
            availability=availability,
            backend=self.config.backend,
            active_job_id=self.active_job_id,
            device=self.config.device,
            metrics=metrics or {},
        )
        response = self.http.post("/heartbeat", request.to_jsonable())
        self.stop_requested = bool(response.get("should_stop_job"))
        self.pause_requested = bool(response.get("should_pause_job"))
        return response

    def claim_job(self) -> Optional[TrainingJobSpec]:
        """Ask the coordinator for a compatible job.

        Returns:
            Assigned job when available.
        """

        request = ClaimJobRequest(
            worker_id=self.config.worker_id,
            backend=self.config.backend,
            capabilities=detect_worker_capabilities(),
        )
        response = ClaimJobResponse.from_jsonable(self.http.post("/claim-job", request.to_jsonable()))
        if response.status != ProtocolStatus.OK:
            raise RuntimeError(response.message)
        return response.job

    def run_forever(self) -> None:
        """Run the worker loop."""

        self.register()
        while True:
            self.heartbeat(WorkerAvailability.AVAILABLE, self._telemetry())
            if not self.config.execute_jobs and not self.config.claim_once:
                time.sleep(self.config.heartbeat_interval_seconds)
                continue
            job = self.claim_job()
            if job is None:
                if self.config.claim_once:
                    return
                time.sleep(self.config.heartbeat_interval_seconds)
                continue
            job = self.sync_job_artifacts(job)
            self.active_job_id = job.job_id
            try:
                if self.config.execute_jobs:
                    self.execute_job(job)
                else:
                    self.fail_job(job.job_id, "Worker execution disabled. Run with --execute to train jobs.", retryable=True)
            finally:
                self.active_job_id = None
            if self.config.claim_once:
                return

    def execute_job(self, job: TrainingJobSpec) -> None:
        """Execute a claimed training job.

        Args:
            job: Claimed training job.
        """

        self.stop_requested = False
        self.pause_requested = False

        def progress(event: Any) -> None:
            metrics = _event_to_metrics(event)
            response = self.http.post(
                "/progress",
                ProgressReportRequest(self.config.worker_id, job.job_id, metrics).to_jsonable(),
            )
            self.stop_requested = bool(response.get("should_stop_job"))
            self.pause_requested = bool(response.get("should_pause_job"))
            while self.pause_requested and not self.stop_requested:
                time.sleep(self.config.heartbeat_interval_seconds)
                heartbeat_response = self.heartbeat(WorkerAvailability.BUSY, {"paused": True})
                self.pause_requested = bool(heartbeat_response.get("should_pause_job"))

        try:
            result = train_from_dataset(
                job.dataset.dataset_dir,
                job.model.config,
                job.training,
                progress=progress,
                should_stop=lambda: self.stop_requested,
            )
        except Exception as exc:
            self.fail_job(job.job_id, str(exc), retryable=False)
            print(f"Job {job.job_id} failed on worker {self.config.worker_id}: {exc}")
            return
        status = JobStatus.CANCELLED if result.stopped else JobStatus.COMPLETED
        artifact_bundle_url = self.upload_result_artifacts(job)
        self.http.post(
            "/complete",
            CompleteJobRequest(
                self.config.worker_id,
                TrainingResultSpec(
                    job_id=job.job_id,
                    status=status,
                    checkpoint_path=result.checkpoint_path,
                    summary_path=result.summary_path,
                    final_train_loss=result.final_train_loss,
                    final_val_loss=result.final_val_loss,
                    stopped=result.stopped,
                    artifact_bundle_url=artifact_bundle_url,
                ),
            ).to_jsonable(),
        )

    def sync_job_artifacts(self, job: TrainingJobSpec) -> TrainingJobSpec:
        """Download and localize remote job artifacts.

        Args:
            job: Claimed job from the coordinator.

        Returns:
            Job rewritten to worker-local paths.
        """

        manifest_url = str(job.metadata.get("project_manifest_url") or "")
        if manifest_url:
            return self._sync_project_delta(job, manifest_url)
        bundle_url = str(job.metadata.get("artifact_bundle_url") or "")
        if not bundle_url:
            return job
        workspace = self._job_workspace(job.job_id)
        bundle_path = workspace / "input_bundle.zip"
        extract_dir = workspace / "input"
        # A manifest makes cache reuse safe and turns an interrupted transfer
        # into a Range-resumed download. Older coordinators may not expose it.
        manifest = None
        try:
            manifest = self.http.get(f"{bundle_url.rstrip('/')}/manifest")
        except Exception:
            pass
        if manifest and bundle_path.is_file() and _file_sha256(bundle_path) == manifest.get("sha256"):
            pass
        else:
            self.http.download(bundle_url, bundle_path)
            if manifest and _file_sha256(bundle_path) != manifest.get("sha256"):
                # Do not ever train on a corrupt or mixed-version partial cache.
                bundle_path.unlink(missing_ok=True)
                self.http.download(bundle_url, bundle_path)
                if _file_sha256(bundle_path) != manifest.get("sha256"):
                    raise ValueError(f"Artifact integrity check failed: {bundle_url}")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract_zip(bundle_path, extract_dir)
        dataset_dir = extract_dir / "dataset"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Downloaded job bundle does not contain dataset/: {bundle_url}")
        output_dir = workspace / "model"
        output_dir.mkdir(parents=True, exist_ok=True)
        job.dataset = DatasetSpec.from_dataset_dir(dataset_dir)
        job.training.output_dir = output_dir
        job.artifacts = ArtifactSpec.from_output_dir(output_dir)
        resume_artifact = job.metadata.get("resume_checkpoint_artifact")
        if resume_artifact:
            resume_path = extract_dir / str(resume_artifact)
            if resume_path.is_file():
                job.training.resume_from_checkpoint = resume_path
        base_artifact = job.metadata.get("base_checkpoint_artifact")
        if base_artifact:
            base_path = extract_dir / str(base_artifact)
            if base_path.is_file():
                job.training.fine_tune_from_checkpoint = base_path
                job.model.base_checkpoint = base_path
        return job

    def _sync_project_delta(self, job: TrainingJobSpec, manifest_url: str) -> TrainingJobSpec:
        """Synchronize only changed project files, with per-file resume/hash checks."""
        manifest = self.http.get(manifest_url)
        base_url = str(job.metadata["project_base_url"]).rstrip("/")
        workspace = self._job_workspace(job.job_id)
        input_dir = workspace / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        expected = set()
        for entry in manifest.get("files", []):
            relative = Path(str(entry["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe project manifest path: {relative}")
            expected.add(relative.as_posix())
            local = input_dir / relative
            digest = str(entry.get("sha256") or "")
            if not local.is_file() or _file_sha256(local) != digest:
                self.http.download(f"{base_url}/{relative.as_posix()}", local)
                if _file_sha256(local) != digest:
                    local.unlink(missing_ok=True)
                    self.http.download(f"{base_url}/{relative.as_posix()}", local)
                    if _file_sha256(local) != digest:
                        raise ValueError(
                            f"Project file integrity check failed: {relative}"
                        )
        for local in input_dir.rglob("*"):
            if local.is_file() and local.relative_to(input_dir).as_posix() not in expected:
                local.unlink()
        dataset_dir = input_dir / "dataset"
        if not dataset_dir.exists():
            raise FileNotFoundError("Project manifest does not contain dataset/")
        output_dir = workspace / "model"
        output_dir.mkdir(parents=True, exist_ok=True)
        job.dataset = DatasetSpec.from_dataset_dir(dataset_dir)
        job.training.output_dir = output_dir
        job.artifacts = ArtifactSpec.from_output_dir(output_dir)
        return job

    def upload_result_artifacts(self, job: TrainingJobSpec) -> Optional[str]:
        """Upload worker output artifacts to the coordinator.

        Args:
            job: Completed job.

        Returns:
            Coordinator artifact URL when upload succeeds.
        """

        output_dir = Path(job.training.output_dir)
        if not output_dir.exists():
            return None
        bundle_path = self._job_workspace(job.job_id) / "result_bundle.zip"
        create_result_artifact_bundle(job.job_id, output_dir, bundle_path)
        remote_path = f"/artifacts/results/{job.job_id}/{bundle_path.name}"
        response = self.http.upload(remote_path, bundle_path)
        return str(response.get("artifact_url") or remote_path)

    def _job_workspace(self, job_id: str) -> Path:
        """Return the worker-local workspace for a job.

        Args:
            job_id: Training job identifier.

        Returns:
            Worker-local job workspace.
        """

        workspace = Path(self.config.workspace_dir) / job_id
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _telemetry(self) -> dict[str, Any]:
        """Collect a compact, cross-platform heartbeat telemetry snapshot."""
        snapshot: dict[str, Any] = {"gpus": [gpu.to_jsonable() for gpu in discover_gpus()]}
        if psutil is not None:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(
                str(Path(self.config.workspace_dir).anchor or Path.cwd())
            )
            network = psutil.net_io_counters()
            snapshot.update({"cpu_percent": psutil.cpu_percent(), "ram_percent": memory.percent,
                "free_ram_gb": memory.available / (1024**3), "free_disk_gb": disk.free / (1024**3),
                "network_bytes_sent": network.bytes_sent, "network_bytes_recv": network.bytes_recv})
        return snapshot

    def fail_job(self, job_id: str, error: str, retryable: bool) -> None:
        """Report job failure to the coordinator.

        Args:
            job_id: Job identifier.
            error: Failure text.
            retryable: Whether the job may be retried.
        """

        self.http.post("/fail", FailJobRequest(self.config.worker_id, job_id, error, retryable).to_jsonable())


def detect_worker_capabilities() -> WorkerCapabilities:
    """Detect local worker hardware capabilities.

    Returns:
        Worker capabilities.
    """

    gpu_records = [gpu.to_jsonable() for gpu in discover_gpus()]
    gpu_names: list[str] = [str(gpu["name"]) for gpu in gpu_records]
    total_vram_gb: Optional[float] = sum(float(gpu.get("vram_total_gb") or 0) for gpu in gpu_records) or None
    system_ram_gb = None
    if psutil is not None:
        system_ram_gb = psutil.virtual_memory().total / (1024**3)
    return WorkerCapabilities(
        hostname=socket.gethostname(),
        platform=f"{platform.system()} {platform.release()}",
        cpu_count=os.cpu_count(),
        system_ram_gb=system_ram_gb,
        gpu_names=gpu_names,
        total_vram_gb=total_vram_gb,
        supports_cuda=torch.cuda.is_available(),
        supports_bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        supports_fp16=torch.cuda.is_available(),
        extra={"gpus": gpu_records},
    )


def run_worker_client(config: WorkerClientConfig) -> None:
    """Run a remote worker client.

    Args:
        config: Worker client configuration.
    """

    RemoteWorkerClient(config).run_forever()


def _event_to_metrics(event: Any) -> TrainingMetrics:
    """Convert a training progress event into protocol metrics.

    Args:
        event: Progress event.

    Returns:
        Training metrics.
    """

    if not isinstance(event, dict):
        return TrainingMetrics(message=str(event))
    return TrainingMetrics(
        step=_int_or_none(event.get("step")),
        total_steps=_int_or_none(event.get("total_steps")),
        epoch=_int_or_none(event.get("epoch")),
        total_epochs=_int_or_none(event.get("epochs") or event.get("total_epochs")),
        train_loss=_float_or_none(event.get("loss") or event.get("train_loss")),
        val_loss=_float_or_none(event.get("val_loss")),
        learning_rate=_float_or_none(event.get("learning_rate") or event.get("lr")),
        tokens_per_second=_float_or_none(event.get("tokens_per_second") or event.get("tokens_per_sec")),
        samples_per_second=_float_or_none(event.get("samples_per_second") or event.get("samples_per_sec")),
        gpu_memory_percent=_float_or_none(event.get("gpu_memory_percent")),
        system_ram_percent=_float_or_none(event.get("system_ram_percent")),
        message=str(event.get("message")) if event.get("message") is not None else None,
    )


def _int_or_none(value: Any) -> Optional[int]:
    """Convert a value to int when possible.

    Args:
        value: Input value.

    Returns:
        Integer or None.
    """

    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> Optional[float]:
    """Convert a value to float when possible.

    Args:
        value: Input value.

    Returns:
        Float or None.
    """

    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a zip file without allowing path traversal.

    Args:
        zip_path: Zip file path.
        target_dir: Destination directory.

    Raises:
        ValueError: If a zip member would escape the target directory.
    """

    root = Path(target_dir).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = (root / member.filename).resolve()
            if root not in member_path.parents and member_path != root:
                raise ValueError(f"Unsafe artifact member path: {member.filename}")
        archive.extractall(root)


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for artifact integrity checks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
