from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import torch

from engine.coordinator.artifacts import create_result_artifact_bundle
from engine.contracts import (
    ArtifactSpec,
    BackendKind,
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
from engine.training_orchestrator import train_from_dataset

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class WorkerClientConfig:
    """Configuration for a remote worker client.

    Attributes:
        coordinator_url: Base URL for the coordinator API.
        worker_id: Stable worker identifier.
        device: Preferred training device.
        labels: Worker scheduling labels.
        heartbeat_interval_seconds: Seconds between heartbeats.
        execute_jobs: Whether to execute claimed jobs.
        claim_once: Whether to claim at most one job and exit.
        workspace_dir: Local folder used for downloaded jobs and outputs.
    """

    coordinator_url: str = "http://127.0.0.1:8765"
    worker_id: str = field(default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    labels: list[str] = field(default_factory=list)
    heartbeat_interval_seconds: int = 10
    execute_jobs: bool = False
    claim_once: bool = False
    workspace_dir: Path = field(default_factory=lambda: Path.home() / ".drunkenbot_ide" / "worker_workspace")


class CoordinatorHttpClient:
    """Small JSON HTTP client for the coordinator API."""

    def __init__(self, base_url: str) -> None:
        """Create an HTTP client.

        Args:
            base_url: Coordinator base URL.
        """

        self.base_url = base_url.rstrip("/")

    def get(self, path: str) -> dict[str, Any]:
        """Send a GET request.

        Args:
            path: API path.

        Returns:
            JSON response payload.
        """

        with urlopen(f"{self.base_url}{path}", timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON POST request.

        Args:
            path: API path.
            payload: Request payload.

        Returns:
            JSON response payload.
        """

        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def download(self, path_or_url: str, output_path: Path) -> None:
        """Download a binary artifact.

        Args:
            path_or_url: Absolute URL or coordinator-relative path.
            output_path: Destination file path.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(self.absolute_url(path_or_url), timeout=300) as response, output_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)

    def upload(self, path_or_url: str, input_path: Path) -> dict[str, Any]:
        """Upload a binary artifact.

        Args:
            path_or_url: Absolute URL or coordinator-relative path.
            input_path: Source file path.

        Returns:
            JSON response payload.
        """

        data = input_path.read_bytes()
        request = Request(
            self.absolute_url(path_or_url),
            data=data,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(data)),
            },
            method="PUT",
        )
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))

    def absolute_url(self, path_or_url: str) -> str:
        """Build an absolute coordinator URL.

        Args:
            path_or_url: Absolute URL or coordinator-relative path.

        Returns:
            Absolute URL string.
        """

        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        return urljoin(f"{self.base_url}/", path_or_url.lstrip("/"))



