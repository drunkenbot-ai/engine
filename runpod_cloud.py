from __future__ import annotations

import json
import logging
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


LOGGER = logging.getLogger(__name__)
RUNPOD_REST_BASE = "https://rest.runpod.io/v1"


@dataclass
class RunPodConfig:
    """Settings used to launch RunPod worker Pods."""

    api_key: str = ""
    image_name: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    gpu_type_id: str = "NVIDIA GeForce RTX 4090"
    gpu_count: int = 1
    cloud_type: str = "COMMUNITY"
    interruptible: bool = True
    container_disk_gb: int = 80
    volume_gb: int = 40
    min_vcpu_per_gpu: int = 4
    min_ram_per_gpu: int = 16
    auto_terminate: bool = True
    worker_labels: str = "runpod,gpu"


@dataclass
class RunPodLaunchResult:
    """Result returned after launching a RunPod worker Pod."""

    pod_id: str
    pod_name: str
    cost_per_hour: str
    gpu_name: str
    worker_id: str
    bootstrap_url: str


def default_runpod_config_path(project_dir: Optional[Path] = None) -> Path:
    """Return the RunPod config path.

    Args:
        project_dir: Optional project root folder.

    Returns:
        Project-specific or user-profile RunPod config path.
    """

    if project_dir is not None:
        return project_dir / "runpod_config.json"
    return Path.home() / ".drunkenbot_ide" / "runpod_config.json"


def ensure_runpod_config(path: Path) -> Path:
    """Create a disabled/default RunPod config when missing.

    Args:
        path: Config path.

    Returns:
        The same config path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(asdict(RunPodConfig()), indent=2), encoding="utf-8")
        LOGGER.info("Created RunPod config: %s", path)
    return path


def load_runpod_config(path: Path) -> RunPodConfig:
    """Load RunPod settings from JSON.

    Args:
        path: Config path.

    Returns:
        Parsed RunPod configuration.
    """

    ensure_runpod_config(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return RunPodConfig(**{**asdict(RunPodConfig()), **data})


def save_runpod_config(path: Path, config: RunPodConfig) -> None:
    """Save RunPod settings to JSON.

    Args:
        path: Config path.
        config: RunPod configuration.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def create_runpod_worker_bundle(project_root: Path, artifact_root: Path, bundle_name: str = "runpod_worker_bootstrap.zip") -> Path:
    """Create a worker source bundle served by the coordinator.

    Args:
        project_root: Local project root containing the engine/ package.
        artifact_root: Coordinator artifact root.
        bundle_name: Output zip file name.

    Returns:
        Path to the created bootstrap bundle.
    """

    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    bundle_path = artifact_root / bundle_name
    source_package = "engine"
    source_root = project_root / source_package
    if not source_root.is_dir():
        # Keep bundles usable for older checkouts while new bundles use the
        # canonical engine package.
        source_package = "llm_trainer"
        source_root = project_root / source_package
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_tree(archive, source_root, f"worker_src/{source_package}")
        archive.writestr("worker_src/requirements.txt", _worker_requirements(project_root))
        run_app = project_root / "run_app.py"
        if run_app.exists():
            archive.write(run_app, "worker_src/run_app.py")
        init_file = project_root / "__init__.py"
        if init_file.exists():
            archive.write(init_file, "worker_src/__init__.py")
        archive.writestr("worker_src/run_worker.py", _worker_runner_script(source_package))
    return bundle_path


def build_runpod_start_command(
    bootstrap_url: str,
    coordinator_url: str,
    worker_id: str,
    labels: str,
    claim_once: bool,
) -> list[str]:
    """Build the Pod start command that installs and runs a worker.

    Args:
        bootstrap_url: Public URL for the worker source bundle.
        coordinator_url: Public coordinator URL.
        worker_id: Stable worker identifier.
        labels: Worker labels.
        claim_once: Whether the worker exits after one job.

    Returns:
        Docker start command array.
    """

    script = f"""
set -e
mkdir -p /workspace/micro_llm_worker
cd /workspace/micro_llm_worker
python - <<'PY'
import urllib.request
urllib.request.urlretrieve({bootstrap_url!r}, 'worker_bootstrap.zip')
PY
python - <<'PY'
import zipfile
with zipfile.ZipFile('worker_bootstrap.zip') as archive:
    archive.extractall('.')
PY
cd worker_src
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_worker.py --coordinator-url {coordinator_url!r} --worker-id {worker_id!r} --labels {labels!r} {'--claim-once' if claim_once else ''}
"""
    return ["bash", "-lc", textwrap.dedent(script).strip()]


class RunPodClient:
    """Small REST client for RunPod Pods."""

    def __init__(self, api_key: str, base_url: str = RUNPOD_REST_BASE) -> None:
        """Create a RunPod API client.

        Args:
            api_key: RunPod API key.
            base_url: REST API base URL.
        """

        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")

    def create_worker_pod(
        self,
        config: RunPodConfig,
        pod_name: str,
        worker_id: str,
        coordinator_url: str,
        bootstrap_url: str,
    ) -> RunPodLaunchResult:
        """Create and start a RunPod worker Pod.

        Args:
            config: RunPod settings.
            pod_name: Pod name.
            worker_id: Worker identifier.
            coordinator_url: Public coordinator URL.
            bootstrap_url: Public worker bootstrap bundle URL.

        Returns:
            Launch summary.
        """

        if not self.api_key:
            raise ValueError("RunPod API key is missing. Edit runpod_config.json first.")
        payload = {
            "name": pod_name,
            "cloudType": config.cloud_type,
            "computeType": "GPU",
            "imageName": config.image_name,
            "gpuCount": max(1, int(config.gpu_count)),
            "gpuTypeIds": [config.gpu_type_id] if config.gpu_type_id.strip() else [],
            "gpuTypePriority": "availability",
            "containerDiskInGb": int(config.container_disk_gb),
            "volumeInGb": int(config.volume_gb),
            "volumeMountPath": "/workspace",
            "minVCPUPerGPU": int(config.min_vcpu_per_gpu),
            "minRAMPerGPU": int(config.min_ram_per_gpu),
            "interruptible": bool(config.interruptible),
            "supportPublicIp": True,
            "ports": ["22/tcp"],
            "env": {
                "MICRO_LLM_COORDINATOR_URL": coordinator_url,
                "MICRO_LLM_WORKER_ID": worker_id,
                "MICRO_LLM_BOOTSTRAP_URL": bootstrap_url,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            "dockerStartCmd": build_runpod_start_command(
                bootstrap_url=bootstrap_url,
                coordinator_url=coordinator_url,
                worker_id=worker_id,
                labels=config.worker_labels,
                claim_once=config.auto_terminate,
            ),
        }
        response = self._request("POST", "/pods", payload)
        pod_id = str(response.get("id") or "")
        if not pod_id:
            raise RuntimeError(f"RunPod did not return a pod id: {response}")
        gpu = response.get("gpu") or response.get("machine", {}) or {}
        return RunPodLaunchResult(
            pod_id=pod_id,
            pod_name=str(response.get("name") or pod_name),
            cost_per_hour=str(response.get("costPerHr") or response.get("adjustedCostPerHr") or "-"),
            gpu_name=str(gpu.get("displayName") or gpu.get("gpuDisplayName") or config.gpu_type_id),
            worker_id=worker_id,
            bootstrap_url=bootstrap_url,
        )

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        """Stop a RunPod Pod.

        Args:
            pod_id: Pod identifier.

        Returns:
            RunPod response.
        """

        return self._request("POST", f"/pods/{pod_id}/stop", {})

    def delete_pod(self, pod_id: str) -> dict[str, Any]:
        """Delete a RunPod Pod.

        Args:
            pod_id: Pod identifier.

        Returns:
            RunPod response.
        """

        return self._request("DELETE", f"/pods/{pod_id}", None)

    def list_pods(self) -> dict[str, Any]:
        """List Pods in the RunPod account.

        Returns:
            RunPod response.
        """

        return self._request("GET", "/pods", None)

    def _request(self, method: str, path: str, payload: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Send a JSON request to RunPod.

        Args:
            method: HTTP method.
            path: REST path.
            payload: Optional JSON payload.

        Returns:
            JSON response.
        """

        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RunPod API error {exc.code}: {detail}") from exc
        if not raw:
            return {}
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"data": parsed}


def public_url_is_cloud_reachable(url: str) -> bool:
    """Return whether a URL looks reachable from RunPod.

    Args:
        url: Coordinator URL.

    Returns:
        True when the URL is not localhost/private loopback.
    """

    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return bool(host and host not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def _write_tree(archive: zipfile.ZipFile, source: Path, archive_root: str) -> None:
    """Write a source tree into a zip archive."""

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Worker source not found: {source}")
    for path in source.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            archive.write(path, Path(archive_root) / path.relative_to(source))


def _worker_runner_script(package_name: str = "engine") -> str:
    """Return the worker runner source used inside RunPod."""

    return f"""from __future__ import annotations

import argparse
from pathlib import Path

from {package_name}.worker import WorkerClientConfig, run_worker_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--labels", default="runpod,gpu")
    parser.add_argument("--workspace-dir", default="/workspace/micro_llm_worker/jobs")
    parser.add_argument("--claim-once", action="store_true")
    args = parser.parse_args()
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    config = WorkerClientConfig(
        coordinator_url=args.coordinator_url,
        worker_id=args.worker_id,
        labels=labels,
        workspace_dir=Path(args.workspace_dir),
        execute_jobs=True,
        claim_once=args.claim_once,
        device="cuda",
    )
    run_worker_client(config)


if __name__ == "__main__":
    main()
"""


def _worker_requirements(project_root: Path) -> str:
    """Return lean worker requirements.

    Args:
        project_root: Local project root.

    Returns:
        Requirements text for cloud workers.
    """

    requirements = project_root / "requirements.txt"
    if not requirements.exists():
        return "\n".join(["numpy", "torch", "tokenizers", "psutil", "datasets", "PyPDF2", "nltk"]) + "\n"
    excluded = {"pyside6", "llama-cpp-python", "pyqtgraph", "markdown", "pygments"}
    lines: list[str] = []
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        package = stripped.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].lower()
        if package in excluded:
            continue
        lines.append(stripped)
    if "torch" not in {line.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].lower() for line in lines}:
        lines.append("torch")
    return "\n".join(lines) + "\n"
