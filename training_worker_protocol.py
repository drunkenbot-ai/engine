from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .contracts import TrainingJobSpec

try:
    import psutil
except ImportError:
    psutil = None

PROCESS_IDENTITY_ERRORS = (OSError, ValueError, IndexError) + (
    (psutil.Error,) if psutil is not None else ()
)


REQUEST_SCHEMA = "drunkenbot.training-worker-request"
MANIFEST_SCHEMA = "drunkenbot.training-run-manifest"
CONTROL_SCHEMA = "drunkenbot.training-worker-control"
PROTOCOL_VERSION = 1
TERMINAL_STATUSES = {"completed", "stopped", "failed"}


def utc_now() -> str:
    """Return a stable UTC timestamp for durable protocol files."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class StandaloneTrainingRequest:
    """Versioned request consumed by the standalone local training worker."""

    run_id: str
    job: TrainingJobSpec
    manifest_path: Path
    control_path: Path
    telemetry_db_path: Path
    heartbeat_interval_seconds: float = 2.0
    telemetry_batch_size: int = 25
    telemetry_flush_interval_seconds: float = 1.0
    notifier_config_path: Optional[Path] = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": REQUEST_SCHEMA,
            "version": PROTOCOL_VERSION,
            "run_id": self.run_id,
            "job": self.job.to_jsonable(),
            "paths": {
                "manifest": str(self.manifest_path),
                "control": str(self.control_path),
                "telemetry_db": str(self.telemetry_db_path),
            },
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "telemetry": {
                "batch_size": self.telemetry_batch_size,
                "flush_interval_seconds": self.telemetry_flush_interval_seconds,
            },
            "notifier": (
                {"config_path": str(self.notifier_config_path)}
                if self.notifier_config_path
                else None
            ),
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "StandaloneTrainingRequest":
        _validate_envelope(data, REQUEST_SCHEMA)
        paths = dict(data.get("paths") or {})
        telemetry = dict(data.get("telemetry") or {})
        notifier = data.get("notifier")
        run_id = str(data.get("run_id") or "").strip()
        if not run_id:
            raise ValueError("Training worker request requires a non-empty run_id")
        for key in ("manifest", "control", "telemetry_db"):
            if not paths.get(key):
                raise ValueError(f"Training worker request is missing paths.{key}")
        return cls(
            run_id=run_id,
            job=TrainingJobSpec.from_jsonable(dict(data["job"])),
            manifest_path=Path(paths["manifest"]),
            control_path=Path(paths["control"]),
            telemetry_db_path=Path(paths["telemetry_db"]),
            heartbeat_interval_seconds=max(
                0.1,
                float(data.get("heartbeat_interval_seconds", 2.0)),
            ),
            telemetry_batch_size=max(1, int(telemetry.get("batch_size", 25))),
            telemetry_flush_interval_seconds=max(
                0.0,
                float(telemetry.get("flush_interval_seconds", 1.0)),
            ),
            notifier_config_path=(
                Path(notifier["config_path"])
                if isinstance(notifier, dict) and notifier.get("config_path")
                else None
            ),
        )


def create_worker_request(
    job: TrainingJobSpec,
    run_id: Optional[str] = None,
    notifier_config_path: Optional[Path] = None,
) -> StandaloneTrainingRequest:
    """Create a request with durable paths rooted in the job output directory."""

    output_dir = job.artifacts.output_dir
    stable_run_id = run_id or f"run_{uuid4().hex}"
    telemetry_path = job.artifacts.telemetry_db or output_dir / "training_telemetry.sqlite"
    return StandaloneTrainingRequest(
        run_id=stable_run_id,
        job=job,
        manifest_path=output_dir / "training_run_manifest.json",
        control_path=output_dir / "training_control.json",
        telemetry_db_path=telemetry_path,
        notifier_config_path=notifier_config_path,
    )


def write_worker_request(path: Path, request: StandaloneTrainingRequest) -> Path:
    """Atomically write a standalone worker request."""

    atomic_write_json(path, request.to_jsonable())
    return path


def load_worker_request(path: Path) -> StandaloneTrainingRequest:
    """Load and validate a standalone worker request."""

    return StandaloneTrainingRequest.from_jsonable(_load_json(path))


def worker_command(
    request_path: Path,
    python_executable: Optional[Path] = None,
) -> list[str]:
    """Build the stable module invocation for a durable request."""

    executable = str(python_executable or Path(sys.executable))
    return [
        executable,
        "-m",
        "engine.training_worker",
        "--request",
        str(Path(request_path).resolve()),
    ]


def launch_worker_process(
    request_path: Path,
    python_executable: Optional[Path] = None,
) -> subprocess.Popen[bytes]:
    """Launch a session-independent worker with output redirected to durable logs."""

    request_path = Path(request_path).resolve()
    request = load_worker_request(request_path)
    output_dir = request.job.artifacts.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "training_worker_stdout.log"
    stderr_path = output_dir / "training_worker_stderr.log"
    environment = os.environ.copy()
    package_parent = str(Path(__file__).resolve().parent.parent)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        package_parent + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else package_parent
    )
    popen_options: dict[str, Any] = {
        "close_fds": True,
        "env": environment,
    }
    if os.name == "nt":
        popen_options["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        popen_options["start_new_session"] = True
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        return subprocess.Popen(
            worker_command(request_path, python_executable),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            **popen_options,
        )


def write_stop_request(path: Path, run_id: str) -> Path:
    """Atomically request cooperative stop for one durable run identity."""

    atomic_write_json(
        path,
        {
            "schema": CONTROL_SCHEMA,
            "version": PROTOCOL_VERSION,
            "run_id": run_id,
            "action": "stop",
            "requested_at": utc_now(),
        },
    )
    return path


def load_stop_request(path: Path, expected_run_id: str) -> Optional[dict[str, Any]]:
    """Return a valid stop request, or None when no sentinel exists."""

    if not path.exists():
        return None
    data = _load_json(path)
    _validate_envelope(data, CONTROL_SCHEMA)
    if data.get("run_id") != expected_run_id:
        raise ValueError("Control request run_id does not match the active worker run")
    if data.get("action") != "stop":
        raise ValueError(f"Unsupported control action: {data.get('action')!r}")
    return data


class AtomicRunManifest:
    """Thread-safe atomic writer for heartbeat and lifecycle state."""

    def __init__(self, path: Path, initial: dict[str, Any]) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data = dict(initial)
        self.write()

    def update(self, **changes: Any) -> None:
        with self._lock:
            self._data.update(changes)
            self._data["updated_at"] = utc_now()
            self.write()

    def heartbeat(self) -> None:
        self.update(heartbeat_at=utc_now())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def write(self) -> None:
        with self._lock:
            atomic_write_json(self.path, self._data)


def initial_run_manifest(
    request: StandaloneTrainingRequest,
    request_path: Path,
) -> dict[str, Any]:
    """Build the initial worker-owned manifest without notifier credentials."""

    now = utc_now()
    return {
        "schema": MANIFEST_SCHEMA,
        "version": PROTOCOL_VERSION,
        "run_id": request.run_id,
        "status": "starting",
        "pid": os.getpid(),
        "process_identity": process_identity(os.getpid()),
        "started_at": now,
        "updated_at": now,
        "heartbeat_at": now,
        "finished_at": None,
        "request_path": str(Path(request_path)),
        "output_paths": {
            "output_dir": str(request.job.artifacts.output_dir),
            "telemetry_db": str(request.telemetry_db_path),
            "checkpoint": None,
            "summary": None,
        },
        "config_fingerprint": config_fingerprint(request.job.to_jsonable()),
        "exit_code": None,
        "failure": None,
        "stop_requested_at": None,
    }


def load_run_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a worker run manifest."""

    data = _load_json(path)
    _validate_envelope(data, MANIFEST_SCHEMA)
    return data


def process_identity(pid: int) -> dict[str, str]:
    """Return an OS process creation identity suitable for PID reuse checks."""

    if psutil is not None:
        return {
            "kind": "psutil-create-time",
            "value": f"{psutil.Process(pid).create_time():.6f}",
        }
    if os.name == "nt":
        value = _windows_process_creation_filetime(pid)
        return {"kind": "windows-creation-filetime", "value": str(value)}
    proc_stat = Path(f"/proc/{pid}/stat")
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    if proc_stat.exists() and boot_id_path.exists():
        fields = proc_stat.read_text(encoding="utf-8").split()
        return {
            "kind": "linux-boot-start-ticks",
            "value": f"{boot_id_path.read_text(encoding='utf-8').strip()}:{fields[21]}",
        }
    return {"kind": "unverifiable", "value": ""}


def process_identity_matches(pid: int, expected: dict[str, Any]) -> bool:
    """Fail closed unless PID and process creation identity both match."""

    try:
        actual = process_identity(int(pid))
    except PROCESS_IDENTITY_ERRORS:
        return False
    return (
        expected.get("kind") != "unverifiable"
        and actual.get("kind") == expected.get("kind")
        and actual.get("value") == expected.get("value")
    )


def manifest_process_is_current(manifest: dict[str, Any]) -> bool:
    """Validate a manifest's PID against its recorded process identity."""

    pid = manifest.get("pid")
    identity = manifest.get("process_identity")
    return isinstance(pid, int) and isinstance(identity, dict) and process_identity_matches(
        pid,
        identity,
    )


def manifest_is_stale(
    manifest: dict[str, Any],
    heartbeat_timeout_seconds: float = 10.0,
) -> bool:
    """Return whether a nonterminal manifest is unsafe to reattach."""

    if manifest.get("status") in TERMINAL_STATUSES:
        return False
    try:
        heartbeat = datetime.fromisoformat(str(manifest["heartbeat_at"]))
    except (KeyError, TypeError, ValueError):
        return True
    age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
    return age > heartbeat_timeout_seconds or not manifest_process_is_current(manifest)


def config_fingerprint(value: dict[str, Any]) -> str:
    """Hash canonical JSON without exposing configuration contents in metadata."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _validate_envelope(data: dict[str, Any], schema: str) -> None:
    if data.get("schema") != schema:
        raise ValueError(f"Unsupported schema: {data.get('schema')!r}")
    if data.get("version") != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported {schema} version: {data.get('version')!r}; "
            f"expected {PROTOCOL_VERSION}"
        )


def _windows_process_creation_filetime(pid: int) -> int:
    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"Cannot open process {pid}")
    creation = ctypes.c_ulonglong()
    exit_time = ctypes.c_ulonglong()
    kernel_time = ctypes.c_ulonglong()
    user_time = ctypes.c_ulonglong()
    try:
        success = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        )
        if not success:
            raise OSError(ctypes.get_last_error(), f"Cannot inspect process {pid}")
        return int(creation.value)
    finally:
        kernel32.CloseHandle(handle)
