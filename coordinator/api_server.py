from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import unquote

from engine.contracts import (
    ClaimJobRequest,
    CompleteJobRequest,
    FailJobRequest,
    HeartbeatRequest,
    ProgressReportRequest,
    RegisterWorkerRequest,
)
from engine.coordinator.job_manager import JobManager
from engine.coordinator.artifacts import default_artifact_root


class CoordinatorApiServer:
    """HTTP API wrapper around the job manager."""

    def __init__(
        self,
        manager: Optional[JobManager] = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        artifact_root: Optional[Path] = None,
    ) -> None:
        """Create a coordinator API server.

        Args:
            manager: Job manager instance.
            host: Host address to bind.
            port: TCP port to bind.
            artifact_root: Root folder served by the artifact endpoint.
        """

        self.manager = manager or JobManager()
        self.host = host
        self.port = port
        self.artifact_root = Path(artifact_root) if artifact_root else default_artifact_root()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.httpd: Optional[ThreadingHTTPServer] = None
        self._manager_lock = threading.RLock()

    def serve_forever(self) -> None:
        """Start serving coordinator API requests."""

        handler = self._handler_class()
        self.httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        """Stop the coordinator API server."""

        if self.httpd is not None:
            self.httpd.shutdown()

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        """Create a request handler bound to this server.

        Returns:
            HTTP request handler class.
        """

        api = self

        class CoordinatorRequestHandler(BaseHTTPRequestHandler):
            """HTTP request handler for coordinator protocol routes."""

            server_version = "MicroLLMCoordinator/0.1"

            def do_GET(self) -> None:
                """Handle GET requests."""

                if self.path.startswith("/artifacts/"):
                    self._send_artifact(self.path.removeprefix("/artifacts/"))
                    return
                routes: dict[str, Callable[[], dict[str, Any]]] = {
                    "/health": api._health,
                    "/workers": api._workers,
                    "/jobs": api._jobs,
                }
                handler = routes.get(self.path)
                if handler is None:
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                with api._manager_lock:
                    self._send_json(handler())

            def do_POST(self) -> None:
                """Handle POST requests."""

                routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
                    "/register": api._register,
                    "/heartbeat": api._heartbeat,
                    "/claim-job": api._claim_job,
                    "/progress": api._progress,
                    "/complete": api._complete,
                    "/fail": api._fail,
                    "/pause-all": api._pause_all,
                    "/resume-all": api._resume_all,
                    "/stop-all": api._stop_all,
                    "/stale-workers": api._stale_workers,
                }
                handler = routes.get(self.path)
                if handler is None:
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                try:
                    payload = self._read_json()
                    with api._manager_lock:
                        response = handler(payload)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                except Exception as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                self._send_json(response)

            def do_PUT(self) -> None:
                """Handle artifact upload requests."""

                if not self.path.startswith("/artifacts/"):
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._receive_artifact(self.path.removeprefix("/artifacts/"))

            def log_message(self, format: str, *args: Any) -> None:
                """Silence default stderr logging."""

            def _read_json(self) -> dict[str, Any]:
                """Read JSON payload from the request.

                Returns:
                    Request payload.

                Raises:
                    ValueError: If the payload is not valid JSON.
                """

                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length <= 0:
                    return {}
                raw = self.rfile.read(content_length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON: {exc}") from exc
                if not isinstance(payload, dict):
                    raise ValueError("JSON payload must be an object")
                return payload

            def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
                """Send a JSON response.

                Args:
                    payload: Response payload.
                    status: HTTP status.
                """

                body = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_artifact(self, relative_url_path: str) -> None:
                """Send an artifact file.

                Args:
                    relative_url_path: URL path relative to the artifact root.
                """

                relative_path = Path(unquote(relative_url_path))
                try:
                    resolved = (api.artifact_root / relative_path).resolve()
                    root = api.artifact_root.resolve()
                    if root not in resolved.parents and resolved != root:
                        raise ValueError("Artifact path escapes artifact root")
                except ValueError:
                    self._send_json({"error": "invalid artifact path"}, HTTPStatus.BAD_REQUEST)
                    return
                if not resolved.is_file():
                    self._send_json({"error": "artifact not found"}, HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
                self.send_response(int(HTTPStatus.OK))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(resolved.stat().st_size))
                self.end_headers()
                with resolved.open("rb") as artifact:
                    while chunk := artifact.read(1024 * 1024):
                        self.wfile.write(chunk)

            def _receive_artifact(self, relative_url_path: str) -> None:
                """Receive an uploaded artifact file.

                Args:
                    relative_url_path: URL path relative to the artifact root.
                """

                relative_path = Path(unquote(relative_url_path))
                try:
                    resolved = (api.artifact_root / relative_path).resolve()
                    root = api.artifact_root.resolve()
                    if root not in resolved.parents and resolved != root:
                        raise ValueError("Artifact path escapes artifact root")
                except ValueError:
                    self._send_json({"error": "invalid artifact path"}, HTTPStatus.BAD_REQUEST)
                    return
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length <= 0:
                    self._send_json({"error": "empty artifact upload"}, HTTPStatus.BAD_REQUEST)
                    return
                resolved.parent.mkdir(parents=True, exist_ok=True)
                remaining = content_length
                with resolved.open("wb") as artifact:
                    while remaining > 0:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        artifact.write(chunk)
                        remaining -= len(chunk)
                if remaining:
                    self._send_json({"error": "incomplete artifact upload"}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_json({"status": "ok", "artifact_url": f"/artifacts/{relative_path.as_posix()}"})

        return CoordinatorRequestHandler

    def _health(self) -> dict[str, Any]:
        """Return server health.

        Returns:
            Health payload.
        """

        return {"status": "ok", "workers": len(self.manager.list_workers()), "jobs": len(self.manager.list_jobs())}

    def _workers(self) -> dict[str, Any]:
        """Return workers.

        Returns:
            Worker payload.
        """

        return {"workers": [worker.to_jsonable() for worker in self.manager.list_workers()]}

    def _jobs(self) -> dict[str, Any]:
        """Return jobs.

        Returns:
            Job payload.
        """

        return {"jobs": [job.to_jsonable() for job in self.manager.list_jobs()]}

    def _register(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Register a worker.

        Args:
            payload: Register worker request payload.

        Returns:
            Register worker response payload.
        """

        return self.manager.register_remote_worker(RegisterWorkerRequest.from_jsonable(payload)).to_jsonable()

    def _heartbeat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle worker heartbeat.

        Args:
            payload: Heartbeat request payload.

        Returns:
            Heartbeat response payload.
        """

        return self.manager.handle_heartbeat(HeartbeatRequest.from_jsonable(payload)).to_jsonable()

    def _claim_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle job claim.

        Args:
            payload: Claim job request payload.

        Returns:
            Claim job response payload.
        """

        return self.manager.handle_claim_job(ClaimJobRequest.from_jsonable(payload)).to_jsonable()

    def _progress(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle progress report.

        Args:
            payload: Progress report request payload.

        Returns:
            Progress response payload.
        """

        return self.manager.handle_progress_report(ProgressReportRequest.from_jsonable(payload)).to_jsonable()

    def _complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle job completion.

        Args:
            payload: Complete job request payload.

        Returns:
            Completion response payload.
        """

        return self.manager.handle_complete_job(CompleteJobRequest.from_jsonable(payload)).to_jsonable()

    def _fail(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle job failure.

        Args:
            payload: Fail job request payload.

        Returns:
            Failure response payload.
        """

        return self.manager.handle_fail_job(FailJobRequest.from_jsonable(payload)).to_jsonable()

    def _pause_all(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Pause all jobs.

        Args:
            payload: Ignored payload.

        Returns:
            Pause summary.
        """

        return {"paused": self.manager.pause_all_jobs()}

    def _resume_all(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Resume all jobs.

        Args:
            payload: Ignored payload.

        Returns:
            Resume summary.
        """

        return {"resumed": self.manager.resume_all_jobs()}

    def _stop_all(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Stop all jobs.

        Args:
            payload: Ignored payload.

        Returns:
            Stop summary.
        """

        return {"stopping": self.manager.stop_all_jobs()}

    def _stale_workers(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Mark stale workers offline.

        Args:
            payload: Payload with optional timeout_seconds.

        Returns:
            Stale worker summary.
        """

        timeout_seconds = int(payload.get("timeout_seconds", 30))
        return {"offline_workers": self.manager.mark_stale_workers_offline(timeout_seconds=timeout_seconds)}


def run_coordinator_api(host: str = "127.0.0.1", port: int = 8765, artifact_root: Optional[Path] = None) -> None:
    """Run the coordinator API server.

    Args:
        host: Host address to bind.
        port: TCP port to bind.
        artifact_root: Root folder served by the artifact endpoint.
    """

    CoordinatorApiServer(host=host, port=port, artifact_root=artifact_root).serve_forever()

