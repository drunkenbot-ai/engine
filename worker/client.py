"""Compatibility facade for remote worker clients."""

from .client_core import CoordinatorHttpClient, WorkerClientConfig
from .client_impl import RemoteWorkerClient, detect_worker_capabilities, run_worker_client

__all__ = ["CoordinatorHttpClient", "WorkerClientConfig", "RemoteWorkerClient", "detect_worker_capabilities", "run_worker_client"]

