from __future__ import annotations

from engine.backends.base import TrainerBackend
from engine.backends.local_backend import LocalTrainerBackend
from engine.contracts import BackendKind


class BackendRegistry:
    """Registry for training backends."""

    def __init__(self) -> None:
        """Create a registry with built-in local backend."""

        self._backends: dict[BackendKind, TrainerBackend] = {
            BackendKind.LOCAL: LocalTrainerBackend(),
        }

    def register(self, kind: BackendKind, backend: TrainerBackend) -> None:
        """Register a backend implementation.

        Args:
            kind: Backend kind.
            backend: Backend implementation.
        """

        self._backends[kind] = backend

    def get(self, kind: BackendKind) -> TrainerBackend:
        """Return a backend by kind.

        Args:
            kind: Backend kind.

        Returns:
            Backend implementation.

        Raises:
            ValueError: If the backend kind is not registered.
        """

        backend = self._backends.get(kind)
        if backend is None:
            raise ValueError(f"No training backend registered for {kind.value}")
        return backend


DEFAULT_BACKEND_REGISTRY = BackendRegistry()

