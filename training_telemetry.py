from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional


@dataclass(frozen=True)
class TelemetrySample:
    """Telemetry work selected for the current optimizer step."""

    metrics: bool
    stability: bool
    preview: bool
    sampled_at: float


class TelemetryCadence:
    """Select telemetry work using wall-clock intervals rather than step counts."""

    def __init__(
        self,
        metrics_interval_seconds: float = 1.0,
        stability_interval_seconds: float = 15.0,
        preview_interval_seconds: float = 30.0,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.metrics_interval_seconds = max(0.0, float(metrics_interval_seconds))
        self.stability_interval_seconds = max(0.0, float(stability_interval_seconds))
        self.preview_interval_seconds = max(0.0, float(preview_interval_seconds))
        self._clock = clock
        self._last_metrics_at: Optional[float] = None
        self._last_stability_at: Optional[float] = None
        self._last_preview_at: Optional[float] = None

    def sample(self) -> TelemetrySample:
        """Return which telemetry categories are due at the current time."""

        now = self._clock()
        metrics = self._due(self._last_metrics_at, self.metrics_interval_seconds, now)
        stability = metrics and self._due(
            self._last_stability_at,
            self.stability_interval_seconds,
            now,
        )
        preview = metrics and self._due(self._last_preview_at, self.preview_interval_seconds, now)
        if metrics:
            self._last_metrics_at = now
        if stability:
            self._last_stability_at = now
        if preview:
            self._last_preview_at = now
        return TelemetrySample(metrics=metrics, stability=stability, preview=preview, sampled_at=now)

    @staticmethod
    def _due(last_at: Optional[float], interval: float, now: float) -> bool:
        return last_at is None or interval == 0.0 or now - last_at >= interval
