from __future__ import annotations

import argparse
import math
import time

from engine.training_telemetry import TelemetryCadence


def _expensive_sample() -> float:
    return sum(math.sqrt(index) for index in range(200))


def run(steps: int) -> tuple[float, float, int]:
    start = time.perf_counter()
    for _ in range(steps):
        _expensive_sample()
    always_seconds = time.perf_counter() - start

    clock = [0.0]
    cadence = TelemetryCadence(1.0, 15.0, 30.0, clock=lambda: clock[0])
    sampled = 0
    start = time.perf_counter()
    for _ in range(steps):
        clock[0] += 0.01
        if cadence.sample().stability:
            _expensive_sample()
            sampled += 1
    cadence_seconds = time.perf_counter() - start
    return always_seconds, cadence_seconds, sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare always-on telemetry work with time-based cadence selection"
    )
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()
    always_seconds, cadence_seconds, sampled = run(max(1, args.steps))
    print(f"steps={args.steps}")
    print(f"always_expensive_seconds={always_seconds:.6f}")
    print(f"cadenced_seconds={cadence_seconds:.6f}")
    print(f"cadenced_expensive_samples={sampled}")
    print(f"wall_clock_speedup={always_seconds / max(cadence_seconds, 1e-12):.2f}x")


if __name__ == "__main__":
    main()
