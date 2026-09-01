"""Engine-owned GPU discovery for standalone and remote workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GPUDevice:
    """One CUDA device visible to the engine process."""

    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: str
    supports_bf16: bool

    def to_jsonable(self) -> dict[str, Any]:
        """Return a protocol-safe GPU record."""

        return asdict(self)


def discover_gpus() -> list[GPUDevice]:
    """Return CUDA devices using only engine-owned PyTorch facilities."""

    if not torch.cuda.is_available():
        return []
    devices: list[GPUDevice] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        major, minor = torch.cuda.get_device_capability(index)
        devices.append(
            GPUDevice(
                index=index,
                name=properties.name,
                vram_total_gb=total_bytes / (1024**3),
                vram_free_gb=free_bytes / (1024**3),
                compute_capability=f"{major}.{minor}",
                supports_bf16=major >= 8,
            )
        )
    return devices


__all__ = ["GPUDevice", "discover_gpus"]
