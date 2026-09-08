"""Auditable hardware defaults for the supported workstation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

@dataclass(frozen=True)
class TeacherAuditHardwareProfile:
    """Measured defaults for the supported 24-core/RTX 4060 Ti workstation."""

    cpu_workers: int = 12
    collision_threads_per_worker: int = 2
    reorder_window_min: int = 32
    reorder_window_per_worker: int = 2

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

TEACHER_AUDIT_HARDWARE = TeacherAuditHardwareProfile()

@dataclass(frozen=True)
class HardwareProfile:
    """Conservative defaults for a 24-core CPU and 16 GiB NVIDIA GPU."""

    cpu_threads: int = 24
    dataloader_workers: int = 8
    dataloader_prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    allow_tf32: bool = True
    cudnn_benchmark: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

def configure_torch_runtime(torch, profile: HardwareProfile) -> Dict[str, Any]:
    """Apply GPU settings and return the exact effective runtime contract."""

    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.backends.cudnn.benchmark = bool(profile.cudnn_benchmark)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = bool(profile.allow_tf32)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(profile.allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    return {
        **profile.as_dict(),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
    }
