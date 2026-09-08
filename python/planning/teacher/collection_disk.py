"""Disk-capacity contracts for bounded Teacher collection runs.

The collector writes one episode NPZ together with journals, progress, logs,
and a final merged index.  This module keeps the capacity estimate and the
runtime threshold policy pure and testable; it never creates or removes run
artifacts.
"""

from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


GIB = 1024 ** 3
DEFAULT_ESTIMATED_EPISODE_BYTES = 1024 * 1024
DEFAULT_METADATA_OVERHEAD_FRACTION = 0.15
DEFAULT_MIN_METADATA_OVERHEAD_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class DiskSnapshot:
    """One filesystem usage observation."""

    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    monotonic_s: float

    @property
    def free_gb(self) -> float:
        return float(self.free_bytes) / float(GIB)


@dataclass(frozen=True)
class DiskCapacityEstimate:
    """Conservative additional space required by a collection target."""

    target_accepted: int
    estimated_episode_bytes: int
    sampled_episode_count: int
    episode_storage_bytes: int
    metadata_overhead_bytes: int
    safety_margin_bytes: int

    @property
    def required_bytes(self) -> int:
        return (
            int(self.episode_storage_bytes)
            + int(self.metadata_overhead_bytes)
            + int(self.safety_margin_bytes)
        )

    @property
    def required_gb(self) -> float:
        return float(self.required_bytes) / float(GIB)


@dataclass(frozen=True)
class DiskCapacityResult:
    """Result of the preflight gate, including the evidence used."""

    snapshot: DiskSnapshot
    estimate: DiskCapacityEstimate
    minimum_free_bytes: int
    required_free_bytes: int
    passed: bool

    @property
    def reason(self) -> str:
        return "PASS" if self.passed else "INSUFFICIENT_FREE_SPACE"


@dataclass(frozen=True)
class DiskWatchEvent:
    """A threshold transition observed by the runtime watcher."""

    level: str
    snapshot: DiskSnapshot
    previous_level: Optional[str]


def disk_snapshot(path: Path) -> DiskSnapshot:
    """Read usage for the filesystem containing ``path``."""

    target = Path(path).expanduser().resolve()
    usage = shutil.disk_usage(str(target))
    return DiskSnapshot(
        path=str(target),
        total_bytes=int(usage.total),
        used_bytes=int(usage.used),
        free_bytes=int(usage.free),
        monotonic_s=time.monotonic(),
    )


def _episode_candidates(roots: Iterable[Path]):
    seen = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root.is_file() and root.suffix == ".npz" and root.parent.name == "episodes":
            candidates = (root,)
        elif root.name == "episodes":
            candidates = root.glob("*.npz")
        elif root.is_dir():
            candidates = root.rglob("*.npz")
        else:
            continue
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if resolved.parent.name != "episodes" or resolved.suffix != ".npz":
                    continue
                key = str(resolved)
                if key in seen:
                    continue
                seen.add(key)
                yield resolved
            except OSError:
                continue


def sample_episode_storage_bytes(
    roots: Iterable[Path], *, limit: int = 64
) -> Tuple[int, int]:
    """Return ``(mean_bytes, sample_count)`` from existing episode NPZs.

    The scan is bounded.  If no episode is available, callers use a
    conservative default rather than silently disabling the gate.
    """

    sizes = []
    for path in _episode_candidates(roots):
        try:
            size = int(path.stat().st_size)
        except OSError:
            continue
        if size > 0:
            sizes.append(size)
        if len(sizes) >= max(1, int(limit)):
            break
    if not sizes:
        return DEFAULT_ESTIMATED_EPISODE_BYTES, 0
    return max(1, int(round(sum(sizes) / float(len(sizes))))), len(sizes)


def estimate_collection_capacity(
    target_accepted: int,
    *,
    safety_margin_gb: float,
    estimated_episode_bytes: int = 0,
    reference_roots: Iterable[Path] = (),
) -> DiskCapacityEstimate:
    """Estimate additional bytes needed before launching runtime workers."""

    target = max(0, int(target_accepted))
    explicit_bytes = int(estimated_episode_bytes)
    if explicit_bytes > 0:
        episode_bytes, sample_count = explicit_bytes, 0
    else:
        episode_bytes, sample_count = sample_episode_storage_bytes(reference_roots)
    episode_storage = target * max(1, int(episode_bytes))
    metadata_overhead = max(
        DEFAULT_MIN_METADATA_OVERHEAD_BYTES,
        int(math.ceil(episode_storage * DEFAULT_METADATA_OVERHEAD_FRACTION)),
    )
    safety_margin = max(0, int(math.ceil(float(safety_margin_gb) * GIB)))
    return DiskCapacityEstimate(
        target_accepted=target,
        estimated_episode_bytes=max(1, int(episode_bytes)),
        sampled_episode_count=int(sample_count),
        episode_storage_bytes=int(episode_storage),
        metadata_overhead_bytes=int(metadata_overhead),
        safety_margin_bytes=int(safety_margin),
    )


def disk_capacity_gate(
    path: Path,
    *,
    target_accepted: int,
    minimum_free_gb: float,
    safety_margin_gb: float,
    estimated_episode_bytes: int = 0,
    reference_roots: Iterable[Path] = (),
) -> DiskCapacityResult:
    """Evaluate capacity without starting any runtime process."""

    snapshot = disk_snapshot(path)
    estimate = estimate_collection_capacity(
        target_accepted,
        safety_margin_gb=safety_margin_gb,
        estimated_episode_bytes=estimated_episode_bytes,
        reference_roots=reference_roots,
    )
    minimum_free_bytes = max(0, int(math.ceil(float(minimum_free_gb) * GIB)))
    required_free_bytes = max(minimum_free_bytes, estimate.required_bytes)
    return DiskCapacityResult(
        snapshot=snapshot,
        estimate=estimate,
        minimum_free_bytes=minimum_free_bytes,
        required_free_bytes=required_free_bytes,
        passed=int(snapshot.free_bytes) >= int(required_free_bytes),
    )


class DiskSpaceWatch:
    """Bounded polling watcher used by the parallel collection supervisor."""

    def __init__(
        self,
        path: Path,
        *,
        warning_free_gb: float,
        stop_free_gb: float,
        interval_s: float,
    ) -> None:
        if float(stop_free_gb) <= 0.0:
            raise ValueError("stop_free_gb must be positive")
        if float(warning_free_gb) < float(stop_free_gb):
            raise ValueError("warning_free_gb must be >= stop_free_gb")
        if float(interval_s) <= 0.0:
            raise ValueError("interval_s must be positive")
        self.path = Path(path).expanduser().resolve()
        self.warning_free_bytes = int(float(warning_free_gb) * GIB)
        self.stop_free_bytes = int(float(stop_free_gb) * GIB)
        self.interval_s = float(interval_s)
        self._last_check_s = None
        self._last_level = None

    def _level(self, free_bytes: int) -> str:
        if int(free_bytes) < self.stop_free_bytes:
            return "STOP"
        if int(free_bytes) < self.warning_free_bytes:
            return "WARNING"
        return "OK"

    def check(self, *, force: bool = False) -> Optional[DiskWatchEvent]:
        now = time.monotonic()
        if (
            not force
            and self._last_check_s is not None
            and now - self._last_check_s < self.interval_s
        ):
            return None
        snapshot = disk_snapshot(self.path)
        self._last_check_s = now
        level = self._level(snapshot.free_bytes)
        previous = self._last_level
        self._last_level = level
        if level == previous:
            return None
        return DiskWatchEvent(level=level, snapshot=snapshot, previous_level=previous)


__all__ = [
    "DEFAULT_ESTIMATED_EPISODE_BYTES",
    "DiskCapacityEstimate",
    "DiskCapacityResult",
    "DiskSnapshot",
    "DiskSpaceWatch",
    "DiskWatchEvent",
    "disk_capacity_gate",
    "disk_snapshot",
    "estimate_collection_capacity",
    "sample_episode_storage_bytes",
]
