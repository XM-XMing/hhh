"""Canonical on-disk identity and paths for one collection worker shard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class RolloutShardIdentity:
    """The worker/run identity that owns one immutable shard directory."""

    worker_id: int
    runtime_instance_id: str
    collection_run_id: str

    def __post_init__(self) -> None:
        if int(self.worker_id) < 0:
            raise ValueError("worker_id must be non-negative")
        if not str(self.runtime_instance_id).strip():
            raise ValueError("runtime_instance_id is missing")
        if not str(self.collection_run_id).strip():
            raise ValueError("collection_run_id is missing")


def worker_shard_dir(workers_dir: Path, worker_id: int) -> Path:
    """Resolve the only supported worker shard directory spelling."""

    parsed_worker_id = int(worker_id)
    if parsed_worker_id < 0:
        raise ValueError("worker_id must be non-negative")
    return Path(workers_dir).expanduser().resolve() / "worker_{:02d}".format(
        parsed_worker_id
    )


def worker_shard_paths(workers_dir: Path, worker_id: int) -> Dict[str, Path]:
    """Return stable report/config/index paths without creating them."""

    root = worker_shard_dir(workers_dir, worker_id)
    return {
        "root": root,
        "report": root / "collection_report.csv",
        "summary": root / "collection_summary.json",
        "resolved_config": root / "resolved_collection_config.json",
        "rollout_index": root / "rollout_index.csv",
        "partial_rollout_index": root / "rollout_index.partial.csv",
        "episodes": root / "episodes",
    }


__all__ = [
    "RolloutShardIdentity",
    "worker_shard_dir",
    "worker_shard_paths",
]
