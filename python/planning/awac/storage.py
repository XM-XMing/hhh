"""Crash-safe AWAC checkpoint retention and storage-budget contracts.

The rolling policy is deliberately separate from checkpoint serialization.  A
transaction marker remains the source of truth: a generation newer than the
committed marker is never considered garbage, and deletion is attempted only
after the marker has validated the new committed artifact.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from planning.common.hashing import file_sha256


GIB = 1024 ** 3
DEFAULT_ROLLING_CHECKPOINT_RETENTION = 2
DEFAULT_MIN_FREE_DISK_BYTES = 20 * GIB
ROLLING_CHECKPOINT_NAME = "checkpoint_last"


@dataclass(frozen=True)
class CheckpointGeneration:
    """One immutable checkpoint generation and its two storage sizes."""

    generation: int
    path: Path
    apparent_bytes: int
    physical_bytes: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generation": int(self.generation),
            "path": str(self.path),
            "filename": self.path.name,
            "apparent_bytes": int(self.apparent_bytes),
            "physical_bytes": int(self.physical_bytes),
        }


def _safe_checkpoint_name(checkpoint_name: str) -> str:
    name = str(checkpoint_name).strip()
    if not name or "/" in name or "\\" in name:
        raise ValueError("checkpoint name is invalid")
    return name


def _physical_bytes(path: Path) -> int:
    stat = path.stat()
    # st_blocks is expressed in 512-byte units on POSIX.  Keeping zero for a
    # sparse file is intentional: apparent and allocated bytes are different
    # storage facts and must not be conflated by the disk guard.
    return int(getattr(stat, "st_blocks", 0)) * 512


def _generation_pattern(checkpoint_name: str) -> re.Pattern:
    name = re.escape(_safe_checkpoint_name(checkpoint_name))
    return re.compile(r"^{}\.generation-(\d{{8}})\.pt$".format(name))


def list_checkpoint_generations(
    output_dir: Path, checkpoint_name: str = ROLLING_CHECKPOINT_NAME
) -> tuple:
    """List immutable generation files in numeric order without mutating disk."""

    root = Path(output_dir).expanduser().resolve()
    name = _safe_checkpoint_name(checkpoint_name)
    if not root.is_dir():
        return tuple()
    pattern = _generation_pattern(name)
    entries = []
    for path in root.iterdir():
        match = pattern.match(path.name)
        if match is None or not path.is_file():
            continue
        generation = int(match.group(1))
        if generation <= 0:
            continue
        entries.append(
            CheckpointGeneration(
                generation=generation,
                path=path,
                apparent_bytes=int(path.stat().st_size),
                physical_bytes=_physical_bytes(path),
            )
        )
    return tuple(sorted(entries, key=lambda item: item.generation))


def _committed_manifest(output_dir: Path, checkpoint_name: str) -> Dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    name = _safe_checkpoint_name(checkpoint_name)
    path = root / "{}.transaction.json".format(name)
    if not path.is_file():
        raise FileNotFoundError("checkpoint transaction manifest is missing: {}".format(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("checkpoint transaction manifest is invalid: {}".format(path)) from exc
    if not isinstance(payload, dict):
        raise ValueError("checkpoint transaction manifest is invalid: {}".format(path))
    required = (
        "generation",
        "checkpoint_name",
        "checkpoint_filename",
        "checkpoint_sha256",
        "transaction_state",
        "checkpoint_final_commit",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(
            "checkpoint transaction manifest missing: {}".format(", ".join(missing))
        )
    if str(payload["checkpoint_name"]) != name:
        raise ValueError("checkpoint transaction manifest name mismatch")
    if str(payload["transaction_state"]) != "COMMITTED":
        raise ValueError("checkpoint transaction is not committed")
    if str(payload["checkpoint_final_commit"]) != "PASS":
        raise ValueError("checkpoint transaction final commit failed")
    generation = int(payload["generation"])
    if generation <= 0:
        raise ValueError("checkpoint transaction generation is invalid")
    filename = Path(str(payload["checkpoint_filename"]))
    if filename.name != str(payload["checkpoint_filename"]):
        raise ValueError("checkpoint transaction filename escapes output directory")
    match = _generation_pattern(name).match(filename.name)
    if match is None or int(match.group(1)) != generation:
        raise ValueError("checkpoint transaction generation filename mismatch")
    artifact = root / filename.name
    if not artifact.is_file():
        raise ValueError("committed checkpoint artifact is missing: {}".format(artifact))
    expected_sha = str(payload["checkpoint_sha256"])
    if len(expected_sha) != 64 or file_sha256(artifact) != expected_sha:
        raise ValueError("committed checkpoint artifact SHA mismatch")
    return dict(payload)


def checkpoint_retention_plan(
    output_dir: Path,
    checkpoint_name: str = ROLLING_CHECKPOINT_NAME,
    *,
    keep_latest_committed: int = DEFAULT_ROLLING_CHECKPOINT_RETENTION,
) -> Dict[str, Any]:
    """Return a dry-run retention plan for one committed rolling checkpoint.

    Only generations at or below the transaction marker are eligible.  Newer
    files are explicitly reported as preserved uncommitted tails, including a
    file left behind by a crash after the atomic checkpoint rename.
    """

    name = _safe_checkpoint_name(checkpoint_name)
    keep_count = int(keep_latest_committed)
    if keep_count <= 0:
        raise ValueError("keep_latest_committed must be positive")
    root = Path(output_dir).expanduser().resolve()
    manifest = _committed_manifest(root, name)
    committed_generation = int(manifest["generation"])
    entries = list(list_checkpoint_generations(root, name))
    committed_entries = [
        item for item in entries if item.generation <= committed_generation
    ]
    if not any(item.generation == committed_generation for item in committed_entries):
        raise ValueError("committed generation is not present on disk")
    keep_entries = committed_entries[-keep_count:]
    keep_generations = [item.generation for item in keep_entries]
    delete_entries = [
        item for item in committed_entries if item.generation not in keep_generations
    ]
    uncommitted_entries = [
        item for item in entries if item.generation > committed_generation
    ]
    return {
        "status": "PASS",
        "checkpoint_name": name,
        "committed_generation": committed_generation,
        "retention": keep_count,
        "keep_generations": keep_generations,
        "keep_files": [item.as_dict() for item in keep_entries],
        "delete_generations": [item.generation for item in delete_entries],
        "delete_candidates": [item.as_dict() for item in delete_entries],
        "delete_candidate_count": len(delete_entries),
        "delete_candidate_apparent_bytes": sum(
            item.apparent_bytes for item in delete_entries
        ),
        "delete_candidate_physical_bytes": sum(
            item.physical_bytes for item in delete_entries
        ),
        "preserved_uncommitted_generations": [
            item.generation for item in uncommitted_entries
        ],
        "preserved_uncommitted_files": [
            item.as_dict() for item in uncommitted_entries
        ],
    }


def garbage_collect_checkpoint_generations(
    output_dir: Path,
    checkpoint_name: str = ROLLING_CHECKPOINT_NAME,
    *,
    keep_latest_committed: int = DEFAULT_ROLLING_CHECKPOINT_RETENTION,
) -> Dict[str, Any]:
    """Delete only old committed rolling generations after a valid commit.

    This function intentionally accepts only ``checkpoint_last``.  Pass,
    milestone, best-dev, and final checkpoints are pinned names and therefore
    cannot be accidentally swept by this rolling GC owner.  Individual unlink
    failures become warnings so a valid transaction is never retroactively
    marked failed.
    """

    name = _safe_checkpoint_name(checkpoint_name)
    if name != ROLLING_CHECKPOINT_NAME:
        raise ValueError(
            "rolling GC cannot delete pinned checkpoint name: {}".format(name)
        )
    plan = checkpoint_retention_plan(
        output_dir,
        name,
        keep_latest_committed=keep_latest_committed,
    )
    root = Path(output_dir).expanduser().resolve()
    deleted = []
    warnings = []
    for candidate in plan["delete_candidates"]:
        path = Path(candidate["path"])
        try:
            path.unlink()
            deleted.append(int(candidate["generation"]))
        except FileNotFoundError:
            # The desired outcome is already true; retain an explicit warning
            # so operators can distinguish it from a clean unlink.
            warnings.append("missing during GC: {}".format(path))
        except OSError as exc:
            warnings.append("{}: {}".format(path, exc))
    if deleted:
        _fsync_directory(root)
    return {
        **plan,
        "status": "PASS" if not warnings else "WARNING",
        "deleted_generations": deleted,
        "gc_warning_count": len(warnings),
        "gc_warnings": warnings,
        "deleted_candidate_physical_bytes": sum(
            candidate["physical_bytes"]
            for candidate in plan["delete_candidates"]
            if int(candidate["generation"]) in deleted
        ),
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def filesystem_free_bytes(path: Path) -> int:
    """Return available bytes for the filesystem containing ``path``."""

    target = Path(path).expanduser().resolve()
    stat = os.statvfs(str(target))
    return int(stat.f_bavail) * int(stat.f_frsize)


def evaluate_disk_guard(
    *,
    free_bytes: int,
    estimated_run_bytes: int,
    min_free_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES,
) -> Dict[str, Any]:
    """Evaluate a conservative preflight budget without changing disk state."""

    free = int(free_bytes)
    estimate = int(estimated_run_bytes)
    minimum = int(min_free_bytes)
    if free < 0 or estimate < 0 or minimum < 0:
        raise ValueError("disk guard values must be non-negative")
    headroom = free - estimate
    return {
        "status": "PASS" if headroom >= minimum else "FAIL",
        "free_bytes": free,
        "estimated_run_bytes": estimate,
        "min_free_bytes": minimum,
        "headroom_bytes": headroom,
        "free_gib": float(free) / GIB,
        "estimated_run_gib": float(estimate) / GIB,
        "headroom_gib": float(headroom) / GIB,
        "min_free_gib": float(minimum) / GIB,
    }


def evaluate_disk_guard_for_path(
    path: Path,
    *,
    estimated_run_bytes: int,
    min_free_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES,
) -> Dict[str, Any]:
    """Evaluate the disk guard using the filesystem containing ``path``."""

    return evaluate_disk_guard(
        free_bytes=filesystem_free_bytes(path),
        estimated_run_bytes=estimated_run_bytes,
        min_free_bytes=min_free_bytes,
    )


__all__ = [
    "CheckpointGeneration",
    "DEFAULT_MIN_FREE_DISK_BYTES",
    "DEFAULT_ROLLING_CHECKPOINT_RETENTION",
    "GIB",
    "ROLLING_CHECKPOINT_NAME",
    "checkpoint_retention_plan",
    "evaluate_disk_guard",
    "evaluate_disk_guard_for_path",
    "filesystem_free_bytes",
    "garbage_collect_checkpoint_generations",
    "list_checkpoint_generations",
]
