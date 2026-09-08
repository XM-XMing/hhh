"""Read-only progress monitor for a formal teacher collection directory."""

from __future__ import annotations

import time
import math
from pathlib import Path
from typing import Any, Dict, Iterable

from planning.common import read_csv, read_json


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _number(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _real(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result > 0.0 else 0.0


def _line_count(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        except OSError:
            continue
    return total


def collection_progress_snapshot(rollout_dir: Path) -> Dict[str, Any]:
    """Return best-effort progress values without mutating collection state."""

    root = Path(rollout_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("rollout directory not found: {}".format(root))
    root_summary = _read_json(root / "collection_summary.json")
    workers = sorted(path for path in (root / "workers").glob("worker_*") if path.is_dir())
    progress = [_read_json(path / "collection_progress.json") for path in workers]
    progress = [item for item in progress if item]
    expected_workers = _number(root_summary.get("workers", 0))
    if expected_workers <= 0:
        specs = _read_json(root / "worker_runtime_specs.json")
        expected_workers = _number(specs.get("worker_count", 0))
    if expected_workers <= 0:
        expected_workers = len(workers)

    attempted = _number(root_summary.get("attempted_total", 0))
    accepted = _number(root_summary.get("accepted_total", 0))
    reliable_rows = _number(root_summary.get("reliable_rows", 0))
    # Worker progress is the live source of truth while the parent summary is
    # either absent or stale.  The files are per-worker, so summing keeps the
    # aggregate counters aligned with the parallel collector.
    if progress:
        attempted = sum(_number(item.get("attempted_total", 0)) for item in progress)
        accepted = sum(_number(item.get("accepted_total", 0)) for item in progress)
        reliable_rows = sum(_number(item.get("reliable_rows", 0)) for item in progress)
    elif not root_summary:
        attempted = 0
        accepted = 0
        reliable_rows = 0
    index_rows = [dict(row) for row in read_csv(root / "rollout_index.csv")]
    persisted_transitions = sum(_number(row.get("reliable_rows", 0)) for row in index_rows)
    journal_paths = list((root / "workers").glob("worker_*/collection_report.journal.jsonl"))
    journal_paths.extend(root.glob("collection_report.journal.jsonl"))
    statuses = [str(item.get("status", "")).strip().lower() for item in progress]
    ready = sum(
        1
        for item, status in zip(progress, statuses)
        if bool(item.get("ready", False))
        or status in {"starting", "running", "completed", "finished", "stopped"}
    )
    finished = sum(
        1
        for item, status in zip(progress, statuses)
        if bool(item.get("complete", False))
        or status in {"completed", "finished", "stopped", "interrupted", "failed"}
    )
    if not progress:
        summaries = list((root / "workers").glob("worker_*/collection_summary.json"))
        ready = len(summaries)
        finished = len(summaries)
    disk_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                disk_bytes += path.stat().st_size
            except OSError:
                continue

    accepted_throughput_per_s = sum(
        _real(
            item.get(
                "accepted_throughput_per_s",
                item.get("throughput_per_s", 0.0),
            )
        )
        for item in progress
    )
    elapsed_s = max((_real(item.get("elapsed_s", 0.0)) for item in progress), default=0.0)
    target_values = [
        _number(item.get("target_accepted", 0))
        for item in progress
        if _number(item.get("target_accepted", 0)) > 0
    ]
    target_accepted = max(
        target_values or [_number(root_summary.get("target_accepted", 0))]
    )
    if target_accepted <= 0:
        target_accepted = sum(
            _number(item.get("estimated_stop", 0)) for item in progress
        )
    active_workers = max(0, expected_workers - finished) if expected_workers > 0 else 0
    mission_pool_exhausted = bool(
        expected_workers > 0
        and active_workers == 0
        and target_accepted > 0
        and accepted < target_accepted
    )
    eta_h = (
        None
        if mission_pool_exhausted
        else (
            0.0
            if target_accepted > 0 and accepted >= target_accepted
            else (
                max(0.0, target_accepted - accepted)
                / accepted_throughput_per_s
                / 3600.0
                if target_accepted > 0 and accepted_throughput_per_s > 0.0
                else None
            )
        )
    )
    if mission_pool_exhausted:
        eta_status = "TARGET_UNREACHABLE"
    elif target_accepted > 0 and accepted >= target_accepted:
        eta_status = "READY"
    elif any(
        str(item.get("eta_status", "")) == "CALIBRATING" for item in progress
    ):
        eta_status = "CALIBRATING"
    else:
        eta_status = "READY" if eta_h is not None else "UNKNOWN"

    progress_target = target_accepted if target_accepted > 0 else attempted
    progress_percent = (
        100.0 * accepted / float(progress_target) if progress_target > 0 else 0.0
    )
    live_counter = lambda key, fallback: (
        sum(_number(item.get(key, 0)) for item in progress)
        if progress
        else _number(root_summary.get(key, fallback))
    )
    live_max_counter = lambda key, fallback: (
        max((_number(item.get(key, 0)) for item in progress), default=0)
        if progress
        else _number(root_summary.get(key, fallback))
    )

    return {
        "rollout_dir": str(root),
        "attempted": attempted,
        "accepted": accepted,
        "reliable_rows": reliable_rows,
        "persisted_transitions": persisted_transitions,
        "journal_rows": _line_count(journal_paths),
        "workers_ready": "{}/{}".format(ready, expected_workers),
        "workers_finished": "{}/{}".format(finished, expected_workers),
        "active_workers": active_workers,
        "mission_pool_exhausted": mission_pool_exhausted,
        "legacy_rows": live_counter("legacy_rows", 0),
        "telemetry_lookup": live_counter("telemetry_lookup_count", 0),
        "snapshot_missing": live_counter("snapshot_missing_count", 0),
        "skew_max_ns": live_max_counter("state_depth_skew_max_ns", 0),
        "frame_failures": live_counter("frame_contract_failures", 0),
        "collector_errors": live_counter("collector_error_total", 0),
        "elapsed_s": elapsed_s,
        "accepted_throughput_per_s": accepted_throughput_per_s,
        "target_accepted": target_accepted,
        "progress_percent": progress_percent,
        "eta_status": eta_status,
        "eta_h": eta_h,
        "disk_bytes": disk_bytes,
    }


def print_collection_progress(snapshot: Dict[str, Any]) -> None:
    print("COLLECTION_MONITOR")
    for key in (
        "attempted",
        "accepted",
        "reliable_rows",
        "persisted_transitions",
        "journal_rows",
        "workers_ready",
        "workers_finished",
        "active_workers",
        "mission_pool_exhausted",
        "target_accepted",
        "progress_percent",
        "legacy_rows",
        "telemetry_lookup",
        "snapshot_missing",
        "skew_max_ns",
        "frame_failures",
        "collector_errors",
        "accepted_throughput_per_s",
        "eta_status",
        "eta_h",
        "disk_bytes",
    ):
        print("  {}: {}".format(key, snapshot[key]))


def run_collection_monitor(
    rollout_dir: Path, *, interval_sec: float, once: bool = False
) -> int:
    """Print progress until interrupted; Ctrl+C is a normal monitor exit."""

    if not once and float(interval_sec) <= 0.0:
        raise ValueError("interval_sec must be positive")
    while True:
        print_collection_progress(collection_progress_snapshot(rollout_dir))
        if once:
            return 0
        time.sleep(float(interval_sec))


__all__ = [
    "collection_progress_snapshot",
    "print_collection_progress",
    "run_collection_monitor",
]
