"""P3 reliable-v4 runtime accounting and post-process artifact binding.

This module only aggregates already-emitted evidence.  It never derives
physical execution from a command send or receipt; that value is accepted
only from Unity's execution-transport audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


RUNTIME_ACCOUNTING_FIELDS = (
    "command_accepted_total",
    "command_forward_total",
    "unity_command_receipt_total",
    "physical_execution_total",
    "command_retry_total",
    "command_duplicate_total",
    "command_conflict_total",
    "result_accepted_total",
    "result_ack_total",
    "result_commit_total",
    "snapshot_request_total",
    "snapshot_response_total",
    "snapshot_hash_match_total",
    "snapshot_cache_hit_total",
    "snapshot_missing_total",
    "pending_command_final",
    "pending_result_final",
    "pending_snapshot_final",
    "protocol_error_total",
    "telemetry_lookup_count",
)


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def _sum(metrics: Iterable[Mapping[str, Any]], *names: str) -> int:
    return sum(_as_int(item.get(name, 0)) for item in metrics for name in names)


def aggregate_runtime_accounting(
    *,
    bridge_metrics: Iterable[Mapping[str, Any]],
    unity_audits: Iterable[Mapping[str, Any]],
    telemetry_lookup_count: int,
) -> dict[str, Optional[int]]:
    """Aggregate per-worker bridge and Unity evidence without inference."""

    bridge = tuple(bridge_metrics)
    unity = tuple(unity_audits)
    physical_values = [
        _as_int(item["physical_execution_total"])
        for item in unity
        if item.get("physical_execution_total") is not None
    ]
    return {
        "command_accepted_total": _sum(bridge, "command_accepted_total"),
        "command_forward_total": _sum(bridge, "command_forward_total"),
        "unity_command_receipt_total": _sum(
            bridge, "unity_command_receipt_total"
        ),
        "physical_execution_total": (
            sum(physical_values) if physical_values else None
        ),
        "command_retry_total": _sum(bridge, "command_retry_total"),
        "command_duplicate_total": _sum(bridge, "command_duplicate_total"),
        "command_conflict_total": _sum(bridge, "command_conflict_total"),
        "result_accepted_total": _sum(bridge, "accepted"),
        "result_ack_total": _sum(bridge, "ack"),
        "result_commit_total": _sum(bridge, "commit"),
        "snapshot_request_total": _sum(bridge, "snapshot_request"),
        "snapshot_response_total": _sum(bridge, "snapshot_response"),
        "snapshot_hash_match_total": _sum(bridge, "snapshot_hash_match"),
        "snapshot_cache_hit_total": _sum(bridge, "snapshot_cache_hit"),
        "snapshot_missing_total": _sum(bridge, "snapshot_missing"),
        "pending_command_final": _sum(bridge, "pending_command_final"),
        "pending_result_final": _sum(bridge, "pending"),
        "pending_snapshot_final": _sum(bridge, "pending_snapshot"),
        "protocol_error_total": _sum(
            bridge,
            "protocol_error",
            "command_protocol_error_total",
            "snapshot_protocol_error",
        ),
        "telemetry_lookup_count": int(telemetry_lookup_count),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_training_artifacts(
    *,
    summary_path: Path,
    checkpoint_path: Optional[Path],
    accounting: Mapping[str, Any],
) -> None:
    """Attach the same immutable accounting namespace to summary/checkpoint."""

    summary = dict(_read_json(Path(summary_path)))
    normalized = {
        key: accounting.get(key) for key in RUNTIME_ACCOUNTING_FIELDS
        if key in accounting
    }
    summary.update(normalized)
    summary["runtime_accounting"] = dict(normalized)
    temporary = Path(summary_path).with_name(Path(summary_path).name + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(summary_path)

    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        return
    checkpoint = Path(checkpoint_path)
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        payload["runtime_accounting"] = dict(normalized)
        checkpoint.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        import torch

        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        payload["runtime_accounting"] = dict(normalized)
        temporary = checkpoint.with_name(checkpoint.name + ".tmp")
        torch.save(payload, str(temporary))
        temporary.replace(checkpoint)


def merge_runtime_accounting_from_directory(
    *,
    summary_path: Path,
    checkpoint_paths: Iterable[Path],
    runtime_dir: Path,
) -> dict[str, Optional[int]]:
    """Read per-worker shutdown artifacts and bind them to P3 outputs."""

    summary = dict(_read_json(Path(summary_path)))
    bridge_paths = sorted(Path(runtime_dir).rglob("bridge_result_metrics_*.json"))
    unity_paths = sorted(
        Path(runtime_dir).rglob("unity_execution_transport_audit_*.json")
    )
    bridge_metrics = [_read_json(path) for path in bridge_paths]
    unity_audits = [_read_json(path) for path in unity_paths]
    accounting = aggregate_runtime_accounting(
        bridge_metrics=bridge_metrics,
        unity_audits=unity_audits,
        telemetry_lookup_count=int(summary.get("telemetry_lookup_count", 0)),
    )
    checkpoints = tuple(Path(path) for path in checkpoint_paths)
    for path in checkpoints:
        merge_training_artifacts(
            summary_path=Path(summary_path),
            checkpoint_path=path if path.is_file() else None,
            accounting=accounting,
        )
    if not checkpoints:
        merge_training_artifacts(
            summary_path=Path(summary_path),
            checkpoint_path=None,
            accounting=accounting,
        )
    write_path = Path(summary_path).with_name("runtime_accounting.json")
    write_path.write_text(
        json.dumps(accounting, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return accounting


__all__ = [
    "RUNTIME_ACCOUNTING_FIELDS",
    "aggregate_runtime_accounting",
    "merge_training_artifacts",
    "merge_runtime_accounting_from_directory",
]
