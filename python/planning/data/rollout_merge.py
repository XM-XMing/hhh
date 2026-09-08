#!/usr/bin/env python3
"""Merge disjoint multi-worker online-teacher rollout indexes."""

from __future__ import annotations

from pathlib import Path

import argparse
import csv
import json
import os
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Set

from planning.contracts.collection import (
    canonical_sha256,
    merge_compatibility_sha256,
)
from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    validate_reliable_exact_metadata,
)
from planning.contracts.task import validate_task_contract
from planning.contracts.pipeline_provenance import (
    validate_episode_provenance,
    validate_collection_row,
    validate_collection_summary,
)
from planning.data.rollout import load_rollout_episode, row_episode_id
from planning.common.config import parse_bool
from planning.common import read_csv, write_csv_atomic, write_json_atomic

TEACHER_COLLECTION_QUALITY_CONTRACT_ID = "teacher_collection_quality_gate"


def is_route_unavailable_rejection(row) -> bool:
    stop_reason = str(row.get("stop_reason", "")).strip()
    if stop_reason == "mission_route_unavailable":
        return True
    if stop_reason != "collector_error":
        return False
    error = str(row.get("error", "")).strip()
    return error.startswith("GlobalRouteUnavailableError(") or error.startswith(
        "RuntimeError('no global route "
    ) or error.startswith("RuntimeError('no free global-route cell ")


def numeric_max(rows, field: str, default: float = 0.0) -> float:
    values = [
        float(row[field])
        for row in rows
        if str(row.get(field, "")).strip()
    ]
    return max(values) if values else float(default)


def _normalise_artifact_reference(
    value: Any,
    *,
    source_dir: Path,
    output_dir: Path,
) -> str:
    """Rebase a worker-relative artifact reference for a merged index."""

    raw = str(value or "").strip()
    if not raw:
        return raw
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(source_dir).expanduser().resolve() / path
    path = path.resolve()
    return os.path.relpath(str(path), str(Path(output_dir).expanduser().resolve()))


def _integer_field(mapping: Dict[str, Any], field: str, *, path: str) -> int:
    try:
        value = int(mapping.get(field, 0))
    except (TypeError, ValueError) as error:
        raise RuntimeError("{} has invalid {}".format(path, field)) from error
    if value < 0:
        raise RuntimeError("{} has negative {}".format(path, field))
    return value


def _merged_artifact_path(value: Any, *, out_dir: Path) -> Path:
    """Resolve a merged artifact reference against the merged rollout root."""

    path = Path(str(value or "").strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (out_dir / path).resolve()


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _row_transition_ids(row: Dict[str, Any]) -> List[str]:
    raw = row.get("transition_ids", "")
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        text = str(raw).strip()
        if not text:
            return []
        values = json.loads(text)
    if not isinstance(values, list):
        raise ValueError("transition_ids must be a list")
    return [str(value).strip() for value in values]


def _validate_episode_against_row(
    row: Dict[str, Any],
    *,
    episode: Dict[str, Any],
    episode_path: Path,
) -> None:
    """Verify the accepted NPZ is the exact artifact described by its row."""

    metadata = dict(episode.get("metadata", {}))
    episode_id = row_episode_id(row)
    reliable_rows = _integer_field(row, "reliable_rows", path=str(episode_path))
    validate_episode_provenance(
        metadata,
        path=str(episode_path),
        expected_episode_id=episode_id,
        expected_transition_count=reliable_rows,
    )
    for field in (
        "collection_run_id",
        "runtime_instance_id",
        "mpl_contract_sha256",
        "mission_index_sha256",
        "collision_cache_sha256",
        "code_version_sha256",
        "resolved_config_sha256",
        "task_contract_id",
        "task_contract_schema_version",
        "task_contract_sha256",
        "max_primitive_steps",
    ):
        row_value = str(row.get(field, "")).strip()
        metadata_value = str(metadata.get(field, "")).strip()
        if row_value and row_value != metadata_value:
            raise ValueError(
                "{} {} disagrees with collection row".format(
                    episode_path, field
                )
            )
    row_transition_ids = _row_transition_ids(row)
    metadata_transition_ids = [
        str(value).strip() for value in metadata.get("transition_ids", [])
    ]
    if row_transition_ids != metadata_transition_ids:
        raise ValueError(
            "{} transition identity list disagrees with collection row".format(
                episode_path
            )
        )
    if int(np.asarray(episode["behavior_actions"]).shape[0]) != reliable_rows:
        raise ValueError("{} reliable row count disagrees with collection row".format(episode_path))


def _accepted_dataset_validation(
    accepted: List[Dict[str, Any]],
    *,
    out_dir: Path,
    artifact_roots: Iterable[Path],
    target_accepted: int,
) -> Dict[str, Any]:
    """Validate clean accepted NPZs and select canonical rows for publication."""

    clean_candidates = [
        row
        for row in accepted
        if not any(
            parse_bool(row.get(field, False))
            for field in (
                "collision",
                "dead_end",
                "hard_altitude",
                "path_length_exceeded",
            )
        )
    ]
    valid_rows: List[Dict[str, Any]] = []
    valid_paths: Set[Path] = set()
    invalid_count = 0
    missing_count = 0
    contract_mismatch_count = 0
    errors: List[str] = []
    all_accepted_paths: Set[Path] = set()
    for row in accepted:
        raw_path = str(row.get("dataset_npz", "")).strip()
        if raw_path:
            all_accepted_paths.add(_merged_artifact_path(raw_path, out_dir=out_dir))
    allowed_roots = tuple(Path(root).resolve() for root in artifact_roots)
    for row in clean_candidates:
        raw_path = str(row.get("dataset_npz", "")).strip()
        if not raw_path:
            missing_count += 1
            errors.append("episode {} missing dataset_npz".format(row_episode_id(row)))
            continue
        episode_path = _merged_artifact_path(raw_path, out_dir=out_dir)
        if not any(_path_is_under(episode_path, root) for root in allowed_roots) or not episode_path.is_file():
            missing_count += 1
            errors.append("episode {} missing NPZ {}".format(row_episode_id(row), episode_path))
            continue
        try:
            episode = load_rollout_episode(episode_path, validate=True)
            _validate_episode_against_row(
                row, episode=episode, episode_path=episode_path
            )
        except Exception as error:
            invalid_count += 1
            message = "episode {} invalid NPZ {}: {}".format(
                row_episode_id(row), episode_path, error
            )
            errors.append(message)
            if "observation" in str(error) or "contract" in str(error):
                contract_mismatch_count += 1
            continue
        valid_rows.append(row)
        valid_paths.add(episode_path)
    valid_rows.sort(key=row_episode_id)
    if int(target_accepted) > 0:
        selected_rows = valid_rows[: int(target_accepted)]
    else:
        selected_rows = list(valid_rows)
    target_met = int(target_accepted) <= 0 or len(valid_rows) >= int(target_accepted)
    quality_gates = {
        "accepted_nonzero": {
            "observed": len(valid_rows),
            "operator": ">",
            "threshold": 0,
            "pass": len(valid_rows) > 0,
        },
        "target_accepted": {
            "observed": len(valid_rows),
            "operator": ">=",
            "threshold": int(target_accepted),
            "pass": target_met,
        },
        "accepted_npz_valid": {
            "observed": int(invalid_count),
            "operator": "==",
            "threshold": 0,
            "pass": invalid_count == 0,
        },
        "accepted_npz_present": {
            "observed": int(missing_count),
            "operator": "==",
            "threshold": 0,
            "pass": missing_count == 0,
        },
        "observation_contract_exact": {
            "observed": int(contract_mismatch_count),
            "operator": "==",
            "threshold": 0,
            "pass": contract_mismatch_count == 0,
        },
    }
    return {
        "clean_candidates": clean_candidates,
        "valid_rows": valid_rows,
        "selected_rows": selected_rows,
        "all_accepted_paths": all_accepted_paths,
        "valid_paths": valid_paths,
        "invalid_count": invalid_count,
        "missing_count": missing_count,
        "contract_mismatch_count": contract_mismatch_count,
        "errors": errors,
        "quality_gates": quality_gates,
        "quality_pass": all(
            bool(gate["pass"]) for gate in quality_gates.values()
        ),
    }


def _safe_unlink(path: Path, *, root: Path) -> bool:
    resolved = path.resolve()
    if not _path_is_under(resolved, root) or not (resolved.is_file() or resolved.is_symlink()):
        return False
    resolved.unlink()
    return True


def _cleanup_failed_artifacts(
    *,
    worker_dirs: List[Path],
    out_dir: Path,
    artifact_root: Path,
    reports: List[Dict[str, Any]],
    accepted_paths: Set[Path],
) -> Dict[str, int]:
    """Remove only artifacts that cannot belong to a valid accepted attempt."""

    failed_attempt_npz_removed = 0
    orphan_artifact_removed_count = 0
    temp_partial_file_count = 0
    nonaccepted_paths = {
        _merged_artifact_path(row.get("dataset_npz"), out_dir=out_dir)
        for row in reports
        if not parse_bool(row.get("execute_ok", False))
        and str(row.get("dataset_npz", "")).strip()
    }
    for path in sorted(nonaccepted_paths):
        if path in accepted_paths:
            continue
        if _safe_unlink(path, root=artifact_root):
            failed_attempt_npz_removed += 1
    for worker_dir in worker_dirs:
        episodes_dir = worker_dir / "episodes"
        if episodes_dir.is_dir():
            for path in sorted(episodes_dir.glob("*.npz")):
                if path.resolve() not in accepted_paths and _safe_unlink(
                    path, root=artifact_root
                ):
                    orphan_artifact_removed_count += 1
        for path in sorted(worker_dir.rglob("*")):
            if not path.is_file():
                continue
            name = path.name.lower()
            if not (
                name.endswith((".tmp", ".partial", ".incomplete", ".part"))
                or ".tmp." in name
                or ".partial." in name
                or ".incomplete." in name
            ):
                continue
            if _safe_unlink(path, root=artifact_root):
                temp_partial_file_count += 1
                orphan_artifact_removed_count += 1
    root_partial = out_dir / "rollout_index.partial.csv"
    if _safe_unlink(root_partial, root=artifact_root):
        temp_partial_file_count += 1
        orphan_artifact_removed_count += 1
    return {
        "failed_attempt_npz_removed": failed_attempt_npz_removed,
        "orphan_artifact_removed_count": orphan_artifact_removed_count,
        "temp_partial_file_count": temp_partial_file_count,
    }


def _assert_raw_report_unchanged(
    path: Path, *, rows: List[Dict[str, Any]], fieldnames: List[str]
) -> None:
    """Never rewrite a raw report that predates accepted-only finalization."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != list(fieldnames):
            raise RuntimeError("existing raw collection report header mismatch")
        existing_rows = list(reader)
    if existing_rows != rows:
        raise RuntimeError("existing raw collection report would be modified")


def _run(args) -> int:
    workers_dir = Path(args.workers_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    worker_dirs = sorted(path for path in workers_dir.glob("worker_*") if path.is_dir())
    if not worker_dirs:
        raise RuntimeError("no worker directories found: {}".format(workers_dir))
    if int(args.expected_workers) > 0 and len(worker_dirs) != int(args.expected_workers):
        raise RuntimeError("worker count {} != expected {}".format(len(worker_dirs), args.expected_workers))

    reports = []
    accepted = []
    seen_ids = set()
    seen_mission_ids: Set[str] = set()
    seen_transition_ids: Set[str] = set()
    worker_summaries = []
    formal_mode: Optional[bool] = None
    for worker_dir in worker_dirs:
        report_path = worker_dir / "collection_report.csv"
        summary_path = worker_dir / "collection_summary.json"
        config_path = worker_dir / "resolved_collection_config.json"
        rows = read_csv(report_path)
        if not rows:
            raise RuntimeError("missing or empty report: {}".format(report_path))
        if not summary_path.exists():
            raise RuntimeError("missing worker summary: {}".format(summary_path))
        if not config_path.exists():
            raise RuntimeError(
                "missing worker resolved config: {}".format(config_path)
            )
        worker_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        worker_config = json.loads(config_path.read_text(encoding="utf-8"))
        worker_is_formal = (
            str(worker_summary.get("observation_contract", "")).strip()
            == EXACT_ENDPOINT_OBSERVATION_CONTRACT
        )
        if formal_mode is None:
            formal_mode = worker_is_formal
        elif worker_is_formal != formal_mode:
            raise RuntimeError(
                "formal and legacy collection shards cannot be merged together"
            )
        if worker_is_formal:
            try:
                validate_collection_summary(
                    worker_summary,
                    path="{}.collection_summary.json".format(worker_dir),
                )
            except ValueError as error:
                raise RuntimeError(str(error)) from error
            shared_config = worker_config.get("shared_collection_config")
            if not isinstance(shared_config, dict):
                raise RuntimeError(
                    "{} shared_collection_config is not a mapping".format(worker_dir)
                )
            try:
                validate_reliable_exact_metadata(
                    shared_config,
                    path="{}.shared_collection_config".format(worker_dir),
                )
            except ValueError as error:
                raise RuntimeError(str(error)) from error
            try:
                validate_task_contract(
                    shared_config,
                    expected_max_primitive_steps=worker_summary.get(
                        "max_steps", worker_summary.get("max_primitive_steps")
                    ),
                    path="{}.shared_collection_config task contract".format(
                        worker_dir
                    ),
                )
            except ValueError as error:
                raise RuntimeError(str(error)) from error
            if worker_config.get("merge_compatibility_sha256") != merge_compatibility_sha256(
                shared_config
            ):
                raise RuntimeError(
                    "{} merge compatibility hash mismatch".format(worker_dir)
                )
            try:
                validate_task_contract(
                    worker_config,
                    expected_max_primitive_steps=worker_summary.get(
                        "max_steps", worker_summary.get("max_primitive_steps")
                    ),
                    path="{}.resolved_collection_config task contract".format(
                        worker_dir
                    ),
                )
            except ValueError as error:
                raise RuntimeError(str(error)) from error
        for field in (
            "collection_config_contract_id",
            "resolved_config_sha256",
            "shared_collection_config",
            "resolved_cli_args",
        ) + (("merge_compatibility_sha256",) if worker_is_formal else ()):
            if worker_summary.get(field) != worker_config.get(field):
                raise RuntimeError(
                    "worker summary/config mismatch for {} in {}".format(
                        field, worker_dir.name
                    )
                )
        computed_config_sha256 = canonical_sha256(
            worker_config["shared_collection_config"]
        )
        if computed_config_sha256 != str(
            worker_config["resolved_config_sha256"]
        ):
            raise RuntimeError(
                "worker resolved config hash mismatch in {}".format(
                    worker_dir.name
                )
            )
        worker_summaries.append(worker_summary)
        for row in rows:
            identifier = row_episode_id(row)
            if identifier in seen_ids:
                raise RuntimeError("duplicate episode_id {}".format(identifier))
            seen_ids.add(identifier)
            if worker_is_formal:
                row_path = "{}.collection_report.csv episode {}".format(
                    worker_dir, identifier
                )
                try:
                    transition_ids = validate_collection_row(
                        row,
                        worker_summary=worker_summary,
                        path=row_path,
                    )
                except ValueError as error:
                    raise RuntimeError(str(error)) from error
                mission_id = str(row.get("mission_id", "")).strip()
                if mission_id in seen_mission_ids:
                    raise RuntimeError("duplicate mission_id {}".format(mission_id))
                seen_mission_ids.add(mission_id)
                duplicate_transitions = seen_transition_ids.intersection(
                    transition_ids
                )
                if duplicate_transitions:
                    raise RuntimeError(
                        "duplicate transition_id(s) {}".format(
                            sorted(duplicate_transitions)
                        )
                    )
                seen_transition_ids.update(transition_ids)
            normalized = dict(row)
            raw_dataset = str(normalized.get("dataset_npz", "")).strip()
            if raw_dataset:
                normalized["dataset_npz"] = _normalise_artifact_reference(
                    raw_dataset,
                    source_dir=report_path.parent,
                    output_dir=out_dir,
                )
            raw_route_store = str(
                normalized.get("mission_route_store", "")
            ).strip()
            if raw_route_store:
                normalized["mission_route_store"] = _normalise_artifact_reference(
                    raw_route_store,
                    source_dir=report_path.parent,
                    output_dir=out_dir,
                )
            reports.append(normalized)
            if parse_bool(normalized.get("execute_ok", False)):
                accepted.append(normalized)

    reports.sort(key=row_episode_id)
    accepted.sort(key=row_episode_id)
    fieldnames = list(reports[0].keys())
    shared_contract_fields = (
        "teacher_planning_contract_id",
        "teacher_config",
        "mpl_duration_s",
        "mpl_forward_distance_m",
        "mpl_contract_sha256",
        "offline_relabel_required",
        "asynchronous_prefetch",
        "teacher_path_length_contract_id",
        "max_actual_path_length_m",
        "max_steps",
        "stream_horizon",
        "max_sensor_skew_ms",
        "max_endpoint_error_m",
        "max_stream_drift_m",
        "collection_config_contract_id",
        "resolved_config_sha256",
        "mission_index_sha256",
        "collision_cache_sha256",
        "code_version_sha256",
        "shared_collection_config",
    )
    if formal_mode:
        shared_contract_fields += (
            "task_contract_id",
            "task_contract_schema_version",
            "task_contract_sha256",
            "max_primitive_steps",
            "collection_run_id",
            "observation_contract",
            "observation_source",
            "reliable_execution",
            "telemetry_observation",
            "protocol_version",
            "state_depth_skew_ns",
            "endpoint_identity_available",
            "asynchronous_prefetch_status",
            "merge_compatibility_sha256",
            "unity_player_sha256",
            "runtime_assembly_sha256",
            "bridge_sha256",
            "runtime_artifact_identity",
        )
    shared_contract = {}
    reference_summary = worker_summaries[0]
    for field in shared_contract_fields:
        if field not in reference_summary:
            raise RuntimeError("worker summary missing contract field: {}".format(field))
        reference_value = reference_summary[field]
        mismatched_workers = [
            index
            for index, worker_summary in enumerate(worker_summaries)
            if worker_summary.get(field) != reference_value
        ]
        if mismatched_workers:
            raise RuntimeError(
                "worker summary contract mismatch for {}: workers {}".format(
                    field, mismatched_workers
                )
            )
        shared_contract[field] = reference_value
    for hash_field in (
        "resolved_config_sha256",
        "mission_index_sha256",
        "collision_cache_sha256",
        "code_version_sha256",
    ):
        value = str(shared_contract[hash_field])
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise RuntimeError(
                "worker summary has invalid {}: {!r}".format(hash_field, value)
            )
        if hash_field != "resolved_config_sha256" and (
            str(
                shared_contract["shared_collection_config"].get(
                    hash_field, ""
                )
            )
            != value
        ):
            raise RuntimeError(
                "{} does not match shared resolved config".format(hash_field)
            )
    if shared_contract["shared_collection_config"].get(
        "collection_config_contract_id"
    ) != shared_contract["collection_config_contract_id"]:
        raise RuntimeError(
            "collection config contract id does not match shared config"
        )
    shared_cli = shared_contract["shared_collection_config"].get(
        "resolved_cli_args", {}
    )
    for summary_field, cli_field in (
        ("max_actual_path_length_m", "max_actual_path_length_m"),
        ("max_steps", "max_steps"),
        ("stream_horizon", "stream_horizon"),
        ("max_sensor_skew_ms", "max_sensor_skew_ms"),
        ("max_endpoint_error_m", "max_endpoint_error_m"),
        ("max_stream_drift_m", "max_stream_drift_m"),
    ):
        if shared_contract[summary_field] != shared_cli.get(cli_field):
            raise RuntimeError(
                "{} does not match shared resolved CLI".format(summary_field)
            )
    worker_ids = [int(summary.get("worker_id", -1)) for summary in worker_summaries]
    expected_worker_ids = list(range(len(worker_dirs)))
    if sorted(worker_ids) != expected_worker_ids:
        raise RuntimeError(
            "worker summary ids {} != expected {}".format(
                sorted(worker_ids), expected_worker_ids
            )
        )
    mismatched_worker_counts = [
        worker_id
        for worker_id, worker_summary in zip(worker_ids, worker_summaries)
        if int(worker_summary.get("workers", -1)) != len(worker_dirs)
    ]
    if mismatched_worker_counts:
        raise RuntimeError(
            "worker summary count mismatch for workers {}".format(
                mismatched_worker_counts
            )
        )
    mismatched_worker_cli = [
        worker_id
        for worker_id, worker_summary in zip(worker_ids, worker_summaries)
        if int(
            worker_summary.get("resolved_cli_args", {}).get("worker_id", -1)
        )
        != worker_id
        or int(
            worker_summary.get("resolved_cli_args", {}).get(
                "num_workers", -1
            )
        )
        != len(worker_dirs)
    ]
    if mismatched_worker_cli:
        raise RuntimeError(
            "worker resolved CLI identity mismatch for workers {}".format(
                mismatched_worker_cli
            )
        )
    raw_collector_errors = [
        row
        for row in reports
        if str(row.get("stop_reason", "")).strip() == "collector_error"
    ]
    route_unavailable_rejections = [
        row for row in reports if is_route_unavailable_rejection(row)
    ]
    fatal_collector_errors = [
        row
        for row in raw_collector_errors
        if not is_route_unavailable_rejection(row)
    ]
    collision_total = sum(
        parse_bool(row.get("collision", False)) for row in reports
    )
    hard_altitude_total = sum(
        parse_bool(row.get("hard_altitude", False)) for row in reports
    )
    accepted_actual_path_length_max_m = numeric_max(
        accepted, "actual_path_length_m"
    )
    global_collision_mask_enabled = all(
        not str(row.get("global_collision_mask_enabled", "")).strip()
        or parse_bool(row.get("global_collision_mask_enabled"))
        for row in reports
    )
    launcher_context = shared_contract["shared_collection_config"].get(
        "launcher_context", {}
    )
    required_accepted = int(
        launcher_context.get("global_target_accepted", 0)
    )
    actual_path_limit_m = float(
        shared_contract["max_actual_path_length_m"]
    )
    formal_counters = {}
    if formal_mode:
        formal_counters = {
            "reliable_rows": sum(
                _integer_field(row, "reliable_rows", path="merged report")
                for row in reports
            ),
            "legacy_rows": sum(
                _integer_field(row, "legacy_rows", path="merged report")
                for row in reports
            ),
            "telemetry_lookup_count": sum(
                _integer_field(row, "telemetry_lookup_count", path="merged report")
                for row in reports
            ),
            "snapshot_missing_count": sum(
                _integer_field(row, "snapshot_missing_count", path="merged report")
                for row in reports
            ),
            "frame_contract_failures": sum(
                _integer_field(row, "frame_contract_failures", path="merged report")
                for row in reports
            ),
            "state_depth_skew_max_ns": max(
                (
                    _integer_field(
                        row, "state_depth_skew_max_ns", path="merged report"
                    )
                    for row in reports
                ),
                default=0,
            ),
            "endpoint_identity_chain_valid": all(
                parse_bool(row.get("endpoint_identity_chain_valid", False))
                for row in reports
            ),
        }
    quality_gates = {
        "accepted_nonzero": {
            "observed": len(accepted),
            "operator": ">",
            "threshold": 0,
            "pass": len(accepted) > 0,
        },
        "required_accepted": {
            "applicable": required_accepted > 0,
            "observed": len(accepted),
            "operator": ">=",
            "threshold": required_accepted,
            "pass": required_accepted <= 0 or len(accepted) >= required_accepted,
        },
        "collector_error_zero": {
            "observed": len(fatal_collector_errors),
            "operator": "==",
            "threshold": 0,
            "pass": len(fatal_collector_errors) == 0,
        },
        "collision_zero": {
            "observed": collision_total,
            "operator": "==",
            "threshold": 0,
            "pass": collision_total == 0,
        },
        "hard_altitude_zero": {
            "observed": hard_altitude_total,
            "operator": "==",
            "threshold": 0,
            "pass": hard_altitude_total == 0,
        },
        "accepted_actual_path_within_limit": {
            "observed": accepted_actual_path_length_max_m,
            "operator": "<=",
            "threshold": actual_path_limit_m,
            "tolerance": 1.0e-6,
            "pass": (
                accepted_actual_path_length_max_m
                <= actual_path_limit_m + 1.0e-6
            ),
        },
        "global_collision_mask_enabled": {
            "observed": global_collision_mask_enabled,
            "operator": "is",
            "threshold": True,
            "pass": global_collision_mask_enabled,
        },
    }
    if formal_mode:
        quality_gates.update(
            {
                "exact_legacy_rows_zero": {
                    "observed": formal_counters["legacy_rows"],
                    "operator": "==",
                    "threshold": 0,
                    "pass": formal_counters["legacy_rows"] == 0,
                },
                "exact_telemetry_lookup_zero": {
                    "observed": formal_counters["telemetry_lookup_count"],
                    "operator": "==",
                    "threshold": 0,
                    "pass": formal_counters["telemetry_lookup_count"] == 0,
                },
                "exact_snapshot_missing_zero": {
                    "observed": formal_counters["snapshot_missing_count"],
                    "operator": "==",
                    "threshold": 0,
                    "pass": formal_counters["snapshot_missing_count"] == 0,
                },
                "exact_state_depth_skew_zero": {
                    "observed": formal_counters["state_depth_skew_max_ns"],
                    "operator": "==",
                    "threshold": 0,
                    "pass": formal_counters["state_depth_skew_max_ns"] == 0,
                },
                "exact_frame_contract_failures_zero": {
                    "observed": formal_counters["frame_contract_failures"],
                    "operator": "==",
                    "threshold": 0,
                    "pass": formal_counters["frame_contract_failures"] == 0,
                },
                "exact_endpoint_identity_chain_valid": {
                    "observed": formal_counters["endpoint_identity_chain_valid"],
                    "operator": "is",
                    "threshold": True,
                    "pass": formal_counters["endpoint_identity_chain_valid"],
                },
            }
        )
    requested_target = int(getattr(args, "target_accepted", 0) or 0)
    if requested_target < 0:
        raise RuntimeError("target_accepted must be non-negative")
    if requested_target == 0:
        requested_target = required_accepted
    if formal_mode:
        accepted_dataset = _accepted_dataset_validation(
            accepted,
            out_dir=out_dir,
            artifact_roots=(out_dir, workers_dir, workers_dir.parent),
            target_accepted=requested_target,
        )
    else:
        # The historical diagnostic merge has no reliable-exact episode
        # provenance to validate.  Keep its old report-only behavior, while
        # making the formal path unambiguously accepted-only.
        accepted_dataset = {
            "clean_candidates": list(accepted),
            "valid_rows": list(accepted),
            "selected_rows": (
                list(accepted[:requested_target])
                if requested_target > 0
                else list(accepted)
            ),
            "all_accepted_paths": set(),
            "valid_paths": set(),
            "invalid_count": 0,
            "missing_count": 0,
            "contract_mismatch_count": 0,
            "errors": [],
            "quality_gates": {
                "accepted_nonzero": {
                    "observed": len(accepted),
                    "operator": ">",
                    "threshold": 0,
                    "pass": len(accepted) > 0,
                },
                "target_accepted": {
                    "observed": len(accepted),
                    "operator": ">=",
                    "threshold": requested_target,
                    "pass": len(accepted) >= requested_target,
                },
            },
            # Preserve the historical diagnostic merge contract.  The
            # accepted-only/runtime split is a formal reliable-exact rule;
            # legacy synthetic/diagnostic reports still fail their old
            # collision/dead-end quality gate.
            "quality_pass": all(
                bool(gate["pass"]) for gate in quality_gates.values()
            ),
        }
    runtime_quality_pass = all(
        bool(gate["pass"]) for gate in quality_gates.values()
    )
    accepted_dataset_quality_pass = bool(accepted_dataset["quality_pass"])
    cleanup_counts = {
        "failed_attempt_npz_removed": 0,
        "orphan_artifact_removed_count": 0,
        "temp_partial_file_count": 0,
    }
    if formal_mode and accepted_dataset_quality_pass:
        cleanup_counts = _cleanup_failed_artifacts(
            worker_dirs=worker_dirs,
            out_dir=out_dir,
            artifact_root=workers_dir.parent,
            reports=reports,
            accepted_paths=accepted_dataset["all_accepted_paths"],
        )
    summary = {
        "workers": len(worker_dirs),
        "attempted_total": len(reports),
        "accepted_total": len(accepted),
        "accepted_dataset_count": len(accepted_dataset["selected_rows"]),
        "accepted_dataset_valid_npz_count": len(accepted_dataset["valid_rows"]),
        "accepted_dataset_invalid_npz_count": accepted_dataset["invalid_count"],
        "accepted_dataset_missing_npz_count": accepted_dataset["missing_count"],
        "accepted_dataset_duplicate_episode_count": 0,
        "accepted_dataset_duplicate_transition_count": 0,
        "accepted_dataset_excluded_outcome_count": (
            len(accepted) - len(accepted_dataset["clean_candidates"])
        ),
        "accepted_dataset_reliable_rows": sum(
            _integer_field(row, "reliable_rows", path="accepted dataset")
            for row in accepted_dataset["selected_rows"]
        ),
        "target_accepted": requested_target,
        "accept_rate": len(accepted) / max(1, len(reports)),
        "success_total": sum(parse_bool(row.get("success", False)) for row in reports),
        "collision_total": collision_total,
        "dead_end_total": sum(parse_bool(row.get("dead_end", False)) for row in reports),
        "hard_altitude_total": hard_altitude_total,
        "collector_error_total": len(fatal_collector_errors),
        "raw_collector_error_total": len(raw_collector_errors),
        "mission_route_unavailable_total": len(route_unavailable_rejections),
        "accepted_actual_path_length_max_m": (
            accepted_actual_path_length_max_m
        ),
        "path_length_rejected_total": sum(
            parse_bool(row.get("path_length_exceeded", False)) for row in reports
        ),
        "global_collision_mask_enabled": global_collision_mask_enabled,
        "teacher_collection_quality_contract_id": (
            TEACHER_COLLECTION_QUALITY_CONTRACT_ID
        ),
        "quality_gates": quality_gates,
        "quality_pass": runtime_quality_pass,
        "quality_pass_semantics": "legacy_alias_of_runtime_quality_pass",
        "runtime_quality_pass": runtime_quality_pass,
        "accepted_dataset_quality_gates": accepted_dataset["quality_gates"],
        "accepted_dataset_quality_pass": accepted_dataset_quality_pass,
        "accepted_dataset_quality_errors": accepted_dataset["errors"][:20],
        "source_collector_error_count": len(raw_collector_errors),
        **cleanup_counts,
        "collection_report": "collection_report.csv",
        "rollout_index": "rollout_index.csv",
        "worker_resolved_configs": [
            os.path.relpath(
                str(worker_dir / "resolved_collection_config.json"),
                str(out_dir),
            )
            for worker_dir in worker_dirs
        ],
        **shared_contract,
    }
    if formal_mode:
        summary.update(
            {
                "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
                "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
                "reliable_execution": True,
                "telemetry_observation": False,
                "state_depth_skew_ns": 0,
                "endpoint_identity_available": True,
                "asynchronous_prefetch": False,
                "asynchronous_prefetch_status": "obsolete_for_reliable_exact",
                "reliable_rows": formal_counters["reliable_rows"],
                "legacy_rows": formal_counters["legacy_rows"],
                "telemetry_lookup_count": formal_counters["telemetry_lookup_count"],
                "snapshot_missing_count": formal_counters["snapshot_missing_count"],
                "state_depth_skew_max_ns": formal_counters["state_depth_skew_max_ns"],
                "exact_primitive_frame_count_failures": formal_counters[
                    "frame_contract_failures"
                ],
                "frame_contract_failures": formal_counters["frame_contract_failures"],
                "endpoint_identity_chain_valid": formal_counters[
                    "endpoint_identity_chain_valid"
                ],
            }
        )
    # Validate every worker contract before publishing any merged artifact.
    # A pre-existing report is raw run evidence and is intentionally immutable.
    raw_report_path = out_dir / "collection_report.csv"
    if raw_report_path.is_file():
        _assert_raw_report_unchanged(
            raw_report_path, rows=reports, fieldnames=fieldnames
        )
    else:
        write_csv_atomic(raw_report_path, reports, fieldnames=fieldnames)
    write_json_atomic(out_dir / "collection_summary.json", summary)
    if not accepted_dataset_quality_pass:
        (out_dir / "rollout_index.csv").unlink(missing_ok=True)
        print("TEACHER_ROLLOUT_MERGE_SUMMARY")
        for key, value in summary.items():
            print("  {}: {}".format(key, value))
        for name, gate in quality_gates.items():
            if not bool(gate["pass"]):
                print(
                    "QUALITY_GATE_FAILED {} observed={} operator={!r} threshold={}".format(
                        name,
                        gate["observed"],
                        gate["operator"],
                        gate["threshold"],
                    )
                )
        for name, gate in accepted_dataset["quality_gates"].items():
            if not bool(gate["pass"]):
                print(
                    "ACCEPTED_DATASET_GATE_FAILED {} observed={} operator={!r} threshold={}".format(
                        name,
                        gate["observed"],
                        gate["operator"],
                        gate["threshold"],
                    )
                )
        print("RESULT=FAIL")
        return 1
    # Publish the formal accepted index last.
    write_csv_atomic(
        out_dir / "rollout_index.csv",
        accepted_dataset["selected_rows"],
        fieldnames=fieldnames,
    )
    print("TEACHER_ROLLOUT_MERGE_SUMMARY")
    for key, value in summary.items():
        print("  {}: {}".format(key, value))
    if not runtime_quality_pass:
        print(
            "RUNTIME_QUALITY_WARNING collector errors or runtime quality gates "
            "remain; accepted-only dataset is still valid"
        )
    print("RESULT=PASS")
    return 0


def merge_worker_rollouts(
    *,
    workers_dir: Path,
    out_dir: Path,
    expected_workers: int = 0,
    target_accepted: int = 0,
) -> int:
    return _run(
        argparse.Namespace(
            workers_dir=str(Path(workers_dir)),
            out_dir=str(Path(out_dir)),
            expected_workers=int(expected_workers),
            target_accepted=int(target_accepted),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-workers", type=int, default=0)
    parser.add_argument("--target-accepted", type=int, default=0)
    return _run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
