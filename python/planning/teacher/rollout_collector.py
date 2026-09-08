#!/usr/bin/env python3
"""Collect formal reliable-exact Unity rollouts for offline relabeling.

The online teacher chooses behavior actions only. Every saved transition uses
an actual Unity observation captured immediately before the executed action.
Soft labels are generated later by ``label_teacher_rollouts.py`` from those
actual states. Legacy asynchronous collection is not a formal entry point;
the retained async-related report fields are historical schema fields.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from planning.contracts.collection import (
    HASH_ENVIRONMENT_FIELDS,
    build_resolved_collection_config,
    canonical_json,
    resolve_input_sha256,
    resolve_source_tree_sha256,
)
from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    exact_endpoint_metadata,
)
from planning.common import print_progress
from planning.common.progress import (
    ProgressRateTracker,
    format_duration,
    format_progress,
    progress_metrics,
)
from planning.common.config import parse_bool
from planning.common import read_csv, write_csv_atomic
from planning.native.geometry import NativeGeometryContext
from planning.data.rollout import (
    make_rollout_metadata,
    save_rollout_episode,
)
from planning.data.csv_journal import CsvJournal
from planning.runtime.unity_env import EnvConfig, UnityForestEnv
from planning.runtime.reliable_training import (
    build_reliable_v4_backend,
    build_reliable_v4_runtime_config,
)
from planning.runtime.identity import runtime_artifact_identity
from planning.teacher.collection_run import CollectionResumeLedger
from planning.teacher.rollout_session import ReliableExactRolloutSession
from planning.contracts.feature import INITIAL_PREV_ACTION, obs_goal_vector, obs_state_vector
from planning.mission.global_route import (
    GlobalRouteConfig,
    GlobalRouteUnavailableError,
    global_route_identity,
)
from planning.mission.spec import (
    DEFAULT_MAX_PRIMITIVE_STEPS,
    MISSION_PLANAR_DISTANCE_M,
    TASK_CONTRACT_ID,
    mission_id_from_row,
    row_start_goal,
    task_contract_sha256,
    validate_mission_rows,
)
from planning.data.mission_routes import (
    MissionRouteStore,
    resolve_route_store_prefix,
    route_store_provenance,
    validate_route_references,
)
from planning.contracts.task import task_contract_fields
from planning.primitives.library import (
    MotionPrimitiveLibrary,
    resolve_package_path,
)
from planning.diagnostics.observability import StructuredRunLogger
from planning.teacher.policy import (
    ReachabilityTeacher,
    add_teacher_arguments,
    config_from_args,
)
from planning.contracts.teacher_path import (
    TEACHER_ACTUAL_PATH_MAX_M,
    TEACHER_PATH_LENGTH_CONTRACT_ID,
    TEACHER_PLAN_PATH_MAX_M,
    actual_path_eligible,
    validate_plan_filtered_mission_rows,
)
from planning.common import write_json_atomic


REPORT_FIELDS = (
    "episode_id",
    "mission_id",
    "global_route_index",
    "global_route_contract_id",
    "mission_route_store_contract_id",
    "mission_route_store_schema_version",
    "mission_route_store",
    "mission_route_store_candidate_index_sha256",
    "mission_route_store_points_sha256",
    "mission_route_store_offsets_sha256",
    "global_collision_mask_enabled",
    "task_contract_id",
    "task_contract_schema_version",
    "task_contract_sha256",
    "max_primitive_steps",
    "resolved_config_sha256",
    "start_x",
    "start_y",
    "start_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "steps",
    "actual_path_length_m",
    "actual_path_limit_m",
    "path_length_exceeded",
    "path_length_contract_id",
    "success",
    "collision",
    "dead_end",
    "hard_altitude",
    "stop_reason",
    "execute_ok",
    "dataset_npz",
    "max_endpoint_error_m",
    "max_stream_drift_m",
    "tracking_drift_replans",
    "invalid_prefetch_replans",
    "predicted_deadend_replans",
    "prefetch_ready_count",
    "prefetch_miss_count",
    "planner_blocking_s",
    "prefetch_compute_s",
    "global_route_length_m",
    "global_route_points",
    "async_prefetch_enabled",
    "elapsed_s",
    "error",
)

PARTIAL_INDEX_ACCEPT_INTERVAL = 50
PARTIAL_INDEX_TIME_INTERVAL_S = 30.0

FORMAL_REPORT_FIELDS = (
    "episode_id",
    "mission_id",
    "global_route_index",
    "global_route_contract_id",
    "mission_route_store_contract_id",
    "mission_route_store_schema_version",
    "mission_route_store",
    "mission_route_store_candidate_index_sha256",
    "mission_route_store_points_sha256",
    "mission_route_store_offsets_sha256",
    "collection_run_id",
    "runtime_instance_id",
    "unity_player_sha256",
    "runtime_assembly_sha256",
    "bridge_sha256",
    "observation_contract",
    "observation_source",
    "reliable_execution",
    "telemetry_observation",
    "state_depth_skew_ns",
    "endpoint_identity_available",
    "asynchronous_prefetch",
    "asynchronous_prefetch_status",
    "reliable_rows",
    "legacy_rows",
    "telemetry_lookup_count",
    "snapshot_missing_count",
    "state_depth_skew_max_ns",
    "frame_contract_failures",
    "endpoint_identity_chain_valid",
    "transition_ids",
    "global_collision_mask_enabled",
    "task_contract_id",
    "task_contract_schema_version",
    "task_contract_sha256",
    "max_primitive_steps",
    "resolved_config_sha256",
    "start_x",
    "start_y",
    "start_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "steps",
    "actual_path_length_m",
    "actual_path_limit_m",
    "path_length_exceeded",
    "path_length_contract_id",
    "success",
    "collision",
    "dead_end",
    "hard_altitude",
    "stop_reason",
    "execute_ok",
    "dataset_npz",
    "max_endpoint_error_m",
    "max_stream_drift_m",
    "tracking_drift_replans",
    "invalid_prefetch_replans",
    "predicted_deadend_replans",
    "prefetch_ready_count",
    "prefetch_miss_count",
    "planner_blocking_s",
    "prefetch_compute_s",
    "global_route_length_m",
    "global_route_points",
    "async_prefetch_enabled",
    "elapsed_s",
    "error",
)


FORMAL_REPORT_BOOLEAN_DEFAULTS = {
    "reliable_execution": True,
    "telemetry_observation": False,
    "endpoint_identity_available": True,
    "asynchronous_prefetch": False,
    "endpoint_identity_chain_valid": True,
    "global_collision_mask_enabled": True,
    "path_length_exceeded": False,
    "success": False,
    "collision": False,
    "dead_end": False,
    "hard_altitude": False,
    "execute_ok": False,
    "async_prefetch_enabled": False,
}


def formal_report_boolean_defaults() -> Dict[str, bool]:
    """Return explicit initial values for every formal report boolean."""

    return dict(FORMAL_REPORT_BOOLEAN_DEFAULTS)


def _formal_report_counter(row: Dict[str, Any], field: str) -> int:
    """Read one persisted exact counter without silently accepting garbage."""

    try:
        value = int(float(row.get(field, 0) or 0))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "formal collection report has invalid {}={!r}".format(
                field, row.get(field)
            )
        ) from error
    if value < 0:
        raise RuntimeError(
            "formal collection report has negative {}={}".format(field, value)
        )
    return value


def empty_episode_buffers() -> Dict[str, list]:
    return {
        "depths": [],
        "states": [],
        "goals": [],
        "height_action_masks": [],
        "execution_action_masks": [],
        "behavior_actions": [],
        "prev_actions": [],
        "poses_before": [],
        "velocities_before": [],
        "state_ids": [],
        "sim_time_ns": [],
        "state_stamp_ns": [],
        "depth_stamp_ns": [],
        "sensor_skew_ns": [],
        "primitive_actual_path_lengths_m": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--route-store-prefix",
        default="",
        help="committed MissionRouteStore prefix; defaults to the mission row reference",
    )
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--target-accepted", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_PRIMITIVE_STEPS)
    parser.add_argument("--stream-horizon", type=int, default=1)
    parser.add_argument("--max-stream-drift-m", type=float, default=0.35)
    parser.add_argument("--max-endpoint-error-m", type=float, default=0.60)
    parser.add_argument("--max-actual-path-length-m", type=float, default=TEACHER_ACTUAL_PATH_MAX_M)
    parser.add_argument("--depth-width", type=int, default=160)
    parser.add_argument("--depth-height", type=int, default=90)
    parser.add_argument("--reset-settle", type=float, default=0.30)
    parser.add_argument("--reset-timeout", type=float, default=10.0)
    parser.add_argument("--worker-ready-timeout", type=float, default=30.0)
    parser.add_argument("--post-wait", type=float, default=0.0)
    parser.add_argument("--max-sensor-skew-ms", type=float, default=80.0)
    parser.add_argument("--stability-timeout", type=float, default=2.0)
    parser.add_argument("--stability-speed-tolerance", type=float, default=0.03)
    parser.add_argument("--stability-frames", type=int, default=3)
    parser.add_argument("--compress-episodes", action="store_true")
    parser.add_argument("--stop-file", default="")
    parser.add_argument("--collision-cache", default="data/map_data/forest_voxels_10cm.npz")
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--inflate-radius", type=float, default=0.35)
    parser.add_argument("--prefetch-inflate-radius", type=float, default=0.40)
    parser.add_argument(
        "--reliable-exact-runtime-instance-id",
        "--reliable-v4-runtime-instance-id",
        dest="reliable_exact_runtime_instance_id",
        default="",
    )
    parser.add_argument(
        "--reliable-exact-command-endpoint",
        "--reliable-v4-command-endpoint",
        dest="reliable_exact_command_endpoint",
        default="",
    )
    parser.add_argument(
        "--reliable-exact-result-endpoint",
        "--reliable-v4-result-endpoint",
        dest="reliable_exact_result_endpoint",
        default="",
    )
    parser.add_argument(
        "--reliable-exact-snapshot-endpoint",
        "--reliable-v4-snapshot-endpoint",
        dest="reliable_exact_snapshot_endpoint",
        default="",
    )
    parser.add_argument(
        "--reliable-exact-timeout",
        "--reliable-v4-timeout",
        dest="reliable_exact_timeout",
        type=float,
        default=10.0,
    )
    add_teacher_arguments(parser, candidate_top_k=12, current_check_step=1, lookahead_check_step=2)
    parser.add_argument("--expected-planar-distance", type=float, default=MISSION_PLANAR_DISTANCE_M)
    parser.add_argument("--planar-distance-tolerance", type=float, default=1.0e-3)
    args = parser.parse_args()
    return _run_reliable_exact(args)


def _formal_append_transition(
    buffers: Dict[str, list],
    *,
    before: Dict[str, Any],
    action_id: int,
    previous_action: int,
    mask: np.ndarray,
    mask_info: Dict[str, Any],
    primitive_path_length_m: float,
) -> None:
    """Append one exact-before action row without changing array schemas."""

    position = np.asarray(before["state"]["position"], dtype=np.float32)
    velocity = np.asarray(before["state"]["velocity"], dtype=np.float32)
    buffers["depths"].append(np.asarray(before["depth"], dtype=np.float16))
    buffers["states"].append(obs_state_vector(before))
    buffers["goals"].append(obs_goal_vector(before))
    buffers["height_action_masks"].append(
        np.asarray(mask_info["height_mask"], dtype=np.bool_)
    )
    buffers["execution_action_masks"].append(np.asarray(mask, dtype=np.bool_))
    buffers["behavior_actions"].append(int(action_id))
    buffers["prev_actions"].append(int(previous_action))
    buffers["poses_before"].append(
        np.asarray(
            [position[0], position[1], position[2], float(before["state"]["yaw"])],
            dtype=np.float32,
        )
    )
    buffers["velocities_before"].append(velocity)
    buffers["state_ids"].append(int(before["state"]["state_id"]))
    buffers["sim_time_ns"].append(int(before["state"]["sim_time_ns"]))
    sensor_time = before["sensor_time"]
    buffers["state_stamp_ns"].append(int(sensor_time["state_stamp_ns"]))
    buffers["depth_stamp_ns"].append(int(sensor_time["depth_stamp_ns"]))
    buffers["sensor_skew_ns"].append(int(sensor_time["skew_ns"]))
    buffers["primitive_actual_path_lengths_m"].append(
        float(max(0.0, primitive_path_length_m))
    )


def _formal_validate_args(args) -> None:
    if int(args.stream_horizon) != 1:
        raise ValueError("formal reliable-exact collection requires --stream-horizon=1")
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be positive")
    if int(args.num_workers) <= 0:
        raise ValueError("--num-workers must be positive")
    if int(args.max_episodes) < 0 or int(args.target_accepted) < 0:
        raise ValueError("formal collection limits must be non-negative")
    if not 0 <= int(args.worker_id) < int(args.num_workers):
        raise ValueError("--worker-id must be in [0, --num-workers)")
    if abs(float(args.max_actual_path_length_m) - TEACHER_ACTUAL_PATH_MAX_M) > 1.0e-9:
        raise ValueError(
            "--max-actual-path-length-m is fixed by {} at {}".format(
                TEACHER_PATH_LENGTH_CONTRACT_ID, TEACHER_ACTUAL_PATH_MAX_M
            )
        )
    required = (
        ("--reliable-exact-runtime-instance-id", args.reliable_exact_runtime_instance_id),
        ("--reliable-exact-command-endpoint", args.reliable_exact_command_endpoint),
        ("--reliable-exact-result-endpoint", args.reliable_exact_result_endpoint),
        ("--reliable-exact-snapshot-endpoint", args.reliable_exact_snapshot_endpoint),
    )
    missing = [name for name, value in required if not str(value).strip()]
    if missing:
        raise ValueError(
            "formal reliable-exact collection requires: {}".format(", ".join(missing))
        )
    if float(args.reliable_exact_timeout) <= 0.0:
        raise ValueError("--reliable-exact-timeout must be positive")


def _build_formal_reliable_backend(args):
    config = build_reliable_v4_runtime_config(
        runtime_instance_id=str(args.reliable_exact_runtime_instance_id),
        command_endpoint=str(args.reliable_exact_command_endpoint),
        result_endpoint=str(args.reliable_exact_result_endpoint),
        snapshot_endpoint=str(args.reliable_exact_snapshot_endpoint),
        timeout_s=float(args.reliable_exact_timeout),
    )
    return build_reliable_v4_backend(config)


def _close_formal_backend(backend: Any) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def _run_reliable_exact(args) -> int:
    """Run one worker of the formal exact collection contract."""

    _formal_validate_args(args)
    index_path = Path(args.index).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    collision_cache_path = resolve_package_path(args.collision_cache)
    episodes_dir = out_dir / "episodes"
    report_path = out_dir / "collection_report.csv"
    report_journal_path = out_dir / "collection_report.journal.jsonl"
    rollout_index_path = out_dir / "rollout_index.csv"
    rollout_partial_index_path = out_dir / "rollout_index.partial.csv"
    progress_path = out_dir / "collection_progress.json"
    structured_log_path = out_dir / "collection.jsonl"
    runtime_instance_id = str(args.reliable_exact_runtime_instance_id).strip()

    source_rows = read_csv(index_path)
    validate_mission_rows(
        source_rows,
        expected_planar_distance=float(args.expected_planar_distance),
        planar_distance_tolerance=float(args.planar_distance_tolerance),
    )
    validate_plan_filtered_mission_rows(source_rows)
    route_store_prefix = resolve_route_store_prefix(
        source_rows,
        index_path,
        Path(args.route_store_prefix) if str(args.route_store_prefix).strip() else None,
    )
    route_store = MissionRouteStore.open(route_store_prefix, validate=True)
    validate_route_references(
        source_rows,
        route_store,
        candidate_index_sha256=route_store.metadata["candidate_index_sha256"],
        require_provenance=True,
    )
    route_provenance = route_store_provenance(route_store, artifact_path=report_path)
    source_rows.sort(key=lambda row: int(float(row["episode_id"])))
    source_rows = [
        row
        for row in source_rows
        if int(float(row["episode_id"])) % int(args.num_workers) == int(args.worker_id)
    ]
    if not source_rows:
        raise RuntimeError("worker shard contains no missions")

    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    if report_path.exists() and report_path.stat().st_size > 0:
        with report_path.open("r", newline="", encoding="utf-8") as handle:
            existing_header = tuple(next(csv.reader(handle), []))
        if existing_header != FORMAL_REPORT_FIELDS:
            raise RuntimeError(
                "formal collection_report.csv contract mismatch; use a clean output directory"
            )
    report_journal = CsvJournal(
        report_journal_path,
        fieldnames=FORMAL_REPORT_FIELDS,
        key_field="episode_id",
        retain_rows=False,
    )
    if report_journal.row_count == 0 and report_path.exists():
        report_journal.import_rows(read_csv(report_path))
    existing_report = report_journal.recover()
    existing_report_count = len(existing_report)
    supplied_run_id = str(os.environ.get("PLANNING_COLLECTION_RUN_ID", "")).strip()
    if supplied_run_id:
        run_id = supplied_run_id
    elif existing_report:
        run_id = str(existing_report[0].get("collection_run_id", "")).strip()
        if not run_id:
            raise RuntimeError("formal resume report is missing collection_run_id")
    else:
        run_id = "teacher-collection-w{}-{}".format(
            int(args.worker_id), datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
    os.environ["PLANNING_COLLECTION_RUN_ID"] = run_id
    resume_ledger = CollectionResumeLedger.from_rows(
        existing_report,
        runtime_instance_id=runtime_instance_id,
        run_id=run_id,
    )
    attempted_at_run_start = int(resume_ledger.attempted_total)
    accepted_count_runtime = int(resume_ledger.accepted_total)
    accepted_at_run_start = int(accepted_count_runtime)
    pending = [
        row
        for row in source_rows
        if int(float(row["episode_id"])) not in resume_ledger.attempted_episode_ids
    ]
    if int(args.max_episodes) > 0:
        pending = pending[: int(args.max_episodes)]

    if not pending and (out_dir / "collection_summary.json").is_file():
        accepted_rows_at_resume = [
            dict(row)
            for row in report_journal.iter_rows()
            if parse_bool(row.get("execute_ok", False))
        ]
        accepted_rows_at_resume.sort(
            key=lambda item: int(float(item["episode_id"]))
        )
        write_csv_atomic(
            rollout_index_path,
            accepted_rows_at_resume,
            fieldnames=FORMAL_REPORT_FIELDS,
        )
        report_journal.commit(report_path)
        report_journal.close()
        rollout_partial_index_path.unlink(missing_ok=True)
        print("TEACHER_RELIABLE_EXACT_RESUME_COMPLETE")
        print("  attempted_total:", resume_ledger.attempted_total)
        print("  accepted_total:", resume_ledger.accepted_total)
        print("RESULT=PASS")
        route_store.close()
        return 0
    del existing_report

    mission_index_sha256 = resolve_input_sha256(
        index_path, HASH_ENVIRONMENT_FIELDS["mission_index_sha256"]
    )
    collision_cache_sha256 = resolve_input_sha256(
        collision_cache_path, HASH_ENVIRONMENT_FIELDS["collision_cache_sha256"]
    )
    code_version_sha256 = resolve_source_tree_sha256()
    mpl = MotionPrimitiveLibrary()
    teacher_cfg = config_from_args(args)
    route_config = GlobalRouteConfig(
        resolution_m=float(teacher_cfg.global_route_resolution_m),
        flight_z_min_m=float(teacher_cfg.z_min),
        flight_z_max_m=float(teacher_cfg.z_max),
        lookahead_m=float(teacher_cfg.global_route_lookahead_m),
        tracking_margin_m=float(teacher_cfg.global_route_tracking_margin_m),
    )
    route_identity = global_route_identity(route_config)
    geometry_context = NativeGeometryContext.from_voxel_cache(
        collision_cache_path,
        voxel_size=float(args.voxel_size),
        route_config=route_config,
    )
    checker = geometry_context.collision_checker(float(args.inflate_radius))
    behavior_teacher = ReachabilityTeacher(mpl, checker, teacher_cfg)
    # The formal path has no predicted-state prefetch; this name documents the
    # preserved route handoff for callers that inspect the collection contract.
    prefetch_teacher = behavior_teacher
    runtime_identity = runtime_artifact_identity(
        player=Path(os.environ.get("UNITY_BIN", ""))
    )
    collection_contract = build_resolved_collection_config(
        args,
        mission_index_sha256=mission_index_sha256,
        collision_cache_sha256=collision_cache_sha256,
        code_version_sha256=code_version_sha256,
        teacher_config=teacher_cfg.__dict__,
        mpl_contract_sha256=str(mpl.contract_sha256),
        async_prefetch_enabled=False,
        observation_contract=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        observation_source=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        reliable_execution=True,
        runtime_artifact_identity=runtime_identity,
        route_store_identity=route_provenance,
    )
    write_json_atomic(out_dir / "resolved_collection_config.json", collection_contract)

    run_logger = StructuredRunLogger(
        structured_log_path,
        run_id=run_id,
        component="teacher_rollout_collection_reliable_exact",
        sample_gpu=False,
    )
    partial_index_last_flush_accepted = accepted_count_runtime
    partial_index_last_flush_time = time.monotonic()
    run_started = time.monotonic()
    attempted_this_run = 0
    interrupted = False
    global_target_accepted = int(
        os.environ.get("PLANNING_COLLECTION_GLOBAL_TARGET_ACCEPTED", args.target_accepted)
    )
    global_max_episodes = int(
        os.environ.get("PLANNING_COLLECTION_GLOBAL_MAX_EPISODES", args.max_episodes)
    )
    backend = None
    env = None
    session = ReliableExactRolloutSession(
        runtime_instance_id=runtime_instance_id,
        depth_shape=(int(args.depth_height), int(args.depth_width)),
        transition_id_offset=resume_ledger.next_transition_id_offset(),
    )
    stop_file = Path(args.stop_file).expanduser().resolve() if str(args.stop_file).strip() else None
    attempt_rate_tracker = ProgressRateTracker(
        initial_completed=attempted_at_run_start
    )
    accepted_rate_tracker = ProgressRateTracker(
        initial_completed=accepted_at_run_start
    )
    reliable_rate_tracker = ProgressRateTracker(initial_completed=0)
    latest_progress_metrics: Dict[str, Any] = {}

    def persist_partial_rollout_index(force: bool = False) -> bool:
        nonlocal partial_index_last_flush_accepted, partial_index_last_flush_time
        accepted_delta = accepted_count_runtime - partial_index_last_flush_accepted
        elapsed = time.monotonic() - partial_index_last_flush_time
        if force and accepted_delta == 0 and rollout_partial_index_path.exists():
            return False
        if not force:
            return False
        accepted_rows = [
            dict(row)
            for row in report_journal.iter_rows()
            if parse_bool(row.get("execute_ok", False))
        ]
        write_csv_atomic(
            rollout_partial_index_path,
            sorted(accepted_rows, key=lambda item: int(float(item["episode_id"]))),
            fieldnames=FORMAL_REPORT_FIELDS,
        )
        partial_index_last_flush_accepted = accepted_count_runtime
        partial_index_last_flush_time = time.monotonic()
        return True

    def persist_progress(status: str, outcome: Optional[dict] = None) -> None:
        elapsed_s = max(0.0, time.monotonic() - run_started)
        estimated_stop = (
            global_target_accepted
            if global_target_accepted > 0
            else (
                global_max_episodes
                if global_max_episodes > 0
                else len(source_rows)
            )
        )
        eta_processed = (
            accepted_count_runtime
            if global_target_accepted > 0
            else resume_ledger.attempted_total
        )
        eta_baseline_processed = (
            accepted_at_run_start
            if global_target_accepted > 0
            else attempted_at_run_start
        )
        attempt_snapshot = attempt_rate_tracker.update(
            resume_ledger.attempted_total
        )
        accepted_snapshot = accepted_rate_tracker.update(
            accepted_count_runtime
        )
        reliable_snapshot = reliable_rate_tracker.update(
            int(session.counters.reliable_rows)
        )
        eta_snapshot = (
            accepted_snapshot
            if global_target_accepted > 0
            else attempt_snapshot
        )
        metrics = progress_metrics(
            processed=resume_ledger.attempted_total,
            passing=accepted_count_runtime,
            estimated_stop=estimated_stop,
            elapsed_s=elapsed_s,
            baseline_processed=attempted_at_run_start,
            eta_processed=eta_processed,
            eta_baseline_processed=eta_baseline_processed,
            eta_target=estimated_stop,
            rolling_throughput_per_s=eta_snapshot["rolling_rate_per_s"],
            ewma_throughput_per_s=eta_snapshot["ewma_rate_per_s"],
            minimum_calibration_s=10.0,
            minimum_calibration_items=5,
        )
        latest_progress_metrics.clear()
        latest_progress_metrics.update(metrics)
        latest_progress_metrics.update(
            {
                "attempts_per_s": attempt_snapshot["stable_rate_per_s"],
                "accepted_per_s": accepted_snapshot["stable_rate_per_s"],
                "reliable_rows_per_s": reliable_snapshot["stable_rate_per_s"],
            }
        )
        progress_line = format_progress(
            component="teacher_rollout_collection_reliable_exact",
            processed=resume_ledger.attempted_total,
            passing=accepted_count_runtime,
            estimated_stop=estimated_stop,
            elapsed_s=elapsed_s,
            baseline_processed=attempted_at_run_start,
            eta_processed=eta_processed,
            eta_baseline_processed=eta_baseline_processed,
            eta_target=estimated_stop,
            rolling_throughput_per_s=eta_snapshot["rolling_rate_per_s"],
            ewma_throughput_per_s=eta_snapshot["ewma_rate_per_s"],
            minimum_calibration_s=10.0,
            minimum_calibration_items=5,
            worker=args.worker_id,
            accepted_per_s=round(float(accepted_snapshot["stable_rate_per_s"]), 6),
            reliable_rows_per_s=round(
                float(reliable_snapshot["stable_rate_per_s"]), 6
            ),
            rejected=max(
                0,
                int(resume_ledger.attempted_total)
                - int(accepted_count_runtime),
            ),
            elapsed=format_duration(elapsed_s),
        )
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": run_id,
            "status": status,
            "complete": status == "completed",
            "worker_id": int(args.worker_id),
            "workers": int(args.num_workers),
            "runtime_instance_id": runtime_instance_id,
            "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            "reliable_rows": int(session.counters.reliable_rows),
            "legacy_rows": 0,
            "telemetry_lookup_count": int(session.counters.telemetry_lookup_count),
            "snapshot_missing_count": int(session.counters.snapshot_missing_count),
            "state_depth_skew_max_ns": int(session.counters.state_depth_skew_max_ns),
            "attempted_total": resume_ledger.attempted_total,
            "accepted_total": accepted_count_runtime,
            "attempted_baseline_at_run_start": attempted_at_run_start,
            "accepted_baseline_at_run_start": accepted_at_run_start,
            "elapsed_s": elapsed_s,
            "estimated_stop": int(estimated_stop),
            "throughput_per_s": float(metrics["throughput_per_s"]),
            "accepted_throughput_per_s": float(metrics["throughput_per_s"]),
            "rolling_throughput_per_s": float(
                metrics["rolling_throughput_per_s"]
            ),
            "ewma_throughput_per_s": float(metrics["ewma_throughput_per_s"]),
            "reliable_rows_per_s": float(
                reliable_snapshot["stable_rate_per_s"]
            ),
            "eta_h": (
                float(metrics["eta_h"])
                if metrics["eta_h"] is not None
                and math.isfinite(float(metrics["eta_h"]))
                else None
            ),
            "eta_processed": int(eta_processed),
            "eta_status": metrics["eta_status"],
            "progress": progress_line,
            "target_accepted": int(global_target_accepted),
            "pending_at_run_start": len(pending),
            "formal_rollout_index": rollout_index_path.name if status == "completed" else "",
            "partial_rollout_index": "" if status == "completed" else rollout_partial_index_path.name,
            "collection_report": report_path.name,
            "collection_report_journal": report_journal_path.name,
            "structured_log": structured_log_path.name,
            "collection_config_contract_id": collection_contract[
                "collection_config_contract_id"
            ]
            if collection_contract is not None
            else "",
            "resolved_config_sha256": collection_contract["resolved_config_sha256"]
            if collection_contract is not None
            else "",
        }
        if outcome is not None:
            payload.update(
                {
                    "last_mission_id": outcome.get("mission_id", ""),
                    "last_episode": int(outcome.get("episode_id", -1)),
                    "last_stop_reason": outcome.get("stop_reason", ""),
                }
            )
        write_json_atomic(progress_path, payload)

    try:
        backend = _build_formal_reliable_backend(args)
        env = UnityForestEnv(
            start=[0.0, 0.0, 2.0],
            goal=[40.0, 0.0, 2.0],
            mpl=mpl,
            config=EnvConfig(
                depth_out_width=int(args.depth_width),
                depth_out_height=int(args.depth_height),
                reset_timeout=float(args.reset_timeout),
                reset_settle_s=float(args.reset_settle),
                primitive_post_wait_s=float(args.post_wait),
                stop_at_primitive_end=False,
                enforce_sensor_sync=True,
                max_sensor_skew_s=max(0.0, float(args.max_sensor_skew_ms) * 1.0e-3),
                max_episode_steps=int(args.max_steps),
                goal_radius_xy=float(args.goal_radius),
                goal_tolerance_z=float(args.goal_tolerance_z),
                use_global_collision_mask=False,
                collision_cache_npz=str(collision_cache_path),
                collision_voxel_size=float(args.voxel_size),
                collision_inflate_radius=float(args.inflate_radius),
                collision_check_step=int(args.current_check_step),
                terminate_on_dead_end=True,
                terminate_on_invalid_action=True,
            ),
            reliable_v4_backend=backend,
            legacy_command_path_enabled=False,
        )
        env._collision_checker = checker
        env.config.use_global_collision_mask = True
        env.wait_until_ready(float(args.worker_ready_timeout))
        persist_progress("started")
        run_logger.event(
            "started",
            attempted_count=resume_ledger.attempted_total,
            accepted_count=accepted_count_runtime,
            pending_count=len(pending),
            resolved_config_sha256=collection_contract["resolved_config_sha256"],
            observation_contract=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        )

        for row in pending:
            if stop_file is not None and stop_file.exists():
                break
            episode_started = time.monotonic()
            episode_id = int(float(row["episode_id"]))
            start, goal_list = row_start_goal(row)
            goal = np.asarray(goal_list, dtype=np.float32)
            counters_before = session.counters.as_dict()
            outcome = {field: "" for field in FORMAL_REPORT_FIELDS}
            outcome.update(
                {
                    **formal_report_boolean_defaults(),
                    "episode_id": episode_id,
                    "mission_id": mission_id_from_row(row),
                    "collection_run_id": run_id,
                    "runtime_instance_id": runtime_instance_id,
                    "unity_player_sha256": runtime_identity["unity_player_sha256"],
                    "runtime_assembly_sha256": runtime_identity[
                        "runtime_assembly_sha256"
                    ],
                    "bridge_sha256": runtime_identity["bridge_sha256"],
                    **exact_endpoint_metadata(),
                    "reliable_rows": 0,
                    "legacy_rows": 0,
                    "telemetry_lookup_count": 0,
                    "snapshot_missing_count": 0,
                    "state_depth_skew_max_ns": 0,
                    "frame_contract_failures": 0,
                    "endpoint_identity_chain_valid": True,
                    "transition_ids": "[]",
                    "global_route_index": int(float(row["global_route_index"])),
                    **route_provenance,
                    "global_collision_mask_enabled": True,
                    **task_contract_fields(int(args.max_steps)),
                    "resolved_config_sha256": collection_contract["resolved_config_sha256"],
                    "start_x": start[0],
                    "start_y": start[1],
                    "start_z": start[2],
                    "goal_x": float(goal[0]),
                    "goal_y": float(goal[1]),
                    "goal_z": float(goal[2]),
                    "execute_ok": False,
                    "dataset_npz": "",
                    "async_prefetch_enabled": False,
                    "error": "",
                }
            )
            buffers = empty_episode_buffers()
            transition_ids: List[str] = []
            execution_ids: List[int] = []
            route_info: Dict[str, Any] = {}
            stop_reason = "max_steps"
            success = False
            collision = False
            dead_end = False
            hard_altitude = False
            max_endpoint_error = 0.0
            actual_path_length_m = 0.0
            path_length_exceeded = False
            planner_blocking_s = 0.0
            steps = 0
            frame_count = 0

            def persist_attempt_provenance() -> None:
                """Keep report provenance truthful even when an episode aborts."""

                current = session.counters.as_dict()
                outcome.update(
                    {
                        "reliable_rows": max(
                            0,
                            int(current["reliable_rows"])
                            - int(counters_before["reliable_rows"]),
                        ),
                        "telemetry_lookup_count": max(
                            0,
                            int(current["telemetry_lookup_count"])
                            - int(counters_before["telemetry_lookup_count"]),
                        ),
                        "snapshot_missing_count": max(
                            0,
                            int(current["snapshot_missing_count"])
                            - int(counters_before["snapshot_missing_count"]),
                        ),
                        "state_depth_skew_max_ns": max(
                            0,
                            int(current["state_depth_skew_max_ns"])
                            - int(counters_before["state_depth_skew_max_ns"]),
                        ),
                        "frame_contract_failures": max(
                            0,
                            int(current["frame_contract_failures"])
                            - int(counters_before["frame_contract_failures"]),
                        ),
                        "endpoint_identity_chain_valid": bool(
                            current["endpoint_identity_chain_valid"]
                        ),
                        "global_collision_mask_enabled": True,
                        "path_length_exceeded": bool(path_length_exceeded),
                        "success": bool(success),
                        "collision": bool(collision),
                        "dead_end": bool(dead_end),
                        "hard_altitude": bool(hard_altitude),
                        "execute_ok": False,
                        "async_prefetch_enabled": False,
                        "transition_ids": json.dumps(
                            transition_ids, sort_keys=True
                        ),
                        "steps": len(buffers["behavior_actions"]),
                    }
                )

            try:
                obs = env.reset(start=start, goal=goal)
                session.reset(obs, episode_id=env.episode_id, reset_id=env.reset_id)
                route_info = behavior_teacher.set_mission_route(
                    route_store.validate(int(float(row["global_route_index"])), goal=goal),
                    goal,
                )
                if prefetch_teacher is not behavior_teacher:
                    prefetch_teacher.set_mission_route(behavior_teacher.mission_route, goal)
                previous_action = INITIAL_PREV_ACTION
                while len(buffers["behavior_actions"]) < int(args.max_steps):
                    plan_started = time.perf_counter()
                    plan = behavior_teacher.plan_horizon(
                        obs,
                        prev_action=previous_action,
                        horizon=1,
                        prefetch_teacher=prefetch_teacher,
                    )
                    planner_blocking_s += time.perf_counter() - plan_started
                    if not plan:
                        dead_end = True
                        stop_reason = "teacher_dead_end_actual_state"
                        break
                    action_id = int(plan[0]["action_id"])
                    mask, mask_info = env.get_action_mask(obs, return_info=True)
                    if bool(mask_info.get("dead_end", False)):
                        dead_end = True
                        stop_reason = "actual_dead_end"
                        break
                    if not bool(mask[action_id]):
                        raise RuntimeError("Teacher action is outside exact endpoint action mask")
                    before = obs
                    after, _reward, done, info = env.step_primitive(
                        action_id,
                        obs_before=before,
                        precomputed_mask=mask,
                        precomputed_mask_info=mask_info,
                    )
                    frame_count = len(mpl.command_sequence(action_id))
                    record = session.record_transition(
                        before=before,
                        after=after,
                        info=info,
                        action_id=action_id,
                        previous_action=previous_action,
                        command_frame_count=frame_count,
                    )
                    transition_ids.append(record.transition_id)
                    execution_ids.append(record.execution_id)
                    observed_path = float(env.actual_trajectory_length_m())
                    _formal_append_transition(
                        buffers,
                        before=before,
                        action_id=action_id,
                        previous_action=previous_action,
                        mask=mask,
                        mask_info=mask_info,
                        primitive_path_length_m=max(0.0, observed_path - actual_path_length_m),
                    )
                    actual_path_length_m = observed_path
                    path_length_exceeded = not actual_path_eligible(actual_path_length_m)
                    previous_action = action_id
                    obs = after
                    max_endpoint_error = max(
                        max_endpoint_error,
                        float(info.get("primitive_endpoint_error_body_m", 0.0)),
                    )
                    success = bool(info.get("success", False))
                    collision = bool(info.get("collided", False))
                    hard_altitude = bool(info.get("hard_altitude_violation", False))
                    if path_length_exceeded:
                        stop_reason = "actual_path_length_exceeded"
                    elif done:
                        stop_reason = str(info.get("done_reason", "terminal"))
                    if success or collision or hard_altitude or path_length_exceeded or done:
                        break
                env.stop()
                steps = len(buffers["behavior_actions"])
                accepted = bool(
                    success
                    and not collision
                    and not dead_end
                    and not hard_altitude
                    and not path_length_exceeded
                    and steps > 0
                    and max_endpoint_error <= float(args.max_endpoint_error_m)
                )
                dataset_path = ""
                if accepted:
                    dataset_path_obj = episodes_dir / "episode_{:06d}.npz".format(episode_id)
                    arrays = {key: np.asarray(values) for key, values in buffers.items()}
                    arrays["start"] = np.asarray(start, dtype=np.float32)
                    arrays["goal"] = goal.astype(np.float32)
                    metadata = make_rollout_metadata(
                        collector="collect_teacher_rollouts.py",
                        episode_id=episode_id,
                        collection_run_id=run_id,
                        runtime_instance_id=runtime_instance_id,
                        observation_contract=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
                        observation_source=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
                        reliable_execution=True,
                        telemetry_observation=False,
                        asynchronous_prefetch=False,
                        asynchronous_prefetch_status="obsolete_for_reliable_exact",
                        endpoint_identity_available=True,
                        state_depth_skew_ns=0,
                        telemetry_lookup_count=0,
                        snapshot_missing_count=0,
                        reliable_rows=steps,
                        legacy_rows=0,
                        transition_ids=transition_ids,
                        execution_ids=execution_ids,
                        exact_primitive_frame_count=frame_count,
                        actual_state_mask_gate=True,
                        offline_relabel_required=True,
                        **{**route_identity, **route_provenance},
                        initial_prev_action=INITIAL_PREV_ACTION,
                        max_steps=int(args.max_steps),
                        mpl_duration_s=float(mpl.duration_s),
                        mpl_forward_distance_m=float(mpl.forward_distance_m),
                        mpl_contract_sha256=str(mpl.contract_sha256),
                        teacher_planning_contract_id=behavior_teacher.teacher_planning_contract_id,
                        teacher_config=teacher_cfg.__dict__,
                        global_route_index=int(float(row["global_route_index"])),
                        global_route_length_m=float(route_info["route_length_m"]),
                        global_route_points=int(route_info["route_points"]),
                        stream_horizon=1,
                        max_sensor_skew_ms=float(args.max_sensor_skew_ms),
                        teacher_path_length_contract_id=TEACHER_PATH_LENGTH_CONTRACT_ID,
                        teacher_plan_path_max_m=TEACHER_PLAN_PATH_MAX_M,
                        teacher_actual_path_max_m=TEACHER_ACTUAL_PATH_MAX_M,
                        actual_path_length_m=float(actual_path_length_m),
                        collection_config_contract_id=collection_contract[
                            "collection_config_contract_id"
                        ],
                        resolved_config_sha256=collection_contract["resolved_config_sha256"],
                        mission_index_sha256=mission_index_sha256,
                        collision_cache_sha256=collision_cache_sha256,
                        code_version_sha256=code_version_sha256,
                        teacher_algorithm_changed=False,
                    )
                    save_rollout_episode(
                        dataset_path_obj,
                        arrays,
                        metadata,
                        compress=bool(args.compress_episodes),
                    )
                    dataset_path = os.path.relpath(str(dataset_path_obj), str(report_path.parent))
                persist_attempt_provenance()
                outcome.update(
                    {
                        "steps": steps,
                        "actual_path_length_m": actual_path_length_m,
                        "actual_path_limit_m": TEACHER_ACTUAL_PATH_MAX_M,
                        "path_length_exceeded": path_length_exceeded,
                        "path_length_contract_id": TEACHER_PATH_LENGTH_CONTRACT_ID,
                        "success": success,
                        "collision": collision,
                        "dead_end": dead_end,
                        "hard_altitude": hard_altitude,
                        "stop_reason": stop_reason,
                        "execute_ok": accepted,
                        "dataset_npz": dataset_path,
                        "max_endpoint_error_m": max_endpoint_error,
                        "max_stream_drift_m": 0.0,
                        "tracking_drift_replans": 0,
                        "invalid_prefetch_replans": 0,
                        "predicted_deadend_replans": 0,
                        "prefetch_ready_count": 0,
                        "prefetch_miss_count": 0,
                        "planner_blocking_s": planner_blocking_s,
                        "prefetch_compute_s": 0.0,
                        "global_route_length_m": float(route_info["route_length_m"]),
                        "global_route_points": int(route_info["route_points"]),
                    }
                )
            except GlobalRouteUnavailableError as error:
                env.stop()
                persist_attempt_provenance()
                outcome.update(
                    {
                        "stop_reason": "mission_route_unavailable",
                        "execute_ok": False,
                        "error": repr(error),
                    }
                )
            except Exception as error:
                env.stop()
                persist_attempt_provenance()
                outcome.update(
                    {
                        "stop_reason": "collector_error",
                        "execute_ok": False,
                        "error": repr(error),
                        "snapshot_missing_count": max(
                            int(outcome.get("snapshot_missing_count", 0) or 0),
                            int(
                                "snapshot" in str(error).lower()
                                or "authoritative" in str(error).lower()
                            ),
                        ),
                    }
                )
            outcome["elapsed_s"] = time.monotonic() - episode_started
            attempted_this_run += 1
            resume_ledger.add_row(outcome)
            report_journal.append(outcome)
            if parse_bool(outcome.get("execute_ok", False)):
                accepted_count_runtime += 1
                persist_partial_rollout_index()
            persist_progress("running", outcome)
            run_logger.event(
                "accepted" if parse_bool(outcome.get("execute_ok", False)) else "rejected",
                mission_id=outcome.get("mission_id", ""),
                episode=episode_id,
                step=int(outcome.get("steps", 0) or 0),
                duration_ms=float(outcome["elapsed_s"]) * 1000.0,
                stop_reason=outcome.get("stop_reason", ""),
                accepted_total=accepted_count_runtime,
                attempted_total=resume_ledger.attempted_total,
            )
            print_progress(
                component="teacher_rollout_collection_reliable_exact",
                processed=resume_ledger.attempted_total,
                passing=accepted_count_runtime,
                estimated_stop=(
                    global_target_accepted
                    if global_target_accepted > 0
                    else (
                        global_max_episodes
                        if global_max_episodes > 0
                        else len(source_rows)
                    )
                ),
                elapsed_s=time.monotonic() - run_started,
                baseline_processed=existing_report_count,
                eta_processed=(
                    accepted_count_runtime
                    if global_target_accepted > 0
                    else resume_ledger.attempted_total
                ),
                eta_baseline_processed=(
                    accepted_at_run_start
                    if global_target_accepted > 0
                    else attempted_at_run_start
                ),
                eta_target=(
                    global_target_accepted
                    if global_target_accepted > 0
                    else (
                        global_max_episodes
                        if global_max_episodes > 0
                        else len(source_rows)
                    )
                ),
                rolling_throughput_per_s=latest_progress_metrics.get(
                    "rolling_throughput_per_s"
                ),
                ewma_throughput_per_s=latest_progress_metrics.get(
                    "ewma_throughput_per_s"
                ),
                minimum_calibration_s=10.0,
                minimum_calibration_items=5,
                accepted_per_s=round(
                    float(latest_progress_metrics.get("accepted_per_s", 0.0)), 6
                ),
                reliable_rows_per_s=round(
                    float(latest_progress_metrics.get("reliable_rows_per_s", 0.0)), 6
                ),
                elapsed=format_duration(time.monotonic() - run_started),
                worker=args.worker_id,
                episode=episode_id,
                outcome=outcome.get("stop_reason"),
                steps=outcome.get("steps", 0),
            )
            if (
                int(args.target_accepted) > 0
                and accepted_count_runtime >= int(args.target_accepted)
            ):
                break

    except KeyboardInterrupt:
        interrupted = True
        persist_partial_rollout_index(force=True)
        persist_progress("interrupted")
        report_journal.close()
    except Exception:
        report_journal.close()
        raise
    finally:
        if env is not None:
            env.stop()
        if backend is not None:
            _close_formal_backend(backend)
        behavior_teacher.close()
        geometry_context.close()
        route_store.close()

    if interrupted:
        run_logger.close()
        print("TEACHER_RELIABLE_EXACT_COLLECTION_INTERRUPTED")
        return 130

    all_reports = report_journal.recover()
    report_journal.commit(report_path)
    report_journal.close()
    accepted_rows = [row for row in all_reports if parse_bool(row.get("execute_ok", False))]
    accepted_rows.sort(key=lambda row: int(float(row["episode_id"])))
    write_csv_atomic(rollout_index_path, accepted_rows, fieldnames=FORMAL_REPORT_FIELDS)
    rollout_partial_index_path.unlink(missing_ok=True)
    counters = {
        "reliable_rows": sum(
            _formal_report_counter(row, "reliable_rows") for row in all_reports
        ),
        "legacy_rows": sum(
            _formal_report_counter(row, "legacy_rows") for row in all_reports
        ),
        "telemetry_lookup_count": sum(
            _formal_report_counter(row, "telemetry_lookup_count")
            for row in all_reports
        ),
        "snapshot_missing_count": sum(
            _formal_report_counter(row, "snapshot_missing_count")
            for row in all_reports
        ),
        "state_depth_skew_max_ns": max(
            (
                _formal_report_counter(row, "state_depth_skew_max_ns")
                for row in all_reports
            ),
            default=0,
        ),
        "exact_primitive_frame_count_failures": sum(
            _formal_report_counter(row, "frame_contract_failures")
            for row in all_reports
        ),
        "endpoint_identity_chain_valid": all(
            parse_bool(row.get("endpoint_identity_chain_valid", False))
            for row in all_reports
        ),
    }
    summary = {
        "workers": int(args.num_workers),
        "worker_id": int(args.worker_id),
        "collection_run_id": run_id,
        "runtime_instance_id": runtime_instance_id,
        "unity_player_sha256": runtime_identity["unity_player_sha256"],
        "runtime_assembly_sha256": runtime_identity["runtime_assembly_sha256"],
        "bridge_sha256": runtime_identity["bridge_sha256"],
        "protocol_version": collection_contract["shared_collection_config"][
            "protocol_version"
        ],
        "attempted_total": len(all_reports),
        "accepted_total": len(accepted_rows),
        "target_accepted": int(global_target_accepted),
        "accept_rate": len(accepted_rows) / max(1, len(all_reports)),
        "rollout_index": rollout_index_path.name,
        "collection_report": report_path.name,
        "collection_report_journal": report_journal_path.name,
        "offline_relabel_required": True,
        **exact_endpoint_metadata(),
        "asynchronous_prefetch": False,
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "reliable_execution": True,
        "telemetry_observation": False,
        "reliable_rows": int(counters["reliable_rows"]),
        "legacy_rows": 0,
        "telemetry_lookup_count": int(counters["telemetry_lookup_count"]),
        "snapshot_missing_count": int(counters["snapshot_missing_count"]),
        "state_depth_skew_max_ns": int(counters["state_depth_skew_max_ns"]),
        "exact_primitive_frame_count_failures": int(
            counters["exact_primitive_frame_count_failures"]
        ),
        "endpoint_identity_chain_valid": bool(counters["endpoint_identity_chain_valid"]),
        "mpl_duration_s": float(mpl.duration_s),
        "mpl_forward_distance_m": float(mpl.forward_distance_m),
        "mpl_contract_sha256": str(mpl.contract_sha256),
        "teacher_planning_contract_id": behavior_teacher.teacher_planning_contract_id,
        "teacher_config": teacher_cfg.__dict__,
        "teacher_path_length_contract_id": TEACHER_PATH_LENGTH_CONTRACT_ID,
        "max_actual_path_length_m": TEACHER_ACTUAL_PATH_MAX_M,
        "max_steps": int(args.max_steps),
        **task_contract_fields(int(args.max_steps)),
        "stream_horizon": 1,
        "max_sensor_skew_ms": float(args.max_sensor_skew_ms),
        "max_endpoint_error_m": float(args.max_endpoint_error_m),
        "max_stream_drift_m": float(args.max_stream_drift_m),
        "mission_index_sha256": mission_index_sha256,
        "collision_cache_sha256": collision_cache_sha256,
        "code_version_sha256": code_version_sha256,
        "collection_config_contract_id": collection_contract[
            "collection_config_contract_id"
        ],
        "resolved_config_sha256": collection_contract["resolved_config_sha256"],
        "merge_compatibility_sha256": collection_contract[
            "merge_compatibility_sha256"
        ],
        "shared_collection_config": collection_contract["shared_collection_config"],
        "resolved_cli_args": collection_contract["resolved_cli_args"],
        "worker_runtime": collection_contract["worker_runtime"],
        "runtime_artifact_identity": runtime_identity,
        "resolved_collection_config": "resolved_collection_config.json",
        "attempted_this_run": int(attempted_this_run),
        "target_accepted_worker_local": int(args.target_accepted),
        **route_provenance,
    }
    write_json_atomic(out_dir / "collection_summary.json", summary)
    persist_progress("completed")
    run_logger.event(
        "completed",
        episode=-1,
        attempted_count=len(all_reports),
        accepted_count=len(accepted_rows),
        reliable_rows=summary["reliable_rows"],
    )
    run_logger.close()
    print("TEACHER_RELIABLE_EXACT_COLLECTION_SUMMARY")
    print("  attempted_total:", summary["attempted_total"])
    print("  accepted_total:", summary["accepted_total"])
    print("  reliable_rows:", summary["reliable_rows"])
    print("  legacy_rows:", summary["legacy_rows"])
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
