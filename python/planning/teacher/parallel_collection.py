"""Parallel teacher rollout collection orchestration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple

from planning.common import canonical_json_sha256, file_sha256
from planning.common.logging import (
    CollectionLogSession,
    bridge_log_path,
    collection_log_path,
    prepare_log_directories,
    unity_log_path,
    worker_log_path,
    write_resume_manifest,
)
from planning.common.paths import planning_package_root, planning_workspace_root
from planning.common.progress import (
    ProgressRateTracker,
    format_duration,
    format_progress,
)
from planning.contracts.collection import source_tree_sha256
from planning.data.rollout_merge import merge_worker_rollouts
from planning.data.rollout_shard import worker_shard_paths
from planning.runtime.ports import build_worker_runtime_specs
from planning.runtime.worker import write_worker_runtime_spec_file
from planning.runtime.supervisor import RuntimeSupervisor
from planning.runtime.identity import runtime_artifact_identity
from planning.runtime.bridge_identity import planning_bridge_binary
from planning.primitives.library import MotionPrimitiveLibrary
from planning.teacher.collection_config import ParallelCollectionConfig
from planning.teacher.collection_run import (
    CollectionResumeLedger,
    aggregate_target_status,
)
from planning.teacher.collection_disk import DiskSpaceWatch, disk_capacity_gate
from planning.common.config import pre_bc_value


DEFAULT_COLLECTION_WORKERS = int(pre_bc_value("collection", "workers"))


def _count_existing_reports(out_dir: Path) -> Tuple[int, int]:
    attempted = 0
    accepted = 0
    workers_root = Path(out_dir) / "workers"
    for report_path in workers_root.glob("worker_*/collection_report.csv"):
        try:
            with report_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    attempted += 1
                    accepted += str(row.get("execute_ok", "")).strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "pass",
                    }
        except FileNotFoundError:
            continue
    return attempted, accepted


def _count_progress(out_dir: Path) -> Tuple[int, int]:
    attempted = 0
    accepted = 0
    workers_root = Path(out_dir) / "workers"
    for progress_path in workers_root.glob("worker_*/collection_progress.json"):
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            attempted += int(payload.get("attempted_total", 0))
            accepted += int(payload.get("accepted_total", 0))
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
    return attempted, accepted


def _collector_args(
    *,
    config: ParallelCollectionConfig,
    spec,
    worker_dir: Path,
    global_stop_file: Path,
    collision_cache: Path,
) -> Tuple[str, ...]:
    backend = spec.python_backend_config()
    args = [
        sys.executable,
        "-u",
        str(planning_package_root() / "scripts" / "collect_teacher_rollouts.py"),
        "--index",
        str(config.mission_index),
        "--out-dir",
        str(worker_dir),
        "--worker-id",
        str(spec.worker_id),
        "--num-workers",
        str(config.active_workers),
        "--max-episodes",
        "0",
        "--target-accepted",
        "0",
        "--stop-file",
        str(global_stop_file),
        "--max-steps",
        str(config.max_steps),
        "--stream-horizon",
        "1",
        "--max-stream-drift-m",
        str(config.max_stream_drift_m),
        "--max-endpoint-error-m",
        str(config.max_endpoint_error_m),
        "--max-actual-path-length-m",
        str(config.max_actual_path_length_m),
        "--worker-ready-timeout",
        str(config.worker_ready_timeout_s),
        "--max-sensor-skew-ms",
        str(config.max_sensor_skew_ms),
        "--collision-cache",
        str(collision_cache),
        "--voxel-size",
        str(config.voxel_size),
        "--inflate-radius",
        str(config.inflate_radius),
        "--prefetch-inflate-radius",
        str(config.prefetch_inflate_radius),
        "--candidate-top-k",
        str(config.candidate_top_k),
        "--current-check-step",
        str(config.current_check_step),
        "--lookahead-check-step",
        str(config.lookahead_check_step),
        "--expected-planar-distance",
        str(config.expected_planar_distance),
        "--reliable-exact-runtime-instance-id",
        str(spec.runtime_instance_id),
        "--reliable-exact-command-endpoint",
        str(backend["command_endpoint"]),
        "--reliable-exact-result-endpoint",
        str(backend["result_endpoint"]),
        "--reliable-exact-snapshot-endpoint",
        str(backend["snapshot_endpoint"]),
        "--reliable-exact-timeout",
        str(config.reliable_timeout_s),
        "--global-route-resolution-m",
        str(config.global_route_resolution_m),
        "--global-route-lookahead-m",
        str(config.global_route_lookahead_m),
        "--global-route-tracking-margin-m",
        str(config.global_route_tracking_margin_m),
        "--beam-depth",
        str(config.beam_depth),
        "--beam-width",
        str(config.beam_width),
        "--beam-branching",
        str(config.beam_branching),
        "--beam-discount",
        str(config.beam_discount),
    ]
    if config.compress_episodes:
        args.append("--compress-episodes")
    # The formal collector is exact by construction.  Keep the historical
    # option in the resolved CLI contract, but do not create an async path.
    return tuple(args)


def _collector_environment(
    *,
    config: ParallelCollectionConfig,
    mission_sha256: str,
    collision_sha256: str,
    code_sha256: str,
    unity_bin: Path | None = None,
    bridge_binary: Path | None = None,
) -> dict[str, str]:
    package_root = planning_package_root()
    inherited_pythonpath = str(os.environ.get("PYTHONPATH", "")).strip()
    pythonpath = [str(package_root / "python")]
    if inherited_pythonpath:
        pythonpath.append(inherited_pythonpath)
    return {
        "PLANNING_COLLISION_THREADS": str(config.collision_threads),
        "PLANNING_COLLECTION_RUN_ID": config.collection_run_id,
        "PLANNING_TRAINING_RUN_ID": config.training_run_id,
        "PLANNING_RUNTIME_LAUNCH_NONCE": config.runtime_launch_nonce,
        "PLANNING_COLLECTION_WORKERS_REQUESTED": str(config.workers_requested),
        "PLANNING_COLLECTION_GLOBAL_MAX_EPISODES": str(config.max_episodes),
        "PLANNING_COLLECTION_GLOBAL_TARGET_ACCEPTED": str(config.target_accepted),
        "PLANNING_COLLECTION_OBSERVATION_CONTRACT": config.observation_contract,
        "PLANNING_COLLECTION_OBSERVATION_SOURCE": config.observation_source,
        "PLANNING_COLLISION_BACKEND": str(config.collision_backend),
        "PLANNING_DEPTH_SAFETY_BACKEND": str(config.depth_safety_backend),
        "PLANNING_BRIDGE_BINARY": str(bridge_binary or config.bridge_binary or ""),
        "UNITY_BIN": str(unity_bin or config.unity_bin or ""),
        "PLANNING_UNITY_WINDOW_WIDTH": str(config.window_width),
        "PLANNING_UNITY_WINDOW_HEIGHT": str(config.window_height),
        "PLANNING_MISSION_INDEX_SHA256": mission_sha256,
        "PLANNING_COLLISION_CACHE_SHA256": collision_sha256,
        "PLANNING_COLLECTION_CODE_SHA256": code_sha256,
        "PYTHONPATH": os.pathsep.join(pythonpath),
    }


def _print_start_summary(
    *,
    config: ParallelCollectionConfig,
    collision_cache: Path,
    mission_sha256: str,
    collision_sha256: str,
    code_sha256: str,
    logger: CollectionLogSession | None = None,
) -> None:
    rows = (
        ("workers_requested", config.workers_requested),
        ("workers_active", config.active_workers),
        ("run_id", config.collection_run_id),
        ("mission_index", config.mission_index),
        ("mission_index_sha256", mission_sha256),
        ("out_dir", config.out_dir),
        ("max_episodes_global", config.max_episodes),
        ("target_accepted_global", config.target_accepted),
        ("collector_program", "collect_teacher_rollouts.py"),
        ("unity_window", f"{config.window_width}x{config.window_height}"),
        ("max_steps", config.max_steps),
        ("stream_horizon", config.stream_horizon),
        ("candidate_top_k", config.candidate_top_k),
        ("max_actual_path_length_m", config.max_actual_path_length_m),
        ("collision_cache", collision_cache),
        ("collision_cache_sha256", collision_sha256),
        ("code_version_sha256", code_sha256),
        ("collision_backend", config.collision_backend),
        ("depth_safety_backend", config.depth_safety_backend),
        ("teacher_beam", "depth=3 width=8 branching=4 discount=0.95"),
        (
            "global_route",
            "resolution={} lookahead={} tracking_margin={}".format(
                config.global_route_resolution_m,
                config.global_route_lookahead_m,
                config.global_route_tracking_margin_m,
            ),
        ),
        ("collision_threads_per_worker", config.collision_threads),
        (
            "total_collision_threads_max",
            config.active_workers * config.collision_threads,
        ),
        ("disk_warning_free_gb", config.disk_warning_free_gb),
        ("disk_stop_free_gb", config.disk_stop_free_gb),
        ("disk_check_interval_s", config.disk_check_interval_s),
        ("disk_safety_margin_gb", config.disk_safety_margin_gb),
    )
    if logger is None:
        print("PARALLEL_TEACHER_COLLECTION_START")
        for key, value in rows:
            print(f"  {key}: {value}")
        return
    logger.event("PARALLEL_TEACHER_COLLECTION_START", **dict(rows))


def _resolved_runtime_paths(config: ParallelCollectionConfig) -> Tuple[Path, Path, Path]:
    """Resolve runtime paths from the config owner, never from ambient env."""

    package_root = planning_package_root().resolve()
    workspace = planning_workspace_root().resolve()
    unity_bin = Path(config.unity_bin or workspace / "src" / "unity" / "XMflight.x86_64").resolve()
    bridge_binary = Path(
        config.bridge_binary
        or planning_bridge_binary(package_root)
    ).resolve()
    collision_cache = Path(
        config.collision_cache
        or package_root / "data" / "map_data" / "forest_voxels_10cm.npz"
    ).resolve()
    return unity_bin, bridge_binary, collision_cache


def _runtime_manifest_payload(
    *,
    config: ParallelCollectionConfig,
    unity_bin: Path,
    bridge_binary: Path,
    collision_cache: Path,
    mission_sha256: str,
    collision_sha256: str,
    code_sha256: str,
) -> Tuple[dict, dict]:
    """Build the safe root identity/config payloads before any worker starts."""

    runtime_identity = runtime_artifact_identity(
        player=unity_bin,
        assembly=(
            unity_bin.parent
            / (unity_bin.stem + "_Data")
            / "Managed"
            / "Assembly-CSharp.dll"
        ),
        bridge=bridge_binary,
    )
    mpl = MotionPrimitiveLibrary()
    mpl_identity = {
        "npz_path": str(mpl.npz_path),
        "npz_sha256": file_sha256(mpl.npz_path),
        "json_path": str(mpl.metadata_path),
        "json_sha256": file_sha256(mpl.metadata_path),
        "contract_sha256": str(mpl.contract_sha256),
        "actions": int(mpl.num_actions),
        "frames": int(mpl.frames),
        "reference_samples": int(mpl.reference_path(0).shape[0]),
    }
    runtime_payload = {
        "run_id": str(config.collection_run_id),
        "runtime_identity": runtime_identity,
        "bridge_binary": str(bridge_binary),
        "collision_cache": str(collision_cache),
        "collision_cache_sha256": collision_sha256,
        "mpl": mpl_identity,
        "observation_contract": config.observation_contract,
    }
    resolved_payload = {
        "run_id": str(config.collection_run_id),
        "training_run_id": str(config.training_run_id),
        "runtime_launch_nonce": str(config.runtime_launch_nonce),
        "mission_index_sha256": mission_sha256,
        "collision_cache_sha256": collision_sha256,
        "code_version_sha256": code_sha256,
        "runtime_identity_sha256": canonical_json_sha256(runtime_payload, ensure_ascii=False),
        "mpl_contract_sha256": mpl_identity["contract_sha256"],
        "observation_contract": config.observation_contract,
        "observation_source": config.observation_source,
        "task_contract": config.task_contract_fields,
        "resolved_config": config.resolved_fields(),
    }
    return runtime_payload, resolved_payload


def run_parallel_collection(config: ParallelCollectionConfig) -> int:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "workers").mkdir(parents=True, exist_ok=True)
    (config.out_dir / "ros").mkdir(parents=True, exist_ok=True)
    prepare_log_directories(config.out_dir)
    logger = CollectionLogSession(
        collection_log_path(config.out_dir),
        run_id=config.collection_run_id,
        command="scripts/collect_rollouts_parallel.py",
    )
    status = "FAIL"
    supervisor = None
    try:
        unity_bin, bridge_binary, collision_cache = _resolved_runtime_paths(config)
        logger.event(
            "resolved runtime paths",
            unity_bin=str(unity_bin),
            bridge_binary=str(bridge_binary),
            collision_cache=str(collision_cache),
        )
        if not unity_bin.is_file() or not os.access(unity_bin, os.X_OK):
            raise FileNotFoundError(
                "Unity executable not found or not executable: {}".format(unity_bin)
            )
        if not collision_cache.is_file():
            raise FileNotFoundError("collision cache not found: {}".format(collision_cache))
        if not bridge_binary.is_file() or not os.access(bridge_binary, os.X_OK):
            raise FileNotFoundError(
                "Bridge executable not found or not executable: {}".format(bridge_binary)
            )

        mission_sha256 = file_sha256(config.mission_index)
        collision_sha256 = file_sha256(collision_cache)
        code_sha256 = source_tree_sha256()
        runtime_payload, resolved_payload = _runtime_manifest_payload(
            config=config,
            unity_bin=unity_bin,
            bridge_binary=bridge_binary,
            collision_cache=collision_cache,
            mission_sha256=mission_sha256,
            collision_sha256=collision_sha256,
            code_sha256=code_sha256,
        )
        resolved_document = write_resume_manifest(
            config.out_dir / "resolved_config.json",
            resolved_payload,
            immutable_keys=tuple(resolved_payload.keys()),
            schema="collection_resolved_config",
        )
        command_payload = {
            "run_id": str(config.collection_run_id),
            "entrypoint": "scripts/collect_rollouts_parallel.py",
            "argv": list(config.command_argv),
            "resolved_config_sha256": canonical_json_sha256(
                resolved_payload, ensure_ascii=False
            ),
        }
        write_resume_manifest(
            config.out_dir / "command.json",
            command_payload,
            immutable_keys=("run_id", "entrypoint", "resolved_config_sha256"),
            schema="collection_command",
        )
        write_resume_manifest(
            config.out_dir / "runtime_identity.json",
            runtime_payload,
            immutable_keys=tuple(runtime_payload.keys()),
            schema="collection_runtime_identity",
        )
        logger.event(
            "resolved configuration",
            resolved_config_path=str(config.out_dir / "resolved_config.json"),
            resolved_config_sha256=canonical_json_sha256(
                resolved_payload, ensure_ascii=False
            ),
            resolved_config=resolved_document.get("resolved_config", {}),
        )
        logger.event("runtime identity", **runtime_payload)

        global_stop_file = config.out_dir / ".collection_stop"
        global_stop_file.unlink(missing_ok=True)
        specs = build_worker_runtime_specs(
            worker_count=config.active_workers,
            master_port_base=config.master_base,
            command_port_base=config.command_base,
            depth_port_base=config.depth_base,
            port_stride=config.port_stride,
            runtime_instance_template=(
                "xmflight-{training_run_id}-{runtime_launch_nonce}-w{worker_id:02d}"
            ),
            ros_root=config.out_dir / "ros",
            training_run_id=config.training_run_id,
            runtime_launch_nonce=config.runtime_launch_nonce,
        )
        write_worker_runtime_spec_file(config.out_dir / "worker_runtime_specs.json", specs)
        logger.event(
            "worker runtime specs resolved",
            workers=int(config.active_workers),
            runtime_instances=[spec.runtime_instance_id for spec in specs],
            port_maps=[spec.port_mapping() for spec in specs],
        )

        # A formal resume may only consume reports produced by the same exact
        # worker identity and collection run.  Legacy reports fail closed here,
        # before any runtime process is started.
        for spec in specs:
            report_path = worker_shard_paths(
                config.out_dir / "workers", spec.worker_id
            )["report"]
            if not report_path.is_file() or report_path.stat().st_size == 0:
                continue
            with report_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            CollectionResumeLedger.from_rows(
                rows,
                runtime_instance_id=spec.runtime_instance_id,
                run_id=config.collection_run_id,
            )

        existing_attempted, existing_accepted = _count_existing_reports(config.out_dir)
        logger.event(
            "resume state inspected",
            attempted=existing_attempted,
            accepted=existing_accepted,
            target_accepted=config.target_accepted,
            max_episodes=config.max_episodes,
        )
        if config.target_accepted > 0 and existing_accepted >= config.target_accepted:
            logger.event(
                "global accepted target already satisfied",
                accepted=existing_accepted,
                target=config.target_accepted,
            )
            merge_status = merge_worker_rollouts(
                workers_dir=config.out_dir / "workers",
                out_dir=config.out_dir,
                expected_workers=config.active_workers,
            )
            status = "PASS" if merge_status == 0 else "FAIL"
            return int(merge_status)
        if config.max_episodes > 0 and existing_attempted >= config.max_episodes:
            logger.event(
                "global attempt limit already satisfied",
                attempted=existing_attempted,
                max_episodes=config.max_episodes,
            )
            merge_status = merge_worker_rollouts(
                workers_dir=config.out_dir / "workers",
                out_dir=config.out_dir,
                expected_workers=config.active_workers,
            )
            status = "PASS" if merge_status == 0 else "FAIL"
            return int(merge_status)

        remaining_target = max(0, int(config.target_accepted) - int(existing_accepted))
        capacity_target = remaining_target if config.target_accepted > 0 else max(
            0, int(config.max_episodes) - int(existing_attempted)
        )
        capacity = disk_capacity_gate(
            config.out_dir,
            target_accepted=capacity_target,
            minimum_free_gb=config.disk_stop_free_gb,
            safety_margin_gb=config.disk_safety_margin_gb,
            estimated_episode_bytes=config.disk_estimated_episode_bytes,
            reference_roots=(
                config.out_dir,
                planning_package_root() / "data",
            ),
        )
        logger.event(
            "disk capacity gate",
            result=capacity.reason,
            passed=capacity.passed,
            free_bytes=capacity.snapshot.free_bytes,
            free_gb=round(capacity.snapshot.free_gb, 3),
            target_accepted=capacity.estimate.target_accepted,
            estimated_episode_bytes=capacity.estimate.estimated_episode_bytes,
            sampled_episode_count=capacity.estimate.sampled_episode_count,
            estimated_required_gb=round(capacity.estimate.required_gb, 3),
            required_free_gb=round(
                float(capacity.required_free_bytes) / float(1024 ** 3), 3
            ),
        )
        if not capacity.passed:
            raise RuntimeError(
                "DISK_CAPACITY_GATE_FAIL: free_gb={:.3f} required_gb={:.3f}".format(
                    capacity.snapshot.free_gb,
                    float(capacity.required_free_bytes) / float(1024 ** 3),
                )
            )
        _print_start_summary(
            config=config,
            collision_cache=collision_cache,
            mission_sha256=mission_sha256,
            collision_sha256=collision_sha256,
            code_sha256=code_sha256,
            logger=logger,
        )

        collector_environment = _collector_environment(
            config=config,
            mission_sha256=mission_sha256,
            collision_sha256=collision_sha256,
            code_sha256=code_sha256,
            unity_bin=unity_bin,
            bridge_binary=bridge_binary,
        )
        supervisor = RuntimeSupervisor(
            specs=specs,
            out_dir=config.out_dir,
            unity_bin=unity_bin,
            bridge_binary=bridge_binary,
            window_width=config.window_width,
            window_height=config.window_height,
            startup_timeout_s=config.startup_timeout_s,
        )

        started = time.monotonic()
        disk_watch = DiskSpaceWatch(
            config.out_dir,
            warning_free_gb=config.disk_warning_free_gb,
            stop_free_gb=config.disk_stop_free_gb,
            interval_s=config.disk_check_interval_s,
        )
        disk_stop_requested = False
        eta_progress_tracker = ProgressRateTracker(
            initial_completed=(
                existing_accepted
                if config.target_accepted > 0
                else existing_attempted
            )
        )
        logger.event(
            "worker launch requested",
            workers=int(config.active_workers),
            mission_index=str(config.mission_index),
            out_dir=str(config.out_dir),
            target_accepted=int(config.target_accepted),
        )
        supervisor.start()
        for worker in supervisor.workers:
            logger.event(
                "worker ready",
                worker_id=worker.spec.worker_id,
                runtime_instance_id=worker.spec.runtime_instance_id,
                worker_log=str(worker_log_path(config.out_dir, worker.spec.worker_id)),
                unity_log=str(unity_log_path(config.out_dir, worker.spec.worker_id)),
                bridge_log=str(bridge_log_path(config.out_dir, worker.spec.worker_id)),
            )
            logger.event(
                "worker collector launch requested",
                worker_id=worker.spec.worker_id,
                mission_shard_rule="episode_id % workers == worker_id",
                worker_count=int(config.active_workers),
            )
            worker.start_collector(
                argv=_collector_args(
                    config=config,
                    spec=worker.spec,
                    worker_dir=worker.worker_dir,
                    global_stop_file=global_stop_file,
                    collision_cache=collision_cache,
                ),
                environment=collector_environment,
            )

        while supervisor.active_collectors() > 0:
            attempted, accepted = _count_progress(config.out_dir)
            active_collectors = supervisor.active_collectors()
            target_status = aggregate_target_status(
                accepted,
                config.target_accepted,
                active_workers=active_collectors,
            )
            estimated_stop = (
                config.target_accepted
                if config.target_accepted > 0
                else config.max_episodes
            )
            eta_completed = (
                accepted if config.target_accepted > 0 else attempted
            )
            eta_baseline = (
                existing_accepted
                if config.target_accepted > 0
                else existing_attempted
            )
            eta_snapshot = eta_progress_tracker.update(eta_completed)
            logger.event(
                "progress",
                progress=format_progress(
                    component="teacher_parallel_collection",
                    processed=attempted,
                    passing=accepted,
                    estimated_stop=estimated_stop,
                    elapsed_s=max(0.0, time.monotonic() - started),
                    baseline_processed=existing_attempted,
                    eta_processed=accepted
                    if config.target_accepted > 0
                    else attempted,
                    eta_baseline_processed=existing_accepted
                    if config.target_accepted > 0
                    else existing_attempted,
                    eta_target=estimated_stop,
                    rolling_throughput_per_s=eta_snapshot["rolling_rate_per_s"],
                    ewma_throughput_per_s=eta_snapshot["ewma_rate_per_s"],
                    minimum_calibration_s=10.0,
                    minimum_calibration_items=5,
                    active_workers=active_collectors,
                    accepted_per_s=round(
                        float(eta_snapshot["stable_rate_per_s"]), 4
                    ),
                    rejected=max(0, int(attempted) - int(accepted)),
                    elapsed=format_duration(max(0.0, time.monotonic() - started)),
                ),
                attempted=attempted,
                accepted=accepted,
                active_workers=active_collectors,
            )
            disk_event = disk_watch.check()
            if disk_event is not None and disk_event.level in {"WARNING", "STOP"}:
                logger.event(
                    "disk capacity warning" if disk_event.level == "WARNING" else "DISK_CAPACITY_STOP",
                    level="WARNING" if disk_event.level == "WARNING" else "ERROR",
                    free_bytes=disk_event.snapshot.free_bytes,
                    free_gb=round(disk_event.snapshot.free_gb, 3),
                    warning_free_gb=config.disk_warning_free_gb,
                    stop_free_gb=config.disk_stop_free_gb,
                    previous_level=disk_event.previous_level,
                )
                if disk_event.level == "STOP" and not global_stop_file.exists():
                    disk_stop_requested = True
                    try:
                        global_stop_file.touch()
                    except OSError as error:
                        logger.exception(
                            error,
                            message="disk stop marker write failed; stopping supervisor",
                        )
                        supervisor.stop()
            if target_status == "TARGET_REACHED" and not global_stop_file.exists():
                logger.event("global accepted target reached; workers stop after current episode")
                global_stop_file.touch()
            if config.max_episodes > 0 and attempted >= config.max_episodes:
                if not global_stop_file.exists():
                    logger.event("global attempt limit reached; workers stop after current episode")
                    global_stop_file.touch()
            time.sleep(2.0)

        failed = []
        for worker in supervisor.workers:
            return_code = worker.wait_collector()
            if return_code == 0:
                logger.event("collector finished", worker_id=worker.spec.worker_id)
            else:
                failed.append(worker.spec.worker_id)
                logger.event(
                    "collector failed",
                    level="ERROR",
                    worker_id=worker.spec.worker_id,
                    worker_log=str(worker_log_path(config.out_dir, worker.spec.worker_id)),
                    exit_code=return_code,
                )
        if failed:
            raise RuntimeError("collection workers failed: {}".format(failed))

        final_attempted, final_accepted = _count_existing_reports(config.out_dir)
        final_target_status = aggregate_target_status(
            final_accepted,
            config.target_accepted,
            active_workers=0,
        )
        if final_target_status == "TARGET_UNREACHED":
            logger.event(
                "AGGREGATE_TARGET_UNREACHED",
                level="ERROR",
                accepted=final_accepted,
                target=config.target_accepted,
                attempted=final_attempted,
                active_workers=0,
                eta_status="TARGET_UNREACHABLE",
                eta_h=None,
                mission_pool_exhausted=True,
            )
        if disk_stop_requested:
            logger.event(
                "disk capacity stop completed",
                level="ERROR" if final_target_status == "TARGET_UNREACHED" else "WARNING",
                accepted=final_accepted,
                attempted=final_attempted,
            )

        merge_status = merge_worker_rollouts(
            workers_dir=config.out_dir / "workers",
            out_dir=config.out_dir,
            expected_workers=config.active_workers,
        )
        logger.event(
            "rollout merge complete",
            merge_status=int(merge_status),
            merged_index=str(config.out_dir / "rollout_index.csv"),
        )
        if final_target_status == "TARGET_UNREACHED":
            status = "FAIL"
            return max(1, int(merge_status))
        if merge_status != 0:
            status = "FAIL"
            return int(merge_status)
        logger.event(
            "PARALLEL_TEACHER_COLLECTION_DONE",
            merged_index=str(config.out_dir / "rollout_index.csv"),
            summary=str(config.out_dir / "collection_summary.json"),
            attempted=final_attempted,
            accepted=final_accepted,
        )
        status = "PASS"
        return 0
    except BaseException as error:
        logger.exception(
            error,
            worker_logs=[
                str(worker_log_path(config.out_dir, worker_id))
                for worker_id in range(max(0, int(config.active_workers)))
            ],
        )
        raise
    finally:
        if supervisor is not None:
            supervisor.stop()
            logger.event("runtime cleanup complete", workers=len(supervisor.workers))
        logger.close(status)


def _install_signal_handlers() -> None:
    def _raise_interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGINT, _raise_interrupt)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the canonical flag-style collector parser.

    Defaults remain in ``ParallelCollectionConfig`` and its canonical owners;
    ``None`` here means "let the config resolver decide".  Positional values
    are retained only for the established shell compatibility seam.
    """

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run the canonical parallel reliable-exact Teacher collector.",
    )
    parser.add_argument("legacy_mission_index", nargs="?")
    parser.add_argument("legacy_out_dir", nargs="?")
    parser.add_argument("legacy_workers", nargs="?", type=int)
    parser.add_argument("--mission-index", dest="mission_index")
    parser.add_argument("--out-dir", dest="out_dir")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--target-accepted", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--unity-bin", type=Path)
    parser.add_argument("--bridge", "--bridge-binary", dest="bridge_binary", type=Path)
    parser.add_argument("--collision-cache", type=Path)
    parser.add_argument("--collision-backend")
    parser.add_argument("--depth-safety-backend")
    parser.add_argument("--collision-threads", type=int)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--inflate-radius", type=float)
    parser.add_argument("--prefetch-inflate-radius", type=float)
    parser.add_argument("--expected-planar-distance", type=float)
    parser.add_argument("--reliable-exact-timeout", dest="reliable_timeout_s", type=float)
    parser.add_argument("--stream-horizon", type=int)
    parser.add_argument("--candidate-top-k", type=int)
    parser.add_argument("--current-check-step", type=int)
    parser.add_argument("--lookahead-check-step", type=int)
    parser.add_argument("--max-stream-drift-m", type=float)
    parser.add_argument("--max-endpoint-error-m", type=float)
    parser.add_argument("--max-actual-path-length-m", type=float)
    parser.add_argument("--max-sensor-skew-ms", type=float)
    parser.add_argument("--worker-ready-timeout", dest="worker_ready_timeout_s", type=float)
    parser.add_argument("--startup-timeout", dest="startup_timeout_s", type=float)
    parser.add_argument("--window-width", type=int)
    parser.add_argument("--window-height", type=int)
    parser.add_argument("--master-base", type=int)
    parser.add_argument("--cmd-base", dest="command_base", type=int)
    parser.add_argument("--depth-base", type=int)
    parser.add_argument("--port-stride", type=int)
    parser.add_argument("--compress-episodes", action="store_true", default=None)
    parser.add_argument("--disable-async-prefetch", action="store_true", default=None)
    parser.add_argument("--run-id", "--collection-run-id", dest="collection_run_id")
    parser.add_argument("--training-run-id", dest="training_run_id")
    parser.add_argument("--runtime-launch-nonce", dest="runtime_launch_nonce")
    parser.add_argument("--global-route-resolution-m", type=float)
    parser.add_argument("--global-route-lookahead-m", type=float)
    parser.add_argument("--global-route-tracking-margin-m", type=float)
    parser.add_argument("--beam-depth", type=int)
    parser.add_argument("--beam-width", type=int)
    parser.add_argument("--beam-branching", type=int)
    parser.add_argument("--beam-discount", type=float)
    parser.add_argument("--disk-warning-free-gb", type=float)
    parser.add_argument("--disk-stop-free-gb", type=float)
    parser.add_argument("--disk-check-interval-sec", dest="disk_check_interval_s", type=float)
    parser.add_argument("--disk-safety-margin-gb", type=float)
    parser.add_argument("--disk-estimated-episode-bytes", type=int)
    return parser


_CLI_CONFIG_FIELDS = (
    "target_accepted",
    "max_episodes",
    "max_steps",
    "unity_bin",
    "bridge_binary",
    "collision_cache",
    "collision_backend",
    "depth_safety_backend",
    "collision_threads",
    "voxel_size",
    "inflate_radius",
    "prefetch_inflate_radius",
    "expected_planar_distance",
    "reliable_timeout_s",
    "stream_horizon",
    "candidate_top_k",
    "current_check_step",
    "lookahead_check_step",
    "max_stream_drift_m",
    "max_endpoint_error_m",
    "max_actual_path_length_m",
    "max_sensor_skew_ms",
    "worker_ready_timeout_s",
    "startup_timeout_s",
    "window_width",
    "window_height",
    "master_base",
    "command_base",
    "depth_base",
    "port_stride",
    "compress_episodes",
    "disable_async_prefetch",
    "collection_run_id",
    "training_run_id",
    "runtime_launch_nonce",
    "global_route_resolution_m",
    "global_route_lookahead_m",
    "global_route_tracking_margin_m",
    "beam_depth",
    "beam_width",
    "beam_branching",
    "beam_discount",
    "disk_warning_free_gb",
    "disk_stop_free_gb",
    "disk_check_interval_s",
    "disk_safety_margin_gb",
    "disk_estimated_episode_bytes",
)


def _resolved_cli_values(parser: argparse.ArgumentParser, args) -> Tuple[Path, Path, int, dict]:
    mission_index = args.mission_index or args.legacy_mission_index
    out_dir = args.out_dir or args.legacy_out_dir
    workers = args.workers
    if workers is None:
        workers = args.legacy_workers
    if mission_index is None:
        parser.error("--mission-index is required")
    if out_dir is None:
        parser.error("--out-dir is required")
    if workers is None:
        workers = DEFAULT_COLLECTION_WORKERS
    overrides = {
        field: getattr(args, field)
        for field in _CLI_CONFIG_FIELDS
        if getattr(args, field) is not None
    }
    return Path(mission_index), Path(out_dir), int(workers), overrides


def main(argv: Iterable[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = build_argument_parser()
    args = parser.parse_args(values)
    mission_index, out_dir, workers, overrides = _resolved_cli_values(parser, args)
    overrides["command_argv"] = tuple(str(value) for value in values)
    default_run_id = "teacher-collection-{}".format(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    config = ParallelCollectionConfig.from_environment(
        mission_index=mission_index,
        out_dir=out_dir,
        workers=workers,
        default_run_id=default_run_id,
        overrides=overrides,
    )
    _install_signal_handlers()
    try:
        return run_parallel_collection(config)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
