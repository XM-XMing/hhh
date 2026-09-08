#!/usr/bin/env python3
"""Offline exact soft-label generation from actual Unity rollout states."""

from __future__ import annotations

from pathlib import Path

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np

from planning.data.rollout import (
    LABEL_CONTRACT_ID,
    load_rollout_episode_fields,
    resolve_dataset_path,
)
from planning.contracts.feature import FEATURE_CONTRACT_ID
from planning.mission.global_route import (
    GLOBAL_ROUTE_CONTRACT_ID,
    GlobalRouteConfig,
    global_route_identity,
)
from planning.mission.spec import task_contract_fields
from planning.safety.collision_checker import (
    COLLISION_BACKEND_CONTRACT_ID,
    COLLISION_SOURCE_FILES,
    COLLISION_SOURCE_ID,
    collision_backend_cli_type,
    resolve_collision_backend,
)
from planning.teacher.label_worker import init_label_worker, label_rollout_episode
from planning.primitives.library import MotionPrimitiveLibrary, resolve_package_path
from planning.teacher.policy import (
    TEACHER_PLANNING_CONTRACT_ID,
    add_teacher_arguments,
    config_from_args,
)
from planning.common.config import pre_bc_value
from planning.contracts.teacher_path import (
    TEACHER_ACTUAL_PATH_MAX_M,
    TEACHER_PATH_LENGTH_CONTRACT_ID,
    TEACHER_PLAN_PATH_MAX_M,
)
from planning.common import file_sha256, print_progress, read_csv, write_json_atomic, write_npz_atomic
from planning.contracts.pipeline_provenance import (
    artifact_provenance_metadata,
    validate_artifact_provenance,
    validate_rollout_provenance,
)
from planning.data.mission_routes import (
    MissionRouteStore,
    resolve_route_store_prefix,
    route_store_provenance,
    validate_route_references,
)


def write_sample_csv(
    path: Path,
    results: List[Dict],
    limit: int = 200,
    arrays: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    rows = []
    for result in results:
        length = int(result["length"])
        offset = int(result.get("offset", 0))
        if arrays is None:
            result_arrays = result
        else:
            result_arrays = {
                "teacher_argmax": arrays["teacher_argmax"][offset : offset + length],
                "soft_targets": arrays["soft_targets"][offset : offset + length],
                "behavior_actions": arrays["behavior_actions"][offset : offset + length],
                "valid_counts": arrays["valid_counts"][offset : offset + length],
                "entropies": arrays["entropies"][offset : offset + length],
            }
        for index in range(length):
            teacher = int(result_arrays["teacher_argmax"][index])
            soft = np.asarray(result_arrays["soft_targets"][index], dtype=np.float32)
            top_ids = np.argsort(soft)[::-1][:5]
            rows.append(
                {
                    "episode_id": int(result["episode_id"]),
                    "transition": index,
                    "behavior_action": int(result_arrays["behavior_actions"][index]),
                    "teacher_argmax": teacher,
                    "valid_count": int(result_arrays["valid_counts"][index]),
                    "entropy": float(result_arrays["entropies"][index]),
                    "top5_actions": " ".join(str(int(value)) for value in top_ids),
                    "top5_probabilities": " ".join("{:.4f}".format(float(soft[value])) for value in top_ids),
                }
            )
            if len(rows) >= int(limit):
                break
        if len(rows) >= int(limit):
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["episode_id"])
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/teach/flight/rollouts/rollout_index.csv")
    parser.add_argument("--out-labels", default="data/teach/flight/rollouts/teacher_labels.npz")
    parser.add_argument(
        "--route-store-prefix",
        default="",
        help="committed MissionRouteStore prefix; defaults to the rollout row reference",
    )
    parser.add_argument("--collision-cache", default="data/map_data/forest_voxels_10cm.npz")
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--inflate-radius", type=float, default=0.35)
    parser.add_argument(
        "--num-workers", type=int, default=int(pre_bc_value("label", "workers"))
    )
    parser.add_argument("--worker-chunksize", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--collision-backend",
        type=collision_backend_cli_type,
        default="cpu",
        help="compatibility alias for the sole C++17/OpenMP CPU backend",
    )
    parser.add_argument("--collision-threads", type=int, default=int(os.environ.get("PLANNING_COLLISION_THREADS", "1")))
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--include-scores", action="store_true")
    parser.add_argument(
        "--direct-mmap",
        action="store_true",
        help="write worker arrays to disjoint temporary mmap slices before NPZ publication",
    )
    add_teacher_arguments(parser, candidate_top_k=12, current_check_step=1, lookahead_check_step=2)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if int(args.num_workers) <= 0:
        raise ValueError("--num-workers must be positive")
    if int(args.worker_chunksize) <= 0:
        raise ValueError("--worker-chunksize must be positive")
    if int(args.max_episodes) < 0:
        raise ValueError("--max-episodes must be non-negative")
    if int(args.collision_threads) <= 0:
        raise ValueError("--collision-threads must be positive")
    resolved_collision_backend = resolve_collision_backend(args.collision_backend)
    os.environ["PLANNING_COLLISION_BACKEND"] = resolved_collision_backend
    os.environ["PLANNING_COLLISION_THREADS"] = str(
        int(args.collision_threads)
    )

    index_path = Path(args.index).expanduser().resolve()
    out_path = Path(args.out_labels).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_provenance = validate_rollout_provenance(index_path)
    max_primitive_steps = int(
        rollout_provenance.root_metadata["max_primitive_steps"]
    )
    rows = [dict(row) for row in rollout_provenance.rows]
    if int(args.max_episodes) > 0:
        rows = rows[: int(args.max_episodes)]
        rollout_provenance = rollout_provenance.subset(
            int(float(row["episode_id"])) for row in rows
        )
    if not rows:
        raise RuntimeError("no accepted rollout rows")
    route_store_prefix = resolve_route_store_prefix(
        rows,
        index_path,
        Path(args.route_store_prefix) if str(args.route_store_prefix).strip() else None,
    )
    route_store = MissionRouteStore.open(route_store_prefix, validate=True)
    validate_route_references(
        rows,
        route_store,
        candidate_index_sha256=route_store.metadata["candidate_index_sha256"],
        require_provenance=True,
    )
    route_provenance = route_store_provenance(route_store, artifact_path=out_path)
    ordered_rows = sorted(rows, key=lambda row: int(float(row["episode_id"])))
    tasks = []
    lengths_from_index = []
    for row in ordered_rows:
        path = resolve_dataset_path(row, index_path)
        raw_length = str(row.get("steps", "")).strip()
        try:
            length = int(float(raw_length))
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            fields = load_rollout_episode_fields(path, ("behavior_actions",))
            length = int(np.asarray(fields["behavior_actions"]).shape[0])
        if length <= 0:
            raise ValueError("rollout has no transitions: {}".format(path))
        lengths_from_index.append(length)
        tasks.append((int(float(row["episode_id"])), str(path)))
    lengths = np.asarray(lengths_from_index, dtype=np.int64)
    offsets = np.zeros_like(lengths)
    if len(offsets) > 1:
        offsets[1:] = np.cumsum(lengths[:-1])
    total_transitions = int(lengths.sum())
    direct_building = None
    direct_arrays = None
    if bool(args.direct_mmap):
        direct_building = out_path.with_name(out_path.name + ".building")
        if direct_building.exists():
            raise FileExistsError("direct label building directory already exists: {}".format(direct_building))
        direct_building.mkdir(parents=True, exist_ok=False)
        direct_specs = {
            "soft_targets": (np.float16, (total_transitions, 105)),
            "global_masks": (np.bool_, (total_transitions, 105)),
            "teacher_argmax": (np.int16, (total_transitions,)),
            "behavior_actions": (np.int16, (total_transitions,)),
            "valid_counts": (np.int16, (total_transitions,)),
            "entropies": (np.float32, (total_transitions,)),
        }
        if bool(args.include_scores):
            direct_specs["teacher_scores"] = (np.float32, (total_transitions, 105))
        direct_arrays = {
            key: np.lib.format.open_memmap(
                str(direct_building / (key + ".npy")),
                mode="w+",
                dtype=dtype,
                shape=shape,
            )
            for key, (dtype, shape) in direct_specs.items()
        }
        direct_tasks = [
            (task[0], task[1], int(offset), int(length))
            for task, offset, length in zip(tasks, offsets, lengths)
        ]
        tasks = direct_tasks

    config = config_from_args(args)
    route_identity = global_route_identity(
        GlobalRouteConfig(
            resolution_m=float(config.global_route_resolution_m),
            flight_z_min_m=float(config.z_min),
            flight_z_max_m=float(config.z_max),
            lookahead_m=float(config.global_route_lookahead_m),
            tracking_margin_m=float(config.global_route_tracking_margin_m),
        )
    )
    route_cache_path = resolve_package_path(args.collision_cache)
    route_identity["global_route_map_identity"] = {
        "cache_path": str(args.collision_cache),
        "cache_sha256": file_sha256(route_cache_path),
        "voxel_size_m": float(args.voxel_size),
        "inflate_radius_m": float(args.inflate_radius),
    }
    collision_payload = {
        "voxel_size": float(args.voxel_size),
        "inflate_radius": float(args.inflate_radius),
        "cache_npz": str(args.collision_cache),
    }
    mpl_contract = MotionPrimitiveLibrary()

    print("OFFLINE_TEACHER_LABELING_START")
    print("  index:", index_path)
    print("  episodes:", len(tasks))
    print("  num_workers:", args.num_workers)
    print("  worker_chunksize:", args.worker_chunksize)
    print("  collision_backend:", args.collision_backend)
    print("  collision_threads:", args.collision_threads)
    print("  candidate_top_k:", args.candidate_top_k)
    print("  actual_state_relabeling: True")
    started = time.time()
    results = []
    direct_output = None if direct_building is None else {
        "building": str(direct_building),
        "total": total_transitions,
        "include_scores": bool(args.include_scores),
    }
    if int(args.num_workers) <= 1:
        init_label_worker(
            config.__dict__,
            collision_payload,
            bool(args.include_scores),
            str(route_store_prefix),
            direct_output,
        )
        for index, task in enumerate(tasks, start=1):
            results.append(label_rollout_episode(task))
            if index % max(1, int(args.progress_interval)) == 0 or index == len(tasks):
                print_progress(
                    component="teacher_labeling",
                    processed=index,
                    passing=index,
                    estimated_stop=len(tasks),
                    elapsed_s=time.time() - started,
                )
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=int(args.num_workers),
            initializer=init_label_worker,
            initargs=(
                config.__dict__,
                collision_payload,
                bool(args.include_scores),
                str(route_store_prefix),
                direct_output,
            ),
        ) as pool:
            iterator = pool.imap(label_rollout_episode, tasks, chunksize=max(1, int(args.worker_chunksize)))
            for index, result in enumerate(iterator, start=1):
                results.append(result)
                if index % max(1, int(args.progress_interval)) == 0 or index == len(tasks):
                    print_progress(
                        component="teacher_labeling",
                        processed=index,
                        passing=index,
                        estimated_stop=len(tasks),
                        elapsed_s=time.time() - started,
                    )

    results.sort(key=lambda result: int(result["episode_id"]))
    if direct_arrays is None:
        soft_targets = np.concatenate([result["soft_targets"] for result in results], axis=0)
        global_masks = np.concatenate([result["global_masks"] for result in results], axis=0)
        teacher_argmax = np.concatenate([result["teacher_argmax"] for result in results], axis=0)
        behavior_actions = np.concatenate([result["behavior_actions"] for result in results], axis=0)
        valid_counts = np.concatenate([result["valid_counts"] for result in results], axis=0)
        entropies = np.concatenate([result["entropies"] for result in results], axis=0)
        lengths = np.asarray([result["length"] for result in results], dtype=np.int64)
        offsets = np.zeros_like(lengths)
        if len(lengths) > 1:
            offsets[1:] = np.cumsum(lengths[:-1])
    else:
        for result, offset in zip(results, offsets):
            result["offset"] = int(offset)
        soft_targets = direct_arrays["soft_targets"]
        global_masks = direct_arrays["global_masks"]
        teacher_argmax = direct_arrays["teacher_argmax"]
        behavior_actions = direct_arrays["behavior_actions"]
        valid_counts = direct_arrays["valid_counts"]
        entropies = direct_arrays["entropies"]
    paths = [os.path.relpath(str(result["path"]), str(out_path.parent)) for result in results]

    metadata = {
        "label_contract_id": LABEL_CONTRACT_ID,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        **task_contract_fields(max_primitive_steps),
        "global_route_contract_id": GLOBAL_ROUTE_CONTRACT_ID,
        "teacher_planning_contract_id": TEACHER_PLANNING_CONTRACT_ID,
        "teacher_path_length_contract_id": TEACHER_PATH_LENGTH_CONTRACT_ID,
        "teacher_plan_path_max_m": TEACHER_PLAN_PATH_MAX_M,
        "teacher_actual_path_max_m": TEACHER_ACTUAL_PATH_MAX_M,
        "labeler": "label_teacher_rollouts.py",
        "actual_state_relabeling": True,
        "episodes": len(results),
        "transitions": int(teacher_argmax.shape[0]),
        "teacher_config": config.__dict__,
        "include_scores": bool(args.include_scores),
        "mpl_duration_s": float(mpl_contract.duration_s),
        "mpl_forward_distance_m": float(mpl_contract.forward_distance_m),
        "mpl_contract_sha256": str(mpl_contract.contract_sha256),
        "collision_voxel_size_m": float(args.voxel_size),
        "collision_inflate_radius_m": float(args.inflate_radius),
        **route_provenance,
    }
    metadata.update(route_identity)
    producer_provenance = artifact_provenance_metadata(
        rollout_provenance,
        artifact_path=out_path,
        artifact_kind="teacher_labels",
        row_count=int(teacher_argmax.shape[0]),
        episode_ids=[int(result["episode_id"]) for result in results],
        transition_offsets=[int(value) for value in offsets],
        transition_lengths=[int(value) for value in lengths],
        extra=route_provenance,
    )
    metadata.update(producer_provenance)
    validate_artifact_provenance(
        metadata,
        artifact_kind="teacher_labels",
        expected_row_count=int(teacher_argmax.shape[0]),
        expected_episode_ids=[int(result["episode_id"]) for result in results],
        expected_offsets=[int(value) for value in offsets],
        expected_lengths=[int(value) for value in lengths],
        require_route_store=True,
    )
    payload = {
        "soft_targets": soft_targets,
        "global_action_masks": global_masks,
        "teacher_argmax": teacher_argmax,
        "behavior_actions": behavior_actions,
        "valid_counts": valid_counts,
        "teacher_entropies": entropies,
        "episode_npz_paths": np.asarray(paths),
        "episode_offsets": offsets,
        "episode_lengths": lengths,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if bool(args.include_scores):
        payload["teacher_scores"] = (
            direct_arrays["teacher_scores"]
            if direct_arrays is not None
            else np.concatenate([result["teacher_scores"] for result in results], axis=0)
        )
    for array in (direct_arrays or {}).values():
        array.flush()
    try:
        write_npz_atomic(out_path, payload, compress=bool(args.compress))
    except Exception:
        raise

    transitions = int(teacher_argmax.shape[0])
    behavior_valid = sum(int(result["behavior_valid"]) for result in results)
    teacher_valid = sum(int(result["teacher_valid"]) for result in results)
    teacher_matches = sum(int(result["teacher_matches_behavior"]) for result in results)
    mask_matches = sum(int(result["mask_match_count"]) for result in results)
    valid_teacher_rows = teacher_argmax >= 0
    soft_sums = soft_targets.astype(np.float32).sum(axis=1)
    summary = {
        "episodes": len(results),
        "transitions": transitions,
        "behavior_valid_rate": behavior_valid / max(1, transitions),
        "teacher_valid_rate": teacher_valid / max(1, int(np.count_nonzero(valid_teacher_rows))),
        "teacher_argmax_equals_behavior_rate": teacher_matches / max(1, transitions),
        "collector_mask_equals_relabel_mask_rate": mask_matches / max(1, transitions),
        "valid_count_mean": float(np.mean(valid_counts)),
        "valid_count_min": int(np.min(valid_counts)),
        "dead_end_transition_count": int(np.count_nonzero(valid_counts <= 0)),
        "training_ready": bool(np.all(valid_counts > 0)),
        "valid_count_max": int(np.max(valid_counts)),
        "teacher_entropy_mean": float(np.mean(entropies[valid_teacher_rows])) if np.any(valid_teacher_rows) else 0.0,
        "soft_sum_error_max": float(np.max(np.abs(soft_sums[valid_teacher_rows] - 1.0))) if np.any(valid_teacher_rows) else 0.0,
        "elapsed_s": time.time() - started,
        "num_workers": int(args.num_workers),
        "worker_chunksize": int(args.worker_chunksize),
        "collision_backend": str(args.collision_backend),
        "collision_backend_requested": str(args.collision_backend),
        "collision_backend_contract_id": COLLISION_BACKEND_CONTRACT_ID,
        "collision_source_id": COLLISION_SOURCE_ID,
        "collision_source_files": list(COLLISION_SOURCE_FILES),
        "collision_threads": int(args.collision_threads),
        "out_labels": out_path.name,
    }
    summary.update(producer_provenance)
    summary.update(route_identity)
    summary_path = out_path.with_suffix(".summary.json")
    write_json_atomic(summary_path, summary)
    sample_path = out_path.with_suffix(".sample.csv")
    write_sample_csv(sample_path, results, arrays=direct_arrays)
    if direct_building is not None:
        shutil.rmtree(str(direct_building))

    print("OFFLINE_TEACHER_LABELING_SUMMARY")
    for key in (
        "episodes",
        "transitions",
        "behavior_valid_rate",
        "teacher_valid_rate",
        "teacher_argmax_equals_behavior_rate",
        "collector_mask_equals_relabel_mask_rate",
        "valid_count_mean",
        "valid_count_min",
        "dead_end_transition_count",
        "training_ready",
        "teacher_entropy_mean",
        "soft_sum_error_max",
        "elapsed_s",
        "num_workers",
    ):
        print("  {}: {}".format(key, summary[key]))
    print("  out_labels:", out_path)
    print("  sample_csv:", sample_path)
    route_store.close()

    behavior_quality_ok = summary["behavior_valid_rate"] >= 0.995
    quality_ok = bool(
        behavior_quality_ok
        and summary["teacher_valid_rate"] >= 0.999
        and summary["soft_sum_error_max"] <= 5.0e-3
    )
    print("RESULT={}".format("PASS" if quality_ok else "FAIL"))
    return 0 if quality_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
