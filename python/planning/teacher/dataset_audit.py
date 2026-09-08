#!/usr/bin/env python3
"""Audit rollout episodes and optional offline teacher labels."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os

import numpy as np

from planning.diagnostics.action_distribution import summarize_action_histogram
from planning.data.rollout import TeacherLabelStore, load_rollout_episode, resolve_dataset_path
from planning.contracts.feature import NUM_ACTIONS
from planning.mission.spec import MISSION_PLANAR_DISTANCE_M, validate_mission_rows
from planning.primitives.library import MotionPrimitiveLibrary
from planning.safety.depth_mask import DepthActionMaskStore
from planning.contracts.pipeline_provenance import (
    validate_cross_artifact_consistency,
    validate_rollout_provenance,
)
from planning.contracts.task import task_contract_fields
from planning.data.mission_routes import (
    MissionRouteStore,
    resolve_route_store_prefix,
    route_store_provenance,
    validate_route_references,
)
from planning.common import read_csv, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument(
        "--route-store-prefix",
        default="",
        help="committed MissionRouteStore prefix; defaults to the rollout row reference",
    )
    parser.add_argument("--labels", default="")
    parser.add_argument(
        "--depth-masks",
        default="",
        help="formal depth-mask artifact; required together with --labels",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--expected-planar-distance", type=float, default=MISSION_PLANAR_DISTANCE_M)
    parser.add_argument("--planar-distance-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--max-endpoint-error-m", type=float, default=0.60)
    parser.add_argument("--max-sensor-skew-ms", type=float, default=80.0)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    index_path = Path(args.index).expanduser().resolve()
    output_path = Path(args.out_json).expanduser().resolve() if args.out_json else index_path.with_suffix(".audit.json")
    rollout_provenance = validate_rollout_provenance(index_path)
    rows = [dict(row) for row in rollout_provenance.rows]
    validate_mission_rows(rows, args.expected_planar_distance, args.planar_distance_tolerance)
    rows.sort(key=lambda row: int(float(row.get("episode_id", 0))))
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
    route_provenance = route_store_provenance(route_store, artifact_path=output_path)

    if bool(args.labels) != bool(args.depth_masks):
        raise ValueError("--labels and --depth-masks must be supplied together")
    label_store = TeacherLabelStore(Path(args.labels)) if args.labels else None
    depth_mask_store = (
        DepthActionMaskStore(Path(args.depth_masks)) if args.depth_masks else None
    )
    cross_artifact = None
    if label_store is not None and depth_mask_store is not None:
        cross_artifact = validate_cross_artifact_consistency(
            rollout_provenance,
            label_store.metadata,
            depth_mask_store.metadata,
        )
    episodes_ok = 0
    transitions = 0
    action_hist = np.zeros((NUM_ACTIONS,), dtype=np.int64)
    invalid_behavior = 0
    invalid_teacher = 0
    soft_sum_errors = []
    endpoint_errors = []
    depth_min = float("inf")
    depth_max = -float("inf")
    sensor_skew_ms_max = 0.0
    failures = []

    for row in rows:
        path = resolve_dataset_path(row, index_path)
        try:
            episode = load_rollout_episode(path, validate=True)
            behavior = np.asarray(episode["behavior_actions"], dtype=np.int64)
            count = int(behavior.shape[0])
            transitions += count
            action_hist += np.bincount(behavior, minlength=NUM_ACTIONS)
            execution_mask = np.asarray(episode["execution_action_masks"], dtype=np.bool_)
            invalid_behavior += int(np.count_nonzero(~execution_mask[np.arange(count), behavior]))
            depths = np.asarray(episode["depths"], dtype=np.float32)
            depth_min = min(depth_min, float(np.min(depths)))
            depth_max = max(depth_max, float(np.max(depths)))
            sensor_skew_ms = np.asarray(episode["sensor_skew_ns"], dtype=np.float64) * 1.0e-6
            sensor_skew_ms_max = max(sensor_skew_ms_max, float(np.max(sensor_skew_ms)))
            if float(np.max(sensor_skew_ms)) > float(args.max_sensor_skew_ms) + 1.0e-6:
                raise ValueError(
                    "sensor skew {:.3f}ms exceeds {:.3f}ms".format(
                        float(np.max(sensor_skew_ms)), float(args.max_sensor_skew_ms)
                    )
                )
            error = float(row.get("max_endpoint_error_m", 0.0) or 0.0)
            endpoint_errors.append(error)
            if error > float(args.max_endpoint_error_m):
                raise ValueError("max endpoint error {:.3f} exceeds {:.3f}".format(error, args.max_endpoint_error_m))
            if label_store is not None:
                labels = label_store.get_episode(path, expected_behavior_actions=behavior)
                teacher = np.asarray(labels["teacher_argmax"], dtype=np.int64)
                masks = np.asarray(labels["global_action_masks"], dtype=np.bool_)
                in_range = (teacher >= 0) & (teacher < NUM_ACTIONS)
                safe_teacher = np.clip(teacher, 0, NUM_ACTIONS - 1)
                invalid_teacher += int(np.count_nonzero(~in_range | ~masks[np.arange(count), safe_teacher]))
                sums = np.asarray(labels["soft_targets"], dtype=np.float32).sum(axis=1)
                soft_sum_errors.extend(np.abs(sums[in_range] - 1.0).tolist())
                local_masks = depth_mask_store.get_episode(path, count)
                if local_masks.shape != (count, NUM_ACTIONS):
                    raise ValueError(
                        "depth-mask shape {} != ({}, {})".format(
                            local_masks.shape, count, NUM_ACTIONS
                        )
                    )
            episodes_ok += 1
        except Exception as error:
            failures.append(
                {
                    "episode_id": row.get("episode_id", "?"),
                    "path": os.path.relpath(str(path), str(output_path.parent)),
                    "error": repr(error),
                }
            )

    motion_primitives = MotionPrimitiveLibrary()
    action_distribution = summarize_action_histogram(action_hist, motion_primitives)
    plan_path_stretches = [
        float(row["teacher_plan_path_stretch"])
        for row in rows
        if str(row.get("teacher_plan_path_stretch", "")).strip()
    ]
    summary = {
        "episodes": len(rows),
        "episodes_ok": episodes_ok,
        "episodes_failed": len(failures),
        "transitions": transitions,
        "action_coverage": int(np.count_nonzero(action_hist)),
        "action_distribution": action_distribution,
        "teacher_plan_path_stretch_mean": (
            float(np.mean(plan_path_stretches)) if plan_path_stretches else None
        ),
        "teacher_plan_path_stretch_gt_1_05_rate": (
            float(np.mean(np.asarray(plan_path_stretches) > 1.05))
            if plan_path_stretches
            else None
        ),
        "invalid_behavior_count": invalid_behavior,
        "invalid_behavior_rate": invalid_behavior / max(1, transitions),
        "invalid_teacher_count": invalid_teacher,
        "invalid_teacher_rate": invalid_teacher / max(1, transitions),
        "soft_sum_error_max": float(max(soft_sum_errors)) if soft_sum_errors else 0.0,
        "endpoint_error_max": float(max(endpoint_errors)) if endpoint_errors else 0.0,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "sensor_skew_ms_max": sensor_skew_ms_max,
        "labels_checked": bool(label_store is not None),
        "depth_masks_checked": bool(depth_mask_store is not None),
        "provenance_checked": bool(label_store is not None and depth_mask_store is not None),
        "observation_contract": (
            "reliable_exact_endpoint_snapshot"
            if label_store is not None and depth_mask_store is not None
            else ""
        ),
        "observation_source": (
            "reliable_exact_endpoint_snapshot"
            if label_store is not None and depth_mask_store is not None
            else ""
        ),
        "input_rollout_index_sha256": (
            cross_artifact.get("input_rollout_index_sha256", "")
            if cross_artifact is not None
            else ""
        ),
        "input_rollout_manifest_sha256": (
            cross_artifact.get("input_rollout_manifest_sha256", "")
            if cross_artifact is not None
            else ""
        ),
        "reliable_rows": int(transitions),
        "legacy_rows": 0,
        "failures": failures[:20],
        **task_contract_fields(
            int(
                rollout_provenance.root_metadata.get(
                    "max_primitive_steps",
                    rollout_provenance.root_metadata.get("max_steps"),
                )
            )
        ),
        "max_steps": int(
            rollout_provenance.root_metadata.get(
                "max_steps", rollout_provenance.root_metadata["max_primitive_steps"]
            )
        ),
        **route_provenance,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, summary)
    print("TEACHER_DATASET_AUDIT_SUMMARY")
    for key, value in summary.items():
        if key != "failures":
            print("  {}: {}".format(key, value))
    print("  out_json:", output_path)
    quality_ok = bool(
        len(failures) == 0
        and summary["invalid_behavior_rate"] == 0.0
        and summary["invalid_teacher_rate"] <= 1.0e-3
        and summary["soft_sum_error_max"] <= 5.0e-3
    )
    print("RESULT={}".format("PASS" if quality_ok else "FAIL"))
    route_store.close()
    return 0 if quality_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
