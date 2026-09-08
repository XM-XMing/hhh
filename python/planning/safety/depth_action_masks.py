#!/usr/bin/env python3
"""Generate reusable local depth-only action masks for rollout supervision."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np

from planning.data.rollout import load_rollout_episode_fields, resolve_dataset_path
from planning.safety.depth_action_mask_worker import build_episode_masks, init_worker
from planning.safety.depth_mask import DEPTH_ACTION_MASK_CONTRACT_ID
from planning.safety.depth_safety import DepthSafetyConfig
from planning.contracts.feature import NUM_ACTIONS
from planning.contracts.pipeline_provenance import (
    DEPTH_MASK_CONTRACT_SHA256,
    artifact_provenance_metadata,
    validate_artifact_provenance,
    validate_rollout_provenance,
)
from planning.common.config import pre_bc_value
from planning.data.mission_routes import (
    MissionRouteStore,
    resolve_route_store_prefix,
    route_store_provenance,
    validate_route_references,
)
from planning.common import print_progress, read_csv, write_json_atomic, write_npz_atomic


DEFAULT_DEPTH_SAFETY_CONFIG = DepthSafetyConfig()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/teach/flight/rollouts/rollout_index.csv")
    parser.add_argument("--out-masks", default="data/teach/flight/rollouts/teacher_labels.npz")
    parser.add_argument(
        "--route-store-prefix",
        default="",
        help="committed MissionRouteStore prefix; defaults to the rollout row reference",
    )
    parser.add_argument(
        "--num-workers", type=int, default=int(pre_bc_value("depth_mask", "workers"))
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument(
        "--direct-mmap",
        action="store_true",
        help="write worker masks to disjoint temporary mmap slices before NPZ publication",
    )
    parser.add_argument("--depth-min-m", type=float, default=0.30)
    parser.add_argument("--depth-max-m", type=float, default=3.00)
    parser.add_argument("--unknown-depth-threshold-m", type=float, default=2.95)
    parser.add_argument(
        "--collision-radius-m",
        type=float,
        default=DEFAULT_DEPTH_SAFETY_CONFIG.collision_radius_m,
    )
    parser.add_argument(
        "--depth-slack-m",
        type=float,
        default=DEFAULT_DEPTH_SAFETY_CONFIG.depth_slack_m,
    )
    parser.add_argument(
        "--path-sample-stride",
        type=int,
        default=DEFAULT_DEPTH_SAFETY_CONFIG.path_sample_stride,
    )
    parser.add_argument(
        "--patch-radius-px",
        type=int,
        default=DEFAULT_DEPTH_SAFETY_CONFIG.patch_radius_px,
    )
    parser.add_argument(
        "--max-patch-radius-px",
        type=int,
        default=DEFAULT_DEPTH_SAFETY_CONFIG.max_patch_radius_px,
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if int(args.num_workers) <= 0:
        raise ValueError("--num-workers must be positive")
    if float(args.unknown_depth_threshold_m) <= float(args.depth_min_m):
        raise ValueError("--unknown-depth-threshold-m must exceed --depth-min-m")

    index_path = Path(args.index).expanduser().resolve()
    out_path = Path(args.out_masks).expanduser().resolve()
    rollout_provenance = validate_rollout_provenance(index_path)
    rows = [dict(row) for row in rollout_provenance.rows]
    if int(args.max_episodes) > 0:
        rows = rows[:int(args.max_episodes)]
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
            fields = load_rollout_episode_fields(path, ("depths",))
            length = int(np.asarray(fields["depths"]).shape[0])
        if length <= 0:
            raise ValueError("rollout has no depth transitions: {}".format(path))
        lengths_from_index.append(length)
        tasks.append((int(float(row["episode_id"])), str(path)))
    lengths = np.asarray(lengths_from_index, dtype=np.int64)
    offsets = np.zeros_like(lengths)
    if len(offsets) > 1:
        offsets[1:] = np.cumsum(lengths[:-1])
    total_transitions = int(lengths.sum())
    if len({path for _, path in tasks}) != len(tasks):
        raise ValueError("rollout index contains duplicate dataset_npz paths")

    direct_building = None
    direct_arrays = None
    if bool(args.direct_mmap):
        direct_building = out_path.with_name(out_path.name + ".building")
        if direct_building.exists():
            raise FileExistsError("direct depth-mask building directory already exists: {}".format(direct_building))
        direct_building.mkdir(parents=True, exist_ok=False)
        direct_arrays = {
            "local_action_masks": np.lib.format.open_memmap(
                str(direct_building / "local_action_masks.npy"),
                mode="w+",
                dtype=np.bool_,
                shape=(total_transitions, NUM_ACTIONS),
            )
        }
        tasks = [
            (task[0], task[1], int(offset), int(length))
            for task, offset, length in zip(tasks, offsets, lengths)
        ]

    config_payload = {
        "min_forward_m": DEFAULT_DEPTH_SAFETY_CONFIG.min_forward_m,
        "path_sample_stride": max(1, int(args.path_sample_stride)),
        "collision_radius_m": float(args.collision_radius_m),
        "depth_slack_m": float(args.depth_slack_m),
        "patch_radius_px": max(0, int(args.patch_radius_px)),
        "max_patch_radius_px": max(0, int(args.max_patch_radius_px)),
        "valid_depth_min_m": float(args.depth_min_m),
        "valid_depth_max_m": float(args.unknown_depth_threshold_m),
    }
    print("DEPTH_ACTION_MASK_GENERATION_START")
    print("  index:", index_path)
    print("  episodes:", len(tasks))
    print("  num_workers:", int(args.num_workers))
    print("  source: saved_depth_only")
    started = time.time()
    results = []
    direct_output = None if direct_building is None else {
        "building": str(direct_building),
        "total": total_transitions,
    }
    if int(args.num_workers) == 1:
        init_worker(
            config_payload,
            args.depth_min_m,
            args.depth_max_m,
            str(route_store_prefix),
            direct_output,
        )
        iterator = map(build_episode_masks, tasks)
        pool = None
    else:
        # fork avoids rosrun relay-module pickling failures seen with spawn.
        context = mp.get_context("fork")
        pool = context.Pool(
            processes=int(args.num_workers),
            initializer=init_worker,
            initargs=(
                config_payload,
                args.depth_min_m,
                args.depth_max_m,
                str(route_store_prefix),
                direct_output,
            ),
        )
        iterator = pool.imap(build_episode_masks, tasks, chunksize=1)
    try:
        for count, result in enumerate(iterator, start=1):
            results.append(result)
            if count % max(1, int(args.progress_interval)) == 0 or count == len(tasks):
                print_progress(
                    component="depth_action_mask_generation",
                    processed=count,
                    passing=count,
                    estimated_stop=len(tasks),
                    elapsed_s=time.time() - started,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    results.sort(key=lambda item: item[0])
    paths = [path for _, path, _ in results]
    if direct_arrays is None:
        lengths = np.asarray([mask.shape[0] for _, _, mask in results], dtype=np.int64)
        offsets = np.zeros_like(lengths)
        if len(offsets) > 1:
            offsets[1:] = np.cumsum(lengths[:-1])
        masks = np.concatenate([mask for _, _, mask in results], axis=0)
    else:
        lengths = np.asarray([int(length) for _, _, length in results], dtype=np.int64)
        offsets = np.zeros_like(lengths)
        if len(offsets) > 1:
            offsets[1:] = np.cumsum(lengths[:-1])
        masks = direct_arrays["local_action_masks"]
    metadata = {
        "depth_action_mask_contract_id": DEPTH_ACTION_MASK_CONTRACT_ID,
        "generator": "generate_depth_action_masks.py",
        "episodes": len(results),
        "transitions": int(masks.shape[0]),
        "depth_source": "saved_normalized_depth",
        "depth_min_m": float(args.depth_min_m),
        "depth_max_m": float(args.depth_max_m),
        "unknown_depth_threshold_m": float(args.unknown_depth_threshold_m),
        "depth_safety_config": config_payload,
        **route_provenance,
    }
    producer_provenance = artifact_provenance_metadata(
        rollout_provenance,
        artifact_path=out_path,
        artifact_kind="depth_masks",
        row_count=int(masks.shape[0]),
        episode_ids=[int(episode_id) for episode_id, _, _ in results],
        transition_offsets=[int(value) for value in offsets],
        transition_lengths=[int(value) for value in lengths],
        extra={
            "depth_mask_contract_id": DEPTH_ACTION_MASK_CONTRACT_ID,
            "depth_mask_contract_sha256": DEPTH_MASK_CONTRACT_SHA256,
            "collision_radius_m": float(args.collision_radius_m),
            "depth_slack_m": float(args.depth_slack_m),
            "path_sample_stride": config_payload["path_sample_stride"],
            "max_patch_radius_px": config_payload["max_patch_radius_px"],
            "action_count": NUM_ACTIONS,
            **route_provenance,
        },
    )
    metadata.update(producer_provenance)
    validate_artifact_provenance(
        metadata,
        artifact_kind="depth_masks",
        expected_row_count=int(masks.shape[0]),
        expected_episode_ids=[int(episode_id) for episode_id, _, _ in results],
        expected_offsets=[int(value) for value in offsets],
        expected_lengths=[int(value) for value in lengths],
        require_route_store=True,
    )
    payload = {
        "local_action_masks": masks,
        "episode_npz_paths": np.asarray(
            [os.path.relpath(str(path), str(out_path.parent)) for path in paths]
        ),
        "episode_offsets": offsets,
        "episode_lengths": lengths,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for array in (direct_arrays or {}).values():
        array.flush()
    write_npz_atomic(out_path, payload, compress=bool(args.compress))
    summary = {
        "episodes": len(results),
        "transitions": int(masks.shape[0]),
        "valid_actions_mean": float(np.mean(masks.sum(axis=1))),
        "valid_actions_min": int(np.min(masks.sum(axis=1))),
        "valid_actions_max": int(np.max(masks.sum(axis=1))),
        "elapsed_s": time.time() - started,
        "out_masks": out_path.name,
    }
    summary.update(producer_provenance)
    write_json_atomic(out_path.with_suffix(".summary.json"), summary)
    print("DEPTH_ACTION_MASK_GENERATION_SUMMARY")
    for key, value in summary.items():
        print("  {}: {}".format(key, value))
    print("RESULT=PASS")
    if direct_building is not None:
        for array in direct_arrays.values():
            array.flush()
        shutil.rmtree(str(direct_building))
    route_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
