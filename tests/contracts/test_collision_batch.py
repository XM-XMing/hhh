#!/usr/bin/env python3
"""Verify and benchmark the C++ multi-pose collision batch API."""

from __future__ import annotations

import argparse
import time


import numpy as np

from planning.safety.collision_checker import VoxelCollisionChecker
from planning.primitives.library import MotionPrimitiveLibrary


def make_checker() -> VoxelCollisionChecker:
    voxel_size = 0.10
    origin = np.asarray([-50, -50, -10], dtype=np.int64)
    shape = np.asarray([140, 140, 40], dtype=np.int64)
    # A sparse wall and two posts create both valid and invalid actions.
    xyz = []
    for y in range(-12, 13):
        for z in range(8, 30):
            xyz.append((22, y, z))
    for x, y in ((35, 18), (42, -18)):
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for z in range(8, 30):
                    xyz.append((x + dx, y + dy, z))
    ijk = np.asarray(xyz, dtype=np.int64)
    keys = VoxelCollisionChecker._pack_keys_static(ijk, origin, shape)
    return VoxelCollisionChecker(np.unique(keys), origin, shape, voxel_size, 0.35)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--check-step", type=int, default=2)
    args = parser.parse_args()

    mpl = MotionPrimitiveLibrary()
    checker = make_checker()
    if not checker.fast_enabled:
        raise RuntimeError("C++ collision backend required: {}".format(checker._fast_error))

    count = max(1, int(args.poses))
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(-0.5, 2.5, count)
    positions[:, 1] = np.linspace(-1.2, 1.2, count)
    positions[:, 2] = 2.0
    yaws = np.linspace(-0.35, 0.35, count, dtype=np.float64)
    action_ids = np.arange(mpl.num_actions, dtype=np.int32)

    repeated = [
        checker.check_actions_array(mpl, action_ids, positions[i], yaws[i], check_step=int(args.check_step))
        for i in range(count)
    ]
    batch = checker.check_pose_actions_array(mpl, positions, yaws, action_ids, check_step=int(args.check_step))
    repeated_collisions = np.stack([~row["valid"] for row in repeated])
    repeated_distances = np.stack([row["minimum_distances"] for row in repeated])
    repeated_endpoints = np.stack([row["endpoints"] for row in repeated])
    if not np.array_equal(repeated_collisions, ~batch["valid"]):
        raise AssertionError("batch collision mask differs from repeated calls")
    finite = np.isfinite(repeated_distances) & np.isfinite(batch["minimum_distances"])
    if np.any(finite) and not np.allclose(
        repeated_distances[finite], batch["minimum_distances"][finite], atol=1e-9, rtol=0.0
    ):
        raise AssertionError("batch minimum distances differ")
    if not np.allclose(repeated_endpoints, batch["endpoints"], atol=1e-6, rtol=0.0):
        raise AssertionError("batch endpoints differ")

    repeats = max(1, int(args.repeats))
    started = time.perf_counter()
    for _ in range(repeats):
        for i in range(count):
            checker.check_all_actions(
                mpl, positions[i], yaws[i], check_step=int(args.check_step), return_path_map=False
            )
    legacy_s = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        checker.check_pose_actions_array(mpl, positions, yaws, action_ids, check_step=int(args.check_step))
    batch_s = time.perf_counter() - started
    speedup = legacy_s / max(batch_s, 1e-12)

    print("COLLISION_BATCH_TEST_SUMMARY")
    print("  backend:", checker._fast.library_path)
    print("  cpp_threads:", checker._fast.thread_count)
    print("  poses:", count)
    print("  actions_per_pose:", mpl.num_actions)
    print("  check_step:", int(args.check_step))
    print("  repeats:", repeats)
    print("  legacy_python_rows_s: {:.6f}".format(legacy_s))
    print("  batch_s: {:.6f}".format(batch_s))
    print("  teacher_scan_speedup: {:.3f}".format(speedup))
    print("  outputs_equal: True")
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
