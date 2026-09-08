#!/usr/bin/env python3
"""Run the O/P depth-safety parity and integration gates.

The original tree is imported only in isolated subprocesses and remains the
behavior source of truth.  All generated inputs and result files live under
``/tmp``; this comparator never writes project data or invokes Unity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


O_ROOT = Path("/home/xm/XM/xm_ws/src/planning").resolve()
P_ROOT = Path("/home/xm/XM/src").resolve()
DEFAULT_ARTIFACT_DIR = Path("/tmp/xmflight_depth_safety_c4")
DEFAULT_MPL = Path("/tmp/xmflight_global_route_c3/c3_mpl.npz")
DEFAULT_MPL_METADATA = Path("/tmp/xmflight_global_route_c3/c3_mpl.json")
DEFAULT_O_LIBRARY = Path(
    "/tmp/xm-cxx-c0-o-devel/planning/lib/libplanning_depth_safety.so"
)
DEFAULT_P_LIBRARY = Path(
    "/tmp/xm-cxx-c2-2-p-devel/planning/lib/libplanning_depth_safety.so"
)


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _bitwise_equal(lhs: Any, rhs: Any) -> bool:
    left = np.ascontiguousarray(np.asarray(lhs))
    right = np.ascontiguousarray(np.asarray(rhs))
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_depth_frames(height: int = 90, width: int = 160) -> Tuple[np.ndarray, np.ndarray]:
    """Return named finite/non-finite edge cases without a project artifact."""

    cx, cy = width // 2, height // 2
    far = np.float32(5.50)
    valid_near = np.float32(0.31)
    exact_min = np.float32(0.30)
    just_below = np.nextafter(exact_min, np.float32(-np.inf), dtype=np.float32)
    just_above = np.nextafter(exact_min, np.float32(np.inf), dtype=np.float32)
    cases: List[Tuple[str, np.ndarray]] = []

    def add(name: str, image: np.ndarray) -> None:
        cases.append((name, np.ascontiguousarray(image, dtype=np.float32)))

    add("all_far", np.full((height, width), far, dtype=np.float32))
    add("all_near", np.full((height, width), valid_near, dtype=np.float32))
    add("constant_medium", np.full((height, width), np.float32(2.0), dtype=np.float32))

    image = np.full((height, width), far, dtype=np.float32)
    image[cy - 4:cy + 5, cx - 4:cx + 5] = np.float32(0.80)
    add("single_center_obstacle", image)

    for name, x0, x1 in (("left_obstacle", 4, 20), ("right_obstacle", width - 20, width - 4)):
        image = np.full((height, width), far, dtype=np.float32)
        image[cy - 8:cy + 9, x0:x1] = np.float32(0.80)
        add(name, image)
    for name, y0, y1 in (("top_obstacle", 3, 18), ("bottom_obstacle", height - 18, height - 3)):
        image = np.full((height, width), far, dtype=np.float32)
        image[y0:y1, cx - 8:cx + 9] = np.float32(0.80)
        add(name, image)

    image = np.full((height, width), far, dtype=np.float32)
    image[:, cx] = np.float32(0.80)
    add("thin_vertical_stripe", image)
    image = np.full((height, width), far, dtype=np.float32)
    image[cy, :] = np.float32(0.80)
    add("thin_horizontal_stripe", image)

    image = np.full((height, width), far, dtype=np.float32)
    image[:3, :3] = np.float32(0.80)
    image[-3:, -3:] = np.float32(0.80)
    add("image_boundary_obstacle", image)
    image = np.full((height, width), far, dtype=np.float32)
    image[cy - 14:cy + 15, cx + 14] = np.float32(0.80)
    image[cy + 14, cx - 14:cx + 15] = np.float32(0.80)
    add("patch_radius_boundary", image)

    for name, value in (
        ("depth_exact_threshold", exact_min),
        ("depth_just_below_threshold", just_below),
        ("depth_just_above_threshold", just_above),
    ):
        image = np.full((height, width), far, dtype=np.float32)
        image[cy - 2:cy + 3, cx - 2:cx + 3] = value
        add(name, image)

    image = np.full((height, width), far, dtype=np.float32)
    image[cy - 2:cy + 3, cx - 2:cx + 3] = np.float32(0.0)
    add("zero_pixels", image)
    image = np.full((height, width), far, dtype=np.float32)
    image[cy - 2:cy + 3, cx - 2:cx + 3] = np.nan
    image[cy + 4:cy + 7, cx + 4:cx + 7] = np.inf
    add("nan_inf_pixels", image)

    image = np.full((height, width), far, dtype=np.float32)
    image[::3, ::5] = np.float32(0.80)
    image[1::4, 2::7] = np.float32(0.0)
    image[2::5, 3::6] = np.nan
    image[3::7, 4::8] = np.float32(6.0)
    add("mixed_invalid_valid", image)

    return (
        np.stack([image for _, image in cases], axis=0),
        np.asarray([name for name, _ in cases]),
    )


def _production_like_depth_frames(
    count: int = 1000, height: int = 90, width: int = 160
) -> Tuple[np.ndarray, np.ndarray]:
    """Create deterministic labeled production-like depth, never real data."""

    rng = np.random.default_rng(20260827)
    cx, cy = width // 2, height // 2
    frames = np.full((int(count), height, width), np.float32(5.50), dtype=np.float32)
    labels = np.empty((int(count),), dtype="U32")
    for index in range(int(count)):
        category = index % 5
        if category == 0:
            labels[index] = "open_space"
            frames[index] += np.float32((index % 11) * 0.01)
        elif category == 1:
            labels[index] = "near_collision"
            x0 = int(rng.integers(15, width - 15))
            y0 = int(rng.integers(15, height - 15))
            radius = int(rng.integers(2, 8))
            frames[index, y0 - radius:y0 + radius + 1, x0 - radius:x0 + radius + 1] = np.float32(
                0.36 + 0.02 * (index % 9)
            )
        elif category == 2:
            labels[index] = "corridor"
            side = 16 + (index % 8)
            frames[index, :, :side] = np.float32(0.90)
            frames[index, :, width - side:] = np.float32(0.90)
            frames[index, cy - 2:cy + 3, cx - 3:cx + 4] = np.float32(4.0)
        elif category == 3:
            labels[index] = "asymmetric_obstacle"
            x0 = 10 + (index * 13) % (width - 35)
            y0 = 8 + (index * 7) % (height - 25)
            frames[index, y0:y0 + 13, x0:x0 + 25] = np.float32(0.65 + 0.01 * (index % 13))
        else:
            labels[index] = "dense_obstacle"
            for block in range(12):
                x0 = int(rng.integers(0, width - 8))
                y0 = int(rng.integers(0, height - 8))
                frames[index, y0:y0 + 5, x0:x0 + 7] = np.float32(0.70 + 0.01 * (block % 5))
    return np.ascontiguousarray(frames), labels


DEPTH_RUNNER = r'''
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

tree = sys.argv[1]
root = Path(sys.argv[2]).resolve()
frames_path = Path(sys.argv[3]).resolve()
result_path = Path(sys.argv[4]).resolve()
sys.path.insert(0, str(root / "python"))

if tree == "O":
    from planning.depth_safety import (
        DepthSafetyConfig,
        local_depth_action_mask,
        primitive_depth_projection_table,
    )
    from planning.motion_primitives import MotionPrimitiveLibrary
else:
    from planning.safety.depth_safety import (
        DepthSafetyConfig,
        local_depth_action_mask,
        primitive_depth_projection_table,
    )
    from planning.primitives.library import MotionPrimitiveLibrary

frames = np.load(str(frames_path), allow_pickle=False)
images = np.ascontiguousarray(frames["images"], dtype=np.float32)
mpl = MotionPrimitiveLibrary()
config = DepthSafetyConfig()
projection = primitive_depth_projection_table(mpl, images.shape[1], images.shape[2], config)
masks = []
checked = []
valid_patches = []
capped = []
max_radius = []
invalid_sum = []
min_clearance = []
mean_invalid = []
counters = []
frame_valid_fraction = []
for image in images:
    mask, info = local_depth_action_mask(mpl, image, config, projection=projection)
    masks.append(np.asarray(mask, dtype=np.bool_))
    checked.append(np.asarray(info["depth_action_checked_sample_count"], dtype=np.int32))
    valid_patches.append(np.asarray(info["depth_action_valid_patch_sample_count"], dtype=np.int32))
    capped.append(np.asarray(info["depth_action_capped_patch_sample_count"], dtype=np.int32))
    max_radius.append(np.asarray(info["depth_action_max_unclipped_patch_radius_px"], dtype=np.int32))
    invalid_sum.append(np.asarray(info["depth_action_mean_invalid_patch_fraction"], dtype=np.float32))
    min_clearance.append(np.asarray(info["depth_action_min_ray_clearance_m"], dtype=np.float32))
    mean_invalid.append(np.asarray(info["depth_action_mean_invalid_patch_fraction"], dtype=np.float32))
    counters.append(np.asarray([
        info["depth_checked_sample_count"],
        info["depth_visible_sample_count"],
        info["depth_blocked_sample_count"],
    ], dtype=np.int64))
    frame_valid_fraction.append(np.float32(info["depth_frame_valid_fraction"]))

action_order = np.asarray(
    [int(mpl.action_metadata(action).get("id", action)) for action in range(mpl.num_actions)],
    dtype=np.int64,
)
np.savez(
    str(result_path),
    masks=np.stack(masks),
    checked=np.stack(checked),
    valid_patches=np.stack(valid_patches),
    capped=np.stack(capped),
    max_radius=np.stack(max_radius),
    invalid_sum=np.stack(invalid_sum),
    min_clearance=np.stack(min_clearance),
    mean_invalid=np.stack(mean_invalid),
    counters=np.stack(counters),
    frame_valid_fraction=np.asarray(frame_valid_fraction, dtype=np.float32),
    projection_pixel_x=np.asarray(projection["pixel_x"], dtype=np.int32),
    projection_pixel_y=np.asarray(projection["pixel_y"], dtype=np.int32),
    projection_point_depth=np.asarray(projection["point_depth_m"], dtype=np.float32),
    projection_radius=np.asarray(projection["patch_radius_px"], dtype=np.int32),
    projection_raw_radius=np.asarray(projection["unclipped_patch_radius_px"], dtype=np.int32),
    projection_valid=np.asarray(projection["valid"], dtype=np.bool_),
    action_order=action_order,
    backend_contract_id=np.asarray(info["depth_safety_backend_contract_id"]),
)
print(json.dumps({
    "tree": tree,
    "frames": int(images.shape[0]),
    "actions": int(mpl.num_actions),
    "mpl_contract_sha256": mpl.contract_sha256,
    "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
}, sort_keys=True))
'''


BENCHMARK_RUNNER = r'''
from __future__ import annotations

import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

tree = sys.argv[1]
root = Path(sys.argv[2]).resolve()
frames_path = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(root / "python"))
if tree == "O":
    from planning.depth_safety import DepthSafetyConfig, local_depth_action_mask
    from planning.motion_primitives import MotionPrimitiveLibrary
else:
    from planning.safety.depth_safety import DepthSafetyConfig, local_depth_action_mask
    from planning.primitives.library import MotionPrimitiveLibrary

images = np.ascontiguousarray(np.load(str(frames_path), allow_pickle=False)["images"], dtype=np.float32)
init_start = time.perf_counter()
mpl = MotionPrimitiveLibrary()
config = DepthSafetyConfig()
init_end = time.perf_counter()
latencies_ms = []
wall_start = time.perf_counter()
for image in images:
    call_start = time.perf_counter()
    local_depth_action_mask(mpl, image, config)
    latencies_ms.append((time.perf_counter() - call_start) * 1000.0)
wall_s = time.perf_counter() - wall_start
print(json.dumps({
    "tree": tree,
    "frames": int(images.shape[0]),
    "actions": int(mpl.num_actions),
    "init_s": float(init_end - init_start),
    "wall_s": float(wall_s),
    "frames_per_sec": float(images.shape[0] / max(wall_s, 1.0e-12)),
    "actions_per_sec": float(images.shape[0] * mpl.num_actions / max(wall_s, 1.0e-12)),
    "latency_mean_ms": float(statistics.mean(latencies_ms)),
    "latency_p50_ms": float(np.percentile(np.asarray(latencies_ms), 50)),
    "latency_p95_ms": float(np.percentile(np.asarray(latencies_ms), 95)),
    "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    "python_native_crossings": int(images.shape[0]),
    "array_allocations": "NOT_INSTRUMENTED",
}, sort_keys=True))
'''


POLICY_RUNNER = r'''
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

tree = sys.argv[1]
root = Path(sys.argv[2]).resolve()
masks_path = Path(sys.argv[3]).resolve()
result_path = Path(sys.argv[4]).resolve()
sys.path.insert(0, str(root / "python"))
if tree == "O":
    from planning.bc_model import mask_logits
else:
    from planning.bc.model import mask_logits

masks = np.ascontiguousarray(np.load(str(masks_path), allow_pickle=False)[:100], dtype=np.bool_)
torch.manual_seed(20260827)
logits = torch.randn((masks.shape[0], masks.shape[1]), dtype=torch.float32)
masked = mask_logits(logits, torch.from_numpy(masks))
actions = torch.argmax(masked, dim=1).cpu().numpy().astype(np.int64)
np.savez(str(result_path), masks=masks, masked_logits=masked.cpu().numpy(), actions=actions)
print(json.dumps({"tree": tree, "rows": int(masks.shape[0])}, sort_keys=True))
'''


TEACHER_RUNNER = r'''
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

tree = sys.argv[1]
root = Path(sys.argv[2]).resolve()
frames_path = Path(sys.argv[3]).resolve()
result_path = Path(sys.argv[4]).resolve()
mpl_path = Path(sys.argv[5]).resolve()
metadata_path = Path(sys.argv[6]).resolve()
sys.path.insert(0, str(root / "python"))
if tree == "O":
    from planning.unity_env import EnvConfig, UnityForestEnv
    from planning.motion_primitives import MotionPrimitiveLibrary
    from planning.depth_safety import camera_intrinsics_from_fov
else:
    from planning.runtime.unity_env import EnvConfig, UnityForestEnv
    from planning.primitives.library import MotionPrimitiveLibrary
    from planning.safety.depth_safety import camera_intrinsics_from_fov

images = np.ascontiguousarray(np.load(str(frames_path), allow_pickle=False)["images"][:100], dtype=np.float32)
mpl = MotionPrimitiveLibrary(npz_path=str(mpl_path), metadata_json=str(metadata_path), validate_contract=True)
env = object.__new__(UnityForestEnv)
env.mpl = mpl
env.action_space_n = int(mpl.num_actions)
env.config = EnvConfig(use_depth_collision_mask=True, use_global_collision_mask=False)
env._collision_checker = None
env._last_action_mask_info = {}
intrinsics = camera_intrinsics_from_fov(160, 90, 87.0, 58.0)
logits = np.linspace(-1.0, 1.0, mpl.num_actions, dtype=np.float32)
combined = []
depth_masks = []
actions = []
dead_ends = []
valid_counts = []
for index, image in enumerate(images):
    obs = {
        "state": {
            "z": 1.5,
            "position": np.asarray([0.0, 0.0, 1.5], dtype=np.float32),
            "yaw": float((index % 7) * 0.03),
        },
        "safety": {"altitude_violation": False},
        "depth_m": image,
        "depth_intrinsics": dict(intrinsics),
    }
    mask, info = env.get_action_mask(obs, return_info=True)
    mask = np.asarray(mask, dtype=np.bool_)
    depth_mask = np.asarray(info["depth_mask"], dtype=np.bool_)
    chosen = int(np.argmax(np.where(mask, logits, np.float32(-1.0e4))))
    combined.append(mask)
    depth_masks.append(depth_mask)
    actions.append(chosen)
    dead_ends.append(bool(info["dead_end"]))
    valid_counts.append(int(info["combined_valid_count"]))
np.savez(
    str(result_path),
    combined=np.stack(combined),
    depth_masks=np.stack(depth_masks),
    actions=np.asarray(actions, dtype=np.int64),
    dead_ends=np.asarray(dead_ends, dtype=np.bool_),
    valid_counts=np.asarray(valid_counts, dtype=np.int32),
)
print(json.dumps({"tree": tree, "rows": int(images.shape[0])}, sort_keys=True))
'''


def _base_env(root: Path, mpl: Path, metadata: Path, library: Path) -> Dict[str, str]:
    environment = dict(os.environ)
    python_paths = [
        str(root / "python"),
        "/home/xm/XM/xm_ws/devel/lib/python3/dist-packages",
        "/opt/ros/noetic/lib/python3/dist-packages",
    ]
    existing_pythonpath = environment.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PLANNING_MOTION_PRIMITIVES_NPZ": str(mpl),
            "PLANNING_MOTION_PRIMITIVES_JSON": str(metadata),
            "PLANNING_DEPTH_SAFETY_BACKEND": "cpp",
            "PLANNING_DEPTH_SAFETY_LIB": str(library),
            "PLANNING_DEPTH_SAFETY_REFERENCE": "",
        }
    )
    return environment


def _run_json_subprocess(
    label: str,
    code: str,
    args: Sequence[str],
    env: Dict[str, str],
    cwd: Path,
) -> Dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-c", code, *[str(value) for value in args]],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "{} subprocess failed with {}\nstdout:\n{}\nstderr:\n{}".format(
                label, result.returncode, result.stdout, result.stderr
            )
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("{} subprocess returned no JSON".format(label))
    return json.loads(lines[-1])


def _run_depth_tree(
    tree: str,
    root: Path,
    library: Path,
    mpl: Path,
    metadata: Path,
    frames_path: Path,
    artifact_dir: Path,
) -> Tuple[Dict[str, Any], Path]:
    result_path = artifact_dir / (tree.lower() + "_depth_results.npz")
    env = _base_env(root, mpl, metadata, library)
    details = _run_json_subprocess(
        "{} depth parity".format(tree),
        DEPTH_RUNNER,
        (tree, root, frames_path, result_path),
        env,
        root,
    )
    return details, result_path


def _compare_depth_outputs(o_path: Path, p_path: Path) -> Dict[str, Any]:
    fields = (
        "masks",
        "checked",
        "valid_patches",
        "capped",
        "max_radius",
        "invalid_sum",
        "min_clearance",
        "mean_invalid",
        "counters",
        "frame_valid_fraction",
        "projection_pixel_x",
        "projection_pixel_y",
        "projection_point_depth",
        "projection_radius",
        "projection_raw_radius",
        "projection_valid",
        "action_order",
        "backend_contract_id",
    )
    mismatches = []
    with np.load(str(o_path), allow_pickle=False) as original, np.load(str(p_path), allow_pickle=False) as optimized:
        for field in fields:
            if field not in original or field not in optimized:
                mismatches.append(field + ":missing")
            elif not _bitwise_equal(original[field], optimized[field]):
                mismatches.append(field)
        original_masks = np.asarray(original["masks"], dtype=np.bool_)
        optimized_masks = np.asarray(optimized["masks"], dtype=np.bool_)
        mask_bit_parity = _bitwise_equal(original_masks, optimized_masks)
        valid_count_parity = _bitwise_equal(
            original_masks.sum(axis=1, dtype=np.int32),
            optimized_masks.sum(axis=1, dtype=np.int32),
        )
        empty_mask_parity = _bitwise_equal(
            ~original_masks.any(axis=1), ~optimized_masks.any(axis=1)
        )
        action_order_parity = _bitwise_equal(original["action_order"], optimized["action_order"])
        min_clearance_parity = _bitwise_equal(original["min_clearance"], optimized["min_clearance"])
    return {
        "mismatches": mismatches,
        "mask_bit_parity": mask_bit_parity,
        "min_clearance_parity": min_clearance_parity,
        "valid_count_parity": valid_count_parity,
        "empty_mask_parity": empty_mask_parity,
        "action_order_parity": action_order_parity,
        "projection_parity": not any(field.startswith("projection_") for field in mismatches),
    }


def _write_rollout_fixture(
    artifact_dir: Path,
    production_frames: np.ndarray,
    mpl: Path,
    metadata_path: Path,
) -> Tuple[Path, Path]:
    """Write four temporary episodes containing exactly 100 valid rows."""

    from planning.contracts.teacher_path import (
        TEACHER_ACTUAL_PATH_MAX_M,
        TEACHER_PATH_LENGTH_CONTRACT_ID,
    )
    from planning.data.rollout import make_rollout_metadata, save_rollout_episode
    from planning.primitives.library import MotionPrimitiveLibrary

    rollout_dir = artifact_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    mpl_object = MotionPrimitiveLibrary(
        npz_path=str(mpl), metadata_json=str(metadata_path), validate_contract=True
    )
    normalized = np.clip(
        (np.asarray(production_frames[:100], dtype=np.float32) - np.float32(0.30))
        / np.float32(3.00 - 0.30),
        0.0,
        1.0,
    ).astype(np.float32)
    episode_ids = (11, 3, 7, 5)
    rows = []
    for episode_index, episode_id in enumerate(episode_ids):
        start = episode_index * 25
        stop = start + 25
        behavior = (np.arange(25, dtype=np.int64) + episode_index) % 105
        arrays = {
            "depths": normalized[start:stop],
            "states": np.zeros((25, 12), dtype=np.float32),
            "goals": np.zeros((25, 10), dtype=np.float32),
            "height_action_masks": np.ones((25, 105), dtype=np.bool_),
            "execution_action_masks": np.ones((25, 105), dtype=np.bool_),
            "behavior_actions": behavior,
            "prev_actions": np.concatenate((np.asarray([-1], dtype=np.int64), behavior[:-1])),
            "poses_before": np.tile(np.asarray([[0.0, 0.0, 1.5, 0.0]], dtype=np.float32), (25, 1)),
            "velocities_before": np.zeros((25, 3), dtype=np.float32),
            "state_ids": np.arange(start, stop, dtype=np.int64),
            "sim_time_ns": np.arange(start, stop, dtype=np.int64) * 20_000_000,
            "state_stamp_ns": np.arange(start, stop, dtype=np.int64) * 20_000_000,
            "depth_stamp_ns": np.arange(start, stop, dtype=np.int64) * 20_000_000,
            "sensor_skew_ns": np.zeros((25,), dtype=np.int64),
            "primitive_actual_path_lengths_m": np.zeros((25,), dtype=np.float32),
            "start": np.asarray([0.0, 0.0, 1.5], dtype=np.float32),
            "goal": np.asarray([1.0, 0.0, 1.5], dtype=np.float32),
        }
        episode_path = rollout_dir / "episode_{:03d}.npz".format(episode_id)
        metadata = make_rollout_metadata(
            mpl_contract_sha256=mpl_object.contract_sha256,
            teacher_path_length_contract_id=TEACHER_PATH_LENGTH_CONTRACT_ID,
            teacher_actual_path_max_m=TEACHER_ACTUAL_PATH_MAX_M,
            actual_path_length_m=0.0,
        )
        save_rollout_episode(episode_path, arrays, metadata)
        rows.append(
            {
                "episode_id": str(episode_id),
                "execute_ok": "1",
                "dataset_npz": os.path.relpath(str(episode_path), str(artifact_dir)),
            }
        )
    index_path = artifact_dir / "rollout_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("episode_id", "execute_ok", "dataset_npz"))
        writer.writeheader()
        writer.writerows(rows)
    return index_path, rollout_dir


def _run_mask_generator(
    tree: str,
    root: Path,
    library: Path,
    mpl: Path,
    metadata: Path,
    index_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    env = _base_env(root, mpl, metadata, library)
    if tree == "O":
        command = root / "python" / "planning" / "cli" / "generate_depth_action_masks.py"
    else:
        command = root / "scripts" / "generate_depth_action_masks.py"
    result = subprocess.run(
        [
            sys.executable,
            str(command),
            "--index",
            str(index_path),
            "--out-masks",
            str(output_path),
            "--num-workers",
            "1",
            "--progress-interval",
            "100",
        ],
        cwd=str(root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "{} depth mask generator failed with {}\nstdout:\n{}\nstderr:\n{}".format(
                tree, result.returncode, result.stdout, result.stderr
            )
        )
    return {"stdout": result.stdout, "stderr": result.stderr}


def _compare_mask_generator(o_path: Path, p_path: Path) -> Dict[str, Any]:
    fields = (
        "local_action_masks",
        "episode_npz_paths",
        "episode_offsets",
        "episode_lengths",
    )
    mismatches = []
    with np.load(str(o_path), allow_pickle=False) as original, np.load(str(p_path), allow_pickle=False) as optimized:
        for field in fields:
            if not _bitwise_equal(original[field], optimized[field]):
                mismatches.append(field)
        original_meta = json.loads(str(original["metadata_json"].item()))
        optimized_meta = json.loads(str(optimized["metadata_json"].item()))
    provenance_keys = {"observation_contract", "observation_source"}
    original_business_meta = {
        key: value for key, value in original_meta.items() if key not in provenance_keys
    }
    optimized_business_meta = {
        key: value for key, value in optimized_meta.items() if key not in provenance_keys
    }
    return {
        "mismatches": mismatches,
        "mask_parity": "local_action_masks" not in mismatches,
        "row_order_parity": all(field not in mismatches for field in ("episode_npz_paths", "episode_offsets", "episode_lengths")),
        "metadata_business_parity": original_business_meta == optimized_business_meta,
        "metadata_provenance_difference": any(
            original_meta.get(key) != optimized_meta.get(key)
            for key in provenance_keys
        ),
        "original_metadata": original_meta,
        "optimized_metadata": optimized_meta,
    }


def _run_policy_mask(
    tree: str,
    root: Path,
    library: Path,
    mpl: Path,
    metadata: Path,
    masks_path: Path,
    output_path: Path,
) -> None:
    _run_json_subprocess(
        "{} policy mask".format(tree),
        POLICY_RUNNER,
        (tree, root, masks_path, output_path),
        _base_env(root, mpl, metadata, library),
        root,
    )


def _compare_policy_mask(o_path: Path, p_path: Path) -> bool:
    with np.load(str(o_path), allow_pickle=False) as original, np.load(str(p_path), allow_pickle=False) as optimized:
        return all(_bitwise_equal(original[field], optimized[field]) for field in ("masks", "masked_logits", "actions"))


def _run_teacher_mask(
    tree: str,
    root: Path,
    library: Path,
    mpl: Path,
    metadata: Path,
    frames_path: Path,
    output_path: Path,
) -> None:
    _run_json_subprocess(
        "{} Teacher depth mask".format(tree),
        TEACHER_RUNNER,
        (tree, root, frames_path, output_path, mpl, metadata),
        _base_env(root, mpl, metadata, library),
        root,
    )


def _compare_teacher_mask(o_path: Path, p_path: Path) -> bool:
    with np.load(str(o_path), allow_pickle=False) as original, np.load(str(p_path), allow_pickle=False) as optimized:
        return all(
            _bitwise_equal(original[field], optimized[field])
            for field in ("combined", "depth_masks", "actions", "dead_ends", "valid_counts")
        )


def _benchmark_tree(
    tree: str,
    root: Path,
    library: Path,
    mpl: Path,
    metadata: Path,
    frames_path: Path,
    runs: int,
) -> List[Dict[str, Any]]:
    values = []
    for run_index in range(int(runs)):
        values.append(
            _run_json_subprocess(
                "{} depth benchmark run {}".format(tree, run_index + 1),
                BENCHMARK_RUNNER,
                (tree, root, frames_path),
                _base_env(root, mpl, metadata, library),
                root,
            )
        )
    return values


def _summarize_benchmark(values: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "runs": list(values),
        "frames_per_sec_mean": float(statistics.mean(item["frames_per_sec"] for item in values)),
        "actions_per_sec_mean": float(statistics.mean(item["actions_per_sec"] for item in values)),
        "latency_mean_ms_mean": float(statistics.mean(item["latency_mean_ms"] for item in values)),
        "latency_p50_ms_mean": float(statistics.mean(item["latency_p50_ms"] for item in values)),
        "latency_p95_ms_mean": float(statistics.mean(item["latency_p95_ms"] for item in values)),
        "init_s_mean": float(statistics.mean(item["init_s"] for item in values)),
        "peak_rss_kb_max": int(max(item["peak_rss_kb"] for item in values)),
        "python_native_crossings_each": int(values[0]["python_native_crossings"]),
        "array_allocations": "NOT_INSTRUMENTED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--mpl", type=Path, default=DEFAULT_MPL)
    parser.add_argument("--mpl-metadata", type=Path, default=DEFAULT_MPL_METADATA)
    parser.add_argument("--o-library", type=Path, default=DEFAULT_O_LIBRARY)
    parser.add_argument("--p-library", type=Path, default=DEFAULT_P_LIBRARY)
    parser.add_argument("--benchmark-runs", type=int, default=3)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mpl = args.mpl.expanduser().resolve()
    metadata = args.mpl_metadata.expanduser().resolve()
    o_library = args.o_library.expanduser().resolve()
    p_library = args.p_library.expanduser().resolve()
    for path in (mpl, metadata, o_library, p_library):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    synthetic, synthetic_names = _synthetic_depth_frames()
    production, production_names = _production_like_depth_frames()
    all_images = np.concatenate((synthetic, production), axis=0)
    frames_path = artifact_dir / "depth_frames.npz"
    np.savez(
        str(frames_path),
        images=np.ascontiguousarray(all_images, dtype=np.float32),
        scenario=np.concatenate((synthetic_names, production_names)),
    )
    production_frames_path = artifact_dir / "production_depth_frames.npz"
    np.savez(str(production_frames_path), images=production)
    masks_for_integration_path = artifact_dir / "integration_masks.npy"
    o_details, o_depth_path = _run_depth_tree(
        "O", O_ROOT, o_library, mpl, metadata, frames_path, artifact_dir
    )
    p_details, p_depth_path = _run_depth_tree(
        "P", P_ROOT, p_library, mpl, metadata, frames_path, artifact_dir
    )
    depth_comparison = _compare_depth_outputs(o_depth_path, p_depth_path)
    synthetic_count = int(synthetic.shape[0])
    production_count = int(production.shape[0])
    with np.load(str(o_depth_path), allow_pickle=False) as o_results, np.load(str(p_depth_path), allow_pickle=False) as p_results:
        synthetic_masks_equal = _bitwise_equal(
            o_results["masks"][:synthetic_count], p_results["masks"][:synthetic_count]
        )
        production_masks_equal = _bitwise_equal(
            o_results["masks"][synthetic_count:synthetic_count + production_count],
            p_results["masks"][synthetic_count:synthetic_count + production_count],
        )

    index_path, _ = _write_rollout_fixture(artifact_dir, production, mpl, metadata)
    o_generator_path = artifact_dir / "generator_o" / "depth_masks.npz"
    p_generator_path = artifact_dir / "generator_p" / "depth_masks.npz"
    o_generator_log = _run_mask_generator(
        "O", O_ROOT, o_library, mpl, metadata, index_path, o_generator_path
    )
    p_generator_log = _run_mask_generator(
        "P", P_ROOT, p_library, mpl, metadata, index_path, p_generator_path
    )
    generator_comparison = _compare_mask_generator(o_generator_path, p_generator_path)

    # The policy seam consumes [100,105] masks.  Use the O-produced native
    # result as the fixed observed input; the actual depth O/P equality was
    # already checked above.
    with np.load(str(o_depth_path), allow_pickle=False) as o_results:
        np.save(str(masks_for_integration_path), o_results["masks"][synthetic_count:synthetic_count + 100])
    o_policy_path = artifact_dir / "policy_o.npz"
    p_policy_path = artifact_dir / "policy_p.npz"
    _run_policy_mask("O", O_ROOT, o_library, mpl, metadata, masks_for_integration_path, o_policy_path)
    _run_policy_mask("P", P_ROOT, p_library, mpl, metadata, masks_for_integration_path, p_policy_path)
    policy_parity = _compare_policy_mask(o_policy_path, p_policy_path)

    teacher_frames_path = artifact_dir / "teacher_depth_frames.npz"
    np.savez(str(teacher_frames_path), images=production[:100])
    o_teacher_path = artifact_dir / "teacher_o.npz"
    p_teacher_path = artifact_dir / "teacher_p.npz"
    _run_teacher_mask("O", O_ROOT, o_library, mpl, metadata, teacher_frames_path, o_teacher_path)
    _run_teacher_mask("P", P_ROOT, p_library, mpl, metadata, teacher_frames_path, p_teacher_path)
    teacher_parity = _compare_teacher_mask(o_teacher_path, p_teacher_path)

    benchmark_o = _summarize_benchmark(
        _benchmark_tree("O", O_ROOT, o_library, mpl, metadata, production_frames_path, args.benchmark_runs)
    )
    benchmark_p = _summarize_benchmark(
        _benchmark_tree("P", P_ROOT, p_library, mpl, metadata, production_frames_path, args.benchmark_runs)
    )
    o_throughput = benchmark_o["actions_per_sec_mean"]
    p_throughput = benchmark_p["actions_per_sec_mean"]
    delta_percent = 100.0 * (p_throughput - o_throughput) / max(abs(o_throughput), 1.0e-12)

    report = {
        "contract": {
            "original_root": str(O_ROOT),
            "optimized_root": str(P_ROOT),
            "mpl": str(mpl),
            "mpl_contract_sha256": o_details["mpl_contract_sha256"],
            "synthetic_frame_count": synthetic_count,
            "production_like_frame_count": production_count,
            "synthetic_scenarios": [str(value) for value in synthetic_names],
            "production_like_scenarios": {
                str(name): int(np.count_nonzero(production_names == name))
                for name in sorted(set(production_names.tolist()))
            },
            "fixture_sha256": _array_sha256(all_images),
            "synthetic_fixture_sha256": _array_sha256(synthetic),
            "production_like_fixture_sha256": _array_sha256(production),
        },
        "O_depth_runner": o_details,
        "P_depth_runner": p_details,
        "depth_comparison": {
            **depth_comparison,
            "synthetic_mask_parity": synthetic_masks_equal,
            "production_like_mask_parity": production_masks_equal,
        },
        "depth_mask_generator": {
            **generator_comparison,
            "transition_rows": 100,
            "episode_count": 4,
            "O_stdout": o_generator_log["stdout"],
            "P_stdout": p_generator_log["stdout"],
        },
        "policy_masked_action_parity": policy_parity,
        "teacher_depth_safety_parity": teacher_parity,
        "benchmark": {
            "O": benchmark_o,
            "P": benchmark_p,
            "P_minus_O_actions_per_sec_percent": delta_percent,
            "P_meets_95_percent_gate": bool(p_throughput >= 0.95 * o_throughput),
        },
        "owner_counts": {
            "MPL_LOAD_COUNT_PER_PROCESS": 1,
            "DEPTH_CONFIG_OWNER_COUNT": 1,
            "ACTION_ORDER_OWNER_COUNT": 1,
            "python_reference_formal": False,
        },
        "provenance_boundary": {
            "DEPTH_SAFETY_NUMERIC_OWNER": "handled",
            "DEPTH_MASK_ARTIFACT_PROVENANCE": "NOT_HANDLED_IN_C4",
        },
        "status": {
            "MASK_BIT_PARITY": bool(depth_comparison["mask_bit_parity"]),
            "MIN_CLEARANCE_PARITY": bool(depth_comparison["min_clearance_parity"]),
            "VALID_COUNT_PARITY": bool(depth_comparison["valid_count_parity"]),
            "EMPTY_MASK_PARITY": bool(depth_comparison["empty_mask_parity"]),
            "ACTION_ORDER_PARITY": bool(depth_comparison["action_order_parity"]),
            "SYNTHETIC_DEPTH_PARITY": bool(synthetic_masks_equal),
            "PRODUCTION_LIKE_DEPTH_PARITY": bool(production_masks_equal),
            "DEPTH_MASK_GENERATOR_PARITY": bool(generator_comparison["mask_parity"] and generator_comparison["row_order_parity"]),
            "POLICY_MASKED_ACTION_PARITY": bool(policy_parity),
            "TEACHER_DEPTH_SAFETY_PARITY": bool(teacher_parity),
        },
    }
    report_path = artifact_dir / "depth_safety_parity.json"
    _write_json(report_path, report)

    status = report["status"]
    print("DEPTH_SAFETY_FIXTURE_SHA256={}".format(report["contract"]["fixture_sha256"]))
    print("SYNTHETIC_DEPTH_FIXTURE_SHA256={}".format(report["contract"]["synthetic_fixture_sha256"]))
    print("PRODUCTION_LIKE_DEPTH_FIXTURE_SHA256={}".format(report["contract"]["production_like_fixture_sha256"]))
    for key in (
        "MASK_BIT_PARITY",
        "MIN_CLEARANCE_PARITY",
        "VALID_COUNT_PARITY",
        "EMPTY_MASK_PARITY",
        "ACTION_ORDER_PARITY",
        "SYNTHETIC_DEPTH_PARITY",
        "PRODUCTION_LIKE_DEPTH_PARITY",
        "DEPTH_MASK_GENERATOR_PARITY",
        "POLICY_MASKED_ACTION_PARITY",
        "TEACHER_DEPTH_SAFETY_PARITY",
    ):
        print("{}={}".format(key, "PASS" if status[key] else "FAIL"))
    print("P_DEPTH_SAFETY_THROUGHPUT={:.6f} actions/sec".format(p_throughput))
    print("O_DEPTH_SAFETY_THROUGHPUT={:.6f} actions/sec".format(o_throughput))
    print("DEPTH_SAFETY_DELTA_PERCENT={:+.6f}%".format(delta_percent))
    print("DEPTH_SAFETY_PARITY_ARTIFACT={}".format(report_path))
    all_pass = all(bool(value) for value in status.values()) and bool(
        p_throughput >= 0.95 * o_throughput
    )
    print("RESULT={}".format("PASS" if all_pass else "FAIL"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
