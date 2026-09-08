#!/usr/bin/env python3
"""Compare mission route validation and Teacher consumption of route results.

The route stage is exercised against the fixed candidate CSV.  The Teacher
stage consumes those exact route arrays through ``set_mission_route`` so this
comparator does not regenerate missions, collect rollouts, or alter Teacher
scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List


O_ROOT = Path("/home/xm/XM/xm_ws/src/planning").resolve()
P_ROOT = Path("/home/xm/XM/src").resolve()
DEFAULT_CACHE = Path("/tmp/xmflight_collision_c2/forest_voxels_10cm.npz")
DEFAULT_CANDIDATES = Path(
    "/home/xm/XM/xm_ws/src/planning/data_back_20260826/teach/flight_20260717/mission_candidates.csv"
)
DEFAULT_ARTIFACT_DIR = Path("/tmp/xmflight_global_route_c3")
DEFAULT_P_ROUTE = Path(
    "/tmp/xm-cxx-c2-2-p-devel/planning/lib/libplanning_global_route.so"
)
DEFAULT_O_COLLISION = Path(
    "/tmp/xm-cxx-c0-o-devel/planning/lib/libplanning_collision_checker.so"
)
DEFAULT_P_COLLISION = Path(
    "/tmp/xm-cxx-c2-2-p-devel/planning/lib/libplanning_collision_checker.so"
)


RUNNER = r'''
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


tree = sys.argv[1]
root = Path(sys.argv[2]).resolve()
cache_path = Path(sys.argv[3]).resolve()
candidates_path = Path(sys.argv[4]).resolve()
mpl_path = Path(sys.argv[5]).resolve()
metadata_path = Path(sys.argv[6]).resolve()
sys.path.insert(0, str(root / "python"))

from planning.mission.global_route import (
    GlobalRouteConfig,
    GlobalRoutePlanner2D,
    GlobalRouteUnavailableError,
)
from planning.safety.collision_checker import VoxelCollisionChecker
from planning.primitives.library import MotionPrimitiveLibrary
from planning.teacher.policy import ReachabilityTeacher, TeacherConfig


def bytes_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def float_bits(value):
    return np.asarray([float(value)], dtype=np.float64).tobytes().hex()


with candidates_path.open(newline="", encoding="utf-8") as handle:
    rows = list(__import__("csv").DictReader(handle))[:1000]
if len(rows) != 1000:
    raise RuntimeError("candidate fixture must provide exactly 1000 rows")

# Load the collision cache once for this process.  The route wrapper receives
# the same immutable cache fields; Python shared binding remains a later C5
# cutover and is intentionally not introduced here.
collision_checker = VoxelCollisionChecker.load_cache(
    cache_path, voxel_size=0.10, inflate_radius=0.35
)
route_checker = SimpleNamespace(
    occupied_keys=np.ascontiguousarray(collision_checker.occupied_keys, dtype=np.int64),
    origin_ijk=np.ascontiguousarray(collision_checker.origin_ijk, dtype=np.int64),
    grid_shape=np.ascontiguousarray(collision_checker.grid_shape, dtype=np.int64),
    voxel_size=float(collision_checker.voxel_size),
    collision_radius=float(collision_checker.collision_radius),
)
route_planner = GlobalRoutePlanner2D(
    route_checker,
    GlobalRouteConfig(
        resolution_m=0.25,
        flight_z_min_m=1.0,
        flight_z_max_m=3.0,
        lookahead_m=3.0,
        tracking_margin_m=0.0,
        nearest_free_radius_cells=6,
    ),
)

route_rows = []
routes = {}
for index, row in enumerate(rows):
    start = [float(row["start_x"]), float(row["start_y"]), float(row["start_z"])]
    goal = [float(row["goal_x"]), float(row["goal_y"]), float(row["goal_z"])]
    mission_id = str(row.get("mission_id", "")) or "candidate-{:04d}".format(index)
    try:
        route = np.ascontiguousarray(route_planner.plan(start, goal), dtype=np.float32)
    except GlobalRouteUnavailableError:
        route_rows.append({
            "index": index,
            "mission_id": mission_id,
            "status": "no_route",
            "route_points": 0,
            "route_length_bits": float_bits(0.0),
            "route_stretch_bits": float_bits(0.0),
            "accepted": False,
        })
        continue
    route_length = float(
        np.sum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1))
    )
    direct_distance = float(
        np.linalg.norm(
            np.asarray(goal[:2], dtype=np.float64)
            - np.asarray(start[:2], dtype=np.float64)
        )
    )
    route_stretch = route_length / max(1.0e-6, direct_distance)
    accepted = route_stretch <= 2.0
    routes[index] = route
    route_rows.append({
        "index": index,
        "mission_id": mission_id,
        "status": "ok",
        "route_points": int(route.shape[0]),
        "route_length_bits": float_bits(route_length),
        "route_stretch_bits": float_bits(route_stretch),
        "accepted": bool(accepted),
    })

accepted_ids = [row["mission_id"] for row in route_rows if row["accepted"]]

# Use the first 100 route-bearing missions as fixed Teacher states.  The state
# is either the route start or a deterministic interior route point; no
# Teacher configuration or scoring code is changed by this test.
teacher = ReachabilityTeacher(
    MotionPrimitiveLibrary(
        npz_path=str(mpl_path),
        metadata_json=str(metadata_path),
        validate_contract=True,
    ),
    collision_checker,
    TeacherConfig(
        candidate_top_k=12,
        current_check_step=1,
        lookahead_check_step=2,
        beam_depth=1,
    ),
)
teacher_rows = []
for index in sorted(routes)[:100]:
    row = rows[index]
    start = np.asarray(
        [float(row["start_x"]), float(row["start_y"]), float(row["start_z"])],
        dtype=np.float32,
    )
    goal = np.asarray(
        [float(row["goal_x"]), float(row["goal_y"]), float(row["goal_z"])],
        dtype=np.float32,
    )
    route = routes[index]
    teacher.set_mission_route(route, goal)
    route_input = teacher.mission_route
    state_index = 0
    if route.shape[0] > 3 and index % 2:
        state_index = min(route.shape[0] - 2, max(1, route.shape[0] // 2))
    position = route[state_index].copy()
    yaw = float((index % 9) - 4) * 0.11
    velocity = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    action, debug = teacher.score_state(
        position=position,
        yaw=yaw,
        velocity=velocity,
        goal=goal,
        prev_action=-1,
    )
    guidance = np.asarray(debug.get("guidance_goal", goal), dtype=np.float32)
    mask = np.asarray(debug["global_action_mask"], dtype=np.bool_)
    teacher_rows.append({
        "index": index,
        "route_input_shape": list(route_input.shape),
        "route_input_hex": route_input.tobytes().hex(),
        "route_input_sha256": bytes_sha256(route_input),
        "state_index": state_index,
        "guidance_goal_hex": guidance.tobytes().hex(),
        "guidance_goal_sha256": bytes_sha256(guidance),
        "chosen_progress_bits": float_bits(debug.get("chosen_progress", 0.0)),
        "action": int(action),
        "dead_end": bool(debug["dead_end_now"]),
        "valid_action_mask": mask.tolist(),
        "raw_action_mask": np.asarray(debug["raw_global_action_mask"], dtype=np.bool_).tolist(),
    })

print(json.dumps({
    "tree": tree,
    "candidate_count": len(route_rows),
    "route_rows": route_rows,
    "accepted_ids": accepted_ids,
    "teacher_rows": teacher_rows,
    "route_backend": getattr(route_planner, "backend_name", "python-reference"),
    "collision_backend": collision_checker.collision_backend,
}, sort_keys=True))
'''


def _write_mpl(artifact_dir: Path) -> tuple[Path, Path]:
    """Generate only a temporary MPL fixture; never populate project data."""
    import copy

    from planning.primitives.generator import generate_library, save_library
    from planning.primitives.library import load_motion_primitive_config

    config = copy.deepcopy(load_motion_primitive_config())
    config["paths"]["motion_primitives_npz"] = str(artifact_dir / "c3_mpl.npz")
    config["paths"]["metadata_json"] = str(artifact_dir / "c3_mpl.json")
    return save_library(generate_library(config), config)


def _run(
    tree: str,
    root: Path,
    cache: Path,
    candidates: Path,
    mpl: Path,
    metadata: Path,
    route_library: Path,
    collision_library: Path,
) -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PLANNING_COLLISION_BACKEND"] = "cpp_cpu" if tree == "P" else "cpu"
    env["PLANNING_COLLISION_THREADS"] = "1"
    env["PLANNING_COLLISION_LIBRARY"] = str(collision_library)
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(collision_library.parent), str(route_library.parent), env.get("LD_LIBRARY_PATH", "")]
    )
    if tree == "P":
        env["PLANNING_GLOBAL_ROUTE_BACKEND"] = "cpp"
        env["PLANNING_GLOBAL_ROUTE_LIBRARY"] = str(route_library)
    else:
        env.pop("PLANNING_GLOBAL_ROUTE_BACKEND", None)
        env.pop("PLANNING_GLOBAL_ROUTE_LIBRARY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            RUNNER,
            tree,
            str(root),
            str(cache),
            str(candidates),
            str(mpl),
            str(metadata),
        ],
        cwd=str(root),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "{} mission/Teacher comparator failed ({}):\n{}\n{}".format(
                tree, result.returncode, result.stdout, result.stderr
            )
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _compare(o: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    o_routes = o["route_rows"]
    p_routes = p["route_rows"]
    route_fields = (
        "status",
        "route_points",
        "route_length_bits",
        "route_stretch_bits",
        "accepted",
    )
    route_mismatches = [
        {"index": index, "fields": [field for field in route_fields if o_row[field] != p_row[field]]}
        for index, (o_row, p_row) in enumerate(zip(o_routes, p_routes))
        if any(o_row[field] != p_row[field] for field in route_fields)
    ]
    teacher_fields = (
        "route_input_shape",
        "route_input_hex",
        "guidance_goal_hex",
        "chosen_progress_bits",
        "action",
        "dead_end",
        "valid_action_mask",
        "raw_action_mask",
    )
    o_teacher = o["teacher_rows"]
    p_teacher = p["teacher_rows"]
    teacher_mismatches = [
        {"index": o_row["index"], "fields": [field for field in teacher_fields if o_row[field] != p_row[field]]}
        for o_row, p_row in zip(o_teacher, p_teacher)
        if any(o_row[field] != p_row[field] for field in teacher_fields)
    ]
    return {
        "mission_count": len(o_routes),
        "route_mismatches": route_mismatches,
        "route_found_parity": not any(
            o_row["status"] != p_row["status"] for o_row, p_row in zip(o_routes, p_routes)
        ),
        "route_length_parity": not any(
            o_row["route_length_bits"] != p_row["route_length_bits"]
            for o_row, p_row in zip(o_routes, p_routes)
        ),
        "route_stretch_parity": not any(
            o_row["route_stretch_bits"] != p_row["route_stretch_bits"]
            for o_row, p_row in zip(o_routes, p_routes)
        ),
        "route_validation_parity": not any(
            o_row["status"] != p_row["status"] for o_row, p_row in zip(o_routes, p_routes)
        ),
        "mission_accept_vector_parity": [row["accepted"] for row in o_routes]
        == [row["accepted"] for row in p_routes],
        "accepted_id_order_parity": o["accepted_ids"] == p["accepted_ids"],
        "teacher_count": len(o_teacher),
        "teacher_mismatches": teacher_mismatches,
        "teacher_route_input_parity": not any(
            o_row["route_input_hex"] != p_row["route_input_hex"]
            for o_row, p_row in zip(o_teacher, p_teacher)
        ),
        "teacher_action_parity": not teacher_mismatches,
        "teacher_action_vector_parity": [row["action"] for row in o_teacher]
        == [row["action"] for row in p_teacher],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--p-route", type=Path, default=DEFAULT_P_ROUTE)
    parser.add_argument("--o-collision", type=Path, default=DEFAULT_O_COLLISION)
    parser.add_argument("--p-collision", type=Path, default=DEFAULT_P_COLLISION)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mpl, metadata = _write_mpl(artifact_dir)
    o = _run(
        "O", O_ROOT, args.cache.resolve(), args.candidates.resolve(),
        mpl, metadata, args.p_route.resolve(), args.o_collision.resolve()
    )
    p = _run(
        "P", P_ROOT, args.cache.resolve(), args.candidates.resolve(),
        mpl, metadata, args.p_route.resolve(), args.p_collision.resolve()
    )
    comparison = _compare(o, p)
    output = {"comparison": comparison, "O": o, "P": p}
    output_path = artifact_dir / "c3_mission_teacher_parity.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("MISSION_COUNT={}".format(comparison["mission_count"]))
    print("MISSION_ROUTE_ACCEPT_VECTOR_PARITY={}".format("PASS" if comparison["mission_accept_vector_parity"] else "FAIL"))
    print("MISSION_ACCEPTED_ID_ORDER_PARITY={}".format("PASS" if comparison["accepted_id_order_parity"] else "FAIL"))
    print("TEACHER_COUNT={}".format(comparison["teacher_count"]))
    print("TEACHER_ROUTE_INPUT_PARITY={}".format("PASS" if comparison["teacher_route_input_parity"] else "FAIL"))
    print("TEACHER_ACTION_PARITY={}".format("PASS" if comparison["teacher_action_parity"] else "FAIL"))
    print("ARTIFACT={}".format(output_path))
    print("RESULT={}".format("PASS" if all([
        comparison["mission_accept_vector_parity"],
        comparison["accepted_id_order_parity"],
        comparison["teacher_route_input_parity"],
        comparison["teacher_action_parity"],
    ]) else "FAIL"))
    return 0 if output["comparison"]["mission_accept_vector_parity"] and output["comparison"]["accepted_id_order_parity"] and output["comparison"]["teacher_route_input_parity"] and output["comparison"]["teacher_action_parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
