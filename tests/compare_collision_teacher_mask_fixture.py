#!/usr/bin/env python3
"""Compare O/P collision-backed candidate acceptance and Teacher masks.

The fixture is deliberately deterministic and runs each source tree in an
isolated subprocess.  It does not generate missions, collect data, or invoke
Unity; it only evaluates the original candidate predicate and the Teacher's
105-action current valid mask against the same temporary MPL and voxel cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


O_ROOT = Path("/home/xm/XM/xm_ws/src/planning").resolve()
P_ROOT = Path("/home/xm/XM/src").resolve()


RUNNER = r'''
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


tree = sys.argv[1]
root = Path(sys.argv[2]).resolve()
cache_path = Path(sys.argv[3]).resolve()
mpl_path = Path(sys.argv[4]).resolve()
metadata_path = Path(sys.argv[5]).resolve()
sys.path.insert(0, str(root / "python"))

if tree == "P":
    from planning.mission.sampling import _candidate, _mission_is_safe
    from planning.primitives.library import MotionPrimitiveLibrary
    from planning.safety.collision_checker import VoxelCollisionChecker
    from planning.teacher.policy import ReachabilityTeacher, TeacherConfig
else:
    from planning.mission_sampling import _candidate, _mission_is_safe
    from planning.motion_primitives import MotionPrimitiveLibrary
    from planning.collision_checker import VoxelCollisionChecker
    from planning.teacher_policy import ReachabilityTeacher, TeacherConfig


def array_sha256(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


mpl = MotionPrimitiveLibrary(
    npz_path=str(mpl_path), metadata_json=str(metadata_path), validate_contract=True
)
checker = VoxelCollisionChecker.load_cache(
    cache_path, voxel_size=0.10, inflate_radius=0.35
)
config = TeacherConfig(
    candidate_top_k=12,
    current_check_step=1,
    lookahead_check_step=2,
    beam_depth=1,
)
teacher = ReachabilityTeacher(mpl, checker, config)

rng = np.random.default_rng(20260827)
missions = [
    _candidate(
        rng,
        episode_id=index,
        start_x_range=(-60.0, 60.0),
        start_y_range=(-60.0, 60.0),
        goal_distance=20.0,
        goal_y_offset_range=(-1.5, 1.5),
        altitude_levels=(1.1, 1.5, 2.0, 2.5, 2.9),
    )
    for index in range(100)
]

candidate_rows = []
teacher_masks = []
teacher_raw_masks = []
teacher_height_masks = []
teacher_actions = []
for mission in missions:
    candidate = dict(mission)
    accepted = _mission_is_safe(
        candidate,
        mpl,
        checker,
        min_start_valid_actions=40,
        min_goal_valid_actions=10,
        map_margin_m=3.0,
        check_step=2,
    )
    candidate_rows.append(
        [
            bool(accepted),
            int(candidate.get("start_valid_action_count", -1)),
            int(candidate.get("goal_valid_action_count", -1)),
        ]
    )

    start = np.asarray(mission["start"][:3], dtype=np.float32)
    goal = np.asarray(mission["goal"], dtype=np.float32)
    teacher.set_mission_route(np.stack([start, goal]), goal)
    action, debug = teacher.score_state(
        start,
        0.0,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        goal,
        prev_action=-1,
    )
    teacher_masks.append(np.asarray(debug["global_action_mask"], dtype=np.bool_).tolist())
    teacher_raw_masks.append(
        np.asarray(debug["raw_global_action_mask"], dtype=np.bool_).tolist()
    )
    teacher_height_masks.append(
        np.asarray(debug["height_action_mask"], dtype=np.bool_).tolist()
    )
    teacher_actions.append(int(action))

candidate_array = np.asarray(candidate_rows, dtype=np.int64)
teacher_mask_array = np.asarray(teacher_masks, dtype=np.bool_)
teacher_raw_array = np.asarray(teacher_raw_masks, dtype=np.bool_)
teacher_height_array = np.asarray(teacher_height_masks, dtype=np.bool_)
print(json.dumps({
    "tree": tree,
    "missions": len(missions),
    "mpl_contract_sha256": mpl.contract_sha256,
    "collision_backend": checker.collision_backend,
    "candidate_accept_rows": candidate_rows,
    "candidate_accept_vector_sha256": array_sha256(candidate_array),
    "teacher_valid_action_mask_rows": teacher_masks,
    "teacher_valid_action_mask_sha256": array_sha256(teacher_mask_array),
    "teacher_raw_action_mask_sha256": array_sha256(teacher_raw_array),
    "teacher_height_action_mask_sha256": array_sha256(teacher_height_array),
    "teacher_actions": teacher_actions,
}, sort_keys=True))
'''


def _run(tree: str, root: Path, cache: Path, mpl: Path, metadata: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PLANNING_COLLISION_BACKEND"] = "cpp_cpu" if tree == "P" else "cpu"
    env["PLANNING_COLLISION_THREADS"] = "1"
    default_library = (
        "/tmp/xm-cxx-c0-p-devel/planning/lib/libplanning_collision_checker.so"
        if tree == "P"
        else "/tmp/xm-cxx-c0-o-devel/planning/lib/libplanning_collision_checker.so"
    )
    library_path = os.environ.get(
        "XM_P_CPU_LIB" if tree == "P" else "XM_O_CPU_LIB", default_library
    )
    env["PLANNING_COLLISION_LIBRARY"] = str(
        Path(library_path)
    )
    default_library_dir = (
        "/tmp/xm-cxx-c0-p-devel/planning/lib"
        if tree == "P"
        else "/tmp/xm-cxx-c0-o-devel/planning/lib"
    )
    library_dir = os.environ.get(
        "XM_P_CPU_LIB_DIR" if tree == "P" else "XM_O_CPU_LIB_DIR",
        default_library_dir,
    )
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            str(Path(library_dir)),
            "/usr/local/cuda-11.8/lib64",
            env.get("LD_LIBRARY_PATH", ""),
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", RUNNER, tree, str(root), str(cache), str(mpl), str(metadata)],
        cwd=str(root),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "{} collision/Teacher fixture failed ({}):\n{}\n{}".format(
                tree, result.returncode, result.stdout, result.stderr
            )
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--mpl", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    p_result = _run("P", P_ROOT, args.cache.resolve(), args.mpl.resolve(), args.metadata.resolve())
    o_result = _run("O", O_ROOT, args.cache.resolve(), args.mpl.resolve(), args.metadata.resolve())
    candidate_equal = p_result["candidate_accept_rows"] == o_result["candidate_accept_rows"]
    teacher_equal = (
        p_result["teacher_valid_action_mask_rows"]
        == o_result["teacher_valid_action_mask_rows"]
    )
    result = {
        "candidate_missions": p_result["missions"],
        "candidate_accept_vector_parity": candidate_equal,
        "teacher_mask_missions": p_result["missions"],
        "teacher_valid_action_mask_parity": teacher_equal,
        "O": o_result,
        "P": p_result,
    }
    if args.output is not None:
        args.output.resolve().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CANDIDATE_ACCEPT_VECTOR_MISSIONS={}".format(p_result["missions"]))
    print("CANDIDATE_ACCEPT_VECTOR_PARITY={}".format("PASS" if candidate_equal else "FAIL"))
    print("TEACHER_VALID_ACTION_MASK_MISSIONS={}".format(p_result["missions"]))
    print("TEACHER_VALID_ACTION_MASK_PARITY={}".format("PASS" if teacher_equal else "FAIL"))
    print("FIXTURE_RESULT_SHA256={}".format(
        hashlib.sha256(json.dumps(result, sort_keys=True).encode("utf-8")).hexdigest()
    ))
    print(json.dumps(result, sort_keys=True))
    return 0 if candidate_equal and teacher_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
