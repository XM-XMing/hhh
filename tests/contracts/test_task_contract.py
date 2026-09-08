#!/usr/bin/env python3
"""Permanent regression checks for the hashed 3D mission contract."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


from planning.mission.spec import (
    FLIGHT_Z_MAX_M,
    FLIGHT_Z_MIN_M,
    GOAL_TOLERANCE_Z_M,
    TASK_CONTRACT_ID,
    goal_reached,
    task_contract_fields,
    task_contract_sha256,
    validate_mission_rows,
)
from planning.data.rollout import (
    load_rollout_episode_fields,
    make_rollout_metadata,
    validate_rollout_episode,
    validate_rollout_metadata,
)
from planning.teacher.policy import ReachabilityTeacher, TeacherConfig, config_from_args
from planning.contracts.teacher_path import (
    TEACHER_ACTUAL_PATH_MAX_M,
    TEACHER_PATH_LENGTH_CONTRACT_ID,
)


def main() -> int:
    valid_row = {
        "episode_id": 1,
        **task_contract_fields(),
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(),
        "start_x": 0.0,
        "start_y": 0.0,
        "start_z": 2.0,
        "goal_x": 40.0,
        "goal_y": 0.0,
        "goal_z": 2.8,
    }
    validate_mission_rows([valid_row])
    old_contract_rejected = False
    try:
        validate_mission_rows([{**valid_row, "task_contract_id": "legacy"}])
    except ValueError:
        old_contract_rejected = True
    rollout_metadata = make_rollout_metadata(
        mpl_contract_sha256="0" * 64,
    )
    old_rollout_rejected = False
    try:
        validate_rollout_episode(
            {"metadata": {**rollout_metadata, "task_contract_id": "legacy"}}
        )
    except ValueError as exc:
        old_rollout_rejected = "task contract" in str(exc)
    contract_override_rejected = False
    try:
        config_from_args(
            SimpleNamespace(
                goal_radius=1.2,
                goal_tolerance_z=GOAL_TOLERANCE_Z_M + 0.1,
            )
        )
    except ValueError:
        contract_override_rejected = True
    sentinel_mpl = SimpleNamespace(
        pos_ref=np.zeros((1, 2, 3), dtype=np.float32),
        cmd_seq=np.zeros((1, 2, 4), dtype=np.float32),
        num_actions=1,
    )
    teacher = ReachabilityTeacher(sentinel_mpl, None, TeacherConfig())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "minimal-label-rollout.npz"
        selective_metadata = make_rollout_metadata(
            mpl_contract_sha256="0" * 64,
            teacher_path_length_contract_id=TEACHER_PATH_LENGTH_CONTRACT_ID,
            teacher_actual_path_max_m=TEACHER_ACTUAL_PATH_MAX_M,
        )
        np.savez(
            path,
            behavior_actions=np.asarray([1], dtype=np.int64),
            goal=np.asarray([40.0, 0.0, 2.0], dtype=np.float32),
            metadata_json=np.asarray(json.dumps(selective_metadata)),
        )
        selective = load_rollout_episode_fields(
            path, ("behavior_actions", "goal")
        )
        validate_rollout_metadata(selective["metadata"], path)
        selective_label_loader = (
            set(selective) == {"behavior_actions", "goal", "metadata"}
            and "depths" not in selective
        )
    checks = {
        "contract_hashed": bool(TASK_CONTRACT_ID and len(task_contract_sha256()) == 64),
        "exact_3d_goal_reached": goal_reached([10.0, 5.0, 2.8], [10.0, 5.0, 2.8]),
        "xy_only_is_not_success": not goal_reached([10.0, 5.0, 1.1], [10.0, 5.0, 2.8]),
        "z_tolerance_inside": goal_reached(
            [10.0, 5.0, 2.8 - GOAL_TOLERANCE_Z_M + 1.0e-6], [10.0, 5.0, 2.8]
        ),
        "z_tolerance_outside": not goal_reached(
            [10.0, 5.0, 2.8 - GOAL_TOLERANCE_Z_M - 1.0e-6], [10.0, 5.0, 2.8]
        ),
        "flight_band_contract": FLIGHT_Z_MIN_M == 1.0 and FLIGHT_Z_MAX_M == 3.0,
        "old_contract_rejected": old_contract_rejected,
        "old_rollout_rejected": old_rollout_rejected,
        "contract_override_rejected": contract_override_rejected,
        "teacher_motion_primitives_identity": (
            teacher.motion_primitives is sentinel_mpl
            and not hasattr(teacher, "mpl")
        ),
        "selective_label_loader_excludes_depth": selective_label_loader,
        "teacher_default_next_state_rule": teacher.config.min_next_valid_count == 1,
    }
    print("TASK_CONTRACT_TEST")
    for name, passed in checks.items():
        print("  {}: {}".format(name, passed))
    passed = all(checks.values())
    print("RESULT={}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
