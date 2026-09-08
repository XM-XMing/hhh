#!/usr/bin/env python3
"""Regression test that the Unity environment enforces the 3D task contract."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np


from planning.runtime.unity_env import EnvConfig, UnityForestEnv
from planning.primitives.library import MotionPrimitiveLibrary


def evaluate(position, goal, distance_xy, dz, unity_altitude_violation=False):
    env = UnityForestEnv.__new__(UnityForestEnv)
    env.config = EnvConfig()
    env.prev_distance_xy = float(distance_xy)
    env.prev_abs_goal_dz = abs(float(dz))
    env.prev_action_id = -1
    env.episode_step = 0
    observation = {
        "goal": {"position": goal, "distance_xy": distance_xy, "dz": dz},
        "state": {"position": position, "z": position[2]},
        "safety": {
            "collided": False,
            "altitude_violation": bool(unity_altitude_violation),
            "min_clearance": 5.0,
        },
        "action_mask": [True],
    }
    _, done, info = env._compute_reward_done(observation, 52)
    return bool(done), info


def main() -> int:
    wrong_z_done, wrong_z = evaluate(
        [40.0, 0.0, 1.1], [40.0, 0.0, 2.8], 0.0, 1.7
    )
    low_done, low = evaluate(
        [30.0, 0.0, 0.99], [40.0, 0.0, 2.0], 10.0, 1.01, True
    )
    goal_done, goal = evaluate(
        [40.0, 0.0, 2.8], [40.0, 0.0, 2.8], 0.0, 0.0
    )
    mask_env = UnityForestEnv.__new__(UnityForestEnv)
    mask_env.config = EnvConfig()
    mask_env.mpl = MotionPrimitiveLibrary()
    mask_env.action_space_n = mask_env.mpl.num_actions
    mask_env._collision_checker = None
    mask_env._last_action_mask_info = {}
    outside_mask = mask_env.get_action_mask(
        {"state": {"z": 0.99}, "safety": {"altitude_violation": True}}
    )
    mask_env._try_get_observation = lambda: None
    missing_state_mask = mask_env.get_action_mask()
    odometer_env = UnityForestEnv.__new__(UnityForestEnv)
    odometer_env._lock = threading.RLock()
    odometer_env._latest_state = None
    odometer_env._state_seq = 0
    odometer_env._trajectory_tracking_enabled = True
    odometer_env._trajectory_last_position = np.zeros(3, dtype=np.float64)
    odometer_env._trajectory_length_m = 0.0
    for position in ((3.0, 0.0, 0.0), (3.0, 4.0, 0.0)):
        odometer_env._on_state(
            SimpleNamespace(
                position=SimpleNamespace(
                    x=position[0], y=position[1], z=position[2]
                )
            )
        )
    checks = {
        "xy_only_not_success": not wrong_z["success"] and not wrong_z_done,
        "strict_altitude_terminal": low["altitude_violation"] and low_done,
        "valid_3d_goal_success": goal["success"] and goal_done,
        "outside_band_mask_fails_closed": not bool(outside_mask.any()),
        "missing_state_mask_fails_closed": not bool(missing_state_mask.any()),
        "actual_path_uses_state_polyline": abs(
            odometer_env.actual_trajectory_length_m() - 7.0
        )
        < 1.0e-9,
    }
    print("UNITY_ENV_TASK_CONTRACT_TEST")
    for name, passed in checks.items():
        print("  {}: {}".format(name, passed))
    passed = all(checks.values())
    print("RESULT={}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
