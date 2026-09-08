#!/usr/bin/env python3
"""Regression test for mutually exclusive post-action terminal outcomes."""

from __future__ import annotations



import numpy as np

from planning.mission.spec import MISSION_PLANAR_DISTANCE_M, REWARD_CONTRACT_ID
from planning.runtime.unity_env import (
    EnvConfig,
    UnityForestEnv,
    apply_post_action_dead_end,
)


def main() -> int:
    config = EnvConfig()
    maximum_potential_progress = (
        config.reward_progress_scale * MISSION_PLANAR_DISTANCE_M
        + config.reward_goal_z_progress_scale * (config.z_max - config.z_min)
    )
    for penalty in (
        config.reward_collision,
        config.reward_altitude_violation,
        config.reward_timeout,
        config.reward_far,
        config.reward_dead_end,
        config.reward_invalid_action,
    ):
        assert penalty + maximum_potential_progress < 0.0

    success_info = {"success": True, "dead_end": False, "done_reason": "success"}
    reward, done, info = apply_post_action_dead_end(
        reward=30.0,
        done=True,
        info=success_info,
        next_mask_info={"dead_end": True},
        terminate_on_dead_end=True,
        reward_dead_end=-20.0,
    )
    assert reward == 30.0
    assert done is True
    assert info["success"] is True
    assert info["dead_end"] is False
    assert info["done_reason"] == "success"

    active_info = {"success": False, "dead_end": False}
    reward, done, info = apply_post_action_dead_end(
        reward=2.0,
        done=False,
        info=active_info,
        next_mask_info={"dead_end": True},
        terminate_on_dead_end=True,
        reward_dead_end=-20.0,
    )
    assert reward == -18.0
    assert done is True
    assert info["dead_end"] is True
    assert info["done_reason"] == "dead_end"

    env = UnityForestEnv.__new__(UnityForestEnv)
    env.config = EnvConfig()
    env.prev_distance_xy = 1.0
    env.prev_abs_goal_dz = 0.0
    env.prev_action_id = -1
    env.episode_step = 0
    collision_at_goal = {
        "state": {"position": np.array([40.0, 0.0, 2.0]), "z": 2.0},
        "goal": {
            "position": np.array([40.0, 0.0, 2.0]),
            "distance_xy": 0.0,
            "dz": 0.0,
        },
        "safety": {
            "collided": True,
            "altitude_violation": False,
            "min_clearance": 0.0,
        },
        "action_mask": np.ones((1,), dtype=np.bool_),
    }
    reward, done, info = env._compute_reward_done(collision_at_goal, action_id=0)
    assert done is True
    assert info["collided"] is True
    assert info["success"] is False
    assert info["done_reason"] == "collision"
    assert reward < float(env.config.reward_success)

    env.prev_distance_xy = 1.0
    env.prev_abs_goal_dz = 0.0
    env.episode_step = env.config.max_episode_steps - 1
    clean_goal = {
        **collision_at_goal,
        "safety": {
            "collided": False,
            "altitude_violation": False,
            "min_clearance": 1.0,
        },
    }
    reward, done, info = env._compute_reward_done(clean_goal, action_id=0)
    assert done is True
    assert info["success"] is True
    assert info["timeout"] is False

    env.prev_distance_xy = 20.0
    env.prev_abs_goal_dz = 0.0
    env.episode_step = env.config.max_episode_steps - 1
    timeout_observation = {
        **collision_at_goal,
        "state": {"position": np.array([20.0, 0.0, 2.0]), "z": 2.0},
        "goal": {
            "position": np.array([40.0, 0.0, 2.0]),
            "distance_xy": 20.0,
            "dz": 0.0,
        },
        "safety": {
            "collided": False,
            "altitude_violation": False,
            "min_clearance": 1.0,
        },
    }
    reward, done, info = env._compute_reward_done(
        timeout_observation, action_id=0
    )
    assert done is True
    assert info["timeout"] is True
    assert info["done_reason"] == "timeout"

    env.prev_distance_xy = 1.0
    env.prev_abs_goal_dz = 0.0
    env.episode_step = 0
    collision_with_empty_mask = {
        **collision_at_goal,
        "state": {"position": np.array([20.0, 0.0, 2.0]), "z": 2.0},
        "goal": {
            "position": np.array([40.0, 0.0, 2.0]),
            "distance_xy": 20.0,
            "dz": 0.0,
        },
        "action_mask": np.zeros((1,), dtype=np.bool_),
    }
    reward, done, info = env._compute_reward_done(collision_with_empty_mask, action_id=0)
    assert done is True
    assert info["collided"] is True
    assert info["dead_end"] is False

    print("UNITY_TERMINAL_PRECEDENCE_TEST")
    print("  reward_contract_id:", REWARD_CONTRACT_ID)
    print("  terminal_failure_dominates_progress: True")
    print("  success_not_reclassified_as_dead_end: True")
    print("  active_dead_end_still_terminates: True")
    print("  collision_at_goal_is_not_success: True")
    print("  success_at_step_limit_is_not_timeout: True")
    print("  timeout_reason_is_explicit: True")
    print("  collision_with_empty_mask_is_not_dead_end: True")
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
