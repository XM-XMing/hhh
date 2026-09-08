#!/usr/bin/env python3
"""Deterministic contract test for bounded teacher beam selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


from planning.teacher.policy import (
    TEACHER_PLANNING_CONTRACT_ID,
    ReachabilityTeacher,
    TeacherConfig,
)


class SyntheticBeamTeacher(ReachabilityTeacher):
    """Small score graph: greedy takes action 0, lookahead must take action 1."""

    def __init__(self, config: TeacherConfig, unsafe_leaves: bool = False):
        self.config = config
        self.motion_primitives = SimpleNamespace(num_actions=3)
        self.calls = 0
        self.successor_modes = []
        self.unsafe_leaves = bool(unsafe_leaves)

    def _score_state_one_step(
        self, position, yaw, velocity, goal, prev_action=-1, evaluate_successors=True
    ):
        del yaw, velocity, goal, prev_action
        self.calls += 1
        self.successor_modes.append(bool(evaluate_successors))
        x_value = int(round(float(np.asarray(position)[0])))
        if x_value == 0:
            scores = np.asarray([10.0, 9.0, -np.inf], dtype=np.float32)
        elif x_value == 1:
            scores = np.asarray([-10.0, -10.0, -np.inf], dtype=np.float32)
        elif x_value == 2:
            scores = np.asarray([10.0, 10.0, -np.inf], dtype=np.float32)
        else:
            scores = np.asarray([0.0, 0.0, -np.inf], dtype=np.float32)
        best = int(np.nanargmax(scores))
        mask = np.isfinite(scores)
        target = np.zeros(3, dtype=np.float32)
        target[best] = 1.0
        zeros = np.zeros(3, dtype=np.float32)
        return best, {
            "dead_end_now": False,
            "valid_count": 2,
            "candidate_count": 2,
            "teacher_scores": scores,
            "teacher_soft_target": target,
            "teacher_entropy": 0.0,
            "global_action_mask": mask,
            "raw_global_action_mask": mask,
            "height_action_mask": mask,
            "guidance_goal": np.asarray([100.0, 0.0, 2.0], dtype=np.float32),
            "_progress_all": zeros,
            "_goal_distance_all": np.full(3, 100.0, dtype=np.float32),
            "_guidance_distance_all": np.full(3, 100.0, dtype=np.float32),
            "_velocity_mismatch_all": zeros,
            "_current_clearance_all": np.ones(3, dtype=np.float32),
            "_evaluated_action_mask": mask,
            "_next_valid_count_all": np.asarray([2, 2, -1], dtype=np.int32),
            "_next_clearance_all": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            "_goal_reached_mask": np.zeros(3, dtype=np.bool_),
        }

    def _predict_action(self, position, yaw, action_id):
        del yaw
        output = np.asarray(position, dtype=np.float32).copy()
        if int(round(float(output[0]))) == 0:
            output[0] = 1.0 if int(action_id) == 0 else 2.0
        else:
            output[0] += 3.0 + float(action_id)
        return output, 0.0, np.zeros(3, dtype=np.float32)

    def _evaluate_candidates(self, action_ids, endpoints, yaw):
        del endpoints, yaw
        return {int(action_id): (2, 1.0) for action_id in action_ids}

    def _evaluate_pose_reachability(self, positions, yaws):
        del yaws
        count = np.asarray(positions).shape[0]
        valid_count = 0 if self.unsafe_leaves else 2
        return (
            np.full(count, valid_count, dtype=np.int32),
            np.ones(count, dtype=np.float32),
        )

    def endpoints_world(self, position, yaw):
        del position, yaw
        return np.zeros((3, 3), dtype=np.float32)


class FastDeadEndBeamTeacher(SyntheticBeamTeacher):
    """Fast beam fixture whose first successor is an explicit dead end."""

    def _score_state_one_step_fast(
        self, position, yaw, velocity, goal, prev_action=-1, evaluate_successors=True
    ):
        del yaw, velocity, goal, prev_action, evaluate_successors
        self.calls += 1
        x_value = int(round(float(np.asarray(position)[0])))
        if x_value != 0:
            return -1, {"dead_end_now": True}
        return 0, {
            "dead_end_now": False,
            "teacher_scores": np.asarray([10.0, 9.0, -np.inf], dtype=np.float32),
        }


def main() -> int:
    config = TeacherConfig(beam_depth=3, beam_width=2, beam_branching=2)
    teacher = SyntheticBeamTeacher(config)
    action_id, debug = teacher.score_state(
        position=[0.0, 0.0, 2.0],
        yaw=0.0,
        velocity=[0.0, 0.0, 0.0],
        goal=[100.0, 0.0, 2.0],
    )
    fallback_teacher = SyntheticBeamTeacher(config, unsafe_leaves=True)
    fallback_action, fallback_debug = fallback_teacher.score_state(
        position=[0.0, 0.0, 2.0],
        yaw=0.0,
        velocity=[0.0, 0.0, 0.0],
        goal=[100.0, 0.0, 2.0],
    )
    fast_dead_end_teacher = FastDeadEndBeamTeacher(config)
    fast_dead_end_action, fast_dead_end_debug = fast_dead_end_teacher.score_state_fast(
        position=[0.0, 0.0, 2.0],
        yaw=0.0,
        velocity=[0.0, 0.0, 0.0],
        goal=[100.0, 0.0, 2.0],
    )
    route_receiver = object.__new__(ReachabilityTeacher)
    route_receiver._route = None
    route_receiver._route_goal = None
    shared_route = np.asarray(
        [[0.0, 0.0, 2.0], [1.0, 1.0, 2.0], [2.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    route_info = route_receiver.set_mission_route(
        shared_route, [2.0, 1.0, 2.0]
    )
    copied_route = route_receiver.mission_route
    shared_route[0, 0] = 99.0
    checks = {
        "beam_beats_greedy_first_action": action_id == 1,
        "contract_is_semantic": debug["teacher_planning_contract_id"]
        == TEACHER_PLANNING_CONTRACT_ID,
        "sequence_starts_with_selected_action": debug["beam_best_sequence"].startswith(
            "1;"
        ),
        "search_is_bounded": teacher.calls <= 1 + config.beam_width * (config.beam_depth - 1),
        "root_uses_real_successor_check": teacher.successor_modes[0] is True,
        "unsafe_leaves_fall_back_to_greedy": fallback_action == 0
        and fallback_debug["beam_fallback"] is True
        and fallback_debug["beam_fallback_reason"] == "no_viable_leaf",
        "fast_dead_end_successor_falls_back": fast_dead_end_action == 0
        and fast_dead_end_debug["dead_end_now"] is False,
        "soft_target_is_normalized": abs(
            float(np.asarray(debug["teacher_soft_target"]).sum()) - 1.0
        )
        < 1.0e-6,
        "audited_route_can_be_reused": (
            route_info["route_points"] == 3
            and abs(route_info["route_length_m"] - (2.0 ** 0.5 + 1.0))
            < 1.0e-6
            and copied_route[0, 0] == 0.0
            and route_receiver.mission_route[0, 0] == 0.0
        ),
    }
    print("TEACHER_BEAM_TEST")
    for name, passed in checks.items():
        print("  {}: {}".format(name, passed))
    passed = all(checks.values())
    print("RESULT={}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
