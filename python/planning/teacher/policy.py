"""Reachability-aware privileged teacher for rollout collection and relabeling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

from planning.mission.global_route import (
    DEFAULT_GLOBAL_ROUTE_RESOLUTION_M,
    DEFAULT_GLOBAL_ROUTE_LOOKAHEAD_M,
    DEFAULT_GLOBAL_ROUTE_TRACKING_MARGIN_M,
    GLOBAL_ROUTE_CONTRACT_ID,
    GlobalRouteConfig,
    GlobalRoutePlanner2D,
)
from planning.mission.spec import (
    ACTION_MASK_Z_MARGIN_M,
    FLIGHT_Z_MAX_M,
    FLIGHT_Z_MIN_M,
    GOAL_RADIUS_XY_M,
    GOAL_TOLERANCE_Z_M,
)
from planning.primitives.library import rotation_z

TEACHER_PLANNING_CONTRACT_ID = "global_route_bounded_beam"

def distance_xy(position: np.ndarray, goal: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.asarray(position, dtype=np.float32)[:2]
            - np.asarray(goal, dtype=np.float32)[:2]
        )
    )

@dataclass(frozen=True)
class TeacherConfig:
    candidate_top_k: int = 12
    current_check_step: int = 1
    lookahead_check_step: int = 2
    z_min: float = FLIGHT_Z_MIN_M
    z_max: float = FLIGHT_Z_MAX_M
    action_mask_z_margin: float = ACTION_MASK_Z_MARGIN_M
    goal_radius: float = GOAL_RADIUS_XY_M
    goal_tolerance_z: float = GOAL_TOLERANCE_Z_M
    min_next_valid_count: int = 1
    progress_weight: float = 1.0
    next_valid_weight: float = 0.01
    current_clearance_weight: float = 0.0
    next_clearance_weight: float = 0.02
    clearance_clip: float = 1.0
    z_weight: float = 0.05
    goal_heading_weight: float = 0.50
    turn_penalty_weight: float = 0.15
    action_change_weight: float = 0.05
    velocity_transition_weight: float = 0.08
    velocity_transition_clip: float = 2.0
    deadend_penalty: float = -100.0
    goal_success_bonus: float = 20.0
    temperature: float = 0.35
    soft_support_margin: float = 1.0
    unscored_penalty: float = 0.50
    unscored_min_gap: float = 0.10
    global_route_resolution_m: float = DEFAULT_GLOBAL_ROUTE_RESOLUTION_M
    global_route_lookahead_m: float = DEFAULT_GLOBAL_ROUTE_LOOKAHEAD_M
    global_route_tracking_margin_m: float = DEFAULT_GLOBAL_ROUTE_TRACKING_MARGIN_M
    beam_depth: int = 3
    beam_width: int = 8
    beam_branching: int = 4
    beam_discount: float = 0.95


DEFAULT_TEACHER_CONFIG = TeacherConfig()


def add_teacher_arguments(
    parser,
    *,
    candidate_top_k: int,
    current_check_step: int = 1,
    lookahead_check_step: int = 2,
) -> None:
    parser.add_argument("--candidate-top-k", type=int, default=int(candidate_top_k))
    parser.add_argument("--current-check-step", type=int, default=int(current_check_step))
    parser.add_argument("--lookahead-check-step", type=int, default=int(lookahead_check_step))
    parser.add_argument(
        "--goal-radius", type=float, default=DEFAULT_TEACHER_CONFIG.goal_radius
    )
    parser.add_argument(
        "--goal-z-tolerance",
        dest="goal_tolerance_z",
        type=float,
        default=DEFAULT_TEACHER_CONFIG.goal_tolerance_z,
    )
    parser.add_argument(
        "--min-next-valid-count",
        type=int,
        default=DEFAULT_TEACHER_CONFIG.min_next_valid_count,
        help="minimum valid actions required at the predicted next state",
    )
    parser.add_argument("--progress-weight", type=float, default=DEFAULT_TEACHER_CONFIG.progress_weight)
    parser.add_argument("--next-valid-weight", type=float, default=DEFAULT_TEACHER_CONFIG.next_valid_weight)
    parser.add_argument("--current-clearance-weight", type=float, default=DEFAULT_TEACHER_CONFIG.current_clearance_weight)
    parser.add_argument("--next-clearance-weight", type=float, default=DEFAULT_TEACHER_CONFIG.next_clearance_weight)
    parser.add_argument("--clearance-clip", type=float, default=DEFAULT_TEACHER_CONFIG.clearance_clip)
    parser.add_argument("--z-weight", type=float, default=DEFAULT_TEACHER_CONFIG.z_weight)
    parser.add_argument("--goal-heading-weight", type=float, default=DEFAULT_TEACHER_CONFIG.goal_heading_weight)
    parser.add_argument("--turn-penalty-weight", type=float, default=DEFAULT_TEACHER_CONFIG.turn_penalty_weight)
    parser.add_argument("--action-change-weight", type=float, default=DEFAULT_TEACHER_CONFIG.action_change_weight)
    parser.add_argument("--velocity-transition-weight", type=float, default=DEFAULT_TEACHER_CONFIG.velocity_transition_weight)
    parser.add_argument("--velocity-transition-clip", type=float, default=DEFAULT_TEACHER_CONFIG.velocity_transition_clip)
    parser.add_argument("--deadend-penalty", type=float, default=DEFAULT_TEACHER_CONFIG.deadend_penalty)
    parser.add_argument("--goal-success-bonus", type=float, default=DEFAULT_TEACHER_CONFIG.goal_success_bonus)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEACHER_CONFIG.temperature)
    parser.add_argument("--soft-support-margin", type=float, default=DEFAULT_TEACHER_CONFIG.soft_support_margin)
    parser.add_argument("--unscored-penalty", type=float, default=DEFAULT_TEACHER_CONFIG.unscored_penalty)
    parser.add_argument("--unscored-min-gap", type=float, default=DEFAULT_TEACHER_CONFIG.unscored_min_gap)
    parser.add_argument("--global-route-resolution-m", type=float, default=DEFAULT_TEACHER_CONFIG.global_route_resolution_m)
    parser.add_argument("--global-route-lookahead-m", type=float, default=DEFAULT_TEACHER_CONFIG.global_route_lookahead_m)
    parser.add_argument("--global-route-tracking-margin-m", type=float, default=DEFAULT_TEACHER_CONFIG.global_route_tracking_margin_m)
    parser.add_argument("--beam-depth", type=int, default=DEFAULT_TEACHER_CONFIG.beam_depth)
    parser.add_argument("--beam-width", type=int, default=DEFAULT_TEACHER_CONFIG.beam_width)
    parser.add_argument("--beam-branching", type=int, default=DEFAULT_TEACHER_CONFIG.beam_branching)
    parser.add_argument("--beam-discount", type=float, default=DEFAULT_TEACHER_CONFIG.beam_discount)

def config_from_args(args) -> TeacherConfig:
    values = {
        name: getattr(args, name)
        for name in TeacherConfig.__dataclass_fields__
        if hasattr(args, name)
    }
    config = TeacherConfig(**values)
    if abs(float(config.goal_radius) - GOAL_RADIUS_XY_M) > 1.0e-9:
        raise ValueError(
            "--goal-radius is fixed by the task contract at {}".format(GOAL_RADIUS_XY_M)
        )
    if abs(float(config.goal_tolerance_z) - GOAL_TOLERANCE_Z_M) > 1.0e-9:
        raise ValueError(
            "--goal-z-tolerance is fixed by the task contract at {}".format(
                GOAL_TOLERANCE_Z_M
            )
        )
    if int(config.beam_depth) <= 0:
        raise ValueError("--beam-depth must be positive")
    if int(config.beam_width) <= 0:
        raise ValueError("--beam-width must be positive")
    if int(config.beam_branching) <= 0:
        raise ValueError("--beam-branching must be positive")
    if not 0.0 < float(config.beam_discount) <= 1.0:
        raise ValueError("--beam-discount must be in (0, 1]")
    return config

class ReachabilityTeacher:
    """Global-map teacher with batched C++ lookahead collision evaluation."""

    def __init__(self, motion_primitives, collision_checker, config: TeacherConfig):
        task_values = (
            ("z_min", config.z_min, FLIGHT_Z_MIN_M),
            ("z_max", config.z_max, FLIGHT_Z_MAX_M),
            ("action_mask_z_margin", config.action_mask_z_margin, ACTION_MASK_Z_MARGIN_M),
            ("goal_radius", config.goal_radius, GOAL_RADIUS_XY_M),
            ("goal_tolerance_z", config.goal_tolerance_z, GOAL_TOLERANCE_Z_M),
        )
        mismatches = [
            "{}={} expected {}".format(name, actual, expected)
            for name, actual, expected in task_values
            if abs(float(actual) - float(expected)) > 1.0e-9
        ]
        if mismatches:
            raise ValueError(
                "TeacherConfig overrides hashed task contract: " + "; ".join(mismatches)
            )
        self.motion_primitives = motion_primitives
        self.collision_checker = collision_checker
        self.config = config
        self._endpoint_body = np.asarray(
            motion_primitives.pos_ref[:, -1, :], dtype=np.float32
        )
        self._terminal_body_velocity = np.asarray(
            motion_primitives.cmd_seq[:, -1, :3], dtype=np.float32
        )
        self._start_body_velocity = np.asarray(
            motion_primitives.cmd_seq[:, 0, :3], dtype=np.float32
        )
        self._all_action_ids = np.arange(
            motion_primitives.num_actions, dtype=np.int32
        )
        self.global_route_contract_id = GLOBAL_ROUTE_CONTRACT_ID
        self.teacher_planning_contract_id = TEACHER_PLANNING_CONTRACT_ID
        native_geometry_context = getattr(
            collision_checker, "native_geometry_context", None
        )
        route_config = GlobalRouteConfig(
            resolution_m=float(config.global_route_resolution_m),
            flight_z_min_m=float(config.z_min),
            flight_z_max_m=float(config.z_max),
            lookahead_m=float(config.global_route_lookahead_m),
            tracking_margin_m=float(config.global_route_tracking_margin_m),
        )
        self._route_planner = (
            native_geometry_context.global_route_planner(
                collision_checker, route_config
            )
            if native_geometry_context is not None
            else GlobalRoutePlanner2D(collision_checker, route_config)
            if collision_checker is not None
            else None
        )
        self._route = None
        self._route_goal = None

    def set_mission(self, start: Sequence[float], goal: Sequence[float]) -> Dict[str, Any]:
        """Build the immutable privileged route used for one mission."""
        if self._route_planner is None:
            self._route = None
            self._route_goal = np.asarray(goal, dtype=np.float32).reshape(3)
            return {"route_points": 0, "route_length_m": 0.0}
        return self.set_mission_route(self._route_planner.plan(start, goal), goal)

    def set_mission_route(
        self, route: Sequence[Sequence[float]], goal: Sequence[float]
    ) -> Dict[str, Any]:
        """Reuse an audited global route while retaining this teacher's local checker."""

        route_array = np.asarray(route, dtype=np.float32)
        goal_array = np.asarray(goal, dtype=np.float32).reshape(3)
        if (
            route_array.ndim != 2
            or route_array.shape[0] < 2
            or route_array.shape[1] != 3
            or not np.all(np.isfinite(route_array))
        ):
            raise ValueError("mission route must be a finite Nx3 array with N >= 2")
        if not np.allclose(route_array[-1], goal_array, atol=1.0e-3):
            raise ValueError("mission route endpoint does not match final goal")
        self._route = np.array(route_array, dtype=np.float32, copy=True)
        self._route_goal = goal_array
        length = float(
            np.sum(np.linalg.norm(np.diff(self._route[:, :2], axis=0), axis=1))
        )
        return {"route_points": int(self._route.shape[0]), "route_length_m": length}

    @property
    def mission_route(self) -> np.ndarray:
        if self._route is None:
            raise RuntimeError("mission route has not been initialized")
        return np.array(self._route, dtype=np.float32, copy=True)

    def _guidance_goal(self, position: np.ndarray, final_goal: np.ndarray) -> np.ndarray:
        if (
            self._route_planner is None
            or self._route is None
            or self._route_goal is None
            or not np.allclose(self._route_goal, final_goal, atol=1.0e-3)
        ):
            return final_goal
        guidance = self._route_planner.guidance_goal(self._route, position)
        # Altitude remains a genuinely 3D final-goal objective; the coarse
        # route only supplies global XY topology.
        guidance[2] = final_goal[2]
        return guidance

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def _collision_info(
        self, position: np.ndarray, yaw: float, check_step: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        arrays = self.collision_checker.check_all_actions_array(
            self.motion_primitives,
            np.asarray(position, dtype=np.float32).reshape(3),
            float(yaw),
            check_step=max(1, int(check_step)),
        )
        return (
            np.asarray(arrays["valid"], dtype=np.bool_),
            np.asarray(arrays["minimum_distances"], dtype=np.float32),
        )

    def endpoints_world(self, position: np.ndarray, yaw: float) -> np.ndarray:
        return np.asarray(position, dtype=np.float32).reshape(1, 3) + self._endpoint_body.dot(
            rotation_z(float(yaw)).T
        )

    def terminal_yaw(self, action_id: int, yaw: float) -> float:
        return float(
            yaw
            + float(
                self.motion_primitives.terminal_heading_rad[int(action_id)]
            )
        )

    @staticmethod
    def _rotate_body_velocities(
        body_velocity: np.ndarray, world_yaw: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(body_velocity, dtype=np.float32).reshape(-1, 3)
        angles = np.asarray(world_yaw, dtype=np.float32).reshape(-1)
        cosine = np.cos(angles)
        sine = np.sin(angles)
        output = np.empty_like(values)
        output[:, 0] = cosine * values[:, 0] - sine * values[:, 1]
        output[:, 1] = sine * values[:, 0] + cosine * values[:, 1]
        output[:, 2] = values[:, 2]
        return output

    def terminal_velocity_world_all(self, yaw: float) -> np.ndarray:
        terminal_yaws = float(yaw) + self.motion_primitives.terminal_heading_rad
        return self._rotate_body_velocities(
            self._terminal_body_velocity, terminal_yaws
        )

    def command_velocity_world(
        self, action_id: int, yaw: float, terminal: bool = True
    ) -> np.ndarray:
        ids = np.asarray([int(action_id)], dtype=np.int64)
        body = (
            self._terminal_body_velocity[ids]
            if terminal
            else self._start_body_velocity[ids]
        )
        angles = (
            float(yaw) + self.motion_primitives.terminal_heading_rad[ids]
            if terminal
            else np.asarray([float(yaw)], dtype=np.float32)
        )
        return self._rotate_body_velocities(body, angles)[0]

    def _evaluate_candidates(
        self,
        action_ids: Sequence[int],
        endpoints: np.ndarray,
        yaw: float,
    ) -> Dict[int, Tuple[int, float]]:
        ids = np.asarray(action_ids, dtype=np.int32).reshape(-1)
        if ids.size == 0:
            return {}
        positions = np.asarray(endpoints[ids], dtype=np.float32)
        next_yaws = (
            float(yaw) + self.motion_primitives.terminal_heading_rad[ids]
        ).astype(np.float64)
        counts, minimum_clearance = self._evaluate_pose_reachability(
            positions, next_yaws
        )
        return {
            int(action_id): (int(counts[index]), float(minimum_clearance[index]))
            for index, action_id in enumerate(ids)
        }

    def _evaluate_pose_reachability(
        self,
        positions: np.ndarray,
        yaws: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch-check whether each predicted pose has a safe next action."""
        pose_array = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        yaw_array = np.asarray(yaws, dtype=np.float64).reshape(-1)
        if pose_array.shape[0] != yaw_array.shape[0]:
            raise ValueError("positions and yaws must have equal lengths")
        if pose_array.shape[0] == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
        arrays = self.collision_checker.check_pose_actions_array(
            self.motion_primitives,
            pose_array,
            yaw_array,
            self._all_action_ids,
            check_step=max(1, int(self.config.lookahead_check_step)),
        )
        global_valid = np.asarray(arrays["valid"], dtype=np.bool_)
        clearance = np.asarray(arrays["minimum_distances"], dtype=np.float32)
        next_z = pose_array[:, 2:3] + self.motion_primitives.z_end.reshape(1, -1)
        height_valid = (
            (next_z >= float(self.config.z_min) + float(self.config.action_mask_z_margin))
            & (next_z <= float(self.config.z_max) - float(self.config.action_mask_z_margin))
        )
        combined = global_valid & height_valid
        counts = np.count_nonzero(combined, axis=1).astype(np.int32)
        minimum_clearance = np.zeros((pose_array.shape[0],), dtype=np.float32)
        for row in range(pose_array.shape[0]):
            finite = clearance[row, combined[row]]
            finite = finite[np.isfinite(finite)]
            minimum_clearance[row] = (
                float(np.min(finite))
                if finite.size
                else (
                    float(self.config.clearance_clip)
                    if np.any(combined[row])
                    else 0.0
                )
            )
        return counts, minimum_clearance

    @staticmethod
    def _softmax(
        scores: np.ndarray, support: np.ndarray, temperature: float
    ) -> np.ndarray:
        output = np.zeros_like(scores, dtype=np.float32)
        ids = np.flatnonzero(np.asarray(support, dtype=np.bool_) & np.isfinite(scores))
        if ids.size == 0:
            return output
        values = scores[ids].astype(np.float64) / max(1.0e-6, float(temperature))
        values -= float(np.max(values))
        weights = np.exp(values)
        output[ids] = (
            weights / max(1.0e-12, float(np.sum(weights)))
        ).astype(np.float32)
        return output

    def _score_state_one_step_impl(
        self,
        position: Sequence[float],
        yaw: float,
        velocity: Sequence[float],
        goal: Sequence[float],
        prev_action: int = -1,
        evaluate_successors: bool = True,
        detailed: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        position_array = np.asarray(position, dtype=np.float32).reshape(3)
        velocity_array = np.asarray(velocity, dtype=np.float32).reshape(3)
        goal_array = np.asarray(goal, dtype=np.float32).reshape(3)
        config = self.config

        height_mask = self.motion_primitives.valid_action_mask(
            float(position_array[2]),
            z_min=config.z_min,
            z_max=config.z_max,
            margin=config.action_mask_z_margin,
        )
        global_mask, current_clearance = self._collision_info(
            position_array, float(yaw), config.current_check_step
        )
        current_mask = height_mask & global_mask
        valid_ids = np.flatnonzero(current_mask).astype(np.int64)
        scores = np.full(
            (self.motion_primitives.num_actions,), -np.inf, dtype=np.float32
        )
        if valid_ids.size == 0:
            if not detailed:
                return -1, {"dead_end_now": True}
            return -1, {
                "dead_end_now": True,
                "valid_count": 0,
                "candidate_count": 0,
                "teacher_scores": scores,
                "teacher_soft_target": np.zeros_like(scores),
                "global_action_mask": current_mask,
                "raw_global_action_mask": global_mask,
                "height_action_mask": height_mask,
                "teacher_entropy": 0.0,
            }

        endpoints = self.endpoints_world(position_array, float(yaw))
        guidance_goal = self._guidance_goal(position_array, goal_array)
        before_distance = distance_xy(position_array, guidance_goal)
        after_guidance_distance = np.linalg.norm(
            endpoints[:, :2] - guidance_goal[None, :2], axis=1
        ).astype(np.float32)
        after_goal_distance = np.linalg.norm(
            endpoints[:, :2] - goal_array[None, :2], axis=1
        ).astype(np.float32)
        progress = (before_distance - after_guidance_distance).astype(np.float32)
        z_error = np.abs(endpoints[:, 2] - float(goal_array[2])).astype(np.float32)
        goal_reached_mask = (
            (after_goal_distance <= float(config.goal_radius))
            & (z_error <= float(config.goal_tolerance_z))
        )
        goal_heading = math.atan2(
            float(guidance_goal[1] - position_array[1]),
            float(guidance_goal[0] - position_array[0]),
        )
        terminal_yaws = float(yaw) + self.motion_primitives.terminal_heading_rad
        heading_error = np.abs(
            np.arctan2(
                np.sin(terminal_yaws - goal_heading),
                np.cos(terminal_yaws - goal_heading),
            )
        ).astype(np.float32)
        turn_magnitude = np.abs(
            self.motion_primitives.terminal_heading_rad
        ).astype(np.float32)
        clipped_clearance = np.minimum(
            float(config.clearance_clip),
            np.maximum(0.0, current_clearance),
        ).astype(np.float32)
        base_scores = (
            float(config.progress_weight) * progress
            - float(config.z_weight) * z_error
            - float(config.goal_heading_weight) * heading_error
            - float(config.turn_penalty_weight) * turn_magnitude
            + float(config.current_clearance_weight) * clipped_clearance
        ).astype(np.float32)
        if 0 <= int(prev_action) < self.motion_primitives.num_actions:
            changed = np.ones(
                (self.motion_primitives.num_actions,), dtype=np.float32
            )
            changed[int(prev_action)] = 0.0
            base_scores -= float(config.action_change_weight) * changed

        terminal_velocity = self.terminal_velocity_world_all(float(yaw))
        velocity_mismatch = np.linalg.norm(
            terminal_velocity - velocity_array.reshape(1, 3), axis=1
        ).astype(np.float32)
        scores[valid_ids] = base_scores[valid_ids] - float(config.unscored_penalty)
        ordered = valid_ids[np.argsort(base_scores[valid_ids])[::-1]]
        if not evaluate_successors:
            candidate_ids = valid_ids
        elif int(config.candidate_top_k) > 0:
            candidate_ids = ordered[: min(int(config.candidate_top_k), ordered.size)]
        else:
            candidate_ids = ordered

        if evaluate_successors:
            next_metrics = self._evaluate_candidates(
                candidate_ids.tolist(), endpoints, float(yaw)
            )
        else:
            next_metrics = {
                int(action_id): (
                    int(config.min_next_valid_count),
                    float(current_clearance[int(action_id)]),
                )
                for action_id in candidate_ids.tolist()
            }
        evaluated = set(int(value) for value in candidate_ids.tolist())

        def refined_score(
            action_id: int, valid_count: int, next_clearance: float
        ) -> float:
            value = float(base_scores[action_id])
            if evaluate_successors:
                value += float(config.next_valid_weight) * math.log1p(
                    max(0, int(valid_count))
                )
                value += float(config.next_clearance_weight) * min(
                    float(config.clearance_clip), max(0.0, float(next_clearance))
                )
            value -= float(config.velocity_transition_weight) * min(
                float(config.velocity_transition_clip),
                float(velocity_mismatch[action_id]),
            )
            if evaluate_successors and int(valid_count) < int(config.min_next_valid_count):
                value += float(config.deadend_penalty)
            if bool(goal_reached_mask[action_id]):
                value += float(config.goal_success_bonus)
            return value

        for action_id, (valid_count, next_clearance) in next_metrics.items():
            scores[action_id] = refined_score(
                action_id, valid_count, next_clearance
            )

        non_deadend = [
            action_id
            for action_id, (valid_count, _) in next_metrics.items()
            if int(valid_count) >= int(config.min_next_valid_count)
            or bool(goal_reached_mask[action_id])
        ]
        if evaluate_successors and not non_deadend and len(evaluated) < valid_ids.size:
            remaining = [
                int(action_id)
                for action_id in ordered.tolist()
                if int(action_id) not in evaluated
            ]
            expanded = self._evaluate_candidates(
                remaining, endpoints, float(yaw)
            )
            next_metrics.update(expanded)
            evaluated.update(remaining)
            for action_id, (valid_count, next_clearance) in expanded.items():
                scores[action_id] = refined_score(
                    action_id, valid_count, next_clearance
                )

        evaluated_ids = np.asarray(sorted(evaluated), dtype=np.int64)
        best_action = int(evaluated_ids[np.argmax(scores[evaluated_ids])])
        best_score = float(scores[best_action])
        non_evaluated = np.asarray(
            [
                int(action_id)
                for action_id in valid_ids.tolist()
                if int(action_id) not in evaluated
            ],
            dtype=np.int64,
        )
        if non_evaluated.size:
            scores[non_evaluated] = np.minimum(
                scores[non_evaluated],
                best_score - float(config.unscored_min_gap),
            )

        if not detailed:
            evaluated_mask = np.zeros(
                (self.motion_primitives.num_actions,), dtype=np.bool_
            )
            next_valid_count_all = np.full(
                (self.motion_primitives.num_actions,), -1, dtype=np.int32
            )
            for action_id, (valid_count, _next_clearance) in next_metrics.items():
                evaluated_mask[int(action_id)] = True
                next_valid_count_all[int(action_id)] = int(valid_count)
            return best_action, {
                "dead_end_now": False,
                "valid_count": int(valid_ids.size),
                "candidate_count": int(len(evaluated)),
                "teacher_scores": scores,
                "_evaluated_action_mask": evaluated_mask,
                "_next_valid_count_all": next_valid_count_all,
                "_goal_reached_mask": goal_reached_mask,
            }

        support = current_mask & np.isfinite(scores)
        if float(config.soft_support_margin) >= 0.0:
            support &= scores >= best_score - float(config.soft_support_margin)
        support[best_action] = True
        soft_target = self._softmax(scores, support, config.temperature)
        probabilities = soft_target[soft_target > 0.0]
        entropy = float(
            -np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-12)))
        )
        chosen_valid_count, chosen_clearance = next_metrics[best_action]
        evaluated_mask = np.zeros(
            (self.motion_primitives.num_actions,), dtype=np.bool_
        )
        next_valid_count_all = np.full(
            (self.motion_primitives.num_actions,), -1, dtype=np.int32
        )
        next_clearance_all = np.zeros(
            (self.motion_primitives.num_actions,), dtype=np.float32
        )
        for action_id, (valid_count, next_clearance) in next_metrics.items():
            evaluated_mask[int(action_id)] = True
            next_valid_count_all[int(action_id)] = int(valid_count)
            next_clearance_all[int(action_id)] = float(next_clearance)
        return best_action, {
            "dead_end_now": False,
            "valid_count": int(valid_ids.size),
            "candidate_count": int(len(evaluated)),
            "chosen_score": best_score,
            "chosen_progress": float(progress[best_action]),
            "chosen_dist_after_pred": float(after_goal_distance[best_action]),
            "chosen_guidance_dist_after_pred": float(after_guidance_distance[best_action]),
            "guidance_goal": guidance_goal.copy(),
            "chosen_next_valid_count": int(chosen_valid_count),
            "chosen_next_min_clearance": float(chosen_clearance),
            "chosen_next_dead_end": bool(
                int(chosen_valid_count) < int(config.min_next_valid_count)
            ),
            "chosen_velocity_mismatch_mps": float(
                velocity_mismatch[best_action]
            ),
            "teacher_scores": scores,
            "teacher_soft_target": soft_target,
            "teacher_entropy": entropy,
            "global_action_mask": current_mask,
            "raw_global_action_mask": global_mask,
            "height_action_mask": height_mask,
            "soft_support_count": int(np.count_nonzero(support)),
            "best_top_actions": ";".join(
                str(int(action_id))
                for action_id in valid_ids[
                    np.argsort(scores[valid_ids])[::-1]
                ][:5]
            ),
            "_progress_all": progress,
            "_goal_distance_all": after_goal_distance,
            "_guidance_distance_all": after_guidance_distance,
            "_velocity_mismatch_all": velocity_mismatch,
            "_current_clearance_all": current_clearance,
            "_evaluated_action_mask": evaluated_mask,
            "_next_valid_count_all": next_valid_count_all,
            "_next_clearance_all": next_clearance_all,
            "_goal_reached_mask": goal_reached_mask,
        }

    def _score_state_one_step(
        self,
        position: Sequence[float],
        yaw: float,
        velocity: Sequence[float],
        goal: Sequence[float],
        prev_action: int = -1,
        evaluate_successors: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        """Compatibility wrapper for the detailed one-step contract."""
        return self._score_state_one_step_impl(
            position,
            yaw,
            velocity,
            goal,
            prev_action,
            evaluate_successors,
            detailed=True,
        )

    def _score_state_one_step_fast(
        self,
        position: Sequence[float],
        yaw: float,
        velocity: Sequence[float],
        goal: Sequence[float],
        prev_action: int = -1,
        evaluate_successors: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select an action while omitting unused detailed diagnostics."""
        return self._score_state_one_step_impl(
            position,
            yaw,
            velocity,
            goal,
            prev_action,
            evaluate_successors,
            detailed=False,
        )

    @staticmethod
    def _public_debug(debug: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in debug.items() if not key.startswith("_")}

    def _predict_action(
        self,
        position: np.ndarray,
        yaw: float,
        action_id: int,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        next_position = np.asarray(position, dtype=np.float32) + rotation_z(yaw).dot(
            self.motion_primitives.endpoint(int(action_id))
        )
        next_yaw = self.terminal_yaw(int(action_id), yaw)
        next_velocity = self.command_velocity_world(int(action_id), yaw, terminal=True)
        return next_position, next_yaw, next_velocity

    def _score_state_impl(
        self,
        position: Sequence[float],
        yaw: float,
        velocity: Sequence[float],
        goal: Sequence[float],
        prev_action: int = -1,
        detailed: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        """Choose the first action of a bounded, collision-checked beam."""
        config = self.config
        one_step = self._score_state_one_step if detailed else self._score_state_one_step_fast
        if int(config.beam_depth) <= 1:
            action_id, debug = one_step(
                position, yaw, velocity, goal, prev_action, evaluate_successors=True
            )
            if not detailed:
                return action_id, {"dead_end_now": bool(debug.get("dead_end_now", False))}
            debug = self._public_debug(debug)
            debug.update(
                {
                    "teacher_planning_contract_id": TEACHER_PLANNING_CONTRACT_ID,
                    "beam_depth": 1,
                    "beam_expanded_nodes": 1,
                    "beam_best_sequence": str(action_id) if action_id >= 0 else "",
                }
            )
            return action_id, debug

        root_position = np.asarray(position, dtype=np.float32).reshape(3)
        root_velocity = np.asarray(velocity, dtype=np.float32).reshape(3)
        goal_array = np.asarray(goal, dtype=np.float32).reshape(3)
        greedy_action, root_debug = one_step(
            root_position,
            float(yaw),
            root_velocity,
            goal_array,
            int(prev_action),
            evaluate_successors=True,
        )
        if bool(root_debug.get("dead_end_now", False)):
            if not detailed:
                return -1, {"dead_end_now": True}
            debug = self._public_debug(root_debug)
            debug.update(
                {
                    "teacher_planning_contract_id": TEACHER_PLANNING_CONTRACT_ID,
                    "beam_depth": int(config.beam_depth),
                    "beam_expanded_nodes": 1,
                    "beam_best_sequence": "",
                }
            )
            return -1, debug

        def fallback(reason: str, expanded_nodes: int) -> Tuple[int, Dict[str, Any]]:
            if not detailed:
                return int(greedy_action), {
                    "dead_end_now": bool(root_debug.get("dead_end_now", False))
                }
            debug = self._public_debug(root_debug)
            debug.update(
                {
                    "teacher_planning_contract_id": TEACHER_PLANNING_CONTRACT_ID,
                    "beam_depth": int(config.beam_depth),
                    "beam_width": int(config.beam_width),
                    "beam_branching": int(config.beam_branching),
                    "beam_expanded_nodes": int(expanded_nodes),
                    "beam_best_sequence": (
                        str(int(greedy_action)) if int(greedy_action) >= 0 else ""
                    ),
                    "beam_fallback": True,
                    "beam_fallback_reason": reason,
                }
            )
            return int(greedy_action), debug

        # A custom/legacy fast one-step scorer may intentionally expose only
        # the public action diagnostics. Beam expansion needs the private
        # arrays below; without them, the only safe behavior is the already
        # computed greedy action. The production fast scorer supplies all of
        # these arrays, so this is a compatibility/fail-safe seam rather than
        # a second selection algorithm.
        required_beam_debug = (
            "teacher_scores",
            "_evaluated_action_mask",
            "_next_valid_count_all",
            "_goal_reached_mask",
        )
        if any(key not in root_debug for key in required_beam_debug):
            return fallback("missing_fast_beam_diagnostics", 1)

        root_scores = np.asarray(root_debug["teacher_scores"], dtype=np.float32)
        root_evaluated = np.asarray(
            root_debug["_evaluated_action_mask"], dtype=np.bool_
        )
        root_next_counts = np.asarray(
            root_debug["_next_valid_count_all"], dtype=np.int32
        )
        root_reaches_goal = np.asarray(
            root_debug["_goal_reached_mask"], dtype=np.bool_
        )
        root_viable = root_evaluated & (
            (root_next_counts >= int(config.min_next_valid_count))
            | root_reaches_goal
        )
        root_ids = np.flatnonzero(np.isfinite(root_scores) & root_viable)
        if root_ids.size == 0:
            return fallback("no_viable_root_candidate", 1)
        root_ids = root_ids[np.argsort(root_scores[root_ids])[::-1]][
            : min(int(config.beam_width), root_ids.size)
        ]
        beam = []
        for action_id in root_ids.tolist():
            next_position, next_yaw, next_velocity = self._predict_action(
                root_position, float(yaw), int(action_id)
            )
            beam.append(
                {
                    "position": next_position,
                    "yaw": next_yaw,
                    "velocity": next_velocity,
                    "previous_action": int(action_id),
                    "first_action": int(action_id),
                    "sequence": (int(action_id),),
                    "score": float(root_scores[int(action_id)]),
                    "terminal": bool(
                        distance_xy(next_position, goal_array) <= float(config.goal_radius)
                        and abs(float(next_position[2] - goal_array[2]))
                        <= float(config.goal_tolerance_z)
                    ),
                }
            )

        def viable_leaf_nodes(nodes):
            nonterminal = [
                index for index, node in enumerate(nodes) if not node["terminal"]
            ]
            if nonterminal:
                counts, clearances = self._evaluate_pose_reachability(
                    np.stack([nodes[index]["position"] for index in nonterminal]),
                    np.asarray(
                        [nodes[index]["yaw"] for index in nonterminal],
                        dtype=np.float64,
                    ),
                )
                for result_index, node_index in enumerate(nonterminal):
                    nodes[node_index]["leaf_next_valid_count"] = int(
                        counts[result_index]
                    )
                    nodes[node_index]["leaf_next_min_clearance"] = float(
                        clearances[result_index]
                    )
            return [
                node
                for node in nodes
                if node["terminal"]
                or int(node.get("leaf_next_valid_count", 0))
                >= int(config.min_next_valid_count)
            ]

        expanded_nodes = 1
        for depth_index in range(1, int(config.beam_depth)):
            expanded = []
            for node in beam:
                if node["terminal"]:
                    expanded.append(node)
                    continue
                _, node_debug = one_step(
                    node["position"],
                    node["yaw"],
                    node["velocity"],
                    goal_array,
                    node["previous_action"],
                    evaluate_successors=False,
                )
                expanded_nodes += 1
                if bool(node_debug.get("dead_end_now", False)):
                    continue
                node_scores = np.asarray(node_debug["teacher_scores"], dtype=np.float32)
                action_ids = np.flatnonzero(np.isfinite(node_scores))
                action_ids = action_ids[np.argsort(node_scores[action_ids])[::-1]][
                    : min(int(config.beam_branching), action_ids.size)
                ]
                for action_id in action_ids.tolist():
                    next_position, next_yaw, next_velocity = self._predict_action(
                        node["position"], node["yaw"], int(action_id)
                    )
                    expanded.append(
                        {
                            "position": next_position,
                            "yaw": next_yaw,
                            "velocity": next_velocity,
                            "previous_action": int(action_id),
                            "first_action": node["first_action"],
                            "sequence": node["sequence"] + (int(action_id),),
                            "score": node["score"]
                            + float(config.beam_discount) ** depth_index
                            * float(node_scores[int(action_id)]),
                            "terminal": bool(
                                distance_xy(next_position, goal_array)
                                <= float(config.goal_radius)
                                and abs(float(next_position[2] - goal_array[2]))
                                <= float(config.goal_tolerance_z)
                            ),
                        }
                    )
            if not expanded:
                return fallback("beam_exhausted_before_horizon", expanded_nodes)
            if depth_index == int(config.beam_depth) - 1:
                expanded = viable_leaf_nodes(expanded)
                if not expanded:
                    return fallback("no_viable_leaf", expanded_nodes)
            beam = sorted(expanded, key=lambda item: item["score"], reverse=True)[
                : int(config.beam_width)
            ]

        best_node = max(beam, key=lambda item: item["score"])
        best_action = int(best_node["first_action"])
        if not detailed:
            return best_action, {"dead_end_now": False}
        beam_scores = np.full_like(root_scores, -np.inf)
        for node in beam:
            action_id = int(node["first_action"])
            beam_scores[action_id] = max(beam_scores[action_id], float(node["score"]))
        best_score = float(beam_scores[best_action])
        support = np.isfinite(beam_scores)
        if float(config.soft_support_margin) >= 0.0:
            support &= beam_scores >= best_score - float(config.soft_support_margin)
        support[best_action] = True
        soft_target = self._softmax(beam_scores, support, config.temperature)
        probabilities = soft_target[soft_target > 0.0]

        chosen_metrics = (
            int(root_debug["_next_valid_count_all"][best_action]),
            float(root_debug["_next_clearance_all"][best_action]),
        )
        debug = self._public_debug(root_debug)
        debug.update(
            {
                "chosen_score": best_score,
                "chosen_progress": float(root_debug["_progress_all"][best_action]),
                "chosen_dist_after_pred": float(
                    root_debug["_goal_distance_all"][best_action]
                ),
                "chosen_guidance_dist_after_pred": float(
                    root_debug["_guidance_distance_all"][best_action]
                ),
                "chosen_next_valid_count": int(chosen_metrics[0]),
                "chosen_next_min_clearance": float(chosen_metrics[1]),
                "chosen_next_dead_end": bool(
                    int(chosen_metrics[0]) < int(config.min_next_valid_count)
                ),
                "chosen_velocity_mismatch_mps": float(
                    root_debug["_velocity_mismatch_all"][best_action]
                ),
                "teacher_scores": beam_scores,
                "teacher_soft_target": soft_target,
                "teacher_entropy": float(
                    -np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-12)))
                ),
                "soft_support_count": int(np.count_nonzero(support)),
                "best_top_actions": ";".join(
                    str(int(action_id))
                    for action_id in np.argsort(beam_scores)[::-1]
                    if np.isfinite(beam_scores[action_id])
                ),
                "teacher_planning_contract_id": TEACHER_PLANNING_CONTRACT_ID,
                "beam_depth": int(config.beam_depth),
                "beam_width": int(config.beam_width),
                "beam_branching": int(config.beam_branching),
                "beam_expanded_nodes": int(expanded_nodes),
                "beam_best_sequence": ";".join(
                    str(action_id) for action_id in best_node["sequence"]
                ),
                "beam_best_cumulative_score": float(best_node["score"]),
                "beam_reaches_goal": bool(best_node["terminal"]),
                "beam_leaf_next_valid_count": int(
                    best_node.get("leaf_next_valid_count", 0)
                ),
                "beam_fallback": False,
                "beam_fallback_reason": "",
            }
        )
        return best_action, debug

    def score_state(
        self,
        position: Sequence[float],
        yaw: float,
        velocity: Sequence[float],
        goal: Sequence[float],
        prev_action: int = -1,
    ) -> Tuple[int, Dict[str, Any]]:
        """Choose an action and return the full diagnostic contract."""
        return self._score_state_impl(
            position, yaw, velocity, goal, prev_action, detailed=True
        )

    def score_state_fast(
        self,
        position: Sequence[float],
        yaw: float,
        velocity: Sequence[float],
        goal: Sequence[float],
        prev_action: int = -1,
    ) -> Tuple[int, Dict[str, Any]]:
        """Choose an action without materializing unused label diagnostics."""
        return self._score_state_impl(
            position, yaw, velocity, goal, prev_action, detailed=False
        )

    def plan_from_state(
        self,
        position: Sequence[float],
        yaw: float,
        velocity: Sequence[float],
        goal: Sequence[float],
        prev_action: int,
        horizon: int,
        prefetch_teacher: Optional["ReachabilityTeacher"] = None,
    ) -> list[Dict[str, Any]]:
        current_position = np.asarray(position, dtype=np.float32).reshape(3)
        current_yaw = float(yaw)
        current_velocity = np.asarray(velocity, dtype=np.float32).reshape(3)
        goal_array = np.asarray(goal, dtype=np.float32).reshape(3)
        current_previous = int(prev_action)
        plan = []
        for step_index in range(max(1, int(horizon))):
            scorer = (
                self
                if step_index == 0 or prefetch_teacher is None
                else prefetch_teacher
            )
            action_id, debug = scorer.score_state(
                current_position,
                current_yaw,
                current_velocity,
                goal_array,
                current_previous,
            )
            if action_id < 0 or bool(debug.get("dead_end_now", False)):
                break
            next_position = current_position + rotation_z(current_yaw).dot(
                self.motion_primitives.endpoint(action_id)
            )
            next_yaw = self.terminal_yaw(action_id, current_yaw)
            next_velocity = self.command_velocity_world(
                action_id, current_yaw, terminal=True
            )
            plan.append(
                {
                    "action_id": int(action_id),
                    "teacher": debug,
                    "predicted_position": next_position,
                    "predicted_yaw": float(next_yaw),
                    "predicted_velocity": next_velocity,
                }
            )
            current_position = next_position
            current_yaw = next_yaw
            current_velocity = next_velocity
            current_previous = int(action_id)
        return plan

    def plan_horizon(
        self,
        observation: Dict[str, Any],
        prev_action: int,
        horizon: int,
        prefetch_teacher: Optional["ReachabilityTeacher"] = None,
    ) -> list[Dict[str, Any]]:
        return self.plan_from_state(
            position=observation["state"]["position"],
            yaw=float(observation["state"]["yaw"]),
            velocity=observation["state"].get(
                "velocity", np.zeros(3, dtype=np.float32)
            ),
            goal=observation["goal"]["position"],
            prev_action=int(prev_action),
            horizon=int(horizon),
            prefetch_teacher=prefetch_teacher,
        )
