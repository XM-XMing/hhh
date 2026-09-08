"""Importable multiprocessing workers for ideal teacher mission auditing."""

from __future__ import annotations
import math
import os
from pathlib import Path
import numpy as np

from planning.mission.spec import goal_reached
from planning.mission.global_route import GlobalRouteConfig
from planning.native.geometry import NativeGeometryContext
from planning.data.mission_routes import MissionRouteStore
from planning.primitives.library import MotionPrimitiveLibrary, rotation_z
from planning.teacher.policy import ReachabilityTeacher, TeacherConfig
from planning.contracts.teacher_path import (
    TEACHER_PATH_LENGTH_CONTRACT_ID,
    TEACHER_PLAN_PATH_MAX_M,
    minimum_remaining_path_to_goal_region_m,
    plan_path_eligible,
    primitive_reference_path_lengths_m,
)

_AUDIT_TEACHER = None
_AUDIT_MPL = None
_AUDIT_MAX_STEPS = 0
_AUDIT_ACTION_PATH_LENGTHS_M = None
_AUDIT_GEOMETRY_CONTEXT = None
_AUDIT_ROUTE_STORE = None

def _create_audit_state(payload):
    global _AUDIT_GEOMETRY_CONTEXT, _AUDIT_ROUTE_STORE
    backend = str(payload.get("collision_backend", "")).strip()
    if backend:
        os.environ["PLANNING_COLLISION_BACKEND"] = backend
        os.environ["PLANNING_COLLISION_THREADS"] = str(
            int(payload.get("collision_threads", 1))
        )
    if _AUDIT_GEOMETRY_CONTEXT is not None:
        _AUDIT_GEOMETRY_CONTEXT.close()
    if _AUDIT_ROUTE_STORE is not None:
        _AUDIT_ROUTE_STORE.close()
    _AUDIT_ROUTE_STORE = MissionRouteStore.open(
        Path(payload["route_store_prefix"]), validate=False
    )
    teacher_config = TeacherConfig(**payload["teacher_config"])
    route_config = GlobalRouteConfig(
        resolution_m=float(teacher_config.global_route_resolution_m),
        flight_z_min_m=float(teacher_config.z_min),
        flight_z_max_m=float(teacher_config.z_max),
        lookahead_m=float(teacher_config.global_route_lookahead_m),
        tracking_margin_m=float(teacher_config.global_route_tracking_margin_m),
    )
    _AUDIT_GEOMETRY_CONTEXT = NativeGeometryContext.from_voxel_cache(
        Path(payload["cache_path"]),
        voxel_size=float(payload["voxel_size"]),
        route_config=route_config,
    )
    checker = _AUDIT_GEOMETRY_CONTEXT.collision_checker(float(payload["inflate_radius"]))
    mpl = MotionPrimitiveLibrary()
    teacher = ReachabilityTeacher(mpl, checker, teacher_config)
    return (
        teacher,
        mpl,
        int(payload["max_steps"]),
        primitive_reference_path_lengths_m(mpl),
    )

def init_audit_worker(payload):
    global _AUDIT_TEACHER, _AUDIT_MPL, _AUDIT_MAX_STEPS, _AUDIT_ACTION_PATH_LENGTHS_M
    (
        _AUDIT_TEACHER,
        _AUDIT_MPL,
        _AUDIT_MAX_STEPS,
        _AUDIT_ACTION_PATH_LENGTHS_M,
    ) = _create_audit_state(payload)

def _audit_mission_row_with_state(
    task, teacher, mpl, max_steps, action_path_lengths_m, route_store
):
    candidate_number, row = task
    if teacher is None or mpl is None or action_path_lengths_m is None:
        raise RuntimeError("teacher audit state is not initialized")
    position = np.asarray(
        [float(row[key]) for key in ("start_x", "start_y", "start_z")],
        dtype=np.float32,
    )
    goal = np.asarray(
        [float(row[key]) for key in ("goal_x", "goal_y", "goal_z")],
        dtype=np.float32,
    )
    yaw = math.radians(float(row.get("start_yaw_deg", 0.0)))
    velocity = np.zeros(3, dtype=np.float32)
    previous_action = -1
    start = position.copy()
    executed_path_length = 0.0
    action_sequence = []
    if route_store is None:
        raise RuntimeError("mission route store is not initialized")
    try:
        route_index = int(float(row["global_route_index"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("mission row is missing a valid global_route_index") from error
    route_info = teacher.set_mission_route(
        route_store.validate(route_index, goal=goal), goal
    )
    reason = "max_steps"
    steps = 0
    remaining_path_lower_bound = minimum_remaining_path_to_goal_region_m(
        position,
        goal,
        teacher.config.goal_radius,
        teacher.config.goal_tolerance_z,
    )
    if not plan_path_eligible(executed_path_length + remaining_path_lower_bound):
        reason = "path_too_long"
    for steps in range(1, max_steps + 1):
        if reason == "path_too_long":
            break
        action_id, debug = teacher.score_state_fast(
            position, yaw, velocity, goal, previous_action
        )
        if action_id < 0 or bool(debug.get("dead_end_now", False)):
            reason = "dead_end"
            break
        old_yaw = yaw
        displacement = rotation_z(old_yaw).dot(mpl.endpoint(action_id))
        position = position + displacement
        executed_path_length += float(action_path_lengths_m[action_id])
        yaw = teacher.terminal_yaw(action_id, old_yaw)
        velocity = teacher.command_velocity_world(
            action_id, old_yaw, terminal=True
        )
        previous_action = int(action_id)
        action_sequence.append(int(action_id))
        if goal_reached(position, goal):
            reason = "success"
            break
        remaining_path_lower_bound = minimum_remaining_path_to_goal_region_m(
            position,
            goal,
            teacher.config.goal_radius,
            teacher.config.goal_tolerance_z,
        )
        if not plan_path_eligible(
            executed_path_length + remaining_path_lower_bound
        ):
            reason = "path_too_long"
            break
    if reason == "success" and not plan_path_eligible(executed_path_length):
        reason = "path_too_long"
    straight_line_distance = float(np.linalg.norm(goal - start))
    return {
        "candidate_number": int(candidate_number),
        "mission_id": row.get("mission_id", ""),
        "global_route_index": int(route_index),
        "result": reason,
        "steps": steps,
        "action_sequence": ";".join(str(value) for value in action_sequence),
        "final_distance_xy_m": float(np.linalg.norm(position[:2] - goal[:2])),
        "final_error_z_m": abs(float(position[2] - goal[2])),
        "route_length_m": float(route_info["route_length_m"]),
        "executed_path_length_m": executed_path_length,
        "remaining_path_lower_bound_m": remaining_path_lower_bound,
        "straight_line_distance_m": straight_line_distance,
        "path_stretch": executed_path_length / max(1.0e-6, straight_line_distance),
        "path_length_contract_id": TEACHER_PATH_LENGTH_CONTRACT_ID,
        "plan_path_max_m": TEACHER_PLAN_PATH_MAX_M,
    }

def audit_mission_row(task):
    if (
        _AUDIT_TEACHER is None
        or _AUDIT_MPL is None
        or _AUDIT_ACTION_PATH_LENGTHS_M is None
    ):
        raise RuntimeError("teacher audit worker is not initialized")
    return _audit_mission_row_with_state(
        task,
        _AUDIT_TEACHER,
        _AUDIT_MPL,
        _AUDIT_MAX_STEPS,
        _AUDIT_ACTION_PATH_LENGTHS_M,
        _AUDIT_ROUTE_STORE,
    )
