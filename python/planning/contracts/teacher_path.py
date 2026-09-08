"""Path-length quality contract for expert demonstrations."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


TEACHER_PATH_LENGTH_CONTRACT_ID = "teacher_plan44_actual46_observed_polyline"
TEACHER_PLAN_PATH_MAX_M = 44.0
TEACHER_ACTUAL_PATH_MAX_M = 46.0
PATH_LENGTH_EPSILON_M = 1.0e-6


def observed_polyline_length_m(positions: Any) -> float:
    """Return distance along ordered 3D position samples."""
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 2:
        return 0.0
    if not np.isfinite(points).all():
        raise ValueError("path positions must be finite")
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def primitive_reference_path_lengths_m(motion_primitives: Any) -> np.ndarray:
    """Return arc length of every reference primitive, including the origin."""
    references = np.asarray(motion_primitives.pos_ref, dtype=np.float64)
    if references.ndim != 3 or references.shape[2] != 3:
        raise ValueError("motion primitive pos_ref must have shape [actions, frames, 3]")
    origins = np.zeros((references.shape[0], 1, 3), dtype=np.float64)
    points = np.concatenate((origins, references), axis=1)
    return np.linalg.norm(np.diff(points, axis=1), axis=2).sum(axis=1)


def plan_path_eligible(path_length_m: float) -> bool:
    return float(path_length_m) <= TEACHER_PLAN_PATH_MAX_M + PATH_LENGTH_EPSILON_M


def minimum_remaining_path_to_goal_region_m(
    position: Any,
    goal: Any,
    goal_radius_m: float,
    goal_z_tolerance_m: float,
) -> float:
    """Return an exact Euclidean lower bound to the cylindrical goal region."""
    current = np.asarray(position, dtype=np.float64).reshape(3)
    target = np.asarray(goal, dtype=np.float64).reshape(3)
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("position and goal must be finite")
    if float(goal_radius_m) < 0.0 or float(goal_z_tolerance_m) < 0.0:
        raise ValueError("goal tolerances must be non-negative")
    remaining_xy = max(
        0.0,
        float(np.linalg.norm(current[:2] - target[:2])) - float(goal_radius_m),
    )
    remaining_z = max(
        0.0,
        abs(float(current[2] - target[2])) - float(goal_z_tolerance_m),
    )
    return float(np.hypot(remaining_xy, remaining_z))


def actual_path_eligible(path_length_m: float) -> bool:
    return float(path_length_m) <= TEACHER_ACTUAL_PATH_MAX_M + PATH_LENGTH_EPSILON_M


def validate_plan_filtered_mission_rows(rows: Sequence[dict]) -> None:
    """Require every collection mission to carry Plan<=44m audit evidence."""
    errors = []
    for index, row in enumerate(rows):
        contract_id = str(row.get("teacher_path_length_contract_id", ""))
        try:
            path_length = float(row.get("teacher_plan_path_length_m", "nan"))
        except (TypeError, ValueError):
            path_length = float("nan")
        if contract_id != TEACHER_PATH_LENGTH_CONTRACT_ID:
            errors.append("row {} path contract {}".format(index, contract_id or "<missing>"))
        elif not np.isfinite(path_length):
            errors.append("row {} missing plan path length".format(index))
        elif not plan_path_eligible(path_length):
            errors.append("row {} plan path length {} exceeds {}".format(
                index, path_length, TEACHER_PLAN_PATH_MAX_M
            ))
        if len(errors) >= 5:
            break
    if errors:
        raise ValueError("mission path-length contract failed: " + "; ".join(errors))
