"""Shared mission contract for the 40 m online-teacher pipeline."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

from planning.common import canonical_json_sha256
from planning.contracts.task import (
    ACTION_MASK_Z_MARGIN_M,
    DEFAULT_MAX_PRIMITIVE_STEPS,
    FLIGHT_Z_MAX_M,
    FLIGHT_Z_MIN_M,
    GOAL_RADIUS_XY_M,
    GOAL_TOLERANCE_Z_M,
    MISSION_PLANAR_DISTANCE_M,
    REWARD_CONTRACT_ID,
    TASK_CONTRACT_ID,
    legacy_task_contract_v1_metadata,
    legacy_task_contract_v1_sha256,
    task_contract_fields,
    task_contract_metadata,
    task_contract_sha256,
    validate_legacy_task_contract_v1,
    validate_task_contract,
)

DEFAULT_ALTITUDE_LEVELS_M = (1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8)

def goal_errors(position: Sequence[float], goal: Sequence[float]) -> Tuple[float, float]:
    """Return planar distance and absolute altitude error for a 3D goal."""
    px, py, pz = [float(value) for value in position[:3]]
    gx, gy, gz = [float(value) for value in goal[:3]]
    return planar_distance_xy(px, py, gx, gy), abs(gz - pz)

def goal_reached(
    position: Sequence[float],
    goal: Sequence[float],
    radius_xy: float = GOAL_RADIUS_XY_M,
    tolerance_z: float = GOAL_TOLERANCE_Z_M,
) -> bool:
    distance_xy, error_z = goal_errors(position, goal)
    return bool(distance_xy <= float(radius_xy) and error_z <= float(tolerance_z))

def planar_distance_xy(start_x: float, start_y: float, goal_x: float, goal_y: float) -> float:
    return math.hypot(float(goal_x) - float(start_x), float(goal_y) - float(start_y))

def mission_id_from_values(
    start_x: float,
    start_y: float,
    start_z: float,
    start_yaw_deg: float,
    goal_x: float,
    goal_y: float,
    goal_z: float,
    map_id: str = "forest",
) -> str:
    """Return a stable identity for grouping rollout data by mission."""
    payload = {
        "map_id": str(map_id),
        "start": [round(float(start_x), 6), round(float(start_y), 6), round(float(start_z), 6), round(float(start_yaw_deg), 6)],
        "goal": [round(float(goal_x), 6), round(float(goal_y), 6), round(float(goal_z), 6)],
    }
    return canonical_json_sha256(payload)[:20]

def mission_id_from_row(row: Dict, map_id: str = "forest") -> str:
    existing = str(row.get("mission_id", "")).strip()
    if existing:
        return existing
    return mission_id_from_values(
        row["start_x"], row["start_y"], row["start_z"], row.get("start_yaw_deg", 0.0),
        row["goal_x"], row["goal_y"], row["goal_z"], map_id=map_id,
    )

def validate_mission_rows(
    rows: Iterable[Dict],
    expected_planar_distance: float = MISSION_PLANAR_DISTANCE_M,
    planar_distance_tolerance: float = 1.0e-3,
    z_min: float = FLIGHT_Z_MIN_M,
    z_max: float = FLIGHT_Z_MAX_M,
    *,
    expected_max_primitive_steps: Optional[int] = None,
    historical_v1: bool = False,
) -> None:
    """Reject rows outside the selected mission contract.

    Formal callers use V2 by default.  V1 is available only through the
    explicit historical seam so old RL fixtures cannot silently become new
    Pre-BC inputs.
    """
    violations = []
    numeric_fields = (
        "start_x",
        "start_y",
        "start_z",
        "goal_x",
        "goal_y",
        "goal_z",
    )
    required = numeric_fields
    for row in rows:
        episode_id = row.get("episode_id", "?")
        missing = [key for key in required if key not in row or str(row.get(key, "")).strip() == ""]
        if missing:
            violations.append("episode {} missing {}".format(episode_id, ",".join(missing)))
            continue
        try:
            if historical_v1:
                validate_legacy_task_contract_v1(row, path="episode {}".format(episode_id))
            else:
                validate_task_contract(
                    row,
                    expected_max_primitive_steps=expected_max_primitive_steps,
                    path="episode {}".format(episode_id),
                )
        except ValueError as error:
            violations.append(str(error))
            continue
        try:
            values = [float(row[key]) for key in numeric_fields]
        except (TypeError, ValueError):
            violations.append("episode {} has non-numeric start/goal".format(episode_id))
            continue
        if not all(math.isfinite(value) for value in values):
            violations.append("episode {} has non-finite start/goal".format(episode_id))
            continue
        start_x, start_y, start_z, goal_x, goal_y, goal_z = values
        if not (float(z_min) <= start_z <= float(z_max) and float(z_min) <= goal_z <= float(z_max)):
            violations.append(
                "episode {} altitude outside [{:.3f}, {:.3f}]".format(episode_id, z_min, z_max)
            )
            continue
        if float(expected_planar_distance) > 0.0:
            distance = planar_distance_xy(start_x, start_y, goal_x, goal_y)
            if abs(distance - float(expected_planar_distance)) > float(planar_distance_tolerance):
                violations.append(
                    "episode {} planar distance {:.6f} != {:.6f} +/- {:.6f}".format(
                        episode_id, distance, expected_planar_distance, planar_distance_tolerance
                    )
                )
    if violations:
        preview = "; ".join(violations[:5])
        extra = "" if len(violations) <= 5 else "; and {} more".format(len(violations) - 5)
        raise ValueError("index violates mission contract: {}{}".format(preview, extra))

def row_start_goal(row: Dict) -> tuple[list[float], list[float]]:
    return (
        [float(row["start_x"]), float(row["start_y"]), float(row["start_z"])],
        [float(row["goal_x"]), float(row["goal_y"]), float(row["goal_z"])],
    )
