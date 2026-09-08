"""Versioned task-contract identities for historical and formal artifacts.

The V1 contract is an immutable compatibility seam for historical artifacts.
New Pre-BC artifacts use V2, whose canonical metadata includes the resolved
primitive horizon.  The two identities are deliberately explicit so a
historical artifact cannot silently select the formal contract.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from planning.common import canonical_json_sha256
from planning.common.config import pre_bc_value
from planning.contracts.reward import REWARD_CONTRACT_ID


MISSION_PLANAR_DISTANCE_M = 40.0
FLIGHT_Z_MIN_M = 1.0
FLIGHT_Z_MAX_M = 3.0
GOAL_RADIUS_XY_M = 1.2
GOAL_TOLERANCE_Z_M = 0.20
ACTION_MASK_Z_MARGIN_M = 0.02
TASK_CONTRACT_ID = "xm_3d_flight_z1_3"

LEGACY_TASK_CONTRACT_SCHEMA_VERSION = 1
TASK_CONTRACT_SCHEMA_VERSION = 2


def _configured_max_primitive_steps() -> int:
    value = pre_bc_value("mission", "max_steps")
    if isinstance(value, bool):
        raise ValueError("mission.max_steps must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("mission.max_steps must be an integer") from error
    if parsed <= 0 or parsed != float(value):
        raise ValueError("mission.max_steps must be a positive integer")
    return parsed


DEFAULT_MAX_PRIMITIVE_STEPS = _configured_max_primitive_steps()


def legacy_task_contract_v1_metadata() -> Dict[str, Any]:
    """Return the frozen V1 metadata, without adding V2 fields."""

    return {
        "task_contract_id": TASK_CONTRACT_ID,
        "mission_planar_distance_m": MISSION_PLANAR_DISTANCE_M,
        "flight_z_min_m": FLIGHT_Z_MIN_M,
        "flight_z_max_m": FLIGHT_Z_MAX_M,
        "goal_radius_xy_m": GOAL_RADIUS_XY_M,
        "goal_tolerance_z_m": GOAL_TOLERANCE_Z_M,
        "action_mask_z_margin_m": ACTION_MASK_Z_MARGIN_M,
        "reward_contract_id": REWARD_CONTRACT_ID,
        "success_rule": "distance_xy<=goal_radius_xy_m and abs(goal_z-current_z)<=goal_tolerance_z_m",
        "altitude_rule": "terminate when current_z is outside [flight_z_min_m,flight_z_max_m]",
    }


def legacy_task_contract_v1_sha256() -> str:
    """Return the immutable SHA256 of the historical V1 metadata bytes."""

    return canonical_json_sha256(legacy_task_contract_v1_metadata())


def _resolve_max_primitive_steps(value: Optional[int]) -> int:
    if value is None:
        return DEFAULT_MAX_PRIMITIVE_STEPS
    if isinstance(value, bool):
        raise ValueError("max_primitive_steps must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("max_primitive_steps must be a positive integer") from error
    if parsed <= 0 or parsed != float(value):
        raise ValueError("max_primitive_steps must be a positive integer")
    return parsed


def task_contract_metadata(max_primitive_steps: Optional[int] = None) -> Dict[str, Any]:
    """Return formal V2 metadata for the resolved primitive horizon."""

    metadata = legacy_task_contract_v1_metadata()
    metadata.update(
        {
            "task_contract_schema_version": TASK_CONTRACT_SCHEMA_VERSION,
            "max_primitive_steps": _resolve_max_primitive_steps(max_primitive_steps),
        }
    )
    return metadata


def task_contract_sha256(max_primitive_steps: Optional[int] = None) -> str:
    """Return the deterministic formal V2 task-contract SHA256."""

    return canonical_json_sha256(task_contract_metadata(max_primitive_steps))


def task_contract_fields(max_primitive_steps: Optional[int] = None) -> Dict[str, Any]:
    """Return the minimum V2 identity fields embedded in every new artifact."""

    resolved = _resolve_max_primitive_steps(max_primitive_steps)
    return {
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_schema_version": TASK_CONTRACT_SCHEMA_VERSION,
        "task_contract_sha256": task_contract_sha256(resolved),
        "max_primitive_steps": resolved,
        "task_contract_mode": "formal_v2",
    }


def _required_integer(metadata: Mapping[str, Any], field: str, path: str) -> int:
    if field not in metadata or str(metadata.get(field, "")).strip() == "":
        raise ValueError("{} missing {}".format(path, field))
    value = metadata[field]
    if isinstance(value, bool):
        raise ValueError("{} {} must be an integer".format(path, field))
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{} {} must be an integer".format(path, field)) from error
    try:
        integral = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{} {} must be an integer".format(path, field)) from error
    if parsed != integral:
        raise ValueError("{} {} must be an integer".format(path, field))
    return parsed


def validate_task_contract(
    metadata: Mapping[str, Any],
    *,
    expected_max_primitive_steps: Optional[int] = None,
    path: str = "task contract",
) -> Dict[str, Any]:
    """Fail closed unless metadata is a formal V2 task contract."""

    if not isinstance(metadata, Mapping):
        raise ValueError("{} must be a mapping".format(path))
    schema_version = _required_integer(metadata, "task_contract_schema_version", path)
    if schema_version != TASK_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            "{} unknown task contract schema version {}".format(path, schema_version)
        )
    if str(metadata.get("task_contract_id", "")).strip() != TASK_CONTRACT_ID:
        raise ValueError("{} task contract id mismatch".format(path))
    max_steps = _required_integer(metadata, "max_primitive_steps", path)
    if max_steps <= 0:
        raise ValueError("{} max_primitive_steps must be positive".format(path))
    if "max_steps" in metadata:
        declared_max_steps = _required_integer(metadata, "max_steps", path)
        if declared_max_steps != max_steps:
            raise ValueError(
                "{} max_steps {} != max_primitive_steps {}".format(
                    path, declared_max_steps, max_steps
                )
            )
    expected = _resolve_max_primitive_steps(expected_max_primitive_steps)
    if max_steps != expected:
        raise ValueError(
            "{} max_primitive_steps {} != resolved {}".format(
                path, max_steps, expected
            )
        )
    expected_sha = task_contract_sha256(max_steps)
    if str(metadata.get("task_contract_sha256", "")).strip().lower() != expected_sha:
        raise ValueError("{} task contract SHA mismatch".format(path))
    validated = dict(metadata)
    validated["task_contract_mode"] = "formal_v2"
    return validated


def resolve_policy_checkpoint_task_contract(
    metadata: Mapping[str, Any],
    *,
    expected_max_primitive_steps: Optional[int] = None,
    path: str = "policy checkpoint task contract",
) -> Dict[str, Any]:
    """Resolve a policy checkpoint's task identity without weakening validation.

    A small number of historical AWAC checkpoints were written with the
    formal task ID and SHA, but without the top-level V2 schema and horizon
    fields.  Those fields can be reconstructed only after the supplied ID and
    SHA match the canonical contract for the evaluator's resolved horizon.
    Checkpoints with an explicit schema continue through the ordinary strict
    validator unchanged.
    """

    if not isinstance(metadata, Mapping):
        raise ValueError("{} must be a mapping".format(path))
    expected = _resolve_max_primitive_steps(expected_max_primitive_steps)
    candidate = dict(metadata)
    schema_missing = "task_contract_schema_version" not in candidate
    if schema_missing:
        if str(candidate.get("task_contract_id", "")).strip() != TASK_CONTRACT_ID:
            raise ValueError("{} task contract id mismatch".format(path))
        expected_sha = task_contract_sha256(expected)
        if str(candidate.get("task_contract_sha256", "")).strip().lower() != expected_sha:
            raise ValueError("{} task contract SHA mismatch".format(path))
        candidate["task_contract_schema_version"] = TASK_CONTRACT_SCHEMA_VERSION
        candidate.setdefault("max_primitive_steps", expected)
    return validate_task_contract(
        candidate,
        expected_max_primitive_steps=expected,
        path=path,
    )


def validate_legacy_task_contract_v1(
    metadata: Mapping[str, Any], *, path: str = "legacy task contract"
) -> Dict[str, Any]:
    """Validate V1 only when a caller explicitly selects historical mode."""

    if not isinstance(metadata, Mapping):
        raise ValueError("{} must be a mapping".format(path))
    if "task_contract_schema_version" in metadata or "max_primitive_steps" in metadata:
        raise ValueError("{} is not a V1 task contract".format(path))
    if str(metadata.get("task_contract_id", "")).strip() != TASK_CONTRACT_ID:
        raise ValueError("{} task contract id mismatch".format(path))
    if str(metadata.get("task_contract_sha256", "")).strip().lower() != legacy_task_contract_v1_sha256():
        raise ValueError("{} task contract SHA mismatch".format(path))
    validated = dict(metadata)
    validated["task_contract_mode"] = "legacy_v1"
    return validated


__all__ = [
    "ACTION_MASK_Z_MARGIN_M",
    "DEFAULT_MAX_PRIMITIVE_STEPS",
    "FLIGHT_Z_MAX_M",
    "FLIGHT_Z_MIN_M",
    "GOAL_RADIUS_XY_M",
    "GOAL_TOLERANCE_Z_M",
    "LEGACY_TASK_CONTRACT_SCHEMA_VERSION",
    "MISSION_PLANAR_DISTANCE_M",
    "REWARD_CONTRACT_ID",
    "TASK_CONTRACT_ID",
    "TASK_CONTRACT_SCHEMA_VERSION",
    "legacy_task_contract_v1_metadata",
    "legacy_task_contract_v1_sha256",
    "task_contract_fields",
    "task_contract_metadata",
    "task_contract_sha256",
    "resolve_policy_checkpoint_task_contract",
    "validate_legacy_task_contract_v1",
    "validate_task_contract",
]
