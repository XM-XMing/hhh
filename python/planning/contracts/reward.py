"""Canonical numeric reward contract for the Unity planning environment.

The function below is deliberately pure.  ``UnityForestEnv`` remains the
runtime owner of terminal classification and state history; this module owns
only the arithmetic and immutable defaults used for a normal transition.
"""

from __future__ import annotations

from typing import Dict

from planning.common import canonical_json_sha256


REWARD_CONTRACT_SCHEMA_VERSION = 1
REWARD_CONTRACT_ID = "terminal_failure_dominates_progress"
TERMINAL_FAILURE_PENALTY = -100.0

_DEFAULTS = {
    "reward_progress_scale": 2.0,
    "reward_goal_z_progress_scale": 1.0,
    "reward_step": -0.02,
    "reward_clearance_scale": -1.0,
    "reward_action_change": -0.02,
    "reward_success": 30.0,
    "reward_collision": TERMINAL_FAILURE_PENALTY,
    "reward_altitude_violation": TERMINAL_FAILURE_PENALTY,
    "reward_timeout": TERMINAL_FAILURE_PENALTY,
    "reward_far": TERMINAL_FAILURE_PENALTY,
    "reward_dead_end": TERMINAL_FAILURE_PENALTY,
}


def reward_contract() -> Dict:
    return {
        "schema_version": REWARD_CONTRACT_SCHEMA_VERSION,
        "contract_id": REWARD_CONTRACT_ID,
        "formula": (
            "progress_scale*progress + goal_z_progress_scale*z_progress + step "
            "+ clearance_scale*max(0, clearance_margin-min_clearance) "
            "+ action_change*action_changed + terminal_bonus_sum"
        ),
        "defaults": dict(_DEFAULTS),
        "terminal_precedence": "success_requires_no_collision_or_altitude_violation",
    }


def reward_contract_sha256() -> str:
    return canonical_json_sha256(reward_contract())


def compute_reward(
    *,
    progress: float,
    z_progress: float,
    min_clearance: float,
    clearance_margin_m: float,
    action_changed: bool,
    success: bool,
    collided: bool,
    altitude_violation: bool,
    timeout: bool,
    far: bool,
    dead_end: bool,
    reward_progress_scale: float = _DEFAULTS["reward_progress_scale"],
    reward_goal_z_progress_scale: float = _DEFAULTS["reward_goal_z_progress_scale"],
    reward_step: float = _DEFAULTS["reward_step"],
    reward_clearance_scale: float = _DEFAULTS["reward_clearance_scale"],
    reward_action_change: float = _DEFAULTS["reward_action_change"],
    reward_success: float = _DEFAULTS["reward_success"],
    reward_collision: float = _DEFAULTS["reward_collision"],
    reward_altitude_violation: float = _DEFAULTS["reward_altitude_violation"],
    reward_timeout: float = _DEFAULTS["reward_timeout"],
    reward_far: float = _DEFAULTS["reward_far"],
    reward_dead_end: float = _DEFAULTS["reward_dead_end"],
) -> float:
    """Compute the exact existing scalar reward without changing precedence."""
    clearance_penalty = max(0.0, float(clearance_margin_m) - float(min_clearance))
    reward = 0.0
    reward += float(reward_progress_scale) * float(progress)
    reward += float(reward_goal_z_progress_scale) * float(z_progress)
    reward += float(reward_step)
    reward += float(reward_clearance_scale) * clearance_penalty
    reward += float(reward_action_change) * float(bool(action_changed))
    if bool(success):
        reward += float(reward_success)
    if bool(collided):
        reward += float(reward_collision)
    if bool(altitude_violation):
        reward += float(reward_altitude_violation)
    if bool(timeout):
        reward += float(reward_timeout)
    if bool(far):
        reward += float(reward_far)
    if bool(dead_end):
        reward += float(reward_dead_end)
    return float(reward)


__all__ = [
    "REWARD_CONTRACT_ID",
    "REWARD_CONTRACT_SCHEMA_VERSION",
    "TERMINAL_FAILURE_PENALTY",
    "compute_reward",
    "reward_contract",
    "reward_contract_sha256",
]

