"""Policy feature schema shared by collection, training, and evaluation.

The feature contract replaces raw yaw with sin/cos and excludes simulator-only collision
metadata (min clearance, collision flags, altitude-violation flags) from policy
inputs. Continuous features are normalized from the training split; the
previous-action one-hot vector is left unchanged.
"""

from __future__ import annotations

from typing import Dict
import numpy as np

from planning.common import canonical_json_sha256

FEATURE_CONTRACT_ID = "depth_goal_state_prev_action"
NUM_ACTIONS = 105
INITIAL_PREV_ACTION = -1

# This is the deployable observation boundary.  Privileged route/map values
# may supervise representation learning offline, but must never be appended to
# the vector consumed by learned policies and Unity evaluation.
POLICY_OBSERVATION_FIELDS = (
    "depth",
    "state",
    "final_goal",
    "previous_action",
)
FORBIDDEN_PRIVILEGED_RUNTIME_FIELDS = (
    "global_map",
    "global_route",
    "route_waypoint",
    "collision_voxel_map",
    "teacher_costs",
    "future_expert_path",
)

STATE_FEATURE_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "yaw_sin",
    "yaw_cos",
    "z_to_min",
    "z_to_max",
    "goal_distance_xy",
    "goal_dz",
)

GOAL_FEATURE_NAMES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "direction_xy_x",
    "direction_xy_y",
    "direction_body_xy_x",
    "direction_body_xy_y",
    "distance_xy",
    "distance_xy_norm40",
    "dz",
)

STATE_DIM = len(STATE_FEATURE_NAMES)
GOAL_DIM = len(GOAL_FEATURE_NAMES)
CONTINUOUS_DIM = STATE_DIM + GOAL_DIM
POLICY_VECTOR_DIM = CONTINUOUS_DIM + NUM_ACTIONS

def policy_input_contract() -> dict:
    return {
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "observation_fields": list(POLICY_OBSERVATION_FIELDS),
        "forbidden_privileged_runtime_fields": list(FORBIDDEN_PRIVILEGED_RUNTIME_FIELDS),
        "continuous_feature_names": list(STATE_FEATURE_NAMES + GOAL_FEATURE_NAMES),
        "previous_action_encoding": "onehot_{}".format(NUM_ACTIONS),
        "vector_dim": POLICY_VECTOR_DIM,
    }

def policy_input_contract_sha256() -> str:
    return canonical_json_sha256(policy_input_contract())

def validate_checkpoint_policy_input_contract(checkpoint: dict) -> None:
    """Reject a changed observation schema or privileged runtime dependency."""
    declared_hash = str(checkpoint.get("policy_input_contract_sha256", ""))
    if declared_hash and declared_hash != policy_input_contract_sha256():
        raise ValueError("checkpoint policy input contract hash mismatch")
    privileged = list(checkpoint.get("privileged_runtime_inputs", []))
    if privileged:
        raise ValueError(
            "checkpoint requires forbidden privileged runtime inputs: {}".format(privileged)
        )

def action_onehot(action_id: int, num_actions: int = NUM_ACTIONS) -> np.ndarray:
    output = np.zeros((int(num_actions),), dtype=np.float32)
    if 0 <= int(action_id) < int(num_actions):
        output[int(action_id)] = 1.0
    return output

def actions_onehot(action_ids: np.ndarray, num_actions: int = NUM_ACTIONS) -> np.ndarray:
    ids = np.asarray(action_ids, dtype=np.int64).reshape(-1)
    output = np.zeros((ids.shape[0], int(num_actions)), dtype=np.float32)
    valid = (ids >= 0) & (ids < int(num_actions))
    output[np.arange(ids.shape[0])[valid], ids[valid]] = 1.0
    return output

def obs_state_vector(obs: Dict) -> np.ndarray:
    state = obs["state"]
    goal = obs["goal"]
    yaw = float(state["yaw"])
    return np.asarray(
        [
            *np.asarray(state["position"], dtype=np.float32).reshape(3),
            *np.asarray(state["velocity"], dtype=np.float32).reshape(3),
            np.sin(yaw),
            np.cos(yaw),
            float(state["z_to_min"]),
            float(state["z_to_max"]),
            float(goal["distance_xy"]),
            float(goal["dz"]),
        ],
        dtype=np.float32,
    )

def obs_goal_vector(obs: Dict) -> np.ndarray:
    goal = obs["goal"]
    return np.asarray(
        [
            *np.asarray(goal["relative"], dtype=np.float32).reshape(3),
            *np.asarray(goal["direction_xy"], dtype=np.float32).reshape(2),
            *np.asarray(goal["direction_body_xy"], dtype=np.float32).reshape(2),
            float(goal["distance_xy"]),
            float(goal["distance_xy_norm40"]),
            float(goal["dz"]),
        ],
        dtype=np.float32,
    )

def obs_continuous_vector(obs: Dict) -> np.ndarray:
    return np.concatenate([obs_state_vector(obs), obs_goal_vector(obs)], axis=0).astype(np.float32)

def validate_feature_arrays(states: np.ndarray, goals: np.ndarray) -> None:
    states = np.asarray(states)
    goals = np.asarray(goals)
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError("states must be [T,{}], got {}".format(STATE_DIM, states.shape))
    if goals.ndim != 2 or goals.shape[1] != GOAL_DIM:
        raise ValueError("goals must be [T,{}], got {}".format(GOAL_DIM, goals.shape))
    if states.shape[0] != goals.shape[0]:
        raise ValueError("state/goal transition count mismatch")
    if not np.isfinite(states).all() or not np.isfinite(goals).all():
        raise ValueError("non-finite policy features")
