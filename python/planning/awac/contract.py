"""Immutable identity for the formal discrete AWAC pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Optional

from planning.common.hashing import canonical_json_sha256
from planning.awac.interaction import behavior_source_contract
from planning.contracts.feature import (
    FEATURE_CONTRACT_ID,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    policy_input_contract_sha256,
)
from planning.contracts.policy_runtime import POLICY_RUNTIME_CONTRACT_ID
from planning.contracts.task import task_contract_sha256
from planning.contracts.reward import REWARD_CONTRACT_ID, reward_contract_sha256
from planning.contracts.task import TASK_CONTRACT_ID
from planning.version import SOFTWARE_VERSION


AWAC_ALGORITHM_ID = "discrete_masked_awac"
AWAC_CHECKPOINT_CONTRACT_ID = "discrete_masked_awac_checkpoint_v2"
AWAC_TRAINING_CONFIG_CONTRACT_ID = "discrete_masked_awac_training_v2"
AWAC_MODEL_TYPE = "awac_discrete_masked"
AWAC_REPLAY_CONTRACT_ID = "discrete_masked_awac_replay_uint8_v1"
AWAC_TRAINING_CONTRACT_SCHEMA_VERSION = 1
AWAC_PHASE_CRITIC_CALIBRATION = "critic_calibration"
AWAC_PHASE_STANDARD_TRAINING = "awac_training"
AWAC_POLICY_INPUT_CONTRACT = "depth_goal_state_prev_action"
AWAC_REWARD_SCALE = 0.10
AWAC_REWARD_SCALE_OWNER = "learner_bellman_target"

REPLAY_FIELDS = (
    "depth",
    "vector",
    "action_mask",
    "action",
    "reward",
    "next_depth",
    "next_vector",
    "next_action_mask",
    "done",
    "behavior_source",
)

TERMINAL_REASONS = (
    "success",
    "collision",
    "dead_end",
    "timeout",
    "hard_altitude",
)


def replay_contract() -> dict:
    """Return the serialized AWAC replay contract without runtime state."""

    return {
        "schema_version": 1,
        "contract_id": AWAC_REPLAY_CONTRACT_ID,
        "fields": list(REPLAY_FIELDS),
        "dtypes": {
            "depth": "uint8",
            "vector": "float32",
            "action_mask": "uint8",
            "action": "int16",
            "reward": "float32",
            "next_depth": "uint8",
            "next_vector": "float32",
            "next_action_mask": "uint8",
            "done": "uint8",
            "behavior_source": "uint8",
        },
        "behavior_sources": behavior_source_contract(),
        "observation_contract": "reliable_exact_endpoint_snapshot",
        "privileged_fields": [],
        "terminal_reasons": list(TERMINAL_REASONS),
        "reward_contract_id": REWARD_CONTRACT_ID,
        "reward_contract_sha256": reward_contract_sha256(),
        "reward_storage_semantics": "raw_environment_reward",
        "reward_scale": AWAC_REWARD_SCALE,
        "reward_scale_owner": AWAC_REWARD_SCALE_OWNER,
    }


def replay_contract_sha256() -> str:
    return canonical_json_sha256(replay_contract())


def build_awac_training_contract(
    *,
    phase: str = AWAC_PHASE_CRITIC_CALIBRATION,
    gamma: float = 0.99,
    tau: float = 0.005,
    max_primitive_steps: int = 45,
    calibration: Optional[Mapping] = None,
) -> dict:
    """Build the immutable Phase-0 AWAC training contract projection."""

    phase = str(phase)
    if phase not in (AWAC_PHASE_CRITIC_CALIBRATION, AWAC_PHASE_STANDARD_TRAINING):
        raise ValueError("unsupported AWAC training phase: {}".format(phase))
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must be in (0,1]")
    if not 0.0 < float(tau) <= 1.0:
        raise ValueError("tau must be in (0,1]")
    if int(max_primitive_steps) <= 0:
        raise ValueError("max_primitive_steps must be positive")
    calibration_config = dict(calibration or {})
    return {
        "schema_version": AWAC_TRAINING_CONTRACT_SCHEMA_VERSION,
        "contract_id": AWAC_TRAINING_CONFIG_CONTRACT_ID,
        "algorithm": AWAC_ALGORITHM_ID,
        "phase": phase,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract": AWAC_POLICY_INPUT_CONTRACT,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(int(max_primitive_steps)),
        "max_primitive_steps": int(max_primitive_steps),
        "observation_contract": "reliable_exact_endpoint_snapshot",
        "observation_source": "reliable_exact_endpoint_snapshot",
        "reward_contract_id": REWARD_CONTRACT_ID,
        "reward_contract_sha256": reward_contract_sha256(),
        "reward_scale": AWAC_REWARD_SCALE,
        "reward_scale_owner": AWAC_REWARD_SCALE_OWNER,
        "gamma": float(gamma),
        "tau": float(tau),
        "replay_contract_id": AWAC_REPLAY_CONTRACT_ID,
        "replay_contract_sha256": replay_contract_sha256(),
        "replay_fields": list(REPLAY_FIELDS),
        "behavior_sources": behavior_source_contract(),
        "actor_initialization": "strict_bc60k_state_dict",
        "critic_initialization": "bc60k_encoder_copy_independent_q_heads",
        "safety_mask_contract": "depth",
        "terminal_contract": {
            "normal_terminal_done": list(TERMINAL_REASONS),
            "normal_terminal_bootstrap": False,
            "runtime_abort": "drop_and_diagnose",
        },
        "actor_update_enabled": phase != AWAC_PHASE_CRITIC_CALIBRATION,
        "privileged_policy_inputs": [],
        "calibration": calibration_config,
    }


def awac_training_contract_sha256(contract: Optional[Mapping] = None) -> str:
    """Hash a resolved AWAC Training Contract using canonical JSON."""

    value = (
        build_awac_training_contract()
        if contract is None
        else dict(contract)
    )
    return canonical_json_sha256(value)


def awac_source_sha256(package_root: Path, relative_files: Iterable[str]) -> str:
    """Hash an explicit AWAC source manifest in deterministic path order."""
    root = Path(package_root).expanduser().resolve()
    digest = hashlib.sha256()
    for relative in sorted(str(value) for value in relative_files):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError("AWAC source-contract file missing: {}".format(path))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_awac_checkpoint_identity(checkpoint: Mapping) -> None:
    """Fail closed on an AWAC checkpoint identity mismatch."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("AWAC checkpoint must be a mapping")
    required = {
        "algorithm_id": AWAC_ALGORITHM_ID,
        "awac_checkpoint_contract_id": AWAC_CHECKPOINT_CONTRACT_ID,
        "model_type": AWAC_MODEL_TYPE,
        "training_config_contract_id": AWAC_TRAINING_CONFIG_CONTRACT_ID,
        "software_version": SOFTWARE_VERSION,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(),
        "reward_contract_id": REWARD_CONTRACT_ID,
    }
    for field, expected in required.items():
        if checkpoint.get(field) != expected:
            raise ValueError(
                "AWAC checkpoint {} mismatch: received={} expected={}".format(
                    field, checkpoint.get(field), expected
                )
            )
    if "alpha" in checkpoint or "log_alpha" in checkpoint:
        raise ValueError("AWAC checkpoint contains unsupported entropy state")
    if checkpoint.get("privileged_runtime_inputs", []):
        raise ValueError("AWAC checkpoint requires privileged runtime inputs")
