"""Checkpoint schema tests for the independent AWAC owner."""

from __future__ import annotations

import copy

import pytest

from planning.awac.checkpoint import (
    build_awac_checkpoint_identity,
    validate_awac_checkpoint_payload,
)
from planning.awac.contract import (
    AWAC_ALGORITHM_ID,
    AWAC_CHECKPOINT_CONTRACT_ID,
    AWAC_MODEL_TYPE,
    AWAC_TRAINING_CONFIG_CONTRACT_ID,
)
from planning.contracts.feature import (
    FEATURE_CONTRACT_ID,
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    policy_input_contract_sha256,
)
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.policy_runtime import POLICY_RUNTIME_CONTRACT_ID
from planning.contracts.reward import REWARD_CONTRACT_ID, reward_contract_sha256
from planning.contracts.task import (
    TASK_CONTRACT_ID,
    TASK_CONTRACT_SCHEMA_VERSION,
    task_contract_sha256,
)
from planning.version import SOFTWARE_VERSION
from planning.contracts.collection import canonical_sha256


def _payload():
    config = {
        "contract_id": AWAC_TRAINING_CONFIG_CONTRACT_ID,
        "algorithm_id": AWAC_ALGORITHM_ID,
        "seed": 55,
    }
    state = {"weight": object()}
    # State values are only checked for non-empty mappings by the schema seam.
    return {
        "algorithm_id": AWAC_ALGORITHM_ID,
        "awac_checkpoint_contract_id": AWAC_CHECKPOINT_CONTRACT_ID,
        "awac_checkpoint_schema_id": "awac_checkpoint_schema_v3",
        "model_type": AWAC_MODEL_TYPE,
        "training_config_contract_id": AWAC_TRAINING_CONFIG_CONTRACT_ID,
        "software_version": SOFTWARE_VERSION,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(),
        "reward_contract_id": REWARD_CONTRACT_ID,
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "mpl_contract_sha256": "m" * 64,
        "global_step": 0,
        "replay_size": 0,
        "update_step": 0,
        "actor_state_dict": state,
        "critic1_state_dict": state,
        "critic2_state_dict": state,
        "target_critic1_state_dict": state,
        "target_critic2_state_dict": state,
        "actor_optimizer_state_dict": state,
        "critic_optimizer_state_dict": state,
        "resolved_training_config": config,
        "resolved_training_config_sha256": canonical_sha256(config),
        "source_code_sha256": "s" * 64,
        "split_manifest_sha256": "d" * 64,
        "vec_dim": 127,
        "num_actions": 105,
    }


@pytest.mark.unit
def test_shared_awac_checkpoint_identity_matches_the_formal_contract():
    identity = build_awac_checkpoint_identity(max_primitive_steps=45)

    assert identity == {
        "awac_checkpoint_schema_id": "awac_checkpoint_schema_v3",
        "algorithm_id": AWAC_ALGORITHM_ID,
        "awac_checkpoint_contract_id": AWAC_CHECKPOINT_CONTRACT_ID,
        "model_type": AWAC_MODEL_TYPE,
        "training_config_contract_id": AWAC_TRAINING_CONFIG_CONTRACT_ID,
        "software_version": SOFTWARE_VERSION,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_schema_version": TASK_CONTRACT_SCHEMA_VERSION,
        "task_contract_sha256": task_contract_sha256(45),
        "max_primitive_steps": 45,
        "reward_contract_id": REWARD_CONTRACT_ID,
        "reward_contract_sha256": reward_contract_sha256(),
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "vec_dim": POLICY_VECTOR_DIM,
        "num_actions": NUM_ACTIONS,
        "depth_history_frames": 1,
        "initial_prev_action": INITIAL_PREV_ACTION,
    }


@pytest.mark.unit
def test_awac_checkpoint_payload_accepts_v3_without_entropy_state():
    assert validate_awac_checkpoint_payload(_payload())["algorithm_id"] == AWAC_ALGORITHM_ID


@pytest.mark.unit
def test_awac_checkpoint_payload_rejects_entropy_state_and_v1_task_hash():
    payload = _payload()
    payload["log_alpha"] = 0.0
    with pytest.raises(ValueError, match="entropy state"):
        validate_awac_checkpoint_payload(payload)
    payload = _payload()
    payload["task_contract_sha256"] = "legacy"
    with pytest.raises(ValueError, match="task_contract_sha256"):
        validate_awac_checkpoint_payload(payload)
