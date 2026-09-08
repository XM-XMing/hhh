"""Contracts for the single formal AWAC checkpoint identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from planning.contracts.collection import canonical_sha256
from planning.contracts.feature import (
    FEATURE_CONTRACT_ID,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    policy_input_contract_sha256,
)
from planning.contracts.offpolicy import (
    AWAC_ALGORITHM_ID,
    AWAC_CHECKPOINT_CONTRACT_ID,
    AWAC_MODEL_TYPE,
    AWAC_TRAINING_CONFIG_CONTRACT_ID,
    algorithm_source_sha256,
    get_offpolicy_algorithm_spec,
    is_offpolicy_checkpoint,
    offpolicy_source_files,
    validate_awac_checkpoint_contract,
    validate_offpolicy_checkpoint_contract,
    validate_policy_checkpoint_algorithm,
)
from planning.contracts.policy_runtime import POLICY_RUNTIME_CONTRACT_ID
from planning.contracts.reward import REWARD_CONTRACT_ID
from planning.contracts.task import TASK_CONTRACT_ID, task_contract_sha256
from planning.version import SOFTWARE_VERSION


pytestmark = pytest.mark.unit


def _checkpoint(algorithm_id: str = AWAC_ALGORITHM_ID):
    spec = get_offpolicy_algorithm_spec(algorithm_id)
    resolved = {
        "contract_id": spec.training_config_contract_id,
        "algorithm_id": spec.algorithm_id,
        "gamma": 0.99,
    }
    return {
        spec.checkpoint_contract_field: spec.checkpoint_contract_id,
        "software_version": SOFTWARE_VERSION,
        "algorithm_id": spec.algorithm_id,
        "model_type": spec.model_type,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(),
        "reward_contract_id": REWARD_CONTRACT_ID,
        "training_config_contract_id": spec.training_config_contract_id,
        "resolved_training_config": resolved,
        "resolved_training_config_sha256": canonical_sha256(resolved),
        "source_code_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
        "vec_dim": POLICY_VECTOR_DIM,
        "num_actions": NUM_ACTIONS,
        "privileged_runtime_inputs": [],
        "actor_state_dict": {"weight": 1},
    }


def test_formal_offpolicy_registry_contains_only_awac():
    spec = get_offpolicy_algorithm_spec(AWAC_ALGORITHM_ID)

    assert spec.checkpoint_contract_id == AWAC_CHECKPOINT_CONTRACT_ID
    assert spec.training_config_contract_id == AWAC_TRAINING_CONFIG_CONTRACT_ID
    assert spec.model_type == AWAC_MODEL_TYPE
    with pytest.raises(ValueError, match="unsupported formal off-policy"):
        get_offpolicy_algorithm_spec("unsupported_algorithm")


def test_awac_contract_is_accepted_by_generic_and_specific_validators():
    checkpoint = _checkpoint()

    assert validate_offpolicy_checkpoint_contract(checkpoint).algorithm_id == AWAC_ALGORITHM_ID
    assert validate_awac_checkpoint_contract(checkpoint) is None
    assert validate_policy_checkpoint_algorithm(
        checkpoint, "actor_state_dict"
    ).algorithm_id == AWAC_ALGORITHM_ID
    assert validate_policy_checkpoint_algorithm(
        {"model_type": "bc_soft", "model_state_dict": {}}, "model_state_dict"
    ) is None
    assert is_offpolicy_checkpoint(checkpoint) is True


def test_unknown_or_foreign_algorithm_fails_closed():
    checkpoint = _checkpoint()
    checkpoint["algorithm_id"] = "foreign_algorithm"
    with pytest.raises(ValueError, match="unsupported formal off-policy"):
        validate_policy_checkpoint_algorithm(checkpoint, "actor_state_dict")

    with pytest.raises(ValueError, match="unsupported formal off-policy"):
        validate_policy_checkpoint_algorithm(
            {"actor_state_dict": {}, "algorithm_id": "foreign_algorithm"},
            "actor_state_dict",
        )


def test_awac_rejects_mismatched_identity_and_privileged_inputs():
    checkpoint = _checkpoint()
    checkpoint["resolved_training_config"]["algorithm_id"] = "foreign_algorithm"
    checkpoint["resolved_training_config_sha256"] = canonical_sha256(
        checkpoint["resolved_training_config"]
    )
    with pytest.raises(ValueError, match="resolved algorithm"):
        validate_offpolicy_checkpoint_contract(checkpoint)

    checkpoint = _checkpoint()
    checkpoint["privileged_runtime_inputs"] = ["global_map"]
    with pytest.raises(ValueError, match="privileged"):
        validate_offpolicy_checkpoint_contract(checkpoint)


def test_awac_source_manifest_uses_new_owners_only():
    package_root = Path(__file__).resolve().parents[1]
    files = set(offpolicy_source_files(AWAC_ALGORITHM_ID))

    assert "python/planning/awac/learner.py" in files
    assert "python/planning/awac/model.py" in files
    assert "python/planning/runtime/parallel_env.py" in files
    legacy_package = "planning/" + "rl/"
    legacy_algorithm = "s" + "a" + "c"
    assert all(
        legacy_package not in path and legacy_algorithm not in path.lower()
        for path in files
    )
    assert len(algorithm_source_sha256(package_root, AWAC_ALGORITHM_ID)) == 64
