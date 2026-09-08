"""RED-first contracts for the AWAC BC-to-RL provenance boundary."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from planning.contracts.feature import (
    CONTINUOUS_DIM,
    FEATURE_CONTRACT_ID,
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    policy_input_contract_sha256,
)
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.policy_runtime import POLICY_RUNTIME_CONTRACT_ID
from planning.mission.spec import TASK_CONTRACT_ID
from planning.contracts.task import task_contract_sha256


pytestmark = pytest.mark.unit


def _checkpoint(**overrides):
    value = {
        "model_state_dict": {"fixture": "tensor"},
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(),
        "mpl_contract_sha256": "m" * 64,
        "vec_dim": POLICY_VECTOR_DIM,
        "state_feature_dim": CONTINUOUS_DIM,
        "previous_action_dim": NUM_ACTIONS,
        "action_ordering": "motion_primitive_index_ascending",
        "num_actions": NUM_ACTIONS,
        "depth_history_frames": 1,
        "initial_prev_action": INITIAL_PREV_ACTION,
        "feature_mean": np.zeros((CONTINUOUS_DIM,), dtype=np.float32),
        "feature_std": np.ones((CONTINUOUS_DIM,), dtype=np.float32),
        "safety_mask": "depth",
        "execution_mode": "continuous",
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
    }
    value.update(overrides)
    return value


def _runtime():
    return {
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "reliable_execution_enabled": True,
    }


def _replay_metadata():
    return {
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_semantics": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "bc_checkpoint_sha256": "b" * 64,
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(),
        "mpl_contract_sha256": "m" * 64,
        "run_identity": "awac-fixture-run",
        "run_contract_sha256": "r" * 64,
        "legacy_replay_transition_count": 0,
        "reliable_v4_transition_count": 3,
    }


def _current():
    return {
        "bc_checkpoint_sha256": "b" * 64,
        "bc_observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "bc_observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "runtime_observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "runtime_observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "expected_observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(),
        "mpl_contract_sha256": "m" * 64,
        "run_identity": "awac-fixture-run",
        "run_contract_sha256": "r" * 64,
    }


def test_missing_bc_observation_contract_is_rejected():
    from planning.contracts.awac_handoff import validate_awac_bc_checkpoint

    checkpoint = _checkpoint()
    checkpoint.pop("observation_contract")
    with pytest.raises(ValueError, match="missing observation contract"):
        validate_awac_bc_checkpoint(checkpoint, expected_mpl_contract_sha256="m" * 64)


def test_legacy_bc_observation_contract_is_rejected():
    from planning.contracts.awac_handoff import validate_awac_bc_checkpoint

    with pytest.raises(ValueError, match="legacy|expected"):
        validate_awac_bc_checkpoint(
            _checkpoint(
                observation_contract="legacy_async_telemetry",
                observation_source="legacy_async_telemetry",
            ),
            expected_mpl_contract_sha256="m" * 64,
        )


def test_bc_runtime_contract_mismatch_is_rejected():
    from planning.contracts.awac_handoff import validate_awac_runtime_provenance

    with pytest.raises(ValueError, match="runtime observation"):
        validate_awac_runtime_provenance(
            observation_contract="legacy_async_telemetry",
            observation_source="legacy_async_telemetry",
            reliable_execution_enabled=False,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("feature_contract_id", "wrong-feature"),
        ("task_contract_sha256", "t" * 64),
        ("mpl_contract_sha256", "x" * 64),
        ("num_actions", 104),
        ("action_ordering", "reverse"),
    ),
)
def test_bc_identity_or_action_mismatch_is_rejected(field, value):
    from planning.contracts.awac_handoff import validate_awac_bc_checkpoint

    with pytest.raises(ValueError):
        validate_awac_bc_checkpoint(
            _checkpoint(**{field: value}),
            expected_mpl_contract_sha256="m" * 64,
        )


def test_valid_exact_bc_checkpoint_is_accepted():
    from planning.contracts.awac_handoff import validate_awac_bc_checkpoint

    resolved = validate_awac_bc_checkpoint(
        _checkpoint(), expected_mpl_contract_sha256="m" * 64
    )
    assert resolved["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    assert resolved["observation_source"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    assert resolved["num_actions"] == NUM_ACTIONS


def test_strict_actor_state_load_rejects_missing_or_unexpected_tensor():
    from planning.bc.model import build_model, require_torch
    from planning.awac.model import load_actor_state_dict_strict

    torch, nn, _, _, _ = require_torch()
    actor = build_model(nn, depth_channels=1)
    state = copy.deepcopy(actor.state_dict())
    load_actor_state_dict_strict(actor, state)
    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="Missing key"):
        load_actor_state_dict_strict(actor, missing)
    unexpected = dict(state)
    unexpected["unexpected.tensor"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_actor_state_dict_strict(actor, unexpected)


def test_replay_missing_or_mismatched_observation_provenance_is_rejected():
    from planning.contracts.awac_handoff import validate_awac_replay_provenance

    missing = _replay_metadata()
    missing.pop("observation_contract")
    with pytest.raises(ValueError, match="missing observation contract"):
        validate_awac_replay_provenance(missing, expected=_current())
    mismatch = _replay_metadata()
    mismatch["observation_source"] = "legacy_async_telemetry"
    with pytest.raises(ValueError, match="mismatch|legacy"):
        validate_awac_replay_provenance(mismatch, expected=_current())


def test_resume_bc_hash_and_observation_mismatch_are_rejected():
    from planning.contracts.awac_handoff import validate_awac_resume_provenance

    resume = _current()
    resume["source_bc_checkpoint_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="BC checkpoint hash"):
        validate_awac_resume_provenance(resume, _replay_metadata(), expected=_current())
    resume = _current()
    resume["runtime_observation_contract"] = "legacy_async_telemetry"
    with pytest.raises(ValueError, match="observation"):
        validate_awac_resume_provenance(resume, _replay_metadata(), expected=_current())


def test_artifact_provenance_projection_contains_run_checkpoint_summary_fields():
    from planning.contracts.awac_handoff import build_awac_provenance

    fields = build_awac_provenance(
        bc_checkpoint_path="bc.pt",
        bc_checkpoint_sha256="b" * 64,
        bc=_checkpoint(),
        runtime=_runtime(),
        replay=_replay_metadata(),
        run_identity="awac-fixture-run",
        run_contract_sha256="r" * 64,
    )
    for key in (
        "bc_checkpoint_path",
        "bc_checkpoint_sha256",
        "bc_observation_contract",
        "bc_observation_source",
        "runtime_observation_contract",
        "runtime_observation_source",
        "replay_observation_contract",
        "replay_observation_source",
        "feature_contract_id",
        "task_contract_sha256",
        "mpl_contract_sha256",
        "run_contract_sha256",
    ):
        assert key in fields
