"""Bounded BC60K-to-AWAC initialization smoke without RL training."""

from __future__ import annotations

import numpy as np
import pytest

from planning.bc.model import build_model, require_torch
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
from planning.contracts.task import TASK_CONTRACT_ID, task_contract_sha256


pytestmark = pytest.mark.unit


def _bc_checkpoint(nn):
    model = build_model(nn, depth_channels=1)
    return {
        "model_state_dict": model.state_dict(),
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
        "reliable_execution": True,
        "legacy_rows": 0,
    }


def test_bc_and_awac_initialized_actor_have_exact_masked_policy_parity():
    from planning.awac.learner import AWACOptimizationConfig
    from planning.awac.model import masked_policy
    from planning.awac.trainer import build_learner, validate_bc_checkpoint_for_awac

    torch, nn, _, _, _ = require_torch()
    torch.manual_seed(20260903)
    checkpoint = _bc_checkpoint(nn)
    validate_bc_checkpoint_for_awac(
        checkpoint,
        mpl_contract_sha256=checkpoint["mpl_contract_sha256"],
    )
    learner = build_learner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_checkpoint=checkpoint,
        config=AWACOptimizationConfig(),
    )
    bc_actor = build_model(nn, depth_channels=1)
    bc_actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    bc_actor.eval()
    learner.actor.eval()

    depth = torch.linspace(0.0, 1.0, 4 * 32 * 32).reshape(4, 1, 32, 32)
    vector = torch.linspace(-1.0, 1.0, 4 * POLICY_VECTOR_DIM).reshape(
        4, POLICY_VECTOR_DIM
    )
    action_mask = torch.ones((4, NUM_ACTIONS), dtype=torch.bool)
    action_mask[0, 0] = False
    action_mask[1, 17:] = False
    action_mask[2, ::2] = False
    action_mask[3, -1] = False

    with torch.no_grad():
        bc_logits = bc_actor(depth, vector)
        awac_logits = learner.actor(depth, vector)
        bc_probabilities, bc_log_probabilities, bc_empty = masked_policy(
            bc_logits, action_mask, torch
        )
        awac_probabilities, awac_log_probabilities, awac_empty = masked_policy(
            awac_logits, action_mask, torch
        )

    assert bc_logits.shape == (4, NUM_ACTIONS)
    assert torch.equal(bc_logits, awac_logits)
    assert torch.equal(bc_probabilities, awac_probabilities)
    assert torch.equal(bc_log_probabilities, awac_log_probabilities)
    assert torch.equal(bc_empty, awac_empty)
    assert torch.equal(torch.argmax(bc_probabilities, dim=1), torch.argmax(awac_probabilities, dim=1))
    assert torch.equal(action_mask, action_mask.clone())
