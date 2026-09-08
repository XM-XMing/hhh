"""Regression tests for the isolated TD-horizon diagnostic seam."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from planning.awac.model import build_critic
from planning.awac.td_horizon_diagnostic import (
    BudgetCritic,
    budget_fraction,
    budget_steps,
    build_sequence_sidecar,
    extend_critic_state_dict,
    masked_expected_value_independent,
    n_step_discounted_return,
    next_budget_steps,
)
from planning.contracts.feature import POLICY_VECTOR_DIM


def _replay_arrays(lengths):
    total = sum(lengths)
    depth = np.arange(total * 4, dtype=np.uint8).reshape(total, 2, 2)
    vector = np.arange(total * 3, dtype=np.float32).reshape(total, 3)
    action_mask = np.ones((total, 2), dtype=np.uint8)
    done = np.zeros(total, dtype=np.uint8)
    cursor = 0
    for length in lengths:
        end = cursor + length - 1
        done[end] = 1
        cursor += length
    return {
        "depth": depth,
        "vector": vector,
        "action_mask": action_mask,
        "next_depth": np.roll(depth, -1, axis=0),
        "next_vector": np.roll(vector, -1, axis=0),
        "next_action_mask": np.roll(action_mask, -1, axis=0),
        "done": done,
    }


def test_budget_is_task_budget_not_recorded_episode_tail():
    assert budget_steps(horizon=45, transition_index=0) == 45
    assert budget_steps(horizon=45, transition_index=44) == 1
    assert next_budget_steps(horizon=45, transition_index=0) == 44
    assert next_budget_steps(horizon=45, transition_index=44) == 0
    assert budget_fraction(horizon=45, transition_index=9) == pytest.approx(36.0 / 45.0)


def test_sequence_sidecar_stays_within_episode_and_records_retrospective_separately():
    arrays = _replay_arrays([3, 7])
    sidecar, report = build_sequence_sidecar(arrays=arrays, count=10, horizon=45, episode_lengths=[3, 7])
    assert report["continuity_failures"] == 0
    assert sidecar["nstep_indices"][1].tolist() == [1, 2, -1, -1, -1]
    assert sidecar["nstep_terminal"][1] == 1
    assert sidecar["bootstrap_index"][0] == -1
    assert sidecar["steps_to_recorded_episode_end"][0] == 2
    assert sidecar["remaining_budget_steps"][0] == 45
    assert report["retrospective_field_is_model_input"] is False


def test_sequence_sidecar_rejects_endpoint_cross_split():
    arrays = _replay_arrays([3, 3])
    arrays["next_vector"][1] = -99.0
    with pytest.raises(ValueError, match="continuity"):
        build_sequence_sidecar(arrays=arrays, count=6, horizon=45, episode_lengths=[3, 3])


def test_n_step_discount_and_terminal_no_bootstrap():
    value, used, terminal = n_step_discounted_return(
        rewards=[1.0, 2.0, 99.0], done=[False, True, True], gamma=0.99, reward_scale=0.1, bootstrap_value=100.0
    )
    assert used == 2
    assert terminal is True
    assert value == pytest.approx(0.1 + 0.99 * 0.2)


def test_n_step_full_prefix_bootstraps_once_and_n1_formula():
    value, used, terminal = n_step_discounted_return(
        rewards=[1.0, 2.0, 3.0, 4.0, 5.0], done=[False] * 5, gamma=0.99, reward_scale=0.1, bootstrap_value=7.0
    )
    expected = sum((0.99 ** index) * 0.1 * reward for index, reward in enumerate([1, 2, 3, 4, 5])) + (0.99 ** 5) * 7.0
    assert used == 5 and terminal is False
    assert value == pytest.approx(expected)
    n1, used1, terminal1 = n_step_discounted_return(rewards=[1.0], done=[False], gamma=0.99, reward_scale=0.1, bootstrap_value=7.0)
    assert n1 == pytest.approx(0.1 + 0.99 * 7.0)
    assert used1 == 1 and terminal1 is False


def test_independent_masked_expected_value_ignores_invalid_actions():
    logits = torch.tensor([[0.0, 100.0, 0.0]])
    q1 = torch.tensor([[1.0, 999.0, 3.0]])
    q2 = torch.tensor([[2.0, 999.0, 5.0]])
    mask = torch.tensor([[True, False, True]])
    value, probabilities = masked_expected_value_independent(logits=logits, q1=q1, q2=q2, action_mask=mask, torch=torch)
    assert float(probabilities[0, 1]) == 0.0
    assert float(probabilities.sum()) == pytest.approx(1.0)
    assert float(value) == pytest.approx(2.0, abs=1.0e-5)


def test_zero_initialized_budget_column_preserves_initial_critic_output():
    source = build_critic(torch.nn, depth_channels=1, vec_dim=POLICY_VECTOR_DIM, num_actions=105)
    extended_state = extend_critic_state_dict(source.state_dict(), torch=torch)
    assert extended_state["vector_encoder.0.weight"].shape[1] == POLICY_VECTOR_DIM + 1
    assert torch.count_nonzero(extended_state["vector_encoder.0.weight"][:, -1]).item() == 0
    diagnostic = BudgetCritic(nn=torch.nn, depth_channels=1, source_state_dict=source.state_dict(), device=torch.device("cpu"))
    depth = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
    vector = torch.randn((2, POLICY_VECTOR_DIM), dtype=torch.float32)
    with torch.no_grad():
        expected = source(depth, vector)
        actual = diagnostic(depth, vector, torch.zeros(2))
    assert torch.allclose(expected, actual, atol=1.0e-6, rtol=1.0e-5)
