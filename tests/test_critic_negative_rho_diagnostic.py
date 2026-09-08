"""Pure-contract tests for the offline Critic-negative-rho diagnostic."""

import math

import numpy as np

from scripts.diagnose_critic_negative_rho import _records_mc_returns, spearman


def _row(index, reward, done, reason="success"):
    return {
        "mission_id": "m0",
        "episode_id": "e0",
        "episode_transition_index": index,
        "holdout_record_index": index,
        "reward": reward,
        "done": done,
        "terminal_reason": reason,
        "depth": np.zeros((1, 1, 1), dtype=np.float32),
        "vector": np.zeros(1, dtype=np.float32),
        "action_mask": np.ones(1, dtype=np.bool_),
        "action": 0,
        "next_depth": np.zeros((1, 1, 1), dtype=np.float32),
        "next_vector": np.zeros(1, dtype=np.float32),
        "next_action_mask": np.ones(1, dtype=np.bool_),
        "behavior_source": 0,
    }


def test_independent_mc_applies_reward_scale_once_and_zero_terminal_tail():
    values, report, _ = _records_mc_returns(
        [_row(0, 1.0, False), _row(1, 2.0, True)],
        gamma=0.99,
        reward_scale=0.10,
    )
    assert math.isclose(values[1], 0.20, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(values[0], 0.10 + 0.99 * 0.20, rel_tol=0.0, abs_tol=1e-12)
    assert report["reward_scale"] == 0.10
    assert report["terminal_bootstrap"].startswith("zero")


def test_independent_spearman_uses_average_ties():
    value = spearman([1.0, 1.0, 2.0], [1.0, 2.0, 2.0])
    expected = float(np.corrcoef([1.5, 1.5, 3.0], [1.5, 3.0, 3.0])[0, 1])
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
