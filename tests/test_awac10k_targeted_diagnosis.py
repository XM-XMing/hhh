"""Pure numeric contracts for the read-only Critic diagnosis."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_descending_rank_and_affine_fit_are_explicit():
    from scripts.diagnose_awac10k_targeted_trace import _affine_fit, _rank_descending

    assert np.array_equal(_rank_descending(np.asarray([3.0, 1.0, 2.0])), [1.0, 3.0, 2.0])
    assert np.array_equal(_rank_descending(np.asarray([1.0, 1.0, 0.0])), [1.5, 1.5, 3.0])
    fit = _affine_fit(np.asarray([1.0, 2.0, 3.0]), np.asarray([5.0, 7.0, 9.0]))
    assert fit["alpha"] == pytest.approx(2.0)
    assert fit["beta"] == pytest.approx(3.0)
    assert fit["r2"] == pytest.approx(1.0)


@pytest.mark.unit
def test_first_divergence_q_preference_uses_only_first_disagreement():
    from scripts.diagnose_awac10k_targeted_trace import _trace_category_summary

    rows = [
        {"episode_id": "1", "step": 0, "top1_disagreement": True, "critic_prefers_awac": True, "q_awac_top1": 1.0, "q_bc_top1": 0.0, "bc_to_awac_kl": 0.1},
        {"episode_id": "1", "step": 1, "top1_disagreement": False, "critic_prefers_awac": False, "q_awac_top1": 0.0, "q_bc_top1": 1.0, "bc_to_awac_kl": 0.2},
        {"episode_id": "2", "step": 0, "top1_disagreement": True, "critic_prefers_awac": False, "q_awac_top1": 0.0, "q_bc_top1": 1.0, "bc_to_awac_kl": 0.3},
    ]

    summary = _trace_category_summary(rows, ["1", "2"])

    assert summary["first_divergence_q_prefers_awac_rate"] == pytest.approx(0.5)
    assert summary["q_prefers_awac_rate"] == pytest.approx(1.0 / 3.0)


@pytest.mark.unit
def test_static_ranking_change_never_claims_high_causal_critic_error():
    from scripts.diagnose_awac10k_targeted_trace import _diagnosis

    diagnosis = _diagnosis(
        {"raw_q_abs_p95_ratio_awac10k_over_v7": 3.0},
        {"q_argmax_change_rate": 0.75},
        {"action_preference_flip_on_policy_disagreement": 0.80},
        {
            "regressed": {"runtime_trace_available": False},
            "recovered": {"runtime_trace_available": False},
        },
        {"timeout_count": 0, "classification": "INCONCLUSIVE"},
    )

    assert diagnosis["ranking_change_observed"] == "YES"
    assert diagnosis["same_state_action_value_rank_validated"] == "NO"
    assert diagnosis["ranking_error_causality"] == "INCONCLUSIVE"
    assert diagnosis["confidence"] == "LOW"
    assert diagnosis["primary_root_cause"] == "INCONCLUSIVE"
    assert diagnosis["next_action"] != "TUNE_CRITIC"
