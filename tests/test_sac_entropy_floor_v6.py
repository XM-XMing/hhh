"""Focused red/green tests for the V6 entropy-floor audit helpers."""

from __future__ import annotations

import numpy as np
import pytest

from planning.sac.entropy_audit import (
    audit_transaction_rows,
    first_entropy_crossing,
    gate_statistic,
    masked_entropy_reference,
    normalized_entropy,
    residual_saturation_metrics,
    valid_action_entropy_max,
)


def test_reference_entropy_masks_invalid_actions_and_normalizes():
    logits = np.asarray([[2.0, 1.0, -4.0]], dtype=np.float64)
    mask = np.asarray([[True, False, True]])
    probabilities, log_probabilities, entropy = masked_entropy_reference(logits, mask)
    assert probabilities[0, 1] == 0.0
    assert log_probabilities[0, 1] == 0.0
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(entropy[0], -(probabilities * log_probabilities).sum())


def test_reference_entropy_handles_zero_times_log_zero():
    probabilities, log_probabilities, entropy = masked_entropy_reference(
        np.asarray([[1000.0, -1000.0]]), np.asarray([[True, True]])
    )
    assert np.isfinite(entropy).all()
    assert np.isfinite(log_probabilities).all()
    assert np.isclose(entropy[0], 0.0, atol=1e-12)


def test_reference_rejects_empty_mask():
    with pytest.raises(ValueError, match="empty action mask"):
        masked_entropy_reference(np.zeros((1, 3)), np.zeros((1, 3), dtype=bool))


def test_single_valid_action_is_zero_entropy_and_not_normalized():
    maximum = valid_action_entropy_max(np.asarray([[False, True, False]]))
    normalized, single = normalized_entropy(np.asarray([0.0]), np.asarray([1]))
    assert np.isnan(maximum[0])
    assert np.isnan(normalized[0])
    assert bool(single[0])


def test_valid_action_entropy_maximum_is_log_n():
    maximum = valid_action_entropy_max(
        np.asarray([[True, True, False, True], [True, False, False, False]])
    )
    assert np.isclose(maximum[0], np.log(3.0))
    assert np.isnan(maximum[1])


def test_gate_statistic_matches_production_eligibility_contract():
    mask = np.asarray([[True, True], [True, False], [True, True]])
    bc_entropy = np.asarray([0.3, 0.0, 0.5])
    entropy = np.asarray([0.1, 0.0, 0.03])
    value, eligible = gate_statistic(entropy, mask, bc_entropy, floor=0.02)
    assert eligible.tolist() == [True, False, True]
    assert np.isclose(value, 0.03)


def test_first_entropy_crossing_is_strict_before_safe_after_unsafe():
    assert first_entropy_crossing([0.03, 0.025], [0.025, 0.019], 0.02) == 2
    assert first_entropy_crossing([0.01], [0.01], 0.02) is None


def test_residual_saturation_metrics_are_finite_and_bounded():
    result = residual_saturation_metrics(
        np.asarray([[-0.49, 0.0, 0.489, 0.1]], dtype=np.float64), cap=0.49
    )
    assert np.isclose(result["abs_max"], 0.49)
    assert 0.0 <= result["saturation_fraction"] <= 1.0
    assert np.isfinite(list(result.values())).all()


def test_transaction_audit_requires_pre_step_safety_and_rollback():
    rows = [
        {"pre_gate_safe": True, "optimizer_step": True, "accepted": True, "rollback": False},
        {"pre_gate_safe": True, "optimizer_step": True, "accepted": False, "rollback": True},
        {"pre_gate_safe": False, "optimizer_step": False, "accepted": False, "rollback": False},
    ]
    result = audit_transaction_rows(rows)
    assert result["status"] == "PASS"
    assert result["accepted_count_rollback_safe"] is True


def test_transaction_audit_rejects_unsafe_pre_step_optimizer_step():
    result = audit_transaction_rows(
        [{"pre_gate_safe": False, "optimizer_step": True, "accepted": False, "rollback": True}]
    )
    assert result["status"] == "FAIL"

