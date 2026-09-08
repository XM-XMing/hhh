"""P3-C public contracts for the first accepted AWAC Actor update."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.unit


def test_actor_schedule_requires_replay_threshold_burnin_and_interval():
    from planning.awac.optimization import actor_update_due

    contract = {
        "actor_learning_starts": 9000,
        "critic_burnin_updates": 2000,
        "actor_update_interval": 2,
    }

    assert not actor_update_due(
        replay_total_added=8999, learner_update_step=1999, **contract
    )
    assert not actor_update_due(
        replay_total_added=9000, learner_update_step=1999, **contract
    )
    assert not actor_update_due(
        replay_total_added=9000, learner_update_step=2000, **contract
    )
    assert actor_update_due(
        replay_total_added=9000, learner_update_step=2001, **contract
    )


def test_first_accepted_actor_update_evidence_requires_identity_delta():
    from planning.awac.actor_evidence import first_actor_update_evidence

    assert first_actor_update_evidence(
        global_step=9000,
        actor_update_step=0,
        metrics={},
        actor_fingerprint_before="a" * 64,
        actor_fingerprint_after="a" * 64,
    ) is None

    metrics = {
        "actor_loss": 1.2,
        "awac_advantage_mean": 0.1,
        "awac_advantage_std": 0.3,
        "awac_advantage_min": -0.4,
        "awac_advantage_max": 0.7,
        "awac_weight_mean": 1.0,
        "awac_weight_std": 0.2,
        "awac_weight_p50": 0.95,
        "awac_weight_p95": 1.4,
        "awac_weight_max": 1.6,
        "awac_weight_cap_fraction": 0.0,
        "awac_data_q_mean": 0.2,
        "awac_data_q_std": 0.4,
        "bc_kl": 0.01,
        "bc_kl_p95": 0.03,
        "bc_kl_max_after_update": 0.04,
        "bc_kl_hard_budget": 0.1,
        "bc_kl_budget_exceeded": 0.0,
        "actor_gradient_norm": 0.5,
        "actor_parameter_delta_norm": 0.01,
        "actor_optimizer_step": 1.0,
        "actor_trust_region_rejection_count": 0.0,
        "parameters_finite": 1.0,
    }

    evidence = first_actor_update_evidence(
        global_step=9000,
        actor_update_step=1,
        metrics=metrics,
        actor_fingerprint_before="a" * 64,
        actor_fingerprint_after="b" * 64,
    )

    assert evidence == {
        "global_step": 9000,
        "actor_update_step": 1,
        "actor_fingerprint_before": "a" * 64,
        "actor_fingerprint_after": "b" * 64,
        **metrics,
    }


def test_first_accepted_actor_update_evidence_rejects_unchanged_actor():
    from planning.awac.actor_evidence import first_actor_update_evidence

    metrics = {
        "actor_loss": 1.2,
        "awac_advantage_mean": 0.1,
        "awac_advantage_std": 0.3,
        "awac_advantage_min": -0.4,
        "awac_advantage_max": 0.7,
        "awac_weight_mean": 1.0,
        "awac_weight_std": 0.2,
        "awac_weight_p50": 0.95,
        "awac_weight_p95": 1.4,
        "awac_weight_max": 1.6,
        "awac_weight_cap_fraction": 0.0,
        "awac_data_q_mean": 0.2,
        "awac_data_q_std": 0.4,
        "bc_kl": 0.01,
        "bc_kl_p95": 0.03,
        "bc_kl_max_after_update": 0.04,
        "bc_kl_hard_budget": 0.1,
        "bc_kl_budget_exceeded": 0.0,
        "actor_gradient_norm": 0.5,
        "actor_parameter_delta_norm": 0.01,
        "actor_optimizer_step": 1.0,
        "actor_trust_region_rejection_count": 0.0,
        "parameters_finite": 1.0,
    }

    with pytest.raises(ValueError, match="fingerprint"):
        first_actor_update_evidence(
            global_step=9000,
            actor_update_step=1,
            metrics=metrics,
            actor_fingerprint_before="a" * 64,
            actor_fingerprint_after="a" * 64,
        )


def test_bc_reference_fingerprint_must_remain_the_frozen_bc_actor():
    from planning.awac.actor_evidence import require_reference_fingerprint

    expected = "a" * 64
    assert require_reference_fingerprint(
        reference_fingerprint=expected,
        expected_fingerprint=expected,
    ) == expected
    with pytest.raises(ValueError, match="BC reference"):
        require_reference_fingerprint(
            reference_fingerprint="b" * 64,
            expected_fingerprint=expected,
        )
