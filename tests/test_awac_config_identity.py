"""Field-level AWAC calibration-to-standard configuration identity tests."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.unit


def test_config_identity_classifies_only_phase_and_actor_enable_changes():
    from planning.awac.config_identity import compare_awac_config_identity

    calibration = {
        "training_contract": {
            "phase": "critic_calibration",
            "actor_update_enabled": False,
            "gamma": 0.99,
        },
        "optimization_config": {"bc_kl_weight": 0.05, "gamma": 0.99},
    }
    standard = {
        "training_contract": {
            "phase": "awac_training",
            "actor_update_enabled": True,
            "gamma": 0.99,
        },
        "optimization_config": {"bc_kl_weight": 0.05, "gamma": 0.99},
        "phase1_training_contract": {"actor_depth_lr": 1.0e-6},
    }
    report = compare_awac_config_identity(calibration, standard)

    assert report["status"] == "PASS"
    assert report["expected_phase_diff_count"] == 1
    assert report["expected_actor_enable_diff_count"] == 2
    assert report["training_semantic_unexpected_diff_count"] == 0
    assert report["unknown_diff_count"] == 0


def test_config_identity_does_not_hide_training_or_unknown_changes():
    from planning.awac.config_identity import compare_awac_config_identity

    calibration = {
        "training_contract": {"phase": "critic_calibration", "gamma": 0.99},
        "optimization_config": {"bc_kl_weight": 0.05},
    }
    standard = {
        "training_contract": {"phase": "critic_calibration", "gamma": 0.98},
        "optimization_config": {"bc_kl_weight": 0.10},
        "future_unowned_field": True,
    }
    report = compare_awac_config_identity(calibration, standard)

    assert report["status"] == "FAIL"
    assert report["training_semantic_unexpected_diff_count"] == 2
    assert report["unknown_diff_count"] == 1

