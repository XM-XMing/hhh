"""Contracts for the read-only value-return recertification report."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_fixed_behavior_and_target_policy_identity_is_explicitly_mismatched():
    from scripts.recertify_awac_value_returns import _policy_contexts

    contexts = _policy_contexts(
        bc_checkpoint_sha256="b" * 64,
        v7_checkpoint_sha256="v" * 64,
        current_checkpoint_sha256="c" * 64,
    )

    assert contexts["v7"]["alignment"]["status"] == "MISMATCH"
    assert contexts["current_25k"]["alignment"]["status"] == "MISMATCH"
    assert contexts["v7"]["return_policy"]["selection_mode"] == "deterministic_argmax"
    assert contexts["v7"]["target_policy"]["selection_mode"] == "masked_softmax"
    assert contexts["current_25k"]["target_policy"]["checkpoint_sha256"] == "c" * 64


@pytest.mark.unit
def test_single_snapshot_cannot_upgrade_to_new_gate_certification():
    from scripts.recertify_awac_value_returns import _offline_value_certification

    certification = _offline_value_certification(
        complete_episode_count=25,
        usable_row_count=604,
        mc_rho=0.90,
        value_policy_alignment="MISMATCH",
        independent_snapshot_count=1,
    )

    assert certification["state"] == "PENDING"
    assert certification["reason"] == "PENDING_INSUFFICIENT_DATA"
    assert certification["historical_pass_not_auto_upgraded"] is True
    assert certification["policy_alignment_blocks_target_value_certification"] is True


@pytest.mark.unit
def test_public_return_contract_keeps_rows_out_of_json_but_hashes_them():
    from scripts.recertify_awac_value_returns import _public_return_contract

    report = _public_return_contract(
        {
            "return_semantics": "episode_monte_carlo_return_v1",
            "episode_keys": [("mission", "episode")],
            "returns": [0.25, -1.0],
        }
    )

    assert "returns" not in report
    assert len(report["returns_array_sha256"]) == 64
    assert report["returns_summary"]["count"] == 2
