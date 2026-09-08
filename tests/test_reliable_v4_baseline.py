"""Tests for the canonical reliable-v4 P3 baseline contract."""

from __future__ import annotations

import copy

import pytest

from planning.evaluation.reliable_v4_baseline import validate_reliable_v4_baseline
from planning.contracts.task import TASK_CONTRACT_ID


MISSION_SHA = "135f21f73eeae2b034eda2ddcec0dd0dcaa915e8bf18426ebf9a04508f050316"
CHECKPOINT_SHA = "188e41301b67cd8c53ffdd45d55684c5f7626b4a4208eca599e322870e8c31f7"
TASK_SHA = "5862af0f354c408d74cf4904dc7ecf609944553443ae1128097136be979fd30a"
PLAYER_SHA = "61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365"
ASSEMBLY_SHA = "ca470709131f27586832749d02687b8697de16c9e08920c1485ba674c3d05045"
BRIDGE_SHA = "1e614e13577c1d6a3c1d46318eddd31668fd235a89e3a7456ddf6da11ddcd6e7"
SCHEMA_SHA = "948056976bd90b89504420712d3b23cc7cadda2090cd8aab5b223aa5b296c40a"


def _summary(**overrides):
    summary = {
        "evaluation_contract_id": "policy_unity_fixed_holdout",
        "policy_unity_outcome_contract_id": "policy_unity_terminal_outcomes",
        "policy_runtime_contract_id": "policy_runtime_mask_execution",
        "task_contract_id": TASK_CONTRACT_ID,
        "episodes": 100,
        "terminal_outcome_total": 100,
        "unclassified_count": 0,
        "ambiguous_outcome_count": 0,
        "outcome_partition_valid": True,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "mission_index_sha256": MISSION_SHA,
        "task_contract_sha256": TASK_SHA,
        "max_steps": 45,
        "policy_action_mode": "deterministic_argmax",
        "policy_temperature": 0.0,
        "safety_mask": "depth",
        "execution_mode": "continuous",
        "reliable_v4": True,
        "depth_mask_config": {
            "collision_radius_m": 0.40,
            "slack_m": 0.08,
            "sample_stride": 4,
            "max_patch_radius_px": 14,
        },
        "success_count": 48,
        "collision_count": 17,
        "dead_end_count": 30,
        "timeout_count": 5,
        "far_count": 0,
        "altitude_violation_count": 0,
    }
    summary.update(overrides)
    return summary


def _runtime_manifest(**overrides):
    manifest = {
        "schema": "p3_runtime_identity_manifest_v1",
        "identity_validated": True,
        "player_sha256": PLAYER_SHA,
        "assembly_csharp_sha256": ASSEMBLY_SHA,
        "bridge_sha256": BRIDGE_SHA,
        "schema_spec_sha256": SCHEMA_SHA,
        "bc_checkpoint_sha256": CHECKPOINT_SHA,
        "capabilities": [
            "reliable_command_v4",
            "primitive_result_v4",
            "snapshot_v4",
            "reset_v4",
        ],
    }
    manifest.update(overrides)
    return manifest


def test_exact_canonical_contract_passes():
    result = validate_reliable_v4_baseline(_summary(), _runtime_manifest())
    assert result["contract_id"] == "reliable_v4_fixed_dev_baseline_v1"
    assert result["reference_outcomes"]["success_count"] == 48


def test_canonical_identity_accepts_baseline_even_when_final_quality_gate_is_red():
    summary = _summary(
        quality_pass=False,
        quality_thresholds={
            "min_success_rate": 0.6,
            "max_collision_rate": 0.05,
            "max_dead_end_rate": 0.25,
            "max_timeout_rate": 0.05,
            "max_far_rate": 0.0,
        },
    )

    result = validate_reliable_v4_baseline(summary, _runtime_manifest())

    assert result["checkpoint_sha256"] == CHECKPOINT_SHA
    assert result["mission_index_sha256"] == MISSION_SHA


@pytest.mark.parametrize("field", ["mission_index_sha256", "checkpoint_sha256"])
def test_wrong_input_identity_fails(field):
    summary = _summary(**{field: "0" * 64})
    with pytest.raises(ValueError, match=field):
        validate_reliable_v4_baseline(summary, _runtime_manifest())


@pytest.mark.parametrize("field", ["player_sha256", "bridge_sha256"])
def test_wrong_runtime_identity_fails(field):
    manifest = _runtime_manifest(**{field: "0" * 64})
    with pytest.raises(ValueError, match=field):
        validate_reliable_v4_baseline(_summary(), manifest)


def test_legacy_76_percent_expectation_is_not_accepted():
    summary = _summary(
        success_count=76,
        collision_count=2,
        dead_end_count=19,
        timeout_count=3,
    )
    with pytest.raises(ValueError, match="success_count"):
        validate_reliable_v4_baseline(summary, _runtime_manifest())


def test_different_mission_set_cannot_reuse_fixed_dev_contract():
    summary = copy.deepcopy(_summary(mission_index_sha256="1" * 64))
    with pytest.raises(ValueError, match="mission_index_sha256"):
        validate_reliable_v4_baseline(summary, _runtime_manifest())
