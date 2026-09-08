from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "aggregate_bc_scale_evaluation.py"
SPEC = importlib.util.spec_from_file_location("aggregate_bc_scale_evaluation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


CONTRACT = "reliable_exact_endpoint_snapshot"


def _reliable_summary():
    return {
        "observation_contract": CONTRACT,
        "observation_source": CONTRACT,
        "checkpoint_observation_contract": CONTRACT,
        "checkpoint_observation_source": CONTRACT,
        "expected_observation_contract": CONTRACT,
        "runtime_observation_contract": CONTRACT,
        "runtime_observation_source": CONTRACT,
        "reliable_v4": True,
        "reliable_execution": True,
        "state_depth_exact_endpoint_binding": True,
        "telemetry_observation": False,
        "telemetry_fallback_enabled": False,
        "runtime_contract_override": False,
        "snapshot_missing_count": 0,
        "telemetry_lookup_count": 0,
        "unclassified_count": 0,
        "ambiguous_outcome_count": 0,
        "legacy_done_timeout_count": 0,
        "outcome_partition_valid": True,
    }


def _reliable_manifest():
    return {
        "observation_contract": CONTRACT,
        "observation_source": CONTRACT,
        "checkpoint_observation_contract": CONTRACT,
        "checkpoint_observation_source": CONTRACT,
        "expected_observation_contract": CONTRACT,
        "runtime_observation_contract": CONTRACT,
        "runtime_observation_source": CONTRACT,
        "reliable_execution": True,
        "state_depth_exact_endpoint_binding": True,
        "telemetry_observation": False,
        "telemetry_fallback_enabled": False,
        "runtime_contract_override": False,
        "snapshot_missing_count": 0,
        "telemetry_lookup_count": 0,
    }


def _row(success):
    return {
        "observation_contract": CONTRACT,
        "observation_source": CONTRACT,
        "checkpoint_observation_contract": CONTRACT,
        "checkpoint_observation_source": CONTRACT,
        "success": success,
    }


def test_reliable_exact_runtime_validation_is_independent_of_quality_gate():
    summary = _reliable_summary()
    summary["quality_pass"] = False
    result = MODULE._validate_runtime(summary, _reliable_manifest(), [_row(True)])
    assert result["reliable_exact_quality"] is True


def test_runtime_validation_rejects_telemetry_or_contract_mismatch():
    summary = _reliable_summary()
    summary["telemetry_lookup_count"] = 1
    assert MODULE._validate_runtime(summary, _reliable_manifest(), [_row(True)])["reliable_exact_quality"] is False

    rows = [_row(True)]
    rows[0]["observation_source"] = "telemetry"
    assert MODULE._validate_runtime(_reliable_summary(), _reliable_manifest(), rows)["reliable_exact_quality"] is False


def test_paired_counts_and_wilson_interval():
    left = {
        "a": {"success": False},
        "b": {"success": True},
        "c": {"success": False},
    }
    right = {
        "a": {"success": True},
        "b": {"success": False},
        "c": {"success": False},
    }
    paired = MODULE._paired(left, right)
    assert paired["left_fail_right_success"] == 1
    assert paired["left_success_right_fail"] == 1
    assert paired["mcnemar_exact_two_sided_p"] == 1.0

    low, high = MODULE.wilson_ci95(66, 100)
    assert 0.56 < low < 0.57
    assert 0.74 < high < 0.75
