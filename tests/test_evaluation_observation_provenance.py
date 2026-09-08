"""TDD contract tests for evaluation observation provenance."""

from __future__ import annotations

import pytest
from pathlib import Path

from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    LEGACY_ASYNC_OBSERVATION_CONTRACT,
)
from planning.evaluation.observation_provenance import (
    build_evaluation_provenance_summary,
    resolve_evaluation_observation_provenance,
)


def _checkpoint(**overrides):
    value = {
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    }
    value.update(overrides)
    return value


def _resolve(checkpoint=None, **overrides):
    options = {
        "expected_observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "runtime_observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "runtime_observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "reliable_execution_enabled": True,
        "telemetry_observation_enabled": False,
        "telemetry_fallback_enabled": False,
        "state_depth_exact_endpoint_binding": True,
        "snapshot_missing_count": 0,
        "telemetry_lookup_count": 0,
        "allow_override": False,
    }
    options.update(overrides)
    return resolve_evaluation_observation_provenance(
        _checkpoint() if checkpoint is None else checkpoint,
        **options,
    )


@pytest.mark.unit
def test_evaluator_rejects_missing_checkpoint_observation_contract():
    checkpoint = _checkpoint()
    checkpoint.pop("observation_contract")
    with pytest.raises(ValueError, match="missing observation contract"):
        _resolve(checkpoint)


@pytest.mark.unit
def test_evaluator_rejects_missing_checkpoint_observation_source():
    checkpoint = _checkpoint()
    checkpoint.pop("observation_source")
    with pytest.raises(ValueError, match="missing observation source"):
        _resolve(checkpoint)


@pytest.mark.unit
def test_evaluator_rejects_checkpoint_contract_mismatch():
    with pytest.raises(ValueError, match="checkpoint observation contract mismatch"):
        _resolve(
            _checkpoint(
                observation_contract=LEGACY_ASYNC_OBSERVATION_CONTRACT,
                observation_source=LEGACY_ASYNC_OBSERVATION_CONTRACT,
            )
        )


@pytest.mark.unit
def test_evaluator_rejects_checkpoint_runtime_source_mismatch():
    with pytest.raises(
        ValueError, match="runtime observation contract/source mismatch"
    ):
        _resolve(runtime_observation_source=LEGACY_ASYNC_OBSERVATION_CONTRACT)


@pytest.mark.unit
def test_evaluator_rejects_legacy_runtime_for_reliable_checkpoint():
    with pytest.raises(ValueError, match="reliable exact endpoint"):
        _resolve(
            runtime_observation_contract=LEGACY_ASYNC_OBSERVATION_CONTRACT,
            runtime_observation_source=LEGACY_ASYNC_OBSERVATION_CONTRACT,
            reliable_execution_enabled=False,
            telemetry_observation_enabled=True,
            telemetry_fallback_enabled=True,
            state_depth_exact_endpoint_binding=False,
        )


@pytest.mark.unit
def test_evaluator_accepts_valid_reliable_exact_fixture():
    provenance = _resolve()
    assert (
        provenance["checkpoint_observation_contract"]
        == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    )
    assert (
        provenance["runtime_observation_contract"]
        == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    )
    assert provenance["runtime_contract_override"] is False


@pytest.mark.unit
def test_evaluator_requires_explicit_override_for_diagnostic_legacy_runtime():
    provenance = _resolve(
        runtime_observation_contract=LEGACY_ASYNC_OBSERVATION_CONTRACT,
        runtime_observation_source=LEGACY_ASYNC_OBSERVATION_CONTRACT,
        reliable_execution_enabled=False,
        telemetry_observation_enabled=True,
        telemetry_fallback_enabled=True,
        state_depth_exact_endpoint_binding=False,
        allow_override=True,
    )
    assert provenance["runtime_contract_override"] is True


@pytest.mark.unit
def test_evaluation_summary_contains_checkpoint_and_runtime_provenance():
    summary = build_evaluation_provenance_summary(
        _resolve(),
        runtime_contract_override=False,
    )
    assert summary["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    assert summary["observation_source"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    assert (
        summary["checkpoint_observation_contract"]
        == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    )
    assert (
        summary["checkpoint_observation_source"]
        == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    )
    assert (
        summary["expected_observation_contract"]
        == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    )
    assert (
        summary["runtime_observation_contract"]
        == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    )
    assert (
        summary["runtime_observation_source"]
        == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    )
    assert summary["runtime_contract_override"] is False


@pytest.mark.unit
def test_reliable_runtime_summary_counters_are_explicitly_zero():
    summary = build_evaluation_provenance_summary(
        _resolve(), runtime_contract_override=False
    )
    assert summary["snapshot_missing_count"] == 0
    assert summary["telemetry_lookup_count"] == 0
    assert summary["telemetry_observation"] is False
    assert summary["telemetry_fallback_enabled"] is False


@pytest.mark.unit
def test_managed_handoff_passes_formal_provenance_and_reliable_endpoints():
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_policy_unity_managed.sh"
    source = script.read_text(encoding="utf-8")
    assert 'EVAL_RELIABLE_V4="${EVAL_RELIABLE_V4:-1}"' in source
    assert 'EVAL_EXPECTED_OBSERVATION_CONTRACT="${EVAL_EXPECTED_OBSERVATION_CONTRACT:-reliable_exact_endpoint_snapshot}"' in source
    assert "--expected-observation-contract" in source
    assert "--reliable-v4-command-endpoint" in source
    assert "--reliable-v4-result-endpoint" in source
    assert "--reliable-v4-snapshot-endpoint" in source
