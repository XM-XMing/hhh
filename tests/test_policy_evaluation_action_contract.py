"""Formal Unity evaluation action-selection contract tests."""

from __future__ import annotations

import inspect

import pytest

from planning.evaluation.policy_evaluator import (
    main,
    resolve_policy_action_contract,
)


@pytest.mark.unit
def test_formal_policy_action_contract_is_deterministic():
    assert resolve_policy_action_contract(0.0, audit_only=False) == (
        "deterministic_argmax",
        0.0,
    )
    with pytest.raises(ValueError, match="formal policy evaluation"):
        resolve_policy_action_contract(0.1, audit_only=False)


@pytest.mark.unit
def test_stochastic_action_selection_requires_explicit_audit():
    assert resolve_policy_action_contract(0.5, audit_only=True) == (
        "masked_categorical",
        0.5,
    )
    for invalid in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and >= 0"):
            resolve_policy_action_contract(invalid, audit_only=True)


@pytest.mark.unit
def test_evaluation_summary_persists_resolved_action_contract():
    source = inspect.getsource(main)
    assert '"policy_action_mode": policy_action_mode' in source
    assert '"policy_temperature": policy_temperature' in source
    assert 'policy_action_mode == "deterministic_argmax"' in source

