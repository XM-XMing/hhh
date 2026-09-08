"""Deterministic contracts for diagnostic-only shadow policy evidence."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_shadow_trace_reports_divergence_without_changing_primary_action():
    from planning.evaluation.policy_evaluator import build_shadow_trace_record

    primary = {
        "action_probabilities": [0.70, 0.20, 0.10],
        "actor_top_k": [
            {"action": 0, "logit": 3.0},
            {"action": 1, "logit": 2.0},
        ],
    }
    shadow = {
        "action_probabilities": [0.10, 0.80, 0.10],
        "actor_top_k": [
            {"action": 1, "logit": 4.0},
            {"action": 0, "logit": 1.0},
        ],
    }
    record = build_shadow_trace_record(
        mission_id="mission",
        episode_id=7,
        step=3,
        previous_action=-1,
        valid_action_count=3,
        primary_action=0,
        shadow_action=1,
        primary_label="bc",
        shadow_label="awac",
        primary_diagnostics=primary,
        shadow_diagnostics=shadow,
    )

    assert record["primary_action"] == 0
    assert record["shadow_action"] == 1
    assert record["primary_top1_action"] == 0
    assert record["shadow_top1_action"] == 1
    assert record["top1_disagreement"] is True
    assert record["selected_action_disagreement"] is True
    assert record["bc_to_awac_kl"] == pytest.approx(
        0.7 * np.log(0.7 / 0.1) + 0.2 * np.log(0.2 / 0.8)
    )
    assert record["js_divergence"] > 0.0


@pytest.mark.unit
def test_shadow_trace_helper_requires_probability_provenance():
    from planning.evaluation.policy_evaluator import build_shadow_trace_record

    diagnostics = {"actor_top_k": [{"action": 0, "logit": 1.0}]}
    with pytest.raises(ValueError, match="full action probabilities"):
        build_shadow_trace_record(
            mission_id="mission",
            episode_id=1,
            step=0,
            previous_action=-1,
            valid_action_count=1,
            primary_action=0,
            shadow_action=0,
            primary_label="bc",
            shadow_label="awac",
            primary_diagnostics=diagnostics,
            shadow_diagnostics=diagnostics,
        )


@pytest.mark.unit
def test_default_choose_action_return_contract_remains_three_values(monkeypatch):
    import planning.evaluation.policy_evaluator as evaluator
    torch, _, _, _, _ = evaluator.require_torch()

    monkeypatch.setattr(
        evaluator,
        "observation_tensors",
        lambda *args, **kwargs: (
            torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            torch.zeros((1, 3), dtype=torch.float32),
        ),
    )

    class Model:
        def __call__(self, depth, vector):
            logits = torch.zeros((1, 105), dtype=torch.float32)
            logits[0, 9] = 2.0
            return logits

    class Normalizer:
        pass

    result = evaluator.choose_action(
        Model(),
        {},
        -1,
        np.ones((105,), dtype=np.bool_),
        Normalizer(),
        torch,
        torch.device("cpu"),
        0.0,
    )
    assert len(result) == 3
    assert result[0] == 9
