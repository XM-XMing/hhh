"""Phase 0 public-contract tests for AWAC calibration."""

from __future__ import annotations

import math

import pytest

from planning.awac.calibration import (
    CALIBRATION_GATE_CONTRACT_ID,
    CALIBRATION_GATE_CONTRACT_VERSION,
    CriticCalibrationConfig,
    HOLDOUT_RETURN_SEMANTICS,
    evaluate_calibration_gate,
    masked_bellman_target,
    summarize_calibration_window,
)
from planning.awac.contract import (
    AWAC_ALGORITHM_ID,
    REPLAY_FIELDS,
    build_awac_training_contract,
    awac_training_contract_sha256,
)
from planning.awac.interaction import BehaviorSource, behavior_source_contract
from planning.contracts.feature import FEATURE_CONTRACT_ID
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.reward import reward_contract_sha256
from planning.contracts.task import task_contract_sha256


def _finite_calibration_window():
    return summarize_calibration_window(
        critic1_td_loss=1.0,
        critic2_td_loss=1.1,
        holdout_td_loss=1.05,
        q1=[4.0, 2.0, 3.0, 1.0],
        q2=[3.8, 1.8, 3.1, 0.9],
        rewards=[1.0, -1.0, 1.0, -1.0],
        returns=[2.0, -2.0, 2.0, -2.0],
        terminal_reasons=["success", "collision", "success", "timeout"],
        holdout_return_semantics=HOLDOUT_RETURN_SEMANTICS,
        holdout_measurement_status="PASS",
        value_policy_alignment="MATCH",
        complete_episode_count=2,
        usable_row_count=4,
    )


def _quality_worsening_windows():
    base = _finite_calibration_window()
    return [
        dict(base, holdout_td_loss=1.0, normalized_twin_disagreement=0.05),
        dict(base, holdout_td_loss=2.0, normalized_twin_disagreement=0.10),
        dict(base, holdout_td_loss=4.0, normalized_twin_disagreement=0.20),
    ]


@pytest.mark.unit
def test_phase0_training_contract_is_complete_and_provenance_bound():
    contract = build_awac_training_contract(
        phase="critic_calibration",
        gamma=0.99,
        tau=0.005,
        calibration=CriticCalibrationConfig().as_dict(),
    )

    assert contract["schema_version"] == 1
    assert contract["algorithm"] == AWAC_ALGORITHM_ID
    assert contract["phase"] == "critic_calibration"
    assert contract["policy_input_contract"] == "depth_goal_state_prev_action"
    assert contract["policy_input_contract_sha256"]
    assert contract["feature_contract_id"] == FEATURE_CONTRACT_ID
    assert contract["task_contract_sha256"] == task_contract_sha256(45)
    assert contract["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    assert contract["reward_contract_sha256"] == reward_contract_sha256()
    assert contract["reward_scale"] == pytest.approx(0.10)
    assert contract["reward_scale_owner"] == "learner_bellman_target"
    assert tuple(contract["replay_fields"]) == REPLAY_FIELDS
    assert contract["behavior_sources"] == {
        "bc_calibration": int(BehaviorSource.BC_CALIBRATION),
        "awac_online": int(BehaviorSource.AWAC_ONLINE),
    }
    assert contract["actor_update_enabled"] is False
    assert contract["privileged_policy_inputs"] == []
    assert contract["calibration"]["calibration_gate_contract_id"] == (
        CALIBRATION_GATE_CONTRACT_ID
    )
    assert contract["calibration"]["calibration_gate_contract_version"] == (
        CALIBRATION_GATE_CONTRACT_VERSION
    )
    assert len(awac_training_contract_sha256(contract)) == 64


@pytest.mark.unit
def test_behavior_source_contract_uses_phase0_names_without_losing_numeric_identity():
    assert behavior_source_contract() == {
        "bc_calibration": int(BehaviorSource.BC_CALIBRATION),
        "awac_online": int(BehaviorSource.AWAC_ONLINE),
    }
    assert int(BehaviorSource.BC_WARMUP) == int(BehaviorSource.BC_CALIBRATION)
    assert int(BehaviorSource.ACCEPTED_COLLECTION_POLICY) == int(BehaviorSource.AWAC_ONLINE)


@pytest.mark.unit
def test_masked_bellman_target_scales_reward_once_and_renormalizes_next_policy():
    torch = pytest.importorskip("torch")
    bc_logits = torch.tensor([[0.0, math.log(2.0), math.log(4.0)]])
    q1 = torch.tensor([[1.0, 3.0, 5.0]])
    q2 = torch.tensor([[2.0, 1.0, 4.0]])
    next_mask = torch.tensor([[True, False, True]])
    target = masked_bellman_target(
        bc_logits=bc_logits,
        target_q1=q1,
        target_q2=q2,
        next_action_mask=next_mask,
        reward=torch.tensor([10.0]),
        done=torch.tensor([0.0]),
        gamma=0.5,
        reward_scale=0.10,
        torch=torch,
    )

    # Valid BC masses are [1, 4], hence pi_valid=[.2, 0, .8],
    # Qmin=[1, 1, 4], V=3.4, and scaled target=.1*10+.5*3.4.
    assert torch.allclose(
        target["normalized_probabilities"],
        torch.tensor([[0.2, 0.0, 0.8]]),
        atol=1.0e-6,
    )
    assert target["next_value"].item() == pytest.approx(3.4, abs=1.0e-6)
    assert target["target"].item() == pytest.approx(2.7, abs=1.0e-6)
    assert target["effective_done"].item() == 0.0


@pytest.mark.unit
def test_masked_bellman_target_uses_reward_only_for_terminal_zero_next_mask():
    torch = pytest.importorskip("torch")
    target = masked_bellman_target(
        bc_logits=torch.zeros((2, 3)),
        target_q1=torch.full((2, 3), 9.0),
        target_q2=torch.full((2, 3), 8.0),
        next_action_mask=torch.tensor([[False, False, False], [True, False, False]]),
        reward=torch.tensor([-100.0, 2.0]),
        done=torch.tensor([1.0, 1.0]),
        gamma=0.99,
        reward_scale=0.10,
        torch=torch,
    )
    assert target["empty_next_mask"].tolist() == [True, False]
    assert target["effective_done"].tolist() == [True, True]
    assert target["next_value"].tolist() == pytest.approx([0.0, 0.0])
    assert target["target"].tolist() == pytest.approx([-10.0, 0.2])
    assert torch.equal(
        target["normalized_probabilities"],
        torch.zeros((2, 3), dtype=torch.float32),
    )
    assert torch.isfinite(target["target"]).all()
    assert torch.isfinite(target["next_value"]).all()
    assert torch.isfinite(target["q_min"]).all()


@pytest.mark.unit
def test_masked_bellman_target_rejects_nonterminal_empty_next_mask():
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="non-terminal.*empty"):
        masked_bellman_target(
            bc_logits=torch.zeros((1, 3)),
            target_q1=torch.zeros((1, 3)),
            target_q2=torch.zeros((1, 3)),
            next_action_mask=torch.zeros((1, 3), dtype=torch.bool),
            reward=torch.tensor([1.0]),
            done=torch.tensor([0.0]),
            gamma=0.99,
            reward_scale=0.10,
            torch=torch,
        )


@pytest.mark.unit
def test_calibration_window_and_gate_are_relative_and_fail_closed_on_nonfinite():
    window = _finite_calibration_window()
    assert window["finite"] is True
    assert window["q_p99"] >= window["q_mean"]
    assert window["normalized_twin_disagreement"] >= 0.0
    assert window["success_q_median"] > window["failure_q_median"]
    assert window["success_minus_failure_q_median"] > 0.0
    assert window["holdout_q_return_rank_correlation"] is not None
    assert 0.0 <= window["holdout_q_return_rank_correlation"] <= 1.0

    config = CriticCalibrationConfig(
        min_replay_transitions=4,
        min_completed_episodes=2,
        min_critic_updates=2,
        min_holdout_episodes=2,
        min_stability_windows=3,
    )
    stable = [dict(window, critic_update_count=index + 1) for index in range(3)]
    gate = evaluate_calibration_gate(
        windows=stable,
        replay_transitions=4,
        completed_episodes=2,
        critic_updates=3,
        holdout_episodes=2,
        config=config,
    )
    assert gate["state"] == "PASS"
    assert gate["would_open"] is True
    assert gate["maturity_ready"] is True
    assert gate["gate_state"] == "PASS"
    assert gate["gate_reason"] == "minimums_stable_finite_ordered_and_q_mc_ranked"
    assert gate["hard_divergence_reason"] == ""

    pending = evaluate_calibration_gate(
        windows=stable[:1],
        replay_transitions=4,
        completed_episodes=2,
        critic_updates=1,
        holdout_episodes=1,
        config=config,
    )
    assert pending["state"] == "PENDING"
    assert pending["would_open"] is False

    divergent = dict(window, finite=False, q_p99=float("nan"))
    failed = evaluate_calibration_gate(
        windows=[divergent],
        replay_transitions=4,
        completed_episodes=2,
        critic_updates=3,
        holdout_episodes=2,
        config=config,
    )
    assert failed["state"] == "FAIL_DIVERGED"
    assert failed["would_open"] is False
    assert failed["hard_divergence_reason"] == "nonfinite_calibration_metrics"


@pytest.mark.unit
def test_calibration_gate_immature_finite_quality_worsening_stays_pending():
    config = CriticCalibrationConfig()
    windows = _quality_worsening_windows()

    zero_update = evaluate_calibration_gate(
        windows=windows,
        replay_transitions=95,
        completed_episodes=5,
        critic_updates=0,
        holdout_episodes=2,
        config=config,
    )
    assert zero_update["state"] == "PENDING"
    assert zero_update["gate_reason"] == "insufficient_data"
    assert zero_update["maturity_ready"] is False
    assert zero_update["hard_divergence_reason"] == ""

    boundary = evaluate_calibration_gate(
        windows=windows,
        replay_transitions=127,
        completed_episodes=15,
        critic_updates=31,
        holdout_episodes=7,
        config=config,
    )
    assert boundary["state"] == "PENDING"
    assert boundary["maturity_ready"] is False


@pytest.mark.unit
def test_calibration_gate_requires_stability_windows_after_count_maturity():
    gate = evaluate_calibration_gate(
        windows=[_finite_calibration_window(), _finite_calibration_window()],
        replay_transitions=128,
        completed_episodes=16,
        critic_updates=32,
        holdout_episodes=8,
        config=CriticCalibrationConfig(),
    )

    assert gate["state"] == "PENDING"
    assert gate["maturity_ready"] is True
    assert gate["stability_windows"] == 2
    assert gate["hard_divergence_reason"] == ""


@pytest.mark.unit
def test_calibration_gate_preserves_mature_quality_failure_semantics():
    gate = evaluate_calibration_gate(
        windows=_quality_worsening_windows(),
        replay_transitions=128,
        completed_episodes=16,
        critic_updates=32,
        holdout_episodes=8,
        config=CriticCalibrationConfig(),
    )

    assert gate["state"] == "FAIL_DIVERGED"
    assert gate["reason"] == "sustained_relative_calibration_worsening"
    assert gate["maturity_ready"] is True
    assert gate["hard_divergence_reason"] == ""


@pytest.mark.unit
def test_calibration_gate_keeps_q_explosion_hard_before_maturity():
    window = _finite_calibration_window()
    window["q_max"] = float(window["q_explosion_bound"]) + 1.0

    gate = evaluate_calibration_gate(
        windows=[window],
        replay_transitions=95,
        completed_episodes=5,
        critic_updates=0,
        holdout_episodes=2,
        config=CriticCalibrationConfig(),
    )

    assert gate["state"] == "FAIL_DIVERGED"
    assert gate["maturity_ready"] is False
    assert (
        gate["hard_divergence_reason"]
        == "q_explosion_relative_to_observed_return_scale"
    )
