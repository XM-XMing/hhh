"""Regression contracts for value-return measurement and calibration gating."""

from __future__ import annotations

import pytest

from planning.awac.calibration import (
    CALIBRATION_GATE_CONTRACT_ID,
    HOLDOUT_RETURN_SEMANTICS,
    CriticCalibrationConfig,
    HoldoutEpisodeValidationError,
    episode_monte_carlo_returns,
    evaluate_calibration_gate,
    spearman_rank_correlation,
)


pytestmark = pytest.mark.unit


def _record(
    mission_id: str,
    episode_id: str,
    index: int,
    reward: float,
    *,
    done: bool,
    terminal_reason: str = "",
    **extra,
):
    return {
        "mission_id": mission_id,
        "episode_id": episode_id,
        "episode_transition_index": index,
        "reward": reward,
        "done": done,
        "terminal_reason": terminal_reason,
        **extra,
    }


def _window(*, rho=0.6, policy_alignment="MATCH", td_loss=1.0, disagreement=0.1):
    return {
        "finite": True,
        "holdout_measurement_status": "PASS",
        "holdout_return_semantics": HOLDOUT_RETURN_SEMANTICS,
        "calibration_gate_contract_id": CALIBRATION_GATE_CONTRACT_ID,
        "value_policy_alignment": policy_alignment,
        "critic1_td_loss": td_loss,
        "critic2_td_loss": td_loss,
        "holdout_td_loss": td_loss,
        "q_mean": 0.0,
        "q_std": 1.0,
        "q_min": -1.0,
        "q_max": 1.0,
        "q_p99": 1.0,
        "normalized_twin_disagreement": disagreement,
        "normalized_twin_disagreement_p95": disagreement,
        "q_explosion_bound": 20.0,
        "success_sample_count": 2,
        "failure_sample_count": 2,
        "value_ordering_status": "PASS",
        "holdout_q_return_rank_correlation": rho,
    }


def _gate(windows):
    return evaluate_calibration_gate(
        windows=windows,
        replay_transitions=4,
        completed_episodes=2,
        critic_updates=3,
        holdout_episodes=2,
        config=CriticCalibrationConfig(
            min_replay_transitions=4,
            min_completed_episodes=2,
            min_critic_updates=3,
            min_holdout_episodes=2,
            min_stability_windows=3,
        ),
    )


def test_episode_monte_carlo_return_scales_each_reward_once():
    result = episode_monte_carlo_returns(
        [
            _record("m1", "e1", 0, 1.0, done=False),
            _record("m1", "e1", 1, 1.0, done=False),
            _record("m1", "e1", 2, -10.0, done=True, terminal_reason="collision"),
        ],
        gamma=0.99,
        reward_scale=0.1,
    )

    assert result["returns"] == pytest.approx([-0.7811, -0.89, -1.0])
    assert result["return_semantics"] == HOLDOUT_RETURN_SEMANTICS
    assert result["complete_episode_count"] == 1
    assert result["usable_row_count"] == 3


def test_episode_monte_carlo_return_handles_interleaved_workers_and_restores_row_order():
    result = episode_monte_carlo_returns(
        [
            _record("m1", "e1", 0, 1.0, done=False),
            _record("m2", "e2", 0, -2.0, done=False),
            _record("m1", "e1", 1, 10.0, done=True, terminal_reason="success"),
            _record("m2", "e2", 1, 4.0, done=True, terminal_reason="dead_end"),
        ],
        gamma=0.5,
        reward_scale=1.0,
    )

    assert result["returns"] == pytest.approx([6.0, 0.0, 10.0, 4.0])
    assert result["complete_episode_count"] == 2
    assert result["episode_keys"] == [["m1", "e1"], ["m2", "e2"]]


@pytest.mark.parametrize(
    "records, reason",
    [
        (
            [
                _record("m", "e", 0, 1.0, done=False),
                _record("m", "e", 2, 1.0, done=True, terminal_reason="success"),
            ],
            "episode_transition_index_not_contiguous",
        ),
        (
            [
                _record("m", "e", 0, 1.0, done=False),
                _record("m", "e", 0, 1.0, done=True, terminal_reason="success"),
            ],
            "duplicate_episode_transition_index",
        ),
        (
            [
                _record("m", "e", 0, 1.0, done=True, terminal_reason="success"),
                _record("m", "e", 1, 1.0, done=True, terminal_reason="success"),
            ],
            "terminal_before_final_transition",
        ),
        (
            [_record("m", "e", 0, 1.0, done=True, terminal_reason="unknown")],
            "illegal_terminal_reason",
        ),
        (
            [_record("m", "e", 0, 1.0, done=False)],
            "episode_not_terminal",
        ),
        (
            [
                _record(
                    "m",
                    "e",
                    0,
                    1.0,
                    done=True,
                    terminal_reason="success",
                    runtime_abort=True,
                )
            ],
            "runtime_abort",
        ),
    ],
)
def test_episode_monte_carlo_return_rejects_incomplete_or_ambiguous_episodes(records, reason):
    with pytest.raises(HoldoutEpisodeValidationError) as error:
        episode_monte_carlo_returns(records, gamma=0.99, reward_scale=0.1)

    assert error.value.report["invalid_episode_count"] == 1
    assert error.value.report["invalid_episodes"][0]["reason"] == reason


def test_rank_correlation_preserves_undefined_constant_evidence():
    assert spearman_rank_correlation([1.0, 1.0], [1.0, 2.0]) is None
    assert spearman_rank_correlation([1.0, 2.0], [2.0, 1.0]) == pytest.approx(-1.0)


def test_gate_requires_strict_positive_mc_rank_evidence_and_policy_alignment():
    stable = [_window(rho=0.6) for _ in range(3)]
    assert _gate(stable)["state"] == "PASS"

    assert _gate([_window(rho=-1.0) for _ in range(3)])["state"] != "PASS"
    missing = _gate([_window(rho=None) for _ in range(3)])
    assert missing["state"] == "PENDING"
    assert missing["evidence_status"] == "PENDING_EVIDENCE"
    assert missing["hard_divergence_reason"] == ""
    assert _gate([_window(rho=0.4) for _ in range(3)])["state"] != "PASS"

    mismatch = _gate([_window(rho=0.6, policy_alignment="MISMATCH") for _ in range(3)])
    assert mismatch["state"] != "PASS"
    assert mismatch["evidence_status"] == "PENDING_EVIDENCE"


def test_holdout_contract_placeholder_is_not_misclassified_as_numerical_failure():
    placeholder = _window(rho=None)
    placeholder["holdout_measurement_status"] = "CONTRACT_ERROR"

    gate = _gate([placeholder for _ in range(3)])

    assert gate["state"] == "PENDING"
    assert gate["evidence_status"] == "CONTRACT_ERROR"
    assert gate["hard_divergence_reason"] == ""


def test_gate_does_not_call_monotonic_improvement_divergence_and_checks_both_q_extremes():
    decreasing = [
        _window(td_loss=1.0, disagreement=0.1),
        _window(td_loss=0.5, disagreement=0.05),
        _window(td_loss=0.25, disagreement=0.025),
    ]
    improving = _gate(decreasing)
    assert improving["state"] == "PENDING"
    assert improving["sustained_td_worsening"] is False
    assert improving["sustained_disagreement_worsening"] is False

    worsening = _gate(
        [
            _window(td_loss=1.0, disagreement=0.1),
            _window(td_loss=2.0, disagreement=0.2),
            _window(td_loss=4.0, disagreement=0.4),
        ]
    )
    assert worsening["state"] == "FAIL_DIVERGED"

    negative_explosion = _window()
    negative_explosion["q_min"] = -100_000_000.0
    negative_explosion["q_max"] = 3.0
    negative = _gate([negative_explosion])
    assert negative["state"] == "FAIL_DIVERGED"
    assert negative["q_explosion"] is True
