from __future__ import annotations

import numpy as np

from planning.diagnostics.reward_horizon_audit import (
    build_mission_level_split,
    reward_horizon_availability,
    within_state_outcome_statistics,
)


def _row(state_id: str, mission_id: str, source: str, reward: float, outcome: float, success: bool) -> dict:
    return {
        "state_id": state_id,
        "mission_id": mission_id,
        "action": 0,
        "action_source": source,
        "step_id": 0,
        "immediate_reward": reward,
        "terminal_return": outcome,
        "success": success,
        "terminal_reason": "success" if success else "dead_end",
    }


def test_reward_horizon_reports_missing_sequence_without_fabricating_targets():
    rows = [_row("s0", "m0", "BC", 1.0, 5.0, True)]
    result = reward_horizon_availability(rows, has_complete_step_reward_sequence=False)

    assert result["exact_future_return_available"] is False
    assert result["exact_one_step_return_available"] is True
    assert result["requested_horizons"]["1"]["status"] == "AVAILABLE"
    assert result["requested_horizons"]["5"]["status"] == "UNAVAILABLE"
    assert result["requested_horizons"]["10"]["status"] == "UNAVAILABLE"
    assert result["requested_horizons"]["20"]["status"] == "UNAVAILABLE"
    assert result["observed_action_reward"]["nonzero_count"] == 1
    assert result["terminal_branch_outcome"]["includes_prefix_return"] is True


def test_within_state_outcome_statistics_preserves_source_and_state_clusters():
    rows = [
        _row("s0", "m0", "BC", 1.0, 1.0, True),
        _row("s0", "m0", "NEIGHBOR", -1.0, 3.0, True),
        _row("s0", "m0", "RANDOM", -2.0, -2.0, False),
        _row("s1", "m1", "BC", 0.0, 4.0, True),
    ]
    result = within_state_outcome_statistics(rows)

    assert result["state_count"] == 2
    assert result["multi_action_state_count"] == 1
    assert result["sources"]["BC"]["row_count"] == 2
    assert result["sources"]["NEIGHBOR"]["success_rate"] == 1.0
    assert np.isclose(result["per_state"]["s0"]["terminal_return_variance"], 38.0 / 9.0)


def test_mission_split_assigns_each_mission_to_exactly_one_fold():
    split = build_mission_level_split(["m0", "m1", "m2", "m3", "m4"], seed=20260910, fold_count=3)

    assert split["mission_count"] == 5
    assert set(split["mission_to_fold"]) == {"m0", "m1", "m2", "m3", "m4"}
    assert all(0 <= fold < 3 for fold in split["mission_to_fold"].values())
    assert sum(split["fold_loads"]) == 5
