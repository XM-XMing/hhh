from __future__ import annotations

import numpy as np
import pytest

from planning.diagnostics.reward_provenance_audit import (
    build_reward_component_breakdown,
    compute_episode_returns,
    validate_episode_rows,
)


def test_compute_episode_returns_stops_at_terminal_without_bootstrap():
    rows = [
        {"step": 0, "reward": 1.0, "done": False},
        {"step": 1, "reward": 2.0, "done": False},
        {"step": 2, "reward": 3.0, "done": True},
    ]

    result = compute_episode_returns(rows, gamma=0.99, horizons=(5, 10, 20))

    assert [item["undiscounted_return"] for item in result] == [6.0, 5.0, 3.0]
    assert np.isclose(result[0]["discounted_return"], 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0)
    assert result[0]["return_5"] == 6.0
    assert result[1]["return_5"] == 5.0
    assert result[2]["return_20"] == 3.0
    assert all(item["bootstrap_status"] == "NOT_USED_TERMINAL_OR_EPISODE_END" for item in result)


def test_component_breakdown_keeps_unobserved_terms_explicit():
    row = {
        "distance_before": 10.0,
        "distance_after": 9.0,
        "prev_action": 1,
        "action": 2,
        "success": True,
        "collision": False,
        "dead_end": False,
        "timeout": False,
        "far": False,
        "hard_altitude": False,
        "reward": 32.0,
    }

    result = build_reward_component_breakdown(row)

    assert result["xy_progress"]["status"] == "EXACT_FROM_STEP_ARTIFACT"
    assert result["xy_progress"]["value"] == 2.0
    assert result["terminal_success"]["value"] == 30.0
    assert result["step"]["value"] == -0.02
    assert result["action_change"]["value"] == -0.02
    assert result["goal_z_progress"]["status"] == "UNAVAILABLE"
    assert result["clearance"]["status"] == "UNAVAILABLE"


def test_validate_episode_rows_requires_contiguous_steps_and_terminal_tail():
    valid = [
        {"step": 0, "reward": 1.0, "done": False},
        {"step": 1, "reward": -2.0, "done": True},
    ]
    validate_episode_rows(valid)

    with pytest.raises(ValueError, match="contiguous"):
        validate_episode_rows([valid[0], {"step": 2, "reward": 1.0, "done": True}])

    with pytest.raises(ValueError, match="terminal row"):
        validate_episode_rows(
            [
                {"step": 0, "reward": -2.0, "done": True},
                {"step": 1, "reward": 1.0, "done": False},
            ]
        )
