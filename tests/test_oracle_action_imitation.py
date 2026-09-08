from __future__ import annotations

import numpy as np

from planning.diagnostics.oracle_action_imitation import (
    _masked_order,
    _weighted_return_advantage,
    build_oracle_labels,
)


def _row(state: str, action: int, source: str, value: float, success: int = 0) -> dict:
    return {
        "state_id": state,
        "mission_id": "mission-" + state,
        "episode_id": "episode-" + state + "-" + str(action),
        "step_id": 3,
        "action": action,
        "action_source": source,
        "episode_return": value,
        "success": success,
    }


def test_oracle_uses_only_observed_actions_and_resolves_ties_by_action_id():
    labels = build_oracle_labels(
        [
            _row("s", 8, "BC", 1.0),
            _row("s", 9, "RANDOM", 4.0, 1),
            _row("s", 7, "TEACHER", 4.0, 1),
        ]
    )
    assert len(labels) == 1
    assert labels[0]["oracle_action"] == 7
    assert labels[0]["oracle_action_source"] == "TEACHER"
    assert labels[0]["observed_actions"] == [7, 8, 9]
    assert labels[0]["return_advantage"] == 3.0


def test_weight_is_proportional_to_return_advantage_and_mean_normalized():
    labels = [
        {"return_advantage": 0.0},
        {"return_advantage": 2.0},
        {"return_advantage": 4.0},
    ]
    weights = _weighted_return_advantage(labels, np.asarray([0, 1, 2], dtype=np.int64))
    assert np.isclose(weights.mean(), 1.0)
    assert weights[0] == 0.0
    assert weights[2] == 2.0 * weights[1]


def test_masked_order_never_returns_an_invalid_action():
    logits = np.asarray([9.0, 8.0, 7.0], dtype=np.float32)
    order = _masked_order(logits, np.asarray([False, True, False]))
    assert int(order[0]) == 1
