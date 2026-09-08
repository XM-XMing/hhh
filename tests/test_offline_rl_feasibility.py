from __future__ import annotations

from planning.diagnostics.offline_rl_feasibility import (
    _conclusion,
    _descending_rank,
    _rank_percentile,
    _state_rows,
)


def _row(state: str, action: int, source: str, value: float, success: int) -> dict:
    return {
        "state_id": state,
        "episode_id": "e-" + state,
        "mission_id": "m-" + state,
        "step_id": 0,
        "action": action,
        "action_source": source,
        "episode_return": value,
        "success": success,
    }


def test_descending_rank_and_percentile_are_tie_safe():
    assert _descending_rank([10.0, 5.0, 5.0], 10.0) == 1.0
    assert _descending_rank([10.0, 5.0, 5.0], 5.0) == 2.5
    assert _rank_percentile(1.0, 3) == 1.0
    assert _rank_percentile(3.0, 3) == 0.0


def test_state_rows_select_only_real_branches_and_keep_bc_baseline():
    rows = _state_rows(
        [
            _row("s", 3, "BC", 1.0, 0),
            _row("s", 4, "RANDOM", 5.0, 1),
            _row("s", 5, "TEACHER", 5.0, 1),
        ]
    )
    assert len(rows) == 1
    assert rows[0]["bc_action"] == 3
    assert rows[0]["oracle_action"] == 4  # deterministic lowest-ID tie representative
    assert rows[0]["oracle_return_delta"] == 4.0
    assert rows[0]["oracle_tie_count"] == 2
    assert rows[0]["oracle_action_source"] == "RANDOM"


def test_conclusion_does_not_call_oracle_a_policy():
    rows = [
        {"oracle_return_delta": 2.0, "oracle_success_delta": 1.0}
        for _ in range(50)
    ]
    bootstrap = {
        "return_delta": {"ci95": [1.0, 3.0]},
        "success_delta": {"ci95": [0.5, 1.0]},
    }
    result = _conclusion(rows, bootstrap)
    assert result["result"] == "A_DATA_HAS_POLICY_IMPROVEMENT_SPACE"
    assert result["oracle_is_not_a_policy"] is True

