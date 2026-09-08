from __future__ import annotations

import json
from pathlib import Path

from planning.diagnostics.hierarchical_task_formulation import (
    _action_cluster_labels,
    _build_action_clusters,
    _build_failure_analysis,
    _finite_stats,
)


def _action(action_id: int, lateral: str = "straight", vertical: str = "level") -> dict:
    return {
        "id": action_id,
        "horizontal_index": 0,
        "vertical_index": 0,
        "lateral_mode": lateral,
        "vertical_mode": vertical,
        "lateral_endpoint_m": 0.0 if lateral == "straight" else (0.2 if lateral == "right" else -0.2),
        "vertical_endpoint_m": 0.0 if vertical == "level" else (0.08 if vertical == "up" else -0.08),
        "terminal_heading_deg": 0.0 if lateral == "straight" else (10.0 if lateral == "right" else -10.0),
        "duration_s": 0.5,
        "command_frames": 25,
        "max_horizontal_speed_mps": 3.0,
        "max_vertical_speed_mps": 0.2,
        "max_horizontal_acceleration_mps2": 2.0,
        "max_vertical_acceleration_mps2": 1.0,
        "max_yaw_rate_radps": 0.5,
    }


def test_action_cluster_labels_keep_high_level_semantics_separate_from_id():
    labels = _action_cluster_labels(_action(1, "left", "up"))
    assert labels["primitive_family"] == "fixed_duration_endpoint_primitive"
    assert labels["direction_intent"] == "left_up"
    assert labels["duration_family"] == "duration_0.500s_frames_25"


def test_finite_stats_is_empty_safe_and_deterministic():
    assert _finite_stats([])["count"] == 0
    assert _finite_stats([1.0, 3.0])["median"] == 2.0


def test_action_cluster_summary_is_explicitly_diagnostic():
    actions = [_action(0), _action(1, "right", "down")]
    library = {
        "definitions": {
            0: {"primitive_id": "mpl_action_000", "neighbors_4": [1], "neighbors_8": [1]},
            1: {"primitive_id": "mpl_action_001", "neighbors_4": [0], "neighbors_8": [0]},
        },
        "actions": actions,
        "json_path": "/tmp/mpl.json",
        "npz_path": "/tmp/mpl.npz",
        "json_sha256": "json",
        "npz_sha256": "npz",
        "meta": {
            "contract_sha256": "contract",
            "num_horizontal": 1,
            "num_vertical": 2,
            "control_dt_s": 0.02,
            "center_action_id": 0,
        },
    }
    # The production builder requires 105 rows; this unit test only checks the
    # semantic builder's output shape by filling the library explicitly.
    library["actions"] = [
        dict(actions[index % 2], id=index, horizontal_index=index % 15, vertical_index=index % 7)
        for index in range(105)
    ]
    for index, action in enumerate(library["actions"]):
        library["definitions"][index] = {"primitive_id": "mpl_action_{:03d}".format(index), "neighbors_4": [], "neighbors_8": []}
    payload = _build_action_clusters(library)
    assert payload["diagnostic_only"] is True
    assert payload["summary"]["action_count"] == 105


def test_failure_analysis_does_not_promote_candidates_to_proof():
    episode = {
        "outcome": "dead_end",
        "terminal_abort_row_count": 0,
        "terminal_abort_reasons": [],
        "terminal_primitive_completed": True,
        "primitive_incomplete_row_count": 0,
        "effective_integration_ticks_available": False,
        "applied_frame_count_available": False,
        "episode_id": "1",
        "mission_id": "m",
        "steps": 1,
        "stop_reason": "dead_end",
        "terminal_action_id": 52,
        "terminal_direction_intent": "straight_level",
    }
    report = _build_failure_analysis([episode], {"episode_count": 1})
    assert report["high_level_decision_error"]["status"] == "INDIRECT_CANDIDATE_ONLY"
    assert report["low_level_control_error"]["status"] == "NOT_ESTABLISHED"
    assert report["unresolved_episode_count"] == 1
