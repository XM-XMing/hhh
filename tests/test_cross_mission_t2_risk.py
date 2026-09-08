from __future__ import annotations

import copy

from planning.diagnostics.cross_mission_t2_risk import (
    build_risk_table,
    select_risk_enriched_missions,
    summarize_cross_mission_screen,
    summarize_mission_repeats,
)


def _trace(*, depth="depth", observation="observation", logits="logits", mask="mask", action=4, after="after"):
    primitive = {
        "step": 0,
        "before": {
            "state_fingerprint": "state",
            "depth_fingerprint": depth,
            "observation_fingerprint": observation,
            "state_sequence": 10,
            "state_id": 10,
            "state_timestamp_ns": 100,
            "depth_sequence": 20,
            "depth_timestamp_ns": 90,
            "sensor_skew_ns": 10,
        },
        "after": {"observation_fingerprint": after},
        "actor_logits_fingerprint": logits,
        "action_mask_fingerprint": mask,
        "action_mask": [True, True],
        "selected_action": action,
        "terminal_reward_input_fingerprint": "terminal",
        "valid_action_count": 2,
        "top1_top2_margin": 0.1,
        "depth_mask_boundary_margin_m": 0.02,
        "minimum_safety_clearance_m": 0.4,
        "primitive_execution": {
            "frame_count": 25,
            "status": "complete",
            "endpoint_state_id": 124,
            "applied_frames": [
                {"frame_index": index, "applied_state_id": 100 + index, "command_id": 200 + index}
                for index in range(25)
            ],
        },
    }
    return {
        "checkpoint_sha256": "checkpoint",
        "mission_id": "mission",
        "episode_id": 1,
        "outcome": "success",
        "steps": 1,
        "primitives": [primitive],
    }


def _rollout():
    return {
        "steps": "1", "success": "True", "collision": "False", "dead_end": "False",
        "timeout": "False", "far": "False", "hard_altitude": "False", "stop_reason": "success",
        "final_x": "1.0", "final_y": "2.0", "final_z": "1.5", "final_distance_xy": "0.2",
    }


def _summary(right):
    mission = {"episode_id": 1, "mission_id": "mission", "baseline_outcome": "success"}
    return summarize_mission_repeats(
        mission=mission,
        repeats=[
            {"repeat": "repeat_001", "trace": _trace(), "rollout": _rollout()},
            {"repeat": "repeat_002", "trace": right, "rollout": _rollout()},
        ],
    )


def test_mission_classification_orders_content_mask_action_and_trajectory():
    assert _summary(_trace())["classification"] == "M0"
    assert _summary(_trace(depth="other", observation="other", logits="other"))["classification"] == "M1"
    assert _summary(_trace(mask="other"))["classification"] == "M2"
    assert _summary(_trace(action=5))["classification"] == "M3"
    assert _summary(_trace(after="other"))["classification"] == "M4"


def test_raw_risk_selection_is_fixed_coverage_without_weighted_score():
    traces = []
    rollouts = {}
    counts = {}
    outcomes = ("success", "collision", "dead_end", "timeout")
    for episode_id in range(1, 25):
        trace = _trace()
        trace["episode_id"] = episode_id
        trace["mission_id"] = "mission-{}".format(episode_id)
        trace["primitives"][0]["top1_top2_margin"] = float(episode_id) / 100.0
        trace["primitives"][0]["depth_mask_boundary_margin_m"] = float(25 - episode_id) / 100.0
        trace["primitives"][0]["before"]["sensor_skew_ns"] = episode_id * 1_000_000
        traces.append(trace)
        row = _rollout()
        for key in ("success", "collision", "dead_end", "timeout"):
            row[key] = "False"
        row[outcomes[(episode_id - 1) % len(outcomes)]] = "True"
        rollouts[episode_id] = row
        counts[episode_id] = [{"boundary_kind": "reset", "admissible_pair_count": episode_id}]
    table = build_risk_table(traces, rollouts, counts)
    selection = select_risk_enriched_missions(table, count=20)
    assert len(selection) == 20
    assert len({row["episode_id"] for row in selection}) == 20
    assert {"success", "collision", "dead_end", "timeout"}.issubset(
        {row["baseline_outcome"] for row in selection}
    )
    assert all(row["selection_reasons"] for row in selection)


def test_cross_mission_t1_blocks_layer3_without_iid_claim():
    m0 = _summary(_trace())
    m3 = _summary(_trace(action=5))
    report = summarize_cross_mission_screen([m0, m3])
    assert report["classification"] == "T1"
    assert not report["layer3_allowed"]
    assert report["mission_classification_counts"]["M3"] == 1
    assert "non-homogeneous" in report["rule_of_three_note"]
