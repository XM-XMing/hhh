from __future__ import annotations

from planning.diagnostics.pairing_counterfactual import (
    enumerate_admissible_pairs,
    summarize_counterfactual_steps,
)


def _state(sequence: int, timestamp_ns: int, fingerprint: str = "state"):
    return {
        "state_sequence": sequence,
        "state_timestamp_ns": timestamp_ns,
        "state_fingerprint": fingerprint,
    }


def _depth(sequence: int, timestamp_ns: int, fingerprint: str = "depth"):
    return {
        "depth_sequence": sequence,
        "depth_timestamp_ns": timestamp_ns,
        "depth_fingerprint": fingerprint,
    }


def test_reset_admissibility_uses_freshness_and_sync_window():
    pairs = enumerate_admissible_pairs(
        boundary_kind="reset",
        states=[
            _state(10, 1_000_000_000),
            _state(11, 1_020_000_000),
            _state(12, 1_200_000_000),
        ],
        depths=[
            _depth(20, 990_000_000),
            _depth(21, 1_040_000_000),
            _depth(22, 1_300_000_000),
        ],
        previous_state_sequence=10,
        previous_depth_sequence=20,
        max_sensor_skew_ns=80_000_000,
    )

    assert [
        (pair["state_sequence"], pair["depth_sequence"])
        for pair in pairs
    ] == [(11, 21)]


def test_endpoint_admissibility_fixes_state_and_requires_post_endpoint_depth():
    pairs = enumerate_admissible_pairs(
        boundary_kind="primitive_endpoint",
        states=[
            _state(31, 2_000_000_000, "endpoint"),
            _state(32, 2_020_000_000, "later"),
        ],
        depths=[
            _depth(40, 1_990_000_000, "too-old-sequence"),
            _depth(41, 1_999_000_000, "before-endpoint"),
            _depth(42, 2_010_000_000, "first"),
            _depth(43, 2_090_000_000, "outside-window"),
        ],
        previous_state_sequence=0,
        previous_depth_sequence=40,
        max_sensor_skew_ns=80_000_000,
        required_state_sequence=31,
        required_state_timestamp_ns=2_000_000_000,
    )

    assert [
        (pair["state_sequence"], pair["depth_sequence"])
        for pair in pairs
    ] == [(31, 42)]


def test_summary_distinguishes_content_logits_mask_action_and_terminal_changes():
    summary = summarize_counterfactual_steps(
        [
            {
                "step": 0,
                "pairs": [
                    {
                        "observation_fingerprint": "o0",
                        "depth_fingerprint": "d0",
                        "mask_fingerprint": "m0",
                        "logits_fingerprint": "l0",
                        "top1_action": 52,
                        "top1_logit": 3.0,
                        "top2_logit": 2.5,
                        "valid_action_count": 100,
                        "terminal_reward_input_fingerprint": "t0",
                        "sensor_skew_ns": 10_000_000,
                    },
                    {
                        "observation_fingerprint": "o1",
                        "depth_fingerprint": "d1",
                        "mask_fingerprint": "m0",
                        "logits_fingerprint": "l1",
                        "top1_action": 52,
                        "top1_logit": 2.9,
                        "top2_logit": 2.6,
                        "valid_action_count": 100,
                        "terminal_reward_input_fingerprint": "t0",
                        "sensor_skew_ns": 20_000_000,
                        "logits_linf_from_selected": 0.1,
                        "mask_hamming_from_selected": 0,
                    },
                ],
            },
            {
                "step": 1,
                "pairs": [
                    {
                        "observation_fingerprint": "o2",
                        "depth_fingerprint": "d2",
                        "mask_fingerprint": "m2",
                        "logits_fingerprint": "l2",
                        "top1_action": 10,
                        "top1_logit": 1.0,
                        "top2_logit": 0.95,
                        "valid_action_count": 99,
                        "terminal_reward_input_fingerprint": "t2",
                        "sensor_skew_ns": 70_000_000,
                    },
                    {
                        "observation_fingerprint": "o3",
                        "depth_fingerprint": "d3",
                        "mask_fingerprint": "m3",
                        "logits_fingerprint": "l3",
                        "top1_action": 11,
                        "top1_logit": 1.1,
                        "top2_logit": 1.0,
                        "valid_action_count": 98,
                        "terminal_reward_input_fingerprint": "t3",
                        "sensor_skew_ns": 75_000_000,
                        "logits_linf_from_selected": 0.25,
                        "mask_hamming_from_selected": 3,
                    },
                ],
            },
        ],
        max_sensor_skew_ns=80_000_000,
    )

    assert summary["policy_decision_count"] == 2
    assert summary["total_admissible_pairs"] == 4
    assert summary["steps_with_multiple_observation_fingerprints"] == [0, 1]
    assert summary["steps_with_multiple_masks"] == [1]
    assert summary["steps_with_multiple_logits_fingerprints"] == [0, 1]
    assert summary["steps_with_multiple_top1_actions"] == [1]
    assert summary["steps_with_terminal_reward_input_changes"] == [1]
    assert summary["minimum_top1_top2_margin"] == 0.05
    assert summary["maximum_logits_linf_delta"] == 0.25
    assert summary["maximum_mask_hamming_distance"] == 3
    assert summary["threshold_margin_min_ns"] == 5_000_000
    assert summary["classification"] == "T2-B"
