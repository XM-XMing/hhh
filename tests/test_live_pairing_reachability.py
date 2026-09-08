from __future__ import annotations

from planning.diagnostics.live_pairing_reachability import summarize_live_reachability


def _trace(
    name: str,
    *,
    depth: str = "depth-a",
    observation: str = "obs-a",
    logits: tuple[float, ...] = (3.0, 2.0, 1.0),
    mask: tuple[bool, ...] = (True, True, True),
    action: int = 0,
    terminal: str = "terminal-a",
    outcome: str = "success",
):
    top = sorted(range(len(logits)), key=lambda index: logits[index], reverse=True)
    return {
        "repeat": name,
        "checkpoint_sha256": "checkpoint",
        "mission_id": "mission",
        "episode_id": 451,
        "outcome": outcome,
        "primitives": [{
            "step": 0,
            "before": {
                "state_sequence": 10,
                "state_id": 10,
                "state_timestamp_ns": 100,
                "state_fingerprint": "state-a",
                "depth_sequence": 20,
                "depth_timestamp_ns": 90,
                "sensor_skew_ns": 10,
                "depth_fingerprint": depth,
                "observation_fingerprint": observation,
            },
            "action_mask_fingerprint": "mask-{}".format(mask),
            "action_mask": list(mask),
            "valid_action_count": sum(mask),
            "actor_logits_fingerprint": "logits-{}".format(logits),
            "actor_logits": list(logits),
            "actor_top_k": [
                {"action": index, "logit": logits[index]} for index in top
            ],
            "selected_action": action,
            "terminal_reward_input_fingerprint": terminal,
            "done": True,
            "done_reason": outcome,
        }],
    }


def test_no_live_content_variation_is_t2_not_live_observed():
    summary = summarize_live_reachability([_trace("r1"), _trace("r2")])

    assert summary["classification"] == "T2-not-live-observed"
    assert summary["reset"]["unique_selected_pair_metadata_count"] == 1
    assert summary["reset"]["unique_depth_fingerprint_count"] == 1
    assert summary["one_sided_95pct_failure_rate_upper_bound"] > 0.0


def test_live_content_and_logits_variation_without_behavior_change_is_t2_live():
    summary = summarize_live_reachability([
        _trace("r1"),
        _trace("r2", depth="depth-b", observation="obs-b", logits=(2.8, 2.0, 1.0)),
    ])

    assert summary["classification"] == "T2-live"
    assert summary["reset"]["unique_depth_fingerprint_count"] == 2
    assert summary["reset"]["unique_logits_fingerprint_count"] == 2
    assert summary["reset"]["unique_top1_action_count"] == 1
    assert summary["first_divergence"]["categories"] == ["content", "logits"]


def test_live_content_reaching_mask_or_action_is_t1():
    summary = summarize_live_reachability([
        _trace("r1"),
        _trace(
            "r2",
            depth="depth-b",
            observation="obs-b",
            logits=(2.0, 3.0, 1.0),
            mask=(False, True, True),
            action=1,
            terminal="terminal-b",
            outcome="collision",
        ),
    ])

    assert summary["classification"] == "T1"
    assert summary["first_divergence"]["step"] == 0
    assert summary["first_divergence"]["categories"] == [
        "content", "mask", "logits", "action", "terminal_reward", "outcome"
    ]
