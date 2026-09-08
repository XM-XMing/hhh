"""Aggregate isolated live evaluator traces for state/depth reachability."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np


LIVE_PAIRING_REACHABILITY_CONTRACT_ID = "live_pairing_reachability"
RESET_THEORETICAL_ADMISSIBLE_PAIRS = 98


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _pairwise_linf(values: Sequence[np.ndarray]) -> float:
    maximum = 0.0
    for left_index, left in enumerate(values):
        for right in values[left_index + 1:]:
            maximum = max(maximum, float(np.max(np.abs(left - right))))
    return maximum


def _pairwise_hamming(values: Sequence[np.ndarray]) -> int:
    maximum = 0
    for left_index, left in enumerate(values):
        for right in values[left_index + 1:]:
            maximum = max(maximum, int(np.count_nonzero(left != right)))
    return maximum


def _step_record(trace: Mapping[str, Any], step: int) -> Mapping[str, Any]:
    primitives = list(trace.get("primitives", []))
    if step < 0 or step >= len(primitives):
        raise ValueError(
            "repeat {} has no primitive step {}".format(trace.get("repeat", "?"), step)
        )
    record = primitives[step]
    required = (
        "action_mask",
        "actor_logits",
        "actor_top_k",
        "terminal_reward_input_fingerprint",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(
            "repeat {} step {} lacks live audit fields: {}".format(
                trace.get("repeat", "?"), step, ",".join(missing)
            )
        )
    return record


def summarize_live_step(traces: Sequence[Mapping[str, Any]], step: int) -> Dict[str, Any]:
    records = [_step_record(trace, step) for trace in traces]
    before = [record["before"] for record in records]
    logits = [np.asarray(record["actor_logits"], dtype=np.float32) for record in records]
    masks = [np.asarray(record["action_mask"], dtype=np.bool_) for record in records]
    if len({tuple(value.shape) for value in logits}) != 1:
        raise ValueError("actor logits shapes differ at step {}".format(step))
    if len({tuple(value.shape) for value in masks}) != 1:
        raise ValueError("action mask shapes differ at step {}".format(step))

    top1_logits = []
    margins = []
    for record in records:
        top = list(record["actor_top_k"])
        if len(top) < 2:
            raise ValueError("step {} has fewer than two valid top-k actions".format(step))
        top1_logits.append(float(top[0]["logit"]))
        margins.append(float(top[0]["logit"]) - float(top[1]["logit"]))
    skews = [int(item["sensor_skew_ns"]) for item in before]

    unique = lambda values: len(set(values))
    return {
        "step": int(step),
        "unique_selected_state_frame_metadata_count": unique(
            (
                int(item["state_sequence"]),
                int(item["state_id"]),
                int(item["state_timestamp_ns"]),
            )
            for item in before
        ),
        "unique_state_fingerprint_count": unique(
            str(item["state_fingerprint"]) for item in before
        ),
        "unique_selected_depth_frame_metadata_count": unique(
            (int(item["depth_sequence"]), int(item["depth_timestamp_ns"]))
            for item in before
        ),
        "unique_selected_pair_metadata_count": unique(
            (
                int(item["state_sequence"]),
                int(item["state_id"]),
                int(item["state_timestamp_ns"]),
                int(item["depth_sequence"]),
                int(item["depth_timestamp_ns"]),
            )
            for item in before
        ),
        "unique_selected_pair_content_count": unique(
            (str(item["state_fingerprint"]), str(item["depth_fingerprint"]))
            for item in before
        ),
        "unique_depth_fingerprint_count": unique(
            str(item["depth_fingerprint"]) for item in before
        ),
        "unique_observation_fingerprint_count": unique(
            str(item["observation_fingerprint"]) for item in before
        ),
        "unique_mask_fingerprint_count": unique(
            str(record["action_mask_fingerprint"]) for record in records
        ),
        "unique_logits_fingerprint_count": unique(
            str(record["actor_logits_fingerprint"]) for record in records
        ),
        "unique_top1_action_count": unique(
            int(record["selected_action"]) for record in records
        ),
        "unique_terminal_reward_input_fingerprint_count": unique(
            str(record["terminal_reward_input_fingerprint"]) for record in records
        ),
        "valid_action_count_min": min(int(record["valid_action_count"]) for record in records),
        "valid_action_count_max": max(int(record["valid_action_count"]) for record in records),
        "sensor_skew_ns": {
            "min": min(skews),
            "p50": _percentile(skews, 50.0),
            "p95": _percentile(skews, 95.0),
            "max": max(skews),
        },
        "top1_logit_min": min(top1_logits),
        "top1_logit_max": max(top1_logits),
        "top1_top2_margin_min": min(margins),
        "top1_top2_margin_max": max(margins),
        "logits_pairwise_max_linf_delta": _pairwise_linf(logits),
        "mask_pairwise_max_hamming_distance": _pairwise_hamming(masks),
    }


def _comparison_categories(
    left_trace: Mapping[str, Any], right_trace: Mapping[str, Any], step: int
) -> list[str]:
    left = _step_record(left_trace, step)
    right = _step_record(right_trace, step)
    left_before = left["before"]
    right_before = right["before"]
    categories = []
    if (
        left_before["depth_fingerprint"] != right_before["depth_fingerprint"]
        or left_before["observation_fingerprint"] != right_before["observation_fingerprint"]
    ):
        categories.append("content")
    if left["action_mask_fingerprint"] != right["action_mask_fingerprint"]:
        categories.append("mask")
    if left["actor_logits_fingerprint"] != right["actor_logits_fingerprint"]:
        categories.append("logits")
    if int(left["selected_action"]) != int(right["selected_action"]):
        categories.append("action")
    if (
        left["terminal_reward_input_fingerprint"]
        != right["terminal_reward_input_fingerprint"]
    ):
        categories.append("terminal_reward")
    return categories


def summarize_live_reachability(
    traces: Sequence[Mapping[str, Any]],
    *,
    sensitive_steps: Sequence[int] = (0, 3, 7, 15, 16, 18, 24, 26, 27),
) -> Dict[str, Any]:
    values = [dict(trace) for trace in traces]
    if not values:
        raise ValueError("at least one live trace is required")
    identity = (
        values[0]["checkpoint_sha256"],
        values[0]["mission_id"],
        int(values[0]["episode_id"]),
    )
    for trace in values:
        actual = (
            trace["checkpoint_sha256"], trace["mission_id"], int(trace["episode_id"])
        )
        if actual != identity:
            raise ValueError("live reachability traces do not share checkpoint/mission identity")

    common_steps = min(len(trace.get("primitives", [])) for trace in values)
    requested_steps = sorted({int(step) for step in sensitive_steps if int(step) < common_steps})
    step_summaries = {
        str(step): summarize_live_step(values, step) for step in requested_steps
    }
    all_step_summaries = [summarize_live_step(values, step) for step in range(common_steps)]

    first_divergence = None
    any_live_content = False
    any_live_action = False
    any_live_mask = False
    any_live_terminal = False
    for step in range(common_steps):
        summary = all_step_summaries[step]
        any_live_content = any_live_content or (
            summary["unique_depth_fingerprint_count"] > 1
            or summary["unique_observation_fingerprint_count"] > 1
        )
        any_live_action = any_live_action or summary["unique_top1_action_count"] > 1
        any_live_mask = any_live_mask or summary["unique_mask_fingerprint_count"] > 1
        any_live_terminal = any_live_terminal or (
            summary["unique_terminal_reward_input_fingerprint_count"] > 1
        )
        if first_divergence is None:
            for left_index, left in enumerate(values):
                for right in values[left_index + 1:]:
                    categories = _comparison_categories(left, right, step)
                    if categories:
                        first_divergence = {
                            "step": int(step),
                            "left": str(left.get("repeat", left_index)),
                            "right": str(right.get("repeat", "?")),
                            "categories": categories,
                            "left_record": _step_record(left, step),
                            "right_record": _step_record(right, step),
                        }
                        break
                if first_divergence is not None:
                    break

    outcomes = [str(trace.get("outcome", "")) for trace in values]
    primitive_counts = [len(trace.get("primitives", [])) for trace in values]
    outcome_divergence = len(set(outcomes)) > 1 or len(set(primitive_counts)) > 1
    if first_divergence is not None and outcome_divergence:
        first_divergence["categories"].append("outcome")
    elif first_divergence is None and outcome_divergence:
        first_divergence = {
            "step": common_steps,
            "left": str(values[0].get("repeat", "0")),
            "right": str(values[1].get("repeat", "1")) if len(values) > 1 else "",
            "categories": ["outcome"],
        }

    behavior_divergence = bool(
        any_live_content
        and (any_live_mask or any_live_action or any_live_terminal or outcome_divergence)
    )
    if behavior_divergence:
        classification = "T1"
    elif any_live_content:
        classification = "T2-live"
    else:
        classification = "T2-not-live-observed"

    repeat_count = len(values)
    upper_bound = 1.0 - math.pow(0.05, 1.0 / float(repeat_count))
    return {
        "contract_id": LIVE_PAIRING_REACHABILITY_CONTRACT_ID,
        "classification": classification,
        "repeat_count": repeat_count,
        "checkpoint_sha256": identity[0],
        "mission_id": identity[1],
        "episode_id": identity[2],
        "reset_theoretical_admissible_pair_count": RESET_THEORETICAL_ADMISSIBLE_PAIRS,
        "reset": all_step_summaries[0],
        "sensitive_steps": step_summaries,
        "any_live_content_divergence": bool(any_live_content),
        "any_live_mask_divergence": bool(any_live_mask),
        "any_live_action_divergence": bool(any_live_action),
        "any_live_terminal_reward_divergence": bool(any_live_terminal),
        "any_live_outcome_divergence": bool(outcome_divergence),
        "outcomes": outcomes,
        "primitive_counts": primitive_counts,
        "first_divergence": first_divergence,
        "one_sided_95pct_failure_rate_upper_bound": upper_bound,
    }
