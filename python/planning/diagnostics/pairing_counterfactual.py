"""Pure contracts for audit-only state/depth counterfactual replay."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


PAIRING_COUNTERFACTUAL_CONTRACT_ID = "admissible_state_depth_pairing_counterfactual"


def enumerate_admissible_pairs(
    *,
    boundary_kind: str,
    states: Sequence[Mapping[str, Any]],
    depths: Sequence[Mapping[str, Any]],
    previous_state_sequence: int,
    previous_depth_sequence: int,
    max_sensor_skew_ns: int,
    required_state_sequence: Optional[int] = None,
    required_state_timestamp_ns: Optional[int] = None,
) -> list[Dict[str, Any]]:
    """Enumerate pairs admitted by the production freshness/sync contract.

    Reset observations admit any state and depth newer than the token captured
    before settling. Primitive endpoints bind the state to frame N-1 and admit
    only a fresh depth captured at or after that post-integration state.
    """

    kind = str(boundary_kind)
    if kind not in ("reset", "primitive_endpoint"):
        raise ValueError("unknown pairing boundary kind: {}".format(kind))
    limit = int(max_sensor_skew_ns)
    if limit < 0:
        raise ValueError("max_sensor_skew_ns must be non-negative")

    output: list[Dict[str, Any]] = []
    for state in states:
        state_sequence = int(state["state_sequence"])
        state_timestamp_ns = int(state["state_timestamp_ns"])
        state_fresh = state_sequence > int(previous_state_sequence)
        if kind == "primitive_endpoint":
            state_fresh = (
                required_state_sequence is not None
                and state_sequence == int(required_state_sequence)
            )
            if required_state_timestamp_ns is not None:
                state_fresh = state_fresh and (
                    state_timestamp_ns == int(required_state_timestamp_ns)
                )
        if not state_fresh:
            continue

        for depth in depths:
            depth_sequence = int(depth["depth_sequence"])
            depth_timestamp_ns = int(depth["depth_timestamp_ns"])
            depth_fresh = depth_sequence > int(previous_depth_sequence)
            endpoint_ok = (
                kind != "primitive_endpoint"
                or depth_timestamp_ns >= state_timestamp_ns
            )
            sensor_skew_ns = abs(state_timestamp_ns - depth_timestamp_ns)
            sync_ok = (
                state_timestamp_ns <= 0
                or depth_timestamp_ns <= 0
                or sensor_skew_ns <= limit
            )
            if not (depth_fresh and endpoint_ok and sync_ok):
                continue
            output.append({
                "state_sequence": state_sequence,
                "depth_sequence": depth_sequence,
                "state_timestamp_ns": state_timestamp_ns,
                "depth_timestamp_ns": depth_timestamp_ns,
                "sensor_skew_ns": int(sensor_skew_ns),
                "state": state,
                "depth": depth,
            })
    return output


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _unique_count(pairs: Sequence[Mapping[str, Any]], key: str) -> int:
    return len({str(pair.get(key, "")) for pair in pairs})


def summarize_counterfactual_steps(
    steps: Sequence[Mapping[str, Any]], *, max_sensor_skew_ns: int
) -> Dict[str, Any]:
    """Summarize replay results without changing evaluator semantics."""

    normalized = [dict(step) for step in steps]
    decision_steps = [step for step in normalized if bool(step.get("policy_decision", True))]
    pair_counts = [len(step.get("pairs", [])) for step in normalized]
    all_pairs = [pair for step in normalized for pair in step.get("pairs", [])]

    def steps_with_multiple(key: str) -> list[int]:
        return [
            int(step["step"])
            for step in normalized
            if _unique_count(step.get("pairs", []), key) > 1
        ]

    multiple_observations = steps_with_multiple("observation_fingerprint")
    multiple_depths = steps_with_multiple("depth_fingerprint")
    multiple_masks = steps_with_multiple("mask_fingerprint")
    multiple_logits = steps_with_multiple("logits_fingerprint")
    multiple_actions = steps_with_multiple("top1_action")
    terminal_changes = steps_with_multiple("terminal_reward_input_fingerprint")

    margins = [
        float(pair["top1_logit"]) - float(pair["top2_logit"])
        for pair in all_pairs
        if "top1_logit" in pair and "top2_logit" in pair
    ]
    logits_linf = [float(pair.get("logits_linf_from_selected", 0.0)) for pair in all_pairs]
    mask_hamming = [int(pair.get("mask_hamming_from_selected", 0)) for pair in all_pairs]
    skews = [int(pair["sensor_skew_ns"]) for pair in all_pairs]
    valid_counts = [int(pair["valid_action_count"]) for pair in all_pairs]
    top1_deltas = [
        float(pair.get("top1_logit_delta_from_selected", 0.0)) for pair in all_pairs
    ]

    behavior_related_changes = bool(
        multiple_observations
        or multiple_depths
        or multiple_masks
        or multiple_logits
        or terminal_changes
    )
    if multiple_actions:
        classification = "T2-B"
    elif behavior_related_changes:
        classification = "T2-A"
    else:
        classification = "T3-candidate"

    count_distribution = {
        "min": int(min(pair_counts)) if pair_counts else 0,
        "p50": _percentile(pair_counts, 50.0),
        "p95": _percentile(pair_counts, 95.0),
        "max": int(max(pair_counts)) if pair_counts else 0,
    }
    skew_distribution = {
        "min_ns": int(min(skews)) if skews else None,
        "p50_ns": _percentile(skews, 50.0),
        "p95_ns": _percentile(skews, 95.0),
        "max_ns": int(max(skews)) if skews else None,
    }
    return {
        "contract_id": PAIRING_COUNTERFACTUAL_CONTRACT_ID,
        "classification": classification,
        "policy_decision_count": len(decision_steps),
        "captured_decision_boundary_count": len(normalized),
        "total_admissible_pairs": len(all_pairs),
        "admissible_pair_count_distribution": count_distribution,
        "steps_with_multiple_admissible_pairs": [
            int(step["step"])
            for step in normalized
            if len(step.get("pairs", [])) > 1
        ],
        "steps_with_multiple_observation_fingerprints": multiple_observations,
        "steps_with_multiple_depth_fingerprints": multiple_depths,
        "steps_with_multiple_masks": multiple_masks,
        "steps_with_multiple_logits_fingerprints": multiple_logits,
        "steps_with_multiple_top1_actions": multiple_actions,
        "steps_with_terminal_reward_input_changes": terminal_changes,
        "minimum_top1_top2_margin": round(min(margins), 12) if margins else None,
        "maximum_top1_logit_delta": (
            max(top1_deltas, default=0.0)
        ),
        "maximum_logits_linf_delta": max(logits_linf, default=0.0),
        "maximum_mask_hamming_distance": max(mask_hamming, default=0),
        "valid_action_count_min": min(valid_counts, default=None),
        "valid_action_count_max": max(valid_counts, default=None),
        "synchronization_skew": skew_distribution,
        "threshold_margin_min_ns": (
            int(max_sensor_skew_ns) - max(skews) if skews else None
        ),
    }


def risk_rankings(steps: Sequence[Mapping[str, Any]], *, limit: int = 10) -> Dict[str, list[Dict[str, Any]]]:
    """Rank individual decision boundaries by the requested sensitivity axes."""

    records = []
    for step in steps:
        pairs = list(step.get("pairs", []))
        if not pairs:
            continue
        margins = [
            float(pair["top1_logit"]) - float(pair["top2_logit"])
            for pair in pairs
        ]
        records.append({
            "step": int(step["step"]),
            "admissible_pair_count": len(pairs),
            "minimum_top1_top2_margin": min(margins),
            "maximum_sensor_skew_ns": max(int(pair["sensor_skew_ns"]) for pair in pairs),
            "maximum_logits_linf_delta": max(
                float(pair.get("logits_linf_from_selected", 0.0)) for pair in pairs
            ),
            "maximum_mask_hamming_distance": max(
                int(pair.get("mask_hamming_from_selected", 0)) for pair in pairs
            ),
            "minimum_mask_boundary_margin_m": min(
                (
                    float(pair.get("mask_boundary_margin_m", float("inf")))
                    for pair in pairs
                    if math.isfinite(float(pair.get("mask_boundary_margin_m", float("nan"))))
                ),
                default=None,
            ),
            "unique_observation_fingerprints": _unique_count(pairs, "observation_fingerprint"),
            "unique_depth_fingerprints": _unique_count(pairs, "depth_fingerprint"),
        })
    n = max(0, int(limit))
    return {
        "smallest_logit_margin": sorted(
            records, key=lambda item: (item["minimum_top1_top2_margin"], item["step"])
        )[:n],
        "closest_to_sync_threshold": sorted(
            records, key=lambda item: (-item["maximum_sensor_skew_ns"], item["step"])
        )[:n],
        "most_admissible_pairs": sorted(
            records, key=lambda item: (-item["admissible_pair_count"], item["step"])
        )[:n],
        "largest_content_difference": sorted(
            records,
            key=lambda item: (
                -item["unique_observation_fingerprints"],
                -item["unique_depth_fingerprints"],
                -item["maximum_logits_linf_delta"],
                item["step"],
            ),
        )[:n],
        "largest_mask_difference": sorted(
            records,
            key=lambda item: (-item["maximum_mask_hamming_distance"], item["step"]),
        )[:n],
        "closest_to_mask_boundary": sorted(
            (item for item in records if item["minimum_mask_boundary_margin_m"] is not None),
            key=lambda item: (item["minimum_mask_boundary_margin_m"], item["step"]),
        )[:n],
    }
