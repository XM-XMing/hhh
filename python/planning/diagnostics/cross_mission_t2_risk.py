"""Pure Layer-2 selection and comparison contracts for live pairing risk."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


CROSS_MISSION_T2_RISK_CONTRACT_ID = "cross_mission_t2_risk_screen"
SYNC_THRESHOLD_NS = 80_000_000
PHYSICS_DT_NS = 20_000_000
_OUTCOME_KEYS = ("success", "collision", "dead_end", "timeout", "far", "hard_altitude")
_OUTCOME_COVERAGE_ORDER = ("collision", "dead_end", "timeout", "success")
_CLASS_RANK = {"M0": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4}


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _minimum(values: Iterable[Any]) -> float | None:
    finite = [number for number in (_finite_or_none(value) for value in values) if number is not None]
    return min(finite) if finite else None


def _maximum(values: Iterable[Any]) -> float | None:
    finite = [number for number in (_finite_or_none(value) for value in values) if number is not None]
    return max(finite) if finite else None


def _outcome_from_rollout(row: Mapping[str, Any]) -> str:
    for key in _OUTCOME_KEYS:
        if str(row.get(key, "")).lower() == "true":
            return key
        if row.get(key) is True:
            return key
    return str(row.get("stop_reason", "unknown")) or "unknown"


class ExecutionPayloadEvidenceError(ValueError):
    """Cross-repeat payload evidence is incomplete and cannot be classified."""


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalized_state_offset(value: Any, first_value: int | None) -> int | None:
    normalized = _integer_or_none(value)
    if normalized is None or first_value is None:
        return None
    return normalized - first_value


def _normalized_physics_tick_offset(value: Any, first_value: int | None) -> int | None:
    normalized = _integer_or_none(value)
    if normalized is None or first_value is None:
        return None
    return int(round((normalized - first_value) / float(PHYSICS_DT_NS)))


def _execution_record(record: Mapping[str, Any]) -> Dict[str, Any] | None:
    """Normalize completed and terminal-abort receipts without correlation IDs."""
    execution = record.get("primitive_execution")
    if isinstance(execution, Mapping):
        frames = list(execution.get("applied_frames", []))
        return {
            "classification": "COMPLETED",
            "frame_count": _integer_or_none(
                execution.get("requested_frame_count", execution.get("frame_count"))
            ),
            "applied_frames": frames,
            "endpoint_state_id": execution.get("endpoint_state_id"),
            "effective_integration_ticks": _integer_or_none(
                execution.get("effective_integration_ticks", len(frames))
            ),
            "primitive_completed": bool(record.get("primitive_completed", True)),
            "terminal_abort": bool(record.get("terminal_abort", False)),
            "terminal_reason": None,
            "terminal_state": None,
            "has_completed_endpoint": "endpoint_state_id" in execution,
        }

    result = record.get("primitive_execution_result")
    if not isinstance(result, Mapping):
        return None
    partial_receipt = result.get("partial_receipt")
    received_frames = (
        list(partial_receipt.get("received_frames", []))
        if isinstance(partial_receipt, Mapping)
        else []
    )
    applied_frames = [
        frame
        for frame in received_frames
        if isinstance(frame, Mapping) and _integer_or_none(frame.get("frame_index")) is not None
        and int(frame["frame_index"]) >= 0
    ]
    failed_frames = [
        frame
        for frame in received_frames
        if isinstance(frame, Mapping) and _integer_or_none(frame.get("frame_index")) == -1
    ]
    terminal_state = result.get("terminal_state")
    if not isinstance(terminal_state, Mapping):
        terminal_state = failed_frames[-1] if failed_frames else None
    return {
        "classification": str(result.get("kind", "")),
        "frame_count": _integer_or_none(
            partial_receipt.get("requested_frame_count")
            if isinstance(partial_receipt, Mapping)
            else None
        ),
        "applied_frames": applied_frames,
        "endpoint_state_id": None,
        "effective_integration_ticks": _integer_or_none(
            result.get(
                "effective_integration_ticks",
                record.get("effective_integration_ticks", len(applied_frames)),
            )
        ),
        "primitive_completed": bool(record.get("primitive_completed", False)),
        "terminal_abort": bool(record.get("terminal_abort", False)),
        "terminal_reason": result.get("terminal_reason"),
        "terminal_state": terminal_state,
        "has_completed_endpoint": "endpoint_state_id" in result,
    }


def _payload_evidence_available(execution: Mapping[str, Any] | None) -> bool | None:
    if execution is None:
        return None
    frames = list(execution["applied_frames"])
    if not frames:
        return False
    present = [
        "command_payload_fingerprint" in frame
        and frame.get("command_payload_fingerprint") is not None
        for frame in frames
    ]
    if any(present) and not all(present):
        raise ExecutionPayloadEvidenceError(
            "payload evidence is incomplete within one primitive execution"
        )
    return bool(present and all(present))


def _payload_evidence_for_traces(traces: Sequence[Mapping[str, Any]]) -> bool:
    evidence = {
        available
        for trace in traces
        for record in trace.get("primitives", [])
        for available in [_payload_evidence_available(_execution_record(record))]
        if available is not None
    }
    if len(evidence) > 1:
        raise ExecutionPayloadEvidenceError(
            "payload evidence is mixed across the cross-repeat comparison set"
        )
    return evidence == {True}


def _canonical_frame(
    frame: Mapping[str, Any],
    *,
    first_state_id: int | None,
    first_sim_time_ns: int | None,
    include_payload: bool,
) -> tuple:
    values = (
        _integer_or_none(frame.get("frame_index")),
        _normalized_state_offset(frame.get("applied_state_id"), first_state_id),
        _normalized_physics_tick_offset(frame.get("sim_time_ns"), first_sim_time_ns),
        _integer_or_none(frame.get("execution_status")),
    )
    if include_payload:
        return values + (frame.get("command_payload_fingerprint"),)
    return values


def _canonical_failed_state(
    state: Mapping[str, Any] | None,
    *,
    first_state_id: int | None,
    first_sim_time_ns: int | None,
) -> tuple | None:
    if not isinstance(state, Mapping):
        return None
    return (
        _integer_or_none(state.get("frame_index")),
        _normalized_state_offset(state.get("applied_state_id"), first_state_id),
        _normalized_physics_tick_offset(state.get("sim_time_ns"), first_sim_time_ns),
        _integer_or_none(state.get("execution_status")),
        bool(state.get("collided", False)),
    )


def _execution_signature(
    record: Mapping[str, Any], *, payload_evidence_available: bool
) -> tuple:
    """Compare execution semantics, excluding run-local correlation identities."""
    execution = _execution_record(record)
    if execution is None:
        return ("missing",)
    frames = list(execution["applied_frames"])
    first_state_id = _integer_or_none(
        frames[0].get("applied_state_id") if frames else None
    )
    first_sim_time_ns = _integer_or_none(
        frames[0].get("sim_time_ns") if frames else None
    )
    frame_signature = tuple(
        _canonical_frame(
            frame,
            first_state_id=first_state_id,
            first_sim_time_ns=first_sim_time_ns,
            include_payload=payload_evidence_available,
        )
        for frame in frames
    )
    endpoint_relation = _normalized_state_offset(
        execution["endpoint_state_id"], first_state_id
    )
    return (
        execution["classification"],
        execution["frame_count"],
        frame_signature,
        endpoint_relation,
        execution["effective_integration_ticks"],
        execution["primitive_completed"],
        execution["terminal_abort"],
        execution["terminal_reason"],
        execution["has_completed_endpoint"],
        _canonical_failed_state(
            execution["terminal_state"],
            first_state_id=first_state_id,
            first_sim_time_ns=first_sim_time_ns,
        ),
    )


def _execution_correlation_identity_signature(record: Mapping[str, Any]) -> tuple:
    """Audit-only identity trace; never use this to classify cross-run divergence."""
    execution = record.get("primitive_execution")
    if isinstance(execution, Mapping):
        return (
            "COMPLETED",
            execution.get("execution_id"),
            tuple(
                frame.get("command_id")
                for frame in execution.get("applied_frames", [])
                if isinstance(frame, Mapping)
            ),
        )
    result = record.get("primitive_execution_result")
    partial_receipt = result.get("partial_receipt") if isinstance(result, Mapping) else None
    if isinstance(partial_receipt, Mapping):
        return (
            str(result.get("kind", "")),
            partial_receipt.get("execution_id"),
            tuple(
                frame.get("command_id")
                for frame in partial_receipt.get("received_frames", [])
                if isinstance(frame, Mapping)
                and _integer_or_none(frame.get("frame_index")) is not None
                and int(frame["frame_index"]) >= 0
            ),
        )
    return ("missing",)


def _trace_outcome_signature(trace: Mapping[str, Any], rollout: Mapping[str, Any]) -> tuple:
    return (
        str(trace.get("outcome", "")),
        int(trace.get("steps", -1)),
        _outcome_from_rollout(rollout),
        int(float(rollout.get("steps", -1))),
        tuple(str(rollout.get(key, "")) for key in _OUTCOME_KEYS),
        float(rollout.get("final_x", float("nan"))),
        float(rollout.get("final_y", float("nan"))),
        float(rollout.get("final_z", float("nan"))),
        float(rollout.get("final_distance_xy", float("nan"))),
    )


def _ordered_unique(values: Iterable[Any]) -> int:
    return len(set(values))


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _risk_order(table: Sequence[Mapping[str, Any]], key: str, *, descending: bool = False) -> list[Mapping[str, Any]]:
    def sort_key(record: Mapping[str, Any]):
        value = _finite_or_none(record.get(key))
        missing = value is None
        numeric = 0.0 if missing else float(value)
        return (missing, -numeric if descending else numeric, int(record["episode_id"]))
    return sorted(table, key=sort_key)


def build_risk_table(
    traces: Sequence[Mapping[str, Any]],
    rollout_by_episode: Mapping[int, Mapping[str, Any]],
    pairing_counts_by_episode: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[Dict[str, Any]]:
    """Extract raw, unweighted mission-risk measures from one BC audit run."""
    table = []
    for trace in traces:
        episode_id = int(trace["episode_id"])
        rollout = rollout_by_episode.get(episode_id)
        if rollout is None:
            raise ValueError("trace episode {} lacks rollout row".format(episode_id))
        primitives = list(trace.get("primitives", []))
        reset_pairs = [
            entry for entry in pairing_counts_by_episode.get(episode_id, [])
            if str(entry.get("boundary_kind", "")) == "reset"
        ]
        skews = [
            int(record.get("before", {}).get("sensor_skew_ns", 0))
            for record in primitives
        ]
        table.append({
            "episode_id": episode_id,
            "mission_id": str(trace["mission_id"]),
            "baseline_outcome": _outcome_from_rollout(rollout),
            "primitive_count": len(primitives),
            "minimum_top1_top2_margin": _minimum(
                record.get("top1_top2_margin") for record in primitives
            ),
            "minimum_depth_mask_margin_m": _minimum(
                record.get("depth_mask_boundary_margin_m") for record in primitives
            ),
            "maximum_sensor_skew_ns": max(skews) if skews else None,
            "minimum_sync_threshold_margin_ns": (
                SYNC_THRESHOLD_NS - max(skews) if skews else None
            ),
            "reset_admissible_pair_count": (
                max(int(entry["admissible_pair_count"]) for entry in reset_pairs)
                if reset_pairs else None
            ),
            "minimum_safety_clearance_m": _minimum(
                record.get("minimum_safety_clearance_m") for record in primitives
            ),
            "minimum_valid_action_count": min(
                (int(record.get("valid_action_count", 0)) for record in primitives),
                default=None,
            ),
        })
    return sorted(table, key=lambda record: int(record["episode_id"]))


def select_risk_enriched_missions(
    table: Sequence[Mapping[str, Any]], *, count: int = 20
) -> list[Dict[str, Any]]:
    """Select fixed missions by coverage, not a fabricated weighted risk score.

    First reserve one representative for each observed terminal class.  Then
    take the next unseen mission in a deterministic round-robin over the raw
    requested risk axes.  Selection reasons are retained so the manifest is
    immutable and independently auditable.
    """
    if int(count) <= 0:
        raise ValueError("count must be positive")
    if len(table) < int(count):
        raise ValueError("risk table has fewer than {} missions".format(count))

    selected: Dict[int, Dict[str, Any]] = {}

    def add(record: Mapping[str, Any], reason: str) -> None:
        episode_id = int(record["episode_id"])
        current = selected.get(episode_id)
        if current is None:
            current = dict(record)
            current["selection_reasons"] = []
            selected[episode_id] = current
        current["selection_reasons"].append(reason)

    outcome_rank = (
        "minimum_top1_top2_margin",
        "minimum_depth_mask_margin_m",
        "minimum_sync_threshold_margin_ns",
        "primitive_count",
        "episode_id",
    )
    for outcome in _OUTCOME_COVERAGE_ORDER:
        members = [record for record in table if record["baseline_outcome"] == outcome]
        if members:
            ranked = sorted(
                members,
                key=lambda record: tuple(
                    (float("inf") if _finite_or_none(record.get(key)) is None else record[key])
                    if key != "primitive_count" else -int(record[key])
                    for key in outcome_rank
                ),
            )
            add(ranked[0], "outcome:{}".format(outcome))
            if len(selected) >= int(count):
                break

    axes = (
        ("minimum_top1_top2_margin", False, "smallest_actor_margin"),
        ("minimum_depth_mask_margin_m", False, "closest_depth_mask_boundary"),
        ("minimum_sync_threshold_margin_ns", False, "closest_sync_threshold"),
        ("reset_admissible_pair_count", True, "most_reset_admissible_pairs"),
        ("primitive_count", True, "longest_primitive_sequence"),
        ("minimum_safety_clearance_m", False, "closest_safety_boundary"),
        ("minimum_valid_action_count", False, "fewest_valid_actions"),
    )
    rankings = [(_risk_order(table, key, descending=descending), label) for key, descending, label in axes]
    rank = 0
    while len(selected) < int(count):
        made_progress = False
        for ordered, label in rankings:
            if rank < len(ordered):
                candidate = ordered[rank]
                if int(candidate["episode_id"]) not in selected:
                    add(candidate, "axis:{}:rank:{}".format(label, rank + 1))
                    made_progress = True
                    if len(selected) >= int(count):
                        break
        if not made_progress and rank >= len(table):
            break
        rank += 1
    if len(selected) != int(count):
        raise RuntimeError("coverage selection stopped at {} missions".format(len(selected)))
    return [selected[episode_id] for episode_id in sorted(selected)]


def _step_categories(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    payload_evidence_available: bool,
) -> list[str]:
    before_left, before_right = left["before"], right["before"]
    categories = []
    if before_left.get("state_fingerprint") != before_right.get("state_fingerprint"):
        categories.append("state_content")
    if before_left.get("depth_fingerprint") != before_right.get("depth_fingerprint"):
        categories.append("depth_content")
    if before_left.get("observation_fingerprint") != before_right.get("observation_fingerprint"):
        categories.append("observation")
    if left.get("actor_logits_fingerprint") != right.get("actor_logits_fingerprint"):
        categories.append("logits")
    if left.get("action_mask_fingerprint") != right.get("action_mask_fingerprint"):
        categories.append("mask")
    if int(left.get("selected_action", -1)) != int(right.get("selected_action", -1)):
        categories.append("action")
    if left.get("terminal_reward_input_fingerprint") != right.get("terminal_reward_input_fingerprint"):
        categories.append("terminal_reward")
    if _execution_signature(
        left, payload_evidence_available=payload_evidence_available
    ) != _execution_signature(
        right, payload_evidence_available=payload_evidence_available
    ):
        categories.append("primitive_execution")
    after_left, after_right = left.get("after", {}), right.get("after", {})
    if after_left.get("observation_fingerprint") != after_right.get("observation_fingerprint"):
        categories.append("endpoint")
    return categories


def _severity(categories: Sequence[str], *, outcome_changed: bool = False) -> str:
    if outcome_changed or "endpoint" in categories or "primitive_execution" in categories:
        return "M4"
    if "action" in categories:
        return "M3"
    if "mask" in categories:
        return "M2"
    if any(key in categories for key in ("state_content", "depth_content", "observation", "logits")):
        return "M1"
    return "M0"


def summarize_mission_repeats(
    *, mission: Mapping[str, Any], repeats: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Compare five isolated traces while explicitly excluding timing-only metadata."""
    if len(repeats) < 2:
        raise ValueError("at least two repeats are required")
    traces = [item["trace"] for item in repeats]
    rollouts = [item["rollout"] for item in repeats]
    identity = (str(mission["mission_id"]), int(mission["episode_id"]))
    for trace in traces:
        if (str(trace["mission_id"]), int(trace["episode_id"])) != identity:
            raise ValueError("repeat trace does not match mission manifest")

    payload_evidence_available = _payload_evidence_for_traces(traces)

    base = traces[0]
    base_rollout = rollouts[0]
    metadata_pair_variation = False
    first_divergence = None
    maximum_class = "M0"
    for repeat_index, (trace, rollout) in enumerate(zip(traces[1:], rollouts[1:]), start=1):
        max_steps = min(len(base.get("primitives", [])), len(trace.get("primitives", [])))
        for step in range(max_steps):
            left = base["primitives"][step]
            right = trace["primitives"][step]
            left_before, right_before = left["before"], right["before"]
            metadata_pair_variation = metadata_pair_variation or (
                (
                    left_before.get("state_sequence"), left_before.get("state_id"), left_before.get("state_timestamp_ns"),
                    left_before.get("depth_sequence"), left_before.get("depth_timestamp_ns"),
                ) != (
                    right_before.get("state_sequence"), right_before.get("state_id"), right_before.get("state_timestamp_ns"),
                    right_before.get("depth_sequence"), right_before.get("depth_timestamp_ns"),
                )
            )
            categories = _step_categories(
                left,
                right,
                payload_evidence_available=payload_evidence_available,
            )
            severity = _severity(categories)
            if _CLASS_RANK[severity] > _CLASS_RANK[maximum_class]:
                maximum_class = severity
            if categories and first_divergence is None:
                first_divergence = {
                    "repeat_a": str(repeats[0]["repeat"]),
                    "repeat_b": str(repeats[repeat_index]["repeat"]),
                    "step": int(step),
                    "categories": categories,
                    "left": left,
                    "right": right,
                }
        outcome_changed = (
            len(base.get("primitives", [])) != len(trace.get("primitives", []))
            or _trace_outcome_signature(base, base_rollout) != _trace_outcome_signature(trace, rollout)
        )
        severity = _severity((), outcome_changed=outcome_changed)
        if _CLASS_RANK[severity] > _CLASS_RANK[maximum_class]:
            maximum_class = severity
        if outcome_changed and first_divergence is None:
            first_divergence = {
                "repeat_a": str(repeats[0]["repeat"]),
                "repeat_b": str(repeats[repeat_index]["repeat"]),
                "step": max_steps,
                "categories": ["trajectory_or_outcome"],
            }

    all_records = [record for trace in traces for record in trace.get("primitives", [])]
    skews = [int(record["before"].get("sensor_skew_ns", 0)) for record in all_records]
    margin = _minimum(record.get("top1_top2_margin") for record in all_records)
    depth_margin = _minimum(record.get("depth_mask_boundary_margin_m") for record in all_records)
    trajectory_signature = lambda trace: tuple(
        (
            record["before"].get("state_fingerprint"),
            record["before"].get("depth_fingerprint"),
            record["before"].get("observation_fingerprint"),
            record.get("action_mask_fingerprint"),
            record.get("actor_logits_fingerprint"),
            int(record.get("selected_action", -1)),
            record.get("after", {}).get("observation_fingerprint"),
        )
        for record in trace.get("primitives", [])
    )
    execution_semantic_trajectories = [
        tuple(
            _execution_signature(
                record, payload_evidence_available=payload_evidence_available
            )
            for record in trace.get("primitives", [])
        )
        for trace in traces
    ]
    execution_correlation_identity_trajectories = [
        tuple(
            _execution_correlation_identity_signature(record)
            for record in trace.get("primitives", [])
        )
        for trace in traces
    ]
    return {
        **dict(mission),
        "repeat_count": len(repeats),
        "primitive_count_range": [
            min(len(trace.get("primitives", [])) for trace in traces),
            max(len(trace.get("primitives", [])) for trace in traces),
        ],
        "unique_depth_trajectories": _ordered_unique(
            tuple(record["before"].get("depth_fingerprint") for record in trace.get("primitives", []))
            for trace in traces
        ),
        "unique_observation_trajectories": _ordered_unique(
            tuple(record["before"].get("observation_fingerprint") for record in trace.get("primitives", []))
            for trace in traces
        ),
        "unique_mask_trajectories": _ordered_unique(
            tuple(record.get("action_mask_fingerprint") for record in trace.get("primitives", []))
            for trace in traces
        ),
        "unique_logits_trajectories": _ordered_unique(
            tuple(record.get("actor_logits_fingerprint") for record in trace.get("primitives", []))
            for trace in traces
        ),
        "unique_action_sequences": _ordered_unique(
            tuple(int(record.get("selected_action", -1)) for record in trace.get("primitives", []))
            for trace in traces
        ),
        "unique_terminal_outcomes": _ordered_unique(
            _trace_outcome_signature(trace, rollout) for trace, rollout in zip(traces, rollouts)
        ),
        "unique_trajectory_signatures": _ordered_unique(trajectory_signature(trace) for trace in traces),
        "unique_execution_semantic_trajectories": _ordered_unique(
            execution_semantic_trajectories
        ),
        "unique_execution_correlation_identity_trajectories": _ordered_unique(
            execution_correlation_identity_trajectories
        ),
        "execution_payload_evidence_available": payload_evidence_available,
        "metadata_pair_variation": bool(metadata_pair_variation),
        "sensor_skew_ns": {
            "min": min(skews) if skews else None,
            "p50": _percentile(skews, 50.0),
            "p95": _percentile(skews, 95.0),
            "max": max(skews) if skews else None,
        },
        "minimum_top1_top2_margin": margin,
        "minimum_depth_mask_margin_m": depth_margin,
        "classification": maximum_class,
        "first_divergence": first_divergence,
    }


def summarize_cross_mission_screen(missions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Produce the requested non-IID mission-level Layer-2 verdict."""
    values = [dict(mission) for mission in missions]
    classes = Counter(str(value["classification"]) for value in values)
    mismatch_classes = {"M1", "M2", "M3", "M4"}
    mismatch_count = sum(1 for value in values if value["classification"] in mismatch_classes)
    behavioral = classes["M3"] + classes["M4"]
    t2_activated = classes["M1"] + classes["M2"]
    if behavioral:
        verdict = "T1"
        layer3_allowed = False
    elif t2_activated:
        verdict = "T2-live"
        layer3_allowed = False
    else:
        verdict = "T2-not-live-observed"
        layer3_allowed = len(values) == 20 and classes["M0"] == len(values)
    return {
        "contract_id": CROSS_MISSION_T2_RISK_CONTRACT_ID,
        "missions": len(values),
        "repeats_per_mission": sorted({int(value["repeat_count"]) for value in values}),
        "total_runs": sum(int(value["repeat_count"]) for value in values),
        "mission_classification_counts": {key: int(classes[key]) for key in ("M0", "M1", "M2", "M3", "M4")},
        "missions_with_metadata_pair_variation": sum(bool(value["metadata_pair_variation"]) for value in values),
        "missions_with_depth_content_variation": sum(value["unique_depth_trajectories"] > 1 for value in values),
        "missions_with_observation_variation": sum(value["unique_observation_trajectories"] > 1 for value in values),
        "missions_with_logits_variation": sum(value["unique_logits_trajectories"] > 1 for value in values),
        "missions_with_mask_variation": sum(value["unique_mask_trajectories"] > 1 for value in values),
        "missions_with_top1_action_variation": sum(value["unique_action_sequences"] > 1 for value in values),
        "missions_with_terminal_outcome_variation": sum(value["unique_terminal_outcomes"] > 1 for value in values),
        "mission_level_mismatch_proportion": float(mismatch_count) / float(len(values)) if values else None,
        "total_observed_mismatch_events": mismatch_count,
        "classification": verdict,
        "layer3_allowed": layer3_allowed,
        "rule_of_three_note": (
            "If all 100 runs are divergence-free, 3/100 is a descriptive ~3% rule-of-three value only; "
            "cross-mission runs are non-homogeneous and it is not a fixed_dev_100 failure-probability bound."
        ),
        "missions_detail": values,
    }
