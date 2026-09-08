"""Pure contracts for the bounded N5/H0 independent confirmation.

This module deliberately contains no Unity, ROS, optimizer, or filesystem
side effects.  It owns the task-identity and fixed-budget seams used by the
offline confirmation entry point so they can be regression-tested separately
from the real four-branch experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from planning.mission.spec import mission_id_from_values


FORMAL_FIELDS = (
    "result",
    "steps",
    "final_distance_xy_m",
    "final_error_z_m",
    "route_length_m",
    "executed_path_length_m",
    "remaining_path_lower_bound_m",
    "straight_line_distance_m",
    "path_stretch",
    "path_length_contract_id",
    "plan_path_max_m",
    "teacher_path_length_contract_id",
    "teacher_plan_path_length_m",
    "teacher_plan_path_stretch",
    "mission_route_store_contract_id",
    "mission_route_store_schema_version",
    "mission_route_store",
    "mission_route_store_candidate_index_sha256",
    "mission_route_store_points_sha256",
    "mission_route_store_offsets_sha256",
    "mission_route_store_total_route_points",
)


def normalized_task_identity(row: Mapping[str, Any], *, map_id: str = "forest") -> str:
    """Return an identity based on normalized task geometry, not file order."""

    return mission_id_from_values(
        row["start_x"],
        row["start_y"],
        row["start_z"],
        row.get("start_yaw_deg", 0.0),
        row["goal_x"],
        row["goal_y"],
        row["goal_z"],
        map_id=map_id,
    )


def identity_record(row: Mapping[str, Any], *, map_id: str = "forest") -> Dict[str, str]:
    """Expose both declared and normalized identities for audit artifacts."""

    return {
        "declared_mission_id": str(row.get("mission_id", "")),
        "normalized_task_identity": normalized_task_identity(row, map_id=map_id),
        "episode_id": str(row.get("episode_id", "")),
    }


def is_formal_teacher_mission(row: Mapping[str, Any]) -> bool:
    """Require the fields emitted by the audited formal mission pipeline."""

    return all(str(row.get(field, "")).strip() != "" for field in FORMAL_FIELDS)


def identity_set(rows: Iterable[Mapping[str, Any]], *, map_id: str = "forest") -> Set[str]:
    result = set()
    for row in rows:
        result.add(normalized_task_identity(row, map_id=map_id))
    return result


def select_confirmation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_identities: Set[str],
    excluded_identity_status: str,
    count: int,
    seed: int,
    map_id: str = "forest",
) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """Select a deterministic, formal-only confirmation set fail-closed.

    ``excluded_identity_status`` must be ``COMPLETE``.  Treating an unknown
    exclusion set as empty would silently turn an unverified overlap into a
    zero-overlap claim, which this function explicitly rejects.
    """

    if excluded_identity_status != "COMPLETE":
        raise ValueError("confirmation task exclusion identity is not complete")
    if int(count) <= 0:
        raise ValueError("confirmation count must be positive")
    eligible = []
    rejected = defaultdict(int)
    seen = set()
    for row in rows:
        identity = normalized_task_identity(row, map_id=map_id)
        if identity in seen:
            rejected["duplicate_normalized_identity"] += 1
            continue
        seen.add(identity)
        if not is_formal_teacher_mission(row):
            rejected["missing_formal_teacher_or_route_store_fields"] += 1
            continue
        if identity in excluded_identities:
            rejected["excluded_identity"] += 1
            continue
        eligible.append((identity, row))

    # Stable selection independent of CSV ordering and Python hash randomization.
    def rank(item):
        identity, _ = item
        digest = hashlib.sha256((str(int(seed)) + ":" + identity).encode("utf-8")).hexdigest()
        return digest, identity

    ranked = sorted(eligible, key=rank)
    selected = [row for _, row in ranked[: int(count)]]
    audit = {
        "requested_count": int(count),
        "candidate_rows": int(len(rows)),
        "formal_candidate_rows": int(len(eligible)),
        "eligible_count": int(len(ranked)),
        "selected_count": int(len(selected)),
        "shortfall": int(max(0, int(count) - len(selected))),
        "selection_seed": int(seed),
        "rejections": dict(sorted(rejected.items())),
        "selection_status": "PASS" if len(selected) >= int(count) else "BLOCKED_INSUFFICIENT_FORMAL_TASKS",
        "selected_identities": [normalized_task_identity(row, map_id=map_id) for row in selected],
    }
    return selected, audit


def derive_episode_seed(root_seed: int, task_identity: str, ordinal: int) -> int:
    """Derive a worker-order-independent per-episode RNG seed."""

    digest = hashlib.sha256(
        (str(int(root_seed)) + ":" + str(task_identity) + ":" + str(int(ordinal))).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def episode_budget_audit(*, count: int, max_steps: int, max_decision_steps: int) -> Dict[str, Any]:
    requested = int(count)
    steps = int(max_steps)
    budget = int(max_decision_steps)
    if requested < 0 or steps <= 0 or budget < 0:
        raise ValueError("invalid confirmation episode budget")
    return {
        "episode_limit": requested,
        "max_steps_per_episode": steps,
        "decision_step_limit": budget,
        "worst_case_steps": requested * steps,
        "within_step_budget": requested * steps <= budget,
        "normal_collision_or_timeout_is_valid": True,
        "infrastructure_failure_is_not_a_complete_episode": True,
    }


def rank_correlation(values: Sequence[float], targets: Sequence[float]) -> Optional[float]:
    """Small dependency-free Spearman helper for confirmation reports/tests."""

    if len(values) != len(targets) or len(values) < 2:
        return None

    def ranks(items):
        order = sorted(range(len(items)), key=lambda index: (items[index], index))
        result = [0.0] * len(items)
        cursor = 0
        while cursor < len(order):
            end = cursor + 1
            while end < len(order) and items[order[end]] == items[order[cursor]]:
                end += 1
            rank = 0.5 * (cursor + end - 1)
            for position in range(cursor, end):
                result[order[position]] = rank
            cursor = end
        return result

    left = ranks([float(value) for value in values])
    right = ranks([float(value) for value in targets])
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = sum((a - left_mean) ** 2 for a in left)
    right_norm = sum((b - right_mean) ** 2 for b in right)
    denominator = (left_norm * right_norm) ** 0.5
    return None if denominator <= 0.0 else numerator / denominator


def cluster_bootstrap_delta(
    left: Mapping[str, Sequence[float]],
    right: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
    *,
    seed: int,
    repeats: int,
) -> Dict[str, Any]:
    """Bootstrap a paired metric by resampling mission clusters."""

    names = sorted(set(left) & set(right) & set(targets))
    if not names:
        return {"estimate": None, "ci95": None, "valid": 0, "cluster_count": 0}

    def flatten(selected):
        prediction_left = []
        prediction_right = []
        actual = []
        for name in selected:
            prediction_left.extend(left[name])
            prediction_right.extend(right[name])
            actual.extend(targets[name])
        return prediction_left, prediction_right, actual

    base_left, base_right, base_target = flatten(names)
    base_left_rho = rank_correlation(base_left, base_target)
    base_right_rho = rank_correlation(base_right, base_target)
    estimate = None if base_left_rho is None or base_right_rho is None else base_left_rho - base_right_rho
    rng = __import__("numpy").random.RandomState(int(seed))
    values = []
    for _ in range(int(repeats)):
        chosen = [names[int(index)] for index in rng.randint(0, len(names), size=len(names))]
        sample_left, sample_right, sample_target = flatten(chosen)
        left_rho = rank_correlation(sample_left, sample_target)
        right_rho = rank_correlation(sample_right, sample_target)
        if left_rho is not None and right_rho is not None:
            values.append(left_rho - right_rho)
    if not values:
        ci = None
    else:
        values.sort()
        def percentile(q):
            position = (len(values) - 1) * q
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            weight = position - lower
            return values[lower] * (1.0 - weight) + values[upper] * weight
        ci = [percentile(0.025), percentile(0.975)]
    return {
        "estimate": estimate,
        "ci95": ci,
        "valid": len(values),
        "requested": int(repeats),
        "cluster_count": len(names),
        "seed": int(seed),
    }
