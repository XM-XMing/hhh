"""Offline Critic ranking diagnostics for the multi-action replay.

The module is intentionally independent of the production AWAC learner.  It
owns only pair labels, mission-grouped folds, ranking loss, deterministic
batch schedules, and cluster bootstrap statistics.  It never creates a
runtime and never mutates a production Replay or checkpoint.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np


RANKING_SCHEMA_ID = "multi_action_ranking_critic_v1"


def ranking_enabled_for_branch(branch_name: str) -> bool:
    """Return the preregistered ranking-loss flag for a training branch."""

    flags = {
        "C1_CONTINUED_TD": False,
        "C2_TD_PLUS_RANK": True,
    }
    try:
        return flags[str(branch_name)]
    except KeyError as exc:
        raise ValueError("unknown ranking branch: {}".format(branch_name)) from exc


def build_pair_label(left_return: float, right_return: float) -> int:
    """Return the ranking label using only observed branch returns."""

    left = float(left_return)
    right = float(right_return)
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("pair returns must be finite")
    if left > right:
        return 1
    if left < right:
        return -1
    return 0


def _pair_type(left: str, right: str) -> str:
    sources = frozenset((str(left), str(right)))
    if sources == frozenset(("BC", "NEIGHBOR")):
        return "BC_vs_NEIGHBOR"
    if sources == frozenset(("BC", "RANDOM")):
        return "BC_vs_RANDOM"
    if sources == frozenset(("NEIGHBOR", "RANDOM")):
        return "NEIGHBOR_vs_RANDOM"
    if "TEACHER" in sources:
        return "CONTAINS_TEACHER"
    return "OTHER"


def build_pair_records(
    branches: Sequence[Mapping[str, Any]],
    row_lookup: Mapping[tuple, int],
) -> List[Dict[str, Any]]:
    """Build all same-state pairs from completed branch returns.

    ``row_lookup`` maps ``(state_id, action)`` to the immutable multi-action
    replay row.  The function refuses incomplete branches, duplicate actions,
    identity disagreement, and fabricated labels.
    """

    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for branch in branches:
        if str(branch.get("status", "")) != "completed":
            raise ValueError("ranking labels require completed branches")
        state_id = str(branch.get("state_id", "")).strip()
        mission_id = str(branch.get("mission_id", "")).strip()
        if not state_id or not mission_id:
            raise ValueError("branch identity is incomplete")
        episode_return = float(branch.get("episode_return"))
        if not math.isfinite(episode_return):
            raise ValueError("branch return is non-finite")
        grouped[state_id].append(branch)

    output: List[Dict[str, Any]] = []
    for state_id in sorted(grouped):
        state_rows = sorted(
            grouped[state_id],
            key=lambda row: (int(row.get("action", -1)), str(row.get("action_source", ""))),
        )
        actions = [int(row.get("action", -1)) for row in state_rows]
        if len(set(actions)) != len(actions):
            raise ValueError("duplicate action branch for state {}".format(state_id))
        mission_ids = {str(row.get("mission_id", "")) for row in state_rows}
        step_ids = {int(row.get("step_id", -1)) for row in state_rows}
        if len(mission_ids) != 1 or len(step_ids) != 1:
            raise ValueError("same-state branches disagree on mission/step")
        for left, right in itertools.combinations(state_rows, 2):
            left_action = int(left["action"])
            right_action = int(right["action"])
            key_left = (state_id, left_action)
            key_right = (state_id, right_action)
            if key_left not in row_lookup or key_right not in row_lookup:
                raise ValueError("pair branch is absent from replay")
            left_return = float(left["episode_return"])
            right_return = float(right["episode_return"])
            label = build_pair_label(left_return, right_return)
            output.append(
                {
                    "state_id": state_id,
                    "mission_id": str(left["mission_id"]),
                    "step_id": int(left["step_id"]),
                    "row_i": int(row_lookup[key_left]),
                    "row_j": int(row_lookup[key_right]),
                    "action_i": left_action,
                    "action_j": right_action,
                    "source_i": str(left.get("action_source", "")),
                    "source_j": str(right.get("action_source", "")),
                    "return_i": left_return,
                    "return_j": right_return,
                    "label": int(label),
                    "tie": bool(label == 0),
                    "pair_type": _pair_type(
                        str(left.get("action_source", "")),
                        str(right.get("action_source", "")),
                    ),
                    "failure_type": (
                        "dead_end"
                        if bool(left.get("dead_end", False))
                        else "collision"
                        if bool(left.get("collision", False))
                        else "other"
                    ),
                }
            )
    return output


def build_mission_group_split(
    pairs: Sequence[Mapping[str, Any]], *, seed: int, fold_count: int = 3
) -> Dict[str, Any]:
    """Assign complete missions to deterministic, approximately stratified folds."""

    if int(fold_count) < 2:
        raise ValueError("fold_count must be at least 2")
    mission_data: Dict[str, Dict[str, Any]] = {}
    for pair in pairs:
        mission_id = str(pair.get("mission_id", "")).strip()
        if not mission_id:
            raise ValueError("pair mission identity is missing")
        item = mission_data.setdefault(
            mission_id,
            {"mission_id": mission_id, "failure_type": str(pair.get("failure_type", "other")), "states": set(), "pair_count": 0},
        )
        if str(pair.get("failure_type", "other")) != item["failure_type"]:
            raise ValueError("mission has multiple failure types")
        item["states"].add(str(pair["state_id"]))
        item["pair_count"] += 1

    rng = np.random.RandomState(int(seed))
    by_type: Dict[str, List[str]] = defaultdict(list)
    for mission_id, item in mission_data.items():
        by_type[str(item["failure_type"])].append(mission_id)
    assignment: Dict[str, int] = {}
    fold_load = [0] * int(fold_count)
    for failure_type in sorted(by_type):
        mission_ids = sorted(by_type[failure_type])
        if mission_ids:
            order = rng.permutation(len(mission_ids)).tolist()
            mission_ids = [mission_ids[index] for index in order]
        for mission_id in mission_ids:
            fold = min(range(int(fold_count)), key=lambda value: (fold_load[value], value))
            assignment[mission_id] = int(fold)
            fold_load[fold] += int(len(mission_data[mission_id]["states"]))

    mission_rows = []
    for mission_id in sorted(mission_data):
        item = mission_data[mission_id]
        mission_rows.append(
            {
                "mission_id": mission_id,
                "failure_type": item["failure_type"],
                "state_count": int(len(item["states"])),
                "transition_count": int(len(item["states"])),
                "pair_count": int(item["pair_count"]),
                "fold_id": int(assignment[mission_id]),
            }
        )
    pair_assignments = [
        {
            "pair_index": int(index),
            "mission_id": str(pair["mission_id"]),
            "state_id": str(pair["state_id"]),
            "fold_id": int(assignment[str(pair["mission_id"])]),
            "tie": bool(pair.get("tie", False)),
        }
        for index, pair in enumerate(pairs)
    ]
    return {
        "schema_id": "mission_group_split_v1",
        "seed": int(seed),
        "fold_count": int(fold_count),
        "mission_count": int(len(mission_rows)),
        "missions": mission_rows,
        "mission_to_fold": assignment,
        "pair_assignments": pair_assignments,
        "fold_state_counts": {
            str(fold): int(sum(row["state_count"] for row in mission_rows if row["fold_id"] == fold))
            for fold in range(int(fold_count))
        },
    }


def ranking_loss(q1_i, q1_j, q2_i, q2_j, labels, *, q_scale: float):
    """Twin-head softplus pairwise loss; labels must be non-tie +/-1."""

    import torch

    scale = float(q_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("q_scale must be finite and positive")
    labels = labels.reshape(-1).to(dtype=q1_i.dtype)
    if labels.numel() == 0 or bool((labels == 0).any().item()):
        raise ValueError("tie pairs cannot enter ranking loss")
    if not bool(torch.all(torch.abs(labels) == 1).item()):
        raise ValueError("ranking labels must be +/-1")
    head1 = torch.nn.functional.softplus(-labels * (q1_i - q1_j) / scale).mean()
    head2 = torch.nn.functional.softplus(-labels * (q2_i - q2_j) / scale).mean()
    return 0.5 * (head1 + head2)


def compute_q_scale(abs_q_differences: Sequence[float]) -> float:
    """Return the preregistered per-fold ranking scale from step-zero Qs."""

    values = np.asarray(abs_q_differences, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("q differences must be finite and non-empty")
    return float(max(float(np.median(values)), 1.0e-3))


def lambda_rank_from_grad_norms(
    td_grad_norm: float, rank_grad_norm: float, *, lower: float = 0.01, upper: float = 10.0
) -> float:
    """Compute the one-shot train-fold ranking-loss gradient calibration."""

    td = float(td_grad_norm)
    rank = float(rank_grad_norm)
    low = float(lower)
    high = float(upper)
    if not all(math.isfinite(value) for value in (td, rank, low, high)):
        raise ValueError("gradient norms and bounds must be finite")
    if td < 0.0 or rank < 0.0 or not 0.0 < low <= high:
        raise ValueError("invalid gradient norm or lambda bounds")
    return float(np.clip(0.25 * td / max(rank, 1.0e-12), low, high))


def sample_pair_schedule(
    mission_state_pairs: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    updates: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Sample mission, then state, then pair uniformly with a fixed RNG."""

    if int(updates) <= 0 or int(batch_size) <= 0:
        raise ValueError("updates and batch_size must be positive")
    missions = sorted(str(value) for value in mission_state_pairs)
    if not missions:
        raise ValueError("pair schedule has no missions")
    rng = np.random.RandomState(int(seed))
    schedule = np.empty((int(updates), int(batch_size)), dtype=np.int64)
    for update in range(int(updates)):
        for column in range(int(batch_size)):
            mission = missions[int(rng.randint(0, len(missions)))]
            states = sorted(str(value) for value in mission_state_pairs[mission])
            if not states:
                raise ValueError("mission has no states")
            state = states[int(rng.randint(0, len(states)))]
            pairs = list(mission_state_pairs[mission][state])
            if not pairs:
                raise ValueError("state has no pairs")
            schedule[update, column] = int(pairs[int(rng.randint(0, len(pairs)))])
    return schedule


def mission_cluster_bootstrap(
    values: Mapping[str, np.ndarray],
    clusters: Sequence[str],
    *,
    left: str,
    right: str,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap paired differences by mission cluster, not by raw pair."""

    if left not in values or right not in values:
        raise KeyError("bootstrap models are missing")
    left_values = np.asarray(values[left], dtype=np.float64).reshape(-1)
    right_values = np.asarray(values[right], dtype=np.float64).reshape(-1)
    cluster_array = np.asarray([str(value) for value in clusters])
    if left_values.size == 0 or left_values.size != right_values.size or left_values.size != cluster_array.size:
        raise ValueError("bootstrap arrays have incompatible sizes")
    names = sorted(set(cluster_array.tolist()))
    if not names:
        raise ValueError("bootstrap requires clusters")
    grouped = {name: np.flatnonzero(cluster_array == name) for name in names}
    rng = np.random.RandomState(int(seed))
    samples = []
    for _ in range(int(repeats)):
        selected = rng.randint(0, len(names), size=len(names))
        indices = np.concatenate([grouped[names[index]] for index in selected])
        samples.append(float(np.mean(left_values[indices] - right_values[indices])))
    array = np.asarray(samples, dtype=np.float64)
    return {
        "left": str(left),
        "right": str(right),
        "seed": int(seed),
        "repeats": int(repeats),
        "cluster_count": int(len(names)),
        "estimate": float(np.mean(left_values - right_values)),
        "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        "samples": [float(value) for value in array.tolist()],
    }


def assert_state_dict_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    """Assert exact tensor/scalar equality for immutable-model proofs."""

    import torch

    if set(left) != set(right):
        raise AssertionError("state-dict keys differ")
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not torch.equal(a, b):
                raise AssertionError("state-dict value differs: {}".format(key))
        elif a != b:
            raise AssertionError("state-dict value differs: {}".format(key))


def validate_no_mc_target_without_sequence(*, has_complete_step_sequence: bool) -> bool:
    """Prevent a pairwise return label from becoming a fabricated MC target."""

    if not bool(has_complete_step_sequence):
        raise ValueError("complete step reward sequence is required for MC regression target")
    return True


def rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("rank data must be finite and non-empty")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]):
    x = rankdata(left)
    y = rankdata(right)
    if x.size != y.size or x.size < 2:
        return None
    x -= x.mean()
    y -= y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return None if denominator <= 0.0 else float(np.dot(x, y) / denominator)


def kendall(left: Sequence[float], right: Sequence[float]):
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return None
    concordant = discordant = 0
    for i, j in itertools.combinations(range(x.size), 2):
        product = (x[i] - x[j]) * (y[i] - y[j])
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
    denominator = concordant + discordant
    return None if denominator == 0 else float((concordant - discordant) / denominator)


__all__ = [
    "RANKING_SCHEMA_ID",
    "ranking_enabled_for_branch",
    "assert_state_dict_equal",
    "build_mission_group_split",
    "build_pair_label",
    "build_pair_records",
    "kendall",
    "compute_q_scale",
    "lambda_rank_from_grad_norms",
    "mission_cluster_bootstrap",
    "ranking_loss",
    "sample_pair_schedule",
    "spearman",
    "validate_no_mc_target_without_sequence",
]
