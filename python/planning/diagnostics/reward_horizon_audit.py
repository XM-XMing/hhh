"""Read-only reward-horizon and credit-assignment audit.

The multi-action diagnostic artifact contains one candidate-action reward and a
completed branch return for each state/action branch.  It does *not* contain a
step-by-step reward sequence or the prefix return.  This module keeps that
boundary explicit: it reports observable immediate-reward and terminal-branch
surrogates, and marks exact multi-step ``G_t`` and reward-density quantities as
unavailable instead of reconstructing them from incomplete data.

No production replay, reward contract, learner, checkpoint, or runtime is
loaded or modified.  The optional ranking models are small CPU diagnostics
trained with mission-level cross-fitting only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from planning.diagnostics.state_sufficiency import (
    ACTION_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_HIDDEN_LAYERS,
    _SmallMLP,
    _Standardizer,
    _one_hot,
    binary_metrics,
    classification_bundle,
    load_audit_dataset,
    mission_cluster_bootstrap,
    sha256_file,
)


REWARD_HORIZON_AUDIT_SCHEMA_ID = "reward_horizon_credit_assignment_audit_v1"
REWARD_HORIZON_METRICS_SCHEMA_ID = "reward_horizon_metrics_v1"
SHORT_HORIZON_SCHEMA_ID = "short_horizon_surrogate_metrics_v1"
REWARD_HORIZON_MODEL_SCHEMA_ID = "reward_horizon_model_config_v1"
REWARD_HORIZON_BOOTSTRAP_SCHEMA_ID = "reward_horizon_bootstrap_v1"
DEFAULT_SEED = 20260910
DEFAULT_BOOTSTRAP_REPEATS = 5000


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "variance": None,
            "p10": None,
            "p90": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("non-finite reward-horizon statistic input")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "variance": float(np.var(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1.0e-12:
        return None
    return float(np.dot(x, y) / denominator)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    index = 0
    while index < array.size:
        end = index + 1
        while end < array.size and array[order[end]] == array[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * float(index + end - 1) + 1.0
        index = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_rankdata(left), _rankdata(right))


def _step_bucket(step_id: int) -> str:
    value = int(step_id)
    if value < 5:
        return "step_0_5"
    if value < 10:
        return "step_5_10"
    if value < 20:
        return "step_10_20"
    return "step_20_plus"


def build_mission_level_split(
    mission_ids: Iterable[str], *, seed: int, fold_count: int = 3
) -> Dict[str, Any]:
    """Create a deterministic mission-only split for small public test seams."""

    names = sorted({str(value) for value in mission_ids})
    if not names:
        raise ValueError("mission split requires at least one mission")
    if int(fold_count) < 2:
        raise ValueError("fold_count must be at least two")
    rng = np.random.RandomState(int(seed))
    shuffled = [names[index] for index in rng.permutation(len(names)).tolist()]
    loads = [0 for _ in range(int(fold_count))]
    assignment: Dict[str, int] = {}
    for mission_id in shuffled:
        fold = min(range(int(fold_count)), key=lambda candidate: (loads[candidate], candidate))
        assignment[mission_id] = int(fold)
        loads[fold] += 1
    return {
        "schema_id": "reward_horizon_mission_split_v1",
        "seed": int(seed),
        "fold_count": int(fold_count),
        "mission_count": len(names),
        "mission_to_fold": assignment,
        "fold_loads": loads,
    }


def reward_horizon_availability(
    rows: Sequence[Mapping[str, Any]], *, has_complete_step_reward_sequence: bool
) -> Dict[str, Any]:
    """Describe exact versus surrogate reward targets without fabricating ``G_t``."""

    rewards = np.asarray([float(row["immediate_reward"]) for row in rows], dtype=np.float64)
    if rewards.size and not np.isfinite(rewards).all():
        raise ValueError("immediate rewards must be finite")
    requested: Dict[str, Dict[str, Any]] = {}
    if has_complete_step_reward_sequence:
        for horizon in (1, 5, 10, 20):
            requested[str(horizon)] = {
                "status": "AVAILABLE",
                "exact_future_return": True,
                "target": "G_t_from_step_reward_sequence",
            }
    else:
        requested["1"] = {
            "status": "AVAILABLE",
            "exact_future_return": True,
            "target": "G_t_1 = recorded_candidate_action_reward",
        }
        for horizon in (5, 10, 20):
            requested[str(horizon)] = {
                "status": "UNAVAILABLE",
                "exact_future_return": False,
                "reason": "complete step reward sequence is not persisted",
            }
    return {
        "exact_future_return_available": bool(has_complete_step_reward_sequence),
        "exact_one_step_return_available": bool(rewards.size > 0),
        "prefix_return_available": False,
        "requested_horizons": requested,
        "observed_action_reward": {
            "count": int(rewards.size),
            "nonzero_count": int(np.count_nonzero(rewards != 0.0)),
            "nonzero_rate": float(np.mean(rewards != 0.0)) if rewards.size else None,
            "statistics": _finite_stats(rewards),
        },
        "terminal_branch_outcome": {
            "status": "OBSERVED_BRANCH_OUTCOME",
            "exact_G_t": False,
            "includes_prefix_return": True,
            "field": "episode_return",
            "reason": "episode_return is prefix + candidate action + BC continuation cumulative return",
        },
        "episode_nonzero_reward_count": {
            "status": "UNAVAILABLE",
            "value": None,
            "reason": "per-step reward sequence is not persisted",
        },
        "terminal_reward_share": {
            "status": "UNAVAILABLE",
            "value": None,
            "reason": "terminal reward is not separately persisted",
        },
        "reward_sparsity": {
            "status": "UNAVAILABLE",
            "value": None,
            "reason": "per-step reward sequence is not persisted",
        },
    }


def _source_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["action_source"])].append(row)
    result: Dict[str, Any] = {}
    for source, values in sorted(grouped.items()):
        state_groups: MutableMapping[str, List[float]] = defaultdict(list)
        for row in values:
            state_groups[str(row["state_id"])].append(float(row["terminal_return"]))
        variances = [float(np.var(value)) for value in state_groups.values() if len(value) >= 2]
        success = np.asarray([int(bool(row["success"])) for row in values], dtype=np.float64)
        result[source] = {
            "row_count": int(len(values)),
            "state_count": int(len(state_groups)),
            "success_rate": float(np.mean(success)) if success.size else None,
            "terminal_return": _finite_stats([float(row["terminal_return"]) for row in values]),
            "immediate_action_reward": _finite_stats(
                [float(row["immediate_reward"]) for row in values]
            ),
            "multi_action_state_count": int(len(variances)),
            "within_state_terminal_return_variance": _finite_stats(variances),
        }
    return result


def within_state_outcome_statistics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize branch outcome variance by state and action source."""

    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    if not grouped:
        raise ValueError("within-state outcome audit has no rows")
    per_state: Dict[str, Any] = {}
    variances: List[float] = []
    success_variances: List[float] = []
    for state_id, values in sorted(grouped.items()):
        returns = [float(row["terminal_return"]) for row in values]
        successes = [int(bool(row["success"])) for row in values]
        terminal_variance = float(np.var(returns))
        success_variance = float(np.var(successes))
        variances.append(terminal_variance)
        success_variances.append(success_variance)
        per_state[state_id] = {
            "action_count": int(len({int(row["action"]) for row in values})),
            "branch_count": int(len(values)),
            "terminal_return": _finite_stats(returns),
            "terminal_return_variance": terminal_variance,
            "success_rate": float(np.mean(successes)),
            "success_variance": success_variance,
            "sources": dict(Counter(str(row["action_source"]) for row in values)),
        }
    return {
        "schema_id": "reward_horizon_within_state_outcome_v1",
        "state_count": int(len(grouped)),
        "multi_action_state_count": int(sum(len(values) >= 2 for values in grouped.values())),
        "single_action_state_count": int(sum(len(values) < 2 for values in grouped.values())),
        "terminal_return_variance": _finite_stats(variances),
        "success_variance": _finite_stats(success_variances),
        "sources": _source_stats(rows),
        "per_state": per_state,
    }


def _correlation_record(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "state_count": 0,
            "mission_count": 0,
            "success_rate": None,
            "immediate_reward_vs_terminal_return_pearson": None,
            "immediate_reward_vs_success_pearson": None,
            "terminal_return_vs_success_pearson": None,
            "within_state_spearman_median": None,
        }
    immediate = [float(row["immediate_reward"]) for row in rows]
    terminal = [float(row["terminal_return"]) for row in rows]
    success = [int(bool(row["success"])) for row in rows]
    by_state: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(row)
    state_correlations = []
    for values in by_state.values():
        correlation = _spearman(
            [float(row["immediate_reward"]) for row in values],
            [float(row["terminal_return"]) for row in values],
        )
        if correlation is not None:
            state_correlations.append(correlation)
    return {
        "row_count": int(len(rows)),
        "state_count": int(len(by_state)),
        "mission_count": int(len({str(row["mission_id"]) for row in rows})),
        "success_rate": float(np.mean(success)),
        "immediate_reward_vs_terminal_return_pearson": _pearson(immediate, terminal),
        "immediate_reward_vs_success_pearson": _pearson(immediate, success),
        "terminal_return_vs_success_pearson": _pearson(terminal, success),
        "within_state_spearman_median": (
            float(np.median(state_correlations)) if state_correlations else None
        ),
    }


def _pairwise_order_agreement(
    pairs: Sequence[Mapping[str, Any]],
    reward_by_state_action: Mapping[Tuple[str, int], float],
) -> Dict[str, Any]:
    considered = 0
    agreed = 0
    immediate_ties = 0
    terminal_ties = 0
    for pair in pairs:
        terminal_delta = float(pair["return1"]) - float(pair["return2"])
        if terminal_delta == 0.0:
            terminal_ties += 1
            continue
        key1 = (str(pair["state_id"]), int(pair["action1"]))
        key2 = (str(pair["state_id"]), int(pair["action2"]))
        if key1 not in reward_by_state_action or key2 not in reward_by_state_action:
            raise ValueError("pair does not join to immediate reward")
        immediate_delta = reward_by_state_action[key1] - reward_by_state_action[key2]
        if immediate_delta == 0.0:
            immediate_ties += 1
            continue
        considered += 1
        agreed += int(immediate_delta * terminal_delta > 0.0)
    return {
        "considered_pair_count": int(considered),
        "agreement_count": int(agreed),
        "agreement_rate": float(agreed / considered) if considered else None,
        "immediate_reward_tie_count": int(immediate_ties),
        "terminal_return_tie_count": int(terminal_ties),
    }


def _load_reward_rows(root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dataset = load_audit_dataset(root)
    transition_path = Path(root) / "replay" / "transitions.npz"
    branch_path = Path(root) / "candidate_branches.jsonl"
    with np.load(str(transition_path), allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    branches: List[Dict[str, Any]] = []
    with branch_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("branch line {} is not an object".format(line_number))
                branches.append(value)
    branch_by_key = {
        (str(row["state_id"]), int(row["action"])): row
        for row in branches
        if str(row.get("status", "")) == "completed"
    }
    rows: List[Dict[str, Any]] = []
    for index, transition in enumerate(dataset["transitions"]):
        key = (str(transition["state_id"]), int(transition["action"]))
        branch = branch_by_key.get(key)
        if branch is None:
            raise ValueError("missing completed branch for {}".format(key))
        rows.append(
            {
                "index": int(index),
                "state_id": str(transition["state_id"]),
                "mission_id": str(transition["mission_id"]),
                "episode_id": str(transition["episode_id"]),
                "step_id": int(transition["step_id"]),
                "action": int(transition["action"]),
                "action_source": str(transition["action_source"]),
                "immediate_reward": float(arrays["reward"][index]),
                "terminal_return": float(branch["episode_return"]),
                "success": bool(branch.get("success", False)),
                "collision": bool(branch.get("collision", False)),
                "dead_end": bool(branch.get("dead_end", False)),
                "timeout": bool(branch.get("timeout", False)),
                "terminal_reason": str(branch.get("terminal_reason", "unknown")),
                "steps": int(branch.get("steps", -1)),
                "state_feature": np.asarray(transition["state_feature"], dtype=np.float32),
            }
        )
    if not np.isfinite(np.asarray([row["immediate_reward"] for row in rows])).all():
        raise ValueError("dataset contains non-finite immediate reward")
    if not np.isfinite(np.asarray([row["terminal_return"] for row in rows])).all():
        raise ValueError("dataset contains non-finite terminal return")
    return dataset, rows


def _validate_prior_audits(
    dataset: Mapping[str, Any], state_audit_root: Path, observation_audit_root: Path
) -> Dict[str, Any]:
    identity = dataset["identity"]
    state_identity_path = Path(state_audit_root) / "dataset_identity.json"
    observation_identity_path = Path(observation_audit_root) / "dataset_identity.json"
    state_identity = json.loads(state_identity_path.read_text(encoding="utf-8"))
    observation_identity = json.loads(observation_identity_path.read_text(encoding="utf-8"))
    for name, prior in (("state_sufficiency", state_identity), ("observation_information", observation_identity)):
        prior_manifest = prior.get("manifest_sha256", prior.get("dataset_manifest_sha256"))
        if prior_manifest is not None and prior_manifest != identity.get("manifest_sha256"):
            raise ValueError("{} audit identity mismatch for manifest_sha256".format(name))
        for field in ("transitions_sha256", "pairwise_sha256"):
            if prior.get(field) is not None and prior.get(field) != identity.get(field):
                raise ValueError("{} audit identity mismatch for {}".format(name, field))
    split_path = Path(state_audit_root) / "mission_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if int(split.get("fold_count", -1)) != 3:
        raise ValueError("state audit mission split must have three folds")
    return {
        "state_sufficiency_identity": state_identity,
        "observation_information_identity": observation_identity,
        "state_sufficiency_identity_sha256": sha256_file(state_identity_path),
        "observation_information_identity_sha256": sha256_file(observation_identity_path),
        "mission_split": split,
        "mission_split_sha256": sha256_file(split_path),
    }


def _pair_features(pairs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not pairs:
        raise ValueError("pair feature construction requires rows")
    states = np.stack([np.asarray(row["state_feature"], dtype=np.float32) for row in pairs])
    actions1 = _one_hot(np.asarray([int(row["action1"]) for row in pairs], dtype=np.int64))
    actions2 = _one_hot(np.asarray([int(row["action2"]) for row in pairs], dtype=np.int64))
    return np.concatenate((states, actions1, actions2), axis=1).astype(np.float32)


def _fit_pair_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> np.ndarray:
    normalizer = _Standardizer.fit(train_x)
    train_values = normalizer.transform(train_x)
    test_values = normalizer.transform(test_x)
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    model = _SmallMLP(train_values.shape[1], DEFAULT_HIDDEN_DIM, DEFAULT_HIDDEN_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    x_tensor = torch.from_numpy(train_values)
    y_tensor = torch.from_numpy(np.asarray(train_y, dtype=np.float32))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 17)
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(x_tensor.shape[0], generator=generator)
        for start in range(0, x_tensor.shape[0], int(batch_size)):
            indices = order[start : start + int(batch_size)]
            logits = model(x_tensor[indices]).reshape(-1)
            loss = loss_fn(logits, y_tensor[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(test_values)).reshape(-1)).numpy()


def _cross_fitted_target(
    pairs: Sequence[Mapping[str, Any]],
    *,
    target_name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    fold_count: int = 3,
) -> Dict[str, Any]:
    rows = [dict(row) for row in pairs]
    labels = np.asarray([int(row["target_label"]) for row in rows], dtype=np.int64)
    missions = np.asarray([str(row["mission_id"]) for row in rows])
    states = np.asarray([str(row["state_id"]) for row in rows])
    folds = np.asarray([int(row["fold_id"]) for row in rows], dtype=np.int64)
    features = _pair_features(rows)
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    fold_records = []
    for fold in range(int(fold_count)):
        train_indices = np.flatnonzero(folds != fold)
        test_indices = np.flatnonzero(folds == fold)
        if not len(train_indices) or not len(test_indices):
            raise ValueError("empty train/test fold {} for {}".format(fold, target_name))
        predictions[test_indices] = _fit_pair_model(
            features[train_indices],
            labels[train_indices],
            features[test_indices],
            seed=int(seed) + fold * 101,
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
        fold_records.append(
            {
                "fold_id": int(fold),
                "train_count": int(len(train_indices)),
                "test_count": int(len(test_indices)),
                "metrics": classification_bundle(
                    labels[test_indices], predictions[test_indices], states[test_indices], missions[test_indices]
                ),
            }
        )
    result = {
        "target": str(target_name),
        "pair_count": int(len(rows)),
        "positive_count": int(np.count_nonzero(labels == 1)),
        "pair_input_dim": int(features.shape[1]),
        "folds": fold_records,
        "oof_metrics": classification_bundle(labels, predictions, states, missions),
        "predictions": predictions,
        "labels": labels,
        "mission_ids": missions,
        "state_ids": states,
    }
    return result


def _build_short_horizon_surrogate(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    *,
    mission_to_fold: Mapping[str, int],
    seed: int,
    bootstrap_repeats: int,
    epochs: int,
    batch_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    reward_map = {(str(row["state_id"]), int(row["action"])): float(row["immediate_reward"]) for row in rows}
    terminal_rows: List[Dict[str, Any]] = []
    immediate_rows: List[Dict[str, Any]] = []
    common_indices: List[int] = []
    for pair_index, pair in enumerate(pairs):
        mission = str(pair["mission_id"])
        if mission not in mission_to_fold:
            raise ValueError("pair mission missing from fixed split: {}".format(mission))
        key1 = (str(pair["state_id"]), int(pair["action1"]))
        key2 = (str(pair["state_id"]), int(pair["action2"]))
        immediate_delta = reward_map[key1] - reward_map[key2]
        terminal_delta = float(pair["return1"]) - float(pair["return2"])
        base = dict(pair)
        base["fold_id"] = int(mission_to_fold[mission])
        base["pair_index"] = int(pair_index)
        if terminal_delta != 0.0:
            terminal = dict(base)
            terminal["target_label"] = int(terminal_delta > 0.0)
            terminal_rows.append(terminal)
        if immediate_delta != 0.0:
            immediate = dict(base)
            immediate["target_label"] = int(immediate_delta > 0.0)
            immediate_rows.append(immediate)
        if terminal_delta != 0.0 and immediate_delta != 0.0:
            common_indices.append(int(pair_index))

    terminal_result = _cross_fitted_target(
        terminal_rows,
        target_name="terminal_branch_return",
        seed=int(seed) + 1000,
        epochs=epochs,
        batch_size=batch_size,
    )
    immediate_result = _cross_fitted_target(
        immediate_rows,
        target_name="immediate_action_reward",
        seed=int(seed) + 2000,
        epochs=epochs,
        batch_size=batch_size,
    )

    terminal_predictions = {
        int(row["pair_index"]): (float(terminal_result["predictions"][index]), int(terminal_result["labels"][index]))
        for index, row in enumerate(terminal_rows)
    }
    immediate_predictions = {
        int(row["pair_index"]): (float(immediate_result["predictions"][index]), int(immediate_result["labels"][index]))
        for index, row in enumerate(immediate_rows)
    }
    common = [
        index
        for index in common_indices
        if index in terminal_predictions and index in immediate_predictions
    ]
    terminal_correct = np.asarray(
        [int((terminal_predictions[index][0] >= 0.5) == bool(terminal_predictions[index][1])) for index in common],
        dtype=np.float64,
    )
    immediate_correct = np.asarray(
        [int((immediate_predictions[index][0] >= 0.5) == bool(immediate_predictions[index][1])) for index in common],
        dtype=np.float64,
    )
    cluster_differences: MutableMapping[str, List[float]] = defaultdict(list)
    for index, terminal_value, immediate_value in zip(
        common,
        terminal_correct.tolist(),
        immediate_correct.tolist(),
    ):
        cluster_differences[str(pairs[index]["mission_id"])].append(
            float(terminal_value - immediate_value)
        )
    paired_bootstrap = mission_cluster_bootstrap(
        cluster_differences,
        repeats=int(bootstrap_repeats),
        seed=int(seed) + 3000,
    )
    paired_bootstrap["left"] = "terminal_branch_return_model_accuracy"
    paired_bootstrap["right"] = "immediate_action_reward_model_accuracy"
    paired_bootstrap["estimate_semantics"] = "terminal minus immediate paired accuracy"
    paired_bootstrap["schema_id"] = REWARD_HORIZON_BOOTSTRAP_SCHEMA_ID
    predictions: List[Dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        predictions.append(
            {
                "pair_index": int(pair_index),
                "state_id": str(pair["state_id"]),
                "mission_id": str(pair["mission_id"]),
                "action1": int(pair["action1"]),
                "action2": int(pair["action2"]),
                "terminal_label": int(pair_index in terminal_predictions and terminal_predictions[pair_index][1]),
                "terminal_prediction": terminal_predictions.get(pair_index, (None, None))[0],
                "immediate_label": int(pair_index in immediate_predictions and immediate_predictions[pair_index][1]),
                "immediate_prediction": immediate_predictions.get(pair_index, (None, None))[0],
            }
        )
    for result in (terminal_result, immediate_result):
        result.pop("predictions", None)
        result.pop("labels", None)
        result.pop("mission_ids", None)
        result.pop("state_ids", None)
    short = {
        "schema_id": SHORT_HORIZON_SCHEMA_ID,
        "label_semantics": {
            "immediate_action_reward": "sign(recorded candidate-action reward difference)",
            "terminal_branch_return": "sign(recorded completed branch episode_return difference)",
            "exact_G_t_available": False,
        },
        "terminal_branch_return_model": terminal_result,
        "immediate_action_reward_model": immediate_result,
        "common_pair_count": int(len(common)),
        "paired_terminal_minus_immediate_accuracy": paired_bootstrap,
    }
    return short, predictions, paired_bootstrap


def _build_horizon_metrics(rows: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "all": list(rows),
        "step_0_5": [row for row in rows if int(row["step_id"]) < 5],
        "step_5_10": [row for row in rows if 5 <= int(row["step_id"]) < 10],
        "step_10_20": [row for row in rows if 10 <= int(row["step_id"]) < 20],
        "step_20_plus": [row for row in rows if int(row["step_id"]) >= 20],
    }
    reward_map = {(str(row["state_id"]), int(row["action"])): float(row["immediate_reward"]) for row in rows}
    bucket_metrics = {}
    for name, values in buckets.items():
        selected_states = {str(row["state_id"]) for row in values}
        selected_pairs = [pair for pair in pairs if str(pair["state_id"]) in selected_states]
        bucket_metrics[name] = {
            "step_range": {
                "min_inclusive": None if not values else int(min(row["step_id"] for row in values)),
                "max_inclusive": None if not values else int(max(row["step_id"] for row in values)),
            },
            "branch_correlations": _correlation_record(values),
            "action_reward_vs_terminal_pairwise": _pairwise_order_agreement(selected_pairs, reward_map),
        }
    terminal_reasons = Counter(str(row["terminal_reason"]) for row in rows)
    return {
        "schema_id": REWARD_HORIZON_METRICS_SCHEMA_ID,
        "branch_count": int(len(rows)),
        "state_count": int(len({str(row["state_id"]) for row in rows})),
        "mission_count": int(len({str(row["mission_id"]) for row in rows})),
        "step_id_range": [int(min(row["step_id"] for row in rows)), int(max(row["step_id"] for row in rows))],
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
        "early_action_predictability": bucket_metrics,
        "action_return_correlation_definition": "immediate candidate-action reward versus recorded completed-branch return; action IDs are not treated as ordered scalars",
    }


def _render_report(
    *,
    identity: Mapping[str, Any],
    availability: Mapping[str, Any],
    horizon_metrics: Mapping[str, Any],
    within_state: Mapping[str, Any],
    short_horizon: Mapping[str, Any],
) -> str:
    all_bucket = horizon_metrics["early_action_predictability"]["all"]
    paired = short_horizon["paired_terminal_minus_immediate_accuracy"]
    source_lines = []
    for source, values in sorted(within_state["sources"].items()):
        source_lines.append(
            "| {} | {} | {:.4f} | {} |".format(
                source,
                values["row_count"],
                values["success_rate"] if values["success_rate"] is not None else float("nan"),
                values["within_state_terminal_return_variance"]["count"],
            )
        )
    requested = availability["requested_horizons"]
    return "\n".join(
        [
            "# Reward Horizon / Credit Assignment Audit V1",
            "",
            "本报告仅分析 diagnostic multi-action replay，不修改 production reward、Replay、模型或运行时；未启动 Unity/Bridge，也未执行任何 RL/Actor/Critic 训练。",
            "",
            "## 数据身份",
            "",
            "- dataset: `{}`；branches={}，states={}，missions={}。".format(
                identity["dataset_id"], identity["transition_count"], identity["state_count"], identity.get("mission_count", 31)
            ),
            "- `production_replay={}`，`training_consumed={}`。".format(identity["production_replay"], identity["training_consumed"]),
            "- pair labels仍来自同 state 的 completed branch `episode_return`；mission-level split复用 state-sufficiency audit，不重新随机拆分。",
            "",
            "## 关键数据边界",
            "",
            "artifact只保存候选动作当步 `reward` 和包含 prefix 的 `episode_return`；没有逐步 reward sequence，也没有 `prefix_return`。因此精确 `G_t`、5/10/20-step return、每集非零 reward 数和 terminal reward 占比均不能合法重建。",
            "",
            "| horizon | 状态 | 说明 |",
            "|---|---|---|",
            "| 1 | {} | `G_t^(1)=r_t`，candidate-action reward 可直接观测 |".format(requested["1"]["status"]),
            "| 5/10/20 | UNAVAILABLE | 未保存逐步 reward sequence |",
            "| terminal | OBSERVED_BRANCH_OUTCOME | `episode_return` 包含 prefix，不等于精确 `G_t` |",
            "",
            "## Early action predictability",
            "",
            "- 全体 immediate reward→terminal branch return Pearson: `{}`。".format(all_bucket["branch_correlations"]["immediate_reward_vs_terminal_return_pearson"]),
            "- 全体 immediate reward→success Pearson: `{}`；terminal return→success Pearson: `{}`。".format(
                all_bucket["branch_correlations"]["immediate_reward_vs_success_pearson"],
                all_bucket["branch_correlations"]["terminal_return_vs_success_pearson"],
            ),
            "- immediate reward 与 terminal return 的同 state pair 顺序一致率：`{}`（{} pairs）。".format(
                all_bucket["action_reward_vs_terminal_pairwise"]["agreement_rate"],
                all_bucket["action_reward_vs_terminal_pairwise"]["considered_pair_count"],
            ),
            "",
            "## Reward density（可证实部分）",
            "",
            "- candidate-action reward 非零率：`{}`。这不是 episode-level reward sparsity。".format(
                availability["observed_action_reward"]["nonzero_rate"]
            ),
            "- episode non-zero reward count / terminal reward share / reward sparsity：`UNAVAILABLE`。",
            "",
            "## Within-state action outcome",
            "",
            "- multi-action states: `{}`；terminal-return variance mean: `{}`；success variance mean: `{}`。".format(
                within_state["multi_action_state_count"],
                within_state["terminal_return_variance"]["mean"],
                within_state["success_variance"]["mean"],
            ),
            "",
            "| source | branches | success rate | states with >=2 source actions |",
            "|---|---:|---:|---:|",
            *source_lines,
            "",
            "## Short-horizon surrogate",
            "",
            "以 candidate-action immediate reward 作为可观测的 1-step surrogate，和 completed branch return 分别训练同结构、mission-level 3-fold cross-fitted ranking MLP；这不是精确 n-step 或 MC target。",
            "",
            "- immediate reward model micro accuracy: `{}`。".format(
                short_horizon["immediate_action_reward_model"]["oof_metrics"]["micro"]["accuracy"]
            ),
            "- terminal branch return model micro accuracy: `{}`。".format(
                short_horizon["terminal_branch_return_model"]["oof_metrics"]["micro"]["accuracy"]
            ),
            "- terminal − immediate paired accuracy delta: `{}`，95% CI `{}`。".format(
                paired["estimate"], paired["ci95"]
            ),
            "",
            "## 结论",
            "",
            "1. reward 是否过稀疏：`UNRESOLVED`；当前 artifact 没有逐步 reward 序列，candidate reward 非零率不足以回答 episode sparsity。",
            "2. 动作影响是否需要长 horizon：`UNRESOLVED`；不能用包含 prefix 的 terminal return 冒充 `G_t`，需要补充逐步 reward/prefix return 证据。",
            "3. 是否需要重新设计 critic target：`NOT_SUPPORTED_BY_THIS_AUDIT`；本轮没有合法 5/10/20-step target。",
            "4. 下一步：先补齐 step-level reward provenance，再决定 A reward shaping / B state augmentation / C hierarchical policy / D 停止 offline RL；本轮不选择其中任何一项。",
            "",
            "结论不构成生产 RL gate、收敛声明或 reward 修改授权。",
            "",
        ]
    )


def run_reward_horizon_audit(
    *,
    dataset_root: Path,
    state_audit_root: Path,
    observation_audit_root: Path,
    out_dir: Path,
    seed: int = DEFAULT_SEED,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Run the offline audit and write only derived diagnostic artifacts."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    state_audit_root = Path(state_audit_root).expanduser().resolve()
    observation_audit_root = Path(observation_audit_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, rows = _load_reward_rows(dataset_root)
    prior = _validate_prior_audits(dataset, state_audit_root, observation_audit_root)
    mission_to_fold = {
        str(mission): int(fold) for mission, fold in prior["mission_split"]["mission_to_fold"].items()
    }
    pairs = [dict(pair) for pair in dataset["pairs"]]
    for pair in pairs:
        pair["fold_id"] = int(mission_to_fold[str(pair["mission_id"])])
    reward_by_key = {(str(row["state_id"]), int(row["action"])): float(row["immediate_reward"]) for row in rows}

    availability = reward_horizon_availability(rows, has_complete_step_reward_sequence=False)
    horizon_metrics = _build_horizon_metrics(rows, pairs)
    within_state = within_state_outcome_statistics(rows)
    short_horizon, prediction_rows, paired_bootstrap = _build_short_horizon_surrogate(
        rows,
        pairs,
        mission_to_fold=mission_to_fold,
        seed=int(seed),
        bootstrap_repeats=int(bootstrap_repeats),
        epochs=int(epochs),
        batch_size=int(batch_size),
    )

    identity = dict(dataset["identity"])
    identity.update(
        {
            "schema_id": REWARD_HORIZON_AUDIT_SCHEMA_ID,
            "mission_count": int(len({str(row["mission_id"]) for row in rows})),
            "branch_count": int(len(rows)),
            "complete_step_reward_sequence_persisted": False,
            "production_replay": False,
            "training_consumed": False,
            "state_audit_root": str(state_audit_root),
            "observation_audit_root": str(observation_audit_root),
        }
    )
    model_config = {
        "schema_id": REWARD_HORIZON_MODEL_SCHEMA_ID,
        "seed": int(seed),
        "fold_count": 3,
        "split": "reuse state_sufficiency_audit_v1 mission_split.json",
        "hidden_dim": int(DEFAULT_HIDDEN_DIM),
        "hidden_layers": int(DEFAULT_HIDDEN_LAYERS),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam(lr=1e-3)",
        "normalizer": "fit train fold only",
        "device": "cpu",
        "q_features_used": False,
        "rl_training": False,
    }
    manifest = {
        "schema_id": REWARD_HORIZON_AUDIT_SCHEMA_ID,
        "dataset_identity": identity,
        "prior_audits": prior,
        "requested_bootstrap_repeats": int(bootstrap_repeats),
        "new_environment_steps": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "unity_started": False,
        "bridge_started": False,
        "production_reward_modified": False,
        "production_replay_modified": False,
        "production_defaults_modified": False,
    }

    _json_dump(out_dir / "dataset_identity.json", identity)
    _json_dump(out_dir / "mission_split.json", prior["mission_split"])
    _json_dump(out_dir / "model_config.json", model_config)
    _json_dump(out_dir / "reward_horizon_metrics.json", {
        "schema_id": REWARD_HORIZON_METRICS_SCHEMA_ID,
        "availability": availability,
        "horizon_analysis": horizon_metrics,
        "within_state_outcome": within_state,
    })
    _json_dump(out_dir / "short_horizon_metrics.json", short_horizon)
    _json_dump(out_dir / "bootstrap.json", paired_bootstrap)
    _json_dump(out_dir / "manifest.json", manifest)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(prediction_rows[0].keys()) if prediction_rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    report = _render_report(
        identity=identity,
        availability=availability,
        horizon_metrics=horizon_metrics,
        within_state=within_state,
        short_horizon=short_horizon,
    )
    (out_dir / "report_zh.md").write_text(report, encoding="utf-8")
    return {
        "manifest": manifest,
        "identity": identity,
        "availability": availability,
        "horizon_metrics": horizon_metrics,
        "within_state": within_state,
        "short_horizon": short_horizon,
        "out_dir": str(out_dir),
    }


__all__ = [
    "build_mission_level_split",
    "reward_horizon_availability",
    "run_reward_horizon_audit",
    "within_state_outcome_statistics",
]
