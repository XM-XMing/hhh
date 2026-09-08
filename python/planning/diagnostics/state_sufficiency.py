"""Independent state/action information audit for the diagnostic replay.

This module is deliberately separate from the production AWAC learner.  It
loads the immutable multi-action artifact, constructs mission-level folds, and
fits small CPU MLPs only to measure whether the recorded observation and an
action contain predictive information about the observed branch outcome.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


MISSION_SPLIT_SCHEMA_ID = "state_sufficiency_mission_split_v1"
DATASET_IDENTITY_SCHEMA_ID = "state_sufficiency_dataset_identity_v1"
MODEL_CONFIG_SCHEMA_ID = "state_sufficiency_model_config_v1"
FOLD_METRICS_SCHEMA_ID = "state_sufficiency_fold_metrics_v1"
BOOTSTRAP_SCHEMA_ID = "state_sufficiency_bootstrap_v1"
EXPECTED_STATE_COUNT = 479
EXPECTED_TRANSITION_COUNT = 2850
EXPECTED_PAIR_COUNT = 7124
ACTION_DIM = 105
DEFAULT_HIDDEN_DIM = 128
DEFAULT_HIDDEN_LAYERS = 2
DEFAULT_EPOCHS = 120
DEFAULT_BATCH_SIZE = 256


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            relative = item.relative_to(path).as_posix().encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _as_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite numeric value")
    return result


def _depth_thumbnail(depth: np.ndarray) -> np.ndarray:
    """Return a fixed 9x8 mean-pool representation for the state depth."""

    array = np.asarray(depth, dtype=np.float32)
    if array.shape == (90, 160):
        return array.reshape(9, 10, 8, 20).mean(axis=(1, 3))
    # The current artifact takes the first branch.  This fallback keeps the
    # diagnostic explicit and deterministic if a future diagnostic resolution
    # changes without silently changing the production observation contract.
    row_index = np.linspace(0, array.shape[0] - 1, 9).round().astype(np.int64)
    col_index = np.linspace(0, array.shape[1] - 1, 8).round().astype(np.int64)
    return array[np.ix_(row_index, col_index)]


def _state_feature(state: Mapping[str, np.ndarray]) -> np.ndarray:
    vector = np.asarray(state["vector"], dtype=np.float32).reshape(-1)
    mask = np.asarray(state["mask"], dtype=np.float32).reshape(-1)
    depth = _depth_thumbnail(state["depth"]).reshape(-1).astype(np.float32)
    previous = np.asarray(state["previous_action"], dtype=np.int64).reshape(-1)
    if previous.size != 1:
        raise ValueError("state previous_action must contain exactly one action")
    previous_one_hot = np.zeros(ACTION_DIM, dtype=np.float32)
    if int(previous[0]) == -1:
        # The first observation in a prefix has no previous action.  The
        # collector's explicit -1 sentinel is represented by an all-zero
        # one-hot block, not as a current action.
        pass
    elif 0 <= int(previous[0]) < ACTION_DIM:
        previous_one_hot[int(previous[0])] = 1.0
    else:
        raise ValueError("state previous_action is outside the action space")
    feature = np.concatenate((vector, mask, depth, previous_one_hot)).astype(np.float32)
    if not np.isfinite(feature).all():
        raise ValueError("state feature contains non-finite values")
    return feature


def _load_states(root: Path) -> Dict[str, np.ndarray]:
    states: Dict[str, np.ndarray] = {}
    for path in sorted((root / "states").glob("*.npz")):
        with np.load(str(path), allow_pickle=False) as loaded:
            required = {"state_id", "vector", "depth", "mask", "previous_action"}
            missing = required.difference(loaded.files)
            if missing:
                raise ValueError("{} missing state fields {}".format(path, sorted(missing)))
            state_id = str(np.asarray(loaded["state_id"]).reshape(-1)[0])
            if state_id in states:
                raise ValueError("duplicate state_id {}".format(state_id))
            mask = np.asarray(loaded["mask"], dtype=np.bool_).reshape(-1)
            if mask.size != ACTION_DIM:
                raise ValueError("state {} has action dimension {}".format(state_id, mask.size))
            states[state_id] = _state_feature({key: loaded[key] for key in required if key != "state_id"})
    if not states:
        raise ValueError("no state artifacts found")
    return states


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("{} line {} is not an object".format(path, line_number))
                rows.append(value)
    return rows


def _load_pairs(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("pairwise dataset is empty")
    required = {
        "state_id",
        "mission_id",
        "step_id",
        "action1",
        "action2",
        "action_source1",
        "action_source2",
        "return1",
        "return2",
        "success1",
        "success2",
        "higher_return_action",
        "return_tie",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("pairwise dataset missing fields {}".format(sorted(missing)))
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        action1 = int(row["action1"])
        action2 = int(row["action2"])
        tie = bool(int(row["return_tie"]))
        higher_text = str(row["higher_return_action"]).strip()
        higher = int(higher_text) if higher_text else -1
        if action1 == action2 or higher not in (action1, action2):
            if not tie:
                raise ValueError("invalid pair action identity")
        parsed.append(
            {
                "state_id": str(row["state_id"]),
                "mission_id": str(row["mission_id"]),
                "step_id": int(row["step_id"]),
                "action1": action1,
                "action2": action2,
                "action_source1": str(row["action_source1"]),
                "action_source2": str(row["action_source2"]),
                "return1": _as_float(row["return1"]),
                "return2": _as_float(row["return2"]),
                "success1": int(row["success1"]),
                "success2": int(row["success2"]),
                "higher_return_action": higher,
                "return_tie": tie,
            }
        )
    return parsed


def load_audit_dataset(root: Path) -> Dict[str, Any]:
    """Load and validate the immutable diagnostic dataset without mutation."""

    root = Path(root).expanduser().resolve()
    manifest_path = root / "dataset_manifest.json"
    transition_path = root / "replay" / "transitions.npz"
    replay_metadata_path = root / "replay" / "metadata.json"
    pair_path = root / "multi_action_pairwise_dataset.csv"
    branch_path = root / "candidate_branches.jsonl"
    for path in (manifest_path, transition_path, replay_metadata_path, pair_path, branch_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replay_metadata = json.loads(replay_metadata_path.read_text(encoding="utf-8"))
    states = _load_states(root)
    branches = _read_jsonl(branch_path)
    branch_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for branch in branches:
        key = (str(branch.get("state_id", "")), int(branch.get("action", -1)))
        if not key[0] or key in branch_by_key:
            raise ValueError("duplicate or missing branch identity {}".format(key))
        branch_by_key[key] = branch
    with np.load(str(transition_path), allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    required = {
        "state_id",
        "mission_id",
        "episode_id",
        "action_source",
        "action",
        "mask",
        "state_vector",
        "state_depth",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError("transitions missing fields {}".format(sorted(missing)))
    count = int(arrays["action"].shape[0])
    if count != len(branches):
        raise ValueError("branch/transition count mismatch")
    transition_rows: List[Dict[str, Any]] = []
    seen_pairs = set()
    for index in range(count):
        state_id = str(arrays["state_id"][index])
        action = int(arrays["action"][index])
        key = (state_id, action)
        if state_id not in states or key in seen_pairs or key not in branch_by_key:
            raise ValueError("transition identity cannot be joined: {}".format(key))
        seen_pairs.add(key)
        branch = branch_by_key[key]
        if str(arrays["mission_id"][index]) != str(branch["mission_id"]):
            raise ValueError("transition/branch mission mismatch for {}".format(key))
        transition_rows.append(
            {
                "index": index,
                "state_id": state_id,
                "mission_id": str(arrays["mission_id"][index]),
                "episode_id": str(arrays["episode_id"][index]),
                "step_id": int(arrays["step_id"][index]),
                "action_source": str(arrays["action_source"][index]),
                "action": action,
                "state_feature": states[state_id],
                "success": int(bool(branch.get("success", False))),
                "episode_return": _as_float(branch["episode_return"]),
                "terminal_reason": str(branch.get("terminal_reason", "unknown")),
            }
        )
    pair_rows = _load_pairs(pair_path)
    for row in pair_rows:
        if row["state_id"] not in states:
            raise ValueError("pair references unknown state {}".format(row["state_id"]))
        if row["mission_id"] != next(
            item["mission_id"] for item in transition_rows if item["state_id"] == row["state_id"]
        ):
            raise ValueError("pair mission mismatch for {}".format(row["state_id"]))
        row["state_feature"] = states[row["state_id"]]
        row["label"] = int(row["higher_return_action"] == row["action1"])
    mission_data: Dict[str, Dict[str, Any]] = {}
    for row in transition_rows:
        item = mission_data.setdefault(
            row["mission_id"], {"state_ids": set(), "transition_count": 0, "pair_count": 0}
        )
        item["state_ids"].add(row["state_id"])
        item["transition_count"] += 1
    for row in pair_rows:
        mission_data[row["mission_id"]]["pair_count"] += 1
    identity = {
        "schema_id": DATASET_IDENTITY_SCHEMA_ID,
        "dataset_root": str(root),
        "dataset_id": str(manifest.get("dataset_id", root.name)),
        "manifest_sha256": sha256_file(manifest_path),
        "transitions_sha256": sha256_file(transition_path),
        "pairwise_sha256": sha256_file(pair_path),
        "candidate_branches_sha256": sha256_file(branch_path),
        "states_tree_sha256": sha256_tree(root / "states"),
        "state_count": len(states),
        "transition_count": count,
        "pair_count": len(pair_rows),
        "non_tie_pair_count": int(sum(not row["return_tie"] for row in pair_rows)),
        "action_dim": ACTION_DIM,
        "observation_contract": manifest.get("observation_contract"),
        "task_contract_schema_version": manifest.get("task_contract_schema_version"),
        "task_contract_sha256": manifest.get("task_contract_sha256"),
        "production_replay": bool(
            manifest.get("production_replay", replay_metadata.get("production_replay", False))
        ),
        "training_consumed": bool(
            manifest.get("training_consumed", replay_metadata.get("training_consumed", False))
        ),
    }
    if identity["state_count"] != EXPECTED_STATE_COUNT:
        raise ValueError("expected {} states, got {}".format(EXPECTED_STATE_COUNT, identity["state_count"]))
    if identity["transition_count"] != EXPECTED_TRANSITION_COUNT:
        raise ValueError("expected {} transitions, got {}".format(EXPECTED_TRANSITION_COUNT, identity["transition_count"]))
    if identity["pair_count"] != EXPECTED_PAIR_COUNT:
        raise ValueError("expected {} pairs, got {}".format(EXPECTED_PAIR_COUNT, identity["pair_count"]))
    if identity["production_replay"] or identity["training_consumed"]:
        raise ValueError("diagnostic input is marked production or consumed")
    return {
        "root": root,
        "manifest": manifest,
        "identity": identity,
        "states": states,
        "transitions": transition_rows,
        "pairs": pair_rows,
        "missions": mission_data,
    }


def build_mission_split(
    missions: Mapping[str, Mapping[str, Any]], *, seed: int, fold_count: int = 3
) -> Dict[str, Any]:
    """Assign complete missions, and therefore all their states, to folds."""

    if int(fold_count) < 2:
        raise ValueError("fold_count must be at least 2")
    rng = np.random.RandomState(int(seed))
    mission_ids = sorted(str(value) for value in missions)
    shuffled = [mission_ids[index] for index in rng.permutation(len(mission_ids)).tolist()]
    loads = [[0, 0, 0] for _ in range(int(fold_count))]
    assignment: Dict[str, int] = {}
    for mission_id in shuffled:
        item = missions[mission_id]
        state_count = len(item["state_ids"])
        transition_count = int(item.get("transition_count", 0))
        pair_count = int(item.get("pair_count", 0))
        fold = min(
            range(int(fold_count)),
            key=lambda value: (loads[value][0], loads[value][1], loads[value][2], value),
        )
        assignment[mission_id] = int(fold)
        loads[fold][0] += state_count
        loads[fold][1] += transition_count
        loads[fold][2] += pair_count
    state_to_fold: Dict[str, int] = {}
    mission_rows = []
    for mission_id in mission_ids:
        item = missions[mission_id]
        fold = assignment[mission_id]
        for state_id in item["state_ids"]:
            state_id = str(state_id)
            if state_id in state_to_fold and state_to_fold[state_id] != fold:
                raise ValueError("state assigned to multiple folds")
            state_to_fold[state_id] = fold
        mission_rows.append(
            {
                "mission_id": mission_id,
                "fold_id": fold,
                "state_count": len(item["state_ids"]),
                "transition_count": int(item.get("transition_count", 0)),
                "pair_count": int(item.get("pair_count", 0)),
            }
        )
    return {
        "schema_id": MISSION_SPLIT_SCHEMA_ID,
        "seed": int(seed),
        "fold_count": int(fold_count),
        "mission_count": len(mission_rows),
        "state_count": len(state_to_fold),
        "mission_to_fold": assignment,
        "state_to_fold": state_to_fold,
        "missions": mission_rows,
        "fold_loads": [
            {
                "state_count": load[0],
                "transition_count": load[1],
                "pair_count": load[2],
            }
            for load in loads
        ],
    }


def _auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = values[labels == 1]
    negatives = values[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positives.size * (positives.size + 1) / 2.0)
        / (positives.size * negatives.size)
    )


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, Optional[float]]:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError("binary metric inputs have incompatible sizes")
    predicted = scores >= 0.5
    tp = int(np.count_nonzero(predicted & (labels == 1)))
    fp = int(np.count_nonzero(predicted & (labels == 0)))
    fn = int(np.count_nonzero((~predicted) & (labels == 1)))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "count": int(labels.size),
        "positive_count": int(np.count_nonzero(labels == 1)),
        "accuracy": float(np.mean(predicted == labels)),
        "auc": _auc(labels, scores),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> Dict[str, Optional[float]]:
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if target.size != predicted.size or target.size == 0:
        raise ValueError("regression metric inputs have incompatible sizes")
    error = predicted - target
    ss_res = float(np.sum(error * error))
    centred = target - float(np.mean(target))
    ss_tot = float(np.sum(centred * centred))
    return {
        "count": int(target.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None,
    }


def _group_mean(
    labels: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    metric_fn,
    keys: Sequence[str],
) -> Dict[str, Optional[float]]:
    values: MutableMapping[str, List[float]] = {key: [] for key in keys}
    for group in sorted(set(str(value) for value in groups.tolist())):
        indices = np.asarray([str(value) == group for value in groups.tolist()], dtype=bool)
        metrics = metric_fn(labels[indices], predictions[indices])
        for key in keys:
            value = metrics.get(key)
            if value is not None and math.isfinite(float(value)):
                values[key].append(float(value))
    return {
        key: float(np.mean(items)) if items else None for key, items in values.items()
    }


def classification_bundle(
    labels: np.ndarray, probabilities: np.ndarray, state_ids: np.ndarray, mission_ids: np.ndarray
) -> Dict[str, Any]:
    return {
        "micro": binary_metrics(labels, probabilities),
        "state_macro": _group_mean(labels, probabilities, state_ids, binary_metrics, ("accuracy", "auc", "f1")),
        "mission_macro": _group_mean(labels, probabilities, mission_ids, binary_metrics, ("accuracy", "auc", "f1")),
    }


def regression_bundle(
    labels: np.ndarray, predictions: np.ndarray, state_ids: np.ndarray, mission_ids: np.ndarray
) -> Dict[str, Any]:
    return {
        "micro": regression_metrics(labels, predictions),
        "state_macro": _group_mean(labels, predictions, state_ids, regression_metrics, ("mae", "rmse", "r2")),
        "mission_macro": _group_mean(labels, predictions, mission_ids, regression_metrics, ("mae", "rmse", "r2")),
    }


def mission_cluster_bootstrap(
    values_by_mission: Mapping[str, Sequence[float]], *, repeats: int, seed: int
) -> Dict[str, Any]:
    missions = sorted(str(value) for value in values_by_mission)
    if not missions or int(repeats) <= 0:
        raise ValueError("bootstrap requires missions and positive repeats")
    mission_values = np.asarray(
        [float(np.mean(np.asarray(values_by_mission[mission], dtype=np.float64))) for mission in missions],
        dtype=np.float64,
    )
    rng = np.random.RandomState(int(seed))
    estimates = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        sample = rng.randint(0, len(missions), size=len(missions))
        estimates[index] = float(np.mean(mission_values[sample]))
    return {
        "schema_id": BOOTSTRAP_SCHEMA_ID,
        "mission_count": len(missions),
        "repeats": int(repeats),
        "seed": int(seed),
        "estimate": float(np.mean(mission_values)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
    }


class _Standardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "_Standardizer":
        mean = np.asarray(values, dtype=np.float32).mean(axis=0)
        scale = np.asarray(values, dtype=np.float32).std(axis=0)
        scale[scale < 1.0e-6] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)


class _SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.extend((nn.Linear(current, int(hidden_dim)), nn.ReLU()))
            current = int(hidden_dim)
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).reshape(-1)


def _train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    task: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    model = _SmallMLP(train_x.shape[1], DEFAULT_HIDDEN_DIM, DEFAULT_HIDDEN_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x_tensor = torch.from_numpy(train_x)
    y_tensor = torch.from_numpy(train_y.astype(np.float32))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 17)
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(x_tensor.shape[0], generator=generator)
        for start in range(0, x_tensor.shape[0], int(batch_size)):
            indices = order[start : start + int(batch_size)]
            prediction = model(x_tensor[indices])
            if task == "regression":
                loss = torch.mean((prediction - y_tensor[indices]) ** 2)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(prediction, y_tensor[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(test_x)).cpu().numpy()
    if task == "classification":
        output = 1.0 / (1.0 + np.exp(-np.clip(output, -40.0, 40.0)))
    return output.astype(np.float64)


def _one_hot(actions: np.ndarray) -> np.ndarray:
    result = np.zeros((len(actions), ACTION_DIM), dtype=np.float32)
    result[np.arange(len(actions)), actions.astype(np.int64)] = 1.0
    return result


def _prediction_row(**kwargs: Any) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def _source_metrics(
    transition_rows: Sequence[Mapping[str, Any]],
    b_probabilities: np.ndarray,
    c_predictions: np.ndarray,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for source in ("BC", "TEACHER", "NEIGHBOR", "RANDOM"):
        indices = np.asarray([row["action_source"] == source for row in transition_rows], dtype=bool)
        if not np.any(indices):
            continue
        labels = np.asarray([row["success"] for row in transition_rows], dtype=np.int64)[indices]
        returns = np.asarray([row["episode_return"] for row in transition_rows], dtype=np.float64)[indices]
        output[source] = {
            "count": int(np.count_nonzero(indices)),
            "success_rate": float(np.mean(labels)),
            "return_mean": float(np.mean(returns)),
            "return_median": float(np.median(returns)),
            "return_p10": float(np.percentile(returns, 10)),
            "return_p90": float(np.percentile(returns, 90)),
            "state_action_success_metrics": binary_metrics(labels, b_probabilities[indices]),
            "state_action_return_metrics": regression_metrics(returns, c_predictions[indices]),
        }
    return output


def _diversity_metrics(
    transition_rows: Sequence[Mapping[str, Any]],
    b_probabilities: np.ndarray,
    c_predictions: np.ndarray,
) -> Dict[str, Any]:
    grouped: Dict[str, List[int]] = {}
    for index, row in enumerate(transition_rows):
        grouped.setdefault(str(row["state_id"]), []).append(index)
    by_count: Dict[str, Dict[str, Any]] = {}
    for state_id, indices in grouped.items():
        count = len({int(transition_rows[index]["action"]) for index in indices})
        labels = np.asarray([transition_rows[index]["success"] for index in indices], dtype=np.int64)
        returns = np.asarray([transition_rows[index]["episode_return"] for index in indices], dtype=np.float64)
        metrics = by_count.setdefault(str(count), {"state_count": 0, "transition_count": 0, "success": [], "return": []})
        metrics["state_count"] += 1
        metrics["transition_count"] += len(indices)
        metrics["success"].extend((b_probabilities[indices] >= 0.5).astype(np.int64).tolist())
        metrics["return"].extend((c_predictions[indices] - returns).tolist())
    result: Dict[str, Any] = {
        "state_count": len(grouped),
        "unique_actions_per_state": {
            "mean": float(np.mean([len({int(transition_rows[index]["action"]) for index in indices}) for indices in grouped.values()])),
            "median": float(np.median([len({int(transition_rows[index]["action"]) for index in indices}) for indices in grouped.values()])),
            "min": int(min(len(indices) for indices in grouped.values())),
            "max": int(max(len(indices) for indices in grouped.values())),
        },
        "by_unique_action_count": {},
    }
    for count, item in sorted(by_count.items(), key=lambda value: int(value[0])):
        result["by_unique_action_count"][count] = {
            "state_count": int(item["state_count"]),
            "transition_count": int(item["transition_count"]),
            "state_action_success_accuracy": float(np.mean(item["success"])),
            "state_action_return_error_mean": float(np.mean(np.abs(item["return"]))),
        }
    return result


def _mission_group_values(
    mission_ids: np.ndarray, values: np.ndarray, *, reduce: str = "mean"
) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {}
    for mission_id, value in zip(mission_ids.tolist(), values.tolist()):
        result.setdefault(str(mission_id), []).append(float(value))
    if reduce == "mean":
        return {key: [float(np.mean(item))] for key, item in result.items()}
    return result


def _write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "model",
        "task",
        "fold_id",
        "mission_id",
        "episode_id",
        "step_id",
        "state_id",
        "state_hash",
        "action",
        "action1",
        "action2",
        "action_source",
        "label",
        "prediction",
        "return_target",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(
    path: Path,
    *,
    identity: Mapping[str, Any],
    split: Mapping[str, Any],
    fold_metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    pair = fold_metrics["models"]["D_pairwise_ranking"]["oof_metrics"]["micro"]
    pair_mission = fold_metrics["models"]["D_pairwise_ranking"]["oof_metrics"]["mission_macro"]
    ci = bootstrap["D_pairwise_accuracy"]["ci95"]
    pair_cluster_estimate = float(bootstrap["D_pairwise_accuracy"]["estimate"])
    b = fold_metrics["models"]["B_state_action_success"]["oof_metrics"]["micro"]
    a = fold_metrics["models"]["A_state_only_success"]["oof_metrics"]["micro"]
    if pair_cluster_estimate >= 0.70 and ci[0] > 0.50:
        result = "STATE_HAS_ACTION_INFORMATION"
        decision = "状态与动作联合输入显示可学习的 action-level 排序信号；这只是信息诊断，不是 AWAC 放行。"
    elif pair["accuracy"] >= 0.45 and pair["accuracy"] <= 0.70:
        result = "STATE_INFORMATION_LIMITED"
        decision = "状态-动作信息较弱或不稳定，不能据此宣称 Critic/AWAC 具备可靠排序能力。"
    else:
        result = "STATE_INSUFFICIENT"
        decision = "当前观测无法在本验证集上提供足够可分的 action-level 信号。"
    lines = [
        "# STATE SUFFICIENCY AUDIT V1",
        "",
        "本报告是只读信息诊断；未加载 N5/Q，未执行任何 AWAC、Actor、Critic、Unity 或 Dev100。",
        "",
        "## 数据与切分",
        "",
        "- dataset: `{}`".format(identity["dataset_id"]),
        "- states / transitions / pairs: `{}` / `{}` / `{}`".format(identity["state_count"], identity["transition_count"], identity["pair_count"]),
        "- non-tie pairs used by Model D: `{}`".format(identity["non_tie_pair_count"]),
        "- missions: `{}`; folds: `{}`; mission-cluster bootstrap: `{}` repeats".format(split["mission_count"], split["fold_count"], config["bootstrap_repeats"]),
        "- normalization: 每个训练 fold 单独拟合，测试 fold 未参与均值/方差拟合。",
        "",
        "## 关键指标",
        "",
        "| 模型 | 任务 | micro accuracy | AUC | F1 | MAE | RMSE | R2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("A_state_only_success", "B_state_action_success", "D_pairwise_ranking"):
        model = fold_metrics["models"][name]["oof_metrics"]["micro"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                "success" if "success" in name else "pairwise",
                model.get("accuracy", "-"),
                model.get("auc", "-"),
                model.get("f1", "-"),
                "-",
                "-",
                "-",
            )
        )
    regression = fold_metrics["models"]["C_state_action_return"]["oof_metrics"]["micro"]
    lines.append("| C_state_action_return | return | - | - | - | {} | {} | {} |".format(regression["mae"], regression["rmse"], regression["r2"]))
    lines.extend(
        [
            "",
            "Model A→B success accuracy: `{:.6f} → {:.6f}` (差 `{:.6f}`).".format(a["accuracy"], b["accuracy"], b["accuracy"] - a["accuracy"]),
            "Model D pairwise accuracy: micro `{:.6f}`, mission-macro `{:.6f}`, 95% mission-cluster CI `[{}, {}]`.".format(pair["accuracy"], pair_mission["accuracy"], ci[0], ci[1]),
            "",
            "## 来源与 action diversity",
            "",
            "详细 source return/success 和按每 state unique action 数分组指标见 `fold_metrics.json`。",
            "",
            "## 结论",
            "",
            "- `STATE_SUFFICIENCY_RESULT = {}`".format(result),
            "- `STATE_ACTION_INFORMATION_SCORE = {:.6f}`（Model D mission-cluster estimate；micro = {:.6f}）".format(pair_cluster_estimate, pair["accuracy"]),
            "- `PAIRWISE_ACCURACY = {:.6f}`; `CI = [{}, {}]`".format(pair_cluster_estimate, ci[0], ci[1]),
            "- `AWAC_NEXT_DECISION = DIAGNOSTIC_ONLY; {}`".format(decision),
            "",
            "限制：pairwise CI 以 mission 为 cluster，不代表训练种子不确定性；三折结果不构成新的独立外部测试集。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_state_sufficiency_audit(
    dataset_root: Path,
    output_root: Path,
    *,
    seed: int = 20260907,
    fold_count: int = 3,
    bootstrap_repeats: int = 5000,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Run the complete read-only offline audit and write all report artifacts."""

    output_root = Path(output_root).expanduser().resolve()
    existing_output = output_root.exists() and any(output_root.iterdir())
    if existing_output and not (output_root / "dataset_identity.json").is_file():
        raise FileExistsError("refusing to overwrite non-audit directory {}".format(output_root))
    data = load_audit_dataset(Path(dataset_root))
    if existing_output:
        previous_identity = json.loads((output_root / "dataset_identity.json").read_text(encoding="utf-8"))
        if previous_identity.get("manifest_sha256") != data["identity"]["manifest_sha256"]:
            raise FileExistsError("refusing to overwrite audit for a different dataset")
    output_root.mkdir(parents=True, exist_ok=True)
    split = build_mission_split(data["missions"], seed=int(seed), fold_count=int(fold_count))
    identity = data["identity"]
    config = {
        "schema_id": MODEL_CONFIG_SCHEMA_ID,
        "seed": int(seed),
        "fold_count": int(fold_count),
        "bootstrap_repeats": int(bootstrap_repeats),
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "hidden_layers": DEFAULT_HIDDEN_LAYERS,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam",
        "learning_rate": 1.0e-3,
        "device": "cpu",
        "state_features": ["vector", "legal_action_mask", "depth_9x8_mean_pool", "previous_action_one_hot"],
        "state_feature_dim": int(len(next(iter(data["states"].values())))),
        "action_features": ["action_one_hot_105"],
        "q_features_used": False,
        "return_features_used": False,
        "normalizer_fit_scope": "training_fold_only",
    }
    _json_dump(output_root / "dataset_identity.json", identity)
    _json_dump(output_root / "mission_split.json", split)
    _json_dump(output_root / "model_config.json", config)

    transitions = data["transitions"]
    pairs = [row for row in data["pairs"] if not row["return_tie"]]
    transition_state = np.stack([row["state_feature"] for row in transitions]).astype(np.float32)
    actions = np.asarray([row["action"] for row in transitions], dtype=np.int64)
    success = np.asarray([row["success"] for row in transitions], dtype=np.int64)
    returns = np.asarray([row["episode_return"] for row in transitions], dtype=np.float64)
    transition_states = np.asarray([row["state_id"] for row in transitions])
    transition_missions = np.asarray([row["mission_id"] for row in transitions])
    transition_folds = np.asarray([split["state_to_fold"][state] for state in transition_states.tolist()], dtype=np.int64)
    episode_by_state = {str(row["state_id"]): str(row["episode_id"]) for row in transitions}
    pair_state = np.stack([row["state_feature"] for row in pairs]).astype(np.float32)
    pair_action1 = np.asarray([row["action1"] for row in pairs], dtype=np.int64)
    pair_action2 = np.asarray([row["action2"] for row in pairs], dtype=np.int64)
    pair_labels = np.asarray([row["label"] for row in pairs], dtype=np.int64)
    pair_states = np.asarray([row["state_id"] for row in pairs])
    pair_missions = np.asarray([row["mission_id"] for row in pairs])
    pair_folds = np.asarray([split["state_to_fold"][state] for state in pair_states.tolist()], dtype=np.int64)

    state_action_features = np.concatenate((transition_state, _one_hot(actions)), axis=1)
    pair_features = np.concatenate((pair_state, _one_hot(pair_action1), _one_hot(pair_action2)), axis=1)
    predictions: List[Dict[str, Any]] = []
    fold_records: Dict[str, Dict[str, Any]] = {
        "A_state_only_success": {"folds": []},
        "B_state_action_success": {"folds": []},
        "C_state_action_return": {"folds": []},
        "D_pairwise_ranking": {"folds": []},
    }
    a_oof = np.zeros(len(transitions), dtype=np.float64)
    b_oof = np.zeros(len(transitions), dtype=np.float64)
    c_oof = np.zeros(len(transitions), dtype=np.float64)
    d_oof = np.zeros(len(pairs), dtype=np.float64)
    for fold in range(int(fold_count)):
        train_t = np.flatnonzero(transition_folds != fold)
        test_t = np.flatnonzero(transition_folds == fold)
        train_p = np.flatnonzero(pair_folds != fold)
        test_p = np.flatnonzero(pair_folds == fold)
        state_normalizer = _Standardizer.fit(transition_state[train_t])
        action_normalizer = _Standardizer.fit(state_action_features[train_t])
        pair_normalizer = _Standardizer.fit(pair_features[train_p])
        a_pred = _train_model(
            state_normalizer.transform(transition_state[train_t]),
            success[train_t],
            state_normalizer.transform(transition_state[test_t]),
            task="classification",
            seed=int(seed) + fold * 100 + 1,
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
        b_pred = _train_model(
            action_normalizer.transform(state_action_features[train_t]),
            success[train_t],
            action_normalizer.transform(state_action_features[test_t]),
            task="classification",
            seed=int(seed) + fold * 100 + 2,
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
        train_return_mean = float(np.mean(returns[train_t]))
        train_return_scale = float(np.std(returns[train_t]))
        if train_return_scale < 1.0e-6:
            train_return_scale = 1.0
        c_pred_scaled = _train_model(
            action_normalizer.transform(state_action_features[train_t]),
            ((returns[train_t] - train_return_mean) / train_return_scale).astype(np.float32),
            action_normalizer.transform(state_action_features[test_t]),
            task="regression",
            seed=int(seed) + fold * 100 + 3,
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
        c_pred = c_pred_scaled * train_return_scale + train_return_mean
        d_pred = _train_model(
            pair_normalizer.transform(pair_features[train_p]),
            pair_labels[train_p],
            pair_normalizer.transform(pair_features[test_p]),
            task="classification",
            seed=int(seed) + fold * 100 + 4,
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
        a_oof[test_t] = a_pred
        b_oof[test_t] = b_pred
        c_oof[test_t] = c_pred
        d_oof[test_p] = d_pred
        fold_records["A_state_only_success"]["folds"].append(
            {"fold_id": fold, "train_count": int(len(train_t)), "test_count": int(len(test_t)), "metrics": classification_bundle(success[test_t], a_pred, transition_states[test_t], transition_missions[test_t])}
        )
        fold_records["B_state_action_success"]["folds"].append(
            {"fold_id": fold, "train_count": int(len(train_t)), "test_count": int(len(test_t)), "metrics": classification_bundle(success[test_t], b_pred, transition_states[test_t], transition_missions[test_t])}
        )
        fold_records["C_state_action_return"]["folds"].append(
            {"fold_id": fold, "train_count": int(len(train_t)), "test_count": int(len(test_t)), "metrics": regression_bundle(returns[test_t], c_pred, transition_states[test_t], transition_missions[test_t])}
        )
        fold_records["D_pairwise_ranking"]["folds"].append(
            {"fold_id": fold, "train_count": int(len(train_p)), "test_count": int(len(test_p)), "metrics": classification_bundle(pair_labels[test_p], d_pred, pair_states[test_p], pair_missions[test_p])}
        )
        for index, prediction in zip(test_t.tolist(), a_pred.tolist()):
            row = transitions[index]
            predictions.append(_prediction_row(model="A_state_only_success", task="success", fold_id=fold, mission_id=row["mission_id"], episode_id=row["episode_id"], step_id=row["step_id"], state_id=row["state_id"], state_hash=row["state_id"], action=row["action"], action_source=row["action_source"], label=row["success"], prediction=prediction))
        for index, prediction in zip(test_t.tolist(), b_pred.tolist()):
            row = transitions[index]
            predictions.append(_prediction_row(model="B_state_action_success", task="success", fold_id=fold, mission_id=row["mission_id"], episode_id=row["episode_id"], step_id=row["step_id"], state_id=row["state_id"], state_hash=row["state_id"], action=row["action"], action_source=row["action_source"], label=row["success"], prediction=prediction))
        for index, prediction in zip(test_t.tolist(), c_pred.tolist()):
            row = transitions[index]
            predictions.append(_prediction_row(model="C_state_action_return", task="return", fold_id=fold, mission_id=row["mission_id"], episode_id=row["episode_id"], step_id=row["step_id"], state_id=row["state_id"], state_hash=row["state_id"], action=row["action"], action_source=row["action_source"], return_target=row["episode_return"], label=row["episode_return"], prediction=prediction))
        for index, prediction in zip(test_p.tolist(), d_pred.tolist()):
            row = pairs[index]
            predictions.append(_prediction_row(model="D_pairwise_ranking", task="pairwise", fold_id=fold, mission_id=row["mission_id"], episode_id=episode_by_state[row["state_id"]], step_id=row["step_id"], state_id=row["state_id"], state_hash=row["state_id"], action1=row["action1"], action2=row["action2"], label=row["label"], prediction=prediction))

    fold_metrics: Dict[str, Any] = {
        "schema_id": FOLD_METRICS_SCHEMA_ID,
        "dataset_identity": identity,
        "fold_count": int(fold_count),
        "models": fold_records,
        "source_metrics": _source_metrics(transitions, b_oof, c_oof),
        "action_diversity_metrics": _diversity_metrics(transitions, b_oof, c_oof),
    }
    fold_metrics["models"]["A_state_only_success"]["oof_metrics"] = classification_bundle(success, a_oof, transition_states, transition_missions)
    fold_metrics["models"]["B_state_action_success"]["oof_metrics"] = classification_bundle(success, b_oof, transition_states, transition_missions)
    fold_metrics["models"]["C_state_action_return"]["oof_metrics"] = regression_bundle(returns, c_oof, transition_states, transition_missions)
    fold_metrics["models"]["D_pairwise_ranking"]["oof_metrics"] = classification_bundle(pair_labels, d_oof, pair_states, pair_missions)

    a_correct = (a_oof >= 0.5) == success
    b_correct = (b_oof >= 0.5) == success
    d_correct = (d_oof >= 0.5) == pair_labels
    missions = sorted(set(transition_missions.tolist()))
    b_minus_a: Dict[str, List[float]] = {}
    d_by_mission: Dict[str, List[float]] = {}
    for mission_id in missions:
        t_indices = np.flatnonzero(transition_missions == mission_id)
        p_indices = np.flatnonzero(pair_missions == mission_id)
        b_minus_a[mission_id] = [float(np.mean(b_correct[t_indices]) - np.mean(a_correct[t_indices]))]
        d_by_mission[mission_id] = [float(np.mean(d_correct[p_indices]))] if p_indices.size else [float("nan")]
    d_by_mission = {key: value for key, value in d_by_mission.items() if math.isfinite(value[0])}
    bootstrap = {
        "schema_id": BOOTSTRAP_SCHEMA_ID,
        "repeats": int(bootstrap_repeats),
        "cluster_unit": "mission_id",
        "B_minus_A_success_accuracy": mission_cluster_bootstrap(b_minus_a, repeats=int(bootstrap_repeats), seed=int(seed) + 1000),
        "D_pairwise_accuracy": mission_cluster_bootstrap(d_by_mission, repeats=int(bootstrap_repeats), seed=int(seed) + 1001),
    }
    _json_dump(output_root / "fold_metrics.json", fold_metrics)
    _json_dump(output_root / "bootstrap.json", bootstrap)
    _write_predictions(output_root / "predictions.csv", predictions)
    _write_report(output_root / "report_zh.md", identity=identity, split=split, fold_metrics=fold_metrics, bootstrap=bootstrap, config=config)
    return {
        "identity": identity,
        "split": split,
        "config": config,
        "fold_metrics": fold_metrics,
        "bootstrap": bootstrap,
        "output_root": str(output_root),
        "state_sufficiency_result": _state_result(fold_metrics, bootstrap),
    }


def _state_result(fold_metrics: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> Dict[str, Any]:
    pair = fold_metrics["models"]["D_pairwise_ranking"]["oof_metrics"]["micro"]
    ci = bootstrap["D_pairwise_accuracy"]["ci95"]
    pair_cluster_estimate = float(bootstrap["D_pairwise_accuracy"]["estimate"])
    if pair_cluster_estimate >= 0.70 and ci[0] > 0.50:
        result = "STATE_HAS_ACTION_INFORMATION"
    elif 0.45 <= pair["accuracy"] <= 0.70:
        result = "STATE_INFORMATION_LIMITED"
    else:
        result = "STATE_INSUFFICIENT"
    return {
        "STATE_SUFFICIENCY_RESULT": result,
        "STATE_ACTION_INFORMATION_SCORE": pair_cluster_estimate,
        "PAIRWISE_ACCURACY": pair_cluster_estimate,
        "PAIRWISE_ACCURACY_MICRO": float(pair["accuracy"]),
        "CI": ci,
        "AWAC_NEXT_DECISION": "DIAGNOSTIC_ONLY",
    }


__all__ = [
    "DATASET_IDENTITY_SCHEMA_ID",
    "MISSION_SPLIT_SCHEMA_ID",
    "binary_metrics",
    "build_mission_split",
    "classification_bundle",
    "load_audit_dataset",
    "mission_cluster_bootstrap",
    "regression_metrics",
    "run_state_sufficiency_audit",
    "sha256_file",
]
