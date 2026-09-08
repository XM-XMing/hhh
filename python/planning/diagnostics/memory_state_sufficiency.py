"""Diagnostic memory models for the observed-action state-aliasing audit.

The history is reconstructed only from the immutable diagnostic
``state_manifest.jsonl`` and state snapshots.  A sequence is ordered by the
recorded source episode/state index, is restricted to the prefix ending at
the current state, and never contains a branch outcome or future frame.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from planning.diagnostics.state_sufficiency import ACTION_DIM, load_audit_dataset


MEMORY_STATE_SUFFICIENCY_AUDIT_SCHEMA_ID = "memory_state_sufficiency_audit_v1"
MEMORY_METRICS_SCHEMA_ID = "memory_state_sufficiency_metrics_v1"
ORACLE_MEMORY_METRICS_SCHEMA_ID = "oracle_memory_action_metrics_v1"
MEMORY_BOOTSTRAP_SCHEMA_ID = "memory_state_sufficiency_bootstrap_v1"
EXPECTED_STATE_COUNT = 479
EXPECTED_TRANSITION_COUNT = 2850
EXPECTED_PAIR_COUNT = 7124
DEFAULT_SEED = 20260908
DEFAULT_FOLD_COUNT = 3
DEFAULT_BOOTSTRAP_REPEATS = 5000
DEFAULT_EPOCHS = 80
DEFAULT_ACTION_BATCH_SIZE = 64
DEFAULT_PAIR_BATCH_SIZE = 128
HIDDEN_DIM = 128
HIDDEN_LAYERS = 2
SEQUENCE_FEATURE_DIM = 304
MODEL_SPECS = {
    "M0_current_state": ("stack", 1),
    "M1_raw_stack_5": ("stack", 5),
    "M2_raw_stack_10": ("stack", 10),
    "M3_gru_10": ("gru", 10),
    "M4_transformer_10": ("transformer", 10),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError("unsupported JSON value {}".format(type(value).__name__))


def _json_dump(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _stats(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    if not np.isfinite(array).all():
        raise ValueError("non-finite memory audit statistic")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


class _Standardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, history: np.ndarray, valid: np.ndarray) -> "_Standardizer":
        frames = np.asarray(history, dtype=np.float32)[np.asarray(valid, dtype=bool)]
        if frames.size == 0:
            raise ValueError("cannot fit history normalizer without valid frames")
        mean = frames.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = frames.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1.0e-6] = 1.0
        return cls(mean, scale)

    def transform(self, history: np.ndarray, valid: np.ndarray) -> np.ndarray:
        values = ((np.asarray(history, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)
        values[~np.asarray(valid, dtype=bool)] = 0.0
        return values


def _sinusoidal(length: int, width: int) -> torch.Tensor:
    position = torch.arange(int(length), dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(torch.arange(0, int(width), 2, dtype=torch.float32) * (-math.log(10000.0) / int(width)))
    encoding = torch.zeros((int(length), int(width)), dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale)
    return encoding.unsqueeze(0)


class _SequenceEncoder(nn.Module):
    def __init__(self, kind: str, sequence_length: int, feature_dim: int) -> None:
        super().__init__()
        self.kind = str(kind)
        self.sequence_length = int(sequence_length)
        self.input_dim = int(feature_dim) + 1  # explicit valid-history bit
        if self.kind == "stack":
            layers: List[nn.Module] = []
            current = self.sequence_length * self.input_dim
            for _ in range(HIDDEN_LAYERS):
                layers.extend((nn.Linear(current, HIDDEN_DIM), nn.ReLU()))
                current = HIDDEN_DIM
            layers.append(nn.Linear(current, HIDDEN_DIM))
            self.backbone = nn.Sequential(*layers)
        elif self.kind == "gru":
            self.gru = nn.GRU(self.input_dim, HIDDEN_DIM, batch_first=True)
        elif self.kind == "transformer":
            self.input_projection = nn.Linear(self.input_dim, HIDDEN_DIM)
            layer = nn.TransformerEncoderLayer(
                d_model=HIDDEN_DIM,
                nhead=4,
                dim_feedforward=HIDDEN_DIM * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
            self.register_buffer("position", _sinusoidal(self.sequence_length, HIDDEN_DIM), persistent=False)
        else:
            raise ValueError("unknown sequence encoder {}".format(kind))

    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        values = torch.cat((history, valid.to(dtype=history.dtype).unsqueeze(-1)), dim=-1)
        if self.kind == "stack":
            return self.backbone(values.reshape(values.shape[0], -1))
        if self.kind == "gru":
            output, _ = self.gru(values)
            return output[:, -1, :]
        projected = self.input_projection(values) + self.position[:, : values.shape[1], :]
        output = self.transformer(projected, src_key_padding_mask=~valid.bool())
        return output[:, -1, :]


class _ActionModel(nn.Module):
    def __init__(self, kind: str, sequence_length: int, feature_dim: int) -> None:
        super().__init__()
        self.encoder = _SequenceEncoder(kind, sequence_length, feature_dim)
        self.head = nn.Linear(HIDDEN_DIM, ACTION_DIM)

    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(history, valid))


class _PairModel(nn.Module):
    def __init__(self, kind: str, sequence_length: int, feature_dim: int) -> None:
        super().__init__()
        self.encoder = _SequenceEncoder(kind, sequence_length, feature_dim)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + ACTION_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, history: torch.Tensor, valid: torch.Tensor, action1: torch.Tensor, action2: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(history, valid)
        return self.head(torch.cat((encoded, action1, action2), dim=1)).reshape(-1)


def _load_history_manifest(dataset_root: Path) -> Dict[str, Dict[str, Any]]:
    path = Path(dataset_root) / "state_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    result: Dict[str, Dict[str, Any]] = {}
    episode_indices: Dict[Tuple[str, int], str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            state_id = str(row.get("state_id", ""))
            episode = str(row.get("source_episode_id", ""))
            index = int(row.get("state_index", -1))
            if not state_id or not episode or index < 0:
                raise ValueError("invalid history manifest line {}".format(line_number))
            if state_id in result or (episode, index) in episode_indices:
                raise ValueError("duplicate history identity at line {}".format(line_number))
            result[state_id] = {
                "state_id": state_id,
                "mission_id": str(row["mission_id"]),
                "source_episode_id": episode,
                "state_index": index,
                "step_id": int(row["step_id"]),
            }
            episode_indices[(episode, index)] = state_id
    if not result:
        raise ValueError("empty history manifest")
    for row in result.values():
        row["episode_state_ids"] = {
            index: state_id
            for (episode, index), state_id in episode_indices.items()
            if episode == row["source_episode_id"]
        }
    return result


def build_history_tensors(
    labels: Sequence[Mapping[str, Any]],
    state_features: Mapping[str, np.ndarray],
    history_manifest: Mapping[str, Mapping[str, Any]],
    *,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if int(sequence_length) <= 0:
        raise ValueError("sequence length must be positive")
    history = np.zeros((len(labels), int(sequence_length), SEQUENCE_FEATURE_DIM), dtype=np.float32)
    valid = np.zeros((len(labels), int(sequence_length)), dtype=bool)
    available_counts: List[int] = []
    for row_index, label in enumerate(labels):
        state_id = str(label["state_id"])
        if state_id not in history_manifest or state_id not in state_features:
            raise ValueError("label state {} missing from history/state features".format(state_id))
        current = history_manifest[state_id]
        episode_map = current["episode_state_ids"]
        current_index = int(current["state_index"])
        count = 0
        for position, source_index in enumerate(range(current_index - int(sequence_length) + 1, current_index + 1)):
            prior_state_id = episode_map.get(source_index)
            if prior_state_id is None:
                continue
            prior = history_manifest[prior_state_id]
            if str(prior["mission_id"]) != str(label["mission_id"]):
                raise ValueError("history crossed mission boundary")
            history[row_index, position] = np.asarray(state_features[prior_state_id], dtype=np.float32)
            valid[row_index, position] = True
            count += 1
        if not valid[row_index, -1]:
            raise ValueError("current state missing from its own history")
        available_counts.append(count)
    provenance = {
        "history_source": "state_manifest.jsonl plus immutable states/*.npz state features",
        "future_information_used": False,
        "branch_outcomes_used_as_features": False,
        "source_episode_count": len({str(row["source_episode_id"]) for row in history_manifest.values()}),
        "sequence_length": int(sequence_length),
        "state_count": len(labels),
        "available_history_frames": _stats(available_counts),
        "full_sequence_state_count": int(sum(value == int(sequence_length) for value in available_counts)),
        "full_sequence_ratio": float(np.mean(np.asarray(available_counts) == int(sequence_length))),
        "zero_prefix_padding_allowed": True,
        "ordering": "oldest available prefix to current state; current state is last token",
    }
    return history, valid, provenance


def _auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(np.unique(labels)) < 2:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive = labels == 1
    negative = labels == 0
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2.0) / (positive.sum() * negative.sum()))


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= 0.5
    return {
        "count": int(len(labels)),
        "accuracy": float(np.mean(predicted == labels)),
        "auc": _auc(labels, probabilities),
        "positive_count": int(np.count_nonzero(labels == 1)),
    }


def _bootstrap(values_by_mission: Mapping[str, Sequence[float]], *, repeats: int, seed: int) -> Dict[str, Any]:
    values = {str(key): np.asarray(value, dtype=np.float64) for key, value in values_by_mission.items() if len(value)}
    if not values:
        return {"schema_id": MEMORY_BOOTSTRAP_SCHEMA_ID, "cluster_unit": "mission_id", "mission_count": 0, "repeats": int(repeats), "seed": int(seed), "estimate": None, "ci95": None}
    missions = sorted(values)
    means = np.asarray([float(np.mean(values[key])) for key in missions], dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    estimates = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        estimates[index] = float(np.mean(means[rng.randint(0, len(missions), size=len(missions))]))
    return {
        "schema_id": MEMORY_BOOTSTRAP_SCHEMA_ID,
        "cluster_unit": "mission_id",
        "mission_count": len(missions),
        "repeats": int(repeats),
        "seed": int(seed),
        "estimate": float(np.mean(means)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
    }


def _fit_action_model(
    kind: str,
    sequence_length: int,
    train_history: np.ndarray,
    train_valid: np.ndarray,
    train_labels: np.ndarray,
    test_history: np.ndarray,
    test_valid: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    model = _ActionModel(kind, sequence_length, SEQUENCE_FEATURE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    train_x = torch.from_numpy(train_history)
    train_v = torch.from_numpy(train_valid)
    train_y = torch.from_numpy(train_labels.astype(np.int64))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 17)
    last_loss = None
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(len(train_y), generator=generator)
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            loss = nn.functional.cross_entropy(model(train_x[indices], train_v[indices]), train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test_history), torch.from_numpy(test_valid)).cpu().numpy()
    return logits.astype(np.float32), {"epochs": int(epochs), "batch_size": int(batch_size), "learning_rate": 1.0e-3, "seed": int(seed), "last_train_loss": last_loss}


def _one_hot(actions: np.ndarray) -> np.ndarray:
    output = np.zeros((len(actions), ACTION_DIM), dtype=np.float32)
    output[np.arange(len(actions)), np.asarray(actions, dtype=np.int64)] = 1.0
    return output


def _fit_pair_model(
    kind: str,
    sequence_length: int,
    train_history: np.ndarray,
    train_valid: np.ndarray,
    train_action1: np.ndarray,
    train_action2: np.ndarray,
    train_labels: np.ndarray,
    test_history: np.ndarray,
    test_valid: np.ndarray,
    test_action1: np.ndarray,
    test_action2: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    model = _PairModel(kind, sequence_length, SEQUENCE_FEATURE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    train_x = torch.from_numpy(train_history)
    train_v = torch.from_numpy(train_valid)
    train_a1 = torch.from_numpy(_one_hot(train_action1))
    train_a2 = torch.from_numpy(_one_hot(train_action2))
    train_y = torch.from_numpy(train_labels.astype(np.float32))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 17)
    last_loss = None
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(len(train_y), generator=generator)
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            logits = model(train_x[indices], train_v[indices], train_a1[indices], train_a2[indices])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(test_history), torch.from_numpy(test_valid),
            torch.from_numpy(_one_hot(test_action1)), torch.from_numpy(_one_hot(test_action2)),
        ).cpu().numpy()
    probabilities = (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).astype(np.float64)
    return probabilities, {"epochs": int(epochs), "batch_size": int(batch_size), "learning_rate": 1.0e-3, "seed": int(seed), "last_train_loss": last_loss}


def _masked_order(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).copy()
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        raise ValueError("state has no legal action")
    values[~valid] = -1.0e30
    return np.argsort(-values, kind="mergesort")


def _action_metrics(logits: np.ndarray, labels: Sequence[Mapping[str, Any]], raw_masks: np.ndarray, state_ids: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, List[List[float]]]]:
    top1: List[float] = []
    top5: List[float] = []
    by_mission: MutableMapping[str, Dict[str, List[float]]] = defaultdict(lambda: {"top1": [], "top5": []})
    for index, row in enumerate(labels):
        order = _masked_order(logits[index], raw_masks[index])
        oracle = int(row["oracle_action"])
        first = float(int(order[0] == oracle))
        fifth = float(int(oracle in set(order[:5].tolist())))
        top1.append(first); top5.append(fifth)
        by_mission[str(row["mission_id"])]["top1"].append(first)
        by_mission[str(row["mission_id"])]["top5"].append(fifth)
    mission_top1 = {mission: values["top1"] for mission, values in by_mission.items()}
    mission_top5 = {mission: values["top5"] for mission, values in by_mission.items()}
    return {
        "state_count": len(labels),
        "mission_count": len(by_mission),
        "top1_accuracy": float(np.mean(top1)),
        "top5_accuracy": float(np.mean(top5)),
        "mission_macro_top1_accuracy": float(np.mean([np.mean(v) for v in mission_top1.values()])),
        "mission_macro_top5_accuracy": float(np.mean([np.mean(v) for v in mission_top5.values()])),
    }, {"top1": mission_top1, "top5": mission_top5}


def _pair_metrics(probabilities: np.ndarray, pairs: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, List[List[float]]]]:
    labels = np.asarray([int(row["label"]) for row in pairs], dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    correct = (probabilities >= 0.5) == labels
    by_mission: MutableMapping[str, List[float]] = defaultdict(list)
    for row, value in zip(pairs, correct.tolist()):
        by_mission[str(row["mission_id"])].append(float(value))
    mission_auc: Dict[str, List[float]] = {}
    for mission in sorted(by_mission):
        indices = np.asarray([str(row["mission_id"]) == mission for row in pairs], dtype=bool)
        value = _auc(labels[indices], probabilities[indices])
        if value is not None:
            mission_auc[mission] = [float(value)]
    return {
        "pair_count": len(pairs),
        "mission_count": len(by_mission),
        "accuracy": float(np.mean(correct)),
        "auc": _auc(labels, probabilities),
        "mission_macro_accuracy": float(np.mean([np.mean(v) for v in by_mission.values()])),
        "mission_macro_auc": float(np.mean([v[0] for v in mission_auc.values()])) if mission_auc else None,
    }, {"accuracy": {key: value for key, value in by_mission.items()}, "auc": mission_auc}


def _paired_delta_bootstrap(values: Mapping[str, Mapping[str, Sequence[float]]], baseline: str, target: str, metric: str, *, repeats: int, seed: int) -> Dict[str, Any]:
    common = sorted(set(values[baseline][metric]).intersection(values[target][metric]))
    delta = {mission: (np.asarray(values[target][metric][mission]) - np.asarray(values[baseline][metric][mission])).tolist() for mission in common}
    return _bootstrap(delta, repeats=repeats, seed=seed)


def _write_predictions(path: Path, labels: Sequence[Mapping[str, Any]], raw_masks: np.ndarray, action_predictions: Mapping[str, np.ndarray], pair_predictions: Mapping[str, np.ndarray], pairs: Sequence[Mapping[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fields = ["kind", "model", "state_id", "mission_id", "oracle_action", "predicted_action", "top5_hit", "action1", "action2", "label", "probability"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for model, logits in action_predictions.items():
            for index, row in enumerate(labels):
                order = _masked_order(logits[index], raw_masks[index])
                writer.writerow({"kind": "oracle_action", "model": model, "state_id": row["state_id"], "mission_id": row["mission_id"], "oracle_action": row["oracle_action"], "predicted_action": int(order[0]), "top5_hit": int(row["oracle_action"] in set(order[:5].tolist()))})
        for model, probabilities in pair_predictions.items():
            for index, row in enumerate(pairs):
                writer.writerow({"kind": "pairwise", "model": model, "state_id": row["state_id"], "mission_id": row["mission_id"], "action1": row["action1"], "action2": row["action2"], "label": row["label"], "probability": float(probabilities[index])})


def _write_report(path: Path, source: Mapping[str, Any], history: Mapping[str, Any], oracle_metrics: Mapping[str, Any], pair_metrics: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> None:
    pair_values = {name: value["accuracy"] for name, value in pair_metrics["models"].items()}
    best_name = max(pair_values, key=pair_values.get)
    best = pair_metrics["models"][best_name]
    best_ci = bootstrap["models"][best_name]["pairwise_accuracy"]["ci95"]
    if best["accuracy"] >= 0.65 and best_ci[0] > 0.50:
        result = "MEMORY_SUPPORTS_STATE_ALIASING_RESOLUTION"
        decision = "至少一个 memory model 达到预设 pairwise 门槛；可进入独立 memory/critic prototype 评审，但本轮仍未训练 RL。"
    elif best["accuracy"] <= 0.60:
        result = "MEMORY_NOT_SUFFICIENT"
        decision = "memory model 仍未达到可靠 action ranking；应优先重新审视状态定义/外部可观测信息，而非直接放行 AWAC。"
    else:
        result = "MEMORY_EVIDENCE_INCONCLUSIVE"
        decision = "结果位于门槛之间，不能把 state aliasing 是否解决作确定结论。"
    lines = [
        "# MEMORY STATE SUFFICIENCY AUDIT V1", "",
        "只读监督诊断；未执行 AWAC、Actor/Critic 训练、Unity/Bridge/ROS，也未修改 production。", "",
        "## 数据与历史边界", "",
        "- dataset=`{}`；states={}；transitions={}；missions={}；pair rows={}。".format(source["dataset_id"], source["state_count"], source["transition_count"], source["mission_count"], source["pair_count"]),
        "- 历史来源：`state_manifest.jsonl` 的 source_episode_id/state_index 与当前 states snapshot；只使用 current state 之前的 prefix。", 
        "- 5-step full-history coverage=`{:.3f}`；10-step full-history coverage=`{:.3f}`；缺失前缀显式 zero-padding。".format(history["M1_raw_stack_5"]["full_sequence_ratio"], history["M2_raw_stack_10"]["full_sequence_ratio"]),
        "- 未使用 future state、branch return、termination reason、Q/Critic 作为输入。", "",
        "## Oracle action classification", "",
        "| model | Top-1 | Top-5 | mission macro Top-1 |", "|---|---:|---:|---:|",
    ]
    for name, item in oracle_metrics["models"].items():
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} |".format(name, item["top1_accuracy"], item["top5_accuracy"], item["mission_macro_top1_accuracy"]))
    lines.extend(["", "## Pairwise ranking", "", "| model | accuracy | AUC | mission macro accuracy | mission macro AUC |", "|---|---:|---:|---:|---:|"])
    for name, item in pair_metrics["models"].items():
        lines.append("| {} | {:.6f} | {} | {:.6f} | {} |".format(name, item["accuracy"], "-" if item["auc"] is None else "{:.6f}".format(item["auc"]), item["mission_macro_accuracy"], "-" if item["mission_macro_auc"] is None else "{:.6f}".format(item["mission_macro_auc"])))
    lines.extend(["", "## Memory comparison", "", "- M1-M0 pairwise accuracy bootstrap CI: `{}`。".format(bootstrap["deltas_vs_M0"]["M1_raw_stack_5"]["pairwise_accuracy"]["ci95"]), "- M2-M0 pairwise accuracy bootstrap CI: `{}`。".format(bootstrap["deltas_vs_M0"]["M2_raw_stack_10"]["pairwise_accuracy"]["ci95"]), "- M3-M0 pairwise accuracy bootstrap CI: `{}`。".format(bootstrap["deltas_vs_M0"]["M3_gru_10"]["pairwise_accuracy"]["ci95"]), "- M4-M0 pairwise accuracy bootstrap CI: `{}`。".format(bootstrap["deltas_vs_M0"]["M4_transformer_10"]["pairwise_accuracy"]["ci95"]), "", "## 结论", "", "- `MEMORY_STATE_SUFFICIENCY_RESULT = {}`。".format(result), "- `BEST_MEMORY_MODEL = {}`。".format(best_name), "- {}".format(decision), "- 该结果只反映当前 31 mission failure-state slice，不能直接推广到完整 Dev100 或生产 RL。", "", "## 身份", "", "- dataset manifest SHA256=`{}`。".format(source["dataset_manifest_sha256"]), "- oracle artifact SHA256=`{}`。".format(source["oracle_artifact_sha256"]), "- production_modified=NO；replay_modified=NO；rl_training_executed=NO；runtime_started=NO。"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_memory_state_sufficiency_audit(*, dataset_root: Path, oracle_root: Path, out_dir: Path, seed: int = DEFAULT_SEED, fold_count: int = DEFAULT_FOLD_COUNT, bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS, epochs: int = DEFAULT_EPOCHS, action_batch_size: int = DEFAULT_ACTION_BATCH_SIZE, pair_batch_size: int = DEFAULT_PAIR_BATCH_SIZE) -> Dict[str, Any]:
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output: {}".format(out_dir))
    dataset_root = Path(dataset_root).expanduser().resolve()
    oracle_root = Path(oracle_root).expanduser().resolve()
    oracle_path = oracle_root / "oracle_dataset.json"
    if not oracle_path.is_file():
        raise FileNotFoundError(str(oracle_path))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    data = load_audit_dataset(dataset_root)
    if data["identity"]["state_count"] != EXPECTED_STATE_COUNT or data["identity"]["transition_count"] != EXPECTED_TRANSITION_COUNT or data["identity"]["pair_count"] != EXPECTED_PAIR_COUNT:
        raise ValueError("unexpected diagnostic dimensions")
    if oracle.get("label_count") != EXPECTED_STATE_COUNT or oracle.get("split", {}).get("fold_count") != int(fold_count):
        raise ValueError("oracle label/split contract mismatch")
    if oracle.get("source", {}).get("dataset_manifest_sha256") != data["identity"]["manifest_sha256"]:
        raise ValueError("oracle and dataset manifest identity mismatch")
    labels = list(oracle["labels"])
    split = oracle["split"]
    state_ids = [str(row["state_id"]) for row in labels]
    if len(set(state_ids)) != EXPECTED_STATE_COUNT:
        raise ValueError("oracle state identities are not unique")
    history_manifest = _load_history_manifest(dataset_root)
    history_by_length: Dict[int, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = {}
    for length in (1, 5, 10):
        history_by_length[length] = build_history_tensors(labels, data["states"], history_manifest, sequence_length=length)
    raw_masks = np.stack([np.asarray(data["states"][state_id][22:127], dtype=np.float32).astype(bool) for state_id in state_ids])
    # state_sufficiency state features are vector(22), mask(105), depth(72), prev(105).
    if raw_masks.shape != (EXPECTED_STATE_COUNT, ACTION_DIM):
        raise ValueError("state feature mask shape mismatch")
    state_index_by_id = {state_id: index for index, state_id in enumerate(state_ids)}
    folds = np.asarray([int(split["state_to_fold"][state_id]) for state_id in state_ids], dtype=np.int64)
    pair_rows = [row for row in data["pairs"] if not bool(row["return_tie"])]
    pair_state_indices = np.asarray([state_index_by_id[str(row["state_id"])] for row in pair_rows], dtype=np.int64)
    pair_folds = folds[pair_state_indices]
    action_labels = np.asarray([int(row["oracle_action"]) for row in labels], dtype=np.int64)
    action_logits: Dict[str, np.ndarray] = {}
    pair_probabilities: Dict[str, np.ndarray] = {}
    histories: Dict[str, Dict[str, Any]] = {}
    fold_training: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    model_order = list(MODEL_SPECS)
    for model_number, (model_name, (kind, length)) in enumerate(MODEL_SPECS.items()):
        history, valid, provenance = history_by_length[length]
        histories[model_name] = provenance
        action_oof = np.zeros((len(labels), ACTION_DIM), dtype=np.float32)
        pair_oof = np.zeros(len(pair_rows), dtype=np.float64)
        for fold in range(int(fold_count)):
            train_states = np.flatnonzero(folds != fold)
            test_states = np.flatnonzero(folds == fold)
            normalizer = _Standardizer.fit(history[train_states], valid[train_states])
            normalized_history = normalizer.transform(history, valid)
            action_test, action_info = _fit_action_model(
                kind, length, normalized_history[train_states], valid[train_states], action_labels[train_states], normalized_history[test_states], valid[test_states], seed=int(seed) + model_number * 1000 + fold * 100 + 1, epochs=int(epochs), batch_size=int(action_batch_size)
            )
            action_oof[test_states] = action_test
            pair_train = np.flatnonzero(pair_folds != fold)
            pair_test = np.flatnonzero(pair_folds == fold)
            pair_train_states = pair_state_indices[pair_train]
            pair_test_states = pair_state_indices[pair_test]
            pair_test_probs, pair_info = _fit_pair_model(
                kind, length, normalized_history[pair_train_states], valid[pair_train_states], np.asarray([int(pair_rows[i]["action1"]) for i in pair_train], dtype=np.int64), np.asarray([int(pair_rows[i]["action2"]) for i in pair_train], dtype=np.int64), np.asarray([int(pair_rows[i]["label"]) for i in pair_train], dtype=np.int64), normalized_history[pair_test_states], valid[pair_test_states], np.asarray([int(pair_rows[i]["action1"]) for i in pair_test], dtype=np.int64), np.asarray([int(pair_rows[i]["action2"]) for i in pair_test], dtype=np.int64), seed=int(seed) + model_number * 1000 + fold * 100 + 2, epochs=int(epochs), batch_size=int(pair_batch_size)
            )
            pair_oof[pair_test] = pair_test_probs
            fold_training[model_name].append({"fold_id": fold, "action": action_info, "pairwise": pair_info, "normalizer_fit_state_count": int(len(train_states))})
        action_logits[model_name] = action_oof
        pair_probabilities[model_name] = pair_oof

    source = {
        "dataset_id": data["identity"]["dataset_id"],
        "dataset_manifest_sha256": data["identity"]["manifest_sha256"],
        "transitions_sha256": data["identity"]["transitions_sha256"],
        "pairwise_sha256": data["identity"]["pairwise_sha256"],
        "dataset_root": str(dataset_root),
        "oracle_artifact": str(oracle_path),
        "oracle_artifact_sha256": sha256_file(oracle_path),
        "state_count": int(data["identity"]["state_count"]),
        "transition_count": int(data["identity"]["transition_count"]),
        "pair_count": len(pair_rows),
        "mission_count": len(data["missions"]),
        "observation_contract": data["identity"]["observation_contract"],
        "task_contract_sha256": data["identity"]["task_contract_sha256"],
        "q_features_used": False,
        "future_information_used": False,
        "production_modified": False,
        "replay_modified": False,
        "rl_training_executed": False,
        "runtime_started": False,
    }
    oracle_metrics: Dict[str, Any] = {"schema_id": ORACLE_MEMORY_METRICS_SCHEMA_ID, "source": source, "history": histories, "models": {}}
    pair_metrics: Dict[str, Any] = {"schema_id": MEMORY_METRICS_SCHEMA_ID, "source": source, "history": histories, "models": {}}
    action_bootstrap_values: Dict[str, Dict[str, Mapping[str, Sequence[float]]]] = {}
    pair_bootstrap_values: Dict[str, Dict[str, Mapping[str, Sequence[float]]]] = {}
    for model_name in model_order:
        metrics, values = _action_metrics(action_logits[model_name], labels, raw_masks, state_ids)
        oracle_metrics["models"][model_name] = metrics
        action_bootstrap_values[model_name] = values
        metrics, values = _pair_metrics(pair_probabilities[model_name], pair_rows)
        pair_metrics["models"][model_name] = metrics
        pair_bootstrap_values[model_name] = values
    bootstrap: Dict[str, Any] = {"schema_id": MEMORY_BOOTSTRAP_SCHEMA_ID, "cluster_unit": "mission_id", "repeats": int(bootstrap_repeats), "models": {}, "deltas_vs_M0": {}}
    for model_number, model_name in enumerate(model_order):
        bootstrap["models"][model_name] = {
            "oracle_action_top1": _bootstrap(action_bootstrap_values[model_name]["top1"], repeats=int(bootstrap_repeats), seed=int(seed) + 1000 + model_number * 10),
            "oracle_action_top5": _bootstrap(action_bootstrap_values[model_name]["top5"], repeats=int(bootstrap_repeats), seed=int(seed) + 1001 + model_number * 10),
            "pairwise_accuracy": _bootstrap(pair_bootstrap_values[model_name]["accuracy"], repeats=int(bootstrap_repeats), seed=int(seed) + 1002 + model_number * 10),
            "pairwise_auc": _bootstrap(pair_bootstrap_values[model_name]["auc"], repeats=int(bootstrap_repeats), seed=int(seed) + 1003 + model_number * 10),
        }
        if model_name != "M0_current_state":
            bootstrap["deltas_vs_M0"][model_name] = {
                "oracle_action_top1": _paired_delta_bootstrap(action_bootstrap_values, "M0_current_state", model_name, "top1", repeats=int(bootstrap_repeats), seed=int(seed) + 2000 + model_number * 10),
                "oracle_action_top5": _paired_delta_bootstrap(action_bootstrap_values, "M0_current_state", model_name, "top5", repeats=int(bootstrap_repeats), seed=int(seed) + 2001 + model_number * 10),
                "pairwise_accuracy": _paired_delta_bootstrap(pair_bootstrap_values, "M0_current_state", model_name, "accuracy", repeats=int(bootstrap_repeats), seed=int(seed) + 2002 + model_number * 10),
                "pairwise_auc": _paired_delta_bootstrap(pair_bootstrap_values, "M0_current_state", model_name, "auc", repeats=int(bootstrap_repeats), seed=int(seed) + 2003 + model_number * 10),
            }
    config = {
        "schema_id": MEMORY_STATE_SUFFICIENCY_AUDIT_SCHEMA_ID,
        "seed": int(seed),
        "fold_count": int(fold_count),
        "bootstrap_repeats": int(bootstrap_repeats),
        "epochs": int(epochs),
        "action_batch_size": int(action_batch_size),
        "pair_batch_size": int(pair_batch_size),
        "hidden_dim": HIDDEN_DIM,
        "hidden_layers": HIDDEN_LAYERS,
        "optimizer": "Adam",
        "learning_rate": 1.0e-3,
        "device": "cpu",
        "normalizer_fit_scope": "training_fold_valid_history_frames_only",
        "models": MODEL_SPECS,
        "state_feature_dim": SEQUENCE_FEATURE_DIM,
        "state_features": ["vector_22", "legal_action_mask_105", "depth_9x8_mean_pool", "previous_action_one_hot_105"],
        "pairwise_rows_used": len(pair_rows),
    }
    all_prediction_rows = []
    for model_name, logits in action_logits.items():
        all_prediction_rows.append((model_name, logits))
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(out_dir / "oracle_memory_metrics.json", {"config": config, **oracle_metrics, "fold_training": dict(fold_training)})
    _json_dump(out_dir / "memory_metrics.json", {"config": config, **pair_metrics, "fold_training": dict(fold_training)})
    _json_dump(out_dir / "bootstrap.json", bootstrap)
    _write_predictions(out_dir / "predictions.csv", labels, raw_masks, action_logits, pair_probabilities, pair_rows)
    _write_report(out_dir / "report_zh.md", source, histories, oracle_metrics, pair_metrics, bootstrap)
    return {"source": source, "config": config, "oracle_metrics": oracle_metrics, "pair_metrics": pair_metrics, "bootstrap": bootstrap, "out_dir": str(out_dir)}
