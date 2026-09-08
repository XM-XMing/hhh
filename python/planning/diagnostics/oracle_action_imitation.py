"""Diagnostic-only supervised imitation of observed better actions.

This module measures whether the action selected by an observed-action oracle
can be recovered from the recorded state.  The oracle is restricted to
actions that were actually executed for the same state; it never evaluates an
unexecuted action and never loads a Q/critic model.  B0 is the frozen BC
checkpoint.  B1 and B2 are small CPU classifiers trained only inside the
diagnostic output boundary.
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

from planning.bc.model import VectorNormalizer, build_model, mask_logits
from planning.contracts.feature import validate_checkpoint_policy_input_contract
from planning.diagnostics.state_sufficiency import (
    ACTION_DIM,
    build_mission_split,
    load_audit_dataset,
)


ORACLE_ACTION_IMITATION_FEASIBILITY_SCHEMA_ID = "oracle_action_imitation_feasibility_v1"
ORACLE_ACTION_DATASET_SCHEMA_ID = "oracle_action_dataset_v1"
ORACLE_ACTION_POLICY_METRICS_SCHEMA_ID = "oracle_action_imitation_policy_metrics_v1"
ORACLE_ACTION_BOOTSTRAP_SCHEMA_ID = "oracle_action_imitation_bootstrap_v1"
EXPECTED_STATE_COUNT = 479
EXPECTED_TRANSITION_COUNT = 2850
DEFAULT_SEED = 20260908
DEFAULT_FOLD_COUNT = 3
DEFAULT_BOOTSTRAP_REPEATS = 5000
DEFAULT_EPOCHS = 120
DEFAULT_BATCH_SIZE = 64
DEFAULT_HIDDEN_DIM = 128
DEFAULT_HIDDEN_LAYERS = 2
DEFAULT_LEARNING_RATE = 1.0e-3


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
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _stats(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("non-finite imitation statistic")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


class _Standardizer:
    """Fold-local feature standardizer; never fitted on a held-out fold."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "_Standardizer":
        array = np.asarray(values, dtype=np.float32)
        mean = array.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = array.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1.0e-6] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)


class _PolicyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int, action_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.extend((nn.Linear(current, int(hidden_dim)), nn.ReLU()))
            current = int(hidden_dim)
        layers.append(nn.Linear(current, int(action_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _load_raw_states(dataset_root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    states: Dict[str, Dict[str, np.ndarray]] = {}
    for path in sorted((Path(dataset_root) / "states").glob("*.npz")):
        with np.load(str(path), allow_pickle=False) as loaded:
            required = {"state_id", "vector", "depth", "mask", "previous_action"}
            missing = required.difference(loaded.files)
            if missing:
                raise ValueError("{} missing state fields {}".format(path, sorted(missing)))
            state_id = str(np.asarray(loaded["state_id"]).reshape(-1)[0])
            if state_id in states:
                raise ValueError("duplicate raw state {}".format(state_id))
            vector = np.asarray(loaded["vector"], dtype=np.float32).reshape(-1)
            depth = np.asarray(loaded["depth"], dtype=np.float32)
            mask = np.asarray(loaded["mask"], dtype=np.bool_).reshape(-1)
            previous = np.asarray(loaded["previous_action"], dtype=np.int64).reshape(-1)
            if vector.size != 22 or depth.shape != (90, 160) or mask.size != ACTION_DIM or previous.size != 1:
                raise ValueError("state {} violates diagnostic observation dimensions".format(state_id))
            if not np.isfinite(vector).all() or not np.isfinite(depth).all():
                raise ValueError("state {} contains non-finite observation".format(state_id))
            if not np.any(mask):
                raise ValueError("state {} has no valid action".format(state_id))
            if int(previous[0]) < -1 or int(previous[0]) >= ACTION_DIM:
                raise ValueError("state {} has invalid previous action".format(state_id))
            states[state_id] = {
                "vector": vector,
                "depth": depth,
                "mask": mask,
                "previous_action": previous,
            }
    if not states:
        raise ValueError("no raw states found")
    return states


def build_oracle_labels(transitions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Create one label per state using only recorded action branches."""

    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in transitions:
        grouped[str(row["state_id"])].append(row)
    labels: List[Dict[str, Any]] = []
    for state_id in sorted(grouped):
        branches = grouped[state_id]
        bc = [row for row in branches if str(row["action_source"]).upper() == "BC"]
        if len(bc) != 1:
            raise ValueError("state {} must have exactly one BC branch".format(state_id))
        by_action: Dict[int, Mapping[str, Any]] = {}
        for row in branches:
            action = int(row["action"])
            if action in by_action:
                raise ValueError("duplicate observed action {} for state {}".format(action, state_id))
            by_action[action] = row
        best_return = max(float(row["episode_return"]) for row in branches)
        candidates = [row for row in branches if float(row["episode_return"]) == best_return]
        oracle = min(candidates, key=lambda row: int(row["action"]))
        bc_row = bc[0]
        bc_return = float(bc_row["episode_return"])
        oracle_return = float(oracle["episode_return"])
        labels.append(
            {
                "state_id": state_id,
                "state_hash": state_id,
                "mission_id": str(bc_row["mission_id"]),
                "episode_id": str(bc_row["episode_id"]),
                "step_id": int(bc_row["step_id"]),
                "oracle_action": int(oracle["action"]),
                "oracle_action_source": str(oracle["action_source"]),
                "oracle_return": oracle_return,
                "oracle_success": int(oracle["success"]),
                "bc_action": int(bc_row["action"]),
                "bc_action_source": str(bc_row["action_source"]),
                "bc_return": bc_return,
                "bc_success": int(bc_row["success"]),
                "return_advantage": max(0.0, oracle_return - bc_return),
                "oracle_tie_count": len(candidates),
                "observed_actions": sorted(by_action),
                "observed_action_returns": {
                    str(action): float(row["episode_return"]) for action, row in sorted(by_action.items())
                },
                "observed_action_success": {
                    str(action): int(row["success"]) for action, row in sorted(by_action.items())
                },
                "observed_action_sources": {
                    str(action): str(row["action_source"]) for action, row in sorted(by_action.items())
                },
            }
        )
    return labels


def _weighted_return_advantage(labels: Sequence[Mapping[str, Any]], indices: np.ndarray) -> np.ndarray:
    raw = np.asarray(
        [max(0.0, float(labels[int(index)]["return_advantage"])) for index in indices.tolist()],
        dtype=np.float32,
    )
    if raw.size == 0:
        raise ValueError("cannot construct weights for an empty training fold")
    scale = float(np.mean(raw))
    if scale <= 1.0e-8:
        return np.ones_like(raw, dtype=np.float32)
    # Proportional to the observed return advantage; mean-one scaling only
    # keeps optimizer magnitudes comparable to B1 and does not change ordering.
    return (raw / scale).astype(np.float32)


def _train_policy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    model = _PolicyMLP(train_x.shape[1], DEFAULT_HIDDEN_DIM, DEFAULT_HIDDEN_LAYERS, ACTION_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    x_tensor = torch.from_numpy(np.asarray(train_x, dtype=np.float32))
    y_tensor = torch.from_numpy(np.asarray(train_y, dtype=np.int64))
    weight_tensor = torch.from_numpy(np.asarray(train_weights, dtype=np.float32))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 17)
    model.train()
    last_loss = None
    for _ in range(int(epochs)):
        order = torch.randperm(x_tensor.shape[0], generator=generator)
        for start in range(0, x_tensor.shape[0], int(batch_size)):
            indices = order[start : start + int(batch_size)]
            logits = model(x_tensor[indices])
            losses = nn.functional.cross_entropy(logits, y_tensor[indices], reduction="none")
            weights = weight_tensor[indices]
            denominator = torch.sum(weights)
            if float(denominator.detach().cpu().item()) <= 1.0e-8:
                loss = torch.mean(losses)
            else:
                loss = torch.sum(losses * weights) / denominator
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(test_x, dtype=np.float32))).cpu().numpy()
    if not np.isfinite(logits).all():
        raise ValueError("policy baseline emitted non-finite logits")
    return logits.astype(np.float32), {
        "train_count": int(len(train_x)),
        "test_count": int(len(test_x)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "last_train_loss": last_loss,
        "seed": int(seed),
        "weight_stats": _stats(weight_tensor.cpu().numpy().tolist()),
    }


def _load_checkpoint(path: Path) -> Tuple[Mapping[str, Any], nn.Module, VectorNormalizer]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    try:
        checkpoint = torch.load(str(resolved), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(resolved), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("BC checkpoint is not a mapping")
    validate_checkpoint_policy_input_contract(dict(checkpoint))
    if int(checkpoint.get("num_actions", ACTION_DIM)) != ACTION_DIM:
        raise ValueError("BC checkpoint action dimension mismatch")
    if int(checkpoint.get("depth_history_frames", 1)) != 1:
        raise ValueError("diagnostic B0 requires the verified one-frame BC checkpoint")
    model = build_model(
        nn,
        vec_dim=int(checkpoint.get("vec_dim", 127)),
        num_actions=int(checkpoint.get("num_actions", ACTION_DIM)),
        depth_channels=int(checkpoint.get("depth_history_frames", 1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return checkpoint, model, VectorNormalizer.from_checkpoint(dict(checkpoint))


def _b0_logits(
    model: nn.Module,
    normalizer: VectorNormalizer,
    raw_states: Mapping[str, Mapping[str, np.ndarray]],
    state_ids: Sequence[str],
) -> np.ndarray:
    depths = []
    vectors = []
    for state_id in state_ids:
        state = raw_states[str(state_id)]
        continuous = normalizer.transform_continuous(state["vector"].reshape(1, -1))
        previous = int(state["previous_action"][0])
        previous_one_hot = np.zeros((1, ACTION_DIM), dtype=np.float32)
        if previous >= 0:
            previous_one_hot[0, previous] = 1.0
        vectors.append(np.concatenate((continuous, previous_one_hot), axis=1)[0])
        depths.append(state["depth"])
    depth_tensor = torch.from_numpy(np.stack(depths, axis=0)[:, None, :, :].astype(np.float32))
    vector_tensor = torch.from_numpy(np.stack(vectors, axis=0).astype(np.float32))
    with torch.no_grad():
        logits = model(depth_tensor, vector_tensor).cpu().numpy()
    if not np.isfinite(logits).all():
        raise ValueError("BC checkpoint emitted non-finite logits")
    return logits.astype(np.float32)


def _masked_order(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).copy()
    valid = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or valid.ndim != 1 or values.size != valid.size:
        raise ValueError("logit/mask shape mismatch")
    if not np.any(valid):
        raise ValueError("no valid action for prediction")
    values[~valid] = -1.0e30
    return np.argsort(-values, kind="mergesort").astype(np.int64)


def _bootstrap(values_by_mission: Mapping[str, Sequence[float]], *, repeats: int, seed: int) -> Dict[str, Any]:
    valid = {
        str(mission): np.asarray(values, dtype=np.float64)
        for mission, values in values_by_mission.items()
        if len(values) > 0
    }
    if not valid:
        return {
            "schema_id": ORACLE_ACTION_BOOTSTRAP_SCHEMA_ID,
            "cluster_unit": "mission_id",
            "mission_count": 0,
            "repeats": int(repeats),
            "seed": int(seed),
            "estimate": None,
            "ci95": None,
            "status": "NOT_ESTIMABLE_NO_OBSERVED_ACTION_COVERAGE",
        }
    missions = sorted(valid)
    means = np.asarray([float(np.mean(valid[mission])) for mission in missions], dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    estimates = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        sample = rng.randint(0, len(missions), size=len(missions))
        estimates[index] = float(np.mean(means[sample]))
    return {
        "schema_id": ORACLE_ACTION_BOOTSTRAP_SCHEMA_ID,
        "cluster_unit": "mission_id",
        "mission_count": len(missions),
        "repeats": int(repeats),
        "seed": int(seed),
        "estimate": float(np.mean(means)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "status": "PASS",
    }


def _prediction_metrics(
    model_name: str,
    logits: np.ndarray,
    labels: Sequence[Mapping[str, Any]],
    raw_states: Mapping[str, Mapping[str, np.ndarray]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(logits) != len(labels):
        raise ValueError("prediction/label count mismatch")
    top1: List[int] = []
    top3: List[int] = []
    top5: List[int] = []
    covered_delta: List[float] = []
    covered_selected_return: List[float] = []
    covered_selected_success: List[float] = []
    rows: List[Dict[str, Any]] = []
    mission_accuracy: MutableMapping[str, List[float]] = defaultdict(list)
    mission_top3: MutableMapping[str, List[float]] = defaultdict(list)
    mission_top5: MutableMapping[str, List[float]] = defaultdict(list)
    mission_delta: MutableMapping[str, List[float]] = defaultdict(list)
    mission_selected_return: MutableMapping[str, List[float]] = defaultdict(list)
    mission_selected_success: MutableMapping[str, List[float]] = defaultdict(list)
    for index, label in enumerate(labels):
        state_id = str(label["state_id"])
        order = _masked_order(logits[index], raw_states[state_id]["mask"])
        oracle_action = int(label["oracle_action"])
        predicted = int(order[0])
        hit1 = int(predicted == oracle_action)
        hit3 = int(oracle_action in set(order[:3].tolist()))
        hit5 = int(oracle_action in set(order[:5].tolist()))
        top1.append(hit1)
        top3.append(hit3)
        top5.append(hit5)
        mission = str(label["mission_id"])
        mission_accuracy[mission].append(float(hit1))
        mission_top3[mission].append(float(hit3))
        mission_top5[mission].append(float(hit5))
        observed_returns = dict(label["observed_action_returns"])
        selected_key = str(predicted)
        selected_return = observed_returns.get(selected_key)
        selected_success = dict(label["observed_action_success"]).get(selected_key)
        if selected_return is not None:
            selected_return = float(selected_return)
            selected_success = float(selected_success)
            delta = selected_return - float(label["bc_return"])
            covered_delta.append(delta)
            covered_selected_return.append(selected_return)
            covered_selected_success.append(selected_success)
            mission_delta[mission].append(delta)
            mission_selected_return[mission].append(selected_return)
            mission_selected_success[mission].append(selected_success)
        rows.append(
            {
                "model": model_name,
                "state_id": state_id,
                "state_hash": state_id,
                "mission_id": mission,
                "episode_id": str(label["episode_id"]),
                "step_id": int(label["step_id"]),
                "oracle_action": oracle_action,
                "bc_action": int(label["bc_action"]),
                "predicted_action": predicted,
                "top3_hit": hit3,
                "top5_hit": hit5,
                "observed_action_coverage": int(selected_return is not None),
                "selected_action_return": selected_return,
                "bc_action_return": float(label["bc_return"]),
                "selected_minus_bc_return": None if selected_return is None else float(selected_return - float(label["bc_return"])),
                "selected_action_success": selected_success,
                "oracle_action_source": str(label["oracle_action_source"]),
                "return_advantage": float(label["return_advantage"]),
            }
        )
    count = len(labels)
    metrics = {
        "model": model_name,
        "state_count": count,
        "mission_count": len(mission_accuracy),
        "action_accuracy": float(np.mean(top1)),
        "top3_accuracy": float(np.mean(top3)),
        "top5_accuracy": float(np.mean(top5)),
        "oracle_action_source_counts": {
            source: int(sum(str(label["oracle_action_source"]).upper() == source for label in labels))
            for source in ("BC", "TEACHER", "NEIGHBOR", "RANDOM")
        },
        "observed_action_return_coverage": float(len(covered_delta) / count),
        "observed_action_return_covered_count": len(covered_delta),
        "observed_action_return": _stats(covered_selected_return),
        "bc_action_return_on_covered_states": _stats(
            float(row["bc_return"]) for row, prediction in zip(labels, rows) if prediction["observed_action_coverage"]
        ),
        "selected_minus_bc_return_on_covered_states": _stats(covered_delta),
        "selected_action_success_on_covered_states": _stats(covered_selected_success),
        "mission_macro_action_accuracy": float(np.mean([np.mean(values) for values in mission_accuracy.values()])),
        "mission_macro_top3_accuracy": float(np.mean([np.mean(values) for values in mission_top3.values()])),
        "mission_macro_top5_accuracy": float(np.mean([np.mean(values) for values in mission_top5.values()])),
        "mission_cluster_values": {
            "action_accuracy": {key: list(values) for key, values in sorted(mission_accuracy.items())},
            "top3_accuracy": {key: list(values) for key, values in sorted(mission_top3.items())},
            "top5_accuracy": {key: list(values) for key, values in sorted(mission_top5.items())},
            "selected_minus_bc_return": {key: list(values) for key, values in sorted(mission_delta.items())},
            "selected_return": {key: list(values) for key, values in sorted(mission_selected_return.items())},
            "selected_success": {key: list(values) for key, values in sorted(mission_selected_success.items())},
        },
    }
    return metrics, rows


def _write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "model",
        "state_id",
        "state_hash",
        "mission_id",
        "episode_id",
        "step_id",
        "oracle_action",
        "bc_action",
        "predicted_action",
        "top3_hit",
        "top5_hit",
        "observed_action_coverage",
        "selected_action_return",
        "bc_action_return",
        "selected_minus_bc_return",
        "selected_action_success",
        "oracle_action_source",
        "return_advantage",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    *,
    source: Mapping[str, Any],
    split: Mapping[str, Any],
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    b0 = metrics["models"]["B0_original_bc"]
    b1 = metrics["models"]["B1_oracle_ce"]
    b2 = metrics["models"]["B2_oracle_advantage_weighted"]
    learned = [b1, b2]
    best = max(learned, key=lambda item: (item["action_accuracy"], item["top5_accuracy"]))
    best_bootstrap = bootstrap[best["model"]]
    coverage_ok = any(item["observed_action_return_coverage"] >= 0.5 for item in learned)
    if best["action_accuracy"] > b0["action_accuracy"] and coverage_ok:
        conclusion = "SUPPORTED_WITH_OBSERVED_ACTION_COVERAGE"
        answer = "至少一个监督 baseline 在 mission-level holdout 上超过冻结 BC，且有足够已执行 action 覆盖；这支持从 state 恢复部分 observed oracle action，但不证明可泛化到未执行动作。"
    elif not coverage_ok:
        conclusion = "INCONCLUSIVE_OBSERVED_RETURN_COVERAGE"
        answer = "监督模型选择的动作多数没有在该 state 被真实执行，因而 selected-action return proxy 不足以判断可实现提升；不能用未执行动作补齐。"
    else:
        conclusion = "NOT_SUPPORTED_OVER_BC"
        answer = "本数据和固定 split 上监督 oracle imitation 没有超过冻结 BC 的 action 选择表现；不能据此把 AWAC 失败单独归因于 Critic/objective。"
    lines = [
        "# ORACLE ACTION IMITATION FEASIBILITY V1",
        "",
        "本报告是诊断-only 结果；未执行 AWAC、Critic/Q 网络、Unity/Bridge/ROS，未修改 production Replay。",
        "",
        "## 数据与标签",
        "",
        "- dataset: `{}`；states=`{}`；transitions=`{}`；missions=`{}`。".format(source["dataset_id"], source["state_count"], source["transition_count"], source["mission_count"]),
        "- split: `{}` missions, `{}` folds, seed `{}`；同一 mission 的 state 没有跨 fold。".format(split["mission_count"], split["fold_count"], split["seed"]),
        "- oracle: 每个 state 只在已真实执行的 action 中选择最高 episode return；并列取最低 action id。",
        "- B2 权重: `max(oracle_return - BC_return, 0)`，仅按训练折归一到 mean=1；没有 Q/未执行 action/future feature。",
        "",
        "## 三个 baseline",
        "",
        "| model | action accuracy | top-3 | top-5 | mission macro top-1 | observed-action coverage | selected return delta (covered) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in ("B0_original_bc", "B1_oracle_ce", "B2_oracle_advantage_weighted"):
        item = metrics["models"][model_name]
        delta = item["selected_minus_bc_return_on_covered_states"]["mean"]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
                model_name,
                item["action_accuracy"],
                item["top3_accuracy"],
                item["top5_accuracy"],
                item["mission_macro_action_accuracy"],
                item["observed_action_return_coverage"],
                "-" if delta is None else "{:.6f}".format(delta),
            )
        )
    lines.extend(
        [
            "",
            "## Mission-cluster bootstrap",
            "",
            "- 5000 次 bootstrap 的 cluster unit 是 mission；不是把 state 当作独立任务。",
            "- B0 action accuracy CI: `{}`。".format(bootstrap["B0_original_bc"]["action_accuracy"]["ci95"]),
            "- B1 action accuracy CI: `{}`。".format(bootstrap["B1_oracle_ce"]["action_accuracy"]["ci95"]),
            "- B2 action accuracy CI: `{}`。".format(bootstrap["B2_oracle_advantage_weighted"]["action_accuracy"]["ci95"]),
            "- B1 selected-return delta CI（仅已执行 action 覆盖的 mission）: `{}`。".format(bootstrap["B1_oracle_ce"]["selected_minus_bc_return"]["ci95"]),
            "- B2 selected-return delta CI（仅已执行 action 覆盖的 mission）: `{}`。".format(bootstrap["B2_oracle_advantage_weighted"]["selected_minus_bc_return"]["ci95"]),
            "",
            "## 结论",
            "",
            "- `ORACLE_ACTION_LEARNABILITY = {}`。".format(conclusion),
            "- `BEST_LEARNED_MODEL = {}`。".format(best["model"]),
            "- `ORACLE_ACTION_CAN_BE_LEARNED_FROM_STATE = {}`。".format("YES" if conclusion == "SUPPORTED_WITH_OBSERVED_ACTION_COVERAGE" else "NOT_ESTABLISHED"),
            "- {}".format(answer),
            "- 即使 observed-action oracle 存在较大上界，也不能把它等同于可部署策略成功率；B0/B1/B2 都只是在 31 个 BC-failure mission 的 state slice 上验证。",
            "",
            "## 身份与保护边界",
            "",
            "- BC checkpoint SHA256: `{}`。".format(source["bc_checkpoint_sha256"]),
            "- dataset manifest SHA256: `{}`。".format(source["dataset_manifest_sha256"]),
            "- `production_replay_modified=NO`, `production_checkpoint_modified=NO`, `rl_training_executed=NO`, `runtime_started=NO`。",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_oracle_action_imitation(
    *,
    dataset_root: Path,
    feasibility_root: Path,
    bc_checkpoint: Path,
    out_dir: Path,
    seed: int = DEFAULT_SEED,
    fold_count: int = DEFAULT_FOLD_COUNT,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output: {}".format(out_dir))
    dataset_root = Path(dataset_root).expanduser().resolve()
    feasibility_root = Path(feasibility_root).expanduser().resolve()
    checkpoint_path = Path(bc_checkpoint).expanduser().resolve()
    if not (feasibility_root / "oracle_upper_bound.json").is_file():
        raise FileNotFoundError(str(feasibility_root / "oracle_upper_bound.json"))
    feasibility = json.loads((feasibility_root / "oracle_upper_bound.json").read_text(encoding="utf-8"))
    data = load_audit_dataset(dataset_root)
    if data["identity"]["state_count"] != EXPECTED_STATE_COUNT or data["identity"]["transition_count"] != EXPECTED_TRANSITION_COUNT:
        raise ValueError("unexpected multi-action dimensions")
    if bool(data["identity"]["production_replay"]) or bool(data["identity"]["training_consumed"]):
        raise ValueError("refusing production or consumed diagnostic replay")
    labels = build_oracle_labels(data["transitions"])
    if len(labels) != EXPECTED_STATE_COUNT:
        raise ValueError("oracle label state count mismatch")
    raw_states = _load_raw_states(dataset_root)
    if set(raw_states) != {str(label["state_id"]) for label in labels}:
        raise ValueError("raw state/label identity mismatch")
    split = build_mission_split(data["missions"], seed=int(seed), fold_count=int(fold_count))
    state_ids = [str(label["state_id"]) for label in labels]
    state_features = np.stack([data["states"][state_id] for state_id in state_ids]).astype(np.float32)
    oracle_actions = np.asarray([int(label["oracle_action"]) for label in labels], dtype=np.int64)
    bc_actions = np.asarray([int(label["bc_action"]) for label in labels], dtype=np.int64)
    folds = np.asarray([int(split["state_to_fold"][str(label["state_id"])]) for label in labels], dtype=np.int64)
    raw_masks = np.stack([raw_states[state_id]["mask"] for state_id in state_ids]).astype(bool)
    checkpoint, b0_model, normalizer = _load_checkpoint(checkpoint_path)
    b0_logits = _b0_logits(b0_model, normalizer, raw_states, state_ids)
    b0_predictions = np.asarray([int(_masked_order(b0_logits[i], raw_masks[i])[0]) for i in range(len(labels))], dtype=np.int64)
    b0_match = float(np.mean(b0_predictions == bc_actions))

    b1_logits = np.zeros((len(labels), ACTION_DIM), dtype=np.float32)
    b2_logits = np.zeros((len(labels), ACTION_DIM), dtype=np.float32)
    fold_training: Dict[str, List[Dict[str, Any]]] = {"B1_oracle_ce": [], "B2_oracle_advantage_weighted": []}
    for fold in range(int(fold_count)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        standardizer = _Standardizer.fit(state_features[train])
        train_x = standardizer.transform(state_features[train])
        test_x = standardizer.transform(state_features[test])
        ones = np.ones(len(train), dtype=np.float32)
        weighted = _weighted_return_advantage(labels, train)
        b1, b1_info = _train_policy(
            train_x,
            oracle_actions[train],
            ones,
            test_x,
            seed=int(seed) + fold * 100 + 1,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=DEFAULT_LEARNING_RATE,
        )
        b2, b2_info = _train_policy(
            train_x,
            oracle_actions[train],
            weighted,
            test_x,
            seed=int(seed) + fold * 100 + 1,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=DEFAULT_LEARNING_RATE,
        )
        b1_logits[test] = b1
        b2_logits[test] = b2
        fold_training["B1_oracle_ce"].append({"fold_id": fold, **b1_info, "normalizer_fit_count": int(len(train))})
        fold_training["B2_oracle_advantage_weighted"].append({"fold_id": fold, **b2_info, "normalizer_fit_count": int(len(train))})

    source = {
        "dataset_id": data["identity"]["dataset_id"],
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": data["identity"]["manifest_sha256"],
        "transitions_sha256": data["identity"]["transitions_sha256"],
        "pairwise_sha256": data["identity"]["pairwise_sha256"],
        "state_count": int(data["identity"]["state_count"]),
        "transition_count": int(data["identity"]["transition_count"]),
        "mission_count": len(data["missions"]),
        "feasibility_artifact": str(feasibility_root / "oracle_upper_bound.json"),
        "feasibility_artifact_sha256": sha256_file(feasibility_root / "oracle_upper_bound.json"),
        "bc_checkpoint": str(checkpoint_path),
        "bc_checkpoint_sha256": sha256_file(checkpoint_path),
        "bc_checkpoint_metadata_sha256": checkpoint.get("normalizer_sha256"),
        "observation_contract": data["identity"]["observation_contract"],
        "task_contract_sha256": data["identity"]["task_contract_sha256"],
        "oracle_source_restriction": "same-state-already-executed-actions-only",
        "bc_checkpoint_matches_recorded_branch_ratio": b0_match,
        "bc_checkpoint_matches_recorded_branch_count": int(np.count_nonzero(b0_predictions == bc_actions)),
    }
    config = {
        "schema_id": ORACLE_ACTION_POLICY_METRICS_SCHEMA_ID,
        "seed": int(seed),
        "fold_count": int(fold_count),
        "bootstrap_repeats": int(bootstrap_repeats),
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "hidden_layers": DEFAULT_HIDDEN_LAYERS,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam",
        "learning_rate": DEFAULT_LEARNING_RATE,
        "device": "cpu",
        "state_features": ["vector_22", "legal_action_mask_105", "depth_9x8_mean_pool", "previous_action_one_hot_105"],
        "state_feature_dim": int(state_features.shape[1]),
        "label": "oracle_action_highest_observed_episode_return",
        "B0": "frozen production BC checkpoint; deterministic masked argmax",
        "B1": "ordinary cross_entropy on oracle action",
        "B2": "cross_entropy weighted proportional to max(oracle_return - bc_return, 0), train-fold mean-one normalized",
        "normalizer_fit_scope": "training_fold_only",
        "q_features_used": False,
        "unexecuted_actions_used": False,
        "future_observation_features_used": False,
        "production_modified": False,
        "replay_modified": False,
        "runtime_started": False,
        "rl_training_executed": False,
    }
    metrics: Dict[str, Any] = {
        "schema_id": ORACLE_ACTION_POLICY_METRICS_SCHEMA_ID,
        "source": source,
        "config": config,
        "feasibility_reference": {
            "schema_id": feasibility.get("schema_id"),
            "conclusion": feasibility.get("conclusion", {}).get("result"),
            "source_sha256": sha256_file(feasibility_root / "oracle_upper_bound.json"),
        },
        "models": {},
        "fold_training": fold_training,
    }
    predictions: List[Dict[str, Any]] = []
    for model_name, logits in (
        ("B0_original_bc", b0_logits),
        ("B1_oracle_ce", b1_logits),
        ("B2_oracle_advantage_weighted", b2_logits),
    ):
        model_metrics, model_predictions = _prediction_metrics(model_name, logits, labels, raw_states)
        metrics["models"][model_name] = model_metrics
        predictions.extend(model_predictions)
    bootstrap: Dict[str, Any] = {
        "schema_id": ORACLE_ACTION_BOOTSTRAP_SCHEMA_ID,
        "cluster_unit": "mission_id",
        "repeats": int(bootstrap_repeats),
        "models": {},
    }
    for model_name in ("B0_original_bc", "B1_oracle_ce", "B2_oracle_advantage_weighted"):
        model_metrics = metrics["models"][model_name]
        values = model_metrics["mission_cluster_values"]
        bootstrap["models"][model_name] = {
            "action_accuracy": _bootstrap(values["action_accuracy"], repeats=int(bootstrap_repeats), seed=int(seed) + 1000),
            "top3_accuracy": _bootstrap(values["top3_accuracy"], repeats=int(bootstrap_repeats), seed=int(seed) + 1001),
            "top5_accuracy": _bootstrap(values["top5_accuracy"], repeats=int(bootstrap_repeats), seed=int(seed) + 1002),
            "selected_minus_bc_return": _bootstrap(values["selected_minus_bc_return"], repeats=int(bootstrap_repeats), seed=int(seed) + 1003),
            "selected_return": _bootstrap(values["selected_return"], repeats=int(bootstrap_repeats), seed=int(seed) + 1004),
            "selected_success": _bootstrap(values["selected_success"], repeats=int(bootstrap_repeats), seed=int(seed) + 1005),
        }
    dataset_payload = {
        "schema_id": ORACLE_ACTION_DATASET_SCHEMA_ID,
        "diagnostic_only": True,
        "oracle_definition": "For each state choose argmax episode_return only among already executed branches for that same state; ties use lowest action id.",
        "forbidden_inputs": ["Q values", "critic checkpoint", "unexecuted actions", "future outcomes as features"],
        "source": source,
        "split": split,
        "label_count": len(labels),
        "labels": labels,
        "production_modified": False,
        "replay_modified": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(out_dir / "oracle_dataset.json", dataset_payload)
    _json_dump(out_dir / "oracle_policy_metrics.json", metrics)
    _json_dump(out_dir / "bootstrap.json", bootstrap)
    _write_predictions(out_dir / "predictions.csv", predictions)
    _write_report(
        out_dir / "report_zh.md",
        source=source,
        split=split,
        metrics=metrics,
        bootstrap={name: bootstrap["models"][name] for name in bootstrap["models"]},
        config=config,
    )
    return {
        "source": source,
        "split": split,
        "config": config,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "out_dir": str(out_dir),
    }
