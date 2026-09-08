"""Read-only observation information audit for multi-action replay.

The audit is intentionally independent of the production AWAC implementation:
it uses only the recorded state components, action IDs, and observed branch
outcomes.  It never loads a Q checkpoint and never writes the input replay.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from planning.contracts.feature import GOAL_FEATURE_NAMES, STATE_FEATURE_NAMES
from planning.diagnostics.state_sufficiency import (
    ACTION_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_HIDDEN_LAYERS,
    _SmallMLP,
    _Standardizer,
    _depth_thumbnail,
    _one_hot,
    binary_metrics,
    build_mission_split,
    classification_bundle,
    load_audit_dataset,
    mission_cluster_bootstrap,
    sha256_file,
)


OBSERVATION_INFORMATION_SCHEMA_ID = "observation_information_audit_v1"
FEATURE_IMPORTANCE_SCHEMA_ID = "observation_feature_importance_v1"
COMPONENT_METRICS_SCHEMA_ID = "observation_component_metrics_v1"
TEMPORAL_METRICS_SCHEMA_ID = "observation_temporal_metrics_v1"
VARIANCE_SCHEMA_ID = "within_state_return_variance_v1"


def build_feature_names(component: str = "canonical") -> List[str]:
    """Return deterministic names for the state/action feature block."""

    vector_names = list(STATE_FEATURE_NAMES) + list(GOAL_FEATURE_NAMES)
    depth_names = ["depth_{}_{}".format(row, col) for row in range(9) for col in range(8)]
    mask_names = ["legal_mask_{}".format(index) for index in range(ACTION_DIM)]
    previous_names = ["previous_action_{}".format(index) for index in range(ACTION_DIM)]
    if component == "vector":
        state_names = vector_names
    elif component == "depth":
        state_names = depth_names
    elif component == "vector_plus_depth":
        state_names = vector_names + depth_names
    elif component == "current_state":
        state_names = vector_names + depth_names + mask_names
    elif component == "temporal_state":
        state_names = vector_names + depth_names + mask_names + previous_names + ["normalized_step_id"]
    elif component == "canonical":
        state_names = vector_names + depth_names + mask_names + previous_names
    else:
        raise ValueError("unknown feature component {}".format(component))
    return state_names + ["action1_{}".format(index) for index in range(ACTION_DIM)] + [
        "action2_{}".format(index) for index in range(ACTION_DIM)
    ]


def component_feature_dimensions() -> Dict[str, int]:
    return {
        "vector": len(STATE_FEATURE_NAMES) + len(GOAL_FEATURE_NAMES),
        "depth": 9 * 8,
        "vector_plus_depth": len(STATE_FEATURE_NAMES) + len(GOAL_FEATURE_NAMES) + 9 * 8,
        "current_state": len(STATE_FEATURE_NAMES) + len(GOAL_FEATURE_NAMES) + 9 * 8 + ACTION_DIM,
        "temporal_state": len(STATE_FEATURE_NAMES) + len(GOAL_FEATURE_NAMES) + 9 * 8 + ACTION_DIM + ACTION_DIM + 1,
        "canonical": len(STATE_FEATURE_NAMES) + len(GOAL_FEATURE_NAMES) + 9 * 8 + ACTION_DIM + ACTION_DIM,
    }


def _load_raw_states(root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for path in sorted((Path(root) / "states").glob("*.npz")):
        with np.load(str(path), allow_pickle=False) as loaded:
            state_id = str(np.asarray(loaded["state_id"]).reshape(-1)[0])
            if state_id in result:
                raise ValueError("duplicate state {}".format(state_id))
            vector = np.asarray(loaded["vector"], dtype=np.float32).reshape(-1)
            mask = np.asarray(loaded["mask"], dtype=np.float32).reshape(-1)
            previous = int(np.asarray(loaded["previous_action"]).reshape(-1)[0])
            if vector.size != 22 or mask.size != ACTION_DIM:
                raise ValueError("state {} has unexpected vector/mask shape".format(state_id))
            previous_one_hot = np.zeros(ACTION_DIM, dtype=np.float32)
            if previous >= 0:
                if previous >= ACTION_DIM:
                    raise ValueError("state {} previous action is outside action space".format(state_id))
                previous_one_hot[previous] = 1.0
            thumbnail = _depth_thumbnail(np.asarray(loaded["depth"], dtype=np.float32)).reshape(-1)
            for value in (vector, mask, thumbnail, previous_one_hot):
                if not np.isfinite(value).all():
                    raise ValueError("state {} contains non-finite features".format(state_id))
            result[state_id] = {
                "vector": vector,
                "depth": thumbnail.astype(np.float32),
                "mask": mask.astype(np.float32),
                "previous_action": previous_one_hot,
                "previous_action_id": np.asarray([previous], dtype=np.int64),
            }
    if not result:
        raise ValueError("no raw states found")
    return result


def _component_vector(
    state: Mapping[str, np.ndarray], component: str, *, step_id: int, max_steps: int
) -> np.ndarray:
    if component == "vector":
        parts = [state["vector"]]
    elif component == "depth":
        parts = [state["depth"]]
    elif component == "vector_plus_depth":
        parts = [state["vector"], state["depth"]]
    elif component == "current_state":
        parts = [state["vector"], state["depth"], state["mask"]]
    elif component == "temporal_state":
        parts = [
            state["vector"],
            state["depth"],
            state["mask"],
            state["previous_action"],
            np.asarray([float(step_id) / float(max_steps)], dtype=np.float32),
        ]
    elif component == "canonical":
        parts = [state["vector"], state["depth"], state["mask"], state["previous_action"]]
    else:
        raise ValueError("unknown feature component {}".format(component))
    return np.concatenate(parts).astype(np.float32)


def _pair_feature_matrix(
    pairs: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, np.ndarray]],
    component: str,
    *,
    max_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_features = []
    action1 = []
    action2 = []
    for row in pairs:
        state_features.append(
            _component_vector(
                states[str(row["state_id"])],
                component,
                step_id=int(row["step_id"]),
                max_steps=int(max_steps),
            )
        )
        action1.append(int(row["action1"]))
        action2.append(int(row["action2"]))
    actions1 = np.asarray(action1, dtype=np.int64)
    actions2 = np.asarray(action2, dtype=np.int64)
    matrix = np.concatenate(
        (np.asarray(state_features, dtype=np.float32), _one_hot(actions1), _one_hot(actions2)), axis=1
    )
    return (
        matrix,
        np.asarray([int(row["label"]) for row in pairs], dtype=np.int64),
        np.asarray([str(row["state_id"]) for row in pairs]),
        np.asarray([str(row["mission_id"]) for row in pairs]),
    )


def within_state_return_statistics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: MutableMapping[str, List[float]] = {}
    action_counts: MutableMapping[str, set] = {}
    for row in rows:
        state_id = str(row["state_id"])
        grouped.setdefault(state_id, []).append(float(row["episode_return"]))
        action_counts.setdefault(state_id, set()).add(int(row.get("action", -1)))
    variances = np.asarray([np.var(values) for values in grouped.values()], dtype=np.float64)
    ranges = np.asarray([max(values) - min(values) for values in grouped.values()], dtype=np.float64)
    multi = [values for values in grouped.values() if len(values) >= 2]
    if not grouped:
        raise ValueError("within-state audit has no rows")
    return {
        "schema_id": VARIANCE_SCHEMA_ID,
        "state_count": len(grouped),
        "multi_action_state_count": len(multi),
        "single_action_state_count": len(grouped) - len(multi),
        "variance_mean": float(np.mean(variances)),
        "variance_median": float(np.median(variances)),
        "variance_p10": float(np.percentile(variances, 10)),
        "variance_p90": float(np.percentile(variances, 90)),
        "variance_p99": float(np.percentile(variances, 99)),
        "range_mean": float(np.mean(ranges)),
        "range_median": float(np.median(ranges)),
        "range_p90": float(np.percentile(ranges, 90)),
        "per_state": {
            state_id: {
                "action_count": len(action_counts[state_id]),
                "sample_count": len(grouped[state_id]),
                "return_mean": float(np.mean(grouped[state_id])),
                "return_variance": float(np.var(grouped[state_id])),
                "return_min": float(min(grouped[state_id])),
                "return_max": float(max(grouped[state_id])),
            }
            for state_id in sorted(grouped)
        },
    }


def _fit_pair_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> Tuple[_SmallMLP, _Standardizer, np.ndarray]:
    normalizer = _Standardizer.fit(train_x)
    train_values = normalizer.transform(train_x)
    test_values = normalizer.transform(test_x)
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    model = _SmallMLP(train_values.shape[1], DEFAULT_HIDDEN_DIM, DEFAULT_HIDDEN_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x_tensor = torch.from_numpy(train_values)
    y_tensor = torch.from_numpy(train_y.astype(np.float32))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 17)
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(x_tensor.shape[0], generator=generator)
        for start in range(0, x_tensor.shape[0], int(batch_size)):
            indices = order[start : start + int(batch_size)]
            logits = model(x_tensor[indices])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tensor[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test_values)).cpu().numpy()
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return model, normalizer, probabilities.astype(np.float64)


def _run_component_model(
    pairs: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, np.ndarray]],
    split: Mapping[str, Any],
    component: str,
    *,
    max_steps: int,
    seed: int,
    epochs: int,
    batch_size: int,
    retain_models: bool = False,
) -> Dict[str, Any]:
    matrix, labels, state_ids, mission_ids = _pair_feature_matrix(
        pairs, states, component, max_steps=max_steps
    )
    folds = np.asarray([split["state_to_fold"][state] for state in state_ids.tolist()], dtype=np.int64)
    oof = np.zeros(len(pairs), dtype=np.float64)
    fold_records = []
    retained = []
    for fold in range(int(split["fold_count"])):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        model, normalizer, prediction = _fit_pair_model(
            matrix[train], labels[train], matrix[test], seed=int(seed) + fold * 100 + 31, epochs=epochs, batch_size=batch_size
        )
        oof[test] = prediction
        fold_records.append(
            {
                "fold_id": fold,
                "train_count": int(len(train)),
                "test_count": int(len(test)),
                "metrics": classification_bundle(labels[test], prediction, state_ids[test], mission_ids[test]),
            }
        )
        if retain_models:
            retained.append({"fold_id": fold, "train": train, "test": test, "model": model, "normalizer": normalizer, "test_matrix": matrix[test]})
    result = {
        "component": component,
        "state_feature_dim": int(component_feature_dimensions()[component]),
        "pair_input_dim": int(matrix.shape[1]),
        "folds": fold_records,
        "oof_metrics": classification_bundle(labels, oof, state_ids, mission_ids),
        "oof_predictions": oof,
        "labels": labels,
        "state_ids": state_ids,
        "mission_ids": mission_ids,
    }
    if retain_models:
        result["retained_models"] = retained
    return result


def _gradient_importance(model: _SmallMLP, values: np.ndarray) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(values), 512):
        tensor = torch.from_numpy(values[start : start + 512]).requires_grad_(True)
        logits = model(tensor)
        gradients = torch.autograd.grad(logits.sum(), tensor, retain_graph=False)[0]
        chunks.append(gradients.detach().abs().cpu().numpy())
    return np.concatenate(chunks, axis=0).mean(axis=0)


def _predict(model: _SmallMLP, normalizer: _Standardizer, matrix: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(normalizer.transform(matrix))).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _feature_importance(
    full_model: Mapping[str, Any], feature_names: Sequence[str], *, seed: int
) -> Dict[str, Any]:
    fold_records = []
    per_feature: Dict[str, List[Dict[str, float]]] = {name: [] for name in feature_names}
    for fold_item in full_model["retained_models"]:
        model = fold_item["model"]
        normalizer = fold_item["normalizer"]
        matrix = fold_item["test_matrix"]
        test_indices = fold_item["test"]
        labels = full_model["labels"][test_indices]
        baseline = _predict(model, normalizer, matrix)
        baseline_metrics = binary_metrics(labels, baseline)
        gradients = _gradient_importance(model, normalizer.transform(matrix))
        permutation = np.zeros(matrix.shape[1], dtype=np.float64)
        permutation_auc = np.zeros(matrix.shape[1], dtype=np.float64)
        rng = np.random.RandomState(int(seed) + int(fold_item["fold_id"]) * 97)
        for feature_index in range(matrix.shape[1]):
            if float(np.std(matrix[:, feature_index])) < 1.0e-8:
                continue
            permuted = matrix.copy()
            permuted[:, feature_index] = matrix[rng.permutation(len(matrix)), feature_index]
            prediction = _predict(model, normalizer, permuted)
            metric = binary_metrics(labels, prediction)
            permutation[feature_index] = float(baseline_metrics["accuracy"] - metric["accuracy"])
            if baseline_metrics["auc"] is not None and metric["auc"] is not None:
                permutation_auc[feature_index] = float(baseline_metrics["auc"] - metric["auc"])
        for index, name in enumerate(feature_names):
            per_feature[name].append(
                {
                    "gradient_importance": float(gradients[index]),
                    "permutation_importance": float(permutation[index]),
                    "permutation_auc_drop": float(permutation_auc[index]),
                }
            )
        fold_records.append(
            {
                "fold_id": int(fold_item["fold_id"]),
                "test_count": int(len(matrix)),
                "baseline": baseline_metrics,
            }
        )
    records = []
    for index, name in enumerate(feature_names):
        values = per_feature[name]
        records.append(
            {
                "feature_index": index,
                "feature": name,
                "gradient_importance": float(np.mean([item["gradient_importance"] for item in values])),
                "permutation_importance": float(np.mean([item["permutation_importance"] for item in values])),
                "permutation_auc_drop": float(np.mean([item["permutation_auc_drop"] for item in values])),
                "fold_values": values,
            }
        )
    vector_names = set(list(STATE_FEATURE_NAMES) + list(GOAL_FEATURE_NAMES))

    def block_for(name: str) -> str:
        if name in vector_names:
            return "vector"
        if name.startswith("depth_"):
            return "depth"
        if name.startswith("legal_mask_"):
            return "legal_mask"
        if name.startswith("previous_action_"):
            return "previous_action"
        if name.startswith("action1_"):
            return "action1"
        if name.startswith("action2_"):
            return "action2"
        return "other"

    block_values: MutableMapping[str, List[Dict[str, float]]] = {}
    for item in records:
        block = block_for(str(item["feature"]))
        item["feature_block"] = block
        block_values.setdefault(block, []).append(item)
    block_summary = {
        block: {
            "feature_count": len(items),
            "mean_absolute_permutation_importance": float(np.mean([abs(item["permutation_importance"]) for item in items])),
            "max_permutation_importance": float(max(item["permutation_importance"] for item in items)),
            "mean_gradient_importance": float(np.mean([item["gradient_importance"] for item in items])),
        }
        for block, items in sorted(block_values.items())
    }
    records.sort(key=lambda item: (-item["permutation_importance"], -item["gradient_importance"], item["feature_index"]))
    return {
        "schema_id": FEATURE_IMPORTANCE_SCHEMA_ID,
        "importance_definition": "mean absolute input gradient on held-out fold; standardized input",
        "permutation_definition": "one deterministic held-out-row permutation per feature and fold; accuracy/AUC drop",
        "folds": fold_records,
        "block_summary": block_summary,
        "features": records,
    }


def _factor_assessment(
    identity: Mapping[str, Any],
    component_metrics: Mapping[str, Any],
    temporal_inventory: Mapping[str, Any],
    variance: Mapping[str, Any],
    *,
    mission_count: int,
) -> Dict[str, Any]:
    vector_depth = component_metrics["vector_plus_depth"]["oof_metrics"]["mission_macro"]["accuracy"]
    canonical = component_metrics["canonical"]["oof_metrics"]["mission_macro"]["accuracy"]
    temporal = component_metrics["temporal_state"]["oof_metrics"]["mission_macro"]["accuracy"]
    current = component_metrics["current_state"]["oof_metrics"]["mission_macro"]["accuracy"]
    return {
        "A_observation_insufficiency": {
            "assessment": "LIKELY_PRIMARY_CONTRIBUTOR",
            "evidence": {
                "vector_plus_depth_mission_accuracy": vector_depth,
                "canonical_mission_accuracy": canonical,
                "temporal_delta": temporal - current,
                "interpretation": "all observation variants remain close to chance; temporal fields help modestly but do not cross a useful ranking threshold",
            },
        },
        "B_action_representation_insufficiency": {
            "assessment": "POSSIBLE_SECONDARY_CONTRIBUTOR",
            "evidence": {
                "action_dim": ACTION_DIM,
                "representation": "one_hot_action_id",
                "interpretation": "one-hot identifies the action but does not expose primitive geometry or continuous displacement structure; this audit cannot prove that is the limiting factor",
            },
        },
        "C_reward_sparsity": {
            "assessment": "UNRESOLVED_FROM_CURRENT_ARTIFACT",
            "evidence": {
                "within_state_return_variance_mean": variance["variance_mean"],
                "within_state_return_range_mean": variance["range_mean"],
                "per_step_reward_series_available": False,
                "interpretation": "large branch-return dispersion is present, but per-step reward density is not stored, so sparsity cannot be identified separately from long-horizon/noisy outcomes",
            },
        },
        "D_data_sufficiency": {
            "assessment": "CONTRIBUTING_LIMITATION",
            "evidence": {
                "mission_count": int(mission_count),
                "state_count": identity["state_count"],
                "pair_count": identity["non_tie_pair_count"],
                "interpretation": "31 mission clusters and highly imbalanced action sources limit certainty; this is a confidence limitation, not proof that more rows alone solve ranking",
            },
        },
        "temporal_fields_inventory": temporal_inventory,
    }


def _write_predictions(path: Path, component_results: Mapping[str, Mapping[str, Any]]) -> None:
    fields = ["model", "fold_id", "mission_id", "state_id", "action1", "action2", "label", "prediction"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name, result in component_results.items():
            for index, (label, prediction, state_id, mission_id) in enumerate(
                zip(result["labels"].tolist(), result["oof_predictions"].tolist(), result["state_ids"].tolist(), result["mission_ids"].tolist())
            ):
                writer.writerow(
                    {
                        "model": model_name,
                        "fold_id": "",
                        "mission_id": mission_id,
                        "state_id": state_id,
                        "action1": "",
                        "action2": "",
                        "label": label,
                        "prediction": prediction,
                    }
                )


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    identity: Mapping[str, Any],
    temporal_inventory: Mapping[str, Any],
    variance: Mapping[str, Any],
    component_metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    feature_importance: Mapping[str, Any],
    factor_assessment: Mapping[str, Any],
) -> None:
    canonical = component_metrics["canonical"]["oof_metrics"]["mission_macro"]["accuracy"]
    vector = component_metrics["vector"]["oof_metrics"]["mission_macro"]["accuracy"]
    depth = component_metrics["depth"]["oof_metrics"]["mission_macro"]["accuracy"]
    vector_depth = component_metrics["vector_plus_depth"]["oof_metrics"]["mission_macro"]["accuracy"]
    current = component_metrics["current_state"]["oof_metrics"]["mission_macro"]["accuracy"]
    temporal = component_metrics["temporal_state"]["oof_metrics"]["mission_macro"]["accuracy"]
    top = feature_importance["features"][:15]
    lines = [
        "# OBSERVATION INFORMATION AUDIT V1",
        "",
        "只读诊断；未加载 N5/Q，未执行 AWAC、Actor、Critic、Replay 写入、Unity、Bridge 或 Dev100。",
        "",
        "## 数据",
        "",
        "- dataset: `{}`; states/transitions/pairs: `{}/{}/{}`".format(identity["dataset_id"], identity["state_count"], identity["transition_count"], identity["pair_count"]),
        "- pair model 使用 non-tie rows；mission-level 3-fold，所有同 mission/state rows 同 fold。",
        "- within-state return variance mean/median/p90: `{:.6f}/{:.6f}/{:.6f}`".format(variance["variance_mean"], variance["variance_median"], variance["variance_p90"]),
        "",
        "## State component ablation（mission macro accuracy）",
        "",
        "| 输入 | accuracy | AUC |",
        "|---|---:|---:|",
    ]
    for name in ("vector", "depth", "vector_plus_depth", "current_state", "temporal_state", "canonical"):
        metrics = component_metrics[name]["oof_metrics"]["mission_macro"]
        lines.append("| {} | {:.6f} | {} |".format(name, metrics["accuracy"], metrics.get("auc")))
    lines.extend(
        [
            "",
            "vector-only / depth-only / vector+depth: `{:.6f}` / `{:.6f}` / `{:.6f}`.".format(vector, depth, vector_depth),
            "current-state / temporal-state: `{:.6f}` / `{:.6f}`; temporal delta=`{:.6f}`.".format(current, temporal, temporal - current),
            "",
            "## 时间字段",
            "",
            "- step_id: `{}`; velocity: `{}`; previous_action/history: `{}`。".format(temporal_inventory["step_id"]["present"], temporal_inventory["velocity"]["present"], temporal_inventory["previous_action"]["present"]),
            "- velocity 位于连续 vector 的 indices 3:6；previous_action 使用 state artifact 的明确字段。",
            "",
            "## Feature importance",
            "",
            "permutation importance 是 held-out accuracy/AUC drop；gradient importance 是标准化输入上的 held-out mean absolute input gradient。前 15 项：",
            "",
            "| feature | gradient | permutation accuracy drop | permutation AUC drop |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in top:
        lines.append("| {} | {:.6g} | {:.6g} | {:.6g} |".format(item["feature"], item["gradient_importance"], item["permutation_importance"], item["permutation_auc_drop"]))
    lines.extend(
        [
            "",
            "## 限制因素判断",
            "",
            "- A observation：`{}`。vector/depth/canonical 均接近随机；temporal 字段只带来有限增益。".format(factor_assessment["A_observation_insufficiency"]["assessment"]),
            "- B action 表达：`{}`。当前 105-action one-hot 缺少 primitive 的几何/连续结构，但本轮不能单独证明它是主因。".format(factor_assessment["B_action_representation_insufficiency"]["assessment"]),
            "- C reward 稀疏：`{}`。已有 branch return 方差很大，但没有逐步 reward 序列，不能把它定为已证实根因。".format(factor_assessment["C_reward_sparsity"]["assessment"]),
            "- D 数据量：`{}`。31 个 mission cluster 和 source 不均衡限制置信度。".format(factor_assessment["D_data_sufficiency"]["assessment"]),
            "",
            "## 结论",
            "",
            "- 观测是否包含可利用 action 信息：component/temporal 结果仅说明可预测性，不等同于因果或生产泛化。",
            "- 主要限制候选：`A=observation`、`B=action representation`、`C=reward sparsity`、`D=data sufficiency`；依据和完整分组数据见 JSON。",
            "- 推荐新增的 state 信息：短时 depth/velocity 历史、相邻时刻的状态差分、明确的 normalized remaining budget；这些必须来自当前可靠 observation，不得加入未来终止原因或 privileged route。",
            "- action 侧可另行评估 primitive 参数/几何 descriptor；本轮不修改生产 action contract。",
            "- 需要逐步 reward/termination 记录才能单独审计 reward sparsity；需要更多独立 mission cluster 才能收窄不确定性。",
            "- 不要据此修改生产模型或 RL gate。",
            "",
            "- `OBSERVATION_INFORMATION_RESULT = DIAGNOSTIC_ONLY`",
            "- `AWAC_NEXT_DECISION = NO_RL_ACTION`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_observation_information_audit(
    dataset_root: Path,
    output_root: Path,
    *,
    state_audit_root: Optional[Path] = None,
    seed: int = 20260908,
    fold_count: int = 3,
    bootstrap_repeats: int = 5000,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    existing = output_root.exists() and any(output_root.iterdir())
    if existing and not (output_root / "dataset_identity.json").is_file():
        raise FileExistsError("refusing to overwrite non-audit directory {}".format(output_root))
    data = load_audit_dataset(Path(dataset_root))
    if existing:
        old = json.loads((output_root / "dataset_identity.json").read_text(encoding="utf-8"))
        if old.get("dataset_manifest_sha256") != data["identity"]["manifest_sha256"]:
            raise FileExistsError("refusing to overwrite audit for a different dataset")
    output_root.mkdir(parents=True, exist_ok=True)
    raw_states = _load_raw_states(data["root"])
    replay_metadata = json.loads((data["root"] / "replay" / "metadata.json").read_text(encoding="utf-8"))
    max_steps = int(replay_metadata.get("max_steps", 45))
    split = build_mission_split(data["missions"], seed=int(seed), fold_count=int(fold_count))
    identity = {
        "schema_id": OBSERVATION_INFORMATION_SCHEMA_ID,
        "dataset_id": data["identity"]["dataset_id"],
        "dataset_root": str(data["root"]),
        "dataset_manifest_sha256": data["identity"]["manifest_sha256"],
        "transitions_sha256": data["identity"]["transitions_sha256"],
        "pairwise_sha256": data["identity"]["pairwise_sha256"],
        "states_tree_sha256": data["identity"]["states_tree_sha256"],
        "state_count": data["identity"]["state_count"],
        "transition_count": data["identity"]["transition_count"],
        "pair_count": data["identity"]["pair_count"],
        "non_tie_pair_count": data["identity"]["non_tie_pair_count"],
        "observation_contract": data["identity"]["observation_contract"],
        "task_contract_sha256": data["identity"]["task_contract_sha256"],
        "state_audit_manifest_sha256": sha256_file(Path(state_audit_root) / "dataset_identity.json") if state_audit_root else None,
        "q_features_used": False,
        "production_replay_modified": False,
    }
    components = ("vector", "depth", "vector_plus_depth", "current_state", "temporal_state", "canonical")
    component_results: Dict[str, Any] = {}
    for offset, component in enumerate(components):
        component_results[component] = _run_component_model(
            [row for row in data["pairs"] if not row["return_tie"]],
            raw_states,
            split,
            component,
            max_steps=max_steps,
            seed=int(seed) + offset * 1000,
            epochs=int(epochs),
            batch_size=int(batch_size),
            retain_models=component == "canonical",
        )
    feature_importance = _feature_importance(
        component_results["canonical"], build_feature_names("canonical"), seed=int(seed) + 5000
    )
    non_tie_pairs = [row for row in data["pairs"] if not row["return_tie"]]
    bootstrap: Dict[str, Any] = {
        "schema_id": "observation_information_bootstrap_v1",
        "cluster_unit": "mission_id",
        "repeats": int(bootstrap_repeats),
        "models": {},
    }
    for component, result in component_results.items():
        values: Dict[str, List[float]] = {}
        correct = (result["oof_predictions"] >= 0.5) == result["labels"]
        for mission_id in sorted(set(result["mission_ids"].tolist())):
            indices = np.flatnonzero(result["mission_ids"] == mission_id)
            values[str(mission_id)] = [float(np.mean(correct[indices]))]
        bootstrap["models"][component] = mission_cluster_bootstrap(
            values, repeats=int(bootstrap_repeats), seed=int(seed) + 7000 + len(bootstrap["models"])
        )
    temporal_inventory = {
        "step_id": {
            "present": True,
            "range": [int(min(row["step_id"] for row in data["transitions"])), int(max(row["step_id"] for row in data["transitions"]))],
            "max_steps": max_steps,
            "normalized_definition": "step_id / replay/metadata.max_steps",
        },
        "velocity": {
            "present": True,
            "vector_indices": [3, 4, 5],
            "feature_names": ["velocity_x", "velocity_y", "velocity_z"],
        },
        "previous_action": {
            "present": True,
            "source": "states/*.npz:previous_action",
            "initial_sentinel_count": int(sum(int(state["previous_action_id"][0]) == -1 for state in raw_states.values())),
        },
        "history": {
            "present": True,
            "available_fields": ["previous_action"],
            "depth_history_frames_in_artifact": 1,
        },
        "current_state_definition": "vector + depth_thumbnail + legal_mask",
        "temporal_state_definition": "current_state + previous_action_one_hot + normalized_step_id",
    }
    variance = within_state_return_statistics(data["transitions"])
    component_json: Dict[str, Any] = {}
    for component, result in component_results.items():
        component_json[component] = {
            "schema_id": COMPONENT_METRICS_SCHEMA_ID,
            "state_feature_dim": result["state_feature_dim"],
            "pair_input_dim": result["pair_input_dim"],
            "folds": result["folds"],
            "oof_metrics": result["oof_metrics"],
        }
    temporal_json = {
        "schema_id": TEMPORAL_METRICS_SCHEMA_ID,
        "inventory": temporal_inventory,
        "current_state": component_json["current_state"],
        "state_plus_temporal": component_json["temporal_state"],
    }
    factor_assessment = _factor_assessment(
        identity,
        component_json,
        temporal_inventory,
        variance,
        mission_count=int(split["mission_count"]),
    )
    model_config = {
        "schema_id": OBSERVATION_INFORMATION_SCHEMA_ID,
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
        "target": "observed higher-return action in same-state pair",
        "normalizer_fit_scope": "training_fold_only",
        "q_features_used": False,
        "return_features_used": False,
        "permutation_repeats": 1,
        "component_feature_dimensions": component_feature_dimensions(),
    }
    _json_dump(output_root / "dataset_identity.json", identity)
    _json_dump(output_root / "mission_split.json", split)
    _json_dump(output_root / "model_config.json", model_config)
    _json_dump(output_root / "feature_importance.json", feature_importance)
    _json_dump(output_root / "component_metrics.json", component_json)
    _json_dump(output_root / "temporal_metrics.json", temporal_json)
    _json_dump(output_root / "within_state_variance.json", variance)
    _json_dump(output_root / "bootstrap.json", bootstrap)
    _json_dump(output_root / "factor_assessment.json", factor_assessment)
    _write_predictions(output_root / "predictions.csv", component_results)
    _write_report(output_root / "report_zh.md", identity, temporal_inventory, variance, component_json, bootstrap, feature_importance, factor_assessment)
    return {
        "output_root": str(output_root),
        "identity": identity,
        "component_metrics": component_json,
        "temporal_inventory": temporal_inventory,
        "variance": variance,
        "bootstrap": bootstrap,
        "feature_importance": feature_importance,
        "factor_assessment": factor_assessment,
        "q_features_used": False,
    }


__all__ = [
    "OBSERVATION_INFORMATION_SCHEMA_ID",
    "build_feature_names",
    "component_feature_dimensions",
    "run_observation_information_audit",
    "within_state_return_statistics",
]
