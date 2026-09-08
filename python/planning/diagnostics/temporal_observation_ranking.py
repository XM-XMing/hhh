"""Offline temporal-observation ranking audit for the diagnostic replay.

The module is intentionally outside the production AWAC path.  It reuses the
immutable multi-action pair labels and the already published mission split,
then trains the same small CPU ranking MLP for four causal observation
variants.  No Q checkpoint, optimizer state, replay writer, or runtime client
is used here.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from planning.diagnostics.observation_information_audit import _load_raw_states
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


TEMPORAL_OBSERVATION_RANKING_SCHEMA_ID = "temporal_observation_ranking_audit_v1"
TEMPORAL_VARIANT_METRICS_SCHEMA_ID = "temporal_variant_metrics_v1"
TEMPORAL_BOOTSTRAP_SCHEMA_ID = "temporal_observation_bootstrap_v1"
TEMPORAL_FEATURE_ABLATION_SCHEMA_ID = "temporal_feature_ablation_v1"
TEMPORAL_VARIANTS = ("S0", "S1", "S2", "S3")
BASE_STATE_DIM = 22 + 72 + ACTION_DIM + ACTION_DIM
HISTORY_BASE_DIM = 22 + 72 + ACTION_DIM
HISTORY_LENGTH_BY_VARIANT = {"S0": 0, "S1": 3, "S2": 5, "S3": 5}
EXPECTED_PAIR_COUNT = 7123


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def temporal_variant_dimensions() -> Dict[str, Dict[str, int]]:
    """Return the fixed state and pair dimensions for S0 through S3."""

    return {
        "S0": {"state_feature_dim": BASE_STATE_DIM, "pair_input_dim": BASE_STATE_DIM + 2 * ACTION_DIM},
        "S1": {
            "state_feature_dim": BASE_STATE_DIM + 3 * HISTORY_BASE_DIM,
            "pair_input_dim": BASE_STATE_DIM + 3 * HISTORY_BASE_DIM + 2 * ACTION_DIM,
        },
        "S2": {
            "state_feature_dim": BASE_STATE_DIM + 5 * HISTORY_BASE_DIM,
            "pair_input_dim": BASE_STATE_DIM + 5 * HISTORY_BASE_DIM + 2 * ACTION_DIM,
        },
        "S3": {
            "state_feature_dim": BASE_STATE_DIM + 5 * HISTORY_BASE_DIM + 5 * ACTION_DIM,
            "pair_input_dim": BASE_STATE_DIM + 5 * HISTORY_BASE_DIM + 5 * ACTION_DIM + 2 * ACTION_DIM,
        },
    }


def build_history_index(
    manifest_rows: Sequence[Mapping[str, Any]], *, history_length: int = 5
) -> Tuple[Dict[str, Tuple[Optional[str], ...]], Dict[str, Any]]:
    """Index only earlier states from the same mission/source episode.

    A missing earlier step is represented by ``None`` and later converted to a
    zero block.  This is causal padding at episode boundaries; no state with a
    larger ``step_id`` can enter a current state's history.
    """

    if int(history_length) <= 0:
        raise ValueError("history_length must be positive")
    state_rows: Dict[str, Mapping[str, Any]] = {}
    groups: MutableMapping[Tuple[str, str], Dict[int, str]] = defaultdict(dict)
    for row in manifest_rows:
        state_id = str(row.get("state_id", ""))
        mission_id = str(row.get("mission_id", ""))
        source_episode_id = str(row.get("source_episode_id", ""))
        if not state_id or not mission_id or not source_episode_id:
            raise ValueError("state manifest row lacks identity")
        if state_id in state_rows:
            raise ValueError("duplicate state_id {}".format(state_id))
        step_id = int(row["step_id"])
        if step_id < 0:
            raise ValueError("negative step_id for {}".format(state_id))
        group = groups[(mission_id, source_episode_id)]
        if step_id in group:
            raise ValueError("duplicate mission/source/step {} {} {}".format(mission_id, source_episode_id, step_id))
        group[step_id] = state_id
        state_rows[state_id] = row

    history: Dict[str, Tuple[Optional[str], ...]] = {}
    available_by_slot = [0] * int(history_length)
    missing_by_slot = [0] * int(history_length)
    full_history_count = 0
    available_history_count = 0
    for state_id, row in state_rows.items():
        group = groups[(str(row["mission_id"]), str(row["source_episode_id"]))]
        step_id = int(row["step_id"])
        slots: List[Optional[str]] = []
        for offset in range(1, int(history_length) + 1):
            previous = group.get(step_id - offset)
            slots.append(previous)
            if previous is None:
                missing_by_slot[offset - 1] += 1
            else:
                available_by_slot[offset - 1] += 1
        history[state_id] = tuple(slots)
        available = sum(item is not None for item in slots)
        if available:
            available_history_count += 1
        if available == int(history_length):
            full_history_count += 1

    return history, {
        "state_count": len(state_rows),
        "history_length": int(history_length),
        "available_history_count": int(available_history_count),
        "no_history_count": int(len(state_rows) - available_history_count),
        "full_history_count": int(full_history_count),
        "partial_history_count": int(available_history_count - full_history_count),
        "available_slot_counts": available_by_slot,
        "missing_slot_counts": missing_by_slot,
        "padding": "zero_for_missing_earlier_step_only",
        "causal_rule": "same mission_id and source_episode_id; step_id - offset",
    }


def _base_current_feature(state: Mapping[str, np.ndarray]) -> np.ndarray:
    parts = [state["vector"], state["depth"], state["mask"], state["previous_action"]]
    feature = np.concatenate([np.asarray(item, dtype=np.float32).reshape(-1) for item in parts]).astype(np.float32)
    if feature.size != BASE_STATE_DIM or not np.isfinite(feature).all():
        raise ValueError("current state feature has invalid shape or values")
    return feature


def _base_history_feature(state: Mapping[str, np.ndarray]) -> np.ndarray:
    parts = [state["vector"], state["depth"], state["mask"]]
    feature = np.concatenate([np.asarray(item, dtype=np.float32).reshape(-1) for item in parts]).astype(np.float32)
    if feature.size != HISTORY_BASE_DIM or not np.isfinite(feature).all():
        raise ValueError("history state feature has invalid shape or values")
    return feature


def _variant_state_feature(
    state_id: str,
    states: Mapping[str, Mapping[str, np.ndarray]],
    history: Mapping[str, Sequence[Optional[str]]],
    variant: str,
) -> np.ndarray:
    if variant not in TEMPORAL_VARIANTS:
        raise ValueError("unknown temporal variant {}".format(variant))
    if state_id not in states or state_id not in history:
        raise ValueError("state {} is not indexed".format(state_id))
    parts: List[np.ndarray] = [_base_current_feature(states[state_id])]
    history_length = HISTORY_LENGTH_BY_VARIANT[variant]
    slots = list(history[state_id])[:history_length]
    for state_ref in slots:
        parts.append(np.zeros(HISTORY_BASE_DIM, dtype=np.float32) if state_ref is None else _base_history_feature(states[state_ref]))
    if variant == "S3":
        for state_ref in slots:
            parts.append(
                np.zeros(ACTION_DIM, dtype=np.float32)
                if state_ref is None
                else np.asarray(states[state_ref]["previous_action"], dtype=np.float32).reshape(-1)
            )
    feature = np.concatenate(parts).astype(np.float32)
    expected = temporal_variant_dimensions()[variant]["state_feature_dim"]
    if feature.size != expected or not np.isfinite(feature).all():
        raise ValueError("{} feature has shape {} expected {}".format(variant, feature.size, expected))
    return feature


def build_pair_feature_matrix(
    pairs: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, np.ndarray]],
    history: Mapping[str, Sequence[Optional[str]]],
    variant: str,
) -> np.ndarray:
    """Build pair inputs without changing pair ordering or labels."""

    state_features = [
        _variant_state_feature(str(row["state_id"]), states, history, variant) for row in pairs
    ]
    action1 = np.asarray([int(row["action1"]) for row in pairs], dtype=np.int64)
    action2 = np.asarray([int(row["action2"]) for row in pairs], dtype=np.int64)
    if np.any(action1 < 0) or np.any(action1 >= ACTION_DIM) or np.any(action2 < 0) or np.any(action2 >= ACTION_DIM):
        raise ValueError("pair action is outside action space")
    matrix = np.concatenate(
        (np.asarray(state_features, dtype=np.float32), _one_hot(action1), _one_hot(action2)), axis=1
    )
    expected = temporal_variant_dimensions()[variant]["pair_input_dim"]
    if matrix.shape != (len(pairs), expected):
        raise ValueError("{} pair matrix shape {} expected {}".format(variant, matrix.shape, (len(pairs), expected)))
    return matrix


def _fit_pair_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> Tuple[_Standardizer, np.ndarray]:
    normalizer = _Standardizer.fit(train_x)
    train_values = normalizer.transform(train_x)
    test_values = normalizer.transform(test_x)
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    model = _SmallMLP(train_values.shape[1], DEFAULT_HIDDEN_DIM, DEFAULT_HIDDEN_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x_tensor = torch.from_numpy(train_values)
    y_tensor = torch.from_numpy(np.asarray(train_y, dtype=np.float32))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 17)
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(x_tensor.shape[0], generator=generator)
        for start in range(0, x_tensor.shape[0], int(batch_size)):
            indices = order[start : start + int(batch_size)]
            logits = model(x_tensor[indices])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_tensor[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test_values)).cpu().numpy()
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return normalizer, probabilities.astype(np.float64)


def _pairwise_bootstrap(values: Mapping[str, Sequence[float]], *, repeats: int, seed: int) -> Dict[str, Any]:
    result = mission_cluster_bootstrap(values, repeats=int(repeats), seed=int(seed))
    return {key: value for key, value in result.items() if key != "schema_id"}


def _load_and_validate_split(
    state_audit_root: Path,
    observation_audit_root: Path,
    data_identity: Mapping[str, Any],
    *,
    fold_count: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    state_audit_root = Path(state_audit_root).expanduser().resolve()
    observation_audit_root = Path(observation_audit_root).expanduser().resolve()
    state_identity_path = state_audit_root / "dataset_identity.json"
    state_split_path = state_audit_root / "mission_split.json"
    observation_identity_path = observation_audit_root / "dataset_identity.json"
    observation_split_path = observation_audit_root / "mission_split.json"
    for path in (state_identity_path, state_split_path, observation_identity_path, observation_split_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    state_identity = json.loads(state_identity_path.read_text(encoding="utf-8"))
    observation_identity = json.loads(observation_identity_path.read_text(encoding="utf-8"))
    split = json.loads(state_split_path.read_text(encoding="utf-8"))
    if state_identity.get("pairwise_sha256") != data_identity.get("pairwise_sha256"):
        raise ValueError("state audit pairwise identity does not match input")
    if observation_identity.get("pairwise_sha256") != data_identity.get("pairwise_sha256"):
        raise ValueError("observation audit pairwise identity does not match input")
    if state_identity.get("dataset_id") != data_identity.get("dataset_id"):
        raise ValueError("state audit dataset identity does not match input")
    if observation_identity.get("dataset_id") != data_identity.get("dataset_id"):
        raise ValueError("observation audit dataset identity does not match input")
    if int(split.get("fold_count", -1)) != int(fold_count):
        raise ValueError("state audit split fold count is not the requested fixed split")
    if set(split.get("state_to_fold", {})) != set(data_identity.get("state_ids", [])):
        # Older identity artifacts do not list state IDs; the caller checks the
        # complete mapping against the loaded state set after this function.
        if int(split.get("state_count", -1)) != int(data_identity.get("state_count", -2)):
            raise ValueError("state audit split does not cover the input states")
    return split, {
        "state_audit_identity_sha256": sha256_file(state_identity_path),
        "state_audit_split_sha256": sha256_file(state_split_path),
        "observation_audit_identity_sha256": sha256_file(observation_identity_path),
        "observation_audit_split_sha256": sha256_file(observation_split_path),
        "observation_audit_split_used": False,
        "mission_split_source": str(state_split_path),
    }


def _write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "variant",
        "fold_id",
        "mission_id",
        "state_id",
        "state_hash",
        "step_id",
        "action1",
        "action2",
        "label",
        "prediction",
        "predicted_label",
        "history_available_slots",
        "history_full",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    *,
    identity: Mapping[str, Any],
    split: Mapping[str, Any],
    coverage: Mapping[str, Any],
    variant_metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    rows = [
        "# TEMPORAL OBSERVATION RANKING AUDIT V1",
        "",
        "本报告是只读 offline diagnostic；未执行 AWAC、Actor、Critic、Unity、Bridge 或 Replay 写入。",
        "",
        "## 固定输入与切分",
        "",
        "- dataset: `{}`；state / pair(non-tie) / mission: `{}` / `{}` / `{}`。".format(
            identity["dataset_id"], identity["state_count"], identity["pair_count"], split["mission_count"]
        ),
        "- pair labels、return 和顺序复用 state_sufficiency_audit_v1；没有重新随机切分。",
        "- folds: `{}`；mission-cluster bootstrap: `{}` repeats；normalizer 只使用训练 fold。".format(
            split["fold_count"], config["bootstrap_repeats"]
        ),
        "",
        "## 因果变体定义",
        "",
        "- S0 = 当前 production state：vector + depth 9x8 mean-pool + legal mask + current previous_action one-hot。",
        "- S1 = S0 + 最近 3 个同 mission/source episode 的较早 state 的 vector/depth/mask。",
        "- S2 = S0 + 最近 5 个同 mission/source episode 的较早 state 的 vector/depth/mask。",
        "- S3 = S2 + 最近 5 个较早 state 的 previous_action one-hot。",
        "- 历史按 `step_id - 1, step_id - 2, ...` 连接；episode 边界使用零填充；不使用 future state/reward/termination/route。",
        "",
        "## 历史覆盖",
        "",
        "- 5-slot history: available `{}` / `{}`；full `{}`；partial `{}`；zero-history `{}`。".format(
            coverage["available_history_count"], coverage["state_count"], coverage["full_history_count"], coverage["partial_history_count"], coverage["no_history_count"]
        ),
        "- 每个 slot 可用数量（从 immediate past 到 oldest）：`{}`；missing：`{}`。".format(
            coverage["available_slot_counts"], coverage["missing_slot_counts"]
        ),
        "",
        "## 三折 OOF 指标",
        "",
        "| variant | state dim | pair dim | micro accuracy | micro AUC | mission-macro accuracy | mission-macro AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in TEMPORAL_VARIANTS:
        item = variant_metrics["variants"][variant]
        micro = item["oof_metrics"]["micro"]
        mission = item["oof_metrics"]["mission_macro"]
        rows.append(
            "| {} | {} | {} | {:.6f} | {} | {:.6f} | {} |".format(
                variant,
                item["state_feature_dim"],
                item["pair_input_dim"],
                micro["accuracy"],
                "-" if micro["auc"] is None else "{:.6f}".format(micro["auc"]),
                mission["accuracy"],
                "-" if mission["auc"] is None else "{:.6f}".format(mission["auc"]),
            )
        )
    rows.extend(["", "## 相对 S0 的 mission-cluster paired bootstrap", "", "| comparison | estimate | 95% CI |", "|---|---:|---:|"])
    for comparison in ("S1_minus_S0", "S2_minus_S0", "S3_minus_S0"):
        item = bootstrap["paired_deltas"][comparison]
        rows.append("| {} | {:.6f} | [{:.6f}, {:.6f}] |".format(comparison, item["estimate"], item["ci95"][0], item["ci95"][1]))
    best_variant = max(TEMPORAL_VARIANTS, key=lambda value: variant_metrics["variants"][value]["oof_metrics"]["mission_macro"]["accuracy"])
    best_score = variant_metrics["variants"][best_variant]["oof_metrics"]["mission_macro"]["accuracy"]
    if any(
        variant_metrics["variants"][variant]["oof_metrics"]["mission_macro"]["accuracy"] > 0.65
        for variant in ("S2", "S3")
    ):
        decision = "S2/S3 超过 0.65；temporal observation 值得进入后续 Critic 设计候选，但本报告不放行任何 RL 训练。"
    else:
        decision = "S2/S3 未超过 0.65；当前证据不足以支持 temporal observation 进入 Critic，继续检查 state/action formulation。"
    rows.extend(
        [
            "",
            "## 结论",
            "",
            "- 最佳 mission-macro variant: `{}` = `{:.6f}`。".format(best_variant, best_score),
            "- TEMPORAL_OBSERVATION_DECISION = `{}`".format(decision),
            "- 该结果是同一 31 mission 数据集上的信息诊断，不是独立泛化集，也不构成生产 gate 或 AWAC 收敛结论。",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_temporal_observation_ranking_audit(
    dataset_root: Path,
    output_root: Path,
    *,
    state_audit_root: Path,
    observation_audit_root: Path,
    seed: int = 20260909,
    fold_count: int = 3,
    bootstrap_repeats: int = 5000,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Run the four-variant causal ranking audit and persist evidence."""

    if int(fold_count) != 3:
        raise ValueError("this audit requires the fixed three-fold mission split")
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty temporal audit directory {}".format(output_root))
    data = load_audit_dataset(Path(dataset_root))
    if int(data["identity"]["pair_count"]) != int(data["identity"]["non_tie_pair_count"]) + 1:
        raise ValueError("expected one explicit tie in the fixed pair artifact")
    pairs = [row for row in data["pairs"] if not row["return_tie"]]
    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise ValueError("expected {} non-tie pairs, got {}".format(EXPECTED_PAIR_COUNT, len(pairs)))
    raw_states = _load_raw_states(Path(dataset_root))
    manifest_rows = _read_jsonl(Path(dataset_root) / "state_manifest.jsonl")
    history, coverage = build_history_index(manifest_rows, history_length=5)
    if set(raw_states) != set(history):
        raise ValueError("state manifest/raw state identity mismatch")
    split, split_identity = _load_and_validate_split(
        Path(state_audit_root), Path(observation_audit_root), data["identity"], fold_count=int(fold_count)
    )
    if set(split["state_to_fold"]) != set(raw_states):
        raise ValueError("fixed mission split does not cover every input state")
    for row in manifest_rows:
        if str(row["state_id"]) not in split["state_to_fold"]:
            raise ValueError("split missing state {}".format(row["state_id"]))

    output_root.mkdir(parents=True, exist_ok=True)
    identity = dict(data["identity"])
    identity.update(
        {
            "schema_id": TEMPORAL_OBSERVATION_RANKING_SCHEMA_ID,
            "source_dataset_manifest_sha256": sha256_file(Path(dataset_root) / "dataset_manifest.json"),
            "state_manifest_sha256": sha256_file(Path(dataset_root) / "state_manifest.jsonl"),
            "state_audit_root": str(Path(state_audit_root).expanduser().resolve()),
            "observation_audit_root": str(Path(observation_audit_root).expanduser().resolve()),
            "non_tie_pair_count_used": len(pairs),
            "future_features_used": False,
            "q_features_used": False,
            "production_observation_modified": False,
        }
    )
    config = {
        "schema_id": "temporal_observation_ranking_model_config_v1",
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
        "loss": "binary_cross_entropy_with_logits",
        "normalizer_fit_scope": "training_fold_only",
        "n5_loaded": False,
        "q_features_used": False,
        "return_features_used": False,
        "variants": temporal_variant_dimensions(),
    }
    _json_dump(output_root / "dataset_identity.json", identity)
    _json_dump(output_root / "mission_split.json", split)
    _json_dump(output_root / "model_config.json", config)

    state_ids = np.asarray([str(row["state_id"]) for row in pairs])
    mission_ids = np.asarray([str(row["mission_id"]) for row in pairs])
    labels = np.asarray([int(row["label"]) for row in pairs], dtype=np.int64)
    steps = np.asarray([int(row["step_id"]) for row in pairs], dtype=np.int64)
    folds = np.asarray([int(split["state_to_fold"][state_id]) for state_id in state_ids.tolist()], dtype=np.int64)
    variant_records: Dict[str, Any] = {}
    prediction_rows: List[Dict[str, Any]] = []
    correctness_by_variant: Dict[str, np.ndarray] = {}
    for variant_index, variant in enumerate(TEMPORAL_VARIANTS):
        matrix = build_pair_feature_matrix(pairs, raw_states, history, variant)
        oof = np.zeros(len(pairs), dtype=np.float64)
        fold_records = []
        for fold in range(int(fold_count)):
            train = np.flatnonzero(folds != fold)
            test = np.flatnonzero(folds == fold)
            _, prediction = _fit_pair_model(
                matrix[train],
                labels[train],
                matrix[test],
                seed=int(seed) + variant_index * 10000 + fold * 100 + 41,
                epochs=int(epochs),
                batch_size=int(batch_size),
            )
            oof[test] = prediction
            fold_records.append(
                {
                    "fold_id": int(fold),
                    "train_count": int(len(train)),
                    "test_count": int(len(test)),
                    "metrics": classification_bundle(labels[test], prediction, state_ids[test], mission_ids[test]),
                }
            )
            for index, value in zip(test.tolist(), prediction.tolist()):
                row = pairs[index]
                slots = list(history[str(row["state_id"])])
                prediction_rows.append(
                    {
                        "variant": variant,
                        "fold_id": int(fold),
                        "mission_id": str(row["mission_id"]),
                        "state_id": str(row["state_id"]),
                        "state_hash": str(row["state_id"]),
                        "step_id": int(row["step_id"]),
                        "action1": int(row["action1"]),
                        "action2": int(row["action2"]),
                        "label": int(labels[index]),
                        "prediction": float(value),
                        "predicted_label": int(value >= 0.5),
                        "history_available_slots": int(sum(item is not None for item in slots[: HISTORY_LENGTH_BY_VARIANT[variant]])),
                        "history_full": bool(all(item is not None for item in slots[: HISTORY_LENGTH_BY_VARIANT[variant]])) if HISTORY_LENGTH_BY_VARIANT[variant] else True,
                    }
                )
        correctness = (oof >= 0.5) == labels
        correctness_by_variant[variant] = correctness
        metrics = classification_bundle(labels, oof, state_ids, mission_ids)
        variant_records[variant] = {
            "variant": variant,
            "state_feature_dim": temporal_variant_dimensions()[variant]["state_feature_dim"],
            "pair_input_dim": temporal_variant_dimensions()[variant]["pair_input_dim"],
            "pair_count": len(pairs),
            "folds": fold_records,
            "oof_metrics": metrics,
            "pairwise_accuracy": float(metrics["micro"]["accuracy"]),
            "pairwise_auc": metrics["micro"]["auc"],
            "mission_macro_accuracy": metrics["mission_macro"]["accuracy"],
            "mission_macro_auc": metrics["mission_macro"]["auc"],
        }

    variant_metrics = {
        "schema_id": TEMPORAL_VARIANT_METRICS_SCHEMA_ID,
        "dataset_identity": identity,
        "fixed_pair_order": True,
        "fixed_mission_split": True,
        "variants": variant_records,
    }
    absolute_bootstrap: Dict[str, Any] = {}
    for index, variant in enumerate(TEMPORAL_VARIANTS):
        by_mission: MutableMapping[str, List[float]] = defaultdict(list)
        for mission_id, value in zip(mission_ids.tolist(), correctness_by_variant[variant].tolist()):
            by_mission[str(mission_id)].append(float(value))
        absolute_bootstrap[variant] = _pairwise_bootstrap(
            by_mission, repeats=int(bootstrap_repeats), seed=int(seed) + 5000 + index
        )
    paired_deltas: Dict[str, Any] = {}
    baseline = correctness_by_variant["S0"]
    for index, variant in enumerate(("S1", "S2", "S3")):
        by_mission_delta: MutableMapping[str, List[float]] = defaultdict(list)
        deltas = correctness_by_variant[variant].astype(np.int8) - baseline.astype(np.int8)
        for mission_id, delta in zip(mission_ids.tolist(), deltas.tolist()):
            by_mission_delta[str(mission_id)].append(float(delta))
        paired_deltas["{}_minus_S0".format(variant)] = _pairwise_bootstrap(
            by_mission_delta, repeats=int(bootstrap_repeats), seed=int(seed) + 6000 + index
        )
    bootstrap = {
        "schema_id": TEMPORAL_BOOTSTRAP_SCHEMA_ID,
        "cluster_unit": "mission_id",
        "repeats": int(bootstrap_repeats),
        "absolute": absolute_bootstrap,
        "paired_deltas": paired_deltas,
        "note": "CI is mission-cluster only and does not include training-seed uncertainty.",
    }
    feature_ablation = {
        "schema_id": TEMPORAL_FEATURE_ABLATION_SCHEMA_ID,
        "variant_definitions": {
            "S0": "current vector + depth_9x8 + legal mask + current previous_action one-hot",
            "S1": "S0 + previous 3 base states (vector + depth_9x8 + legal mask)",
            "S2": "S0 + previous 5 base states (vector + depth_9x8 + legal mask)",
            "S3": "S2 + previous 5 previous_action one-hot blocks",
        },
        "dimensions": temporal_variant_dimensions(),
        "history_coverage": coverage,
        "causality": {
            "future_state_used": False,
            "future_reward_used": False,
            "termination_reason_used": False,
            "privileged_route_used": False,
            "history_join": "same mission_id/source_episode_id and strictly smaller step_id",
        },
        "same_pair_labels_and_returns": True,
        "same_mission_split": True,
    }
    _json_dump(output_root / "variant_metrics.json", variant_metrics)
    _json_dump(output_root / "bootstrap.json", bootstrap)
    _json_dump(output_root / "feature_ablation.json", feature_ablation)
    _write_predictions(output_root / "predictions.csv", prediction_rows)
    _write_report(
        output_root / "report_zh.md",
        identity=identity,
        split=split,
        coverage=coverage,
        variant_metrics=variant_metrics,
        bootstrap=bootstrap,
        config=config,
    )
    return {
        "output_root": str(output_root),
        "identity": identity,
        "split": split,
        "coverage": coverage,
        "config": config,
        "variant_metrics": variant_metrics,
        "bootstrap": bootstrap,
        "feature_ablation": feature_ablation,
        "split_identity": split_identity,
    }


__all__ = [
    "TEMPORAL_VARIANTS",
    "build_history_index",
    "build_pair_feature_matrix",
    "run_temporal_observation_ranking_audit",
    "temporal_variant_dimensions",
]
