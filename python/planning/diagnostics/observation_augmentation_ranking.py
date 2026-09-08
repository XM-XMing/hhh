"""Offline ranking audit for minimal non-privileged observation augmentations.

This module is diagnostic-only.  It reuses the immutable multi-action replay,
the fixed observation-information mission split, and the fixed non-tie pair
labels.  It never loads a Q checkpoint, writes replay, starts a runtime, or
changes the production observation contract.

The artifact does not contain timestamps.  Consequently the acceleration
augmentation is explicitly a per-decision velocity difference, not a claim of
physical m/s^2 acceleration.  Missing history at an episode boundary is zero
padded and recorded in the feature-delta artifact.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
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
    classification_bundle,
    load_audit_dataset,
    mission_cluster_bootstrap,
    sha256_file,
)


OBSERVATION_AUGMENTATION_RANKING_SCHEMA_ID = "observation_augmentation_ranking_audit_v1"
VARIANT_METRICS_SCHEMA_ID = "observation_augmentation_variant_metrics_v1"
BOOTSTRAP_SCHEMA_ID = "observation_augmentation_bootstrap_v1"
FEATURE_DELTA_SCHEMA_ID = "observation_augmentation_feature_delta_v1"
VARIANTS = ("S0", "S1", "S2", "S3", "S4")
EXPECTED_NON_TIE_PAIR_COUNT = 7123
MAX_STEPS = 45
BASE_STATE_DIM = 22 + 72 + ACTION_DIM + ACTION_DIM
ACTION_PAIR_DIM = 2 * ACTION_DIM


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("{} line {} is not an object".format(path, line_number))
            rows.append(value)
    return rows


def _stats(values: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("feature statistics require finite non-empty values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def variant_dimensions() -> Dict[str, Dict[str, int]]:
    """Return fixed dimensions for the five pre-registered variants."""

    return {
        "S0": {"state_feature_dim": BASE_STATE_DIM, "pair_input_dim": BASE_STATE_DIM + ACTION_PAIR_DIM},
        "S1": {"state_feature_dim": BASE_STATE_DIM + 3, "pair_input_dim": BASE_STATE_DIM + 3 + ACTION_PAIR_DIM},
        "S2": {"state_feature_dim": BASE_STATE_DIM + 72, "pair_input_dim": BASE_STATE_DIM + 72 + ACTION_PAIR_DIM},
        "S3": {"state_feature_dim": BASE_STATE_DIM + 3 + 72, "pair_input_dim": BASE_STATE_DIM + 3 + 72 + ACTION_PAIR_DIM},
        "S4": {"state_feature_dim": BASE_STATE_DIM + 3 + 72 + 1, "pair_input_dim": BASE_STATE_DIM + 3 + 72 + 1 + ACTION_PAIR_DIM},
    }


def build_previous_state_index(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    max_steps: int = MAX_STEPS,
) -> Tuple[Dict[str, Optional[str]], Dict[str, Any]]:
    """Join only the immediately preceding state of the same source episode."""

    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    by_group: MutableMapping[Tuple[str, str], Dict[int, str]] = defaultdict(dict)
    seen_state_ids = set()
    rows_by_id: Dict[str, Mapping[str, Any]] = {}
    for row in manifest_rows:
        state_id = str(row.get("state_id", ""))
        mission_id = str(row.get("mission_id", ""))
        source_episode_id = str(row.get("source_episode_id", ""))
        if not state_id or not mission_id or not source_episode_id:
            raise ValueError("state manifest row lacks identity")
        if state_id in seen_state_ids:
            raise ValueError("duplicate state_id {}".format(state_id))
        step_id = int(row["step_id"])
        if step_id < 0 or step_id >= int(max_steps):
            raise ValueError("step_id {} is outside [0, {})".format(step_id, max_steps))
        group = by_group[(mission_id, source_episode_id)]
        if step_id in group:
            raise ValueError("duplicate state step {} {} {}".format(mission_id, source_episode_id, step_id))
        group[step_id] = state_id
        seen_state_ids.add(state_id)
        rows_by_id[state_id] = row

    previous: Dict[str, Optional[str]] = {}
    available = 0
    for state_id, row in rows_by_id.items():
        group = by_group[(str(row["mission_id"]), str(row["source_episode_id"]))]
        prior = group.get(int(row["step_id"]) - 1)
        previous[state_id] = prior
        available += int(prior is not None)
    return previous, {
        "state_count": len(rows_by_id),
        "previous_state_available_count": int(available),
        "previous_state_missing_count": int(len(rows_by_id) - available),
        "history_join": "same mission_id and source_episode_id at step_id - 1",
        "history_is_past_only": True,
        "zero_padding_for_missing_previous_state": True,
        "max_steps": int(max_steps),
    }


def _base_feature(state: Mapping[str, np.ndarray]) -> np.ndarray:
    feature = np.concatenate(
        [
            np.asarray(state["vector"], dtype=np.float32).reshape(-1),
            np.asarray(state["depth"], dtype=np.float32).reshape(-1),
            np.asarray(state["mask"], dtype=np.float32).reshape(-1),
            np.asarray(state["previous_action"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)
    if feature.size != BASE_STATE_DIM or not np.isfinite(feature).all():
        raise ValueError("invalid base observation feature shape or value")
    return feature


def build_augmentation_features(
    states: Mapping[str, Mapping[str, np.ndarray]],
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    max_steps: int = MAX_STEPS,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """Build S0-S4 state blocks using current and strictly past artifacts."""

    previous, coverage = build_previous_state_index(manifest_rows, max_steps=max_steps)
    rows_by_id = {str(row["state_id"]): row for row in manifest_rows}
    if set(rows_by_id) != set(states):
        raise ValueError("state manifest/raw state identity mismatch")
    features: Dict[str, Dict[str, np.ndarray]] = {}
    acceleration_values: List[np.ndarray] = []
    depth_delta_values: List[np.ndarray] = []
    budget_values: List[float] = []
    for state_id in sorted(states):
        current = states[state_id]
        base = _base_feature(current)
        prior_id = previous[state_id]
        if prior_id is None:
            acceleration = np.zeros(3, dtype=np.float32)
            depth_delta = np.zeros(72, dtype=np.float32)
        else:
            prior = states[prior_id]
            acceleration = (
                np.asarray(current["vector"], dtype=np.float32).reshape(-1)[3:6]
                - np.asarray(prior["vector"], dtype=np.float32).reshape(-1)[3:6]
            ).astype(np.float32)
            depth_delta = (
                np.asarray(current["depth"], dtype=np.float32).reshape(-1)
                - np.asarray(prior["depth"], dtype=np.float32).reshape(-1)
            ).astype(np.float32)
        step_id = int(rows_by_id[state_id]["step_id"])
        budget = np.asarray([(float(max_steps) - float(step_id)) / float(max_steps)], dtype=np.float32)
        if not np.isfinite(acceleration).all() or not np.isfinite(depth_delta).all() or not np.isfinite(budget).all():
            raise ValueError("non-finite augmentation for {}".format(state_id))
        features[state_id] = {
            "S0": base,
            "S1": np.concatenate((base, acceleration)).astype(np.float32),
            "S2": np.concatenate((base, depth_delta)).astype(np.float32),
            "S3": np.concatenate((base, acceleration, depth_delta)).astype(np.float32),
            "S4": np.concatenate((base, acceleration, depth_delta, budget)).astype(np.float32),
            "acceleration": acceleration,
            "depth_temporal_difference": depth_delta,
            "normalized_remaining_budget": budget,
        }
        acceleration_values.append(acceleration)
        depth_delta_values.append(depth_delta)
        budget_values.append(float(budget[0]))

    coverage = dict(coverage)
    coverage["acceleration_unit"] = "velocity difference per decision step; timestamp unavailable"
    coverage["acceleration_physical_unit"] = "NOT_CLAIMED"
    coverage["max_step_id"] = int(max(int(row["step_id"]) for row in manifest_rows))
    coverage["acceleration_feature_statistics"] = {
        "velocity_x_difference": _stats(np.asarray(acceleration_values)[:, 0]),
        "velocity_y_difference": _stats(np.asarray(acceleration_values)[:, 1]),
        "velocity_z_difference": _stats(np.asarray(acceleration_values)[:, 2]),
    }
    coverage["depth_temporal_difference_statistics"] = {
        "all_pixels": _stats(np.asarray(depth_delta_values).reshape(-1)),
        "per_state_l2": _stats(np.linalg.norm(np.asarray(depth_delta_values), axis=1)),
    }
    coverage["normalized_remaining_budget_statistics"] = _stats(np.asarray(budget_values))
    return features, coverage


def build_pair_feature_matrix(
    pairs: Sequence[Mapping[str, Any]],
    features: Mapping[str, Mapping[str, np.ndarray]],
    variant: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the same ordered pair matrix for one observation variant."""

    if variant not in VARIANTS:
        raise ValueError("unknown observation variant {}".format(variant))
    state_blocks = []
    actions1 = []
    actions2 = []
    labels = []
    state_ids = []
    mission_ids = []
    for row in pairs:
        state_id = str(row["state_id"])
        if state_id not in features:
            raise ValueError("pair references unknown state {}".format(state_id))
        action1 = int(row["action1"])
        action2 = int(row["action2"])
        if not 0 <= action1 < ACTION_DIM or not 0 <= action2 < ACTION_DIM:
            raise ValueError("pair action outside action space")
        state_ids.append(state_id)
        mission_ids.append(str(row["mission_id"]))
        actions1.append(action1)
        actions2.append(action2)
        labels.append(int(row["higher_return_action"] == action1))
        state_blocks.append(features[state_id][variant])
    matrix = np.concatenate(
        (np.asarray(state_blocks, dtype=np.float32), _one_hot(np.asarray(actions1)), _one_hot(np.asarray(actions2))),
        axis=1,
    )
    expected = variant_dimensions()[variant]["pair_input_dim"]
    if matrix.shape != (len(pairs), expected) or not np.isfinite(matrix).all():
        raise ValueError("{} pair matrix shape/value mismatch: {}".format(variant, matrix.shape))
    return (
        matrix,
        np.asarray(labels, dtype=np.int64),
        np.asarray(state_ids),
        np.asarray(mission_ids),
    )


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
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).astype(np.float64)


def _bootstrap_absolute(
    correctness: np.ndarray,
    mission_ids: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    grouped: MutableMapping[str, List[float]] = defaultdict(list)
    for mission_id, value in zip(mission_ids.tolist(), correctness.tolist()):
        grouped[str(mission_id)].append(float(value))
    result = mission_cluster_bootstrap(grouped, repeats=int(repeats), seed=int(seed))
    result.pop("schema_id", None)
    return result


def _bootstrap_delta(
    delta: np.ndarray,
    mission_ids: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    grouped: MutableMapping[str, List[float]] = defaultdict(list)
    for mission_id, value in zip(mission_ids.tolist(), delta.tolist()):
        grouped[str(mission_id)].append(float(value))
    result = mission_cluster_bootstrap(grouped, repeats=int(repeats), seed=int(seed))
    result.pop("schema_id", None)
    return result


def _write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "variant",
        "fold_id",
        "mission_id",
        "state_id",
        "step_id",
        "action1",
        "action2",
        "label",
        "prediction",
        "predicted_label",
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
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> None:
    rows = [
        "# OBSERVATION AUGMENTATION RANKING AUDIT V1",
        "",
        "只读 offline diagnostic；未训练 AWAC/Actor/Critic，未启动 Unity/Bridge/ROS，未修改 production observation 或 Replay。",
        "",
        "## 固定输入",
        "",
        "- dataset: `{}`；states: `{}`；non-tie pairs: `{}`；missions: `{}`。".format(
            identity["dataset_id"], identity["state_count"], identity["non_tie_pair_count"], split["mission_count"]
        ),
        "- 复用 observation_information_audit_v1 的同一 3-fold mission split、同一 pair 顺序和同一 7123 个非平局标签。",
        "- MLP: CPU、2 hidden layers、128 hidden units、Adam lr=1e-3；normalizer 只 fit train fold。",
        "",
        "## Variant 定义",
        "",
        "- S0: production canonical diagnostic projection = vector(22) + depth 9x8 mean-pool + legal mask(105) + previous-action one-hot(105)。",
        "- S1: S0 + acceleration proxy(3) = 当前 velocity 与严格上一决策步 velocity 的差；无 timestamp，因此不声称 m/s²。",
        "- S2: S0 + 当前 depth thumbnail - 严格上一决策步 depth thumbnail。",
        "- S3: S1 + S2。",
        "- S4: S3 + normalized remaining budget = (45 - step_id) / 45。",
        "- 所有历史只从同 mission/source episode 的 `step_id-1` 获取；缺失历史零填充；不使用 future、route、termination reason 或 privileged 信息。",
        "",
        "## History coverage",
        "",
        "- previous state available: `{}` / `{}`；missing and zero-padded: `{}`。".format(
            coverage["previous_state_available_count"], coverage["state_count"], coverage["previous_state_missing_count"]
        ),
        "",
        "## 三折 OOF ranking 指标",
        "",
        "| variant | state dim | pair dim | micro accuracy | micro AUC | mission macro accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = metrics["variants"][variant]
        rows.append(
            "| {} | {} | {} | {:.6f} | {:.6f} | {:.6f} |".format(
                variant,
                item["state_feature_dim"],
                item["pair_input_dim"],
                item["oof_metrics"]["micro"]["accuracy"],
                item["oof_metrics"]["micro"]["auc"],
                item["oof_metrics"]["mission_macro"]["accuracy"],
            )
        )
    rows.extend(
        [
            "",
            "## 相对 S0 的 mission-cluster bootstrap",
            "",
            "| comparison | estimate | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for variant in ("S1", "S2", "S3", "S4"):
        item = bootstrap["paired_deltas"]["{}_minus_S0".format(variant)]
        rows.append("| {}-S0 | {:.6f} | [{:.6f}, {:.6f}] |".format(variant, item["estimate"], item["ci95"][0], item["ci95"][1]))
    qualifying = []
    for variant in VARIANTS[1:]:
        accuracy = metrics["variants"][variant]["oof_metrics"]["micro"]["accuracy"]
        ci_lower = bootstrap["absolute"][variant]["ci95"][0]
        if accuracy >= 0.65 and ci_lower > 0.50:
            qualifying.append(variant)
    decision = (
        "ENTER_CRITIC_REDESIGN_CANDIDATE_QUALIFIED_" + ",".join(qualifying)
        if qualifying
        else "NO_VARIANT_PASSES_0.65_AND_CI_LOWER_GT_0.50"
    )
    rows.extend(
        [
            "",
            "## 结论",
            "",
            "- `AUGMENTATION_DECISION = {}`。".format(decision),
            "- 该 gate 只表示 offline ranking 信息诊断结果，不放行任何 Critic/AWAC 训练，也不改变 production 默认配置。",
            "- 不满足 gate 时，应继续审视 task formulation / hierarchical policy；不能仅凭本实验选择算法。",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_observation_augmentation_ranking_audit(
    dataset_root: Path,
    output_root: Path,
    *,
    observation_audit_root: Path,
    seed: int = 202609071,
    fold_count: int = 3,
    bootstrap_repeats: int = 5000,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_steps: int = MAX_STEPS,
) -> Dict[str, Any]:
    """Run S0-S4 on fixed pairs and persist all diagnostic evidence."""

    if int(fold_count) != 3:
        raise ValueError("this audit requires the fixed three-fold mission split")
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory {}".format(output_root))
    dataset_root = Path(dataset_root).expanduser().resolve()
    observation_audit_root = Path(observation_audit_root).expanduser().resolve()
    data = load_audit_dataset(dataset_root)
    pairs = [row for row in data["pairs"] if not row["return_tie"]]
    if len(pairs) != EXPECTED_NON_TIE_PAIR_COUNT:
        raise ValueError("expected {} non-tie pairs, got {}".format(EXPECTED_NON_TIE_PAIR_COUNT, len(pairs)))
    split_path = observation_audit_root / "mission_split.json"
    identity_path = observation_audit_root / "dataset_identity.json"
    if not split_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError("observation audit split/identity is required")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    observation_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if int(split.get("fold_count", -1)) != 3 or int(split.get("state_count", -1)) != data["identity"]["state_count"]:
        raise ValueError("observation audit split is not the fixed 3-fold input split")
    if observation_identity.get("pairwise_sha256") != data["identity"]["pairwise_sha256"]:
        raise ValueError("observation audit pair identity mismatch")
    if observation_identity.get("dataset_id") != data["identity"]["dataset_id"]:
        raise ValueError("observation audit dataset identity mismatch")
    raw_states = _load_raw_states(dataset_root)
    manifest_rows = _read_jsonl(dataset_root / "state_manifest.jsonl")
    features, coverage = build_augmentation_features(raw_states, manifest_rows, max_steps=int(max_steps))
    if set(split.get("state_to_fold", {})) != set(raw_states):
        raise ValueError("fixed mission split does not cover all states")
    state_ids = np.asarray([str(row["state_id"]) for row in pairs])
    mission_ids = np.asarray([str(row["mission_id"]) for row in pairs])
    folds = np.asarray([int(split["state_to_fold"][state_id]) for state_id in state_ids.tolist()], dtype=np.int64)

    output_root.mkdir(parents=True, exist_ok=True)
    identity = dict(data["identity"])
    identity.update(
        {
            "schema_id": OBSERVATION_AUGMENTATION_RANKING_SCHEMA_ID,
            "source_dataset_manifest_sha256": sha256_file(dataset_root / "dataset_manifest.json"),
            "source_state_manifest_sha256": sha256_file(dataset_root / "state_manifest.jsonl"),
            "observation_audit_identity_sha256": sha256_file(identity_path),
            "observation_audit_split_sha256": sha256_file(split_path),
            "same_pair_order": True,
            "same_mission_split": True,
            "future_features_used": False,
            "production_observation_modified": False,
            "production_replay_modified": False,
            "q_features_used": False,
            "return_features_used": False,
        }
    )
    config = {
        "schema_id": "observation_augmentation_ranking_model_config_v1",
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
        "max_steps": int(max_steps),
        "variants": variant_dimensions(),
        "primary_metric": "micro_pairwise_accuracy",
        "qualification_gate": "micro_pairwise_accuracy >= 0.65 and mission_cluster_absolute_ci95_lower > 0.50",
    }
    _json_dump(output_root / "dataset_identity.json", identity)
    _json_dump(output_root / "mission_split.json", split)
    _json_dump(output_root / "model_config.json", config)

    variant_records: Dict[str, Any] = {}
    correctness_by_variant: Dict[str, np.ndarray] = {}
    prediction_rows: List[Dict[str, Any]] = []
    pair_order_sha = sha256_file(dataset_root / "multi_action_pairwise_dataset.csv")
    for variant in VARIANTS:
        matrix, labels, pair_state_ids, pair_mission_ids = build_pair_feature_matrix(pairs, features, variant)
        oof = np.zeros(len(pairs), dtype=np.float64)
        fold_records: List[Dict[str, Any]] = []
        for fold in range(int(fold_count)):
            train = np.flatnonzero(folds != fold)
            test = np.flatnonzero(folds == fold)
            prediction = _fit_pair_model(
                matrix[train], labels[train], matrix[test], seed=int(seed) + fold * 100 + 41, epochs=int(epochs), batch_size=int(batch_size)
            )
            oof[test] = prediction
            fold_records.append(
                {
                    "fold_id": int(fold),
                    "train_count": int(train.size),
                    "test_count": int(test.size),
                    "metrics": classification_bundle(labels[test], prediction, pair_state_ids[test], pair_mission_ids[test]),
                }
            )
            for index, value in zip(test.tolist(), prediction.tolist()):
                row = pairs[index]
                prediction_rows.append(
                    {
                        "variant": variant,
                        "fold_id": int(fold),
                        "mission_id": str(row["mission_id"]),
                        "state_id": str(row["state_id"]),
                        "step_id": int(row["step_id"]),
                        "action1": int(row["action1"]),
                        "action2": int(row["action2"]),
                        "label": int(labels[index]),
                        "prediction": float(value),
                        "predicted_label": int(value >= 0.5),
                    }
                )
        correctness = (oof >= 0.5) == labels
        correctness_by_variant[variant] = correctness
        bundle = classification_bundle(labels, oof, pair_state_ids, pair_mission_ids)
        variant_records[variant] = {
            "variant": variant,
            "state_feature_dim": int(variant_dimensions()[variant]["state_feature_dim"]),
            "pair_input_dim": int(variant_dimensions()[variant]["pair_input_dim"]),
            "pair_count": int(len(pairs)),
            "folds": fold_records,
            "oof_metrics": bundle,
            "pairwise_accuracy": float(bundle["micro"]["accuracy"]),
            "pairwise_auc": bundle["micro"]["auc"],
            "mission_macro_accuracy": float(bundle["mission_macro"]["accuracy"]),
            "mission_macro_auc": bundle["mission_macro"]["auc"],
        }

    absolute: Dict[str, Any] = {}
    for index, variant in enumerate(VARIANTS):
        absolute[variant] = _bootstrap_absolute(
            correctness_by_variant[variant], mission_ids, repeats=int(bootstrap_repeats), seed=int(seed) + 5000 + index
        )
    paired_deltas: Dict[str, Any] = {}
    baseline = correctness_by_variant["S0"].astype(np.int8)
    for index, variant in enumerate(VARIANTS[1:]):
        paired_deltas["{}_minus_S0".format(variant)] = _bootstrap_delta(
            correctness_by_variant[variant].astype(np.int8) - baseline,
            mission_ids,
            repeats=int(bootstrap_repeats),
            seed=int(seed) + 6000 + index,
        )

    metrics = {
        "schema_id": VARIANT_METRICS_SCHEMA_ID,
        "dataset_identity": identity,
        "pairwise_source_sha256": pair_order_sha,
        "fixed_pair_order": True,
        "fixed_mission_split": True,
        "variants": variant_records,
    }
    bootstrap = {
        "schema_id": BOOTSTRAP_SCHEMA_ID,
        "cluster_unit": "mission_id",
        "repeats": int(bootstrap_repeats),
        "absolute": absolute,
        "paired_deltas": paired_deltas,
        "note": "CI is mission-cluster only; it excludes training-seed uncertainty.",
    }
    feature_delta = {
        "schema_id": FEATURE_DELTA_SCHEMA_ID,
        "history": coverage,
        "variant_definitions": {
            "S0": "production canonical diagnostic projection",
            "S1": "S0 + velocity difference to step_id - 1",
            "S2": "S0 + depth thumbnail difference to step_id - 1",
            "S3": "S0 + velocity difference + depth difference",
            "S4": "S3 + (max_steps - step_id) / max_steps",
        },
        "dimensions": variant_dimensions(),
        "same_pair_count": len(pairs),
        "same_pair_order": True,
        "same_mission_split": True,
        "causality": {
            "future_state_used": False,
            "future_reward_used": False,
            "termination_reason_used": False,
            "route_used": False,
            "privileged_information_used": False,
        },
    }
    _json_dump(output_root / "variant_metrics.json", metrics)
    _json_dump(output_root / "bootstrap.json", bootstrap)
    _json_dump(output_root / "feature_delta_analysis.json", feature_delta)
    _write_predictions(output_root / "predictions.csv", prediction_rows)
    _write_report(output_root / "report_zh.md", identity=identity, split=split, coverage=coverage, metrics=metrics, bootstrap=bootstrap)
    return {
        "output_root": str(output_root),
        "identity": identity,
        "split": split,
        "coverage": coverage,
        "config": config,
        "variant_metrics": metrics,
        "bootstrap": bootstrap,
        "feature_delta": feature_delta,
    }


__all__ = [
    "VARIANTS",
    "build_augmentation_features",
    "build_pair_feature_matrix",
    "build_previous_state_index",
    "run_observation_augmentation_ranking_audit",
    "variant_dimensions",
]
