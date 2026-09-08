"""Read-only high-level action abstraction audit.

The module maps the immutable 105-action MPL into semantic candidates without
looking at return labels.  It then compares the same small pairwise-ranking
MLP on the fixed mission split.  It never imports a production learner and
does not modify Replay, action definitions, or runtime configuration.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from planning.diagnostics.action_formulation_audit import _load_action_library
from planning.diagnostics.state_sufficiency import (
    _Standardizer,
    _one_hot,
    _train_model,
    classification_bundle,
    load_audit_dataset,
    mission_cluster_bootstrap,
    sha256_file,
)


HIGH_LEVEL_ACTION_ABSTRACTION_AUDIT_SCHEMA_ID = "high_level_action_abstraction_audit_v1"
MAPPING_SCHEMA_ID = "high_level_action_mapping_v1"
ABSTRACTION_METRICS_SCHEMA_ID = "high_level_abstraction_metrics_v1"
RANKING_METRICS_SCHEMA_ID = "high_level_ranking_metrics_v1"
EXPECTED_ACTION_COUNT = 105
EXPECTED_STATE_COUNT = 479
EXPECTED_TRANSITION_COUNT = 2850
EXPECTED_PAIR_COUNT = 7124
DEFAULT_EPOCHS = 120
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 20260908


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_stats(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    if not np.isfinite(array).all():
        raise ValueError("non-finite statistic input")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _action_cluster_labels(item: Mapping[str, Any]) -> Dict[str, str]:
    lateral_mode = str(item.get("lateral_mode", "")).strip().lower()
    vertical_mode = str(item.get("vertical_mode", "")).strip().lower()
    lateral = lateral_mode if lateral_mode in {"left", "straight", "right"} else "unknown_lateral"
    vertical = vertical_mode if vertical_mode in {"down", "level", "up"} else "unknown_vertical"
    return {
        "primitive_family": "fixed_duration_endpoint_primitive",
        "horizontal_family": lateral,
        "vertical_family": vertical,
        "motion_direction": "{}_{}".format(lateral, vertical),
    }


def _geometry_kmeans(descriptors: np.ndarray, *, cluster_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic farthest-point seeded geometry clustering.

    Geometry is the normalized MPL descriptor only.  No reward, return,
    terminal result, or pair label is used to seed or fit this mapping.
    """

    points = np.asarray(descriptors, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < int(cluster_count):
        raise ValueError("geometry matrix cannot support requested cluster count")
    centers = [0]
    min_distance = np.full(points.shape[0], np.inf, dtype=np.float64)
    for _ in range(1, int(cluster_count)):
        distance = np.sum((points - points[centers[-1]]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
        next_index = int(np.argmax(min_distance))
        if next_index in centers:
            next_index = next(index for index in range(len(points)) if index not in centers)
        centers.append(next_index)
    centroids = points[np.asarray(centers)].copy()
    for _ in range(50):
        distances = np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        updated = centroids.copy()
        for index in range(len(centroids)):
            members = points[labels == index]
            if len(members):
                updated[index] = np.mean(members, axis=0)
        if np.allclose(updated, centroids, rtol=0.0, atol=1.0e-12):
            centroids = updated
            break
        centroids = updated
    # Canonicalize labels by centroid coordinates so IDs are reproducible even
    # if a future implementation changes the deterministic seed order.
    order = sorted(range(len(centroids)), key=lambda index: tuple(float(v) for v in centroids[index]))
    remap = {old: new for new, old in enumerate(order)}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int64), centroids[np.asarray(order)]


def _candidate_mappings(library: Mapping[str, Any]) -> Dict[str, Any]:
    actions = sorted(library["actions"], key=lambda row: int(row["id"]))
    if len(actions) != EXPECTED_ACTION_COUNT:
        raise ValueError("expected 105 actions")
    descriptors = np.asarray(
        [library["definitions"][int(row["id"])] ["descriptor"] for row in actions], dtype=np.float64
    )
    geometry_labels, geometry_centroids = _geometry_kmeans(descriptors, cluster_count=9)
    candidates: Dict[str, Dict[str, Any]] = {
        "primitive_family": {
            "description": "The fixed-duration endpoint primitive family declared by the current MPL.",
            "source": "motion primitive family metadata, never return labels",
            "label_for_action": {},
        },
        "motion_direction": {
            "description": "lateral_mode x vertical_mode semantic direction intent.",
            "source": "MPL lateral_mode and vertical_mode, never return labels",
            "label_for_action": {},
        },
        "duration_bucket": {
            "description": "duration_s and command_frames bucket.",
            "source": "MPL duration metadata, never return labels",
            "label_for_action": {},
        },
        "geometry_cluster": {
            "description": "Deterministic geometry-only k-means over normalized endpoint/heading descriptors.",
            "source": "normalized MPL geometry descriptors, never return labels",
            "label_for_action": {},
            "cluster_count": 9,
            "centroids": {
                "geometry_cluster_{:02d}".format(index): [float(value) for value in centroid]
                for index, centroid in enumerate(geometry_centroids)
            },
        },
    }
    for index, raw in enumerate(actions):
        action_id = int(raw["id"])
        semantic = _action_cluster_labels(raw)
        candidates["primitive_family"]["label_for_action"][str(action_id)] = "primitive_family_fixed_duration_endpoint"
        candidates["motion_direction"]["label_for_action"][str(action_id)] = semantic["motion_direction"]
        candidates["duration_bucket"]["label_for_action"][str(action_id)] = "duration_{:.3f}s_frames_{}".format(
            float(raw["duration_s"]), int(raw["command_frames"])
        )
        candidates["geometry_cluster"]["label_for_action"][str(action_id)] = "geometry_cluster_{:02d}".format(
            int(geometry_labels[index])
        )
    for candidate in candidates.values():
        labels = sorted(set(candidate["label_for_action"].values()))
        candidate["high_level_ids"] = labels
        candidate["high_level_count"] = len(labels)
        candidate["primitive_to_high_level_count"] = EXPECTED_ACTION_COUNT / float(len(labels))
    return candidates


def _mapping_payload(library: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    per_action = []
    for raw in sorted(library["actions"], key=lambda row: int(row["id"])):
        action_id = int(raw["id"])
        per_action.append(
            {
                "action_id": action_id,
                "primitive_id": "mpl_action_{:03d}".format(action_id),
                "lateral_mode": str(raw["lateral_mode"]),
                "vertical_mode": str(raw["vertical_mode"]),
                "lateral_endpoint_m": float(raw["lateral_endpoint_m"]),
                "vertical_endpoint_m": float(raw["vertical_endpoint_m"]),
                "terminal_heading_deg": float(raw["terminal_heading_deg"]),
                "duration_s": float(raw["duration_s"]),
                "command_frames": int(raw["command_frames"]),
                "high_level_ids": {
                    name: str(candidate["label_for_action"][str(action_id)])
                    for name, candidate in candidates.items()
                },
            }
        )
    candidate_summary = {}
    for name, candidate in candidates.items():
        candidate_summary[name] = {
            key: value
            for key, value in candidate.items()
            if key != "label_for_action"
        }
    return {
        "schema_id": MAPPING_SCHEMA_ID,
        "diagnostic_only": True,
        "source": {
            "motion_primitives_json": library["json_path"],
            "motion_primitives_npz": library["npz_path"],
            "motion_primitives_json_sha256": library["json_sha256"],
            "motion_primitives_npz_sha256": library["npz_sha256"],
            "mpl_contract_sha256": library["meta"].get("contract_sha256"),
        },
        "mapping_rule": "All candidate labels are derived from MPL geometry/metadata before any outcome or return is read.",
        "candidates": candidate_summary,
        "actions": per_action,
    }


def _stats_by_candidate(
    transitions: Sequence[Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for name, candidate in candidates.items():
        groups: Dict[str, Dict[str, Any]] = {}
        for row in transitions:
            high_level = str(candidate["label_for_action"][str(int(row["action"]))])
            group = groups.setdefault(
                high_level,
                {"actions": set(), "missions": set(), "returns": [], "success_returns": [], "failure_returns": [], "success_count": 0, "failure_count": 0},
            )
            group["actions"].add(int(row["action"]))
            group["missions"].add(str(row["mission_id"]))
            group["returns"].append(float(row["episode_return"]))
            if int(row["success"]):
                group["success_count"] += 1
                group["success_returns"].append(float(row["episode_return"]))
            else:
                group["failure_count"] += 1
                group["failure_returns"].append(float(row["episode_return"]))
        total_missions = len({str(row["mission_id"]) for row in transitions})
        output[name] = {}
        for high_level in sorted(groups):
            group = groups[high_level]
            output[name][high_level] = {
                "primitive_action_ids": sorted(group["actions"]),
                "primitive_action_count": len(group["actions"]),
                "mission_count": len(group["missions"]),
                "mission_frequency": float(len(group["missions"]) / max(1, total_missions)),
                "transition_count": len(group["returns"]),
                "success_transition_count": group["success_count"],
                "failure_transition_count": group["failure_count"],
                "success_rate": float(group["success_count"] / max(1, len(group["returns"]))),
                "return": _finite_stats(group["returns"]),
                "success_return": _finite_stats(group["success_returns"]),
                "failure_return": _finite_stats(group["failure_returns"]),
            }
    return output


def _branching_metrics(states: Mapping[str, np.ndarray], candidates: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    primitive_counts = []
    for feature in states.values():
        mask = np.asarray(feature[22 : 22 + EXPECTED_ACTION_COUNT], dtype=np.float32) > 0.5
        primitive_counts.append(int(np.count_nonzero(mask)))
    output: Dict[str, Any] = {
        "state_count": len(states),
        "primitive_valid_count": _finite_stats(primitive_counts),
        "candidates": {},
    }
    for name, candidate in candidates.items():
        counts = []
        reductions = []
        for feature, primitive_count in zip(states.values(), primitive_counts):
            mask = np.asarray(feature[22 : 22 + EXPECTED_ACTION_COUNT], dtype=np.float32) > 0.5
            high_level = {
                candidate["label_for_action"][str(action_id)]
                for action_id in np.flatnonzero(mask).tolist()
            }
            counts.append(len(high_level))
            reductions.append(1.0 - len(high_level) / max(1, primitive_count))
        output["candidates"][name] = {
            "valid_high_level_count": _finite_stats(counts),
            "branching_reduction_ratio": _finite_stats(reductions),
            "branching_factor_reduction_absolute_mean": float(np.mean(np.asarray(primitive_counts) - np.asarray(counts))),
        }
    return output


def _pair_features(
    pairs: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    label_to_index: Mapping[str, int],
) -> np.ndarray:
    state = np.stack([np.asarray(row["state_feature"], dtype=np.float32) for row in pairs])
    first = np.asarray(
        [label_to_index[str(candidate["label_for_action"][str(int(row["action1"]))])] for row in pairs], dtype=np.int64
    )
    second = np.asarray(
        [label_to_index[str(candidate["label_for_action"][str(int(row["action2"]))])] for row in pairs], dtype=np.int64
    )
    suffix_first = np.zeros((len(pairs), len(label_to_index)), dtype=np.float32)
    suffix_second = np.zeros((len(pairs), len(label_to_index)), dtype=np.float32)
    suffix_first[np.arange(len(pairs)), first] = 1.0
    suffix_second[np.arange(len(pairs)), second] = 1.0
    return np.concatenate((state, suffix_first, suffix_second), axis=1).astype(np.float32)


def _pairwise_ranking(
    data: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    mission_split: Mapping[str, Any],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    bootstrap_repeats: int,
) -> Dict[str, Any]:
    all_pairs = list(data["pairs"])
    if len(all_pairs) != EXPECTED_PAIR_COUNT:
        raise ValueError("expected {} pair rows".format(EXPECTED_PAIR_COUNT))
    pairs = [row for row in all_pairs if not row["return_tie"]]
    tie_count = len(all_pairs) - len(pairs)
    mission_to_fold = {str(key): int(value) for key, value in mission_split["mission_to_fold"].items()}
    pair_missions = np.asarray([str(row["mission_id"]) for row in pairs])
    labels = np.asarray([int(row["label"]) for row in pairs], dtype=np.int64)
    folds = np.asarray([mission_to_fold[str(mission)] for mission in pair_missions.tolist()], dtype=np.int64)
    records: Dict[str, Any] = {}
    for candidate_index, (name, candidate) in enumerate(
        [("original_105", {"label_for_action": {str(index): str(index) for index in range(EXPECTED_ACTION_COUNT)}, "high_level_ids": [str(index) for index in range(EXPECTED_ACTION_COUNT)]})]
        + list(candidates.items())
    ):
        labels_sorted = sorted(set(candidate["label_for_action"].values()))
        label_to_index = {label: index for index, label in enumerate(labels_sorted)}
        features = _pair_features(pairs, candidate, label_to_index)
        oof = np.zeros(len(pairs), dtype=np.float64)
        folds_out = []
        for fold in sorted(set(folds.tolist())):
            train = np.flatnonzero(folds != fold)
            test = np.flatnonzero(folds == fold)
            normalizer = _Standardizer.fit(features[train])
            prediction = _train_model(
                normalizer.transform(features[train]),
                labels[train],
                normalizer.transform(features[test]),
                task="classification",
                seed=int(seed) + candidate_index * 1000 + int(fold),
                epochs=int(epochs),
                batch_size=int(batch_size),
            )
            oof[test] = prediction
            folds_out.append(
                {
                    "fold_id": int(fold),
                    "train_count": int(len(train)),
                    "test_count": int(len(test)),
                    "metrics": classification_bundle(
                        labels[test], prediction,
                        np.asarray([str(pairs[index]["state_id"]) for index in test]),
                        pair_missions[test],
                    ),
                }
            )
        mission_values: Dict[str, List[float]] = defaultdict(list)
        for mission in sorted(set(pair_missions.tolist())):
            indices = np.flatnonzero(pair_missions == mission)
            mission_values[str(mission)] = [
                float(np.mean((oof[indices] >= 0.5) == labels[indices]))
            ]
        records[name] = {
            "high_level_count": len(labels_sorted),
            "high_level_ids": labels_sorted,
            "mapped_same_pair_count": int(
                sum(candidate["label_for_action"][str(int(row["action1"]))] == candidate["label_for_action"][str(int(row["action2"]))] for row in pairs)
            ),
            "folds": folds_out,
            "oof_metrics": classification_bundle(
                labels,
                oof,
                np.asarray([str(row["state_id"]) for row in pairs]),
                pair_missions,
            ),
            "mission_cluster_accuracy_bootstrap": mission_cluster_bootstrap(
                mission_values, repeats=int(bootstrap_repeats), seed=int(seed) + candidate_index + 5000
            ),
        }
    baseline = records["original_105"]
    baseline_accuracy = np.asarray([0.0])
    # Store paired mission accuracy deltas against the same OOF folds.  The
    # per-mission values are recomputed from the persisted pair order to avoid
    # treating pair rows as independent clusters.
    baseline_pred = _retrain_oof_for_delta(data, candidates, mission_split, seed=seed, epochs=epochs, batch_size=batch_size)
    for name in candidates:
        candidate_pred = baseline_pred[name]
        baseline_values: Dict[str, List[float]] = defaultdict(list)
        delta_values: Dict[str, List[float]] = defaultdict(list)
        for index, row in enumerate(pairs):
            mission = str(row["mission_id"])
            baseline_values[mission].append(float((baseline_pred["original_105"][index] >= 0.5) == labels[index]))
            delta_values[mission].append(float((candidate_pred[index] >= 0.5) == labels[index]))
        delta_by_mission = {
            mission: [float(np.mean(delta_values[mission]) - np.mean(baseline_values[mission]))]
            for mission in baseline_values
        }
        records[name]["paired_delta_vs_original_105"] = mission_cluster_bootstrap(
            delta_by_mission, repeats=int(bootstrap_repeats), seed=int(seed) + 7000 + list(candidates).index(name)
        )
    return {
        "schema_id": RANKING_METRICS_SCHEMA_ID,
        "diagnostic_only": True,
        "pair_count_all": len(all_pairs),
        "pair_count_non_tie": len(pairs),
        "tie_pair_count_excluded_from_binary_ranking": tie_count,
        "same_pair_labels": True,
        "fold_count": int(mission_split["fold_count"]),
        "training": {"model": "state_feature + candidate_action_one_hot_pair", "hidden_dim": 128, "hidden_layers": 2, "epochs": int(epochs), "batch_size": int(batch_size), "optimizer": "Adam(lr=1e-3)", "seed": int(seed)},
        "models": records,
        "threshold": {"high_level_ranking_accuracy_required": 0.65, "decision_rule": "any abstraction candidate micro accuracy > 0.65 is required for prototype consideration"},
    }


def _retrain_oof_for_delta(
    data: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]], mission_split: Mapping[str, Any], *, seed: int, epochs: int, batch_size: int
) -> Dict[str, np.ndarray]:
    """Produce the same deterministic OOF sequence used for paired deltas."""

    pairs = [row for row in data["pairs"] if not row["return_tie"]]
    labels = np.asarray([int(row["label"]) for row in pairs], dtype=np.int64)
    missions = np.asarray([str(row["mission_id"]) for row in pairs])
    folds = np.asarray([int(mission_split["mission_to_fold"][str(mission)]) for mission in missions.tolist()])
    ordered = [("original_105", {"label_for_action": {str(index): str(index) for index in range(EXPECTED_ACTION_COUNT)}})] + list(candidates.items())
    result = {}
    for candidate_index, (name, candidate) in enumerate(ordered):
        labels_sorted = sorted(set(candidate["label_for_action"].values()))
        label_to_index = {label: index for index, label in enumerate(labels_sorted)}
        features = _pair_features(pairs, candidate, label_to_index)
        oof = np.zeros(len(pairs), dtype=np.float64)
        for fold in sorted(set(folds.tolist())):
            train = np.flatnonzero(folds != fold)
            test = np.flatnonzero(folds == fold)
            normalizer = _Standardizer.fit(features[train])
            oof[test] = _train_model(
                normalizer.transform(features[train]), labels[train], normalizer.transform(features[test]),
                task="classification", seed=int(seed) + candidate_index * 1000 + int(fold), epochs=int(epochs), batch_size=int(batch_size)
            )
        result[name] = oof
    return result


def _build_report(mapping: Mapping[str, Any], abstraction: Mapping[str, Any], ranking: Mapping[str, Any]) -> str:
    original = ranking["models"]["original_105"]["oof_metrics"]
    lines = [
        "# HIGH-LEVEL ACTION ABSTRACTION AUDIT V1",
        "",
        "本报告为只读 offline diagnostic；未执行 AWAC、Actor、Critic、Unity/Bridge，未修改 production action、Replay 或训练配置。",
        "",
        "## 输入与映射",
        "",
        "- 105 action MPL 映射全部由几何/metadata 生成，未使用 return、success、failure 或 pair label。",
        "- pair rows：全部 {}，其中 tie={}；二元 ranking 使用既有口径的 non-tie={}，没有重写标签。".format(
            ranking["pair_count_all"], ranking["tie_pair_count_excluded_from_binary_ranking"], ranking["pair_count_non_tie"]
        ),
        "- 固定 mission split：{} folds；state/mission 未重新随机划分。".format(ranking["fold_count"]),
        "",
        "## Abstraction 压缩",
        "",
    ]
    for name, candidate in mapping["candidates"].items():
        lines.append(
            "- `{}`：{} 个 high-level id，平均每个 high-level 对应 {:.2f} 个 primitive。".format(
                name, candidate["high_level_count"], candidate["primitive_to_high_level_count"]
            )
        )
    lines.extend(
        [
            "",
            "## Ranking 结果",
            "",
            "| candidate | high-level count | micro accuracy | micro AUC | mission macro accuracy | mapped same-pair |",
            "|---|---:|---:|---:|---:|---:|",
            "| original_105 | {} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
                ranking["models"]["original_105"]["high_level_count"],
                original["micro"]["accuracy"],
                original["micro"]["auc"],
                original["mission_macro"]["accuracy"],
                ranking["models"]["original_105"]["mapped_same_pair_count"],
            ),
        ]
    )
    high_level_pass = False
    best = None
    for name in mapping["candidates"]:
        record = ranking["models"][name]
        metrics = record["oof_metrics"]
        lines.append(
            "| {} | {} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
                name,
                record["high_level_count"],
                metrics["micro"]["accuracy"],
                metrics["micro"]["auc"],
                metrics["mission_macro"]["accuracy"],
                record["mapped_same_pair_count"],
            )
        )
        if best is None or metrics["micro"]["accuracy"] > best[1]:
            best = (name, metrics["micro"]["accuracy"])
        if metrics["micro"]["accuracy"] > 0.65:
            high_level_pass = True
    lines.extend(
        [
            "",
            "## Branching factor",
            "",
            "每个 state 的 valid primitive/high-level 数量及减少比例保存在 `abstraction_metrics.json`；mask 只用于 valid action branching 统计，不参与 high-level mapping。",
            "",
            "## 结论",
            "",
            "1. 高层 abstraction 是否减少 ambiguity：是，尤其 direction/geometry candidate 会压缩 action branching；primitive_family 与 duration_bucket 在当前 MPL 中各只有 1 类，不能提供有意义的 action discrimination。",
            "2. ranking 是否提升：见上表和 `ranking_metrics.json` 的同 mission OOF paired CI；不能把降维本身当作提升。",
            "3. 是否进入 hierarchical prototype：{}。判定阈值为 high-level micro ranking accuracy > 0.65；当前最佳 candidate 为 `{}`，accuracy={:.6f}。".format(
                "YES" if high_level_pass else "NO",
                best[0] if best else "UNKNOWN",
                best[1] if best else float("nan"),
            ),
            "",
            "因此：若未达到阈值，继续研究 state/reward，不切换 production action；若达到阈值，也只代表进入隔离 prototype，不代表 production gate 或 AWAC 成功。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_high_level_action_abstraction_audit(
    *,
    dataset_root: Path,
    hierarchical_root: Path,
    mission_split_path: Path,
    motion_primitives_json: Path,
    motion_primitives_npz: Path,
    out_dir: Path,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    bootstrap_repeats: int = 5000,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    hierarchical_root = Path(hierarchical_root).expanduser().resolve()
    mission_split_path = Path(mission_split_path).expanduser().resolve()
    motion_primitives_json = Path(motion_primitives_json).expanduser().resolve()
    motion_primitives_npz = Path(motion_primitives_npz).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output: {}".format(out_dir))
    if not hierarchical_root.is_dir():
        raise FileNotFoundError(str(hierarchical_root))
    split = json.loads(mission_split_path.read_text(encoding="utf-8"))
    data = load_audit_dataset(dataset_root)
    if data["identity"]["state_count"] != EXPECTED_STATE_COUNT or data["identity"]["transition_count"] != EXPECTED_TRANSITION_COUNT:
        raise ValueError("unexpected diagnostic dataset dimensions")
    if data["identity"]["pair_count"] != EXPECTED_PAIR_COUNT:
        raise ValueError("unexpected pair count")
    data_missions = set(data["missions"])
    if data_missions != set(split["mission_to_fold"]):
        raise ValueError("mission split does not cover exactly the input missions")
    library = _load_action_library(motion_primitives_json, motion_primitives_npz)
    candidates = _candidate_mappings(library)
    mapping = _mapping_payload(library, candidates)
    abstraction = {
        "schema_id": ABSTRACTION_METRICS_SCHEMA_ID,
        "diagnostic_only": True,
        "dataset_identity": data["identity"],
        "mission_split_sha256": sha256_file(mission_split_path),
        "high_level_statistics": _stats_by_candidate(data["transitions"], candidates),
        "branching_factor": _branching_metrics(data["states"], candidates),
        "ambiguity": {
            name: {
                "primitive_action_count": EXPECTED_ACTION_COUNT,
                "high_level_action_count": int(candidate["high_level_count"]),
                "compression_ratio": float(EXPECTED_ACTION_COUNT / candidate["high_level_count"]),
            }
            for name, candidate in candidates.items()
        },
        "return_free_mapping": True,
    }
    ranking = _pairwise_ranking(
        data, candidates, split, seed=int(seed), epochs=int(epochs), batch_size=int(batch_size), bootstrap_repeats=int(bootstrap_repeats)
    )
    ranking["dataset_identity"] = data["identity"]
    ranking["mission_split_sha256"] = sha256_file(mission_split_path)
    ranking["motion_primitives_json_sha256"] = sha256_file(motion_primitives_json)
    ranking["motion_primitives_npz_sha256"] = sha256_file(motion_primitives_npz)
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(out_dir / "high_level_action_mapping.json", mapping)
    _json_dump(out_dir / "abstraction_metrics.json", abstraction)
    _json_dump(out_dir / "ranking_metrics.json", ranking)
    (out_dir / "report_zh.md").write_text(_build_report(mapping, abstraction, ranking), encoding="utf-8")
    return {
        "schema_id": HIGH_LEVEL_ACTION_ABSTRACTION_AUDIT_SCHEMA_ID,
        "out_dir": str(out_dir),
        "mapping": mapping,
        "abstraction": abstraction,
        "ranking": ranking,
        "production_modified": False,
        "rl_training_executed": False,
        "runtime_started": False,
    }


__all__ = ["run_high_level_action_abstraction_audit"]
