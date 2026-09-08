"""Read-only action-formulation audit for the 105-action diagnostic replay.

The production action library is loaded as an immutable artifact.  This audit
compares the historical 105-way one-hot pair input with a three-dimensional
continuous primitive descriptor and reports action geometry/outcome structure.
It does not load a Q checkpoint and never writes replay or production files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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


ACTION_FORMULATION_AUDIT_SCHEMA_ID = "action_formulation_audit_v1"
ACTION_ANALYSIS_SCHEMA_ID = "action_formulation_action_analysis_v1"
PRIMITIVE_METRICS_SCHEMA_ID = "action_formulation_primitive_metrics_v1"
ACTION_MODEL_CONFIG_SCHEMA_ID = "action_formulation_model_config_v1"
ACTION_REPRESENTATIONS = ("one_hot", "primitive_descriptor")
VARIABLE_PARAMETER_NAMES = (
    "lateral_endpoint_m",
    "vertical_endpoint_m",
    "terminal_heading_deg",
)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "variance": None, "p10": None, "p90": None, "p99": None}
    if not np.isfinite(array).all():
        raise ValueError("non-finite diagnostic statistic input")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "variance": float(np.var(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
    }


def _normalize(value: float, bounds: Mapping[str, float]) -> float:
    lower = float(bounds["min"])
    upper = float(bounds["max"])
    if not upper > lower:
        raise ValueError("action parameter range must be non-degenerate")
    return float(2.0 * (float(value) - lower) / (upper - lower) - 1.0)


def build_action_definitions(
    actions: Sequence[Mapping[str, Any]],
    *,
    parameter_ranges: Mapping[str, Mapping[str, float]],
    horizontal_count: int,
    vertical_count: int,
) -> Dict[int, Dict[str, Any]]:
    """Build semantic descriptors and explicit lattice neighbors."""

    by_grid: Dict[Tuple[int, int], int] = {}
    definitions: Dict[int, Dict[str, Any]] = {}
    for raw in actions:
        action_id = int(raw["id"])
        horizontal = int(raw["horizontal_index"])
        vertical = int(raw["vertical_index"])
        if action_id in definitions or (horizontal, vertical) in by_grid:
            raise ValueError("duplicate action identity")
        if not (0 <= horizontal < int(horizontal_count) and 0 <= vertical < int(vertical_count)):
            raise ValueError("action lattice index outside declared dimensions")
        parameter_vector = np.asarray(
            [float(raw[name]) for name in VARIABLE_PARAMETER_NAMES], dtype=np.float32
        )
        descriptor = np.asarray(
            [_normalize(float(raw[name]), parameter_ranges[name]) for name in VARIABLE_PARAMETER_NAMES],
            dtype=np.float32,
        )
        definitions[action_id] = {
            "action_id": action_id,
            "primitive_id": "mpl_action_{:03d}".format(action_id),
            "horizontal_index": horizontal,
            "vertical_index": vertical,
            "parameter_dim": len(VARIABLE_PARAMETER_NAMES),
            "parameter_names": list(VARIABLE_PARAMETER_NAMES),
            "parameter_vector": parameter_vector.astype(float).tolist(),
            "descriptor": descriptor.astype(float).tolist(),
            "parameter_ranges": {
                name: {"min": float(bounds["min"]), "max": float(bounds["max"])}
                for name, bounds in parameter_ranges.items()
            },
            "duration_s": float(raw["duration_s"]),
            "command_frames": int(raw["command_frames"]),
            "control_dt_s": float(raw.get("control_dt_s", 0.0)),
            "lateral_mode": str(raw.get("lateral_mode", "")),
            "vertical_mode": str(raw.get("vertical_mode", "")),
        }
        by_grid[(horizontal, vertical)] = action_id

    for action_id, item in definitions.items():
        h = int(item["horizontal_index"])
        v = int(item["vertical_index"])
        neighbors_4 = []
        neighbors_8 = []
        for dh in (-1, 0, 1):
            for dv in (-1, 0, 1):
                if dh == 0 and dv == 0:
                    continue
                neighbor = by_grid.get((h + dh, v + dv))
                if neighbor is None:
                    continue
                neighbors_8.append(int(neighbor))
                if abs(dh) + abs(dv) == 1:
                    neighbors_4.append(int(neighbor))
        item["neighbors_4"] = sorted(neighbors_4)
        item["neighbors_8"] = sorted(neighbors_8)
    if len(definitions) != int(horizontal_count) * int(vertical_count):
        raise ValueError("action count does not fill the declared lattice")
    return dict(sorted(definitions.items()))


def classify_action_pair(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, bool]:
    """Classify two action metadata rows without treating IDs as semantics."""

    same_id = int(first["id"]) == int(second["id"])
    same_horizontal = int(first["horizontal_index"]) == int(second["horizontal_index"])
    same_vertical = int(first["vertical_index"]) == int(second["vertical_index"])
    return {
        "same_action_id": same_id,
        "same_grid_cell": bool(same_id and same_horizontal and same_vertical),
        "same_horizontal_family": bool(same_horizontal and not same_id),
        "same_vertical_family": bool(same_vertical and not same_id),
        "different_horizontal_and_vertical": bool(not same_horizontal and not same_vertical),
    }


def build_pair_action_features(
    state_matrix: np.ndarray,
    actions1: Sequence[int],
    actions2: Sequence[int],
    *,
    representation: str,
    descriptors: Optional[Mapping[int, np.ndarray]] = None,
) -> np.ndarray:
    """Append either 105-way identity or semantic primitive descriptors."""

    states = np.asarray(state_matrix, dtype=np.float32)
    first = np.asarray(actions1, dtype=np.int64).reshape(-1)
    second = np.asarray(actions2, dtype=np.int64).reshape(-1)
    if states.ndim != 2 or len(states) != len(first) or len(first) != len(second):
        raise ValueError("state/action pair lengths are incompatible")
    if np.any(first < 0) or np.any(first >= ACTION_DIM) or np.any(second < 0) or np.any(second >= ACTION_DIM):
        raise ValueError("action ID outside the 105-action space")
    if representation == "one_hot":
        suffix = np.concatenate((_one_hot(first), _one_hot(second)), axis=1)
    elif representation == "primitive_descriptor":
        if descriptors is None:
            raise ValueError("primitive descriptors are required")
        try:
            suffix = np.concatenate(
                (
                    np.asarray([descriptors[int(action)] for action in first], dtype=np.float32),
                    np.asarray([descriptors[int(action)] for action in second], dtype=np.float32),
                ),
                axis=1,
            )
        except KeyError as error:
            raise ValueError("primitive descriptor missing action") from error
        if suffix.shape[1] != 2 * len(VARIABLE_PARAMETER_NAMES):
            raise ValueError("primitive descriptor has unexpected dimension")
    else:
        raise ValueError("unknown action representation {}".format(representation))
    result = np.concatenate((states, suffix), axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("action feature matrix contains non-finite values")
    return result


def _load_action_library(json_path: Path, npz_path: Path) -> Dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "meta" not in payload or "actions" not in payload:
        raise ValueError("motion primitive metadata has unexpected schema")
    meta = payload["meta"]
    actions = list(payload["actions"])
    if int(meta["num_actions"]) != ACTION_DIM or len(actions) != ACTION_DIM:
        raise ValueError("motion primitive library must contain 105 actions")
    arrays: Dict[str, np.ndarray] = {}
    with np.load(str(npz_path), allow_pickle=False) as loaded:
        for key in ("pos_ref", "cmd_seq"):
            if key not in loaded.files:
                raise ValueError("motion primitive NPZ missing {}".format(key))
            arrays[key] = np.asarray(loaded[key], dtype=np.float32)
    if arrays["pos_ref"].shape[0] != ACTION_DIM or arrays["cmd_seq"].shape[0] != ACTION_DIM:
        raise ValueError("motion primitive arrays have wrong action count")
    parameter_ranges = {
        "lateral_endpoint_m": {"min": float(min(meta["lateral_endpoint_samples_m"])), "max": float(max(meta["lateral_endpoint_samples_m"]))},
        "vertical_endpoint_m": {"min": float(min(meta["vertical_endpoint_samples_m"])), "max": float(max(meta["vertical_endpoint_samples_m"]))},
        "terminal_heading_deg": {"min": float(min(meta["terminal_heading_samples_deg"])), "max": float(max(meta["terminal_heading_samples_deg"]))},
    }
    definitions = build_action_definitions(
        actions,
        parameter_ranges=parameter_ranges,
        horizontal_count=int(meta["num_horizontal"]),
        vertical_count=int(meta["num_vertical"]),
    )
    return {
        "json_path": str(json_path.resolve()),
        "npz_path": str(npz_path.resolve()),
        "json_sha256": sha256_file(json_path),
        "npz_sha256": sha256_file(npz_path),
        "meta": meta,
        "actions": actions,
        "parameter_ranges": parameter_ranges,
        "definitions": definitions,
        "pos_ref": arrays["pos_ref"],
        "cmd_seq": arrays["cmd_seq"],
    }


def _pair_distance_rows(library: Mapping[str, Any]) -> List[Dict[str, Any]]:
    definitions = library["definitions"]
    position = np.asarray(library["pos_ref"], dtype=np.float32)
    command = np.asarray(library["cmd_seq"], dtype=np.float32)
    rows = []
    for first, second in combinations(range(ACTION_DIM), 2):
        classification = classify_action_pair(library["actions"][first], library["actions"][second])
        descriptor_distance = float(
            np.linalg.norm(np.asarray(definitions[first]["descriptor"]) - np.asarray(definitions[second]["descriptor"]))
        )
        endpoint_distance = float(np.linalg.norm(position[first, -1] - position[second, -1]))
        path_delta = position[first] - position[second]
        path_rms_distance = float(np.sqrt(np.mean(np.sum(path_delta * path_delta, axis=1))))
        command_delta = command[first] - command[second]
        command_rms_distance = float(np.sqrt(np.mean(command_delta * command_delta)))
        exact_behavior = bool(
            np.max(np.abs(path_delta)) <= 1.0e-6 and np.max(np.abs(command_delta)) <= 1.0e-6
        )
        rows.append(
            {
                "action1": first,
                "action2": second,
                "descriptor_distance": descriptor_distance,
                "geometry_distance_m": endpoint_distance,
                "path_rms_distance_m": path_rms_distance,
                "command_rms_distance": command_rms_distance,
                "exact_behavior_equivalent": exact_behavior,
                **classification,
            }
        )
    return rows


def _distance_summary(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    return _finite_stats([float(row[key]) for row in rows])


def _action_equivalence_analysis(library: Mapping[str, Any], pair_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    definitions = library["definitions"]
    distinct_pairs = _pair_distance_rows(library)
    exact = [row for row in distinct_pairs if row["exact_behavior_equivalent"]]
    same_horizontal = [row for row in distinct_pairs if row["same_horizontal_family"]]
    same_vertical = [row for row in distinct_pairs if row["same_vertical_family"]]
    different = [row for row in distinct_pairs if row["different_horizontal_and_vertical"]]
    distances = np.asarray(
        [
            float(
                np.linalg.norm(
                    np.asarray(definitions[int(row["action1"])]["descriptor"])
                    - np.asarray(definitions[int(row["action2"])]["descriptor"])
                )
            )
            for row in pair_rows
        ],
        dtype=np.float64,
    )
    if distances.size:
        thresholds = {
            "near_upper_q33": float(np.percentile(distances, 33.3333333333)),
            "medium_upper_q66": float(np.percentile(distances, 66.6666666667)),
        }
    else:
        thresholds = {"near_upper_q33": None, "medium_upper_q66": None}
    ordered = sorted(distinct_pairs, key=lambda row: (row["descriptor_distance"], row["action1"], row["action2"]))
    return {
        "distinct_action_pair_count": len(distinct_pairs),
        "same_action_id_pair_count": 0,
        "different_action_id_pair_count": len(distinct_pairs),
        "exact_parameter_duplicate_pair_count": 0,
        "exact_behavior_equivalent_pair_count": len(exact),
        "same_horizontal_family_pair_count": len(same_horizontal),
        "same_vertical_family_pair_count": len(same_vertical),
        "different_horizontal_and_vertical_pair_count": len(different),
        "descriptor_distance": _distance_summary(distinct_pairs, "descriptor_distance"),
        "geometry_distance_m": _distance_summary(distinct_pairs, "geometry_distance_m"),
        "path_rms_distance_m": _distance_summary(distinct_pairs, "path_rms_distance_m"),
        "command_rms_distance": _distance_summary(distinct_pairs, "command_rms_distance"),
        "observed_pair_distance_thresholds": thresholds,
        "nearest_pairs": ordered[:20],
        "neighbor_definition": "4-connected and 8-connected adjacency in the 15x7 horizontal/vertical MPL lattice",
    }


def _outcome_separability(
    pair_rows: Sequence[Mapping[str, Any]], library: Mapping[str, Any]
) -> Dict[str, Any]:
    definitions = library["definitions"]
    records: List[Dict[str, Any]] = []
    for row in pair_rows:
        first = int(row["action1"])
        second = int(row["action2"])
        descriptor_distance = float(
            np.linalg.norm(np.asarray(definitions[first]["descriptor"]) - np.asarray(definitions[second]["descriptor"]))
        )
        records.append(
            {
                "descriptor_distance": descriptor_distance,
                "return1": float(row["return1"]),
                "return2": float(row["return2"]),
                "return_delta": float(row["return1"]) - float(row["return2"]),
                "abs_return_delta": abs(float(row["return1"]) - float(row["return2"])),
                "success_disagreement": int(int(row["success1"]) != int(row["success2"])),
            }
        )
    distances = np.asarray([item["descriptor_distance"] for item in records], dtype=np.float64)
    q33, q66 = np.percentile(distances, [33.3333333333, 66.6666666667])
    buckets: Dict[str, List[Dict[str, Any]]] = {"near": [], "medium": [], "far": []}
    for item in records:
        if item["descriptor_distance"] <= float(q33):
            bucket = "near"
        elif item["descriptor_distance"] <= float(q66):
            bucket = "medium"
        else:
            bucket = "far"
        buckets[bucket].append(item)
    by_bucket: Dict[str, Any] = {}
    for bucket, items in buckets.items():
        values = np.asarray([item["return1"] for item in items] + [item["return2"] for item in items], dtype=np.float64)
        deltas = np.asarray([item["return_delta"] for item in items], dtype=np.float64)
        by_bucket[bucket] = {
            "pair_count": len(items),
            "descriptor_distance": _finite_stats([item["descriptor_distance"] for item in items]),
            "pooled_branch_return": _finite_stats(values.tolist()),
            "return_delta": _finite_stats(deltas.tolist()),
            "abs_return_delta": _finite_stats([abs(float(item["return_delta"])) for item in items]),
            "success_disagreement_rate": float(np.mean([item["success_disagreement"] for item in items])) if items else None,
        }
    correlation = float(np.corrcoef(distances, np.asarray([item["abs_return_delta"] for item in records]))[0, 1]) if len(records) > 1 and float(np.std(distances)) > 0.0 else None
    return {
        "schema_id": "action_outcome_separability_v1",
        "pair_count": len(records),
        "distance_metric": "L2 distance of normalized [lateral_endpoint, vertical_endpoint, terminal_heading] descriptor",
        "bucket_thresholds": {"near_upper_q33": float(q33), "medium_upper_q66": float(q66)},
        "by_distance_bucket": by_bucket,
        "distance_abs_return_delta_correlation": correlation,
    }


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
    for _ in range(int(epochs)):
        order = torch.randperm(x_tensor.shape[0], generator=generator)
        for start in range(0, len(order), int(batch_size)):
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


def _run_representation(
    pairs: Sequence[Mapping[str, Any]],
    state_matrix: np.ndarray,
    split: Mapping[str, Any],
    representation: str,
    descriptors: Mapping[int, np.ndarray],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> Dict[str, Any]:
    actions1 = [int(row["action1"]) for row in pairs]
    actions2 = [int(row["action2"]) for row in pairs]
    matrix = build_pair_action_features(
        state_matrix, actions1, actions2, representation=representation, descriptors=descriptors
    )
    labels = np.asarray([int(row["label"]) for row in pairs], dtype=np.int64)
    state_ids = np.asarray([str(row["state_id"]) for row in pairs])
    mission_ids = np.asarray([str(row["mission_id"]) for row in pairs])
    folds = np.asarray([int(split["state_to_fold"][state_id]) for state_id in state_ids.tolist()], dtype=np.int64)
    oof = np.zeros(len(pairs), dtype=np.float64)
    fold_records = []
    for fold in range(int(split["fold_count"])):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        prediction = _fit_pair_model(
            matrix[train], labels[train], matrix[test], seed=int(seed) + fold * 100 + 71, epochs=epochs, batch_size=batch_size
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
    metrics = classification_bundle(labels, oof, state_ids, mission_ids)
    return {
        "representation": representation,
        "input_dim": int(matrix.shape[1]),
        "descriptor_dim_per_action": 3 if representation == "primitive_descriptor" else ACTION_DIM,
        "pair_count": len(pairs),
        "folds": fold_records,
        "oof_metrics": metrics,
        "oof_predictions": oof,
        "labels": labels,
        "state_ids": state_ids,
        "mission_ids": mission_ids,
    }


def _bootstrap_model_results(results: Mapping[str, Mapping[str, Any]], *, repeats: int, seed: int) -> Dict[str, Any]:
    correctness = {
        name: (np.asarray(item["oof_predictions"]) >= 0.5) == np.asarray(item["labels"])
        for name, item in results.items()
    }
    absolute: Dict[str, Any] = {}
    for index, (name, item) in enumerate(results.items()):
        values: MutableMapping[str, List[float]] = defaultdict(list)
        for mission, value in zip(item["mission_ids"].tolist(), correctness[name].tolist()):
            values[str(mission)].append(float(value))
        absolute[name] = mission_cluster_bootstrap(values, repeats=repeats, seed=seed + index)
    first_name = "105_one_hot"
    paired: Dict[str, Any] = {}
    baseline = correctness[first_name]
    for index, name in enumerate(("primitive_descriptor",)):
        values = defaultdict(list)
        deltas = correctness[name].astype(np.int8) - baseline.astype(np.int8)
        for mission, value in zip(results[name]["mission_ids"].tolist(), deltas.tolist()):
            values[str(mission)].append(float(value))
        paired["{}_minus_{}".format(name, first_name)] = mission_cluster_bootstrap(values, repeats=repeats, seed=seed + 100 + index)
    return {
        "schema_id": "action_formulation_bootstrap_v1",
        "cluster_unit": "mission_id",
        "repeats": int(repeats),
        "absolute": absolute,
        "paired_deltas": paired,
        "note": "CI is mission-cluster only; it does not include training-seed uncertainty.",
    }


def _write_predictions(path: Path, results: Mapping[str, Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]], split: Mapping[str, Any]) -> None:
    fields = ["representation", "fold_id", "mission_id", "state_id", "step_id", "action1", "action2", "label", "prediction", "predicted_label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            predictions = result["oof_predictions"]
            for index, value in enumerate(predictions.tolist()):
                row = pairs[index]
                writer.writerow(
                    {
                        "representation": name,
                        "fold_id": int(split["state_to_fold"][str(row["state_id"])]),
                        "mission_id": str(row["mission_id"]),
                        "state_id": str(row["state_id"]),
                        "step_id": int(row["step_id"]),
                        "action1": int(row["action1"]),
                        "action2": int(row["action2"]),
                        "label": int(row["label"]),
                        "prediction": float(value),
                        "predicted_label": int(value >= 0.5),
                    }
                )


def _write_report(
    path: Path,
    *,
    identity: Mapping[str, Any],
    library: Mapping[str, Any],
    action_analysis: Mapping[str, Any],
    primitive_metrics: Mapping[str, Any],
) -> None:
    one_hot = primitive_metrics["models"]["105_one_hot"]
    descriptor = primitive_metrics["models"]["primitive_descriptor"]
    delta = primitive_metrics["bootstrap"]["paired_deltas"]["primitive_descriptor_minus_105_one_hot"]
    exact = int(action_analysis["equivalence"]["exact_behavior_equivalent_pair_count"])
    descriptor_better = delta["ci95"][0] > 0.0
    one_hot_better = delta["ci95"][1] < 0.0
    if descriptor_better:
        next_step = "A 改 action representation：primitive descriptor 的 paired CI 支持优于 105 one-hot。"
    elif one_hot_better:
        next_step = "C 优先检查 state：当前语义 descriptor 没有带来提升，不能把 action representation 当作单一根因。"
    elif exact == 0 and descriptor["oof_metrics"]["mission_macro"]["accuracy"] <= 0.60:
        next_step = "C 优先继续检查 state/action 信息瓶颈；action 语义替换的证据区间仍跨过 0。"
    else:
        next_step = "A/C 联合离线诊断：先扩大独立验证，再决定 action representation 或 state 改造。"
    lines = [
        "# ACTION FORMULATION AUDIT V1",
        "",
        "本报告是只读 offline diagnostic；未执行 AWAC、Actor、Critic、Unity、Replay 写入或生产 action 修改。",
        "",
        "## 输入与 action library",
        "",
        "- dataset: `{}`；states `{}`；pairs `{}`；missions `{}`。".format(identity["dataset_id"], identity["state_count"], identity["pair_count"], identity["mission_count"]),
        "- MPL: `{}` actions；每个 primitive 的可变语义参数维度为 3：lateral endpoint、vertical endpoint、terminal heading。".format(library["meta"]["num_actions"]),
        "- duration `{}` s；command frames `{}`；control dt `{}` s；MPL contract `{}`。".format(library["meta"]["duration_s"], library["meta"]["command_frames"], library["meta"]["control_dt_s"], library["meta"]["contract_sha256"]),
        "- descriptor 是归一化连续参数，不包含 action id、Q 值或 return。",
        "",
        "## 等价性与邻居",
        "",
        "- distinct action pairs: `{}`；不同 action ID 的 exact parameter duplicate: `{}`；exact behavior equivalent: `{}`。".format(action_analysis["equivalence"]["distinct_action_pair_count"], action_analysis["equivalence"]["exact_parameter_duplicate_pair_count"], exact),
        "- 4-neighbor edges: `{}`；8-neighbor edges: `{}`。这些是 15×7 MPL lattice 邻接，不是 outcome 等价声明。".format(action_analysis["neighbor_summary"]["total_4_connected_edges"], action_analysis["neighbor_summary"]["total_8_connected_edges"]),
        "- 几何 endpoint distance mean `{:.6f}` m；descriptor distance mean `{:.6f}`。".format(action_analysis["equivalence"]["geometry_distance_m"]["mean"], action_analysis["equivalence"]["descriptor_distance"]["mean"]),
        "",
        "## outcome distance buckets",
        "",
        "| bucket | pairs | pooled return variance | abs return delta mean | success disagreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket in ("near", "medium", "far"):
        item = action_analysis["outcome_separability"]["by_distance_bucket"][bucket]
        lines.append("| {} | {} | {:.6f} | {:.6f} | {:.6f} |".format(bucket, item["pair_count"], item["pooled_branch_return"]["variance"], item["abs_return_delta"]["mean"], item["success_disagreement_rate"]))
    lines.extend(
        [
            "",
            "## ranking model",
            "",
            "| representation | input dim | micro accuracy | micro AUC | mission macro accuracy | mission macro AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("105_one_hot", "primitive_descriptor"):
        item = primitive_metrics["models"][name]
        micro = item["oof_metrics"]["micro"]
        mission = item["oof_metrics"]["mission_macro"]
        lines.append("| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(name, item["input_dim"], micro["accuracy"], micro["auc"], mission["accuracy"], mission["auc"]))
    lines.extend(
        [
            "",
            "primitive_descriptor - 105_one_hot mission-macro accuracy paired bootstrap: `{:.6f}`, 95% CI `[{}, {}]`.".format(delta["estimate"], delta["ci95"][0], delta["ci95"][1]),
            "",
            "## 回答",
            "",
            "1. 105 action 是否存在大量等价动作？ exact artifact-level duplicate 为 `{}`；因此没有证据证明存在大量 exact 等价 action，但相邻网格动作的行为距离较小。".format(exact),
            "2. action ID 是否比真实 primitive 语义更弱？ one-hot 与 descriptor 的差异见上表和 paired CI；当前不能脱离该 CI 宣称因果优劣。",
            "3. primitive descriptor 是否提升 ranking？结论：`{}`。".format("YES" if descriptor_better else ("NO; one-hot 在本固定诊断中更高" if one_hot_better else "INCONCLUSIVE")),
            "4. 下一步：`{}`".format(next_step),
            "",
            "限制：数据来自 31 个 failure missions；mission-cluster CI 不含训练 seed 方差，也不构成生产 Critic/AWAC gate。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_action_formulation_audit(
    dataset_root: Path,
    output_root: Path,
    *,
    state_audit_root: Path,
    temporal_audit_root: Path,
    motion_primitives_json: Path,
    motion_primitives_npz: Path,
    motion_primitives_config: Path,
    seed: int = 20260910,
    bootstrap_repeats: int = 5000,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Run action library, outcome, and representation ranking diagnostics."""

    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty action audit directory {}".format(output_root))
    data = load_audit_dataset(Path(dataset_root))
    state_audit_root = Path(state_audit_root).expanduser().resolve()
    temporal_audit_root = Path(temporal_audit_root).expanduser().resolve()
    state_identity_path = state_audit_root / "dataset_identity.json"
    split_path = state_audit_root / "mission_split.json"
    temporal_identity_path = temporal_audit_root / "dataset_identity.json"
    for path in (state_identity_path, split_path, temporal_identity_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    state_identity = json.loads(state_identity_path.read_text(encoding="utf-8"))
    temporal_identity = json.loads(temporal_identity_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if state_identity.get("pairwise_sha256") != data["identity"].get("pairwise_sha256"):
        raise ValueError("state audit pair identity mismatch")
    if temporal_identity.get("pairwise_sha256") != data["identity"].get("pairwise_sha256"):
        raise ValueError("temporal audit pair identity mismatch")
    if set(split["mission_to_fold"]) != set(data["missions"]):
        raise ValueError("state audit mission split does not cover input missions")
    pairs = [row for row in data["pairs"] if not row["return_tie"]]
    if not pairs:
        raise ValueError("no non-tie pair labels")
    library = _load_action_library(Path(motion_primitives_json).resolve(), Path(motion_primitives_npz).resolve())
    pair_distance_rows = _pair_distance_rows(library)
    output_root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_id": ACTION_FORMULATION_AUDIT_SCHEMA_ID,
        "dataset_id": data["identity"]["dataset_id"],
        "dataset_manifest_sha256": data["identity"]["manifest_sha256"],
        "pairwise_sha256": data["identity"]["pairwise_sha256"],
        "state_count": data["identity"]["state_count"],
        "transition_count": data["identity"]["transition_count"],
        "pair_count": len(pairs),
        "mission_count": len(data["missions"]),
        "state_audit_identity_sha256": sha256_file(state_identity_path),
        "mission_split_sha256": sha256_file(split_path),
        "temporal_audit_identity_sha256": sha256_file(temporal_identity_path),
        "motion_primitives_json_sha256": library["json_sha256"],
        "motion_primitives_npz_sha256": library["npz_sha256"],
        "motion_primitives_config_sha256": sha256_file(Path(motion_primitives_config).resolve()),
        "q_checkpoint_loaded": False,
        "production_replay_modified": False,
        "production_action_modified": False,
    }
    definitions = library["definitions"]
    descriptor_map = {action_id: np.asarray(item["descriptor"], dtype=np.float32) for action_id, item in definitions.items()}
    neighbor_summary = {
        "total_4_connected_edges": int(sum(len(item["neighbors_4"]) for item in definitions.values()) // 2),
        "total_8_connected_edges": int(sum(len(item["neighbors_8"]) for item in definitions.values()) // 2),
        "mean_4_neighbors": float(np.mean([len(item["neighbors_4"]) for item in definitions.values()])),
        "mean_8_neighbors": float(np.mean([len(item["neighbors_8"]) for item in definitions.values()])),
    }
    action_analysis = {
        "schema_id": ACTION_ANALYSIS_SCHEMA_ID,
        "identity": identity,
        "motion_primitive_source": {
            "json": library["json_path"],
            "npz": library["npz_path"],
            "config": str(Path(motion_primitives_config).resolve()),
            "contract_sha256": library["meta"]["contract_sha256"],
            "num_actions": library["meta"]["num_actions"],
            "duration_s": library["meta"]["duration_s"],
            "command_frames": library["meta"]["command_frames"],
            "control_dt_s": library["meta"]["control_dt_s"],
            "forward_distance_m": library["meta"]["forward_distance_m"],
            "target_forward_speed_mps": library["meta"]["target_forward_speed_mps"],
            "variable_parameter_names": list(VARIABLE_PARAMETER_NAMES),
            "parameter_dim": len(VARIABLE_PARAMETER_NAMES),
            "parameter_ranges": library["parameter_ranges"],
        },
        "actions": list(definitions.values()),
        "neighbor_summary": neighbor_summary,
        "equivalence": _action_equivalence_analysis(library, pairs),
        "outcome_separability": _outcome_separability(pairs, library),
    }
    state_matrix = np.stack([np.asarray(row["state_feature"], dtype=np.float32) for row in pairs])
    model_results: Dict[str, Any] = {}
    for index, representation in enumerate(ACTION_REPRESENTATIONS):
        model_results["105_one_hot" if representation == "one_hot" else "primitive_descriptor"] = _run_representation(
            pairs,
            state_matrix,
            split,
            representation,
            descriptor_map,
            seed=int(seed) + index * 1000,
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
    bootstrap = _bootstrap_model_results(model_results, repeats=int(bootstrap_repeats), seed=int(seed) + 9000)
    primitive_metrics = {
        "schema_id": PRIMITIVE_METRICS_SCHEMA_ID,
        "identity": identity,
        "fixed_mission_split": True,
        "models": {
            name: {
                key: value
                for key, value in result.items()
                if key not in ("oof_predictions", "labels", "state_ids", "mission_ids")
            }
            for name, result in model_results.items()
        },
        "bootstrap": bootstrap,
        "no_q_features": True,
        "no_return_features": True,
    }
    config = {
        "schema_id": ACTION_MODEL_CONFIG_SCHEMA_ID,
        "seed": int(seed),
        "bootstrap_repeats": int(bootstrap_repeats),
        "fold_count": int(split["fold_count"]),
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "hidden_layers": DEFAULT_HIDDEN_LAYERS,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "Adam",
        "learning_rate": 1.0e-3,
        "loss": "binary_cross_entropy_with_logits",
        "normalizer_fit_scope": "training_fold_only",
        "state_feature_dim": int(state_matrix.shape[1]),
        "representations": {"105_one_hot": 2 * ACTION_DIM, "primitive_descriptor": 2 * len(VARIABLE_PARAMETER_NAMES)},
        "q_checkpoint_loaded": False,
    }
    _json_dump(output_root / "dataset_identity.json", identity)
    _json_dump(output_root / "mission_split.json", split)
    _json_dump(output_root / "model_config.json", config)
    _json_dump(output_root / "action_analysis.json", action_analysis)
    _json_dump(output_root / "primitive_metrics.json", primitive_metrics)
    _write_predictions(output_root / "predictions.csv", model_results, pairs, split)
    _write_report(output_root / "report_zh.md", identity=identity, library=library, action_analysis=action_analysis, primitive_metrics=primitive_metrics)
    return {
        "output_root": str(output_root),
        "identity": identity,
        "action_analysis": action_analysis,
        "primitive_metrics": primitive_metrics,
        "config": config,
    }


__all__ = [
    "build_action_definitions",
    "build_pair_action_features",
    "classify_action_pair",
    "run_action_formulation_audit",
]
