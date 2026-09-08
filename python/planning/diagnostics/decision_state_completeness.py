"""Read-only decision-state completeness audit for the diagnostic replay.

This module does not change the production observation contract, replay, model,
or runtime.  It cross-checks the existing observation-information audit with
the immutable multi-action artifact and reports which decision variables are
present, absent, or unidentifiable at the policy boundary.
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from planning.contracts.feature import GOAL_FEATURE_NAMES, STATE_FEATURE_NAMES
from planning.diagnostics.observation_information_audit import build_feature_names
from planning.diagnostics.state_sufficiency import _depth_thumbnail


DECISION_STATE_SCHEMA_ID = "decision_state_completeness_audit_v1"
ACTION_DIM = 105
VECTOR_DIM = len(STATE_FEATURE_NAMES) + len(GOAL_FEATURE_NAMES)
CANONICAL_STATE_DIM = VECTOR_DIM + 9 * 8 + ACTION_DIM + ACTION_DIM


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    root = Path(path).resolve()
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _finite_stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    if not np.isfinite(array).all():
        raise ValueError("non-finite statistic input")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _corr(left: Sequence[float], right: Sequence[float], *, rank: bool = False) -> Optional[float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or y.size != x.size or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    if rank:
        value = spearmanr(x, y).statistic
    else:
        value = np.corrcoef(x, y)[0, 1]
    return float(value) if math.isfinite(float(value)) else None


def build_feature_inventory() -> Dict[str, Any]:
    """Describe every policy-bound state feature and its temporal scope."""

    state_semantics = {
        "position_x": ("geometry", "m", "current absolute position"),
        "position_y": ("geometry", "m", "current absolute position"),
        "position_z": ("geometry", "m", "current absolute position"),
        "velocity_x": ("dynamics", "m/s", "current instantaneous velocity"),
        "velocity_y": ("dynamics", "m/s", "current instantaneous velocity"),
        "velocity_z": ("dynamics", "m/s", "current instantaneous velocity"),
        "yaw_sin": ("geometry", "dimensionless", "current orientation encoding"),
        "yaw_cos": ("geometry", "dimensionless", "current orientation encoding"),
        "z_to_min": ("task progress", "m", "current altitude margin"),
        "z_to_max": ("task progress", "m", "current altitude margin"),
        "goal_distance_xy": ("task progress", "m", "current goal-relative scalar"),
        "goal_dz": ("task progress", "m", "current goal-relative scalar"),
    }
    goal_semantics = {
        "relative_x": ("geometry", "m", "current goal-relative vector"),
        "relative_y": ("geometry", "m", "current goal-relative vector"),
        "relative_z": ("geometry", "m", "current goal-relative vector"),
        "direction_xy_x": ("geometry", "dimensionless", "current normalized goal direction"),
        "direction_xy_y": ("geometry", "dimensionless", "current normalized goal direction"),
        "direction_body_xy_x": ("action context", "dimensionless", "current body-frame goal direction"),
        "direction_body_xy_y": ("action context", "dimensionless", "current body-frame goal direction"),
        "distance_xy": ("task progress", "m", "current goal-relative scalar"),
        "distance_xy_norm40": ("task progress", "dimensionless", "current normalized goal distance"),
        "dz": ("task progress", "m", "current goal-relative scalar"),
    }
    fields: List[Dict[str, Any]] = []
    for index, name in enumerate(STATE_FEATURE_NAMES):
        group, unit, temporal = state_semantics[name]
        fields.append({
            "index": index,
            "name": name,
            "block": "continuous_vector.state",
            "category": group,
            "unit": unit,
            "temporal_attribute": temporal,
            "status": "AVAILABLE_IN_POLICY_STATE",
            "source": "planning.contracts.feature.obs_state_vector",
        })
    for offset, name in enumerate(GOAL_FEATURE_NAMES, len(STATE_FEATURE_NAMES)):
        group, unit, temporal = goal_semantics[name]
        fields.append({
            "index": offset,
            "name": name,
            "block": "continuous_vector.goal",
            "category": group,
            "unit": unit,
            "temporal_attribute": temporal,
            "status": "AVAILABLE_IN_POLICY_STATE",
            "source": "planning.contracts.feature.obs_goal_vector",
        })
    return {
        "continuous_vector_dim": VECTOR_DIM,
        "policy_vector_dim": 127,
        "state_fields": fields,
        "additional_state_fields": [
            {
                "name": "depth",
                "shape": [90, 160],
                "dtype": "float32",
                "unit": "normalized depth in [0,1]",
                "category": "geometry",
                "temporal_attribute": "one current endpoint frame",
                "status": "AVAILABLE_IN_POLICY_STATE",
                "source": "multi_action_replay_v1 states/*.npz:depth",
                "audit_representation": "9x8 mean-pool only for prior diagnostic model",
            },
            {
                "name": "legal_action_mask",
                "shape": [ACTION_DIM],
                "dtype": "bool",
                "unit": "valid-action indicator",
                "category": "action context",
                "temporal_attribute": "current endpoint mask",
                "status": "AVAILABLE_IN_POLICY_STATE",
                "source": "states/*.npz:mask",
            },
            {
                "name": "previous_action",
                "shape": [1],
                "dtype": "int64",
                "unit": "action id or -1 sentinel",
                "category": "history",
                "temporal_attribute": "one-step history only; one-hot in policy vector",
                "status": "AVAILABLE_IN_POLICY_STATE",
                "source": "states/*.npz:previous_action",
            },
        ],
        "identity_fields_not_policy_features": [
            "state_id", "mission_id", "episode_id", "step_id", "route_id"
        ],
        "runtime_fields_excluded_from_policy_vector": [
            {
                "name": "acceleration",
                "shape": [3],
                "unit": "m/s^2",
                "status": "AVAILABLE_IN_RUNTIME_EXCLUDED_FROM_POLICY_STATE",
                "source": "Unity state snapshot and UnityForestEnv._build_observation",
            },
            {
                "name": "min_clearance/front_clearances",
                "unit": "m",
                "status": "AVAILABLE_IN_RUNTIME_EXCLUDED_FROM_POLICY_STATE",
                "source": "Unity state/safety metadata; forbidden privileged policy fields",
            },
        ],
    }


def build_missing_variable_inventory() -> Dict[str, Any]:
    return {
        "variables": [
            {
                "name": "goal_relative_state",
                "status": "AVAILABLE",
                "evidence": "relative xyz, direction_xy, direction_body_xy, distance_xy, dz in continuous vector indices 12:22",
                "scope": "policy state",
            },
            {
                "name": "path_deviation",
                "status": "MISSING",
                "evidence": "no route waypoint, path tangent, or cross-track error in policy vector; route_id is identity metadata only",
                "scope": "policy state",
            },
            {
                "name": "heading_error",
                "status": "AVAILABLE_DERIVED_GOAL_HEADING_PROXY",
                "evidence": "yaw sin/cos plus direction_body_xy provide goal-relative heading proxy; explicit path-heading error is absent",
                "scope": "policy state",
            },
            {
                "name": "acceleration",
                "status": "AVAILABLE_RUNTIME_EXCLUDED_FROM_POLICY_STATE",
                "evidence": "present in reliable runtime state snapshot but absent from 22-dim policy vector and multi-action state NPZ",
                "scope": "runtime only",
            },
            {
                "name": "depth_temporal_change",
                "status": "MISSING",
                "evidence": "artifact has one 90x160 depth frame per state; prior temporal audit reports depth_history_frames=1",
                "scope": "policy state",
            },
            {
                "name": "obstacle_motion",
                "status": "UNKNOWN",
                "evidence": "single depth frame cannot distinguish static geometry from moving obstacles; no obstacle velocity field is persisted",
                "scope": "environment/state identifiability",
            },
            {
                "name": "local_free_space_representation",
                "status": "AVAILABLE_PROXY",
                "evidence": "current depth frame plus 105-action legal mask; no explicit metric free-space geometry is persisted",
                "scope": "policy state",
            },
        ],
        "forbidden_additions": [
            "future state",
            "future reward",
            "termination reason",
            "privileged route/global map",
        ],
    }


def _load_states(root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    state_manifest: Dict[str, Dict[str, Any]] = {}
    with (Path(root) / "state_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                state_manifest[str(row["state_id"])] = row
    states: Dict[str, Dict[str, Any]] = {}
    for path in sorted((Path(root) / "states").glob("*.npz")):
        with np.load(str(path), allow_pickle=False) as loaded:
            state_id = _decode_text(np.asarray(loaded["state_id"]).reshape(-1)[0])
            vector = np.asarray(loaded["vector"], dtype=np.float32).reshape(-1)
            depth = np.asarray(loaded["depth"], dtype=np.float32)
            mask = np.asarray(loaded["mask"], dtype=np.float32).reshape(-1)
            previous_action = int(np.asarray(loaded["previous_action"]).reshape(-1)[0])
        if vector.shape != (VECTOR_DIM,):
            raise ValueError("state {} vector shape {} != {}".format(state_id, vector.shape, VECTOR_DIM))
        if depth.shape != (90, 160):
            raise ValueError("state {} depth shape {} != (90,160)".format(state_id, depth.shape))
        if mask.shape != (ACTION_DIM,):
            raise ValueError("state {} mask shape {} != {}".format(state_id, mask.shape, ACTION_DIM))
        if previous_action < -1 or previous_action >= ACTION_DIM:
            raise ValueError("state {} previous action outside contract".format(state_id))
        if not np.isfinite(vector).all() or not np.isfinite(depth).all() or not np.isfinite(mask).all():
            raise ValueError("state {} contains non-finite values".format(state_id))
        previous_one_hot = np.zeros(ACTION_DIM, dtype=np.float32)
        if previous_action >= 0:
            previous_one_hot[previous_action] = 1.0
        feature = np.concatenate(
            (vector, _depth_thumbnail(depth).reshape(-1), mask, previous_one_hot), axis=0
        ).astype(np.float32)
        if feature.shape != (CANONICAL_STATE_DIM,):
            raise ValueError("state {} canonical feature shape mismatch".format(state_id))
        states[state_id] = {
            "state_id": state_id,
            "vector": vector,
            "depth": depth,
            "mask": mask,
            "previous_action": previous_action,
            "feature": feature,
        }
    if not states:
        raise ValueError("multi-action state artifact is empty")
    missing = sorted(set(states).difference(state_manifest))
    if missing:
        raise ValueError("state manifest missing {} state identities".format(len(missing)))
    return states, state_manifest


def _load_branches(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with (Path(root) / "candidate_branches.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("candidate branch artifact is empty")
    return rows


def _load_transitions(root: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    path = Path(root) / "replay" / "transitions.npz"
    with np.load(str(path), allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    required = {"state_id", "action", "state_vector", "next_state_vector", "next_state_depth", "next_mask"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError("multi-action transitions missing {}".format(missing))
    result: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for index in range(int(arrays["action"].shape[0])):
        key = (_decode_text(arrays["state_id"][index]), int(arrays["action"][index]))
        if key in result:
            raise ValueError("duplicate transition identity {}".format(key))
        result[key] = {
            "state_vector": np.asarray(arrays["state_vector"][index], dtype=np.float32),
            "next_state_vector": np.asarray(arrays["next_state_vector"][index], dtype=np.float32),
            "next_state_depth": np.asarray(arrays["next_state_depth"][index], dtype=np.float32),
            "next_mask": np.asarray(arrays["next_mask"][index], dtype=np.bool_),
        }
    return result


def _load_pairs(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with (Path(root) / "multi_action_pairwise_dataset.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            row["action1"] = int(row["action1"])
            row["action2"] = int(row["action2"])
            row["return1"] = float(row["return1"])
            row["return2"] = float(row["return2"])
            row["return_tie"] = str(row["return_tie"]).strip().lower() in {"1", "true"}
            rows.append(row)
    return rows


def _load_original_failure_types(root: Path) -> Dict[str, str]:
    provenance_path = Path(root) / "collection_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest = json.loads((Path(root) / "dataset_manifest.json").read_text(encoding="utf-8"))
    raw = provenance.get("failure_state_source", {}).get("source_bc_index", "")
    if not raw:
        raw = manifest.get("failure_state_source", {}).get("source_bc_index", "")
    path = Path(str(raw))
    if not path.is_file():
        return {}
    result: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[str(row["episode_id"])] = str(row.get("stop_reason", "unknown"))
    return result


def _state_outcomes(
    states: Mapping[str, Mapping[str, Any]],
    state_manifest: Mapping[str, Mapping[str, Any]],
    branches: Sequence[Mapping[str, Any]],
    original_failure_types: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in branches:
        grouped[str(row["state_id"])].append(row)
    rows: List[Dict[str, Any]] = []
    by_state: Dict[str, Dict[str, Any]] = {}
    for state_id in sorted(states):
        entries = grouped.get(state_id, [])
        if not entries:
            raise ValueError("state {} has no candidate branches".format(state_id))
        reasons = Counter(str(row.get("terminal_reason", "unknown")) for row in entries)
        source = state_manifest[state_id]
        failure_type = original_failure_types.get(str(source.get("source_episode_id", "")), "unknown")
        summary = {
            "state_id": state_id,
            "mission_id": str(source["mission_id"]),
            "step_id": int(source["step_id"]),
            "source_episode_id": str(source.get("source_episode_id", "")),
            "original_failure_type": failure_type,
            "branch_count": len(entries),
            "success_count": int(sum(bool(row.get("success", False)) for row in entries)),
            "collision_count": int(sum(bool(row.get("collision", False)) for row in entries)),
            "dead_end_count": int(sum(bool(row.get("dead_end", False)) for row in entries)),
            "timeout_count": int(sum(bool(row.get("timeout", False)) for row in entries)),
            "terminal_reason_counts": dict(sorted(reasons.items())),
            "return_statistics": _finite_stats([float(row["episode_return"]) for row in entries]),
        }
        summary.update(
            {
                "has_success_branch": summary["success_count"] > 0,
                "has_collision_branch": summary["collision_count"] > 0,
                "has_dead_end_branch": summary["dead_end_count"] > 0,
                "branch_outcome_type_count": len(reasons),
            }
        )
        rows.append(summary)
        by_state[state_id] = summary
    return rows, by_state


def _binary_metrics(y_true: np.ndarray, probability: np.ndarray, missions: Sequence[str]) -> Dict[str, Any]:
    prediction = probability >= 0.5
    result: Dict[str, Any] = {
        "count": int(y_true.size),
        "positive_count": int(y_true.sum()),
        "accuracy": float(np.mean(prediction == y_true)) if y_true.size else None,
        "auc": None,
    }
    if np.unique(y_true).size == 2:
        result["auc"] = float(roc_auc_score(y_true, probability))
    grouped: MutableMapping[str, List[int]] = defaultdict(list)
    for index, mission in enumerate(missions):
        grouped[str(mission)].append(index)
    mission_accuracy = [float(np.mean(prediction[indexes] == y_true[indexes])) for indexes in grouped.values()]
    mission_auc = [
        float(roc_auc_score(y_true[indexes], probability[indexes]))
        for indexes in grouped.values()
        if np.unique(y_true[indexes]).size == 2
    ]
    result["mission_macro_accuracy"] = float(np.mean(mission_accuracy)) if mission_accuracy else None
    result["mission_macro_auc"] = float(np.mean(mission_auc)) if mission_auc else None
    return result


def _fit_state_models(
    state_rows: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
    mission_to_fold: Mapping[str, int],
) -> Dict[str, Any]:
    state_ids = [str(row["state_id"]) for row in state_rows]
    missions = [str(row["mission_id"]) for row in state_rows]
    features = np.stack([np.asarray(states[state_id]["feature"], dtype=np.float64) for state_id in state_ids])
    targets = {
        "state_has_success_branch": np.asarray([bool(row["has_success_branch"]) for row in state_rows], dtype=np.int64),
        "state_has_collision_branch": np.asarray([bool(row["has_collision_branch"]) for row in state_rows], dtype=np.int64),
        "state_has_dead_end_branch": np.asarray([bool(row["has_dead_end_branch"]) for row in state_rows], dtype=np.int64),
        "original_failure_is_collision": np.asarray(
            [str(row["original_failure_type"]) == "collision" for row in state_rows], dtype=np.int64
        ),
    }
    feature_names = build_feature_names("canonical")[:CANONICAL_STATE_DIM]
    output: Dict[str, Any] = {
        "model": "state-only standardized logistic regression; diagnostic only",
        "input": "canonical state features only; action id excluded",
        "mission_split": "reused observation_information_audit_v1 mission_to_fold",
        "targets": {},
        "predictions": [],
    }
    for target_name, labels in targets.items():
        oof = np.full(labels.shape, np.nan, dtype=np.float64)
        fold_results = []
        coefficient_rows: List[np.ndarray] = []
        for fold_id in sorted(set(int(value) for value in mission_to_fold.values())):
            test = np.asarray([mission_to_fold[mission] == fold_id for mission in missions], dtype=bool)
            train = ~test
            if not train.any() or not test.any():
                continue
            if np.unique(labels[train]).size < 2:
                probability = np.full(int(test.sum()), float(np.mean(labels[train])), dtype=np.float64)
                coefficients = np.zeros(features.shape[1], dtype=np.float64)
            else:
                scaler = StandardScaler()
                train_features = scaler.fit_transform(features[train])
                test_features = scaler.transform(features[test])
                model = LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=20260908,
                    solver="lbfgs",
                )
                model.fit(train_features, labels[train])
                probability = model.predict_proba(test_features)[:, 1]
                coefficients = np.asarray(model.coef_[0], dtype=np.float64)
            oof[test] = probability
            coefficient_rows.append(coefficients)
            fold_results.append(
                {
                    "fold_id": fold_id,
                    "train_count": int(train.sum()),
                    "test_count": int(test.sum()),
                    "train_positive_count": int(labels[train].sum()),
                    "test_metrics": _binary_metrics(labels[test], probability, [missions[index] for index in np.flatnonzero(test)]),
                }
            )
        valid = np.isfinite(oof)
        if not valid.all():
            raise ValueError("OOF prediction missing for target {}".format(target_name))
        mean_abs_coef = np.mean(np.abs(np.stack(coefficient_rows)), axis=0) if coefficient_rows else np.zeros(features.shape[1])
        order = np.argsort(-mean_abs_coef)
        output["targets"][target_name] = {
            "label_semantics": (
                "state has at least one recorded branch with this outcome"
                if target_name.startswith("state_has_")
                else "original BC failure type from source rollout index"
            ),
            "positive_count": int(labels.sum()),
            "negative_count": int(labels.size - labels.sum()),
            "majority_baseline_accuracy": float(
                max(int(labels.sum()), int(labels.size - labels.sum())) / labels.size
            ),
            "folds": fold_results,
            "oof_metrics": _binary_metrics(labels, oof, missions),
            "top_standardized_logistic_features": [
                {
                    "feature": feature_names[int(index)],
                    "feature_index": int(index),
                    "mean_abs_coefficient": float(mean_abs_coef[index]),
                }
                for index in order[:20]
            ],
        }
        for index, state_id in enumerate(state_ids):
            output["predictions"].append(
                {
                    "state_id": state_id,
                    "mission_id": missions[index],
                    "target": target_name,
                    "label": int(labels[index]),
                    "oof_probability": float(oof[index]),
                    "fold_id": int(mission_to_fold[missions[index]]),
                }
            )
    output["state_feature_dim"] = int(features.shape[1])
    return output


def _pair_analysis(
    pairs: Sequence[Mapping[str, Any]],
    transitions: Mapping[Tuple[str, int], Mapping[str, Any]],
    vector_names: Sequence[str],
) -> Dict[str, Any]:
    non_ties = [row for row in pairs if not bool(row["return_tie"])]
    deltas: List[np.ndarray] = []
    current_deltas: List[np.ndarray] = []
    return_differences: List[float] = []
    next_depth_differences: List[float] = []
    next_mask_differences: List[int] = []
    high_reasons: Counter[str] = Counter()
    low_reasons: Counter[str] = Counter()
    missing = 0
    for row in non_ties:
        key1 = (str(row["state_id"]), int(row["action1"]))
        key2 = (str(row["state_id"]), int(row["action2"]))
        if key1 not in transitions or key2 not in transitions:
            missing += 1
            continue
        if float(row["return1"]) >= float(row["return2"]):
            high, low = transitions[key1], transitions[key2]
            high_reason, low_reason = str(row.get("terminal_reason1", "unknown")), str(row.get("terminal_reason2", "unknown"))
        else:
            high, low = transitions[key2], transitions[key1]
            high_reason, low_reason = str(row.get("terminal_reason2", "unknown")), str(row.get("terminal_reason1", "unknown"))
        current_deltas.append(np.asarray(high["state_vector"], dtype=np.float64) - np.asarray(low["state_vector"], dtype=np.float64))
        deltas.append(np.asarray(high["next_state_vector"], dtype=np.float64) - np.asarray(low["next_state_vector"], dtype=np.float64))
        next_depth_differences.append(float(np.mean(np.abs(np.asarray(high["next_state_depth"]) - np.asarray(low["next_state_depth"])))) )
        next_mask_differences.append(int(np.count_nonzero(np.asarray(high["next_mask"]) != np.asarray(low["next_mask"]))))
        return_differences.append(abs(float(row["return1"]) - float(row["return2"])))
        high_reasons[high_reason] += 1
        low_reasons[low_reason] += 1
    if not deltas:
        raise ValueError("no pair transitions could be aligned")
    delta = np.stack(deltas)
    current_delta = np.stack(current_deltas)
    return_difference = np.asarray(return_differences, dtype=np.float64)
    feature_stats = []
    for index, name in enumerate(vector_names):
        values = delta[:, index]
        feature_stats.append(
            {
                "feature": name,
                "feature_index": index,
                "mean_signed_next_delta_high_minus_low": float(np.mean(values)),
                "mean_abs_next_delta_high_minus_low": float(np.mean(np.abs(values))),
                "p90_abs_next_delta_high_minus_low": float(np.percentile(np.abs(values), 90)),
                "return_difference_pearson": _corr(values, return_difference),
                "return_difference_spearman": _corr(values, return_difference, rank=True),
            }
        )
    feature_stats.sort(key=lambda row: -float(row["mean_abs_next_delta_high_minus_low"]))
    return {
        "pair_count": len(pairs),
        "non_tie_pair_count": len(non_ties),
        "aligned_pair_count": int(len(deltas)),
        "missing_transition_alignment_count": int(missing),
        "current_state_delta": {
            "max_abs": float(np.max(np.abs(current_delta))),
            "mean_abs": float(np.mean(np.abs(current_delta))),
            "interpretation": "same-state current features are identical by construction; no current-state delta separates the two actions",
        },
        "post_action_next_state_delta": {
            "vector_feature_statistics_sorted_by_mean_abs": feature_stats,
            "mean_abs_depth_delta": _finite_stats(next_depth_differences),
            "next_mask_hamming": _finite_stats(next_mask_differences),
            "interpretation": "next-state changes are post-action outcomes, not variables available at decision time",
        },
        "terminal_reason_counts": {
            "higher_return_branch": dict(sorted(high_reasons.items())),
            "lower_return_branch": dict(sorted(low_reasons.items())),
        },
    }


def _load_prior_feature_importance(path: Path) -> Dict[str, Any]:
    if not Path(path).is_file():
        return {"status": "UNAVAILABLE"}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = [row for row in payload.get("features", []) if row.get("feature_block") in {"vector", "depth", "legal_mask", "previous_action"}]
    return {
        "status": "AVAILABLE_READ_ONLY_PRIOR_PAIRWISE_AUDIT",
        "top_features": features[:20],
        "block_summary": payload.get("block_summary", {}),
        "source_sha256": _sha256_file(path),
    }


def _write_predictions(path: Path, predictions: Sequence[Mapping[str, Any]]) -> None:
    fields = ["state_id", "mission_id", "target", "label", "oof_probability", "fold_id"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    inventory = result["feature_inventory"]
    missing = result["missing_variables"]
    models = result["success_failure_metrics"]
    pairs = result["pair_state_outcome_analysis"]
    prior = result["prior_observation_audit"]
    target_success = models["targets"]["state_has_success_branch"]
    target_collision = models["targets"]["state_has_collision_branch"]
    target_dead_end = models["targets"]["state_has_dead_end_branch"]
    lines = [
        "# DECISION STATE COMPLETENESS AUDIT V1",
        "",
        "只读诊断；未训练 AWAC/Actor/Critic，未启动 Unity/Bridge/ROS，未修改 Replay、reward 或 production observation。",
        "",
        "## 数据边界",
        "",
        "- state count: `{}`; transition count: `{}`; non-tie pair count: `{}`。".format(
            result["counts"]["state_count"], result["counts"]["transition_count"], result["counts"]["non_tie_pair_count"]
        ),
        "- state-only success 标签定义为“该 state 的记录分支中至少一个 success”，不是 action-independent 的任务成功标签。",
        "- 同一 state 的不同 action 可有不同终止结果，因此不能把 state-only 预测当成 action outcome 的充分模型。",
        "",
        "## State feature inventory",
        "",
        "- continuous vector: `{}`；depth: `90x160` 单帧归一化深度；legal mask: `105`；previous action: 单步历史 one-hot。".format(VECTOR_DIM),
        "- policy vector: `127`；depth 单独进入模型；prior audit 的 9x8 depth thumbnail 只属于诊断表示。",
        "",
        "| index/block | field | category | unit | temporal attribute |",
        "|---|---|---|---|---|",
    ]
    for field in inventory["state_fields"]:
        lines.append("| {} | `{}` | {} | {} | {} |".format(field["index"], field["name"], field["category"], field["unit"], field["temporal_attribute"]))
    lines.extend([
        "| depth | `depth` | geometry | normalized [0,1] | one current endpoint frame |",
        "| mask | `legal_action_mask` | action context | valid-action indicator | current endpoint |",
        "| history | `previous_action` | history | action id / -1 | one-step history |",
        "",
        "## 缺失变量审计",
        "",
        "| variable | status | evidence |",
        "|---|---|---|",
    ])
    for item in missing["variables"]:
        lines.append("| `{}` | **{}** | {} |".format(item["name"], item["status"], item["evidence"]))
    lines.extend([
        "",
        "## Success/failure separability",
        "",
        "state-only standardized logistic regression（3 个既有 mission folds；无 action feature）：",
        "",
        "| target | positives | OOF accuracy | OOF AUC | mission macro accuracy |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("state_has_success_branch", "state_has_collision_branch", "state_has_dead_end_branch", "original_failure_is_collision"):
        target = models["targets"][name]
        metric = target["oof_metrics"]
        lines.append("| `{}` | {} | {:.6f} | {} | {:.6f} |".format(
            name,
            target["positive_count"],
            metric["accuracy"],
            "NA" if metric["auc"] is None else "{:.6f}".format(metric["auc"]),
            metric["mission_macro_accuracy"],
        ))
    lines.extend([
        "",
        "success target 的 top state features（标准化 logistic coefficient）：",
    ])
    for item in target_success["top_standardized_logistic_features"][:10]:
        lines.append("- `{}`: {:.6f}".format(item["feature"], item["mean_abs_coefficient"]))
    lines.extend([
        "",
        "prior observation-information audit 的 pairwise feature importance 作为参考：`{}`。".format(prior["status"]),
        "该参考结果不能证明 feature 对 action outcome 具有因果作用。",
        "",
        "## Pair outcome 与 feature change",
        "",
        "- non-tie pairs: `{}`; aligned: `{}`。".format(pairs["non_tie_pair_count"], pairs["aligned_pair_count"]),
        "- 当前 decision state 的 high/low action feature delta max abs: `{:.9f}`；这是同 state 配对的结构性结果。".format(pairs["current_state_delta"]["max_abs"]),
        "- 因此高 reward pair 与当前 state feature 没有可比较的 state 差异；可见的 next-state 差异属于 action 执行后的结果，不能作为决策时输入。",
        "- next-state 差异统计和每个 vector feature 的 return-difference correlation 已写入 `pair_state_outcome_analysis.json`。",
        "",
        "## 结论",
        "",
        "1. 当前 state 是 **PARTIAL_MARKOV_APPROXIMATION / NOT_PROVEN_MARKOV**：它覆盖位置、速度、姿态、目标相对量、单帧局部深度、mask 和 previous action；但不覆盖 path deviation、depth temporal change，acceleration 也未进入 policy vector，obstacle motion 为 UNKNOWN。",
        "2. 最可能影响 Q ranking 的缺口是 **短时动态上下文**（acceleration 与 depth temporal change）；既有 temporal audit 的 mission-macro accuracy 从 current-state `0.519707` 到 temporal-state `0.542217`，只能作为支持性相关证据，不能视为因果证明。",
        "3. 最小新增 observation 建议（未实施）：当前 acceleration 三维 + 一帧过去 depth 的低分辨率差分 + normalized current step/budget；全部只使用过去/当前信息，不加入未来终止原因、future reward、route/map privileged 信息。",
        "4. path deviation 仍是明确缺口，但当前 artifact 没有 route geometry 对照，不能把它排在 temporal dynamics 之前。",
        "",
        "`DECISION_STATE_COMPLETENESS_RESULT = DIAGNOSTIC_ONLY`",
        "`AWAC_NEXT_DECISION = NO_RL_ACTION`",
        "",
        "## 运行边界",
        "",
        "`awac_training_executed=false`, `actor_updates=0`, `critic_updates=0`, `unity_started=false`, `bridge_started=false`, `production_replay_modified=false`。",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_decision_state_completeness_audit(
    *,
    observation_audit_root: Path,
    reward_audit_root: Path,
    multi_action_root: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    observation_audit_root = Path(observation_audit_root).resolve()
    reward_audit_root = Path(reward_audit_root).resolve()
    multi_action_root = Path(multi_action_root).resolve()
    out_dir = Path(out_dir).resolve()
    states, state_manifest = _load_states(multi_action_root)
    branches = _load_branches(multi_action_root)
    transitions = _load_transitions(multi_action_root)
    pairs = _load_pairs(multi_action_root)
    original_failure_types = _load_original_failure_types(multi_action_root)
    state_rows, state_outcomes = _state_outcomes(states, state_manifest, branches, original_failure_types)

    split_path = observation_audit_root / "mission_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    mission_to_fold = {str(key): int(value) for key, value in split["mission_to_fold"].items()}
    missing_missions = sorted({str(row["mission_id"]) for row in state_rows}.difference(mission_to_fold))
    if missing_missions:
        raise ValueError("mission split missing {} missions".format(len(missing_missions)))
    success_failure_metrics = _fit_state_models(state_rows, states, mission_to_fold)
    pair_state_outcome_analysis = _pair_analysis(pairs, transitions, list(STATE_FEATURE_NAMES) + list(GOAL_FEATURE_NAMES))
    feature_inventory = build_feature_inventory()
    missing_variables = build_missing_variable_inventory()
    prior_feature_importance = _load_prior_feature_importance(observation_audit_root / "feature_importance.json")
    reward_provenance = json.loads((reward_audit_root / "reward_provenance.json").read_text(encoding="utf-8"))
    reward_metrics = json.loads((reward_audit_root / "credit_metrics.json").read_text(encoding="utf-8"))
    multi_manifest = json.loads((multi_action_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = success_failure_metrics.pop("predictions")
    _write_predictions(out_dir / "state_outcome_predictions.csv", predictions)
    _write_json(out_dir / "feature_inventory.json", feature_inventory)
    _write_json(out_dir / "missing_variables.json", missing_variables)
    _write_json(out_dir / "success_failure_metrics.json", success_failure_metrics)
    _write_json(out_dir / "pair_state_outcome_analysis.json", pair_state_outcome_analysis)
    result: Dict[str, Any] = {
        "schema_id": DECISION_STATE_SCHEMA_ID,
        "status": "PASS_DIAGNOSTIC_ONLY",
        "out_dir": str(out_dir),
        "counts": {
            "state_count": len(states),
            "transition_count": len(branches),
            "pair_count": len(pairs),
            "non_tie_pair_count": sum(not bool(row["return_tie"]) for row in pairs),
            "mission_count": len({str(row["mission_id"]) for row in state_rows}),
            "multi_outcome_state_count": int(sum(int(row["branch_outcome_type_count"]) > 1 for row in state_rows)),
        },
        "feature_inventory": feature_inventory,
        "missing_variables": missing_variables,
        "success_failure_metrics": success_failure_metrics,
        "pair_state_outcome_analysis": pair_state_outcome_analysis,
        "prior_observation_audit": prior_feature_importance,
        "reward_provenance_input": {
            "audit_status": reward_provenance.get("status"),
            "credit_metrics_status": reward_metrics.get("status"),
            "sequence_recovery": reward_provenance.get("sequence_recovery"),
            "source_sha256": {
                "reward_provenance": _sha256_file(reward_audit_root / "reward_provenance.json"),
                "credit_metrics": _sha256_file(reward_audit_root / "credit_metrics.json"),
            },
        },
        "input_identity": {
            "observation_audit_root": str(observation_audit_root),
            "observation_dataset_identity_sha256": _sha256_file(observation_audit_root / "dataset_identity.json"),
            "observation_feature_importance_sha256": _sha256_file(observation_audit_root / "feature_importance.json"),
            "observation_mission_split_sha256": _sha256_file(split_path),
            "reward_audit_root": str(reward_audit_root),
            "multi_action_root": str(multi_action_root),
            "multi_action_manifest_sha256": _sha256_file(multi_action_root / "dataset_manifest.json"),
            "multi_action_transition_sha256": _sha256_file(multi_action_root / "replay" / "transitions.npz"),
            "multi_action_states_tree_sha256": _sha256_tree(multi_action_root / "states"),
            "mission_split_sha256": _sha256_file(split_path),
        },
        "contracts": {
            "observation_contract": str(multi_manifest.get("observation_contract", "")),
            "task_contract_sha256": str(multi_manifest.get("task_contract_sha256", "")),
            "production_observation_modified": False,
            "production_replay_modified": False,
            "q_features_used": False,
            "return_features_used": False,
        },
        "runtime_boundary": {
            "awac_training_executed": False,
            "actor_updates": 0,
            "critic_updates": 0,
            "unity_started": False,
            "bridge_started": False,
            "ros_started": False,
        },
    }
    _write_json(out_dir / "decision_state_completeness.json", result)
    _write_report(out_dir / "report_zh.md", result)
    manifest = {
        "schema_id": DECISION_STATE_SCHEMA_ID,
        "output_files": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
        "counts": result["counts"],
        "production_modified": False,
        "input_identity": result["input_identity"],
    }
    _write_json(out_dir / "manifest.json", manifest)
    return result


__all__ = [
    "DECISION_STATE_SCHEMA_ID",
    "build_feature_inventory",
    "build_missing_variable_inventory",
    "run_decision_state_completeness_audit",
]
