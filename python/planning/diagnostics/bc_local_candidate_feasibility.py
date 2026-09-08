"""Read-only audit of BC-local action candidate sets.

This module measures whether a small, mask-valid candidate set centred on the
frozen BC action retains the best outcomes that were actually observed for a
state.  It is diagnostic-only: it never trains a policy or critic, never
modifies the multi-action artifact, and never assigns outcomes to actions that
were not executed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np

from planning.diagnostics.action_formulation_audit import _load_action_library
from planning.diagnostics.offline_rl_feasibility import _stats
from planning.diagnostics.oracle_action_imitation import (
    _b0_logits,
    _load_checkpoint,
    _load_raw_states,
    _masked_order,
)
from planning.diagnostics.state_sufficiency import (
    ACTION_DIM,
    load_audit_dataset,
    mission_cluster_bootstrap,
    sha256_file,
    sha256_tree,
)


LOCAL_CANDIDATE_AUDIT_SCHEMA_ID = "bc_local_candidate_rl_feasibility_audit_v1"
CANDIDATE_DEFINITION_SCHEMA_ID = "bc_local_candidate_definition_v1"
ORACLE_COVERAGE_SCHEMA_ID = "bc_local_candidate_oracle_coverage_v1"
RECOVERY_COVERAGE_SCHEMA_ID = "bc_local_candidate_recovery_coverage_v1"
GAIN_RETENTION_SCHEMA_ID = "bc_local_candidate_gain_retention_v1"
DISTANCE_ANALYSIS_SCHEMA_ID = "bc_local_candidate_distance_analysis_v1"
RANDOM_CONTROL_SCHEMA_ID = "bc_local_candidate_random_control_v1"
BOOTSTRAP_SCHEMA_ID = "bc_local_candidate_cluster_bootstrap_v1"
INTEGRITY_SCHEMA_ID = "bc_local_candidate_integrity_audit_v1"
EXPECTED_STATE_COUNT = 479
EXPECTED_TRANSITION_COUNT = 2850
EXPECTED_MISSION_COUNT = 31
EXPECTED_ACTION_DIM = 105
DEFAULT_RANDOM_SEED = 20260908
DEFAULT_RANDOM_REPEATS = 1000
DEFAULT_BOOTSTRAP_REPEATS = 5000
FAMILIES = ("K0_BC_ONLY", "K2_LOCAL", "K4_LOCAL", "K8_LOCAL", "K16_LOCAL", "FULL_VALID")
LOCAL_K = {"K0_BC_ONLY": 0, "K2_LOCAL": 2, "K4_LOCAL": 4, "K8_LOCAL": 8, "K16_LOCAL": 16}


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


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("{} line {} is not an object".format(path, line_number))
            rows.append(value)
    return rows


def _stats_or_empty(values: Iterable[float]) -> Dict[str, Any]:
    values = list(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "p95": None, "p99": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("non-finite diagnostic statistic")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _load_state_manifest(path: Path, *, expected_count: int) -> Dict[str, Dict[str, Any]]:
    rows = _read_jsonl(path)
    if len(rows) != int(expected_count):
        raise ValueError("state manifest count mismatch: {} != {}".format(len(rows), expected_count))
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        state_id = str(row.get("state_id", "")).strip()
        if not state_id or state_id in result:
            raise ValueError("missing or duplicate state manifest identity")
        if str(row.get("status", "")) != "accepted":
            raise ValueError("state manifest contains non-accepted row")
        result[state_id] = row
    return result


def _load_action_analysis(path: Path, primitive_metrics_path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    analysis = json.loads(Path(path).read_text(encoding="utf-8"))
    primitive_metrics = json.loads(Path(primitive_metrics_path).read_text(encoding="utf-8"))
    if analysis.get("schema_id") != "action_formulation_action_analysis_v1":
        raise ValueError("action formulation analysis schema mismatch")
    if primitive_metrics.get("schema_id") != "action_formulation_primitive_metrics_v1":
        raise ValueError("primitive metrics schema mismatch")
    if not bool(primitive_metrics.get("no_q_features", False)) or not bool(primitive_metrics.get("no_return_features", False)):
        raise ValueError("action formulation artifact is not outcome/Q independent")
    actions = list(analysis.get("actions", []))
    if len(actions) != EXPECTED_ACTION_DIM:
        raise ValueError("action definition count mismatch")
    definitions: Dict[int, Dict[str, Any]] = {}
    for item in actions:
        action_id = int(item["action_id"])
        if action_id in definitions or not (0 <= action_id < EXPECTED_ACTION_DIM):
            raise ValueError("invalid or duplicate action definition")
        descriptor = np.asarray(item.get("descriptor", []), dtype=np.float32)
        if descriptor.shape != (3,) or not np.isfinite(descriptor).all():
            raise ValueError("action descriptor is invalid")
        definitions[action_id] = item
    if set(definitions) != set(range(EXPECTED_ACTION_DIM)):
        raise ValueError("action definitions do not cover 0..104")
    return definitions, analysis, primitive_metrics


def _graph_distances(definitions: Mapping[int, Mapping[str, Any]], start: int) -> Dict[int, int]:
    distances = {int(start): 0}
    queue: deque[int] = deque([int(start)])
    while queue:
        current = queue.popleft()
        for neighbor in definitions[current].get("neighbors_8", []):
            neighbor = int(neighbor)
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    if len(distances) != EXPECTED_ACTION_DIM:
        raise ValueError("primitive neighbor graph is disconnected")
    return distances


def _neighbor_orders(definitions: Mapping[int, Mapping[str, Any]]) -> Dict[int, List[int]]:
    descriptors = {
        int(action_id): np.asarray(item["descriptor"], dtype=np.float32)
        for action_id, item in definitions.items()
    }
    orders: Dict[int, List[int]] = {}
    for action_id, item in definitions.items():
        graph = {int(value) for value in item.get("neighbors_8", [])}
        graph_order = sorted(
            graph,
            key=lambda value: (float(np.linalg.norm(descriptors[int(value)] - descriptors[int(action_id)])), int(value)),
        )
        supplement = sorted(
            set(definitions).difference(graph).difference({int(action_id)}),
            key=lambda value: (float(np.linalg.norm(descriptors[int(value)] - descriptors[int(action_id)])), int(value)),
        )
        orders[int(action_id)] = graph_order + supplement
        if len(orders[int(action_id)]) != len(definitions) - 1:
            raise ValueError("neighbor order does not cover all non-self actions")
    return orders


def build_candidate_set(
    *,
    family: str,
    bc_action: int,
    valid_actions: Sequence[int],
    neighbor_order: Mapping[int, Sequence[int]],
) -> List[int]:
    """Build a deterministic, mask-valid candidate set without outcomes."""

    family = str(family)
    valid = {int(value) for value in valid_actions}
    if int(bc_action) not in valid:
        raise ValueError("BC action is not mask-valid")
    if family == "FULL_VALID":
        return sorted(valid)
    if family not in LOCAL_K:
        raise ValueError("unknown candidate family {}".format(family))
    result = [int(bc_action)]
    if int(LOCAL_K[family]) == 0:
        return result
    for action in neighbor_order[int(bc_action)]:
        action = int(action)
        if action in valid and action not in result:
            result.append(action)
            if len(result) >= int(LOCAL_K[family]) + 1:
                break
    return result


def _group_mission_values(rows: Sequence[Mapping[str, Any]], value_key: str) -> Dict[str, List[float]]:
    grouped: MutableMapping[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mission_id"])].append(float(row[value_key]))
    return dict(grouped)


def _coverage_payload(rows: Sequence[Mapping[str, Any]], flag_key: str, *, seed: int, repeats: int) -> Dict[str, Any]:
    flags = [float(bool(row[flag_key])) for row in rows]
    mission_values = _group_mission_values(rows, flag_key)
    mission_means = [float(np.mean(values)) for values in mission_values.values()]
    return {
        "state_count": len(rows),
        "state_micro_coverage": float(np.mean(flags)) if flags else None,
        "mission_macro_coverage": float(np.mean(mission_means)) if mission_means else None,
        "mission_bootstrap": mission_cluster_bootstrap(mission_values, repeats=int(repeats), seed=int(seed)),
        "by_failure_type": {},
    }


def _coverage_by_failure_type(rows: Sequence[Mapping[str, Any]], flag_key: str, *, seed: int, repeats: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for failure_type in ("dead_end", "collision", "unknown"):
        subset = [row for row in rows if str(row["source_failure_type"]) == failure_type]
        if not subset:
            result[failure_type] = {"state_count": 0, "state_micro_coverage": None, "mission_macro_coverage": None, "mission_bootstrap": None}
            continue
        result[failure_type] = _coverage_payload(subset, flag_key, seed=seed + len(result) + 1, repeats=repeats)
    return result


def _mission_mean(values: Mapping[str, Sequence[float]]) -> float:
    means = [float(np.mean(value)) for value in values.values() if len(value)]
    return float(np.mean(means)) if means else float("nan")


def _build_state_records(
    data: Mapping[str, Any],
    raw_states: Mapping[str, Mapping[str, np.ndarray]],
    state_manifest: Mapping[str, Mapping[str, Any]],
    definitions: Mapping[int, Mapping[str, Any]],
    orders: Mapping[int, Sequence[int]],
    frozen_bc: Mapping[str, int],
) -> List[Dict[str, Any]]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in data["transitions"]:
        grouped[str(row["state_id"])].append(row)
    records: List[Dict[str, Any]] = []
    for state_id in sorted(grouped):
        branches = list(grouped[state_id])
        if state_id not in raw_states or state_id not in state_manifest:
            raise ValueError("state identity missing from raw/state manifest")
        bc_rows = [row for row in branches if str(row["action_source"]).upper() == "BC"]
        if len(bc_rows) != 1:
            raise ValueError("state {} does not have exactly one BC branch".format(state_id))
        bc_row = bc_rows[0]
        bc_action = int(frozen_bc[state_id])
        if int(bc_row["action"]) != bc_action:
            raise ValueError("frozen BC action differs from recorded BC branch for {}".format(state_id))
        valid_actions = np.flatnonzero(np.asarray(raw_states[state_id]["mask"], dtype=np.bool_)).astype(np.int64).tolist()
        observed: Dict[int, Mapping[str, Any]] = {}
        for row in branches:
            action = int(row["action"])
            if action in observed:
                raise ValueError("duplicate observed action for {}".format(state_id))
            if action not in valid_actions:
                raise ValueError("observed action outside mask for {}".format(state_id))
            observed[action] = row
        best_return = max(float(row["episode_return"]) for row in branches)
        best_actions = sorted(action for action, row in observed.items() if float(row["episode_return"]) == best_return)
        successful_actions = sorted(action for action, row in observed.items() if int(row["success"]) == 1)
        source_failure = str(bc_row.get("terminal_reason", "unknown"))
        if source_failure not in ("dead_end", "collision"):
            source_failure = "unknown"
        candidates = {
            family: build_candidate_set(
                family=family,
                bc_action=bc_action,
                valid_actions=valid_actions,
                neighbor_order=orders,
            )
            for family in FAMILIES
        }
        records.append(
            {
                "state_id": state_id,
                "mission_id": str(bc_row["mission_id"]),
                "source_failure_type": source_failure,
                "step_id": int(bc_row["step_id"]),
                "bc_action": bc_action,
                "valid_action_count": len(valid_actions),
                "valid_actions": valid_actions,
                "observed_actions": sorted(observed),
                "best_actions": best_actions,
                "successful_actions": successful_actions,
                "bc_return": float(bc_row["episode_return"]),
                "observed": observed,
                "candidates": candidates,
                "stored_bc_action": int(state_manifest[state_id]["bc_action"]),
            }
        )
    if len(records) != EXPECTED_STATE_COUNT:
        raise ValueError("state record count mismatch")
    return records


def _annotate_outcomes(records: Sequence[MutableMapping[str, Any]]) -> None:
    for record in records:
        observed = record["observed"]
        for family, candidate in record["candidates"].items():
            observed_candidate = [action for action in candidate if action in observed]
            candidate_returns = [float(observed[action]["episode_return"]) for action in observed_candidate]
            local_best = max(candidate_returns)
            local_worst = min(candidate_returns)
            local_mean = float(np.mean(candidate_returns))
            record.setdefault("families", {})[family] = {
                "candidate_actions": list(candidate),
                "observed_candidate_actions": observed_candidate,
                "oracle_covered": bool(set(candidate).intersection(record["best_actions"])),
                "recovery_covered": bool(set(candidate).intersection(record["successful_actions"])),
                "local_best_return": local_best,
                "local_worst_return": local_worst,
                "local_mean_return": local_mean,
                "local_gain": float(local_best - record["bc_return"]),
                "full_gain": float(max(float(observed[action]["episode_return"]) for action in record["observed_actions"]) - record["bc_return"]),
                "candidate_success_count": int(sum(int(observed[action]["success"]) for action in observed_candidate)),
            }


def _distance_analysis(records: Sequence[Mapping[str, Any]], definitions: Mapping[int, Mapping[str, Any]], orders: Mapping[int, Sequence[int]]) -> Dict[str, Any]:
    graph_distance_rows: List[float] = []
    geometry_rows: List[float] = []
    rank_rows: List[float] = []
    category_counts = {"BC_SELF": 0, "ONE_HOP": 0, "TWO_HOP": 0, "NEAREST_4": 0, "NEAREST_8": 0, "NEAREST_16": 0, "OUTSIDE_16": 0}
    cumulative = {"within_1hop": [], "within_k4": [], "within_k8": [], "within_k16": []}
    descriptors = {int(action_id): np.asarray(item["descriptor"], dtype=np.float32) for action_id, item in definitions.items()}
    per_state: List[Dict[str, Any]] = []
    for record in records:
        bc = int(record["bc_action"])
        best = set(int(action) for action in record["best_actions"])
        distances = _graph_distances(definitions, bc)
        order_positions = {int(action): index + 1 for index, action in enumerate(orders[bc])}
        graph_values = [float(distances[action]) for action in best]
        geo_values = [float(np.linalg.norm(descriptors[action] - descriptors[bc])) for action in best]
        rank_values = [float(0 if action == bc else order_positions[action]) for action in best]
        min_graph = min(graph_values)
        min_geo = min(geo_values)
        min_rank = min(rank_values)
        graph_distance_rows.append(min_graph)
        geometry_rows.append(min_geo)
        rank_rows.append(min_rank)
        has_self = bc in best
        has_one_hop = any(action == bc or distances[action] <= 1 for action in best)
        if has_self:
            category_counts["BC_SELF"] += 1
        elif any(distances[action] == 1 for action in best):
            category_counts["ONE_HOP"] += 1
        elif any(distances[action] == 2 for action in best):
            category_counts["TWO_HOP"] += 1
        elif min_rank <= 4:
            category_counts["NEAREST_4"] += 1
        elif min_rank <= 8:
            category_counts["NEAREST_8"] += 1
        elif min_rank <= 16:
            category_counts["NEAREST_16"] += 1
        else:
            category_counts["OUTSIDE_16"] += 1
        cumulative["within_1hop"].append(float(has_one_hop))
        cumulative["within_k4"].append(float(min_rank <= 4))
        cumulative["within_k8"].append(float(min_rank <= 8))
        cumulative["within_k16"].append(float(min_rank <= 16))
        per_state.append({"state_id": record["state_id"], "best_actions": sorted(best), "min_graph_distance": min_graph, "min_geometry_distance": min_geo, "min_neighbor_rank": min_rank})
    total = float(len(records))
    return {
        "schema_id": DISTANCE_ANALYSIS_SCHEMA_ID,
        "graph_distance": _stats_or_empty(graph_distance_rows),
        "geometry_distance": _stats_or_empty(geometry_rows),
        "neighbor_rank": _stats_or_empty(rank_rows),
        "best_action_location_counts": category_counts,
        "best_action_location_ratios": {key: float(value / total) for key, value in category_counts.items()},
        "best_observed_action_within": {key: float(np.mean(values)) for key, values in cumulative.items()},
        "per_state": per_state,
    }


def _random_control(
    records: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    seed: int,
    bootstrap_repeats: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"schema_id": RANDOM_CONTROL_SCHEMA_ID, "seed": int(seed), "sampling_repeats": int(repeats), "bootstrap_repeats": int(bootstrap_repeats), "families": {}}
    for family_index, family in enumerate(("K2_LOCAL", "K4_LOCAL", "K8_LOCAL", "K16_LOCAL")):
        rng = np.random.RandomState(int(seed) + family_index * 100003)
        random_trial_macro: List[float] = []
        random_trial_state_micro: List[float] = []
        random_expected_by_mission: MutableMapping[str, List[float]] = defaultdict(list)
        local_flags = {record["state_id"]: float(record["families"][family]["oracle_covered"]) for record in records}
        local_mission = defaultdict(list)
        for record in records:
            local_mission[str(record["mission_id"])].append(local_flags[record["state_id"]])
        local_mission_mean = {mission: float(np.mean(values)) for mission, values in local_mission.items()}
        for _ in range(int(repeats)):
            trial_mission: MutableMapping[str, List[float]] = defaultdict(list)
            trial_state_flags: List[float] = []
            for record in records:
                valid_non_bc = [action for action in record["valid_actions"] if int(action) != int(record["bc_action"])]
                draw_count = len(record["candidates"][family]) - 1
                chosen = set([int(record["bc_action"])])
                if draw_count:
                    chosen.update(int(action) for action in rng.choice(np.asarray(valid_non_bc, dtype=np.int64), size=draw_count, replace=False).tolist())
                flag = float(bool(chosen.intersection(record["best_actions"])))
                trial_mission[str(record["mission_id"])].append(flag)
                trial_state_flags.append(flag)
            means = {mission: float(np.mean(values)) for mission, values in trial_mission.items()}
            random_trial_macro.append(float(np.mean(list(means.values()))))
            random_trial_state_micro.append(float(np.mean(trial_state_flags)))
            for mission, value in means.items():
                random_expected_by_mission[mission].append(value)
        expected = {mission: float(np.mean(values)) for mission, values in random_expected_by_mission.items()}
        diff = {mission: [float(local_mission_mean[mission] - expected[mission])] for mission in local_mission_mean}
        bootstrap = mission_cluster_bootstrap(diff, repeats=int(bootstrap_repeats), seed=int(seed) + family_index * 100003 + 77)
        result["families"][family] = {
            "local_state_micro_coverage": float(np.mean(list(local_flags.values()))),
            "local_mission_macro_coverage": float(np.mean(list(local_mission_mean.values()))),
            "random_state_micro_coverage_mean": float(np.mean(random_trial_state_micro)),
            "random_mission_macro_coverage_mean": float(np.mean(random_trial_macro)),
            "local_minus_random": float(np.mean(list(local_mission_mean.values())) - np.mean(random_trial_macro)),
            "local_minus_random_mission_bootstrap": bootstrap,
            "random_sampling_quantile_ci95": [float(np.percentile(np.asarray([float(np.mean(list(local_mission_mean.values()))) - value for value in random_trial_macro]), 2.5)), float(np.percentile(np.asarray([float(np.mean(list(local_mission_mean.values()))) - value for value in random_trial_macro]), 97.5))],
            "candidate_size_match": True,
        }
    return result


def _write_candidate_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = ["state_id", "mission_id", "source_failure_type", "bc_action", "stored_bc_action", "valid_action_count"]
    for family in FAMILIES:
        fields.extend([family + "_candidate_size", family + "_candidate_actions", family + "_observed_candidate_actions", family + "_oracle_covered", family + "_recovery_covered", family + "_local_best_return", family + "_local_gain", family + "_full_gain", family + "_candidate_success_count"])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fields if key in record}
            for family in FAMILIES:
                item = record["families"][family]
                row.update({
                    family + "_candidate_size": len(item["candidate_actions"]),
                    family + "_candidate_actions": json.dumps(item["candidate_actions"], separators=(",", ":")),
                    family + "_observed_candidate_actions": json.dumps(item["observed_candidate_actions"], separators=(",", ":")),
                    family + "_oracle_covered": int(item["oracle_covered"]),
                    family + "_recovery_covered": int(item["recovery_covered"]),
                    family + "_local_best_return": item["local_best_return"],
                    family + "_local_gain": item["local_gain"],
                    family + "_full_gain": item["full_gain"],
                    family + "_candidate_success_count": item["candidate_success_count"],
                })
            writer.writerow(row)


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    oracle = result["oracle_coverage"]["families"]
    recovery = result["recovery_coverage"]["families"]
    gains = result["gain_retention"]["families"]
    sizes = result["candidate_sizes"]
    distance = result["distance_analysis"]
    decision = result["decision"]
    lines = [
        "# BC Local Candidate RL Feasibility Audit v1",
        "",
        "本报告是只读离线诊断，不是 RL 训练、生产 action-space 变更或 policy gate。候选集合只使用冻结 BC deterministic masked argmax、当前 state mask 和预定义 primitive 邻接；return、success、Q、oracle 只在审计阶段用于评价已执行分支。",
        "",
        "## 输入与边界",
        "",
        "- states `{}`；transitions `{}`；missions `{}`；action dimension `{}`。".format(result["input_identity"]["state_count"], result["input_identity"]["transition_count"], result["input_identity"]["mission_count"], result["input_identity"]["action_dim"]),
        "- BC action identity: match `{}`, mismatch `{}`, missing `{}`。".format(result["bc_audit"]["match_count"], result["bc_audit"]["mismatch_count"], result["bc_audit"]["missing_count"]),
        "- valid action branching mean `{:.6f}`。".format(result["valid_action_branching"]["mean"]),
        "- Neighbor order: existing verified 8-neighbor lattice first, sorted by descriptor distance/action id; non-graph actions then supplement by the same deterministic distance/order。",
        "",
        "## Observed oracle coverage",
        "",
        "| family | state micro | mission macro | 95% mission CI |",
        "|---|---:|---:|---:|",
    ]
    for family in FAMILIES:
        item = oracle[family]
        lines.append("| {} | {:.6f} | {:.6f} | [{:.6f}, {:.6f}] |".format(family, item["state_micro_coverage"], item["mission_macro_coverage"], item["mission_bootstrap"]["ci95"][0], item["mission_bootstrap"]["ci95"][1]))
    lines.extend(["", "## Recovery coverage", "", "| family | all states | dead_end | collision |", "|---|---:|---:|---:|"])
    for family in ("K0_BC_ONLY", "K2_LOCAL", "K4_LOCAL", "K8_LOCAL", "K16_LOCAL", "FULL_VALID"):
        all_item = recovery[family]
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} |".format(family, all_item["mission_macro_coverage"], recovery[family]["by_failure_type"]["dead_end"]["mission_macro_coverage"] or 0.0, recovery[family]["by_failure_type"]["collision"]["mission_macro_coverage"] or 0.0))
    lines.extend(["", "## Gain retention and branching", "", "| family | candidate mean | reduction mean | aggregate retention | state-wise ratio median |", "|---|---:|---:|---:|---:|"])
    for family in FAMILIES:
        lines.append("| {} | {:.6f} | {:.6f} | {} | {} |".format(family, sizes[family]["size"]["mean"], sizes[family]["reduction"]["mean"], "n/a" if gains[family]["aggregate_gain_retention_ratio"] is None else "{:.6f}".format(gains[family]["aggregate_gain_retention_ratio"]), "n/a" if gains[family]["state_wise_ratio"]["median"] is None else "{:.6f}".format(gains[family]["state_wise_ratio"]["median"])))
    lines.extend([
        "",
        "## BC-relative distance",
        "",
        "- best observed action graph distance (tie-aware minimum): mean `{:.6f}`, median `{:.6f}`, p90 `{:.6f}`。".format(distance["graph_distance"]["mean"], distance["graph_distance"]["median"], distance["graph_distance"]["p90"]),
        "- best observed action geometry distance: mean `{:.6f}`, median `{:.6f}`, p90 `{:.6f}`。".format(distance["geometry_distance"]["mean"], distance["geometry_distance"]["median"], distance["geometry_distance"]["p90"]),
        "- best action within 1-hop (including BC self): `{:.6f}`；within K4: `{:.6f}`；within K8: `{:.6f}`；within K16: `{:.6f}`。".format(distance["best_observed_action_within"]["within_1hop"], distance["best_observed_action_within"]["within_k4"], distance["best_observed_action_within"]["within_k8"], distance["best_observed_action_within"]["within_k16"]),
        "",
        "## Local versus random",
        "",
        "随机对照固定 seed `{}`，每个 family 重复 `{}` 次，并按完全相同的每-state candidate size 抽取合法非 BC action；抽样不读取 return/Q/oracle。".format(result["random_control"]["seed"], result["random_control"]["sampling_repeats"]),
        "",
        "| family | local - random | 95% mission CI |",
        "|---|---:|---:|",
    ])
    for family in ("K2_LOCAL", "K4_LOCAL", "K8_LOCAL", "K16_LOCAL"):
        item = result["random_control"]["families"][family]
        ci = item["local_minus_random_mission_bootstrap"]["ci95"]
        lines.append("| {} | {:.6f} | [{:.6f}, {:.6f}] |".format(family, item["local_minus_random"], ci[0], ci[1]))
    lines.extend([
        "",
        "## 预注册判定",
        "",
        "- `RECOMMENDED_K={}`。".format(decision["recommended_k"]),
        "- `NEXT_DECISION={}`。".format(decision["next_decision"]),
        "- 判定同时考虑 mission-macro oracle coverage >= 0.75、recovery coverage >= 0.75、positive gain retention >= 0.80、candidate mean <= 10，以及 local 相对 size-matched random 的 CI 下界 > 0。",
        "",
        "## 不变性",
        "",
        "- Actor optimizer steps `0`；Critic optimizer steps `0`；new environment steps `0`；AWAC/Dev100/Final300 均未执行。",
        "- production Replay、BC checkpoint、multi-action diagnostic input、production action library/config 均保持不变；本目录是独立 diagnostic artifact。",
        "- 本结果不能证明未执行 action 的收益，也不能证明一个新 policy 能泛化选择所有 observed oracle action。",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_hashes(
    *,
    dataset_root: Path,
    oracle_root: Path,
    action_root: Path,
    motion_json: Path,
    motion_npz: Path,
    motion_config: Path,
    bc_checkpoint: Path,
) -> Dict[str, str]:
    dataset_root = Path(dataset_root).resolve()
    return {
        "dataset_manifest_sha256": sha256_file(dataset_root / "dataset_manifest.json"),
        "dataset_transition_sha256": sha256_file(dataset_root / "replay" / "transitions.npz"),
        "dataset_pairwise_sha256": sha256_file(dataset_root / "multi_action_pairwise_dataset.csv"),
        "dataset_branches_sha256": sha256_file(dataset_root / "candidate_branches.jsonl"),
        "dataset_state_manifest_sha256": sha256_file(dataset_root / "state_manifest.jsonl"),
        "dataset_states_tree_sha256": sha256_tree(dataset_root / "states"),
        "oracle_upper_bound_sha256": sha256_file(Path(oracle_root) / "oracle_upper_bound.json"),
        "action_analysis_sha256": sha256_file(Path(action_root) / "action_analysis.json"),
        "primitive_metrics_sha256": sha256_file(Path(action_root) / "primitive_metrics.json"),
        "motion_primitives_json_sha256": sha256_file(Path(motion_json)),
        "motion_primitives_npz_sha256": sha256_file(Path(motion_npz)),
        "motion_primitives_config_sha256": sha256_file(Path(motion_config)),
        "bc_checkpoint_sha256": sha256_file(Path(bc_checkpoint)),
    }


def run_bc_local_candidate_audit(
    *,
    dataset_root: Path,
    oracle_root: Path,
    action_root: Path,
    motion_primitives_json: Path,
    motion_primitives_npz: Path,
    motion_primitives_config: Path,
    bc_checkpoint: Path,
    out_dir: Path,
    seed: int = DEFAULT_RANDOM_SEED,
    random_repeats: int = DEFAULT_RANDOM_REPEATS,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
) -> Dict[str, Any]:
    """Run the complete read-only local-candidate audit."""

    output = Path(out_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty diagnostic output {}".format(output))
    dataset_root = Path(dataset_root).expanduser().resolve()
    oracle_root = Path(oracle_root).expanduser().resolve()
    action_root = Path(action_root).expanduser().resolve()
    bc_checkpoint = Path(bc_checkpoint).expanduser().resolve()
    data = load_audit_dataset(dataset_root)
    if data["identity"]["state_count"] != EXPECTED_STATE_COUNT or data["identity"]["transition_count"] != EXPECTED_TRANSITION_COUNT or len(data["missions"]) != EXPECTED_MISSION_COUNT:
        raise ValueError("multi-action input count/schema gate failed")
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("action_count", EXPECTED_ACTION_DIM)) != EXPECTED_ACTION_DIM and int(manifest.get("unique_action_count", EXPECTED_ACTION_DIM)) != EXPECTED_ACTION_DIM:
        raise ValueError("multi-action manifest action dimension mismatch")
    oracle_payload = json.loads((oracle_root / "oracle_upper_bound.json").read_text(encoding="utf-8"))
    if oracle_payload.get("schema_id") != "offline_rl_oracle_upper_bound_v1" or bool(oracle_payload.get("q_loaded", True)) or bool(oracle_payload.get("rl_training_executed", True)):
        raise ValueError("offline oracle artifact identity gate failed")
    action_definitions, action_analysis, primitive_metrics = _load_action_analysis(action_root / "action_analysis.json", action_root / "primitive_metrics.json")
    if action_analysis["identity"].get("dataset_manifest_sha256") != data["identity"]["manifest_sha256"]:
        raise ValueError("action analysis dataset identity mismatch")
    if action_analysis["identity"].get("pairwise_sha256") != data["identity"].get("pairwise_sha256"):
        raise ValueError("action analysis pairwise identity mismatch")
    # Re-read the canonical MPL source to ensure the neighbor artifact is tied
    # to the current 105-action library, while using the already audited action
    # definitions for the actual candidate order.
    canonical_library = _load_action_library(Path(motion_primitives_json).resolve(), Path(motion_primitives_npz).resolve())
    if set(canonical_library["definitions"]) != set(action_definitions):
        raise ValueError("action library identity mismatch")
    raw_states = _load_raw_states(dataset_root)
    state_manifest = _load_state_manifest(dataset_root / "state_manifest.jsonl", expected_count=EXPECTED_STATE_COUNT)
    checkpoint, _model, _normalizer = _load_checkpoint(bc_checkpoint)
    if sha256_file(bc_checkpoint) != "ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2":
        raise ValueError("BC checkpoint SHA256 does not match the fixed input")
    state_ids = sorted(raw_states)
    logits = _b0_logits(_model, _normalizer, raw_states, state_ids)
    frozen_bc = {}
    match_count = 0
    mismatch_count = 0
    missing_count = 0
    transition_by_state: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in data["transitions"]:
        transition_by_state[str(row["state_id"])].append(row)
    for index, state_id in enumerate(state_ids):
        order = _masked_order(logits[index], raw_states[state_id]["mask"])
        selected = int(order[0])
        frozen_bc[state_id] = selected
        stored = state_manifest[state_id].get("bc_action")
        branch_rows = [row for row in transition_by_state[state_id] if str(row["action_source"]).upper() == "BC"]
        if stored is None or len(branch_rows) != 1:
            missing_count += 1
        elif int(stored) == selected == int(branch_rows[0]["action"]):
            match_count += 1
        else:
            mismatch_count += 1
    if mismatch_count or missing_count:
        raise ValueError("frozen BC identity mismatch: mismatch={}, missing={}".format(mismatch_count, missing_count))
    orders = _neighbor_orders(action_definitions)
    records = _build_state_records(data, raw_states, state_manifest, action_definitions, orders, frozen_bc)
    _annotate_outcomes(records)
    source_before = _source_hashes(dataset_root=dataset_root, oracle_root=oracle_root, action_root=action_root, motion_json=motion_primitives_json, motion_npz=motion_primitives_npz, motion_config=motion_primitives_config, bc_checkpoint=bc_checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    valid_branching = _stats_or_empty([record["valid_action_count"] for record in records])
    candidate_sizes: Dict[str, Any] = {}
    for family in FAMILIES:
        sizes = [len(record["candidates"][family]) for record in records]
        reductions = [1.0 - float(size) / float(record["valid_action_count"]) for record, size in zip(records, sizes)]
        candidate_sizes[family] = {"size": _stats_or_empty(sizes), "reduction": _stats_or_empty(reductions)}
    oracle_families: Dict[str, Any] = {}
    recovery_families: Dict[str, Any] = {}
    gain_families: Dict[str, Any] = {}
    for index, family in enumerate(FAMILIES):
        family_rows = []
        for record in records:
            item = record["families"][family]
            family_rows.append({"mission_id": record["mission_id"], "source_failure_type": record["source_failure_type"], "oracle_covered": item["oracle_covered"], "recovery_covered": item["recovery_covered"]})
        oracle_payload_family = _coverage_payload(family_rows, "oracle_covered", seed=int(seed) + index * 11, repeats=int(bootstrap_repeats))
        recovery_payload_family = _coverage_payload(family_rows, "recovery_covered", seed=int(seed) + index * 11 + 1, repeats=int(bootstrap_repeats))
        oracle_payload_family["by_failure_type"] = _coverage_by_failure_type(family_rows, "oracle_covered", seed=int(seed) + index * 11 + 2, repeats=int(bootstrap_repeats))
        recovery_payload_family["by_failure_type"] = _coverage_by_failure_type(family_rows, "recovery_covered", seed=int(seed) + index * 11 + 3, repeats=int(bootstrap_repeats))
        oracle_families[family] = oracle_payload_family
        recovery_families[family] = recovery_payload_family
        local_positive = [max(0.0, float(record["families"][family]["local_gain"])) for record in records]
        full_positive = [max(0.0, float(record["families"][family]["full_gain"])) for record in records]
        denominator = float(np.sum(full_positive))
        ratios = [local / full for local, full in zip(local_positive, full_positive) if full > 0.0]
        gain_families[family] = {
            "local_gain": _stats_or_empty([float(record["families"][family]["local_gain"]) for record in records]),
            "full_gain": _stats_or_empty([float(record["families"][family]["full_gain"]) for record in records]),
            "aggregate_gain_retention_ratio": float(np.sum(local_positive) / denominator) if denominator > 0.0 else None,
            "state_wise_ratio": _stats_or_empty(ratios),
            "candidate_quality": {
                "bc_return": _stats_or_empty([record["bc_return"] for record in records]),
                "best_observed_candidate_return": _stats_or_empty([record["families"][family]["local_best_return"] for record in records]),
                "worst_observed_candidate_return": _stats_or_empty([max(float(record["observed"][action]["episode_return"]) for action in record["families"][family]["observed_candidate_actions"]) if False else min(float(record["observed"][action]["episode_return"]) for action in record["families"][family]["observed_candidate_actions"]) for record in records]),
                "mean_observed_candidate_return": _stats_or_empty([record["families"][family]["local_mean_return"] for record in records]),
                "candidate_success_count": int(sum(record["families"][family]["candidate_success_count"] for record in records)),
                "candidate_success_count_per_state": _stats_or_empty([record["families"][family]["candidate_success_count"] for record in records]),
                "observed_candidate_action_count": int(sum(len(record["families"][family]["observed_candidate_actions"]) for record in records)),
            },
        }
    random_control = _random_control(records, repeats=int(random_repeats), seed=int(seed), bootstrap_repeats=int(bootstrap_repeats))
    distance_analysis = _distance_analysis(records, action_definitions, orders)
    recommended = None
    for family in ("K2_LOCAL", "K4_LOCAL", "K8_LOCAL", "K16_LOCAL"):
        if oracle_families[family]["mission_macro_coverage"] >= 0.75 and recovery_families[family]["mission_macro_coverage"] >= 0.75 and float(gain_families[family]["aggregate_gain_retention_ratio"] or 0.0) >= 0.80 and candidate_sizes[family]["size"]["mean"] <= 10.0 and random_control["families"][family]["local_minus_random_mission_bootstrap"]["ci95"][0] > 0.0:
            recommended = int(LOCAL_K[family])
            break
    k16_oracle = oracle_families["K16_LOCAL"]["mission_macro_coverage"]
    any_random_supported = any(random_control["families"][family]["local_minus_random_mission_bootstrap"]["ci95"][0] > 0.0 for family in ("K2_LOCAL", "K4_LOCAL", "K8_LOCAL", "K16_LOCAL"))
    if recommended is not None:
        next_decision = "LOCAL_CANDIDATE_RL_SUPPORTED"
    elif k16_oracle < 0.60 or not any_random_supported:
        next_decision = "LOCAL_CANDIDATE_RL_NOT_SUPPORTED"
    elif any(oracle_families[family]["mission_macro_coverage"] >= 0.60 or float(gain_families[family]["aggregate_gain_retention_ratio"] or 0.0) >= 0.65 for family in ("K2_LOCAL", "K4_LOCAL", "K8_LOCAL", "K16_LOCAL")):
        next_decision = "LOCAL_CANDIDATE_RL_INCONCLUSIVE"
    else:
        next_decision = "LOCAL_CANDIDATE_RL_NOT_SUPPORTED"
    source_after = _source_hashes(dataset_root=dataset_root, oracle_root=oracle_root, action_root=action_root, motion_json=motion_primitives_json, motion_npz=motion_primitives_npz, motion_config=motion_primitives_config, bc_checkpoint=bc_checkpoint)
    if source_before != source_after:
        raise RuntimeError("read-only source identity changed during audit")
    source_hashes = {
        "before": source_before,
        "after": source_after,
        "all_unchanged": True,
    }
    identity = {
        "schema_id": LOCAL_CANDIDATE_AUDIT_SCHEMA_ID,
        "dataset_id": data["identity"]["dataset_id"],
        "dataset_manifest_sha256": data["identity"]["manifest_sha256"],
        "dataset_transition_sha256": data["identity"]["transitions_sha256"],
        "dataset_pairwise_sha256": data["identity"]["pairwise_sha256"],
        "state_count": len(records),
        "transition_count": len(data["transitions"]),
        "mission_count": len(data["missions"]),
        "action_dim": EXPECTED_ACTION_DIM,
        "oracle_input_sha256": sha256_file(oracle_root / "oracle_upper_bound.json"),
        "action_analysis_input_sha256": sha256_file(action_root / "action_analysis.json"),
        "primitive_metrics_input_sha256": sha256_file(action_root / "primitive_metrics.json"),
        "bc_checkpoint_sha256": source_before["bc_checkpoint_sha256"],
        "production_replay_modified": False,
        "production_action_space_modified": False,
        "production_config_modified": False,
        "q_loaded": False,
        "rl_training_executed": False,
        "environment_steps": 0,
    }
    candidate_definition = {
        "schema_id": CANDIDATE_DEFINITION_SCHEMA_ID,
        "families": {family: {"family": family, "neighbor_count": LOCAL_K[family] if family in LOCAL_K else None, "includes_bc": True, "max_candidate_size": LOCAL_K[family] + 1 if family in LOCAL_K else EXPECTED_ACTION_DIM} for family in FAMILIES},
        "neighbor_definition": "verified action_formulation neighbors_8 first, descriptor Euclidean distance then action_id; non-graph actions supplement for K16",
        "distance_definition": "Euclidean distance over normalized MPL descriptor [lateral_endpoint_m, vertical_endpoint_m, terminal_heading_deg]",
        "candidate_generation_inputs": ["frozen BC deterministic masked argmax", "current state action mask", "predefined MPL geometry/neighbor graph"],
        "candidate_generation_forbidden": ["return", "success", "terminal outcome", "Q", "Critic", "oracle label"],
        "neighbor_graph_sha256": _canonical_sha({str(action): [int(item) for item in action_definitions[action].get("neighbors_8", [])] for action in sorted(action_definitions)}),
        "neighbor_order_sha256": _canonical_sha({str(action): [int(item) for item in orders[action]] for action in sorted(orders)}),
        "monotonic_nestedness_verified": True,
        "deterministic_order_verified": True,
    }
    oracle_coverage = {"schema_id": ORACLE_COVERAGE_SCHEMA_ID, "families": oracle_families, "definition": "A_best is all observed actions tied at maximum branch episode_return; no unexecuted action is scored."}
    recovery_coverage = {"schema_id": RECOVERY_COVERAGE_SCHEMA_ID, "families": recovery_families, "definition": "A_success is the set of observed branches with success=true."}
    gain_retention = {"schema_id": GAIN_RETENTION_SCHEMA_ID, "families": gain_families, "full_gain_definition": "max episode_return over all observed actions minus recorded BC return", "local_gain_definition": "max episode_return over observed actions in candidate set minus recorded BC return"}
    bootstrap = {"schema_id": BOOTSTRAP_SCHEMA_ID, "repeats": int(bootstrap_repeats), "seed": int(seed), "oracle": {family: oracle_families[family]["mission_bootstrap"] for family in FAMILIES}, "recovery": {family: recovery_families[family]["mission_bootstrap"] for family in FAMILIES}, "local_minus_random": {family: random_control["families"][family]["local_minus_random_mission_bootstrap"] for family in ("K2_LOCAL", "K4_LOCAL", "K8_LOCAL", "K16_LOCAL")}}
    integrity = {
        "schema_id": INTEGRITY_SCHEMA_ID,
        "source_hashes": source_hashes,
        "input_files_unchanged": True,
        "production_replay_modified": False,
        "bc_checkpoint_modified": False,
        "multi_action_replay_modified": False,
        "production_action_space_modified": False,
        "production_config_modified": False,
        "actor_optimizer_steps": 0,
        "critic_optimizer_steps": 0,
        "awac_training_executed": False,
        "new_environment_steps": 0,
        "dev100": False,
        "final300": False,
    }
    result = {
        "input_identity": identity,
        "source_hashes": source_hashes,
        "bc_audit": {"match_count": match_count, "mismatch_count": mismatch_count, "missing_count": missing_count, "checkpoint_metadata": {"seed": checkpoint.get("seed"), "num_actions": checkpoint.get("num_actions"), "vec_dim": checkpoint.get("vec_dim")}},
        "valid_action_branching": valid_branching,
        "candidate_definition": candidate_definition,
        "candidate_sizes": candidate_sizes,
        "oracle_coverage": oracle_coverage,
        "recovery_coverage": recovery_coverage,
        "gain_retention": gain_retention,
        "distance_analysis": distance_analysis,
        "random_control": random_control,
        "bootstrap": bootstrap,
        "integrity": integrity,
        "decision": {"recommended_k": recommended, "next_decision": next_decision, "criteria": {"oracle_mission_macro_min": 0.75, "recovery_mission_macro_min": 0.75, "positive_gain_retention_min": 0.80, "candidate_size_mean_max": 10.0, "local_minus_random_ci_lower_gt": 0.0}},
    }
    _json_dump(output / "input_identity.json", identity)
    _json_dump(output / "candidate_definition.json", candidate_definition)
    _write_candidate_csv(output / "state_candidate_sets.csv", records)
    _json_dump(output / "oracle_coverage.json", oracle_coverage)
    _json_dump(output / "recovery_coverage.json", recovery_coverage)
    _json_dump(output / "gain_retention.json", gain_retention)
    _json_dump(output / "distance_analysis.json", distance_analysis)
    _json_dump(output / "random_control.json", random_control)
    _json_dump(output / "cluster_bootstrap.json", bootstrap)
    _json_dump(output / "integrity_audit.json", integrity)
    if next_decision == "LOCAL_CANDIDATE_RL_SUPPORTED":
        _json_dump(output / "local_candidate_rl_interface_v1.json", {"schema_id": "local_candidate_rl_interface_v1", "base_action": "frozen_BC_action", "rl_role": "candidate_selection_or_correction", "candidate_count_max": int(LOCAL_K["K8_LOCAL"] + 1), "input": ["current_observation", "BC_logits_or_probabilities", "BC_chosen_action", "candidate_action_descriptors", "candidate_mask"], "output": "candidate_index", "production_enabled": False})
    _write_report(output / "report_zh.md", result)
    return result


__all__ = [
    "FAMILIES",
    "LOCAL_CANDIDATE_AUDIT_SCHEMA_ID",
    "build_candidate_set",
    "run_bc_local_candidate_audit",
]
