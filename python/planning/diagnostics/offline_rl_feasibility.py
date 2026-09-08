"""Read-only empirical upper-bound audit for offline policy improvement.

The oracle is deliberately restricted to actions that were actually executed
for the same recorded state.  It does not use a Q function and is not a policy
or a production gate: it measures how much observed branch outcome is above
the recorded BC branch before any learning algorithm is considered.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from planning.diagnostics.state_sufficiency import load_audit_dataset, mission_cluster_bootstrap, sha256_file


OFFLINE_RL_FEASIBILITY_AUDIT_SCHEMA_ID = "offline_rl_feasibility_audit_v1"
ORACLE_UPPER_BOUND_SCHEMA_ID = "offline_rl_oracle_upper_bound_v1"
EXPECTED_STATE_COUNT = 479
EXPECTED_TRANSITION_COUNT = 2850
DEFAULT_SEED = 20260908
DEFAULT_BOOTSTRAP_REPEATS = 5000


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stats(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "p99": None, "min": None, "max": None}
    if not np.isfinite(array).all():
        raise ValueError("non-finite offline feasibility statistic")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _descending_rank(values: Sequence[float], target: float) -> float:
    """Competition rank with average rank for exact return ties."""

    greater = sum(float(value) > float(target) for value in values)
    equal = sum(float(value) == float(target) for value in values)
    return float(1.0 + greater + max(0, equal - 1) / 2.0)


def _rank_percentile(rank: float, count: int) -> float:
    if int(count) <= 1:
        return 1.0
    return float(1.0 - (float(rank) - 1.0) / float(count - 1))


def _state_rows(transitions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in transitions:
        grouped[str(row["state_id"])].append(row)
    rows: List[Dict[str, Any]] = []
    for state_id in sorted(grouped):
        branches = grouped[state_id]
        bc = [row for row in branches if str(row["action_source"]).upper() == "BC"]
        if len(bc) != 1:
            raise ValueError("expected exactly one BC branch for state {}".format(state_id))
        returns = [float(row["episode_return"]) for row in branches]
        best_return = max(returns)
        oracle_candidates = [row for row in branches if float(row["episode_return"]) == best_return]
        oracle = min(oracle_candidates, key=lambda row: int(row["action"]))
        bc_row = bc[0]
        bc_return = float(bc_row["episode_return"])
        oracle_return = float(oracle["episode_return"])
        bc_rank = _descending_rank(returns, bc_return)
        oracle_rank = _descending_rank(returns, oracle_return)
        rows.append(
            {
                "state_id": state_id,
                "mission_id": str(bc_row["mission_id"]),
                "episode_id": str(bc_row["episode_id"]),
                "step_id": int(bc_row["step_id"]),
                "observed_action_count": len(branches),
                "observed_action_ids": sorted(int(row["action"]) for row in branches),
                "bc_action": int(bc_row["action"]),
                "bc_action_source": str(bc_row["action_source"]),
                "bc_return": bc_return,
                "bc_success": int(bc_row["success"]),
                "bc_rank_descending": bc_rank,
                "bc_rank_percentile": _rank_percentile(bc_rank, len(branches)),
                "oracle_action": int(oracle["action"]),
                "oracle_action_source": str(oracle["action_source"]),
                "oracle_return": oracle_return,
                "oracle_success": int(oracle["success"]),
                "oracle_rank_descending": oracle_rank,
                "oracle_rank_percentile": _rank_percentile(oracle_rank, len(branches)),
                "oracle_tie_count": len(oracle_candidates),
                "oracle_return_delta": oracle_return - bc_return,
                "oracle_success_delta": int(oracle["success"]) - int(bc_row["success"]),
                "oracle_action_is_bc": int(oracle["action"]) == int(bc_row["action"]),
            }
        )
    return rows


def _mission_bootstrap(rows: Sequence[Mapping[str, Any]], *, repeats: int, seed: int) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"return": [], "success": [], "bc_return": [], "oracle_return": []})
    for row in rows:
        mission = str(row["mission_id"])
        grouped[mission]["return"].append(float(row["oracle_return_delta"]))
        grouped[mission]["success"].append(float(row["oracle_success_delta"]))
        grouped[mission]["bc_return"].append(float(row["bc_return"]))
        grouped[mission]["oracle_return"].append(float(row["oracle_return"]))
    values = {
        mission: [float(np.mean(item["return"]))]
        for mission, item in grouped.items()
    }
    success_values = {
        mission: [float(np.mean(item["success"])),]
        for mission, item in grouped.items()
    }
    bc_values = {mission: [float(np.mean(item["bc_return"]))] for mission, item in grouped.items()}
    oracle_values = {mission: [float(np.mean(item["oracle_return"]))] for mission, item in grouped.items()}
    return {
        "cluster_unit": "mission_id",
        "mission_count": len(grouped),
        "return_delta": mission_cluster_bootstrap(values, repeats=int(repeats), seed=int(seed)),
        "success_delta": mission_cluster_bootstrap(success_values, repeats=int(repeats), seed=int(seed) + 1),
        "bc_return": mission_cluster_bootstrap(bc_values, repeats=int(repeats), seed=int(seed) + 2),
        "oracle_return": mission_cluster_bootstrap(oracle_values, repeats=int(repeats), seed=int(seed) + 3),
    }


def _conclusion(rows: Sequence[Mapping[str, Any]], bootstrap: Mapping[str, Any]) -> Dict[str, Any]:
    return_deltas = np.asarray([float(row["oracle_return_delta"]) for row in rows], dtype=np.float64)
    success_deltas = np.asarray([float(row["oracle_success_delta"]) for row in rows], dtype=np.float64)
    return_ci = bootstrap["return_delta"]["ci95"]
    success_ci = bootstrap["success_delta"]["ci95"]
    positive_return_states = int(np.count_nonzero(return_deltas > 0.0))
    positive_success_states = int(np.count_nonzero(success_deltas > 0.0))
    if len(rows) < 50:
        result = "C_DATA_INSUFFICIENT_TO_JUDGE"
    elif positive_return_states == 0 and positive_success_states == 0:
        result = "B_DATA_CLOSE_TO_BC_OBSERVED_UPPER_BOUND"
    elif float(return_ci[0]) > 0.0 or float(success_ci[0]) > 0.0:
        result = "A_DATA_HAS_POLICY_IMPROVEMENT_SPACE"
    else:
        result = "C_DATA_INSUFFICIENT_TO_JUDGE_ROBUSTLY"
    return {
        "result": result,
        "observed_positive_return_state_count": positive_return_states,
        "observed_positive_success_state_count": positive_success_states,
        "return_ci_lower_gt_zero": bool(float(return_ci[0]) > 0.0),
        "success_ci_lower_gt_zero": bool(float(success_ci[0]) > 0.0),
        "oracle_is_not_a_policy": True,
        "caveat": "The oracle selects among already executed actions at each state. It is an empirical observed-action upper bound, not proof that one learned policy can select all oracle actions or generalize to unseen states.",
    }


def run_offline_rl_feasibility_audit(
    *,
    dataset_root: Path,
    bc_dev_root: Path,
    out_dir: Path,
    seed: int = DEFAULT_SEED,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    bc_dev_root = Path(bc_dev_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output: {}".format(out_dir))
    if not (bc_dev_root / "summary.json").is_file():
        raise FileNotFoundError(str(bc_dev_root / "summary.json"))
    data = load_audit_dataset(dataset_root)
    if data["identity"]["state_count"] != EXPECTED_STATE_COUNT or data["identity"]["transition_count"] != EXPECTED_TRANSITION_COUNT:
        raise ValueError("unexpected multi-action dimensions")
    rows = _state_rows(data["transitions"])
    if len(rows) != EXPECTED_STATE_COUNT:
        raise ValueError("state grouping lost rows")
    bootstrap = _mission_bootstrap(rows, repeats=int(bootstrap_repeats), seed=int(seed))
    source_manifest = dataset_root / "dataset_manifest.json"
    bc_summary_path = bc_dev_root / "summary.json"
    bc_summary = json.loads(bc_summary_path.read_text(encoding="utf-8"))
    return_deltas = [float(row["oracle_return_delta"]) for row in rows]
    success_deltas = [float(row["oracle_success_delta"]) for row in rows]
    bc_returns = [float(row["bc_return"]) for row in rows]
    oracle_returns = [float(row["oracle_return"]) for row in rows]
    bc_success = [float(row["bc_success"]) for row in rows]
    oracle_success = [float(row["oracle_success"]) for row in rows]
    action_counts = [float(row["observed_action_count"]) for row in rows]
    payload = {
        "schema_id": ORACLE_UPPER_BOUND_SCHEMA_ID,
        "diagnostic_only": True,
        "oracle_definition": "For each state, choose the highest-return action among only the already executed branches for that same state; ties are resolved by lowest action id for the representative row.",
        "forbidden_inputs": ["Q values", "critic checkpoint", "unexecuted actions", "future outcomes"],
        "source": {
            "dataset_id": data["identity"]["dataset_id"],
            "dataset_root": str(dataset_root),
            "dataset_manifest_sha256": sha256_file(source_manifest),
            "transitions_sha256": data["identity"]["transitions_sha256"],
            "candidate_branches_sha256": data["identity"]["candidate_branches_sha256"],
            "bc_dev_root": str(bc_dev_root),
            "bc_summary_sha256": sha256_file(bc_summary_path),
            "bc_checkpoint_sha256": bc_summary.get("checkpoint_sha256"),
            "bc_mission_index_sha256": bc_summary.get("mission_index_sha256"),
            "multi_action_failure_state_source": data["manifest"].get("failure_state_source"),
            "observation_contract": data["identity"]["observation_contract"],
            "task_contract_sha256": data["identity"]["task_contract_sha256"],
        },
        "bc_dev_context": {
            "full_dev_success_count": bc_summary.get("success_count"),
            "full_dev_collision_count": bc_summary.get("collision_count"),
            "full_dev_dead_end_count": bc_summary.get("dead_end_count"),
            "note": "The oracle state slice is drawn from BC-failure missions; its conditional BC success rate is not the full Dev100 success rate.",
        },
        "counts": {
            "state_count": len(rows),
            "transition_count": len(data["transitions"]),
            "mission_count": len({str(row["mission_id"]) for row in rows}),
            "bc_branch_count": len(rows),
            "oracle_branch_count": len(rows),
            "bc_action_coverage_ratio": 1.0,
        },
        "behavior_policy_baseline": {
            "bc_return": _stats(bc_returns),
            "bc_success": _stats(bc_success),
            "action_count_per_state": _stats(action_counts),
        },
        "oracle_upper_bound": {
            "oracle_return": _stats(oracle_returns),
            "oracle_success": _stats(oracle_success),
            "return_delta_oracle_minus_bc": _stats(return_deltas),
            "success_delta_oracle_minus_bc": _stats(success_deltas),
            "strict_return_improvement_state_count": int(sum(value > 0.0 for value in return_deltas)),
            "strict_return_improvement_state_ratio": float(np.mean(np.asarray(return_deltas) > 0.0)),
            "strict_success_improvement_state_count": int(sum(value > 0.0 for value in success_deltas)),
            "strict_success_improvement_state_ratio": float(np.mean(np.asarray(success_deltas) > 0.0)),
            "oracle_equals_bc_state_count": int(sum(bool(row["oracle_action_is_bc"]) for row in rows)),
        },
        "action_diversity_gain": {
            "bc_action_rank_descending": _stats(row["bc_rank_descending"] for row in rows),
            "oracle_action_rank_descending": _stats(row["oracle_rank_descending"] for row in rows),
            "bc_action_rank_percentile": _stats(row["bc_rank_percentile"] for row in rows),
            "oracle_action_rank_percentile": _stats(row["oracle_rank_percentile"] for row in rows),
            "oracle_tie_count": int(sum(int(row["oracle_tie_count"]) > 1 for row in rows)),
        },
        "mission_bootstrap": bootstrap,
        "conclusion": _conclusion(rows, bootstrap),
        "state_rows": rows,
        "production_modified": False,
        "replay_modified": False,
        "q_loaded": False,
        "rl_training_executed": False,
        "runtime_started": False,
    }
    report = _build_report(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(out_dir / "oracle_upper_bound.json", payload)
    (out_dir / "report_zh.md").write_text(report, encoding="utf-8")
    return {"out_dir": str(out_dir), "payload": payload}


def _build_report(payload: Mapping[str, Any]) -> str:
    baseline = payload["behavior_policy_baseline"]
    oracle = payload["oracle_upper_bound"]
    conclusion = payload["conclusion"]
    bootstrap = payload["mission_bootstrap"]
    return "\n".join(
        [
            "# OFFLINE RL FEASIBILITY AUDIT V1",
            "",
            "本报告是只读 offline diagnostic；未执行 AWAC、Actor、Critic、Unity，未修改 Replay。",
            "",
            "## 口径",
            "",
            "Oracle 只能在同一 state 已真实执行的 action branches 中选择 episode return 最大者，不使用 Q、不访问未执行 action。Tie 仅以最低 action id 作为代表行，统计仍保留 tie count。",
            "",
            "- states={}；transitions={}；missions={}；BC branch coverage=100%。".format(
                payload["counts"]["state_count"], payload["counts"]["transition_count"], payload["counts"]["mission_count"]
            ),
            "- 这些 state 来自 BC failure-state slice；因此下方 BC success=0 是条件基线，不能替代完整 Dev100 的历史/独立评估结果。",
            "- BC return mean={:.6f}；oracle observed upper-bound return mean={:.6f}；delta mean={:.6f}。".format(
                baseline["bc_return"]["mean"], oracle["oracle_return"]["mean"], oracle["return_delta_oracle_minus_bc"]["mean"]
            ),
            "- BC success mean={:.6f}；oracle observed upper-bound success mean={:.6f}；strict success-gain states={}。".format(
                baseline["bc_success"]["mean"], oracle["oracle_success"]["mean"], oracle["strict_success_improvement_state_count"]
            ),
            "",
            "## Action diversity gain",
            "",
            "- BC descending return rank mean={:.3f}，percentile mean={:.6f}。".format(
                payload["action_diversity_gain"]["bc_action_rank_descending"]["mean"], payload["action_diversity_gain"]["bc_action_rank_percentile"]["mean"]
            ),
            "- Oracle descending return rank mean={:.3f}，percentile mean={:.6f}。".format(
                payload["action_diversity_gain"]["oracle_action_rank_descending"]["mean"], payload["action_diversity_gain"]["oracle_action_rank_percentile"]["mean"]
            ),
            "",
            "## Mission-cluster bootstrap",
            "",
            "- return delta estimate={:.6f}，95% CI=[{:.6f}, {:.6f}]。".format(
                bootstrap["return_delta"]["estimate"], bootstrap["return_delta"]["ci95"][0], bootstrap["return_delta"]["ci95"][1]
            ),
            "- success delta estimate={:.6f}，95% CI=[{:.6f}, {:.6f}]。".format(
                bootstrap["success_delta"]["estimate"], bootstrap["success_delta"]["ci95"][0], bootstrap["success_delta"]["ci95"][1]
            ),
            "bootstrap unit=mission_id；CI 不包含训练随机性，也不代表可泛化 policy performance。",
            "",
            "## 结论",
            "",
            "`{}`。这表示 observed branches 中存在超过 BC 的动作支持，但 oracle 不是一个已实现的策略；它不能证明单一 offline policy 能在每个 state 选中 oracle，也不能证明未见 state 会提升。".format(conclusion["result"]),
            "",
            "后续若继续，应先做受约束的 action-selection/coverage 验证；本审计本身不触发 AWAC、Actor、Critic 或 production gate。",
            "",
        ]
    )


__all__ = ["run_offline_rl_feasibility_audit"]
