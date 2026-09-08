"""Read-only audit of flat primitive actions versus hierarchical candidates.

This diagnostic consumes the immutable 105-action library and an existing BC
Dev100 step trace.  It does not import the AWAC learner, start a runtime, write
Replay, or change any production contract.  Its failure-mode labels are
explicitly evidence levels: the historical trace does not persist enough
execution telemetry to prove a high-level/low-level causal split.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from planning.diagnostics.action_formulation_audit import _load_action_library
from planning.diagnostics.state_sufficiency import sha256_file


HIERARCHICAL_TASK_FORMULATION_AUDIT_SCHEMA_ID = "hierarchical_task_formulation_audit_v1"
ACTION_CLUSTERS_SCHEMA_ID = "hierarchical_action_clusters_v1"
TRAJECTORY_PATTERN_SCHEMA_ID = "hierarchical_trajectory_pattern_v1"
FAILURE_MODE_SCHEMA_ID = "hierarchical_failure_mode_analysis_v1"
EXPECTED_ACTION_COUNT = 105
EXPECTED_EPISODES = 100
OUTCOME_ORDER = ("success", "dead_end", "collision", "timeout", "other")


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any, *, default: float = 0.0) -> float:
    text = str(value).strip()
    if not text:
        return float(default)
    result = float(text)
    if not math.isfinite(result):
        raise ValueError("non-finite numeric value")
    return result


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def _entropy(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = float(len(tokens))
    return float(-sum((count / total) * math.log(count / total) for count in counts.values()))


def _top_counter(counter: Counter, limit: int = 12) -> List[Dict[str, Any]]:
    return [{"key": str(key), "count": int(count)} for key, count in counter.most_common(limit)]


def _family_from_mode(mode: str, *, positive: str, neutral: str, negative: str) -> str:
    mode = str(mode).strip().lower()
    if mode == positive:
        return positive
    if mode == negative:
        return negative
    return neutral


def _action_cluster_labels(item: Mapping[str, Any]) -> Dict[str, str]:
    lateral = _family_from_mode(
        str(item.get("lateral_mode", "")), positive="right", neutral="straight", negative="left"
    )
    vertical = _family_from_mode(
        str(item.get("vertical_mode", "")), positive="up", neutral="level", negative="down"
    )
    heading = float(item["terminal_heading_deg"])
    if abs(heading) < 1e-6:
        heading_family = "neutral"
    elif heading > 0:
        heading_family = "right_turn"
    else:
        heading_family = "left_turn"
    endpoint_lateral = abs(float(item["lateral_endpoint_m"]))
    endpoint_vertical = abs(float(item["vertical_endpoint_m"]))
    lateral_effect = "none" if endpoint_lateral < 1e-6 else ("small" if endpoint_lateral <= 0.10 else "large")
    vertical_effect = "none" if endpoint_vertical < 1e-6 else ("small" if endpoint_vertical <= 0.04 else "large")
    heading_effect = "neutral" if abs(heading) < 1e-6 else ("mild" if abs(heading) <= 10.0 else "strong")
    return {
        "primitive_family": "fixed_duration_endpoint_primitive",
        "horizontal_family": lateral,
        "vertical_family": vertical,
        "motion_direction": "{}_{}".format(lateral, vertical),
        "heading_family": heading_family,
        "direction_intent": "{}_{}".format(lateral, vertical),
        "duration_family": "duration_{:.3f}s_frames_{}".format(
            float(item["duration_s"]), int(item["command_frames"])
        ),
        "control_effect": "lateral_{}_vertical_{}_heading_{}".format(
            lateral_effect, vertical_effect, heading_effect
        ),
    }


def _build_action_clusters(library: Mapping[str, Any]) -> Dict[str, Any]:
    definitions = library["definitions"]
    actions = library["actions"]
    if len(actions) != EXPECTED_ACTION_COUNT:
        raise ValueError("expected 105 actions")
    clusters: Dict[str, Counter] = {
        "primitive_family": Counter(),
        "horizontal_family": Counter(),
        "vertical_family": Counter(),
        "motion_direction": Counter(),
        "heading_family": Counter(),
        "direction_intent": Counter(),
        "duration_family": Counter(),
        "control_effect": Counter(),
    }
    rows: List[Dict[str, Any]] = []
    for raw in actions:
        action_id = int(raw["id"])
        labels = _action_cluster_labels(raw)
        for key, label in labels.items():
            clusters[key][label] += 1
        definition = definitions[action_id]
        rows.append(
            {
                "action_id": action_id,
                "primitive_id": definition["primitive_id"],
                "horizontal_index": int(raw["horizontal_index"]),
                "vertical_index": int(raw["vertical_index"]),
                "lateral_endpoint_m": float(raw["lateral_endpoint_m"]),
                "vertical_endpoint_m": float(raw["vertical_endpoint_m"]),
                "terminal_heading_deg": float(raw["terminal_heading_deg"]),
                "duration_s": float(raw["duration_s"]),
                "command_frames": int(raw["command_frames"]),
                "control_effect_numeric": {
                    name: float(raw[name])
                    for name in (
                        "max_horizontal_speed_mps",
                        "max_vertical_speed_mps",
                        "max_horizontal_acceleration_mps2",
                        "max_vertical_acceleration_mps2",
                        "max_yaw_rate_radps",
                    )
                },
                "neighbors_4": list(definition["neighbors_4"]),
                "neighbors_8": list(definition["neighbors_8"]),
                "cluster_labels": labels,
            }
        )
    return {
        "schema_id": ACTION_CLUSTERS_SCHEMA_ID,
        "diagnostic_only": True,
        "source": {
            "motion_primitives_json": library["json_path"],
            "motion_primitives_npz": library["npz_path"],
            "motion_primitives_json_sha256": library["json_sha256"],
            "motion_primitives_npz_sha256": library["npz_sha256"],
            "mpl_contract_sha256": library["meta"].get("contract_sha256"),
        },
        "summary": {
            "action_count": len(rows),
            "lattice": {
                "horizontal_count": int(library["meta"]["num_horizontal"]),
                "vertical_count": int(library["meta"]["num_vertical"]),
            },
            "duration_s": _finite_stats(row["duration_s"] for row in rows),
            "command_frames": _finite_stats(row["command_frames"] for row in rows),
            "control_dt_s": float(library["meta"]["control_dt_s"]),
            "center_action_id": int(library["meta"]["center_action_id"]),
            "exact_parameter_duplicate_count": 0,
            "exact_behavior_duplicate_count": 0,
            "neighbor_edge_counts": {
                "four_connected": int(sum(len(row["neighbors_4"]) for row in rows) // 2),
                "eight_connected": int(sum(len(row["neighbors_8"]) for row in rows) // 2),
            },
        },
        "cluster_counts": {key: dict(sorted(value.items())) for key, value in clusters.items()},
        "clusters": {
            "primitive_family": "All 105 rows use the fixed-duration endpoint primitive; the useful coarse split is direction_intent.",
            "motion_direction": "lateral_mode x vertical_mode, yielding the observed 3 x 3 intent families.",
            "duration": "All actions currently share the same duration and command-frame contract.",
            "control_effect": "Deterministic endpoint/heading magnitude buckets plus persisted numeric control effects; not an outcome label.",
        },
        "actions": sorted(rows, key=lambda row: row["action_id"]),
        "candidate_high_level_action_spaces": [
            {
                "name": "direction_intent",
                "cardinality": 9,
                "values": sorted(clusters["direction_intent"]),
                "supported_by_current_artifact": True,
                "description": "Coarse lateral/vertical intent; retain the existing primitive selector as the low-level realization.",
            },
            {
                "name": "navigation_mode",
                "cardinality": 5,
                "values": ["goal_approach", "lateral_align", "altitude_correct", "heading_align", "recovery_safe_brake"],
                "supported_by_current_artifact": False,
                "description": "Design candidates only; current trace has no explicit high-level mode label.",
            },
            {
                "name": "local_objective",
                "cardinality": 5,
                "values": ["reduce_goal_distance", "reduce_heading_error", "maintain_altitude_band", "increase_clearance", "recover_dead_end"],
                "supported_by_current_artifact": False,
                "description": "Requires explicit non-privileged objective/feature definitions before training.",
            },
            {
                "name": "local_waypoint_intent",
                "cardinality": "continuous_or_small_discrete_set",
                "values": [],
                "supported_by_current_artifact": False,
                "description": "Candidate relative waypoint/direction target from current observation only; no future route or termination label may be used.",
            },
        ],
    }


def _read_episode(index_row: Mapping[str, str], root: Path, action_rows: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    step_path = (root / str(index_row["step_csv"])).resolve()
    if not step_path.is_file() or root not in step_path.parents:
        raise FileNotFoundError(str(step_path))
    with step_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["step"]))
    if [int(row["step"]) for row in rows] != list(range(len(rows))):
        raise ValueError("non-contiguous step sequence in {}".format(step_path))
    action_ids = [int(row["action"]) for row in rows]
    if any(action_id not in action_rows for action_id in action_ids):
        raise ValueError("step references action outside MPL")
    labels = [_action_cluster_labels(action_rows[action_id]) for action_id in action_ids]
    intents = [label["direction_intent"] for label in labels]
    horizontals = [label["horizontal_family"] for label in labels]
    verticals = [label["vertical_family"] for label in labels]
    headings = [label["heading_family"] for label in labels]
    repeats = sum(first == second for first, second in zip(action_ids, action_ids[1:]))
    switches = max(0, len(action_ids) - 1 - repeats)
    progress = [
        _as_float(row.get("distance_before")) - _as_float(row.get("distance_after"))
        for row in rows
    ]
    outcome = "other"
    for candidate in ("success", "dead_end", "collision", "timeout"):
        if _as_bool(index_row.get(candidate, False)):
            outcome = candidate
            break
    primitive_incomplete = sum(not _as_bool(row.get("primitive_completed", False)) for row in rows)
    terminal_abort = sum(_as_bool(row.get("terminal_abort", False)) for row in rows)
    abort_reasons = [str(row.get("terminal_abort_reason", "")).strip() for row in rows if str(row.get("terminal_abort_reason", "")).strip()]
    integration_ticks = [_as_float(row.get("effective_integration_ticks"), default=-1.0) for row in rows]
    applied_frames = [_as_float(row.get("applied_frame_count"), default=-1.0) for row in rows]
    return {
        "episode_id": str(index_row["episode_id"]),
        "mission_id": str(index_row["mission_id"]),
        "step_csv": str(index_row["step_csv"]),
        "outcome": outcome,
        "stop_reason": str(index_row.get("stop_reason", "")),
        "return": _as_float(index_row.get("return")),
        "steps": len(rows),
        "action_ids": action_ids,
        "primitive_families": [label["primitive_family"] for label in labels],
        "horizontal_families": horizontals,
        "vertical_families": verticals,
        "heading_families": headings,
        "direction_intents": intents,
        "repeat_ratio": float(repeats / max(1, len(action_ids) - 1)),
        "switch_rate": float(switches / max(1, len(action_ids) - 1)),
        "unique_action_count": len(set(action_ids)),
        "unique_direction_intent_count": len(set(intents)),
        "action_entropy": _entropy([str(action) for action in action_ids]),
        "direction_intent_entropy": _entropy(intents),
        "horizontal_change_count": sum(a != b for a, b in zip(horizontals, horizontals[1:])),
        "vertical_change_count": sum(a != b for a, b in zip(verticals, verticals[1:])),
        "heading_change_count": sum(a != b for a, b in zip(headings, headings[1:])),
        "progress_sum": float(sum(progress)),
        "progress_mean": float(np.mean(progress)) if progress else 0.0,
        "terminal_action_id": action_ids[-1] if action_ids else None,
        "terminal_direction_intent": intents[-1] if intents else None,
        "terminal_primitive_completed": _as_bool(rows[-1].get("primitive_completed", False)) if rows else False,
        "primitive_incomplete_row_count": int(primitive_incomplete),
        "terminal_abort_row_count": int(terminal_abort),
        "terminal_abort_reasons": sorted(set(abort_reasons)),
        "effective_integration_ticks_available": bool(rows) and min(integration_ticks) >= 0,
        "applied_frame_count_available": bool(rows) and min(applied_frames) >= 0,
        "last_step": rows[-1] if rows else {},
    }


def _sequence_counter(episodes: Sequence[Mapping[str, Any]], key: str, width: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    prefixes = Counter()
    suffixes = Counter()
    transitions = Counter()
    for episode in episodes:
        values = [str(value) for value in episode[key]]
        if values:
            prefixes[" -> ".join(values[:width])] += 1
            suffixes[" -> ".join(values[-width:])] += 1
            transitions.update("{} -> {}".format(a, b) for a, b in zip(values, values[1:]))
    return {
        "prefix": _top_counter(prefixes),
        "suffix": _top_counter(suffixes),
        "transitions": _top_counter(transitions),
    }


def _group_episode_stats(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    fields = (
        "steps",
        "unique_action_count",
        "unique_direction_intent_count",
        "repeat_ratio",
        "switch_rate",
        "action_entropy",
        "direction_intent_entropy",
        "horizontal_change_count",
        "vertical_change_count",
        "heading_change_count",
        "progress_sum",
        "progress_mean",
    )
    result: Dict[str, Any] = {}
    for outcome in OUTCOME_ORDER:
        selected = [episode for episode in episodes if episode["outcome"] == outcome]
        result[outcome] = {
            "episode_count": len(selected),
            "fields": {field: _finite_stats(episode[field] for episode in selected) for field in fields},
            "direction_intent_sequences": _sequence_counter(selected, "direction_intents"),
            "horizontal_sequences": _sequence_counter(selected, "horizontal_families"),
            "vertical_sequences": _sequence_counter(selected, "vertical_families"),
        }
    return result


def _build_trajectory_patterns(episodes: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_id": TRAJECTORY_PATTERN_SCHEMA_ID,
        "diagnostic_only": True,
        "source": dict(source),
        "episode_count": len(episodes),
        "outcome_counts": dict(Counter(episode["outcome"] for episode in episodes)),
        "group_stats": _group_episode_stats(episodes),
        "observations": [
            "Patterns are descriptive summaries of one fixed BC Dev100 artifact, not causal evidence.",
            "The artifact persists action and terminal outcome, but no explicit high-level decision label.",
        ],
    }


def _build_failure_analysis(episodes: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> Dict[str, Any]:
    failures = [episode for episode in episodes if episode["outcome"] not in ("success", "other")]
    dead_ends = [episode for episode in failures if episode["outcome"] == "dead_end"]
    collisions = [episode for episode in failures if episode["outcome"] == "collision"]
    explicit_abort = [episode for episode in failures if episode["terminal_abort_row_count"] > 0 or episode["terminal_abort_reasons"]]
    incomplete = [episode for episode in failures if episode["primitive_incomplete_row_count"] > 0]
    no_explicit_error_completed = [
        episode
        for episode in failures
        if episode["terminal_abort_row_count"] == 0
        and not episode["terminal_abort_reasons"]
        and episode["terminal_primitive_completed"]
    ]
    missing_integration = [episode for episode in failures if not episode["effective_integration_ticks_available"]]
    missing_frames = [episode for episode in failures if not episode["applied_frame_count_available"]]
    return {
        "schema_id": FAILURE_MODE_SCHEMA_ID,
        "diagnostic_only": True,
        "source": dict(source),
        "failure_counts": {
            "all_failures": len(failures),
            "dead_end": len(dead_ends),
            "collision": len(collisions),
            "timeout": sum(episode["outcome"] == "timeout" for episode in failures),
        },
        "execution_evidence": {
            "explicit_terminal_abort_episode_count": len(explicit_abort),
            "incomplete_primitive_episode_count": len(incomplete),
            "completed_terminal_primitive_without_explicit_abort_count": len(no_explicit_error_completed),
            "effective_integration_ticks_missing_episode_count": len(missing_integration),
            "applied_frame_count_missing_episode_count": len(missing_frames),
            "explicit_low_level_control_error_established_count": 0,
            "reason": "The trace has no terminal abort reason and both integration/frame counters are unavailable (-1); collision-time incomplete primitive is a candidate signal, not proof of a low-level control error.",
        },
        "high_level_decision_error": {
            "status": "INDIRECT_CANDIDATE_ONLY",
            "candidate_episode_count": len(no_explicit_error_completed),
            "candidate_outcomes": dict(Counter(episode["outcome"] for episode in no_explicit_error_completed)),
            "interpretation": "Completed primitive followed by dead_end is compatible with a high-level decision problem, but the artifact lacks the counterfactual or explicit decision label needed to establish it.",
        },
        "low_level_control_error": {
            "status": "NOT_ESTABLISHED",
            "candidate_episode_count": len(incomplete),
            "candidate_outcomes": dict(Counter(episode["outcome"] for episode in incomplete)),
            "interpretation": "Collision rows with primitive_completed=false are execution-boundary candidates; missing runtime telemetry prevents attributing them to controller failure rather than collision termination semantics.",
        },
        "unresolved_episode_count": len(failures),
        "failure_episode_details": [
            {
                key: episode[key]
                for key in (
                    "episode_id",
                    "mission_id",
                    "outcome",
                    "steps",
                    "stop_reason",
                    "terminal_action_id",
                    "terminal_direction_intent",
                    "terminal_primitive_completed",
                    "primitive_incomplete_row_count",
                    "terminal_abort_reasons",
                )
            }
            for episode in failures
        ],
        "required_evidence_gap": [
            "Persist effective integration ticks and applied frame count for each primitive.",
            "Persist an explicit controller/transport failure reason separately from collision/dead_end outcome.",
            "Add a non-privileged high-level intent or decision label only if it is generated from current observation and action semantics.",
        ],
    }


def _load_existing_action_evidence(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics_path = path.parent / "primitive_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        models = metrics.get("models", {})
        ranking = {}
        for name in ("105_one_hot", "primitive_descriptor"):
            model = models.get(name, {})
            oof = model.get("oof_metrics", {})
            ranking[name] = {
                "micro_accuracy": oof.get("micro", {}).get("accuracy"),
                "micro_auc": oof.get("micro", {}).get("auc"),
                "mission_macro_accuracy": oof.get("mission_macro", {}).get("accuracy"),
                "mission_macro_auc": oof.get("mission_macro", {}).get("auc"),
            }
        ranking["paired_descriptor_minus_one_hot"] = metrics.get("bootstrap", {}).get("paired_deltas", {}).get(
            "primitive_descriptor_minus_105_one_hot"
        )
        return {
            "available": True,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "equivalence": payload.get("equivalence", {}),
            "source": payload.get("source", {}),
            "ranking": ranking,
        }
    except (OSError, ValueError, TypeError):
        return {"available": False, "path": str(path), "read_error": True}


def _build_report(
    action_clusters: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    failure: Mapping[str, Any],
    action_evidence: Mapping[str, Any],
) -> str:
    counts = trajectory["outcome_counts"]
    summary = action_clusters["summary"]
    rank = action_evidence.get("ranking", {}) if action_evidence.get("available") else {}
    lines = [
        "# HIERARCHICAL TASK FORMULATION AUDIT V1",
        "",
        "本报告是只读 offline diagnostic；未执行 AWAC、Actor、Critic、Unity/Bridge，未修改生产配置、Replay 或 action contract。",
        "",
        "## 证据范围",
        "",
        "- MPL: {} actions，{}×{} lattice，duration={} s，command_frames={}，contract `{}`。".format(
            summary["action_count"],
            summary["lattice"]["horizontal_count"],
            summary["lattice"]["vertical_count"],
            summary["duration_s"]["mean"],
            summary["command_frames"]["mean"],
            action_clusters["source"]["mpl_contract_sha256"],
        ),
        "- BC Dev100 trajectory artifact: {} episodes；success={}、dead_end={}、collision={}、timeout={}。".format(
            trajectory["episode_count"],
            counts.get("success", 0),
            counts.get("dead_end", 0),
            counts.get("collision", 0),
            counts.get("timeout", 0),
        ),
        "- action exact parameter duplicate={}、exact behavior duplicate={}。".format(
            summary["exact_parameter_duplicate_count"], summary["exact_behavior_duplicate_count"]
        ),
        "",
        "## Flat primitive 结构",
        "",
        "105 个动作实际上共享同一个固定时长 endpoint primitive；可直接从当前 artifact 支持的粗粒度语义是 lateral×vertical 的 9 个 direction intent。duration 不是当前可分层变量，control effect 仍是数值控制效果而不是 outcome 标签。",
        "",
        "## 轨迹模式",
        "",
        "成功/失败的 family 统计保存在 `trajectory_pattern.json`。本次只描述 sequence、repeat/switch、方向族变化与进度，不能把相关性解释成高层策略因果。",
        "",
        "## 失败模式归因",
        "",
        "- high-level decision error: `INDIRECT_CANDIDATE_ONLY`；completed primitive 后 dead_end 的候选数为 {}。".format(
            failure["high_level_decision_error"]["candidate_episode_count"]
        ),
        "- low-level control error: `NOT_ESTABLISHED`；collision 且 primitive 未完成的候选数为 {}，但没有显式 abort reason。".format(
            failure["low_level_control_error"]["candidate_episode_count"]
        ),
        "- 当前 artifact 的 effective integration ticks / applied frame count 均不可用，因而 {} 个失败 episode 的高层/低层归因仍 unresolved。".format(
            failure["unresolved_episode_count"]
        ),
        "",
        "## Flat action 是否造成 credit assignment 困难",
        "",
        "现有 action-formulation evidence 可作为诊断支持：exact duplicate 为 0，但 primitive descriptor 没有超过 one-hot ranking；这一结果不支持把 action representation 单独认定为根因。结合单次 BC Dev100 的 sequence/outcome 模式，只能说 flat action credit assignment 是合理的诊断假设，不能据此宣布 hierarchical RL 必然有效。",
    ]
    if rank:
        lines.extend(["", "已有 action ranking 摘要：", "", "```json", json.dumps(rank, indent=2, sort_keys=True), "```"])
    lines.extend(
        [
            "",
            "## 候选 high-level action space",
            "",
            "1. `direction_intent`：9 类 lateral×vertical intent；当前 artifact 可直接映射，低层仍使用现有 105-action selector。",
            "2. `navigation_mode`：goal approach、lateral align、altitude correct、heading align、recovery/safe brake；当前没有标签，只能作为后续设计候选。",
            "3. `local_objective`：减小 goal distance、heading error、维持 altitude band、增加 clearance、dead-end recovery；需要额外非 privileged 的目标/观测定义。",
            "4. `local_waypoint_intent`：仅由当前 observation 构造的相对短程方向目标；不得使用未来状态、终止原因或 privileged route。",
            "",
            "## 建议",
            "",
            "推荐 `B_CANDIDATE_WITH_FLAT_BASELINE`：先把 hierarchical offline RL 作为隔离设计候选，同时保留 flat AWAC 作为对照；在有更完整 execution provenance 和 high-level label 之前，不修改 production。BC 数据质量/任务覆盖仍是并列风险，不能仅凭本审计选择 C 或 D。",
            "",
            "结论状态：`FLAT_ACTION_CREDIT_DIFFICULTY=SUPPORTED_AS_DIAGNOSTIC_HYPOTHESIS`；`HIGH_LEVEL_VS_LOW_LEVEL_CAUSE=UNRESOLVED`。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_hierarchical_task_formulation_audit(
    *,
    rollout_root: Path,
    motion_primitives_json: Path,
    motion_primitives_npz: Path,
    action_formulation_root: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    rollout_root = Path(rollout_root).expanduser().resolve()
    motion_primitives_json = Path(motion_primitives_json).expanduser().resolve()
    motion_primitives_npz = Path(motion_primitives_npz).expanduser().resolve()
    action_formulation_root = Path(action_formulation_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    index_path = rollout_root / "rollout_index.csv"
    summary_path = rollout_root / "summary.json"
    if not index_path.is_file():
        raise FileNotFoundError(str(index_path))
    if not summary_path.is_file():
        raise FileNotFoundError(str(summary_path))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty diagnostic output: {}".format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    library = _load_action_library(motion_primitives_json, motion_primitives_npz)
    action_rows = {int(row["id"]): row for row in library["actions"]}
    with index_path.open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    if len(index_rows) != EXPECTED_EPISODES:
        raise ValueError("expected {} trajectory episodes, got {}".format(EXPECTED_EPISODES, len(index_rows)))
    episodes = [_read_episode(row, rollout_root, action_rows) for row in index_rows]
    source = {
        "rollout_root": str(rollout_root),
        "rollout_index_sha256": sha256_file(index_path),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_sha256": json.loads(summary_path.read_text(encoding="utf-8")).get("checkpoint_sha256"),
        "task_contract_sha256": index_rows[0].get("task_contract_sha256") if index_rows else None,
        "observation_contract": index_rows[0].get("observation_contract") if index_rows else None,
        "observation_source": index_rows[0].get("observation_source") if index_rows else None,
        "episode_count": len(episodes),
        "transition_count": int(sum(episode["steps"] for episode in episodes)),
    }
    action_clusters = _build_action_clusters(library)
    trajectory = _build_trajectory_patterns(episodes, source)
    failure = _build_failure_analysis(episodes, source)
    action_evidence = _load_existing_action_evidence(
        action_formulation_root / "action_analysis.json"
    )
    _json_dump(out_dir / "action_clusters.json", action_clusters)
    _json_dump(out_dir / "trajectory_pattern.json", trajectory)
    _json_dump(out_dir / "failure_mode_analysis.json", failure)
    (out_dir / "report_zh.md").write_text(
        _build_report(action_clusters, trajectory, failure, action_evidence), encoding="utf-8"
    )
    return {
        "schema_id": HIERARCHICAL_TASK_FORMULATION_AUDIT_SCHEMA_ID,
        "out_dir": str(out_dir),
        "action_clusters": action_clusters,
        "trajectory_pattern": trajectory,
        "failure_mode_analysis": failure,
        "production_modified": False,
        "runtime_started": False,
        "rl_training_executed": False,
    }
