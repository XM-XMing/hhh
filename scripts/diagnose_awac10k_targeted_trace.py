#!/usr/bin/env python3
"""Read-only Critic ranking and targeted shadow-trace diagnosis.

The command loads frozen policy/Critic checkpoints and a read-only replay
prefix.  It never calls an optimizer, writes a checkpoint, changes replay, or
starts Unity.  Optional JSONL inputs are copied into the diagnostic directory
and are summarized without being treated as complete evidence when absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from planning.awac.model import build_actor, build_critic, masked_policy
from planning.awac.replay import AWACReplayBuffer
from planning.bc.model import require_torch
from planning.common.checkpoint import load_torch
from planning.common.hashing import file_sha256


FIXED_PREFIX_ROWS = 5491
REPLAY_FIELDS = (
    "depth",
    "vector",
    "action_mask",
    "action",
    "reward",
    "next_depth",
    "next_vector",
    "next_action_mask",
    "done",
    "behavior_source",
)
EXPECTED_BC_SHA = "ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2"
EXPECTED_CALIBRATION_SHA = "1a0edc9949f8e67290dfd4d84d455123302f44afbebe92e8e80ed4a352dec8ea"
EXPECTED_AWAC_SHA = "abfc553b0bf87bd87d348794e39d525dccdec5e0e00875f74bace336e2da6e3e"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite diagnostic value")
    return result


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise ValueError("diagnostic array contains non-finite values")
    return {
        "count": int(array.size),
        "mean": _finite(array.mean()),
        "p05": _finite(np.percentile(array, 5)),
        "p50": _finite(np.percentile(array, 50)),
        "p95": _finite(np.percentile(array, 95)),
        "max": _finite(array.max()),
    }


def _rank_descending(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, with the largest value ranked first."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < order.size:
        end = cursor + 1
        while end < order.size and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or right.size != left.size:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1.0e-12:
        # A constant vector has no rank/linear association to certify.  Do
        # not turn undefined evidence into a convenient numeric value.
        return None
    return _finite(float(np.dot(left_centered, right_centered) / denominator))


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_rank_descending(left), _rank_descending(right))


def _affine_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return {"alpha": None, "beta": None, "r2": None}
    variance = float(np.dot(x - x.mean(), x - x.mean()))
    if variance <= 1.0e-12:
        return {"alpha": 0.0, "beta": _finite(y.mean()), "r2": 0.0}
    alpha = float(np.dot(x - x.mean(), y - y.mean()) / variance)
    beta = float(y.mean() - alpha * x.mean())
    predicted = alpha * x + beta
    residual = float(np.square(y - predicted).sum())
    total = float(np.square(y - y.mean()).sum())
    r2 = 1.0 if total <= 1.0e-12 and residual <= 1.0e-12 else 1.0 - residual / max(total, 1.0e-12)
    return {"alpha": _finite(alpha), "beta": _finite(beta), "r2": _finite(r2)}


def _checkpoint_state(payload: Mapping[str, Any], actor: bool) -> Mapping[str, Any]:
    key = (
        "actor_state_dict"
        if actor and "actor_state_dict" in payload
        else "model_state_dict"
        if actor
        else "critic1_state_dict"
    )
    if key not in payload or not isinstance(payload[key], Mapping):
        raise ValueError("checkpoint missing {}".format(key))
    return payload[key]


def _load_models(*, torch, nn, bc_payload, calibration_payload, awac_payload):
    depth_channels = int(awac_payload.get("depth_history_frames", 1))
    bc_actor = build_actor(nn, depth_channels=depth_channels).to(torch.device("cpu"))
    bc_actor.load_state_dict(_checkpoint_state(bc_payload, actor=True), strict=True)
    v7_critic1 = build_critic(nn, depth_channels=depth_channels).to(torch.device("cpu"))
    v7_critic2 = build_critic(nn, depth_channels=depth_channels).to(torch.device("cpu"))
    v7_critic1.load_state_dict(_checkpoint_state(calibration_payload, actor=False), strict=True)
    v7_critic2.load_state_dict(calibration_payload["critic2_state_dict"], strict=True)
    awac_actor = build_actor(nn, depth_channels=depth_channels).to(torch.device("cpu"))
    awac_actor.load_state_dict(_checkpoint_state(awac_payload, actor=True), strict=True)
    awac_critic1 = build_critic(nn, depth_channels=depth_channels).to(torch.device("cpu"))
    awac_critic2 = build_critic(nn, depth_channels=depth_channels).to(torch.device("cpu"))
    awac_critic1.load_state_dict(_checkpoint_state(awac_payload, actor=False), strict=True)
    awac_critic2.load_state_dict(awac_payload["critic2_state_dict"], strict=True)
    for module in (bc_actor, v7_critic1, v7_critic2, awac_actor, awac_critic1, awac_critic2):
        module.eval()
    return bc_actor, v7_critic1, v7_critic2, awac_actor, awac_critic1, awac_critic2


def _read_batch(replay: AWACReplayBuffer, offset: int, end: int, *, torch):
    return replay.sample_indices(
        np.arange(int(offset), int(end), dtype=np.int64),
        torch=torch,
        device=torch.device("cpu"),
    )


def _top_k_overlap(left: np.ndarray, right: np.ndarray, valid: np.ndarray, k: int) -> float:
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return 0.0
    k = min(int(k), int(valid_indices.size))
    left_top = set(valid_indices[np.argsort(-left[valid_indices], kind="stable")[:k]].tolist())
    right_top = set(valid_indices[np.argsort(-right[valid_indices], kind="stable")[:k]].tolist())
    return float(len(left_top.intersection(right_top)) / max(1, k))


def _load_jsonl(path: Path | None) -> List[Dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("trace line {} is not an object".format(line_number))
            rows.append(dict(value))
    return rows


def _copy_traces(sources: Sequence[Path], destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as output:
        for source in sources:
            if source.is_file():
                output.write(source.read_text(encoding="utf-8"))


def _trace_category_summary(rows: Sequence[Mapping[str, Any]], episode_ids: Sequence[str]) -> Dict[str, Any]:
    wanted = {str(value) for value in episode_ids}
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        episode_id = str(row.get("episode_id", ""))
        if episode_id in wanted:
            grouped.setdefault(episode_id, []).append(row)
    first_divergence = []
    q_preferences = []
    first_divergence_q_preferences = []
    kls = []
    for episode_id in episode_ids:
        episode_rows = grouped.get(str(episode_id), [])
        disagreements = [row for row in episode_rows if bool(row.get("top1_disagreement", False))]
        first = disagreements[0] if disagreements else None
        if first is not None:
            first_divergence.append({
                "episode_id": str(episode_id),
                "step": int(first.get("step", -1)),
                "bc_to_awac_kl": first.get("bc_to_awac_kl"),
                "primary_top1_probability": first.get("primary_top1_probability"),
                "shadow_top1_probability": first.get("shadow_top1_probability"),
                "q_bc_top1": first.get("q_bc_top1"),
                "q_awac_top1": first.get("q_awac_top1"),
                "q_awac_minus_bc": first.get("q_awac_minus_bc"),
                "critic_prefers_awac": first.get("critic_prefers_awac"),
            })
            if first.get("critic_prefers_awac") is not None:
                first_divergence_q_preferences.append(bool(first["critic_prefers_awac"]))
        for row in episode_rows:
            if row.get("bc_to_awac_kl") is not None:
                kls.append(float(row["bc_to_awac_kl"]))
            q_awac = row.get("q_awac_top1")
            q_bc = row.get("q_bc_top1")
            if q_awac is not None and q_bc is not None:
                q_preferences.append(float(q_awac) > float(q_bc))
    return {
        "requested_episode_ids": [str(value) for value in episode_ids],
        "traced_episode_ids": sorted(grouped),
        "traced_count": len(grouped),
        "runtime_trace_available": bool(grouped),
        "first_divergence_count": len(first_divergence),
        "first_divergence": first_divergence,
        "first_divergence_q_prefers_awac_rate": (
            None
            if not first_divergence_q_preferences
            else float(
                sum(first_divergence_q_preferences) / len(first_divergence_q_preferences)
            )
        ),
        "q_prefers_awac_rate": (
            None if not q_preferences else float(sum(q_preferences) / len(q_preferences))
        ),
        "bc_to_awac_kl_summary": _summary(kls),
    }


def _csv_trace_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    actions = [int(float(row["action"])) for row in rows if row.get("action", "") != ""]
    rewards = [float(row["reward"]) for row in rows if row.get("reward", "") != ""]
    progress = [
        float(row["distance_before"]) - float(row["distance_after"])
        for row in rows
        if row.get("distance_before", "") != "" and row.get("distance_after", "") != ""
    ]
    switches = sum(left != right for left, right in zip(actions, actions[1:]))
    streaks: List[int] = []
    previous = None
    current = 0
    for action in actions:
        if action == previous:
            current += 1
        else:
            if current:
                streaks.append(current)
            current = 1
            previous = action
    if current:
        streaks.append(current)
    terminal = rows[-1] if rows else {}
    if terminal.get("success") == "True":
        outcome = "success"
    elif terminal.get("collision") == "True":
        outcome = "collision"
    elif terminal.get("dead_end") == "True":
        outcome = "dead_end"
    elif terminal.get("timeout") == "True":
        outcome = "timeout"
    else:
        outcome = "unknown"
    return {
        "available": True,
        "path": str(path),
        "steps": len(rows),
        "outcome": outcome,
        "episode_return": float(sum(rewards)),
        "unique_actions": len(set(actions)),
        "action_switch_rate": float(switches / max(1, len(actions) - 1)),
        "max_action_streak": max(streaks, default=0),
        "progress_sum_m": float(sum(progress)),
        "final_distance_after": (
            float(rows[-1]["distance_after"]) if rows and rows[-1].get("distance_after", "") else None
        ),
        "last10_progress_m": float(sum(progress[-10:])),
        "last20_progress_m": float(sum(progress[-20:])),
    }


def _paired_trace_signatures(
    paired: Mapping[str, Any], awac_trace_rows: Sequence[Mapping[str, Any]] = ()
) -> Dict[str, Any]:
    trace_by_episode: Dict[str, List[Mapping[str, Any]]] = {}
    for row in awac_trace_rows:
        trace_by_episode.setdefault(str(row.get("episode_id", "")), []).append(row)
    values = []
    for row in paired.get("awac_timeouts", []):
        episode_id = str(row.get("episode_id", ""))
        awac = row.get("awac_trace", {})
        bc = row.get("bc_trace", {})
        awac_summary = _csv_trace_summary(Path(awac["path"])) if awac.get("path") else {"available": False}
        bc_summary = _csv_trace_summary(Path(bc["path"])) if bc.get("path") else {"available": False}
        episode_rows = trace_by_episode.get(episode_id, [])
        disagreements = [item for item in episode_rows if bool(item.get("top1_disagreement", False))]
        first = disagreements[0] if disagreements else None
        late = episode_rows[-10:]
        late_q_preferences = [
            bool(item["critic_prefers_awac"])
            for item in late
            if item.get("critic_prefers_awac") is not None
        ]
        first_divergence = None
        if first is not None:
            first_divergence = {
                "step": int(first.get("step", -1)),
                "awac_action": first.get("primary_action"),
                "bc_action": first.get("shadow_action"),
                "awac_probability_for_awac_action": first.get("primary_selected_probability"),
                "bc_probability_for_bc_action": first.get("shadow_selected_probability"),
                "awac_probability_for_bc_action": None,
                "bc_probability_for_awac_action": None,
                "q10k_awac_action": first.get("q_awac_top1"),
                "q10k_bc_action": first.get("q_bc_top1"),
                "q10k_awac_minus_bc": first.get("q_awac_minus_bc"),
                "v7_q_awac_action": None,
                "v7_q_bc_action": None,
                "v7_q_awac_minus_bc": None,
                "distance_before": first.get("distance_before"),
                "distance_after": first.get("distance_after"),
                "progress_m": first.get("progress_m"),
                "minimum_safety_clearance_before_m": first.get(
                    "minimum_safety_clearance_before_m"
                ),
                "minimum_safety_clearance_after_m": first.get(
                    "minimum_safety_clearance_after_m"
                ),
            }
        values.append({
            "episode_id": episode_id,
            "mission_id": str(row.get("mission_id", "")),
            "awac": awac_summary,
            "bc": bc_summary,
            "first_divergence": first_divergence,
            "late_timeout_phase": {
                "trace_rows": len(late),
                "progress_sum_m": (
                    None
                    if not late
                    else float(sum(float(item.get("progress_m", 0.0)) for item in late))
                ),
                "q_prefers_awac_count": int(sum(late_q_preferences)),
                "q_preference_rate": (
                    None
                    if not late_q_preferences
                    else float(sum(late_q_preferences) / len(late_q_preferences))
                ),
            },
        })
    awac_switch = [item["awac"]["action_switch_rate"] for item in values if item["awac"].get("available")]
    awac_streak = [item["awac"]["max_action_streak"] for item in values if item["awac"].get("available")]
    awac_progress = [item["awac"]["progress_sum_m"] for item in values if item["awac"].get("available")]
    return {
        "source": "existing_primary_evaluation_csv",
        "timeout_count": len(values),
        "episodes": values,
        "awac_action_switch_rate": _summary(awac_switch),
        "awac_max_action_streak": _summary(awac_streak),
        "awac_progress_sum_m": _summary(awac_progress),
        "loop_proven": False,
        "classification": "INCONCLUSIVE",
        "interpretation": "primary-only timeout traces do not prove an action loop without shadow state alignment",
    }


def _critic_decomposition(
    *,
    replay: AWACReplayBuffer,
    torch,
    bc_actor,
    v7_critic1,
    v7_critic2,
    awac_actor,
    awac_critic1,
    awac_critic2,
    rows: int,
    batch_size: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    flat_v7: List[float] = []
    flat_awac: List[float] = []
    centered_v7: List[float] = []
    centered_awac: List[float] = []
    per_state_rank: List[float] = []
    per_state_rank_unavailable_count = 0
    top1_agreement = []
    top3_overlap = []
    top5_overlap = []
    bc_top1_actions: List[int] = []
    awac_top1_actions: List[int] = []
    policy_disagreement_deltas: List[Tuple[float, float]] = []
    with torch.no_grad():
        for offset in range(0, int(rows), int(batch_size)):
            end = min(int(rows), offset + int(batch_size))
            batch = _read_batch(replay, offset, end, torch=torch)
            masks = batch["action_mask"].bool().cpu().numpy().astype(bool)
            v7_q = torch.minimum(
                v7_critic1(batch["depth"], batch["vector"]),
                v7_critic2(batch["depth"], batch["vector"]),
            ).cpu().numpy().astype(np.float64)
            awac_q = torch.minimum(
                awac_critic1(batch["depth"], batch["vector"]),
                awac_critic2(batch["depth"], batch["vector"]),
            ).cpu().numpy().astype(np.float64)
            bc_prob, _, _ = masked_policy(
                bc_actor(batch["depth"], batch["vector"]), batch["action_mask"], torch
            )
            awac_prob, _, _ = masked_policy(
                awac_actor(batch["depth"], batch["vector"]), batch["action_mask"], torch
            )
            bc_top1 = bc_prob.argmax(dim=1).cpu().numpy().astype(np.int64)
            awac_top1 = awac_prob.argmax(dim=1).cpu().numpy().astype(np.int64)
            bc_top1_actions.extend(bc_top1.tolist())
            awac_top1_actions.extend(awac_top1.tolist())
            for row_index, valid in enumerate(masks):
                valid_indices = np.flatnonzero(valid)
                if valid_indices.size == 0:
                    continue
                v7_values = v7_q[row_index, valid]
                awac_values = awac_q[row_index, valid]
                flat_v7.extend(v7_values.tolist())
                flat_awac.extend(awac_values.tolist())
                centered_v7.extend((v7_values - v7_values.mean()).tolist())
                centered_awac.extend((awac_values - awac_values.mean()).tolist())
                rank_association = _spearman(v7_values, awac_values)
                if rank_association is None:
                    per_state_rank_unavailable_count += 1
                else:
                    per_state_rank.append(rank_association)
                v7_order = valid_indices[np.argsort(-v7_q[row_index, valid], kind="stable")]
                awac_order = valid_indices[np.argsort(-awac_q[row_index, valid], kind="stable")]
                top1_agreement.append(bool(v7_order[0] == awac_order[0]))
                top3_overlap.append(_top_k_overlap(v7_q[row_index], awac_q[row_index], valid, 3))
                top5_overlap.append(_top_k_overlap(v7_q[row_index], awac_q[row_index], valid, 5))
                if int(bc_top1[row_index]) != int(awac_top1[row_index]):
                    bc_action = int(bc_top1[row_index])
                    awac_action = int(awac_top1[row_index])
                    policy_disagreement_deltas.append((
                        float(v7_q[row_index, awac_action] - v7_q[row_index, bc_action]),
                        float(awac_q[row_index, awac_action] - awac_q[row_index, bc_action]),
                    ))
    flat_v7_array = np.asarray(flat_v7, dtype=np.float64)
    flat_awac_array = np.asarray(flat_awac, dtype=np.float64)
    affine = _affine_fit(flat_v7_array, flat_awac_array)
    raw_ratio = float(
        np.percentile(np.abs(flat_awac_array), 95)
        / max(np.percentile(np.abs(flat_v7_array), 95), 1.0e-12)
    )
    scale = {
        "fixed_replay_contract": "calibration_replay_prefix",
        "rows": int(rows),
        "valid_action_values": int(flat_v7_array.size),
        "qmin_v7_summary": _summary(flat_v7),
        "qmin_awac10k_summary": _summary(flat_awac),
        "raw_q_abs_p95_ratio_awac10k_over_v7": _finite(raw_ratio),
        "global_pearson": _pearson(flat_v7_array, flat_awac_array),
        "global_spearman": _spearman(flat_v7_array, flat_awac_array),
        "affine_fit_q_awac10k_equals_alpha_q_v7_plus_beta": affine,
        "centered_qmin_v7_summary": _summary(centered_v7),
        "centered_qmin_awac10k_summary": _summary(centered_awac),
        "centered_qmin_pearson": _pearson(
            np.asarray(centered_v7, dtype=np.float64),
            np.asarray(centered_awac, dtype=np.float64),
        ),
        "interpretation": "raw scale/offset is not treated as action degradation unless rank metrics also change",
    }
    ranking = {
        "fixed_replay_contract": "calibration_replay_prefix",
        "rows": int(rows),
        "per_state_q_rank_spearman": _summary(per_state_rank),
        "per_state_q_rank_spearman_unavailable_count": int(
            per_state_rank_unavailable_count
        ),
        "q_top1_action_agreement_v7_vs_10k": float(np.mean(top1_agreement)) if top1_agreement else None,
        "q_argmax_change_rate": (
            None
            if not top1_agreement
            else float(1.0 - float(np.mean(top1_agreement)))
        ),
        "q_argmax_change_count": int(len(top1_agreement) - sum(top1_agreement)),
        "q_argmax_comparable_state_count": int(len(top1_agreement)),
        "q_top3_overlap_intersection_over_k": float(np.mean(top3_overlap)) if top3_overlap else None,
        "q_top5_overlap_intersection_over_k": float(np.mean(top5_overlap)) if top5_overlap else None,
        "top1_agreement_count": int(sum(top1_agreement)),
        "top1_disagreement_count": int(len(top1_agreement) - sum(top1_agreement)),
        "policy_top1_disagreement_count": int(sum(left != right for left, right in zip(bc_top1_actions, awac_top1_actions))),
        "policy_top1_disagreement_rate": float(
            sum(left != right for left, right in zip(bc_top1_actions, awac_top1_actions))
            / max(1, len(bc_top1_actions))
        ),
        "rank_definition": "descending average ranks over valid actions; invalid actions excluded",
        "q_argmax_change_rate_definition": "changed Qmin argmax / comparable states, comparing V7 Critic with AWAC Critic",
        "top_k_overlap_definition": "intersection divided by k, with k clipped to valid action count",
    }
    deltas_v7 = [value[0] for value in policy_disagreement_deltas]
    deltas_awac = [value[1] for value in policy_disagreement_deltas]
    v7_prefers_awac = [value > 0.0 for value in deltas_v7]
    awac_prefers_awac = [value > 0.0 for value in deltas_awac]
    action_preference_changed = [
        left != right for left, right in zip(v7_prefers_awac, awac_prefers_awac)
    ]
    disagreement = {
        "fixed_replay_contract": "calibration_replay_prefix",
        "states_where_bc_top1_differs_awac_top1": len(policy_disagreement_deltas),
        "q_v7_awac_minus_bc_delta": _summary(deltas_v7),
        "q_awac10k_awac_minus_bc_delta": _summary(deltas_awac),
        "v7_fraction_prefers_awac_top1": (
            None if not v7_prefers_awac else float(sum(v7_prefers_awac) / len(v7_prefers_awac))
        ),
        "awac10k_fraction_prefers_awac_top1": (
            None if not awac_prefers_awac else float(sum(awac_prefers_awac) / len(awac_prefers_awac))
        ),
        "action_preference_flip_on_policy_disagreement": (
            None
            if not action_preference_changed
            else float(
                sum(action_preference_changed) / len(action_preference_changed)
            )
        ),
        "action_preference_flip_count": int(sum(action_preference_changed)),
        "action_preference_flip_denominator": int(len(action_preference_changed)),
        "preference_definition": "critic prefers AWAC when Qmin(AWAC actor top1)-Qmin(BC actor top1) > 0",
    }
    return scale, ranking, disagreement


def _diagnosis(scale, ranking, disagreement, trace_summaries, timeout_signature) -> Dict[str, Any]:
    q_argmax_change_rate = ranking.get("q_argmax_change_rate")
    action_preference_flip = disagreement.get(
        "action_preference_flip_on_policy_disagreement"
    )
    ranking_change_observed = (
        "YES"
        if any(
            value is not None and float(value) > 0.0
            for value in (q_argmax_change_rate, action_preference_flip)
        )
        else "NO"
        if any(value is not None for value in (q_argmax_change_rate, action_preference_flip))
        else "UNKNOWN"
    )
    regressed = trace_summaries["regressed"]
    recovered = trace_summaries["recovered"]
    runtime_trace = bool(regressed.get("runtime_trace_available") and recovered.get("runtime_trace_available"))
    critical_drift = "INCONCLUSIVE"
    if runtime_trace:
        regressed_kl = regressed.get("bc_to_awac_kl_summary", {}).get("mean")
        recovered_kl = recovered.get("bc_to_awac_kl_summary", {}).get("mean")
        if regressed_kl is not None and recovered_kl is not None:
            critical_drift = "YES" if regressed_kl > recovered_kl * 1.25 else "NO"
    timeout_available = bool(timeout_signature.get("timeout_count", 0))
    timeout_pathology = "INCONCLUSIVE" if timeout_available else "UNKNOWN"
    if critical_drift == "YES":
        primary = "CRITICAL_STATE_POLICY_DRIFT"
        secondary = "TIMEOUT_CONTROL_BEHAVIOR"
        next_action = "TUNE_ACTOR_REGULARIZATION"
    else:
        primary = "INCONCLUSIVE"
        secondary = "TIMEOUT_CONTROL_BEHAVIOR"
        next_action = "COLLECT_ACTION_RETURN_EVIDENCE_BEFORE_CRITIC_TUNING"
    return {
        "ranking_change_observed": ranking_change_observed,
        "q_vs_behavior_mc_association": "NOT_MEASURED",
        "same_state_action_value_rank_validated": "NO",
        "ranking_error_causality": "INCONCLUSIVE",
        "critic_action_ranking_degradation": (
            "OBSERVED_UNVALIDATED"
            if ranking_change_observed == "YES"
            else "NO_STATIC_CHANGE_OBSERVED"
            if ranking_change_observed == "NO"
            else "INCONCLUSIVE"
        ),
        "critic_raw_scale_inflation": (
            "SUPPORTED"
            if scale.get("raw_q_abs_p95_ratio_awac10k_over_v7", 1.0) > 1.0
            else "NOT_SUPPORTED"
        ),
        "critic_raw_scale_inflation_causality": (
            "INCONCLUSIVE"
        ),
        "critical_state_policy_drift": critical_drift,
        "timeout_control_pathology": timeout_pathology,
        "timeout_loop_signature": str(
            timeout_signature.get("classification", "INCONCLUSIVE")
        ),
        "primary_root_cause": primary,
        "secondary_root_cause": secondary,
        "confidence": "MEDIUM" if critical_drift == "YES" else "LOW",
        "next_action": next_action,
        "evidence_limits": [
            "No live shadow conclusion is made when evaluator JSONL is absent.",
            "Primary-only timeout CSVs do not prove a control loop.",
            "Q scale drift alone is not classified as action-ranking degradation.",
            "Static Q ranking changes have no same-state action-return or counterfactual validation.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", required=True, type=Path)
    parser.add_argument("--calibration-checkpoint", required=True, type=Path)
    parser.add_argument("--awac-checkpoint", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--calibration-replay-dir", required=True, type=Path)
    parser.add_argument("--paired-audit", required=True, type=Path)
    parser.add_argument("--bc-trace-jsonl", action="append", default=[], type=Path)
    parser.add_argument("--awac-trace-jsonl", action="append", default=[], type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    paths = {
        name: Path(getattr(args, name)).expanduser().resolve()
        for name in (
            "bc_checkpoint",
            "calibration_checkpoint",
            "awac_checkpoint",
            "replay_dir",
            "calibration_replay_dir",
            "paired_audit",
            "out_dir",
        )
    }
    bc_traces = [Path(value).expanduser().resolve() for value in args.bc_trace_jsonl]
    awac_traces = [Path(value).expanduser().resolve() for value in args.awac_trace_jsonl]
    for name in ("bc_checkpoint", "calibration_checkpoint", "awac_checkpoint", "paired_audit"):
        if not paths[name].is_file():
            raise FileNotFoundError(str(paths[name]))
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    torch, nn, _, _, _ = require_torch()
    torch.set_num_threads(int(args.cpu_threads))
    bc_payload = load_torch(paths["bc_checkpoint"], torch=torch, map_location="cpu")
    calibration_payload = load_torch(paths["calibration_checkpoint"], torch=torch, map_location="cpu")
    awac_payload = load_torch(paths["awac_checkpoint"], torch=torch, map_location="cpu")
    expected_hashes = {
        "bc_checkpoint": EXPECTED_BC_SHA,
        "calibration_checkpoint": EXPECTED_CALIBRATION_SHA,
        "awac_checkpoint": EXPECTED_AWAC_SHA,
    }
    actual_hashes = {name: file_sha256(paths[name]) for name in expected_hashes}
    for name, expected in expected_hashes.items():
        if actual_hashes[name] != expected:
            raise ValueError("{} SHA mismatch".format(name))
    paired = json.loads(paths["paired_audit"].read_text(encoding="utf-8"))
    regressed_ids = [str(row["episode_id"]) for row in paired.get("regressed", [])]
    recovered_ids = [str(row["episode_id"]) for row in paired.get("recovered", [])]
    timeout_ids = [str(row["episode_id"]) for row in paired.get("awac_timeouts", [])]
    if len(regressed_ids) != 15 or len(recovered_ids) != 8 or len(timeout_ids) != 6:
        raise ValueError("paired audit target counts do not match 15/8/6 contract")
    models = _load_models(
        torch=torch,
        nn=nn,
        bc_payload=bc_payload,
        calibration_payload=calibration_payload,
        awac_payload=awac_payload,
    )
    bc_actor, v7_critic1, v7_critic2, awac_actor, awac_critic1, awac_critic2 = models
    replay = AWACReplayBuffer.open(paths["replay_dir"], read_only=True)
    calibration_replay = AWACReplayBuffer.open(paths["calibration_replay_dir"], read_only=True)
    try:
        if int(calibration_replay.size) < FIXED_PREFIX_ROWS or int(replay.size) < FIXED_PREFIX_ROWS:
            raise ValueError("replay does not contain fixed calibration prefix")
        prefix_checks = {}
        for field in REPLAY_FIELDS:
            prefix_checks[field] = bool(
                np.array_equal(
                    calibration_replay.arrays[field][:FIXED_PREFIX_ROWS],
                    replay.arrays[field][:FIXED_PREFIX_ROWS],
                )
            )
        if not all(prefix_checks.values()):
            raise ValueError("AWAC replay prefix differs from calibration replay")
        scale, ranking, disagreement = _critic_decomposition(
            replay=calibration_replay,
            torch=torch,
            bc_actor=bc_actor,
            v7_critic1=v7_critic1,
            v7_critic2=v7_critic2,
            awac_actor=awac_actor,
            awac_critic1=awac_critic1,
            awac_critic2=awac_critic2,
            rows=FIXED_PREFIX_ROWS,
            batch_size=int(args.batch_size),
        )
    finally:
        replay.close()
        calibration_replay.close()
    bc_trace_rows = [row for path in bc_traces for row in _load_jsonl(path)]
    awac_trace_rows = [row for path in awac_traces for row in _load_jsonl(path)]
    _copy_traces(bc_traces, paths["out_dir"] / "trace_bc_primary.jsonl")
    _copy_traces(awac_traces, paths["out_dir"] / "trace_awac_primary.jsonl")
    all_trace_rows = bc_trace_rows + awac_trace_rows
    trace_summaries = {
        "regressed": _trace_category_summary(all_trace_rows, regressed_ids),
        "recovered": _trace_category_summary(all_trace_rows, recovered_ids),
        "timeout": _trace_category_summary(all_trace_rows, timeout_ids),
    }
    timeout_signature = _paired_trace_signatures(paired, awac_trace_rows)
    trace_summaries["timeout"]["timeout_runtime_details"] = timeout_signature
    first_divergence = {
        "regressed": trace_summaries["regressed"],
        "recovered": trace_summaries["recovered"],
        "timeout": trace_summaries["timeout"],
        "comparison": {
            "regressed_runtime_trace_available": trace_summaries["regressed"]["runtime_trace_available"],
            "recovered_runtime_trace_available": trace_summaries["recovered"]["runtime_trace_available"],
            "timeout_runtime_trace_available": trace_summaries["timeout"]["runtime_trace_available"],
        },
    }
    diagnosis = _diagnosis(scale, ranking, disagreement, trace_summaries, timeout_signature)
    diagnosis.update({
        "fixed_replay_prefix_rows": FIXED_PREFIX_ROWS,
        "source_hashes": actual_hashes,
        "paired_target_counts": {
            "regressed": len(regressed_ids),
            "recovered": len(recovered_ids),
            "timeout": len(timeout_ids),
        },
        "critic_scale_decomposition": scale,
        "critic_ranking_audit": ranking,
        "policy_disagreement_q_audit": disagreement,
        "first_divergence_analysis": first_divergence,
        "timeout_loop_evidence": timeout_signature,
        "runtime_trace_inputs": {
            "bc_primary": [str(path) for path in bc_traces],
            "awac_primary": [str(path) for path in awac_traces],
        },
        "training_executed": False,
        "checkpoint_modified": False,
        "replay_modified": False,
        "final300_used": False,
    })
    _json(paths["out_dir"] / "critic_scale_decomposition.json", scale)
    _json(paths["out_dir"] / "critic_ranking_audit.json", ranking)
    _json(paths["out_dir"] / "regressed_trace_summary.json", trace_summaries["regressed"])
    _json(paths["out_dir"] / "recovered_trace_summary.json", trace_summaries["recovered"])
    _json(paths["out_dir"] / "timeout_trace_summary.json", trace_summaries["timeout"])
    _json(paths["out_dir"] / "first_divergence_analysis.json", first_divergence)
    _json(paths["out_dir"] / "diagnosis.json", diagnosis)
    lines = [
        "# Standard AWAC targeted trace and Critic ranking diagnosis",
        "",
        "- Fixed replay prefix rows: `{}`".format(FIXED_PREFIX_ROWS),
        "- Regressed / recovered / timeout targets: `{}` / `{}` / `{}`".format(
            len(regressed_ids), len(recovered_ids), len(timeout_ids)
        ),
        "- Raw Q absolute p95 ratio: `{}`".format(scale["raw_q_abs_p95_ratio_awac10k_over_v7"]),
        "- Raw Q scale inflation: `{}`; scale-only causality: `{}`".format(
            diagnosis["critic_raw_scale_inflation"],
            diagnosis["critic_raw_scale_inflation_causality"],
        ),
        "- Global Pearson / Spearman: `{}` / `{}`".format(
            scale["global_pearson"], scale["global_spearman"]
        ),
        "- Per-state rank Spearman mean: `{}`".format(
            ranking["per_state_q_rank_spearman"]["mean"]
        ),
        "- Q top-1 agreement: `{}`".format(ranking["q_top1_action_agreement_v7_vs_10k"]),
        "- Q top-3 overlap: `{}`".format(ranking["q_top3_overlap_intersection_over_k"]),
        "- Q argmax change rate: `{}`".format(
            ranking["q_argmax_change_rate"]
        ),
        "- Action-preference flip on policy disagreements: `{}`".format(
            disagreement["action_preference_flip_on_policy_disagreement"]
        ),
        "- Live shadow trace regressed/recovered/timeout: `{}` / `{}` / `{}`".format(
            trace_summaries["regressed"]["runtime_trace_available"],
            trace_summaries["recovered"]["runtime_trace_available"],
            trace_summaries["timeout"]["runtime_trace_available"],
        ),
        "- First-divergence Q prefers AWAC regressed/recovered/timeout: `{}` / `{}` / `{}`".format(
            trace_summaries["regressed"]["first_divergence_q_prefers_awac_rate"],
            trace_summaries["recovered"]["first_divergence_q_prefers_awac_rate"],
            trace_summaries["timeout"]["first_divergence_q_prefers_awac_rate"],
        ),
        "- Critical-state policy drift: `{}`".format(diagnosis["critical_state_policy_drift"]),
        "- Ranking change observed / same-state action-value rank validated / causality: `{}` / `{}` / `{}`".format(
            diagnosis["ranking_change_observed"],
            diagnosis["same_state_action_value_rank_validated"],
            diagnosis["ranking_error_causality"],
        ),
        "- Timeout loop signature: `{}`".format(diagnosis["timeout_loop_signature"]),
        "- Primary root cause: `{}`".format(diagnosis["primary_root_cause"]),
        "- Secondary root cause: `{}`".format(diagnosis["secondary_root_cause"]),
        "- Confidence: `{}`".format(diagnosis["confidence"]),
        "- Next action: `{}`".format(diagnosis["next_action"]),
        "",
        "The report is diagnostic-only. No training, checkpoint mutation, replay mutation, or Final300 evaluation was performed.",
    ]
    (paths["out_dir"] / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "out_dir": str(paths["out_dir"]),
        "scale": scale,
        "ranking": ranking,
        "disagreement": disagreement,
        "trace_summaries": trace_summaries,
        "timeout_signature": timeout_signature,
        "diagnosis": diagnosis,
    }


def main(argv: Iterable[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print("AWAC10K_TARGETED_TRACE_AND_CRITIC_DIAGNOSIS=PASS")
    print(json.dumps({
        "out_dir": result["out_dir"],
        "primary_root_cause": result["diagnosis"]["primary_root_cause"],
        "confidence": result["diagnosis"]["confidence"],
        "next_action": result["diagnosis"]["next_action"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
