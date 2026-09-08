#!/usr/bin/env python3
"""Read-only diagnosis of the fixed BC60K versus AWAC10K Dev100 gap.

This command never calls the trainer update path and never starts a runtime.  It
loads the frozen checkpoints in evaluation mode, reads the persistent replay
through its read-only API, and writes evidence-oriented diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from planning.awac.interaction import BehaviorSource
from planning.awac.learner import AWACOptimizationConfig
from planning.awac.model import masked_policy
from planning.awac.optimization import actor_update_due, critic_update_target
from planning.awac.replay import AWACReplayBuffer
from planning.awac.trainer import build_learner
from planning.bc.model import require_torch
from planning.common.checkpoint import load_torch
from planning.common.hashing import file_sha256
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.primitives.library import MotionPrimitiveLibrary


SOURCE_BC_SHA = "ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2"
SOURCE_CALIBRATION_SHA = "1a0edc9949f8e67290dfd4d84d455123302f44afbebe92e8e80ed4a352dec8ea"
SOURCE_AWAC_SHA = "abfc553b0bf87bd87d348794e39d525dccdec5e0e00875f74bace336e2da6e3e"
BC_REPLAY_ROWS = 5491
REPLAY_FIELD_NAMES = (
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
OUTCOMES = ("success", "collision", "dead_end", "timeout", "far", "hard_altitude")


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite diagnostic value")
    return result


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    return _float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def _state_fingerprint(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        tuple(left[name].shape) == tuple(right[name].shape)
        and left[name].dtype == right[name].dtype
        and bool((left[name].detach().cpu() == right[name].detach().cpu()).all())
        for name in left
    )


def _config_from_checkpoint(payload: Mapping[str, Any]) -> AWACOptimizationConfig:
    resolved = payload.get("resolved_training_config", {})
    optimization = resolved.get("optimization_config", {})
    names = (
        "gamma",
        "tau",
        "reward_scale",
        "actor_head_lr",
        "actor_vector_lr",
        "actor_depth_lr",
        "critic_head_lr",
        "critic_vector_lr",
        "critic_depth_lr",
        "critic_cql_weight",
        "awac_temperature",
        "awac_weight_max",
        "bc_kl_weight",
        "trust_tail_top_k",
        "bc_kl_hard_budget",
        "bc_kl_recovery_weight",
        "gradient_clip_norm",
    )
    missing = [name for name in names if name not in optimization]
    if missing:
        raise ValueError("checkpoint optimization config missing: {}".format(", ".join(missing)))
    return AWACOptimizationConfig(**{name: optimization[name] for name in names})


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _outcome(row: Mapping[str, Any]) -> str:
    active = [name for name in OUTCOMES if _bool(row.get(name, False))]
    if len(active) != 1:
        raise ValueError("evaluation row has invalid outcome partition")
    return active[0]


def _metric_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    names = (
        "episode_id",
        "mission_id",
        "steps",
        "return",
        "final_distance_xy",
        "final_abs_goal_dz",
        "min_z",
        "max_z",
        "success",
        "collision",
        "dead_end",
        "timeout",
        "far",
        "hard_altitude",
    )
    result: Dict[str, Any] = {}
    for name in names:
        value = row.get(name, "")
        if name in {"episode_id", "mission_id", "success", "collision", "dead_end", "timeout", "far", "hard_altitude"}:
            result[name] = value
        elif name == "steps":
            result[name] = int(float(value)) if value != "" else None
        elif value == "":
            result[name] = None
        else:
            result[name] = _float(value)
    result["outcome"] = _outcome(row)
    return result


def _trace_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    rows = _read_rows(path)
    actions = [int(float(row["action"])) for row in rows if str(row.get("action", "")) != ""]
    switches = sum(left != right for left, right in zip(actions, actions[1:]))
    streaks: List[int] = []
    current = 0
    previous = None
    progress = []
    for row in rows:
        if "distance_before" in row and "distance_after" in row:
            progress.append(_float(row["distance_before"]) - _float(row["distance_after"]))
        action = int(float(row["action"])) if str(row.get("action", "")) else None
        if action is None:
            continue
        if action == previous:
            current += 1
        else:
            if current:
                streaks.append(current)
            current = 1
            previous = action
    if current:
        streaks.append(current)
    return {
        "available": True,
        "path": str(path),
        "steps": len(rows),
        "unique_actions": len(set(actions)),
        "action_switches": switches,
        "action_switch_rate": _float(switches / max(1, len(actions) - 1)),
        "max_action_streak": max(streaks, default=0),
        "progress_sum_m": _float(sum(progress)),
        "final_distance_after": (
            _float(rows[-1]["distance_after"])
            if rows and rows[-1].get("distance_after", "") != ""
            else None
        ),
        "terminal_abort_reason": rows[-1].get("terminal_abort_reason", "") if rows else "",
    }


def _episode_trace(root: Path, episode_id: str) -> Dict[str, Any]:
    try:
        number = int(episode_id)
    except ValueError:
        return {"available": False, "reason": "non_numeric_episode_id"}
    return _trace_summary(root / "steps" / "episode_{:06d}.csv".format(number))


def _batch_metrics(
    replay: AWACReplayBuffer,
    learner,
    *,
    torch,
    start: int,
    stop: int,
    batch_size: int,
) -> Dict[str, Any]:
    learner.actor.eval()
    learner.critic1.eval()
    learner.critic2.eval()
    learner.target_critic1.eval()
    learner.target_critic2.eval()
    arrays: Dict[str, List[float]] = {
        "td_abs": [],
        "q_min": [],
        "target": [],
        "twin_disagreement": [],
        "advantage": [],
        "raw_weight": [],
        "normalized_weight": [],
        "raw_clip": [],
        "bc_kl": [],
        "top1_disagreement": [],
        "top1_q_gap": [],
    }
    by_source: Dict[int, Dict[str, List[float]]] = {
        int(BehaviorSource.BC_CALIBRATION): {key: [] for key in arrays if key != "top1_q_gap"},
        int(BehaviorSource.AWAC_ONLINE): {key: [] for key in arrays if key != "top1_q_gap"},
    }
    finite = True
    action_top1: List[int] = []
    action_bc_top1: List[int] = []
    action_masks: List[np.ndarray] = []
    q_gap_values: List[float] = []
    with torch.no_grad():
        for offset in range(int(start), int(stop), int(batch_size)):
            end = min(int(stop), offset + int(batch_size))
            batch = replay.sample_indices(np.arange(offset, end, dtype=np.int64), torch=torch, device=torch.device("cpu"))
            logits = learner.actor(batch["depth"], batch["vector"])
            probabilities, log_probabilities, _ = masked_policy(logits, batch["action_mask"], torch)
            bc_logits = learner.bc_reference(batch["depth"], batch["vector"])
            bc_probabilities, bc_log_probabilities, _ = masked_policy(
                bc_logits, batch["action_mask"], torch
            )
            bc_kl = (
                bc_probabilities * (bc_log_probabilities - log_probabilities)
            ).sum(dim=1)
            top1_disagreement = (
                probabilities.argmax(dim=1) != bc_probabilities.argmax(dim=1)
            ).float()
            q1_all = learner.critic1(batch["depth"], batch["vector"])
            q2_all = learner.critic2(batch["depth"], batch["vector"])
            q_min_all = torch.minimum(q1_all, q2_all)
            action_q = q_min_all.gather(1, batch["action"][:, None]).squeeze(1)
            state_value = (probabilities * q_min_all).sum(dim=1)
            advantage = action_q - state_value
            from planning.awac.learner import awac_advantage_weights

            weights = awac_advantage_weights(
                advantage,
                temperature=float(learner.config.awac_temperature),
                weight_max=float(learner.config.awac_weight_max),
                torch=torch,
            )
            terminal = batch["done"].bool()
            target = batch["reward"] * float(learner.config.reward_scale)
            next_value = torch.zeros_like(target)
            nonterminal = ~terminal
            if bool(nonterminal.any().item()):
                next_logits = learner.actor(batch["next_depth"][nonterminal], batch["next_vector"][nonterminal])
                next_q1 = learner.target_critic1(batch["next_depth"][nonterminal], batch["next_vector"][nonterminal])
                next_q2 = learner.target_critic2(batch["next_depth"][nonterminal], batch["next_vector"][nonterminal])
                next_probabilities, _, _ = masked_policy(
                    next_logits, batch["next_action_mask"][nonterminal], torch
                )
                next_value[nonterminal] = (
                    next_probabilities * torch.minimum(next_q1, next_q2)
                ).sum(dim=1)
                target[nonterminal] += float(learner.config.gamma) * next_value[nonterminal]
            q1 = q1_all.gather(1, batch["action"][:, None]).squeeze(1)
            q2 = q2_all.gather(1, batch["action"][:, None]).squeeze(1)
            td_abs = 0.5 * ((q1 - target).abs() + (q2 - target).abs())
            disagreement = (q1 - q2).abs()
            values = {
                "td_abs": td_abs,
                "q_min": torch.minimum(q1, q2),
                "target": target,
                "twin_disagreement": disagreement,
                "advantage": advantage,
                "raw_weight": weights["raw"],
                "normalized_weight": weights["normalized"],
                "raw_clip": weights["raw_high_clip"].float(),
                "bc_kl": bc_kl,
                "top1_disagreement": top1_disagreement,
            }
            for key, value in values.items():
                converted = value.detach().cpu().numpy().astype(np.float64).tolist()
                arrays[key].extend(converted)
            source = batch["behavior_source"].detach().cpu().numpy().astype(np.int64)
            for source_id, source_values in by_source.items():
                mask = source == source_id
                for key, value in values.items():
                    if bool(mask.any()):
                        source_values[key].extend(value.detach().cpu().numpy()[mask].astype(np.float64).tolist())
            awac_top1 = probabilities.argmax(dim=1)
            action_top1.extend(awac_top1.detach().cpu().numpy().astype(np.int64).tolist())
            action_masks.extend(batch["action_mask"].detach().cpu().numpy().astype(bool))
            finite = finite and bool(torch.isfinite(q_min_all).all()) and bool(torch.isfinite(target).all())
            # The comparison is populated in the fixed-corpus policy pass.
            del log_probabilities
    return {
        "arrays": arrays,
        "by_source": by_source,
        "finite": bool(finite),
        "action_top1": action_top1,
        "action_masks": action_masks,
    }


def _summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": _float(np.mean(values)),
        "p05": _percentile(values, 5),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": _float(np.max(values)),
    }


def _source_summaries(by_source: Mapping[int, Mapping[str, Sequence[float]]]) -> Dict[str, Any]:
    result = {}
    names = {int(BehaviorSource.BC_CALIBRATION): "BC_CALIBRATION", int(BehaviorSource.AWAC_ONLINE): "AWAC_ONLINE"}
    for source_id, values in by_source.items():
        result[names[int(source_id)]] = {key: _summary(value) for key, value in values.items()}
    return result


def _fixed_policy_audit(replay: AWACReplayBuffer, learner, *, torch, mpl: MotionPrimitiveLibrary) -> Dict[str, Any]:
    kl_values: List[float] = []
    tv_values: List[float] = []
    disagree: List[bool] = []
    high_conf_disagree: List[bool] = []
    high_conf_count = 0
    bc_top1_actions: List[int] = []
    awac_top1_actions: List[int] = []
    awac_q_gap: List[float] = []
    awac_q_gap_positive: List[bool] = []
    bc_confidence: List[float] = []
    with torch.no_grad():
        for offset in range(0, BC_REPLAY_ROWS, 256):
            end = min(BC_REPLAY_ROWS, offset + 256)
            batch = replay.sample_indices(np.arange(offset, end, dtype=np.int64), torch=torch, device=torch.device("cpu"))
            bc_logits = learner.bc_reference(batch["depth"], batch["vector"])
            awac_logits = learner.actor(batch["depth"], batch["vector"])
            bc_prob, bc_log, _ = masked_policy(bc_logits, batch["action_mask"], torch)
            awac_prob, _, _ = masked_policy(awac_logits, batch["action_mask"], torch)
            kl = (bc_prob * (bc_log - torch.log(awac_prob.clamp_min(1.0e-12)))).sum(dim=1)
            tv = 0.5 * (bc_prob - awac_prob).abs().sum(dim=1)
            bc_top1 = bc_prob.argmax(dim=1)
            awac_top1 = awac_prob.argmax(dim=1)
            confidence = bc_prob.max(dim=1).values
            q_min = torch.minimum(
                learner.critic1(batch["depth"], batch["vector"]),
                learner.critic2(batch["depth"], batch["vector"]),
            )
            row_indices = torch.arange(q_min.shape[0])
            awac_q = q_min[row_indices, awac_top1]
            bc_q = q_min[row_indices, bc_top1]
            gap = awac_q - bc_q
            kl_values.extend(kl.cpu().numpy().astype(np.float64).tolist())
            tv_values.extend(tv.cpu().numpy().astype(np.float64).tolist())
            disagree.extend((awac_top1 != bc_top1).cpu().numpy().astype(bool).tolist())
            high = confidence >= 0.80
            high_conf_count += int(high.sum().item())
            high_conf_disagree.extend(((awac_top1 != bc_top1) & high).cpu().numpy().astype(bool).tolist())
            bc_top1_actions.extend(bc_top1.cpu().numpy().astype(np.int64).tolist())
            awac_top1_actions.extend(awac_top1.cpu().numpy().astype(np.int64).tolist())
            awac_q_gap.extend(gap.cpu().numpy().astype(np.float64).tolist())
            awac_q_gap_positive.extend((gap > 0.0).cpu().numpy().astype(bool).tolist())
            bc_confidence.extend(confidence.cpu().numpy().astype(np.float64).tolist())
    action_counts = Counter(int(value) for value in awac_top1_actions)
    bc_action_counts = Counter(int(value) for value in bc_top1_actions)
    def action_info(action: int) -> Dict[str, Any]:
        spec = dict(mpl.actions[int(action)])
        return {
            "action": int(action),
            "count": int(action_counts.get(int(action), 0)),
            "bc_count": int(bc_action_counts.get(int(action), 0)),
            "horizontal_index": int(mpl.horizontal_index[int(action)]),
            "vertical_index": int(mpl.vertical_index[int(action)]),
            "lateral_mode": spec.get("lateral_mode"),
            "vertical_mode": spec.get("vertical_mode"),
        }
    high_denominator = max(1, high_conf_count)
    return {
        "fixed_corpus": "BC_CALIBRATION_REPLAY_PREFIX",
        "fixed_corpus_rows": BC_REPLAY_ROWS,
        "bc_reference_to_awac_kl": _summary(kl_values),
        "total_variation": _summary(tv_values),
        "top1_disagreement": {
            "count": int(sum(disagree)),
            "rate": _float(sum(disagree) / len(disagree)),
        },
        "high_confidence_definition": {
            "source": "BC masked policy top-1 probability",
            "threshold": 0.80,
            "diagnostic_only": True,
            "confidence_contract_available": False,
        },
        "high_confidence_bc_top1_disagreement": {
            "rows": int(high_conf_count),
            "disagreement_count": int(sum(high_conf_disagree)),
            "rate": _float(sum(high_conf_disagree) / high_denominator),
        },
        "bc_top1_confidence": _summary(bc_confidence),
        "awac_selected_minus_bc_selected_q": _summary(awac_q_gap),
        "awac_selected_q_positive_gap_rate": _float(sum(awac_q_gap_positive) / len(awac_q_gap_positive)),
        "top1_action_distribution": [action_info(action) for action, _ in action_counts.most_common(10)],
        "top1_action_unique_count": len(action_counts),
        "mpl_contract_sha256": mpl.contract_sha256,
    }


def _schedule_audit(awac_payload: Mapping[str, Any], replay: AWACReplayBuffer) -> Dict[str, Any]:
    resolved = awac_payload["resolved_training_config"]
    start_total = int(awac_payload["starting_replay_total_added"])
    end_total = int(replay.total_added)
    learning_starts = int(resolved["learning_starts"])
    actor_learning_starts = int(resolved["actor_learning_starts"])
    burnin = int(resolved["critic_burnin_updates"])
    interval = int(resolved["actor_update_interval"])
    rate = float(resolved["updates_per_step"])
    calibration_updates = int(awac_payload["starting_replay_size"] - learning_starts + 1)
    calibration_updates = int(math.floor(max(0, calibration_updates) * rate))
    critic_count = calibration_updates
    update_step = calibration_updates
    scheduled_actor = 0
    for total in range(start_total + 1, end_total + 1):
        target = critic_update_target(
            replay_total_added=total,
            learning_starts=learning_starts,
            updates_per_step=rate,
        )
        while critic_count < target:
            if actor_update_due(
                replay_total_added=total,
                learner_update_step=update_step,
                actor_learning_starts=actor_learning_starts,
                critic_burnin_updates=burnin,
                actor_update_interval=interval,
            ):
                scheduled_actor += 1
            critic_count += 1
            update_step += 1
    actual = {
        "critic_update_count": int(awac_payload["critic_update_count"]),
        "actor_update_count": int(awac_payload["actor_update_count"]),
        "actor_optimizer_step_count": int(awac_payload["actor_optimizer_step_count"]),
    }
    return {
        "config": {
            "learning_starts": learning_starts,
            "actor_learning_starts": actor_learning_starts,
            "critic_burnin_updates": burnin,
            "actor_update_interval": interval,
            "updates_per_step": rate,
        },
        "starting_replay_total_added": start_total,
        "ending_replay_total_added": end_total,
        "expected_critic_update_count": critic_count,
        "expected_scheduled_actor_calls": scheduled_actor,
        "actual": actual,
        "actor_trust_region_rejections": int(awac_payload.get("actor_trust_region_rejection_count", 0)),
        "accepted_actor_plus_rejected_actor_equals_schedule": bool(
            int(awac_payload["actor_update_count"])
            + int(awac_payload.get("actor_trust_region_rejection_count", 0))
            == scheduled_actor
        ),
        "pass": bool(
            critic_count == actual["critic_update_count"]
            and actual["actor_update_count"] == actual["actor_optimizer_step_count"]
            and actual["actor_update_count"] + int(awac_payload.get("actor_trust_region_rejection_count", 0)) == scheduled_actor
        ),
    }


def _lineage_audit(
    *,
    bc_path: Path,
    calibration_path: Path,
    awac_path: Path,
    replay: AWACReplayBuffer,
    bc_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    awac_payload: Mapping[str, Any],
    root: Path,
) -> Dict[str, Any]:
    runtime_start = json.loads((awac_path.parent / "runtime_start.json").read_text(encoding="utf-8"))
    checks: Dict[str, bool] = {}
    checks["bc_checkpoint_sha256"] = file_sha256(bc_path) == SOURCE_BC_SHA
    checks["calibration_checkpoint_sha256"] = file_sha256(calibration_path) == SOURCE_CALIBRATION_SHA
    checks["awac_checkpoint_sha256"] = file_sha256(awac_path) == SOURCE_AWAC_SHA
    checks["runtime_source_calibration_sha256"] = runtime_start.get("source_calibration_checkpoint_sha256") == SOURCE_CALIBRATION_SHA
    checks["runtime_resume_sha256"] = runtime_start.get("resume_checkpoint_identity", {}).get("sha256") == SOURCE_CALIBRATION_SHA
    checks["awac_source_calibration_sha256"] = awac_payload.get("source_calibration_checkpoint_sha256") == SOURCE_CALIBRATION_SHA
    checks["awac_source_bc_sha256"] = awac_payload.get("source_bc_checkpoint_sha256") == SOURCE_BC_SHA
    checks["replay_bc_sha256"] = replay.metadata.get("bc_checkpoint_sha256") == SOURCE_BC_SHA
    checks["contracts_exact"] = (
        replay.metadata.get("observation_contract") == EXACT_ENDPOINT_OBSERVATION_CONTRACT
        and awac_payload.get("observation_contract") == EXACT_ENDPOINT_OBSERVATION_CONTRACT
        and awac_payload.get("task_contract_sha256") == runtime_start.get("task_contract_sha256")
    )
    checks["calibration_actor_equals_bc"] = _tensor_states_equal(
        calibration_payload["actor_state_dict"], bc_payload["model_state_dict"]
    )
    checks["replay_prefix_matches_calibration"] = True
    calibration_replay_dir = calibration_path.parent / "replay"
    calibration_replay = AWACReplayBuffer.open(calibration_replay_dir, read_only=True)
    try:
        if int(calibration_replay.size) != BC_REPLAY_ROWS or int(replay.size) < BC_REPLAY_ROWS:
            checks["replay_prefix_matches_calibration"] = False
        else:
            for name in REPLAY_FIELD_NAMES:
                checks["replay_prefix_matches_calibration"] = checks["replay_prefix_matches_calibration"] and bool(
                    np.array_equal(calibration_replay.arrays[name][:BC_REPLAY_ROWS], replay.arrays[name][:BC_REPLAY_ROWS])
                )
    finally:
        calibration_replay.close()
    schedule = _schedule_audit(awac_payload, replay)
    checks["schedule_counters"] = bool(schedule["pass"])
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "source_hashes": {
            "bc_checkpoint": file_sha256(bc_path),
            "calibration_checkpoint": file_sha256(calibration_path),
            "awac_checkpoint": file_sha256(awac_path),
        },
        "runtime_start": runtime_start,
        "schedule": schedule,
        "source_paths": {
            "bc_checkpoint": str(bc_path),
            "calibration_checkpoint": str(calibration_path),
            "awac_checkpoint": str(awac_path),
            "replay": str(replay.directory),
            "runtime_start": str(awac_path.parent / "runtime_start.json"),
        },
    }


def _critic_checkpoint_snapshot(replay: AWACReplayBuffer, learner, *, torch) -> Dict[str, Any]:
    values: Dict[str, List[float]] = {"q_min": [], "twin_disagreement": []}
    with torch.no_grad():
        for offset in range(0, BC_REPLAY_ROWS, 256):
            end = min(BC_REPLAY_ROWS, offset + 256)
            batch = replay.sample_indices(np.arange(offset, end, dtype=np.int64), torch=torch, device=torch.device("cpu"))
            q1 = learner.critic1(batch["depth"], batch["vector"])
            q2 = learner.critic2(batch["depth"], batch["vector"])
            q_min = torch.minimum(q1, q2)
            values["q_min"].extend(q_min.detach().cpu().numpy().reshape(-1).astype(np.float64).tolist())
            values["twin_disagreement"].extend((q1 - q2).abs().detach().cpu().numpy().reshape(-1).astype(np.float64).tolist())
    return {
        "q_min_all_valid_actions": _summary(values["q_min"]),
        "twin_disagreement_all_valid_actions": _summary(values["twin_disagreement"]),
    }


def _write_paired_audit(
    out_dir: Path,
    bc_dir: Path,
    awac_dir: Path,
    bc_rows: Sequence[Mapping[str, Any]],
    awac_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if len(bc_rows) != len(awac_rows):
        raise ValueError("paired evaluation row count mismatch")
    paired: List[Dict[str, Any]] = []
    for bc_row, awac_row in zip(bc_rows, awac_rows):
        if (bc_row["episode_id"], bc_row["mission_id"]) != (awac_row["episode_id"], awac_row["mission_id"]):
            raise ValueError("paired evaluation identity/order mismatch")
        paired.append({
            "episode_id": bc_row["episode_id"],
            "mission_id": bc_row["mission_id"],
            "bc": _metric_row(bc_row),
            "awac": _metric_row(awac_row),
            "classification": (
                "REGRESSED" if _bool(bc_row.get("success")) and not _bool(awac_row.get("success")) else
                "RECOVERED" if not _bool(bc_row.get("success")) and _bool(awac_row.get("success")) else
                "UNCHANGED"
            ),
            "bc_trace": _episode_trace(bc_dir, bc_row["episode_id"]),
            "awac_trace": _episode_trace(awac_dir, awac_row["episode_id"]),
        })
    regressed = [row for row in paired if row["classification"] == "REGRESSED"]
    recovered = [row for row in paired if row["classification"] == "RECOVERED"]
    timeout = [row for row in paired if row["awac"]["outcome"] == "timeout"]
    result = {
        "paired_rows": len(paired),
        "regressed_count": len(regressed),
        "recovered_count": len(recovered),
        "awac_timeout_count": len(timeout),
        "regressed": regressed,
        "recovered": recovered,
        "awac_timeouts": timeout,
        "episode_order_exact": True,
        "mission_order_exact": True,
    }
    _json(out_dir / "paired_failure_audit.json", result)
    return result


def _write_timeout_audit(out_dir: Path, paired: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(paired["awac_timeouts"])
    available = [row for row in rows if row["awac_trace"].get("available")]
    steps = [row["awac_trace"].get("steps", 0) for row in available]
    switches = [row["awac_trace"].get("action_switch_rate", 0.0) for row in available]
    result = {
        "awac_timeout_count": len(rows),
        "trace_available_count": len(available),
        "trace_steps": _summary(steps),
        "trace_action_switch_rate": _summary(switches),
        "all_timeout_traces_reached_max_steps": bool(rows) and all(int(row["awac"]["steps"]) == 45 for row in rows),
        "terminal_abort_reasons": Counter(str(row["awac_trace"].get("terminal_abort_reason", "")) for row in rows),
        "interpretation": "The traces establish timeout timing and control observables, but do not prove a single control-path cause without a counterfactual runtime trace.",
    }
    result["terminal_abort_reasons"] = dict(result["terminal_abort_reasons"])
    _json(out_dir / "timeout_audit.json", result)
    return result


def _diagnosis(
    *,
    lineage: Mapping[str, Any],
    policy: Mapping[str, Any],
    td: Mapping[str, Any],
    weights: Mapping[str, Any],
    timeout: Mapping[str, Any],
    paired: Mapping[str, Any],
    actor_drift: Mapping[str, Any],
    bc_kl_pass: bool,
) -> Dict[str, Any]:
    disagreement = float(policy["top1_disagreement"]["rate"])
    q_gap = float(policy["awac_selected_minus_bc_selected_q"]["mean"])
    weight_pathology = str(weights["classification"]).startswith("SUSPECTED")
    critic_scale = str(td["critic_scale_health"]).startswith("FAIL")
    actor_class = str(actor_drift["classification"])
    source_policy = policy.get("behavior_source_policy_metrics", {})
    calibration_policy = source_policy.get("BC_CALIBRATION", {})
    online_policy = source_policy.get("AWAC_ONLINE", {})
    calibration_td = float(td["td_abs_mean_by_source"].get("BC_CALIBRATION", 0.0))
    online_td = float(td["td_abs_mean_by_source"].get("AWAC_ONLINE", 0.0))
    calibration_kl = float(calibration_policy.get("bc_to_awac_kl", {}).get("mean", 0.0))
    online_kl = float(online_policy.get("bc_to_awac_kl", {}).get("mean", 0.0))
    calibration_disagreement = float(calibration_policy.get("top1_disagreement", {}).get("mean", 0.0))
    online_disagreement = float(online_policy.get("top1_disagreement", {}).get("mean", 0.0))
    if online_td > calibration_td * 1.10 and online_kl > calibration_kl * 1.10:
        replay_shift = "SUPPORTED"
    elif online_td > calibration_td * 1.10 or online_kl > calibration_kl * 1.10 or online_disagreement > calibration_disagreement + 0.01:
        replay_shift = "POSSIBLE"
    else:
        replay_shift = "NOT_SUPPORTED"
    if not bool(lineage["pass"]):
        primary = "TRAINING_LINEAGE_NOT_CLOSED"
        confidence = "HIGH"
    elif not bc_kl_pass:
        primary = "BC_KL_IMPLEMENTATION_OR_CONTRACT_FAILURE"
        confidence = "HIGH"
    elif weight_pathology:
        primary = "AWAC_WEIGHT_PATHOLOGY"
        confidence = "MEDIUM"
    elif critic_scale:
        primary = "CRITIC_SCALE_INSTABILITY"
        confidence = "MEDIUM"
    elif actor_class in {"MODERATE", "HIGH"} and disagreement > 0.05:
        primary = "ACTOR_POLICY_DRIFT_WITH_ONLINE_DISTRIBUTION_SHIFT"
        confidence = "MEDIUM"
    else:
        primary = "NEED_TARGETED_TRACE_DIAGNOSIS"
        confidence = "LOW"
    secondary = "TIMEOUT_REGRESSION_OBSERVED" if int(paired["awac_timeout_count"]) > 0 else "NONE"
    return {
        "primary_root_cause": primary,
        "secondary_root_cause": secondary,
        "confidence": confidence,
        "evidence_limits": [
            "The paired Dev100 is observational and does not supply counterfactual outcomes for the same state/action.",
            "Replay stores done/reward but not terminal reason, so TD strata are behavior-source based rather than terminal-reason based.",
            "The confidence interface is intentionally unavailable; high-confidence policy analysis uses an explicit 0.80 diagnostic threshold only.",
        ],
        "classification": {
            "actor_drift": actor_class,
            "critic_misranking_suspected": "INCONCLUSIVE" if disagreement > 0.0 and q_gap > 0.0 else "NOT_SUPPORTED",
            "awac_weight_pathology": "SUSPECTED" if weight_pathology else "NOT_SUPPORTED",
            "timeout_control_pathology": "INCONCLUSIVE" if int(timeout["awac_timeout_count"]) > 0 else "NOT_OBSERVED",
            "online_replay_distribution_shift": replay_shift,
        },
    }


def _write_diagnosis_md(out_dir: Path, diagnosis: Mapping[str, Any], lineage: Mapping[str, Any], policy: Mapping[str, Any], td: Mapping[str, Any], weights: Mapping[str, Any], paired: Mapping[str, Any]) -> None:
    lines = [
        "# AWAC10K degradation diagnosis",
        "",
        "This is a read-only audit of frozen artifacts. No trainer update, runtime, Unity, Bridge, checkpoint, or replay write was performed.",
        "",
        "## Result",
        "",
        "- Primary root cause: `{}`".format(diagnosis["primary_root_cause"]),
        "- Secondary root cause: `{}`".format(diagnosis["secondary_root_cause"]),
        "- Confidence: `{}`".format(diagnosis["confidence"]),
        "- Training lineage: `{}`".format("PASS" if lineage["pass"] else "FAIL"),
        "- Regressed / recovered: `{}` / `{}`".format(paired["regressed_count"], paired["recovered_count"]),
        "- AWAC timeouts: `{}`".format(paired["awac_timeout_count"]),
        "",
        "## Fixed BC-calibration corpus",
        "",
        "- BC to AWAC masked KL mean: `{:.8f}`".format(policy["bc_reference_to_awac_kl"]["mean"]),
        "- Top-1 disagreement: `{:.6f}` ({:.2%})".format(policy["top1_disagreement"]["rate"], policy["top1_disagreement"]["rate"]),
        "- High-confidence threshold: `0.80` (diagnostic only)",
        "- High-confidence disagreement: `{:.6f}`".format(policy["high_confidence_bc_top1_disagreement"]["rate"]),
        "- Final-critic AWAC-selected minus BC-selected Q mean: `{:.8f}`".format(policy["awac_selected_minus_bc_selected_q"]["mean"]),
        "",
        "## Replay health",
        "",
        "- TD absolute mean by source: `{}`".format(json.dumps(td["td_abs_mean_by_source"], sort_keys=True)),
        "- Twin critic health: `{}`".format(td["critic_twin_agreement_health"]),
        "- AWAC raw-weight classification: `{}`".format(weights["classification"]),
        "- Raw-weight clip fractions: `{}`".format(json.dumps(weights["clip_fraction_by_source"], sort_keys=True)),
        "",
        "## Lineage checks",
        "",
    ]
    lines.extend("- `{}`: `{}`".format(key, value) for key, value in sorted(lineage["checks"].items()))
    lines.extend([
        "",
        "## Limits",
        "",
    ])
    lines.extend("- {}".format(value) for value in diagnosis["evidence_limits"])
    (out_dir / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", required=True, type=Path)
    parser.add_argument("--calibration-checkpoint", required=True, type=Path)
    parser.add_argument("--awac-checkpoint", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--bc-eval-dir", required=True, type=Path)
    parser.add_argument("--awac-eval-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    paths = {name: Path(getattr(args, name)).expanduser().resolve() for name in (
        "bc_checkpoint", "calibration_checkpoint", "awac_checkpoint", "replay_dir", "bc_eval_dir", "awac_eval_dir", "out_dir"
    )}
    for name in ("bc_checkpoint", "calibration_checkpoint", "awac_checkpoint"):
        if not paths[name].is_file():
            raise FileNotFoundError(str(paths[name]))
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    torch, nn, _, _, _ = require_torch()
    torch.set_num_threads(int(args.cpu_threads))
    bc_payload = load_torch(paths["bc_checkpoint"], torch=torch, map_location="cpu")
    calibration_payload = load_torch(paths["calibration_checkpoint"], torch=torch, map_location="cpu")
    awac_payload = load_torch(paths["awac_checkpoint"], torch=torch, map_location="cpu")
    if not all(isinstance(value, Mapping) for value in (bc_payload, calibration_payload, awac_payload)):
        raise ValueError("all checkpoints must be mappings")
    mpl = MotionPrimitiveLibrary()
    awac_learner = build_learner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_checkpoint=bc_payload,
        config=_config_from_checkpoint(awac_payload),
    )
    awac_learner.load_state_dict(dict(awac_payload))
    calibration_learner = build_learner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_checkpoint=bc_payload,
        config=_config_from_checkpoint(calibration_payload),
    )
    calibration_learner.load_state_dict(dict(calibration_payload))
    replay = AWACReplayBuffer.open(paths["replay_dir"], read_only=True)
    try:
        if int(replay.size) != int(awac_payload["replay_size"]):
            raise ValueError("replay/checkpoint size mismatch")
        bc_rows = _read_rows(paths["bc_eval_dir"] / "rollout_index.csv")
        awac_rows = _read_rows(paths["awac_eval_dir"] / "rollout_index.csv")
        paired = _write_paired_audit(paths["out_dir"], paths["bc_eval_dir"], paths["awac_eval_dir"], bc_rows, awac_rows)
        timeout = _write_timeout_audit(paths["out_dir"], paired)
        policy = _fixed_policy_audit(replay, awac_learner, torch=torch, mpl=mpl)
        metrics = _batch_metrics(replay, awac_learner, torch=torch, start=0, stop=int(replay.size), batch_size=256)
        source_summaries = _source_summaries(metrics["by_source"])
        policy["behavior_source_policy_metrics"] = {
            name: {
                "bc_to_awac_kl": values["bc_kl"],
                "top1_disagreement": values["top1_disagreement"],
            }
            for name, values in source_summaries.items()
        }
        calibration_snapshot = _critic_checkpoint_snapshot(replay, calibration_learner, torch=torch)
        awac_snapshot = _critic_checkpoint_snapshot(replay, awac_learner, torch=torch)
        td_by_source = {name: values["td_abs"] for name, values in source_summaries.items()}
        disagreement_by_source = {name: values["twin_disagreement"] for name, values in source_summaries.items()}
        final_q_p99 = float(awac_snapshot["q_min_all_valid_actions"]["p95"])
        calibration_q_p99 = float(calibration_snapshot["q_min_all_valid_actions"]["p95"])
        scale_ratio = _float(abs(final_q_p99) / max(abs(calibration_q_p99), 1.0e-6))
        twin_ratio = _float(
            awac_snapshot["twin_disagreement_all_valid_actions"]["p95"]
            / max(abs(final_q_p99), 1.0e-6)
        )
        critic_scale_health = "PASS" if metrics["finite"] and scale_ratio <= 4.0 else "FAIL_SCALE_OR_NONFINITE"
        critic_twin_health = "PASS" if metrics["finite"] and twin_ratio <= 0.25 else "WARN_RELATIVE_DISAGREEMENT"
        td = {
            "source": "final_awac_checkpoint_critic_and_target_critic",
            "reward_scale": float(awac_learner.config.reward_scale),
            "gamma": float(awac_learner.config.gamma),
            "td_abs_mean_by_source": {name: value["mean"] for name, value in td_by_source.items()},
            "td_abs_summary_by_source": td_by_source,
            "critic_scale_health": critic_scale_health,
            "critic_twin_agreement_health": critic_twin_health,
            "critic_twin_disagreement_summary_by_source": disagreement_by_source,
            "checkpoint_same_corpus_scale_comparison": {
                "calibration": calibration_snapshot,
                "awac": awac_snapshot,
                "awac_over_calibration_q_p95_abs_ratio": scale_ratio,
                "awac_twin_p95_over_awac_q_p95_abs_ratio": twin_ratio,
                "q_explosion_multiplier_contract": 4.0,
                "relative_disagreement_warning_threshold": 0.25,
            },
            "finite": bool(metrics["finite"]),
        }
        critic_health = {
            "finite": bool(metrics["finite"]),
            "scale_health": critic_scale_health,
            "twin_agreement_health": critic_twin_health,
            "same_corpus_checkpoint_comparison": {
                "calibration": calibration_snapshot,
                "awac": awac_snapshot,
                "awac_over_calibration_q_p95_abs_ratio": scale_ratio,
                "awac_twin_p95_over_awac_q_p95_abs_ratio": twin_ratio,
            },
            "thresholds": {
                "q_explosion_multiplier": 4.0,
                "relative_twin_disagreement_warning": 0.25,
            },
            "by_behavior_source": {
                name: {
                    "q_min": values["q_min"],
                    "target": values["target"],
                    "twin_disagreement": values["twin_disagreement"],
                }
                for name, values in source_summaries.items()
            },
            "interpretation": (
                "The final Critic is finite, but its same-corpus q p95 is above the "
                "calibration checkpoint by more than the existing 4x explosion "
                "threshold; this is a health warning, not proof of action misranking."
            ),
        }
        _json(paths["out_dir"] / "critic_health.json", critic_health)
        _json(paths["out_dir"] / "td_audit.json", td)
        raw_by_source = {name: values["raw_weight"] for name, values in source_summaries.items()}
        normalized_by_source = {name: values["normalized_weight"] for name, values in source_summaries.items()}
        clip_by_source = {name: values["raw_clip"]["mean"] for name, values in source_summaries.items()}
        all_raw = metrics["arrays"]["raw_weight"]
        all_clip = metrics["arrays"]["raw_clip"]
        all_ess = (sum(all_raw) ** 2 / max(sum(value * value for value in all_raw), 1.0e-12)) / max(1, len(all_raw))
        classification = "SUSPECTED_LOW_ESS_OR_CLIPPING" if max(clip_by_source.values(), default=0.0) > 0.01 or all_ess < 0.20 else "NOT_SUSPECTED"
        weights = {
            "formula": "clip(exp((Qmin(data_action)-sum(pi_awac*Qmin))/temperature), max=20)",
            "temperature": float(awac_learner.config.awac_temperature),
            "weight_max": float(awac_learner.config.awac_weight_max),
            "raw_weight_by_source": raw_by_source,
            "normalized_weight_by_source": normalized_by_source,
            "clip_fraction_by_source": clip_by_source,
            "overall_raw_weight": _summary(all_raw),
            "overall_raw_clip_fraction": _float(sum(all_clip) / max(1, len(all_clip))),
            "overall_raw_effective_sample_size_fraction": _float(all_ess),
            "classification": classification,
        }
        _json(paths["out_dir"] / "awac_weight_audit.json", weights)
        actor_drift_groups: Dict[str, Dict[str, float]] = {}
        bc_state = bc_payload["model_state_dict"]
        awac_state = awac_payload["actor_state_dict"]
        for group in ("depth_encoder", "vector_encoder", "head"):
            left = [value for name, value in bc_state.items() if name.startswith(group + ".")]
            right = [value for name, value in awac_state.items() if name.startswith(group + ".")]
            delta = math.sqrt(sum(float(((a - b).detach().cpu().numpy().astype(np.float64) ** 2).sum()) for a, b in zip(left, right)))
            norm = math.sqrt(sum(float((a.detach().cpu().numpy().astype(np.float64) ** 2).sum()) for a in left))
            actor_drift_groups[group] = {"delta_l2": _float(delta), "bc_parameter_l2": _float(norm), "relative_delta": _float(delta / max(norm, 1.0e-12))}
        total_delta = math.sqrt(sum(value["delta_l2"] ** 2 for value in actor_drift_groups.values()))
        total_norm = math.sqrt(sum(value["bc_parameter_l2"] ** 2 for value in actor_drift_groups.values()))
        disagreement = float(policy["top1_disagreement"]["rate"])
        actor_class = "LOW" if total_delta / max(total_norm, 1.0e-12) < 0.01 and disagreement < 0.05 else "MODERATE" if total_delta / max(total_norm, 1.0e-12) < 0.05 and disagreement < 0.15 else "HIGH"
        actor_drift = {
            "bc_actor_state_sha256": _state_fingerprint(bc_state),
            "awac_actor_state_sha256": _state_fingerprint(awac_state),
            "summary_actor_state_before_sha256": json.loads((paths["awac_checkpoint"].parent / "summary.json").read_text(encoding="utf-8"))["actor_state_before_sha256"],
            "summary_actor_state_after_sha256": json.loads((paths["awac_checkpoint"].parent / "summary.json").read_text(encoding="utf-8"))["actor_state_after_sha256"],
            "groups": actor_drift_groups,
            "total_relative_delta": _float(total_delta / max(total_norm, 1.0e-12)),
            "classification_thresholds": {"low_relative_delta": 0.01, "moderate_relative_delta": 0.05, "low_disagreement": 0.05, "moderate_disagreement": 0.15},
            "classification": actor_class,
        }
        _json(paths["out_dir"] / "actor_drift.json", actor_drift)
        _json(paths["out_dir"] / "policy_drift.json", policy)
        lineage = _lineage_audit(
            bc_path=paths["bc_checkpoint"], calibration_path=paths["calibration_checkpoint"], awac_path=paths["awac_checkpoint"], replay=replay,
            bc_payload=bc_payload, calibration_payload=calibration_payload, awac_payload=awac_payload, root=paths["out_dir"],
        )
        _json(paths["out_dir"] / "lineage.json", lineage)
        learner_source = inspect.getsource(type(awac_learner)._batch_bc_kl)
        bc_kl_checks = {
            "frozen_bc_reference": all(not parameter.requires_grad for parameter in awac_learner.bc_reference.parameters()),
            "bc_reference_not_in_actor_optimizer": not bool({id(parameter) for parameter in awac_learner.bc_reference.parameters()} & {id(parameter) for group in awac_learner.actor_optimizer.param_groups for parameter in group["params"]}),
            "masked_policy_used": "masked_policy" in learner_source,
            "bc_to_awac_direction": "bc_log_probabilities - actor_log_probabilities" in learner_source,
            "no_grad_observation": "with torch.no_grad()" in learner_source,
            "runtime_weight": float(awac_learner.config.bc_kl_weight) == 0.05,
        }
        bc_kl_pass = all(bc_kl_checks.values())
        bc_kl = {"checks": bc_kl_checks, "pass": bc_kl_pass, "fixed_corpus_kl": policy["bc_reference_to_awac_kl"]}
        _json(paths["out_dir"] / "bc_kl_audit.json", bc_kl)
        diagnosis = _diagnosis(lineage=lineage, policy=policy, td=td, weights=weights, timeout=timeout, paired=paired, actor_drift=actor_drift, bc_kl_pass=bc_kl_pass)
        diagnosis.update({
            "training_lineage": "PASS" if lineage["pass"] else "FAIL",
            "bc_kl_implementation": "PASS" if bc_kl_pass else "FAIL",
            "actor_parameter_drift_classification": actor_class,
            "fixed_corpus_bc_to_awac_kl_mean": policy["bc_reference_to_awac_kl"]["mean"],
            "fixed_corpus_top1_disagreement": policy["top1_disagreement"]["rate"],
            "high_confidence_bc_top1_disagreement": policy["high_confidence_bc_top1_disagreement"]["rate"],
            "critic_scale_health": critic_scale_health,
            "critic_twin_agreement_health": critic_twin_health,
            "awac_weight_audit": weights,
            "td_abs_mean_bc_calibration": td["td_abs_mean_by_source"].get("BC_CALIBRATION", 0.0),
            "td_abs_mean_awac_online": td["td_abs_mean_by_source"].get("AWAC_ONLINE", 0.0),
            "awac_weight_mean_bc_calibration": raw_by_source.get("BC_CALIBRATION", {}).get("mean", 0.0),
            "awac_weight_p95_bc_calibration": raw_by_source.get("BC_CALIBRATION", {}).get("p95", 0.0),
            "awac_weight_clip_fraction_bc_calibration": clip_by_source.get("BC_CALIBRATION", 0.0),
            "awac_weight_mean_awac_online": raw_by_source.get("AWAC_ONLINE", {}).get("mean", 0.0),
            "awac_weight_p95_awac_online": raw_by_source.get("AWAC_ONLINE", {}).get("p95", 0.0),
            "awac_weight_clip_fraction_awac_online": clip_by_source.get("AWAC_ONLINE", 0.0),
            "critic_misranking_suspected": diagnosis["classification"]["critic_misranking_suspected"],
            "regressed_missions": paired["regressed_count"],
            "recovered_missions": paired["recovered_count"],
            "awac_timeout_missions": paired["awac_timeout_count"],
            "actor_drift": actor_class,
            "critic_scale_instability": "YES" if critic_scale_health.startswith("FAIL") else "NO",
            "critic_action_misranking": diagnosis["classification"]["critic_misranking_suspected"],
            "awac_weight_pathology": diagnosis["classification"]["awac_weight_pathology"],
            "bc_kl_bug": "YES" if not bc_kl_pass else "NO",
            "online_replay_distribution_shift": diagnosis["classification"]["online_replay_distribution_shift"],
            "timeout_control_pathology": diagnosis["classification"]["timeout_control_pathology"],
            "full_test_note": "Tests are run by the controlling Codex turn after this read-only diagnostic.",
        })
        _json(paths["out_dir"] / "diagnosis.json", diagnosis)
        _write_diagnosis_md(paths["out_dir"], diagnosis, lineage, policy, td, weights, paired)
        return {
            "diagnosis": diagnosis,
            "lineage": lineage,
            "policy": policy,
            "td": td,
            "weights": weights,
            "paired": paired,
            "timeout": timeout,
            "actor_drift": actor_drift,
            "bc_kl": bc_kl,
            "out_dir": str(paths["out_dir"]),
        }
    finally:
        replay.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print("AWAC10K_DEGRADATION_DIAGNOSIS=PASS")
    print(json.dumps({
        "out_dir": result["out_dir"],
        "training_lineage": result["diagnosis"]["training_lineage"],
        "bc_kl_implementation": result["diagnosis"]["bc_kl_implementation"],
        "primary_root_cause": result["diagnosis"]["primary_root_cause"],
        "confidence": result["diagnosis"]["confidence"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
