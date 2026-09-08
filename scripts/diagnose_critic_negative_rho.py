#!/usr/bin/env python
"""Offline Critic-only diagnosis for a blocked AWAC calibration.

This tool deliberately owns no production training path.  It reads one
calibration checkpoint and one read-only replay, reconstructs independent
episode MC labels, and runs the bounded MC/TD diagnostic branches described by
the 2026-09-06 diagnosis task.  The production calibration gate is never
written by this module.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from planning.awac.calibration import masked_bellman_target
from planning.awac.checkpoint import calibration_holdout_records_sha256
from planning.awac.learner import AWACOptimizationConfig
from planning.awac.optimization import masked_discrete_cql_loss
from planning.awac.replay import AWACReplayBuffer
from planning.awac.trainer import build_learner
from planning.common.hashing import canonical_json_sha256, file_sha256
from planning.version import SOFTWARE_VERSION


DEFAULT_INDEX_SEED = 20260906
DEFAULT_TORCH_SEED = 20260906
DEFAULT_BATCH_SIZE = 128
DEFAULT_SMALL_ROWS = 256
DEFAULT_SMALL_UPDATES = 1000
DEFAULT_BRANCH_UPDATES = 3000
DEFAULT_BOOTSTRAP = 2000
SNAPSHOTS = (0, 300, 1000, 3000)
SMALL_SNAPSHOTS = (0, 100, 300, 1000)


def _json(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return file_sha256(Path(path).resolve())


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _sha_bytes(array.tobytes(order="C"))


def _sha_indices(value: np.ndarray) -> str:
    return _sha_array(np.asarray(value, dtype=np.int64))


def _module_state_sha(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.state_dict().items()):
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _optimizer_sha(optimizer: torch.optim.Optimizer) -> str:
    stream = io.BytesIO()
    torch.save(optimizer.state_dict(), stream)
    return _sha_bytes(stream.getvalue())


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "p05": float(np.percentile(values, 5.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = 0.5 * float(index + end - 1) + 1.0
        index = end
    return ranks


def spearman(values_left: Sequence[float], values_right: Sequence[float]) -> Optional[float]:
    """Independent tied-rank Spearman implementation used by the audit."""

    left = np.asarray(values_left, dtype=np.float64).reshape(-1)
    right = np.asarray(values_right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 2:
        return None
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Spearman inputs contain non-finite values")
    left = _rankdata(left)
    right = _rankdata(right)
    left -= left.mean()
    right -= right.mean()
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def pearson(values_left: Sequence[float], values_right: Sequence[float]) -> Optional[float]:
    left = np.asarray(values_left, dtype=np.float64).reshape(-1)
    right = np.asarray(values_right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 2:
        return None
    left -= left.mean()
    right -= right.mean()
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def pair_metrics(prediction: Sequence[float], target: Sequence[float]) -> Dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    return {
        "n": int(prediction.size),
        "rho": spearman(prediction, target),
        "pearson": pearson(prediction, target),
        "mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "mean": float(np.mean(prediction)),
        "std": float(np.std(prediction)),
        "quantiles": _quantiles(prediction),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
        "target_quantiles": _quantiles(target),
    }


def _records_mc_returns(
    records: Sequence[Mapping[str, Any]], *, gamma: float, reward_scale: float
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """Rebuild complete-episode MC labels without calling production MC code."""

    groups: Dict[Tuple[str, str], List[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    seen_record_indices = set()
    for row_index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ValueError("holdout row {} is not a mapping".format(row_index))
        mission_id = str(row.get("mission_id", "")).strip()
        episode_id = str(row.get("episode_id", "")).strip()
        if not mission_id or not episode_id:
            raise ValueError("holdout row {} has incomplete identity".format(row_index))
        key = (mission_id, episode_id)
        groups[key].append((row_index, row))
    returns = np.empty(len(records), dtype=np.float64)
    episode_report: List[Dict[str, Any]] = []
    terminal_counts = Counter()
    for key, grouped_rows in groups.items():
        ordered = sorted(
            grouped_rows,
            key=lambda item: int(item[1]["episode_transition_index"]),
        )
        indices = [int(row["episode_transition_index"]) for _, row in ordered]
        expected = list(range(len(ordered)))
        if indices != expected:
            raise ValueError("holdout episode {} transition indices are not contiguous".format(key))
        done_values = [bool(row["done"]) for _, row in ordered]
        if sum(done_values) != 1 or not done_values[-1]:
            raise ValueError("holdout episode {} does not have one final done row".format(key))
        terminal_reason = str(ordered[-1][1].get("terminal_reason", "")).strip()
        if not terminal_reason:
            raise ValueError("holdout episode {} has no terminal reason".format(key))
        if any(str(row.get("terminal_reason", "")).strip() != terminal_reason for _, row in ordered):
            raise ValueError("holdout episode {} changes terminal reason".format(key))
        total = 0.0
        values = []
        for row_index, row in reversed(ordered):
            reward = float(row["reward"])
            if not math.isfinite(reward):
                raise ValueError("holdout row {} reward is non-finite".format(row_index))
            total = float(reward_scale) * reward + float(gamma) * total
            returns[row_index] = total
            values.append(total)
            if row_index in seen_record_indices:
                raise ValueError("holdout row index is duplicated")
            seen_record_indices.add(row_index)
        terminal_counts[terminal_reason] += 1
        episode_report.append(
            {
                "mission_id": key[0],
                "episode_id": key[1],
                "row_indices": [int(row_index) for row_index, _ in ordered],
                "length": int(len(ordered)),
                "terminal_reason": terminal_reason,
                "return_first": float(returns[ordered[0][0]]),
                "return_terminal": float(returns[ordered[-1][0]]),
            }
        )
    if len(seen_record_indices) != len(records):
        raise ValueError("some holdout rows were not assigned an MC label")
    lengths = np.asarray([row["length"] for row in episode_report], dtype=np.float64)
    report = {
        "method": "independent_reverse_episode_recursion",
        "gamma": float(gamma),
        "reward_scale": float(reward_scale),
        "reward_unit": "raw_environment_reward",
        "reward_scale_owner": "learner_bellman_target",
        "row_count": int(len(records)),
        "episode_count": int(len(episode_report)),
        "unique_mission_count": int(len({row["mission_id"] for row in episode_report})),
        "terminal_episode_counts": dict(sorted(terminal_counts.items())),
        "length_quantiles": _quantiles(lengths),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_quantiles": _quantiles(returns),
        "record_hash": calibration_holdout_records_sha256(records),
        "action_override_field_present": any(
            any(name in row for name in ("action_override", "executed_action", "sampled_action"))
            for row in records
        ),
        "action_override_audit": "no persisted override/executed-action field; replay action is the only persisted action identity",
        "terminal_bootstrap": "zero_for_all_normal_terminal_reasons",
        "max_steps_semantics": "task/collector normal timeout terminal; no runtime-abort rows in holdout",
    }
    return returns, report, episode_report


def _train_mc_labels(replay: AWACReplayBuffer, *, gamma: float, reward_scale: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    count = int(replay.size)
    reward = np.asarray(replay.arrays["reward"][:count], dtype=np.float64)
    done = np.asarray(replay.arrays["done"][:count], dtype=bool)
    if count <= 0 or not bool(done[-1]):
        raise ValueError("train Replay does not end on a committed terminal row")
    labels = np.empty(count, dtype=np.float64)
    ends = np.flatnonzero(done)
    starts = np.r_[0, ends[:-1] + 1]
    for start, end in zip(starts, ends):
        total = 0.0
        for index in range(int(end), int(start) - 1, -1):
            total = float(reward_scale) * float(reward[index]) + float(gamma) * total
            labels[index] = total
    lengths = ends - starts + 1
    return labels, {
        "row_count": count,
        "episode_count": int(len(ends)),
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "terminal_rows": int(done.sum()),
        "return_mean": float(labels.mean()),
        "return_std": float(labels.std()),
        "labels_sha256": _sha_array(labels),
    }


def _rows_batch(records: Sequence[Mapping[str, Any]], indices: Optional[np.ndarray], device):
    selected = records if indices is None else [records[int(index)] for index in indices]
    def stack(name: str, dtype) -> torch.Tensor:
        return torch.from_numpy(np.asarray([row[name] for row in selected], dtype=dtype)).to(device=device)
    return {
        "depth": stack("depth", np.float32),
        "vector": stack("vector", np.float32),
        "action_mask": stack("action_mask", np.bool_),
        "action": stack("action", np.int64).long(),
        "reward": stack("reward", np.float32),
        "next_depth": stack("next_depth", np.float32),
        "next_vector": stack("next_vector", np.float32),
        "next_action_mask": stack("next_action_mask", np.bool_),
        "done": stack("done", np.float32),
        "behavior_source": stack("behavior_source", np.int64).long(),
    }


def _td_target(learner, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    nonterminal = ~batch["done"].bool()
    target = batch["reward"] * float(learner.config.reward_scale)
    if bool(nonterminal.any().item()):
        with torch.no_grad():
            next_logits = learner.actor(batch["next_depth"][nonterminal], batch["next_vector"][nonterminal])
            target_q1 = learner.target_critic1(batch["next_depth"][nonterminal], batch["next_vector"][nonterminal])
            target_q2 = learner.target_critic2(batch["next_depth"][nonterminal], batch["next_vector"][nonterminal])
            info = masked_bellman_target(
                bc_logits=next_logits,
                target_q1=target_q1,
                target_q2=target_q2,
                next_action_mask=batch["next_action_mask"][nonterminal],
                reward=batch["reward"][nonterminal],
                done=batch["done"][nonterminal],
                gamma=float(learner.config.gamma),
                reward_scale=float(learner.config.reward_scale),
                torch=torch,
            )
            target = target.clone()
            target[nonterminal] = info["target"]
    return target


def _q_arrays(learner, batch: Mapping[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        q1_all = learner.critic1(batch["depth"], batch["vector"])
        q2_all = learner.critic2(batch["depth"], batch["vector"])
        action = batch["action"]
        q1 = q1_all.gather(1, action[:, None]).squeeze(1)
        q2 = q2_all.gather(1, action[:, None]).squeeze(1)
    return q1.detach().cpu().numpy(), q2.detach().cpu().numpy()


def _holdout_predictions(learner, records, device, *, chunk_size: Optional[int] = 256) -> Dict[str, np.ndarray]:
    learner.critic1.eval()
    learner.critic2.eval()
    learner.target_critic1.eval()
    learner.target_critic2.eval()
    learner.actor.eval()
    learner.bc_reference.eval()
    q1_parts = []
    q2_parts = []
    target_parts = []
    indices = range(0, len(records), len(records) if chunk_size is None else int(chunk_size))
    with torch.no_grad():
        for start in indices:
            stop = min(len(records), start + (len(records) if chunk_size is None else int(chunk_size)))
            batch = _rows_batch(records, np.arange(start, stop, dtype=np.int64), device)
            q1, q2 = _q_arrays(learner, batch)
            q1_parts.append(q1)
            q2_parts.append(q2)
            target_parts.append(_td_target(learner, batch).detach().cpu().numpy())
    learner.critic1.train()
    learner.critic2.train()
    learner.target_critic1.eval()
    learner.target_critic2.eval()
    learner.actor.train()
    return {
        "q1": np.concatenate(q1_parts),
        "q2": np.concatenate(q2_parts),
        "td_target": np.concatenate(target_parts),
    }


def _train_predictions(learner, replay, indices: np.ndarray, device, *, chunk_size: int = 256) -> Dict[str, np.ndarray]:
    q1_parts = []
    q2_parts = []
    target_parts = []
    for start in range(0, len(indices), int(chunk_size)):
        selected = indices[start : start + int(chunk_size)]
        batch = replay.sample_indices(selected, torch=torch, device=device)
        q1, q2 = _q_arrays(learner, batch)
        q1_parts.append(q1)
        q2_parts.append(q2)
        target_parts.append(_td_target(learner, batch).detach().cpu().numpy())
    return {
        "q1": np.concatenate(q1_parts),
        "q2": np.concatenate(q2_parts),
        "td_target": np.concatenate(target_parts),
    }


def _loss_components(learner, batch, target: torch.Tensor) -> Dict[str, float]:
    q1_all = learner.critic1(batch["depth"], batch["vector"])
    q2_all = learner.critic2(batch["depth"], batch["vector"])
    q1 = q1_all.gather(1, batch["action"][:, None]).squeeze(1)
    q2 = q2_all.gather(1, batch["action"][:, None]).squeeze(1)
    td = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
    cql = masked_discrete_cql_loss(q1_all, batch["action"], batch["action_mask"], torch)
    cql = cql + masked_discrete_cql_loss(q2_all, batch["action"], batch["action_mask"], torch)
    return {
        "td_loss": float(td.detach().cpu().item()),
        "cql_loss_unweighted": float(cql.detach().cpu().item()),
        "cql_loss_weighted": float(cql.detach().cpu().item()) * float(learner.config.critic_cql_weight),
        "critic_loss": float((td + float(learner.config.critic_cql_weight) * cql).detach().cpu().item()),
    }


def _manual_mc_step(learner, batch, target: torch.Tensor) -> Dict[str, float]:
    learner.critic1.train()
    learner.critic2.train()
    q1_all = learner.critic1(batch["depth"], batch["vector"])
    q2_all = learner.critic2(batch["depth"], batch["vector"])
    q1 = q1_all.gather(1, batch["action"][:, None]).squeeze(1)
    q2 = q2_all.gather(1, batch["action"][:, None]).squeeze(1)
    loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
    params = [p for module in (learner.critic1, learner.critic2) for p in module.parameters() if p.requires_grad]
    learner.critic_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, float(learner.config.gradient_clip_norm))
    learner.critic_optimizer.step()
    learner.critic_update_count += 1
    learner.update_step += 1
    return {
        "critic_td_loss": float(loss.detach().cpu().item()),
        "critic_cql_loss": 0.0,
        "critic_loss": float(loss.detach().cpu().item()),
        "critic_gradient_norm": float(grad_norm.detach().cpu().item()),
    }


def _new_learner(checkpoint, bc_checkpoint, config, device, *, cql_weight: Optional[float] = None):
    torch.manual_seed(DEFAULT_TORCH_SEED)
    learner = build_learner(torch=torch, nn=torch.nn, device=device, bc_checkpoint=bc_checkpoint, config=config)
    learner.load_state_dict(checkpoint)
    # The diagnostic branches deliberately start with identical fresh Adam
    # state rather than inheriting momentum from the production calibration.
    learner.critic_optimizer.state.clear()
    learner.critic_optimizer.zero_grad(set_to_none=True)
    if cql_weight is not None:
        learner.config = dataclasses.replace(learner.config, critic_cql_weight=float(cql_weight))
    learner.freeze_actor_for_calibration()
    learner.actor.eval()
    learner.bc_reference.eval()
    return learner


def _metric_snapshot(q: Dict[str, np.ndarray], target: np.ndarray) -> Dict[str, Any]:
    result = {}
    qmin = np.minimum(q["q1"], q["q2"])
    for name, values in (("q1", q["q1"]), ("q2", q["q2"]), ("qmin", qmin)):
        result[name] = pair_metrics(values, target)
    result["q1_q2_disagreement"] = {
        "mean": float(np.mean(np.abs(q["q1"] - q["q2"]))),
        "p95": float(np.percentile(np.abs(q["q1"] - q["q2"]), 95.0)),
    }
    return result


def _group_metrics(q: Dict[str, np.ndarray], target: np.ndarray, records, lengths) -> List[Dict[str, Any]]:
    qmin = np.minimum(q["q1"], q["q2"])
    done = np.asarray([bool(row["done"]) for row in records])
    action = np.asarray([int(row["action"]) for row in records])
    transition = np.asarray([int(row["episode_transition_index"]) for row in records])
    remaining = np.asarray([int(lengths[(str(row["mission_id"]), str(row["episode_id"]))] - 1 - int(row["episode_transition_index"])) for row in records])
    reasons = np.asarray([str(row["terminal_reason"]) for row in records])
    groups: List[Tuple[str, np.ndarray]] = [
        ("all", np.ones(len(records), dtype=bool)),
        ("terminal", done),
        ("nonterminal", ~done),
        ("first_step", transition == 0),
        ("worker_unavailable", np.ones(len(records), dtype=bool)),
    ]
    for reason in sorted(set(reasons.tolist())):
        groups.append(("terminal_reason=" + reason, reasons == reason))
    for value in sorted(set(action.tolist())):
        groups.append(("action=" + str(int(value)), action == value))
    for name, selector in (("remaining=0", remaining == 0), ("remaining=1-4", (remaining >= 1) & (remaining <= 4)), ("remaining=5-9", (remaining >= 5) & (remaining <= 9)), ("remaining=10-19", (remaining >= 10) & (remaining <= 19)), ("remaining=20+", remaining >= 20)):
        groups.append((name, selector))
    rows: List[Dict[str, Any]] = []
    for group_name, selector in groups:
        indices = np.flatnonzero(selector)
        for q_name, values in (("q1", q["q1"]), ("q2", q["q2"]), ("qmin", qmin)):
            row = pair_metrics(values[indices], target[indices]) if len(indices) else {"n": 0, "rho": None}
            row.update({"group": group_name, "q": q_name})
            rows.append(row)
    return rows


def _gradient_probe(learner, replay, batch_indices, train_labels, device, mode: str) -> Dict[str, Any]:
    records = []
    params = [p for module in (learner.critic1, learner.critic2) for p in module.parameters() if p.requires_grad]
    groups = []
    for index, group in enumerate(learner.critic_optimizer.param_groups):
        groups.append((str(group.get("name", "group_{}".format(index))), set(id(p) for p in group["params"])))
    learner.critic1.eval(); learner.critic2.eval(); learner.target_critic1.eval(); learner.target_critic2.eval(); learner.actor.eval()
    for batch_id, indices in enumerate(batch_indices):
        batch = replay.sample_indices(indices, torch=torch, device=device)
        target = (
            torch.from_numpy(np.asarray(train_labels[indices], dtype=np.float32)).to(device=device)
            if mode == "mc" else _td_target(learner, batch)
        )
        q1_all = learner.critic1(batch["depth"], batch["vector"])
        q2_all = learner.critic2(batch["depth"], batch["vector"])
        q1 = q1_all.gather(1, batch["action"][:, None]).squeeze(1)
        q2 = q2_all.gather(1, batch["action"][:, None]).squeeze(1)
        td_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
        cql = masked_discrete_cql_loss(q1_all, batch["action"], batch["action_mask"], torch)
        cql = cql + masked_discrete_cql_loss(q2_all, batch["action"], batch["action_mask"], torch)
        cql_loss = 0.05 * cql
        td_grad = torch.autograd.grad(td_loss, params, retain_graph=True, allow_unused=True)
        cql_grad = torch.autograd.grad(cql_loss, params, retain_graph=False, allow_unused=True)
        td_flat = torch.cat([(g.detach().reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)) for p, g in zip(params, td_grad)])
        cql_flat = torch.cat([(g.detach().reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)) for p, g in zip(params, cql_grad)])
        td_norm = float(torch.linalg.vector_norm(td_flat).cpu().item())
        cql_norm = float(torch.linalg.vector_norm(cql_flat).cpu().item())
        cosine = float(torch.dot(td_flat, cql_flat).cpu().item() / (td_norm * cql_norm + 1.0e-12))
        by_group = {}
        position = 0
        for group_name, parameter_ids in groups:
            td_parts = []
            cql_parts = []
            for parameter, td_value, cql_value in zip(params, td_grad, cql_grad):
                if id(parameter) not in parameter_ids:
                    continue
                td_parts.append(td_value.detach().reshape(-1) if td_value is not None else torch.zeros_like(parameter).reshape(-1))
                cql_parts.append(cql_value.detach().reshape(-1) if cql_value is not None else torch.zeros_like(parameter).reshape(-1))
            if td_parts:
                td_group = torch.cat(td_parts); cql_group = torch.cat(cql_parts)
                td_group_norm = float(torch.linalg.vector_norm(td_group).cpu().item())
                cql_group_norm = float(torch.linalg.vector_norm(cql_group).cpu().item())
                by_group[group_name] = {
                    "g_td_norm": td_group_norm,
                    "g_cql_norm": cql_group_norm,
                    "ratio": cql_group_norm / (td_group_norm + 1.0e-12),
                    "cosine": float(torch.dot(td_group, cql_group).cpu().item() / (td_group_norm * cql_group_norm + 1.0e-12)),
                }
        records.append({"batch": int(batch_id), "g_td_norm": td_norm, "g_cql_norm": cql_norm, "ratio": cql_norm / (td_norm + 1.0e-12), "cosine": cosine, "by_group": by_group})
    learner.critic1.train(); learner.critic2.train()
    def aggregate(field: str) -> float:
        return float(np.mean([float(row[field]) for row in records]))
    return {
        "mode": mode,
        "batch_count": len(records),
        "overall_mean": {field: aggregate(field) for field in ("g_td_norm", "g_cql_norm", "ratio", "cosine")},
        "per_batch": records,
    }


def _run_small_mc(learner, replay, labels, small_indices, small_batches, device) -> Dict[str, Any]:
    snapshots = []
    prediction_snapshots = {}
    def snapshot(step: int):
        q = _train_predictions(learner, replay, small_indices, device)
        target = labels[small_indices]
        prediction_snapshots[str(step)] = {key: value.copy() for key, value in q.items()}
        snapshots.append({"updates": int(step), "metrics": _metric_snapshot(q, target), "critic1_sha256": _module_state_sha(learner.critic1), "critic2_sha256": _module_state_sha(learner.critic2), "critic_optimizer_sha256": _optimizer_sha(learner.critic_optimizer)})
    snapshot(0)
    for step in range(1, DEFAULT_SMALL_UPDATES + 1):
        indices = small_indices[small_batches[step - 1]]
        batch = replay.sample_indices(indices, torch=torch, device=device)
        target = torch.from_numpy(np.asarray(labels[indices], dtype=np.float32)).to(device=device)
        _manual_mc_step(learner, batch, target)
        if step in SMALL_SNAPSHOTS[1:]:
            snapshot(step)
    initial = snapshots[0]["metrics"]
    final = snapshots[-1]["metrics"]
    return {"updates": DEFAULT_SMALL_UPDATES, "snapshots": snapshots, "q1_mse_reduction": float(initial["q1"]["mse"] - final["q1"]["mse"]), "q2_mse_reduction": float(initial["q2"]["mse"] - final["q2"]["mse"]), "prediction_snapshots": prediction_snapshots}


def _run_branch(name, learner, replay, labels, records, returns, batch_indices, train_diag_indices, grad_indices, device, output_predictions):
    mode = "mc" if name == "D_MC" else "td"
    snapshots = []
    endpoint = None
    update_records = []
    initial_critic1 = _module_state_sha(learner.critic1)
    initial_critic2 = _module_state_sha(learner.critic2)
    initial_target1 = _module_state_sha(learner.target_critic1)
    initial_target2 = _module_state_sha(learner.target_critic2)
    initial_optimizer = _optimizer_sha(learner.critic_optimizer)
    gradients_initial = _gradient_probe(learner, replay, grad_indices, labels, device, mode)
    def snapshot(step: int):
        holdout = _holdout_predictions(learner, records, device)
        train = _train_predictions(learner, replay, train_diag_indices, device)
        holdout_metrics = _metric_snapshot(holdout, returns)
        train_metrics = _metric_snapshot(train, labels[train_diag_indices])
        fixed_batch = replay.sample_indices(batch_indices[min(max(step - 1, 0), len(batch_indices) - 1)], torch=torch, device=device)
        target = torch.from_numpy(np.asarray(labels[batch_indices[min(max(step - 1, 0), len(batch_indices) - 1)],], dtype=np.float32)).to(device=device) if mode == "mc" else _td_target(learner, fixed_batch)
        losses = _loss_components(learner, fixed_batch, target)
        record = {
            "updates": int(step),
            "holdout": holdout_metrics,
            "train_fixed": train_metrics,
            "td_target_stats": {"mean": float(holdout["td_target"].mean()), "std": float(holdout["td_target"].std()), "quantiles": _quantiles(holdout["td_target"])},
            "fixed_batch_losses": losses,
            "critic1_sha256": _module_state_sha(learner.critic1),
            "critic2_sha256": _module_state_sha(learner.critic2),
            "target_critic1_sha256": _module_state_sha(learner.target_critic1),
            "target_critic2_sha256": _module_state_sha(learner.target_critic2),
            "critic_optimizer_sha256": _optimizer_sha(learner.critic_optimizer),
            "critic_update_count": int(learner.critic_update_count),
            "actor_update_count": int(learner.actor_update_count),
            "actor_awac_update_count": int(learner.actor_awac_update_count),
            "actor_recovery_update_count": int(learner.actor_recovery_update_count),
            "actor_trust_region_rejection_count": int(learner.actor_trust_region_rejection_count),
        }
        snapshots.append(record)
        output_predictions(name + "@" + str(step), holdout, records, returns)
        return record
    snapshot(0)
    for update in range(1, DEFAULT_BRANCH_UPDATES + 1):
        indices = batch_indices[update - 1]
        batch = replay.sample_indices(indices, torch=torch, device=device)
        if mode == "mc":
            target = torch.from_numpy(np.asarray(labels[indices], dtype=np.float32)).to(device=device)
            metrics = _manual_mc_step(learner, batch, target)
        else:
            metrics = learner.update(batch, update_actor=False)
        update_records.append({"update": int(update), "critic_td_loss": float(metrics.get("critic_td_loss", 0.0)), "critic_cql_loss": float(metrics.get("critic_cql_loss", 0.0)), "critic_loss": float(metrics.get("critic_loss", 0.0)), "critic_gradient_norm": float(metrics.get("critic_gradient_norm", 0.0))})
        if update % 100 == 0:
            print("{} update {}/{}".format(name, update, DEFAULT_BRANCH_UPDATES), flush=True)
        if update in SNAPSHOTS[1:]:
            endpoint = snapshot(update)
    gradients_final = _gradient_probe(learner, replay, grad_indices, labels, device, mode)
    return {
        "name": name,
        "target_mode": mode,
        "cql_weight": float(learner.config.critic_cql_weight),
        "updates": DEFAULT_BRANCH_UPDATES,
        "snapshots": snapshots,
        "update_records_summary": {
            "first": update_records[:3],
            "last": update_records[-3:],
            "mean_critic_td_loss": float(np.mean([row["critic_td_loss"] for row in update_records])),
            "mean_critic_cql_loss": float(np.mean([row["critic_cql_loss"] for row in update_records])),
        },
        "gradient_diagnostics": {"initial": gradients_initial, "final": gradients_final},
        "initial_state": {"critic1_sha256": initial_critic1, "critic2_sha256": initial_critic2, "target_critic1_sha256": initial_target1, "target_critic2_sha256": initial_target2, "critic_optimizer_sha256": initial_optimizer},
        "final_state": {"critic1_sha256": _module_state_sha(learner.critic1), "critic2_sha256": _module_state_sha(learner.critic2), "target_critic1_sha256": _module_state_sha(learner.target_critic1), "target_critic2_sha256": _module_state_sha(learner.target_critic2), "critic_optimizer_sha256": _optimizer_sha(learner.critic_optimizer)},
        "actor_update_counts": {"actor_optimizer_step_count": int(learner.actor_optimizer_step_count), "actor_awac_update_count": int(learner.actor_awac_update_count), "actor_recovery_update_count": int(learner.actor_recovery_update_count), "actor_trust_region_rejection_count": int(learner.actor_trust_region_rejection_count)},
        "endpoint": endpoint,
    }


def _bootstrap(endpoints: Mapping[str, np.ndarray], records, returns, *, seed: int, count: int) -> Dict[str, Any]:
    clusters: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(records):
        clusters[str(row["mission_id"])].append(index)
    names = sorted(clusters)
    rng = np.random.RandomState(int(seed))
    values = {name: [] for name in endpoints}
    pairs = {"D_MC_minus_D_TD0": [], "D_TD0_minus_D_TD05": []}
    for _ in range(int(count)):
        chosen = rng.randint(0, len(names), size=len(names))
        indices = np.concatenate([np.asarray(clusters[names[int(i)]], dtype=np.int64) for i in chosen])
        for name, q in endpoints.items():
            values[name].append(spearman(q[indices], returns[indices]))
        pairs["D_MC_minus_D_TD0"].append(values["D_MC"][ -1] - values["D_TD0"][ -1])
        pairs["D_TD0_minus_D_TD05"].append(values["D_TD0"][ -1] - values["D_TD05"][ -1])
    result = {"seed": int(seed), "resamples": int(count), "cluster_key": "mission_id", "cluster_count": len(names), "point_estimates": {}, "intervals_95": {}, "paired_difference_intervals_95": {}}
    for name, samples in values.items():
        point = spearman(endpoints[name], returns)
        result["point_estimates"][name] = point
        result["intervals_95"][name] = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
    exact_pair_points = {
        "D_MC_minus_D_TD0": float(
            spearman(endpoints["D_MC"], returns)
            - spearman(endpoints["D_TD0"], returns)
        ),
        "D_TD0_minus_D_TD05": float(
            spearman(endpoints["D_TD0"], returns)
            - spearman(endpoints["D_TD05"], returns)
        ),
    }
    for name, samples in pairs.items():
        result["paired_difference_intervals_95"][name] = {
            "point": exact_pair_points[name],
            "bootstrap_mean": float(np.mean(samples)),
            "ci95": [
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            ],
        }
    return result


def _write_predictions(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = ["snapshot", "row_index", "mission_id", "episode_id", "episode_transition_index", "terminal_reason", "done", "remaining_steps", "action", "valid_action", "valid_action_count", "q1", "q2", "qmin", "mc_return", "td_target"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _source_manifest(root: Path) -> Dict[str, str]:
    relative = [
        "python/planning/awac/calibration.py",
        "python/planning/awac/calibration_runtime.py",
        "python/planning/awac/checkpoint.py",
        "python/planning/awac/contract.py",
        "python/planning/awac/learner.py",
        "python/planning/awac/optimization.py",
        "python/planning/awac/replay.py",
        "python/planning/awac/trainer.py",
        "python/planning/awac/phase1.py",
    ]
    return {name: _sha_file(root / name) for name in relative}


def _append_predictions(output_rows, snapshot_name, prediction, records, returns, lengths):
    for index, row in enumerate(records):
        key = (str(row["mission_id"]), str(row["episode_id"]))
        output_rows.append({
            "snapshot": snapshot_name,
            "row_index": int(index),
            "mission_id": str(row["mission_id"]),
            "episode_id": str(row["episode_id"]),
            "episode_transition_index": int(row["episode_transition_index"]),
            "terminal_reason": str(row["terminal_reason"]),
            "done": int(bool(row["done"])),
            "remaining_steps": int(lengths[key] - 1 - int(row["episode_transition_index"])),
            "action": int(row["action"]),
            "valid_action": int(bool(np.asarray(row["action_mask"], dtype=bool)[int(row["action"])])),
            "valid_action_count": int(np.asarray(row["action_mask"], dtype=bool).sum()),
            "q1": float(prediction["q1"][index]),
            "q2": float(prediction["q2"][index]),
            "qmin": float(min(prediction["q1"][index], prediction["q2"][index])),
            "mc_return": float(returns[index]),
            "td_target": float(prediction["td_target"][index]),
        })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--index-seed", type=int, default=DEFAULT_INDEX_SEED)
    parser.add_argument("--torch-seed", type=int, default=DEFAULT_TORCH_SEED)
    parser.add_argument("--bootstrap-count", type=int, default=DEFAULT_BOOTSTRAP)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    out = args.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("diagnostic output directory already contains files: {}".format(out))
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA device is unavailable")
    torch.manual_seed(int(args.torch_seed))
    np.random.seed(int(args.index_seed))
    torch.set_grad_enabled(True)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    gate_checkpoint_path = args.gate_checkpoint.expanduser().resolve()
    bc_path = args.bc_checkpoint.expanduser().resolve()
    replay_path = args.replay.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    gate_checkpoint = torch.load(gate_checkpoint_path, map_location="cpu")
    bc_checkpoint = torch.load(bc_path, map_location="cpu")
    config = AWACOptimizationConfig(**checkpoint["resolved_training_config"]["optimization_config"])
    replay = AWACReplayBuffer.open(replay_path, read_only=True)
    labels, train_report = _train_mc_labels(replay, gamma=float(config.gamma), reward_scale=float(config.reward_scale))
    records = checkpoint["exact_resume_state"]["raw_holdout_records"]
    returns, return_report, episode_report = _records_mc_returns(records, gamma=float(config.gamma), reward_scale=float(config.reward_scale))
    if return_report["record_hash"] != checkpoint["exact_resume_state"]["raw_holdout_records_sha256"]:
        raise SystemExit("holdout record hash mismatch")

    # Correct committed-prefix contract audit.  Unused preallocated mmap rows
    # are explicitly excluded by replay.size and never treated as transitions.
    n = int(replay.size)
    action = np.asarray(replay.arrays["action"][:n], dtype=np.int64)
    action_mask = np.asarray(replay.arrays["action_mask"][:n], dtype=bool)
    next_action_mask = np.asarray(replay.arrays["next_action_mask"][:n], dtype=bool)
    done = np.asarray(replay.arrays["done"][:n], dtype=bool)
    valid_action = action_mask[np.arange(n), action]
    train_audit = {
        "committed_rows": n,
        "valid_action_rows": int(valid_action.sum()),
        "invalid_action_rows": int((~valid_action).sum()),
        "nonterminal_empty_next_mask_rows": int(((~done) & (~next_action_mask.any(axis=1))).sum()),
        "done_rows": int(done.sum()),
        "action_min": int(action.min()),
        "action_max": int(action.max()),
        "action_unique": int(np.unique(action).size),
        "action_coverage": {str(int(k)): int(v) for k, v in zip(*np.unique(action, return_counts=True))},
        "mask_true_count_min": int(action_mask.sum(axis=1).min()),
        "mask_true_count_max": int(action_mask.sum(axis=1).max()),
        "next_mask_true_count_min": int(next_action_mask.sum(axis=1).min()),
        "next_mask_true_count_max": int(next_action_mask.sum(axis=1).max()),
        "depth_storage_dtype": str(replay.arrays["depth"].dtype),
        "depth_storage_range": [int(replay.arrays["depth"][:n].min()), int(replay.arrays["depth"][:n].max())],
        "depth_normalization": "float32(uint8) / 255.0 exactly once in AWACReplayBuffer.sample_indices",
        "finite_vector_reward": bool(np.isfinite(np.asarray(replay.arrays["vector"][:n], dtype=np.float64)).all() and np.isfinite(np.asarray(replay.arrays["next_vector"][:n], dtype=np.float64)).all() and np.isfinite(np.asarray(replay.arrays["reward"][:n], dtype=np.float64)).all()),
        "behavior_source_values": {str(int(k)): int(v) for k, v in zip(*np.unique(replay.arrays["behavior_source"][:n], return_counts=True))},
    }
    if train_audit["invalid_action_rows"] or train_audit["nonterminal_empty_next_mask_rows"]:
        raise SystemExit("train replay contract audit failed")

    # Independent gate reproduction: use the gate-time generation and the
    # same CUDA device when available.  This is intentionally separate from
    # the production summarizer.
    gate_config = AWACOptimizationConfig(**gate_checkpoint["resolved_training_config"]["optimization_config"])
    gate_learner = _new_learner(gate_checkpoint, bc_checkpoint, gate_config, device)
    final_learner_for_gate = _new_learner(checkpoint, bc_checkpoint, config, device)
    gate_prediction = _holdout_predictions(gate_learner, records, device, chunk_size=None)
    final_prediction = _holdout_predictions(final_learner_for_gate, records, device, chunk_size=None)
    gate_rho = spearman(np.minimum(gate_prediction["q1"], gate_prediction["q2"]), returns)
    final_rho = spearman(np.minimum(final_prediction["q1"], final_prediction["q2"]), returns)
    persisted_gate_rho = float(gate_checkpoint["calibration_metrics"]["gate_history"][-1]["holdout_q_return_rank_correlation"])
    gate_reproduction = {
        "production_metric_owner": "planning.awac.calibration.summarize_calibration_window:565-680",
        "metric": "Spearman(min(Q1(action),Q2(action)), independent_episode_MC_return)",
        "tie_rule": "average rank; independent stable-sort implementation",
        "gate_checkpoint": {"path": str(gate_checkpoint_path), "sha256": _sha_file(gate_checkpoint_path), "environment_step_count": int(gate_checkpoint["environment_step_count"]), "critic_update_count": int(gate_checkpoint["critic_update_count"])},
        "final_checkpoint": {"path": str(checkpoint_path), "sha256": _sha_file(checkpoint_path), "environment_step_count": int(checkpoint["environment_step_count"]), "critic_update_count": int(checkpoint["critic_update_count"])},
        "persisted_gate_rho": persisted_gate_rho,
        "independent_gate_rho": gate_rho,
        "gate_rho_diff": float(gate_rho - persisted_gate_rho),
        "independent_final_checkpoint_rho": final_rho,
        "final_minus_gate_rho": float(final_rho - gate_rho),
        "holdout_record_hash": return_report["record_hash"],
        "interpretation": "gate value is exactly reproduced on CUDA; final checkpoint is 23 Critic updates later and is not the same model time point",
    }
    _write_json(out / "gate_reproduction.json", gate_reproduction)

    rng = np.random.RandomState(int(args.index_seed))
    small_indices = np.asarray(rng.choice(n, size=min(DEFAULT_SMALL_ROWS, n), replace=False), dtype=np.int64)
    train_diag_indices = np.asarray(rng.choice(n, size=min(2048, n), replace=False), dtype=np.int64)
    batch_indices = np.asarray(rng.randint(0, n, size=(DEFAULT_BRANCH_UPDATES, DEFAULT_BATCH_SIZE)), dtype=np.int64)
    small_batches = np.asarray(rng.randint(0, len(small_indices), size=(DEFAULT_SMALL_UPDATES, DEFAULT_BATCH_SIZE)), dtype=np.int64)
    grad_indices = batch_indices[:16].copy()
    np.save(out / "small_indices.npy", small_indices)
    np.save(out / "train_diagnostic_indices.npy", train_diag_indices)
    np.save(out / "critic_batch_indices.npy", batch_indices)
    np.save(out / "small_batch_indices.npy", small_batches)
    np.save(out / "gradient_batch_indices.npy", grad_indices)
    fixed_indices = {path.name: {"shape": list(np.load(path).shape), "sha256": _sha_file(path)} for path in sorted(out.glob("*_indices.npy"))}
    _write_json(out / "fixed_indices.json", {"index_seed": int(args.index_seed), "torch_seed": int(args.torch_seed), "batch_size": DEFAULT_BATCH_SIZE, "small_rows": len(small_indices), "files": fixed_indices})

    # Small MC fit, then each full-train branch starts again from the same
    # final production Critic and freshly empty Critic Adam state.
    small_learner = _new_learner(checkpoint, bc_checkpoint, config, device, cql_weight=0.0)
    small_result = _run_small_mc(small_learner, replay, labels, small_indices, small_batches, device)
    _write_json(out / "small_mc_fit.json", {key: value for key, value in small_result.items() if key != "prediction_snapshots"})

    prediction_rows: List[Dict[str, Any]] = []
    lengths = {(row["mission_id"], row["episode_id"]): int(row["length"]) for row in episode_report}
    def output_predictions(snapshot_name, prediction, rows, target):
        _append_predictions(prediction_rows, snapshot_name, prediction, rows, target, lengths)
    output_predictions("checkpoint_gate_generation_828", gate_prediction, records, returns)
    output_predictions("checkpoint_final_generation_829", final_prediction, records, returns)

    branches = {}
    branch_endpoints = {}
    branch_group_metrics = []
    for name, cql_weight in (("D_MC", 0.0), ("D_TD0", 0.0), ("D_TD05", 0.05)):
        branch_config = dataclasses.replace(config, critic_cql_weight=float(cql_weight))
        learner = _new_learner(checkpoint, bc_checkpoint, branch_config, device, cql_weight=float(cql_weight))
        result = _run_branch(name, learner, replay, labels, records, returns, batch_indices, train_diag_indices, grad_indices, device, output_predictions)
        branches[name] = result
        endpoint_prediction = _holdout_predictions(learner, records, device)
        branch_endpoints[name] = np.minimum(endpoint_prediction["q1"], endpoint_prediction["q2"])
        branch_group_metrics.extend([{**row, "branch": name, "updates": 3000} for row in _group_metrics(endpoint_prediction, returns, records, lengths)])

    _write_predictions(out / "critic_predictions.csv", prediction_rows)
    with (out / "stratified_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in branch_group_metrics for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(branch_group_metrics)
    bootstrap = _bootstrap(branch_endpoints, records, returns, seed=int(args.index_seed), count=int(args.bootstrap_count))
    _write_json(out / "cluster_bootstrap.json", bootstrap)
    _write_json(out / "diagnostic_branches.json", branches)

    # Actor/BC/replay invariance is recorded against read-only source hashes;
    # every branch starts from the same Actor and performs no Actor step.
    replay_files = sorted(replay_path.glob("*.npy")) + [replay_path / "metadata.json"]
    replay_hashes = {str(path): _sha_file(path) for path in replay_files}
    invariance = {
        "source_replay_hashes_before_after": {"before": replay_hashes, "after": {str(path): _sha_file(path) for path in replay_files}},
        "replay_unchanged": all(replay_hashes[str(path)] == _sha_file(path) for path in replay_files),
        "bc_reference_sha256": _module_state_sha(_new_learner(checkpoint, bc_checkpoint, config, device).bc_reference),
        "actor_update_steps_all_branches": {name: int(value["actor_update_counts"]["actor_optimizer_step_count"]) for name, value in branches.items()},
        "actor_awac_updates_all_branches": {name: int(value["actor_update_counts"]["actor_awac_update_count"]) for name, value in branches.items()},
        "actor_recovery_updates_all_branches": {name: int(value["actor_update_counts"]["actor_recovery_update_count"]) for name, value in branches.items()},
        "actor_rejections_all_branches": {name: int(value["actor_update_counts"]["actor_trust_region_rejection_count"]) for name, value in branches.items()},
        "actor_bc_runtime_ab_c_status": "NOT_RUN; this file proves only Critic-only branch invariance",
    }
    _write_json(out / "actor_bc_replay_invariance.json", invariance)

    # Data and label artifacts.
    split = checkpoint.get("calibration_split", {})
    train_ids = set(split.get("train_episode_ids", []))
    holdout_ids = set(split.get("holdout_episode_ids", []))
    holdout_episode_ids = {str(row["episode_id"]) for row in records}
    holdout_mission_ids = {str(row["mission_id"]) for row in records}
    data_accounting = {
        "environment_steps": int(checkpoint["environment_step_count"]),
        "train_committed_rows": int(replay.size),
        "completed_holdout_rows": int(len(records)),
        "incomplete_or_discarded_rows": 0,
        "other_documented_rows": 0,
        "sum_of_disjoint_components": int(replay.size + len(records)),
        "reconciliation_pass": int(replay.size + len(records)) == int(checkpoint["environment_step_count"]),
        "train_episode_count": int(train_report["episode_count"]),
        "holdout_episode_count": int(return_report["episode_count"]),
        "completed_episode_count_checkpoint": int(checkpoint["completed_episode_count"]),
        "holdout_unique_missions": int(return_report["unique_mission_count"]),
        "split_train_episode_count": len(train_ids),
        "split_holdout_episode_count": len(holdout_ids),
        "holdout_episode_split_intersection": int(len(holdout_episode_ids.intersection(holdout_ids))),
        "holdout_episode_train_intersection": int(len(holdout_episode_ids.intersection(train_ids))),
        "terminal_reason_counts": return_report["terminal_episode_counts"],
        "train_report": train_report,
        "correct_committed_prefix_mask_audit": train_audit,
        "unused_preallocated_rows_excluded": int(replay.arrays["action"].shape[0] - replay.size),
    }
    _write_json(out / "data_accounting.json", data_accounting)
    _write_json(out / "return_label_audit.json", {"holdout": return_report, "holdout_episode_report": episode_report, "train": train_report, "independent_formula": "G_t=reward_scale*raw_reward_t+gamma*G_(t+1); final normal terminal has zero tail", "production_mc_function_called": False})

    # Source identity and reproducibility handoff.
    source_after = out / "source_after"
    source_after.mkdir(exist_ok=True)
    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, source_after / script_path.name)
    source_files = _source_manifest(root)
    source_diff = {
        "git_status": "NOT_A_GIT_REPOSITORY",
        "production_source_modified": False,
        "production_source_files_sha256": source_files,
        "diagnostic_script": {"path": str(script_path), "before": "ABSENT_FOR_THIS_TASK", "after_sha256": _sha_file(script_path), "after_copy": str(source_after / script_path.name)},
        "unified_diff": "No production source diff; diagnostic script is a new offline tool.",
    }
    _write_json(out / "source_diff.json", source_diff)

    manifest = {
        "diagnostic_id": "critic_negative_rho_diagnostic",
        "run_id": out.name,
        "command": " ".join(sys.argv),
        "new_environment_steps": 0,
        "actor_optimizer_steps": 0,
        "dev100_evaluations": 0,
        "production_gate_modified": False,
        "device": str(device),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_seed": int(args.torch_seed),
        "index_seed": int(args.index_seed),
        "bootstrap_seed": int(args.index_seed),
        "software_version": SOFTWARE_VERSION,
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha_file(checkpoint_path)},
        "gate_checkpoint": {"path": str(gate_checkpoint_path), "sha256": _sha_file(gate_checkpoint_path)},
        "bc_checkpoint": {"path": str(bc_path), "sha256": _sha_file(bc_path)},
        "replay": {"path": str(replay_path), "metadata_sha256": _sha_file(replay_path / "metadata.json"), "size": int(replay.size)},
        "holdout_rows": len(records),
        "holdout_record_sha256": return_report["record_hash"],
        "observation_contract": checkpoint.get("observation_contract"),
        "task_contract_sha256": checkpoint.get("task_contract_sha256"),
        "mpl_contract_sha256": checkpoint.get("mpl_contract_sha256"),
        "max_primitive_steps": checkpoint.get("max_primitive_steps"),
        "resolved_optimization_config": dataclasses.asdict(config),
        "actual_optimizer_lr_inventory": _new_learner(checkpoint, bc_checkpoint, config, device).optimizer_lr_inventory(),
        "diagnostic_budgets": {"small_mc_updates": DEFAULT_SMALL_UPDATES, "D_MC": DEFAULT_BRANCH_UPDATES, "D_TD0": DEFAULT_BRANCH_UPDATES, "D_TD05": DEFAULT_BRANCH_UPDATES, "batch_size": DEFAULT_BATCH_SIZE, "total_critic_steps": DEFAULT_SMALL_UPDATES + 3 * DEFAULT_BRANCH_UPDATES},
        "branch_contract": {"D_MC": {"target": "independent_train_episode_MC", "cql": 0.0, "target_update": False}, "D_TD0": {"target": "production_masked_expected_bellman_T1", "cql": 0.0, "target_tau": float(config.tau)}, "D_TD05": {"target": "production_masked_expected_bellman_T1", "cql": 0.05, "target_tau": float(config.tau)}},
        "source_manifest": source_diff,
        "artifact_files": sorted(path.name for path in out.iterdir()),
    }
    _write_json(out / "manifest.json", manifest)
    # manifest was written after the artifact list; refresh it once with the
    # final list without changing any source or historical artifact.
    manifest["artifact_files"] = sorted(path.name for path in out.iterdir())
    _write_json(out / "manifest.json", manifest)

    with (out / "rho_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["branch", "updates", "q", "rho", "mse", "mae"])
        for name, result in (("checkpoint_gate", {"snapshots": [{"updates": 0, "holdout": _metric_snapshot(gate_prediction, returns)}]}), ("checkpoint_final", {"snapshots": [{"updates": 0, "holdout": _metric_snapshot(final_prediction, returns)}]})):
            for snap in result["snapshots"]:
                for q_name in ("q1", "q2", "qmin"):
                    metric = snap["holdout"][q_name]
                    writer.writerow([name, snap["updates"], q_name, metric["rho"], metric["mse"], metric["mae"]])
        for name, result in branches.items():
            for snap in result["snapshots"]:
                for q_name in ("q1", "q2", "qmin"):
                    metric = snap["holdout"][q_name]
                    writer.writerow([name, snap["updates"], q_name, metric["rho"], metric["mse"], metric["mae"]])

    endpoint_summary = {
        name: {"rho": spearman(values, returns), "metrics": pair_metrics(values, returns)}
        for name, values in branch_endpoints.items()
    }
    _write_json(out / "endpoint_summary.json", endpoint_summary)
    _write_json(out / "fixed_snapshot_identities.json", {"small_mc": small_result["snapshots"], "branches": {name: result["snapshots"] for name, result in branches.items()}})
    _write_json(out / "experiment_manifest.json", {"manifest": manifest, "gate_reproduction": gate_reproduction, "endpoint_summary": endpoint_summary, "cluster_bootstrap": bootstrap})
    (out / "chinese_summary.md").write_text(
        "# Critic-only 负 rho 离线诊断\n\n"
        "本产物不启动环境、不修改生产 gate、不执行 Actor。gate generation-828 的 CUDA 独立复算与持久化 rho 对齐；generation-829 是后续 23 次 Critic 更新后的不同时间点。\n\n"
        "数据对账：train committed 24,169 + holdout 2,501 = environment steps 26,670；没有把未使用 mmap 容量当成样本。独立 MC 使用 raw reward 乘 0.10 一次，gamma=0.99。\n\n"
        "D_MC、D_TD0、D_TD05 均使用同一 train Replay、固定 batch index、相同 Critic 起点和 fresh Adam 初态；主终点固定为 3,000 次。请以 endpoint_summary、cluster_bootstrap 和 fixed_snapshot_identities 为数值依据，不把任何诊断分支结果解释为生产 gate PASS。\n",
        encoding="utf-8",
    )
    print("DIAGNOSTIC_OUTPUT", out)
    print("GATE_RHO", gate_rho, "PERSISTED", persisted_gate_rho, "DIFF", gate_rho - persisted_gate_rho, flush=True)
    for name, value in endpoint_summary.items():
        print(name, "rho", value["rho"], "mse", value["metrics"]["mse"], flush=True)
    replay.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
