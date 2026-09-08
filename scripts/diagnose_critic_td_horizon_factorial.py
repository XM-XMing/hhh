#!/usr/bin/env python
"""Run the bounded TD-horizon x finite-budget Critic-only factorial.

This is a diagnostic entry point.  It never constructs a Unity/ROS runtime,
never changes the production learner defaults, and never writes a production
checkpoint or Replay.  The four branches share the generation-829 Critic
state, fresh Adam, fixed batches, and the same frozen 127-dimensional Actor.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import difflib
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from planning.awac.calibration import masked_bellman_target
from planning.awac.checkpoint import calibration_holdout_records_sha256
from planning.awac.learner import AWACOptimizationConfig
from planning.awac.model import optimizer_parameter_groups, soft_update
from planning.awac.replay import AWACReplayBuffer
from planning.awac.td_horizon_diagnostic import (
    BudgetCritic,
    DIAGNOSTIC_CRITIC_SCHEMA_ID,
    budget_fraction,
    build_sequence_sidecar,
    masked_expected_value_independent,
    next_budget_steps,
    n_step_discounted_return,
)
from planning.awac.trainer import build_learner
from planning.common.hashing import file_sha256
from planning.version import SOFTWARE_VERSION


DEFAULT_BATCH_SIZE = 128
DEFAULT_UPDATES = 3000
DEFAULT_BOOTSTRAP = 2000
SNAPSHOTS = (0, 300, 1000, 3000)
INDEX_SEED = 20260906
TORCH_SEED = 20260906
BOOTSTRAP_SEED = 20260907
EXPECTED_BATCH_SHA = "8c435ad62a62fe80af564aa9b23825911d053e36b0f2cff589129e9b4dc0b631"
EXPECTED_START_SHA = "05791a5ffe5d1736f1b8c10dcca3e605772918aaf370273b4f2240f9d19dbcaa"
EXPECTED_GATE_SHA = "5928d395c786a9cc7f859f483d570e17f83dab8ff0f2609dc8e8e48069823c2b"
EXPECTED_BC_SHA = "ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2"
EXPECTED_HOLDOUT_SHA = "b819905262e03fba535ff0b9ccd5e500e46baeb24c195d5ce0d609126a73d898"


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
    path.write_text(json.dumps(_json(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_array(value: np.ndarray) -> str:
    return _sha_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def _module_sha(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _optimizer_sha(optimizer: torch.optim.Optimizer) -> str:
    stream = io.BytesIO()
    torch.save(optimizer.state_dict(), stream)
    return _sha_bytes(stream.getvalue())


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _capture_rng() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _fresh_output(base: Path) -> Path:
    base = base.expanduser().resolve()
    if not base.exists() or not any(base.iterdir()):
        base.mkdir(parents=True, exist_ok=True)
        return base
    index = 2
    while True:
        candidate = Path(str(base) + "_r{}".format(index))
        if not candidate.exists() or not any(candidate.iterdir()):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        index += 1


def _hash_tree(path: Path) -> Dict[str, str]:
    root = Path(path).resolve()
    if root.is_file():
        return {str(root): file_sha256(root)}
    return {str(item): file_sha256(item) for item in sorted(root.rglob("*")) if item.is_file()}


def _write_code_changes(path: Path, source_paths: Sequence[Path]) -> None:
    chunks = []
    for source in source_paths:
        content = source.read_text(encoding="utf-8").splitlines(True)
        chunks.extend(difflib.unified_diff([], content, fromfile="/dev/null", tofile=str(source)))
    path.write_text("".join(chunks), encoding="utf-8")


def _records_mc_returns(records: Sequence[Mapping[str, Any]], *, gamma: float, reward_scale: float):
    groups = defaultdict(list)
    for index, row in enumerate(records):
        key = (str(row.get("mission_id", "")), str(row.get("episode_id", "")))
        if not key[0] or not key[1]:
            raise ValueError("holdout row has incomplete identity")
        groups[key].append((index, row))
    returns = np.empty(len(records), dtype=np.float64)
    meta = {
        "episode_index": np.empty(len(records), dtype=np.int64),
        "t": np.empty(len(records), dtype=np.int64),
        "T": np.empty(len(records), dtype=np.int64),
        "recorded_episode_length": np.empty(len(records), dtype=np.int64),
        "steps_to_recorded_episode_end": np.empty(len(records), dtype=np.int64),
        "remaining_budget_steps": np.empty(len(records), dtype=np.int64),
        "next_remaining_budget_steps": np.empty(len(records), dtype=np.int64),
        "remaining_budget_fraction": np.empty(len(records), dtype=np.float32),
        "next_index": np.full(len(records), -1, dtype=np.int64),
        "terminal_row": np.zeros(len(records), dtype=np.uint8),
    }
    episode_report = []
    for episode_no, (key, grouped) in enumerate(sorted(groups.items(), key=lambda item: min(v[0] for v in item[1]))):
        ordered = sorted(grouped, key=lambda item: int(item[1]["episode_transition_index"]))
        indices = [int(index) for index, _ in ordered]
        transition = [int(row["episode_transition_index"]) for _, row in ordered]
        if transition != list(range(len(ordered))) or sum(bool(row["done"]) for _, row in ordered) != 1 or not bool(ordered[-1][1]["done"]):
            raise ValueError("holdout episode is not one contiguous terminal sequence")
        length = len(ordered)
        total = 0.0
        for index, row in reversed(ordered):
            total = float(reward_scale) * float(row["reward"]) + float(gamma) * total
            returns[index] = total
        for position, (index, row) in enumerate(ordered):
            t = int(row["episode_transition_index"])
            meta["episode_index"][index] = episode_no
            meta["t"][index] = t
            meta["T"][index] = 45
            meta["recorded_episode_length"][index] = length
            meta["steps_to_recorded_episode_end"][index] = length - 1 - t
            meta["remaining_budget_steps"][index] = 45 - t
            meta["next_remaining_budget_steps"][index] = 44 - t
            meta["remaining_budget_fraction"][index] = np.float32((45.0 - t) / 45.0)
            meta["terminal_row"][index] = 1 if bool(row["done"]) else 0
            if position + 1 < length:
                meta["next_index"][index] = indices[position + 1]
        episode_report.append({"mission_id": key[0], "episode_id": key[1], "row_indices": indices, "length": length, "terminal_reason": str(ordered[-1][1].get("terminal_reason", ""))})
    return returns, meta, episode_report


def _train_mc_labels(replay: AWACReplayBuffer, *, gamma: float, reward_scale: float) -> Tuple[np.ndarray, List[int]]:
    n = int(replay.size)
    reward = np.asarray(replay.arrays["reward"][:n], dtype=np.float64)
    done = np.asarray(replay.arrays["done"][:n], dtype=bool)
    if n <= 0 or not bool(done[-1]):
        raise ValueError("train Replay does not end on a terminal row")
    labels = np.empty(n, dtype=np.float64)
    ends = np.flatnonzero(done)
    starts = np.concatenate((np.asarray([0]), ends[:-1] + 1))
    lengths = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        total = 0.0
        for row in range(end, start - 1, -1):
            total = float(reward_scale) * float(reward[row]) + float(gamma) * total
            labels[row] = total
        lengths.append(int(end - start + 1))
    return labels, lengths


def _record_batch(records: Sequence[Mapping[str, Any]], indices: np.ndarray, device):
    selected = [records[int(index)] for index in np.asarray(indices).reshape(-1)]
    def stack(name, dtype):
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
    }


def _independent_target(learner, batch: Mapping[str, torch.Tensor], *, device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Independent TD1 calculation; production masked_bellman_target is not called here."""

    active = ~batch["done"].bool()
    target = batch["reward"] * float(learner.config.reward_scale)
    value = torch.zeros_like(target)
    mean_value = torch.zeros_like(target)
    entropy = torch.zeros_like(target)
    effective = torch.zeros_like(target)
    q_min_at_action = torch.zeros_like(target)
    if bool(active.any().item()):
        logits = learner.actor(batch["next_depth"][active], batch["next_vector"][active])
        q1 = learner.target_critic1(batch["next_depth"][active], batch["next_vector"][active])
        q2 = learner.target_critic2(batch["next_depth"][active], batch["next_vector"][active])
        v_min, probabilities = masked_expected_value_independent(
            logits=logits, q1=q1, q2=q2, action_mask=batch["next_action_mask"][active], torch=torch
        )
        q_mean = (q1 + q2) * 0.5
        v_mean = (probabilities * q_mean).sum(dim=1)
        log_prob = torch.log(probabilities.clamp_min(1.0e-12))
        entropy_active = -(probabilities * log_prob).sum(dim=1)
        value[active] = v_min
        mean_value[active] = v_mean
        entropy[active] = entropy_active
        effective[active] = batch["next_action_mask"][active].bool().sum(dim=1).to(dtype=target.dtype)
        target[active] = target[active] + float(learner.config.gamma) * v_min
    return target, {"value_min": value, "value_mean": mean_value, "entropy": entropy, "effective": effective, "q_min_at_action": q_min_at_action}


def _production_target(learner, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    active = ~batch["done"].bool()
    target = batch["reward"] * float(learner.config.reward_scale)
    if bool(active.any().item()):
        logits = learner.actor(batch["next_depth"][active], batch["next_vector"][active])
        q1 = learner.target_critic1(batch["next_depth"][active], batch["next_vector"][active])
        q2 = learner.target_critic2(batch["next_depth"][active], batch["next_vector"][active])
        result = masked_bellman_target(
            bc_logits=logits,
            target_q1=q1,
            target_q2=q2,
            next_action_mask=batch["next_action_mask"][active],
            reward=batch["reward"][active],
            done=batch["done"][active],
            gamma=float(learner.config.gamma),
            reward_scale=float(learner.config.reward_scale),
            torch=torch,
        )
        target = target.clone()
        target[active] = result["target"]
    return target


def _build_horizon_evidence(root: Path, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    env_path = root / "python/planning/runtime/unity_env.py"
    collector_path = root / "python/planning/teacher/rollout_collector.py"
    def lines(path: Path, start: int, end: int) -> List[str]:
        content = path.read_text(encoding="utf-8").splitlines()
        return ["{}:{}".format(index + 1, content[index]) for index in range(start - 1, min(end, len(content)))]
    return {
        "effective_horizon_T": int(checkpoint["max_primitive_steps"]),
        "checkpoint_max_primitive_steps": int(checkpoint["max_primitive_steps"]),
        "task_contract_sha256": str(checkpoint["task_contract_sha256"]),
        "collector_max_steps_argument": lines(collector_path, 295, 308),
        "collector_env_config_assignment": lines(collector_path, 826, 838),
        "unity_default_is_not_used": lines(env_path, 150, 162),
        "unity_timeout_done_logic": lines(env_path, 2292, 2354),
        "timeout_semantics": "task-level max_steps terminal; timeout is included in done and has reward-only target with no bootstrap",
    }


def _make_base_learner(checkpoint: Mapping[str, Any], bc_checkpoint: Mapping[str, Any], config: AWACOptimizationConfig, device):
    learner = build_learner(torch=torch, nn=torch.nn, device=device, bc_checkpoint=bc_checkpoint, config=config)
    learner.load_state_dict(checkpoint)
    learner.critic_optimizer.state.clear()
    learner.actor.eval()
    learner.bc_reference.eval()
    for parameter in learner.actor.parameters():
        parameter.requires_grad_(False)
    for parameter in learner.bc_reference.parameters():
        parameter.requires_grad_(False)
    return learner


class FactorialBranch:
    def __init__(self, *, name: str, n_step: int, use_budget: bool, source: Any, config: AWACOptimizationConfig, replay: AWACReplayBuffer, sidecar: Mapping[str, np.ndarray], device):
        self.name = str(name)
        self.n_step = int(n_step)
        self.use_budget = bool(use_budget)
        self.replay = replay
        self.sidecar = sidecar
        self.device = device
        self.actor = source.actor
        self.bc_reference = source.bc_reference
        self.critic1 = BudgetCritic(nn=torch.nn, depth_channels=source.depth_channels, source_state_dict=source.critic1.state_dict(), device=device)
        self.critic2 = BudgetCritic(nn=torch.nn, depth_channels=source.depth_channels, source_state_dict=source.critic2.state_dict(), device=device)
        self.target_critic1 = BudgetCritic(nn=torch.nn, depth_channels=source.depth_channels, source_state_dict=source.target_critic1.state_dict(), device=device)
        self.target_critic2 = BudgetCritic(nn=torch.nn, depth_channels=source.depth_channels, source_state_dict=source.target_critic2.state_dict(), device=device)
        for target in (self.target_critic1, self.target_critic2):
            target.eval()
            for parameter in target.parameters():
                parameter.requires_grad_(False)
        groups = []
        for prefix, critic in (("critic1", self.critic1), ("critic2", self.critic2)):
            for group in optimizer_parameter_groups(
                critic.base,
                head_lr=float(config.critic_head_lr),
                vector_lr=float(config.critic_vector_lr),
                depth_lr=float(config.critic_depth_lr),
            ):
                copied = dict(group)
                copied["name"] = "{}_{}".format(prefix, group["name"])
                groups.append(copied)
        self.optimizer = torch.optim.Adam(groups)
        self.config = dataclasses.replace(config, critic_cql_weight=0.0)
        self.updates = 0
        self.rng_initial = _capture_rng()

    def _budget(self, row_indices: np.ndarray) -> torch.Tensor:
        if not self.use_budget:
            return torch.zeros(len(row_indices), dtype=torch.float32, device=self.device)
        return torch.from_numpy(np.asarray(self.sidecar["remaining_budget_fraction"][row_indices], dtype=np.float32)).to(device=self.device)

    def _next_budget(self, row_indices: np.ndarray) -> torch.Tensor:
        if not self.use_budget:
            return torch.zeros(len(row_indices), dtype=torch.float32, device=self.device)
        values = np.asarray(self.sidecar["next_remaining_budget_steps"][row_indices], dtype=np.float32) / float(self.sidecar["T"][row_indices[0]])
        return torch.from_numpy(values).to(device=self.device)

    def _expected_value(self, depth, vector, mask, budget):
        logits = self.actor(depth, vector)
        q1 = self.target_critic1(depth, vector, budget)
        q2 = self.target_critic2(depth, vector, budget)
        return masked_expected_value_independent(logits=logits, q1=q1, q2=q2, action_mask=mask, torch=torch)[0]

    def target_for_starts(self, starts: np.ndarray, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        starts = np.asarray(starts, dtype=np.int64).reshape(-1)
        if self.n_step == 1:
            active = ~batch["done"].bool()
            target = batch["reward"] * float(self.config.reward_scale)
            if bool(active.any().item()):
                next_indices = self.sidecar["next_index"][starts[active.detach().cpu().numpy()]]
                # The bootstrap state is the current state of the next row;
                # its input is T-(t+1), not the budget after that next row's
                # action (which would be one step too small).
                next_budget = self._budget(next_indices)
                value = self._expected_value(
                    batch["next_depth"][active], batch["next_vector"][active],
                    batch["next_action_mask"][active], next_budget,
                )
                target = target.clone()
                target[active] = target[active] + float(self.config.gamma) * value
            return target

        indices = np.asarray(self.sidecar["nstep_indices"][starts], dtype=np.int64)
        lengths = np.asarray(self.sidecar["nstep_length"][starts], dtype=np.int64)
        terminal = np.asarray(self.sidecar["nstep_terminal"][starts], dtype=bool)
        safe = np.maximum(indices, 0)
        rewards = np.asarray(self.replay.arrays["reward"][safe], dtype=np.float32)
        target_np = np.zeros(len(starts), dtype=np.float32)
        for offset in range(5):
            valid = offset < lengths
            target_np[valid] += (float(self.config.gamma) ** offset) * float(self.config.reward_scale) * rewards[valid, offset]
        boot_mask = ~terminal
        if bool(boot_mask.any()):
            boot_indices = np.asarray(self.sidecar["bootstrap_index"][starts[boot_mask]], dtype=np.int64)
            boot_batch = self.replay.sample_indices(boot_indices, torch=torch, device=self.device)
            boot_budget = self._budget(boot_indices)
            value = self._expected_value(
                boot_batch["depth"], boot_batch["vector"], boot_batch["action_mask"], boot_budget
            )
            target_np[boot_mask] += (float(self.config.gamma) ** 5) * value.detach().cpu().numpy().astype(np.float32)
        return torch.from_numpy(target_np).to(device=self.device)

    def update(self, starts: np.ndarray) -> Dict[str, float]:
        starts = np.asarray(starts, dtype=np.int64)
        batch = self.replay.sample_indices(starts, torch=torch, device=self.device)
        target = self.target_for_starts(starts, batch)
        budget = self._budget(starts)
        q1_all = self.critic1(batch["depth"], batch["vector"], budget)
        q2_all = self.critic2(batch["depth"], batch["vector"], budget)
        q1 = q1_all.gather(1, batch["action"][:, None]).squeeze(1)
        q2 = q2_all.gather(1, batch["action"][:, None]).squeeze(1)
        loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
        params = [parameter for module in (self.critic1, self.critic2) for parameter in module.parameters() if parameter.requires_grad]
        self.optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("{} Critic loss is non-finite".format(self.name))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(self.config.gradient_clip_norm))
        self.optimizer.step()
        soft_update(self.target_critic1.base, self.critic1.base, float(self.config.tau))
        soft_update(self.target_critic2.base, self.critic2.base, float(self.config.tau))
        self.updates += 1
        return {"critic_loss": float(loss.detach().cpu().item()), "target_mean": float(target.detach().mean().cpu().item())}

    def q_from_batch(self, batch: Mapping[str, torch.Tensor], indices: np.ndarray, sidecar: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            values = self.sidecar if sidecar is None else sidecar
            if self.use_budget:
                budget = torch.from_numpy(np.asarray(values["remaining_budget_fraction"][np.asarray(indices, dtype=np.int64)], dtype=np.float32)).to(device=self.device)
            else:
                budget = torch.zeros(len(indices), dtype=torch.float32, device=self.device)
            q1_all = self.critic1(batch["depth"], batch["vector"], budget)
            q2_all = self.critic2(batch["depth"], batch["vector"], budget)
            q1 = q1_all.gather(1, batch["action"][:, None]).squeeze(1)
            q2 = q2_all.gather(1, batch["action"][:, None]).squeeze(1)
        return {"q1": q1.detach().cpu().numpy(), "q2": q2.detach().cpu().numpy()}

    def save_checkpoint(self, path: Path, *, source_sha: str, sidecar_sha: str, fixed_sha: str, actor_sha: str, bc_sha: str) -> None:
        payload = {
            "schema_id": DIAGNOSTIC_CRITIC_SCHEMA_ID,
            "branch": self.name,
            "updates": int(self.updates),
            "n_step": int(self.n_step),
            "use_budget": bool(self.use_budget),
            "critic_vector_dim": 128,
            "source_checkpoint_sha256": source_sha,
            "sequence_sidecar_sha256": sidecar_sha,
            "fixed_batch_indices_sha256": fixed_sha,
            "actor_state_sha256": actor_sha,
            "bc_reference_sha256": bc_sha,
            "critic1_state_dict": _cpu_copy(self.critic1.state_dict()),
            "critic2_state_dict": _cpu_copy(self.critic2.state_dict()),
            "target_critic1_state_dict": _cpu_copy(self.target_critic1.state_dict()),
            "target_critic2_state_dict": _cpu_copy(self.target_critic2.state_dict()),
            "critic_optimizer_state_dict": _cpu_copy(self.optimizer.state_dict()),
            "rng_state": _cpu_copy(_capture_rng()),
            "resolved_config": dataclasses.asdict(self.config),
        }
        torch.save(payload, str(path))


def _metric(prediction: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.size != target.size or prediction.size == 0:
        return {"n": int(prediction.size), "rho": None, "mse": None, "mae": None}
    def rho(left, right):
        if left.size < 2:
            return None
        def rankdata(values):
            order = np.argsort(values, kind="mergesort")
            ranks = np.empty(values.size, dtype=np.float64)
            index = 0
            while index < values.size:
                end = index + 1
                while end < values.size and values[order[end]] == values[order[index]]:
                    end += 1
                ranks[order[index:end]] = 0.5 * (index + end - 1)
                index = end
            return ranks
        lrank = rankdata(left)
        rrank = rankdata(right)
        lrank -= lrank.mean(); rrank -= rrank.mean()
        denom = math.sqrt(float(np.dot(lrank, lrank) * np.dot(rrank, rrank)))
        return None if denom <= 0 else float(np.dot(lrank, rrank) / denom)
    return {
        "n": int(prediction.size),
        "rho": rho(prediction, target),
        "mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "mean": float(prediction.mean()),
        "std": float(prediction.std()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "p05": float(np.percentile(prediction, 5)),
        "p50": float(np.percentile(prediction, 50)),
        "p95": float(np.percentile(prediction, 95)),
    }


def _metrics_by_group(q: Mapping[str, np.ndarray], target: np.ndarray, meta: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    qmin = np.minimum(q["q1"], q["q2"])
    done = np.asarray(meta["terminal_row"], dtype=bool)
    remaining = np.asarray(meta["remaining_budget_steps"], dtype=np.int64)
    selectors = {
        "all": np.ones(len(target), dtype=bool),
        "terminal": done,
        "nonterminal": ~done,
        "budget_1_4": (remaining >= 1) & (remaining <= 4),
        "budget_5_9": (remaining >= 5) & (remaining <= 9),
        "budget_10_19": (remaining >= 10) & (remaining <= 19),
        "budget_20_plus": remaining >= 20,
    }
    result = {}
    for name, selector in selectors.items():
        indices = np.flatnonzero(selector)
        result[name] = {qname: _metric(values[indices], target[indices]) for qname, values in (("q1", q["q1"]), ("q2", q["q2"]), ("qmin", qmin))}
    return result


def _holdout_prediction(branch: FactorialBranch, records: Sequence[Mapping[str, Any]], meta: Mapping[str, np.ndarray], device, chunk: int = 256) -> Dict[str, np.ndarray]:
    parts = {"q1": [], "q2": []}
    for start in range(0, len(records), chunk):
        indices = np.arange(start, min(len(records), start + chunk), dtype=np.int64)
        batch = _record_batch(records, indices, device)
        values = branch.q_from_batch(batch, indices, sidecar=meta)
        for key in parts:
            parts[key].append(values[key])
    result = {key: np.concatenate(value) for key, value in parts.items()}
    result["qmin"] = np.minimum(result["q1"], result["q2"])
    return result


def _train_prediction(branch: FactorialBranch, replay: AWACReplayBuffer, indices: np.ndarray, device) -> Dict[str, np.ndarray]:
    batch = replay.sample_indices(indices, torch=torch, device=device)
    result = branch.q_from_batch(batch, indices)
    result["qmin"] = np.minimum(result["q1"], result["q2"])
    return result


def _write_prediction_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["branch", "snapshot", "source", "row_index", "episode_index", "mission_id", "episode_id", "t", "T", "remaining_budget_steps", "remaining_budget_fraction", "done", "q1", "q2", "qmin", "target_return"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_bootstrap(path: Path, endpoints: Mapping[str, np.ndarray], records: Sequence[Mapping[str, Any]], returns: np.ndarray, *, seed: int, count: int) -> Dict[str, Any]:
    clusters = defaultdict(list)
    for index, row in enumerate(records):
        clusters[str(row["mission_id"])].append(index)
    names = sorted(clusters)
    rng = np.random.RandomState(int(seed))
    keys = ["E_N1_H1_minus_E_N1_H0", "E_N5_H0_minus_E_N1_H0", "E_N5_H1_minus_E_N5_H0", "E_N5_H1_minus_E_N1_H1", "E_N5_H1_minus_E_N1_H0", "interaction"]
    deltas = {key: [] for key in keys}
    endpoint_rho = {name: _metric(values, returns)["rho"] for name, values in endpoints.items()}
    for _ in range(int(count)):
        chosen = rng.randint(0, len(names), size=len(names))
        selected = np.concatenate([np.asarray(clusters[names[int(index)]], dtype=np.int64) for index in chosen])
        rho = {name: _metric(values[selected], returns[selected])["rho"] for name, values in endpoints.items()}
        if any(value is None for value in rho.values()):
            continue
        d_budget_1 = rho["E_N1_H1"] - rho["E_N1_H0"]
        d_nstep_0 = rho["E_N5_H0"] - rho["E_N1_H0"]
        d_budget_5 = rho["E_N5_H1"] - rho["E_N5_H0"]
        d_nstep_1 = rho["E_N5_H1"] - rho["E_N1_H1"]
        d_joint = rho["E_N5_H1"] - rho["E_N1_H0"]
        deltas["E_N1_H1_minus_E_N1_H0"].append(d_budget_1)
        deltas["E_N5_H0_minus_E_N1_H0"].append(d_nstep_0)
        deltas["E_N5_H1_minus_E_N5_H0"].append(d_budget_5)
        deltas["E_N5_H1_minus_E_N1_H1"].append(d_nstep_1)
        deltas["E_N5_H1_minus_E_N1_H0"].append(d_joint)
        deltas["interaction"].append(d_budget_5 - d_budget_1)
    result = {"seed": int(seed), "requested": int(count), "valid": int(len(deltas[keys[0]])), "endpoint_rho": endpoint_rho, "deltas": {}}
    for key, values in deltas.items():
        array = np.asarray(values, dtype=np.float64)
        result["deltas"][key] = {"estimate": float(endpoint_rho[key.split("_minus_")[0]] - endpoint_rho[key.split("_minus_")[1]]) if "_minus_" in key and key != "interaction" else None, "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))] if array.size else None, "n": int(array.size)}
    result["deltas"]["interaction"]["estimate"] = float((endpoint_rho["E_N5_H1"] - endpoint_rho["E_N5_H0"]) - (endpoint_rho["E_N1_H1"] - endpoint_rho["E_N1_H0"]))
    _write_json(path, result)
    return result


def _audit_td1(*, learner, replay: AWACReplayBuffer, train_sidecar: Mapping[str, np.ndarray], train_labels: np.ndarray, records: Sequence[Mapping[str, Any]], holdout_meta: Mapping[str, np.ndarray], holdout_returns: np.ndarray, out_csv: Path, device) -> Dict[str, Any]:
    fields = ["source", "row_index", "episode_index", "t", "T", "recorded_episode_length", "steps_to_recorded_episode_end", "remaining_budget_steps", "remaining_budget_fraction", "done", "scaled_reward", "G", "G_next", "next_policy_entropy", "next_policy_effective_action_count", "V_target_expected_min", "V_target_expected_mean", "Q_target_min_at_recorded_next_action", "Y_TD1", "Y_minus_G", "V_minus_G_next", "V_minus_Q_recorded", "Q_recorded_minus_G_next", "production_Y_TD1", "production_abs_diff"]
    max_abs = 0.0
    max_rel = 0.0
    rows = []
    def consume(source: str, source_records, meta, labels, count: int, fetch):
        nonlocal max_abs, max_rel
        for start in range(0, count, 256):
            indices = np.arange(start, min(count, start + 256), dtype=np.int64)
            batch, next_actions = fetch(indices)
            independent, info = _independent_target(learner, batch, device=device)
            production = _production_target(learner, batch)
            active = ~batch["done"].bool()
            q_recorded = torch.zeros_like(independent)
            if bool(active.any().item()):
                next_rows = np.asarray(meta["next_index"][indices[active.detach().cpu().numpy()]], dtype=np.int64)
                next_batch, next_action_values = fetch(next_rows)
                q1 = learner.target_critic1(next_batch["depth"], next_batch["vector"])
                q2 = learner.target_critic2(next_batch["depth"], next_batch["vector"])
                q_recorded[active] = torch.minimum(q1, q2).gather(1, next_action_values[:, None]).squeeze(1)
            diff = torch.abs(independent - production).detach().cpu().numpy()
            denom = torch.maximum(torch.abs(independent), torch.full_like(independent, 1.0e-12))
            rel = (torch.abs(independent - production) / denom).detach().cpu().numpy()
            if diff.size:
                max_abs = max(max_abs, float(diff.max()))
                max_rel = max(max_rel, float(rel.max()))
            independent_np = independent.detach().cpu().numpy()
            production_np = production.detach().cpu().numpy()
            value_min = info["value_min"].detach().cpu().numpy()
            value_mean = info["value_mean"].detach().cpu().numpy()
            entropy = info["entropy"].detach().cpu().numpy()
            effective = info["effective"].detach().cpu().numpy()
            q_recorded_np = q_recorded.detach().cpu().numpy()
            g = np.asarray(labels[indices], dtype=np.float64)
            next_indices = np.asarray(meta["next_index"][indices], dtype=np.int64)
            g_next = np.zeros(len(indices), dtype=np.float64)
            valid_next = next_indices >= 0
            g_next[valid_next] = np.asarray(labels[next_indices[valid_next]], dtype=np.float64)
            for local, row_index in enumerate(indices.tolist()):
                row = source_records[int(row_index)] if source_records is not None else None
                rows.append({
                    "source": source,
                    "row_index": int(row_index),
                    "episode_index": int(meta["episode_index"][row_index]),
                    "t": int(meta["t"][row_index]),
                    "T": int(meta["T"][row_index]),
                    "recorded_episode_length": int(meta["recorded_episode_length"][row_index]),
                    "steps_to_recorded_episode_end": int(meta["steps_to_recorded_episode_end"][row_index]),
                    "remaining_budget_steps": int(meta["remaining_budget_steps"][row_index]),
                    "remaining_budget_fraction": float(meta["remaining_budget_fraction"][row_index]),
                    "done": int(bool(meta["terminal_row"][row_index])),
                    "scaled_reward": float(batch["reward"][local].detach().cpu().item() * float(learner.config.reward_scale)),
                    "G": float(g[local]),
                    "G_next": float(g_next[local]),
                    "next_policy_entropy": float(entropy[local]),
                    "next_policy_effective_action_count": float(effective[local]),
                    "V_target_expected_min": float(value_min[local]),
                    "V_target_expected_mean": float(value_mean[local]),
                    "Q_target_min_at_recorded_next_action": float(q_recorded_np[local]),
                    "Y_TD1": float(independent_np[local]),
                    "Y_minus_G": float(independent_np[local] - g[local]),
                    "V_minus_G_next": float(value_min[local] - g_next[local]),
                    "V_minus_Q_recorded": float(value_min[local] - q_recorded_np[local]),
                    "Q_recorded_minus_G_next": float(q_recorded_np[local] - g_next[local]),
                    "production_Y_TD1": float(production_np[local]),
                    "production_abs_diff": float(diff[local]),
                })
    train_n = int(replay.size)
    train_actions = np.asarray(replay.arrays["action"][:train_n], dtype=np.int64)
    def fetch_train(indices):
        batch = replay.sample_indices(indices, torch=torch, device=device)
        return batch, torch.from_numpy(train_actions[indices]).to(device=device).long()
    consume("train", None, train_sidecar, train_labels, train_n, fetch_train)
    holdout_actions = np.asarray([int(row["action"]) for row in records], dtype=np.int64)
    def fetch_holdout(indices):
        batch = _record_batch(records, indices, device)
        return batch, torch.from_numpy(holdout_actions[indices]).to(device=device).long()
    consume("holdout", records, holdout_meta, holdout_returns, len(records), fetch_holdout)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    terminal = np.asarray([row["done"] for row in rows], dtype=bool)
    residual = np.asarray([row["Y_minus_G"] for row in rows], dtype=np.float64)
    group_report = {}
    for name, selector in (("all", np.ones(len(rows), dtype=bool)), ("terminal", terminal), ("nonterminal", ~terminal)):
        values = residual[selector]
        group_report[name] = {"n": int(values.size), "mean": float(values.mean()) if values.size else None, "p50": float(np.percentile(values, 50)) if values.size else None, "p95_abs": float(np.percentile(np.abs(values), 95)) if values.size else None, "mse": float(np.mean(values * values)) if values.size else None}
    return {"rows": len(rows), "max_abs_diff": max_abs, "max_rel_diff": max_rel, "allclose_atol_1e-6_rtol_1e-5": bool(max_abs <= 1.0e-6 + 1.0e-5 * 1.0), "residual_groups": group_report, "formula": "Y-G=gamma*(V_target_expected_min-G_next) for nonterminal; terminal target is reward_scale*reward"}


def _initial_equivalence(*, source, branches: Mapping[str, FactorialBranch], replay, sidecar, records, holdout_meta, device, train_indices, source_sha, bc_sha) -> Dict[str, Any]:
    rows = []
    old_q = []
    new_q = {name: [] for name in branches}
    for start in range(0, len(train_indices), 256):
        indices = train_indices[start : start + 256]
        batch = replay.sample_indices(indices, torch=torch, device=device)
        with torch.no_grad():
            old1 = source.critic1(batch["depth"], batch["vector"])
            old2 = source.critic2(batch["depth"], batch["vector"])
        old_q.extend([old1.detach().cpu().numpy(), old2.detach().cpu().numpy()])
        for name, branch in branches.items():
            with torch.no_grad():
                zeros = torch.zeros(len(indices), dtype=torch.float32, device=device)
                q1 = branch.critic1(batch["depth"], batch["vector"], zeros)
                q2 = branch.critic2(batch["depth"], batch["vector"], zeros)
            new_q[name].append({"q1": q1.detach().cpu().numpy(), "q2": q2.detach().cpu().numpy()})
    old1 = np.concatenate([value for value in old_q[0::2]])
    old2 = np.concatenate([value for value in old_q[1::2]])
    output = {"source_checkpoint_sha256": source_sha, "bc_checkpoint_sha256": bc_sha, "train_rows": int(len(train_indices)), "old_vs_extended": {}}
    for name, parts in new_q.items():
        q1 = np.concatenate([value["q1"] for value in parts]); q2 = np.concatenate([value["q2"] for value in parts])
        output["old_vs_extended"][name] = {"q1_max_abs_diff": float(np.max(np.abs(old1 - q1))), "q2_max_abs_diff": float(np.max(np.abs(old2 - q2))), "budget_column_initialization": "zero", "actor_vector_dim": 127, "critic_vector_dim": 128}
    output["H0_vs_H1_initial_q_max_abs_diff"] = max(
        output["old_vs_extended"]["E_N1_H0"]["q1_max_abs_diff"],
        output["old_vs_extended"]["E_N1_H1"]["q1_max_abs_diff"],
    )
    output["target_lag"] = {
        name: {
            "critic1_parameter_l2": float(sum(torch.linalg.vector_norm(a - b).detach().cpu().item() for a, b in zip(branch.critic1.parameters(), branch.target_critic1.parameters()))),
            "critic2_parameter_l2": float(sum(torch.linalg.vector_norm(a - b).detach().cpu().item() for a, b in zip(branch.critic2.parameters(), branch.target_critic2.parameters()))),
        }
        for name, branch in branches.items()
    }
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--bootstrap-count", type=int, default=DEFAULT_BOOTSTRAP)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if int(args.updates) != DEFAULT_UPDATES or int(args.batch_size) != DEFAULT_BATCH_SIZE:
        raise SystemExit("this registered diagnostic requires exactly 3000 updates and batch_size=128")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA device is unavailable")
    root = Path(__file__).resolve().parents[1]
    out = _fresh_output(args.out_dir)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    gate_path = args.gate_checkpoint.expanduser().resolve()
    bc_path = args.bc_checkpoint.expanduser().resolve()
    replay_path = args.replay.expanduser().resolve()
    source_before = {
        "checkpoint": {"path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path)},
        "gate_checkpoint": {"path": str(gate_path), "sha256": file_sha256(gate_path)},
        "bc_checkpoint": {"path": str(bc_path), "sha256": file_sha256(bc_path)},
        "replay_files": _hash_tree(replay_path),
        "diagnostic_source_files_at_start": {
            "script": file_sha256(Path(__file__).resolve()),
            "module": file_sha256(root / "python/planning/awac/td_horizon_diagnostic.py"),
        },
    }
    _write_json(out / "source_before_after_sha256.json", {"before": source_before, "after": None, "before_captured_before_training": True})
    _write_code_changes(out / "code_changes.diff", [Path(__file__).resolve(), root / "python/planning/awac/td_horizon_diagnostic.py", root / "tests/test_td_horizon_diagnostic.py"])

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    gate_checkpoint = torch.load(str(gate_path), map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(str(bc_path), map_location="cpu", weights_only=False)
    if source_before["checkpoint"]["sha256"] != EXPECTED_START_SHA:
        raise SystemExit("generation-829 checkpoint SHA mismatch")
    if source_before["gate_checkpoint"]["sha256"] != EXPECTED_GATE_SHA:
        raise SystemExit("generation-828 gate checkpoint SHA mismatch")
    if source_before["bc_checkpoint"]["sha256"] != EXPECTED_BC_SHA:
        raise SystemExit("BC checkpoint SHA mismatch")
    if int(checkpoint.get("max_primitive_steps")) != 45:
        raise SystemExit("effective horizon is not the validated T=45")
    if int(checkpoint.get("environment_step_count")) != 26670 or int(checkpoint.get("completed_episode_count")) != 1000:
        raise SystemExit("source checkpoint accounting identity mismatch")
    records = checkpoint["exact_resume_state"]["raw_holdout_records"]
    holdout_hash = calibration_holdout_records_sha256(records)
    if holdout_hash != EXPECTED_HOLDOUT_SHA or holdout_hash != checkpoint["exact_resume_state"]["raw_holdout_records_sha256"]:
        raise SystemExit("holdout identity mismatch")
    config = AWACOptimizationConfig(**checkpoint["resolved_training_config"]["optimization_config"])
    config_cql0 = dataclasses.replace(config, critic_cql_weight=0.0)
    replay = AWACReplayBuffer.open(replay_path, read_only=True)
    if int(replay.size) != 24169:
        raise SystemExit("source Replay size mismatch")
    train_labels, train_lengths = _train_mc_labels(replay, gamma=float(config.gamma), reward_scale=float(config.reward_scale))
    train_sidecar, sidecar_report = build_sequence_sidecar(arrays=replay.arrays, count=int(replay.size), horizon=45, episode_lengths=train_lengths)
    np.savez_compressed(out / "sequence_sidecar.npz", **train_sidecar)
    sidecar_sha = file_sha256(out / "sequence_sidecar.npz")
    _write_json(out / "sequence_sidecar.json", {**sidecar_report, "path": str(out / "sequence_sidecar.npz"), "sha256": sidecar_sha})
    holdout_returns, holdout_meta, episode_report = _records_mc_returns(records, gamma=float(config.gamma), reward_scale=float(config.reward_scale))
    np.savez_compressed(out / "holdout_sequence_sidecar.npz", **holdout_meta)
    _write_json(out / "holdout_sequence_sidecar.json", {"row_count": len(records), "episode_count": len(episode_report), "sha256": file_sha256(out / "holdout_sequence_sidecar.npz"), "holdout_record_sha256": holdout_hash})
    fixed_root = root / "data/awac/diagnostics/critic_negative_rho_v1_20260906_w16"
    fixed_source = fixed_root / "critic_batch_indices.npy"
    fixed_sha = file_sha256(fixed_source)
    if fixed_sha != EXPECTED_BATCH_SHA:
        raise SystemExit("fixed Critic batch index SHA mismatch")
    fixed_batches = np.load(fixed_source, allow_pickle=False)
    if fixed_batches.shape != (3000, 128):
        raise SystemExit("fixed Critic batch shape mismatch")
    np.save(out / "fixed_critic_batch_indices.npy", fixed_batches)
    fixed_output_sha = file_sha256(out / "fixed_critic_batch_indices.npy")
    train_diag_source = fixed_root / "train_diagnostic_indices.npy"
    train_diag_indices = np.load(train_diag_source, allow_pickle=False).astype(np.int64)
    np.save(out / "fixed_train_diagnostic_indices.npy", train_diag_indices)
    fixed_report = {"source_path": str(fixed_source), "source_sha256": fixed_sha, "output_sha256": fixed_output_sha, "shape": list(fixed_batches.shape), "index_seed": INDEX_SEED, "torch_seed": TORCH_SEED}
    _write_json(out / "fixed_batch_indices.json", fixed_report)

    evidence = _build_horizon_evidence(root, checkpoint)
    _write_json(out / "horizon_evidence.json", evidence)
    source_learner = _make_base_learner(checkpoint, bc_checkpoint, config_cql0, device)
    actor_sha = _module_sha(source_learner.actor)
    bc_sha = _module_sha(source_learner.bc_reference)
    _write_json(out / "actor_bc_identity_before.json", {"actor_state_sha256": actor_sha, "bc_reference_sha256": bc_sha, "actor_vector_dim": 127, "bc_checkpoint_sha256": source_before["bc_checkpoint"]["sha256"]})

    td1_report = _audit_td1(
        learner=source_learner,
        replay=replay,
        train_sidecar=train_sidecar,
        train_labels=train_labels,
        records=records,
        holdout_meta=holdout_meta,
        holdout_returns=holdout_returns,
        out_csv=out / "bootstrap_decomposition.csv",
        device=device,
    )
    _write_json(out / "bootstrap_decomposition.json", td1_report)

    # Build all four branches from independent copies of the same generation-
    # 829 online/target Critics.  The Actor/BC is shared, frozen, and never
    # passed through an optimizer in this diagnostic.
    branch_specs = (("E_N1_H0", 1, False), ("E_N1_H1", 1, True), ("E_N5_H0", 5, False), ("E_N5_H1", 5, True))
    branches = {}
    for name, n_step, use_budget in branch_specs:
        torch.manual_seed(TORCH_SEED)
        np.random.seed(INDEX_SEED)
        random.seed(INDEX_SEED)
        branch = FactorialBranch(name=name, n_step=n_step, use_budget=use_budget, source=source_learner, config=config_cql0, replay=replay, sidecar=train_sidecar, device=device)
        branches[name] = branch

    equivalence = _initial_equivalence(
        source=source_learner,
        branches={name: branches[name] for name in ("E_N1_H0", "E_N1_H1")},
        replay=replay,
        sidecar=train_sidecar,
        records=records,
        holdout_meta=holdout_meta,
        device=device,
        train_indices=train_diag_indices,
        source_sha=source_before["checkpoint"]["sha256"],
        bc_sha=source_before["bc_checkpoint"]["sha256"],
    )
    _write_json(out / "initial_equivalence.json", equivalence)

    prediction_rows = []
    branch_reports = {}
    endpoints = {}
    train_meta_fixed = {key: np.asarray(value)[train_diag_indices] for key, value in train_sidecar.items()}
    for name, branch in branches.items():
        branch_dir = out / "branches" / name
        branch_dir.mkdir(parents=True, exist_ok=True)
        snapshots = []
        update_summaries = []

        def take_snapshot(step):
            if int(branch.updates) != int(step):
                raise RuntimeError("branch snapshot schedule mismatch")
            holdout_q = _holdout_prediction(branch, records, holdout_meta, device)
            train_q = _train_prediction(branch, replay, train_diag_indices, device)
            holdout_metrics = {qname: _metric(holdout_q[qname], holdout_returns) for qname in ("q1", "q2", "qmin")}
            train_metrics = {qname: _metric(train_q[qname], train_labels[train_diag_indices]) for qname in ("q1", "q2", "qmin")}
            record = {"updates": int(step), "holdout": holdout_metrics, "train_fixed": train_metrics, "holdout_groups": _metrics_by_group(holdout_q, holdout_returns, holdout_meta), "actor_optimizer_steps": 0, "actor_update_steps": 0, "critic_updates": int(branch.updates), "critic1_sha256": _module_sha(branch.critic1), "critic2_sha256": _module_sha(branch.critic2), "target_critic1_sha256": _module_sha(branch.target_critic1), "target_critic2_sha256": _module_sha(branch.target_critic2), "critic_optimizer_sha256": _optimizer_sha(branch.optimizer)}
            snapshots.append(record)
            for source_name, values, target, meta, source_indices in (("holdout", holdout_q, holdout_returns, holdout_meta, np.arange(len(records), dtype=np.int64)), ("train_fixed", train_q, train_labels[train_diag_indices], train_meta_fixed, train_diag_indices)):
                for local, row_index in enumerate(source_indices.tolist()):
                    row = records[row_index] if source_name == "holdout" else None
                    prediction_rows.append({"branch": name, "snapshot": int(step), "source": source_name, "row_index": int(row_index), "episode_index": int(meta["episode_index"][local]), "mission_id": str(row.get("mission_id", "")) if row else "", "episode_id": str(row.get("episode_id", "")) if row else "", "t": int(meta["t"][local]), "T": int(meta["T"][local]), "remaining_budget_steps": int(meta["remaining_budget_steps"][local]), "remaining_budget_fraction": float(meta["remaining_budget_fraction"][local]), "done": int(meta["terminal_row"][local]), "q1": float(values["q1"][local]), "q2": float(values["q2"][local]), "qmin": float(values["qmin"][local]), "target_return": float(target[local])})
            branch.save_checkpoint(branch_dir / "checkpoint_{}.pt".format(step), source_sha=source_before["checkpoint"]["sha256"], sidecar_sha=sidecar_sha, fixed_sha=fixed_sha, actor_sha=actor_sha, bc_sha=bc_sha)
            if int(step) == DEFAULT_UPDATES:
                endpoints[name] = holdout_q["qmin"].copy()

        take_snapshot(0)
        for update in range(1, DEFAULT_UPDATES + 1):
            update_summaries.append(branch.update(fixed_batches[update - 1]))
            if update % 100 == 0:
                print("{} update {}/{}".format(name, update, DEFAULT_UPDATES), flush=True)
            if update in SNAPSHOTS[1:]:
                take_snapshot(update)
        branch_reports[name] = {"name": name, "n_step": int(branch.n_step), "use_budget": bool(branch.use_budget), "cql_weight": 0.0, "updates": int(branch.updates), "snapshots": snapshots, "update_summary": {"mean_loss": float(np.mean([row["critic_loss"] for row in update_summaries])), "p95_loss": float(np.percentile([row["critic_loss"] for row in update_summaries], 95)), "final_loss": float(update_summaries[-1]["critic_loss"])}}
        branches[name] = None
        del branch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_prediction_rows(out / "critic_predictions.csv", prediction_rows)
    _write_json(out / "branch_metrics.json", branch_reports)
    bootstrap = _write_bootstrap(out / "paired_mission_bootstrap.json", endpoints, records, holdout_returns, seed=BOOTSTRAP_SEED, count=int(args.bootstrap_count))
    source_after = {"checkpoint": {"path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path)}, "gate_checkpoint": {"path": str(gate_path), "sha256": file_sha256(gate_path)}, "bc_checkpoint": {"path": str(bc_path), "sha256": file_sha256(bc_path)}, "replay_files": _hash_tree(replay_path), "actor_state_sha256": _module_sha(source_learner.actor), "bc_reference_sha256": _module_sha(source_learner.bc_reference)}
    _write_json(out / "source_before_after_sha256.json", {"before": source_before, "after": source_after, "before_captured_before_training": True, "historical_inputs_unchanged": source_before["checkpoint"] == source_after["checkpoint"] and source_before["gate_checkpoint"] == source_after["gate_checkpoint"] and source_before["bc_checkpoint"] == source_after["bc_checkpoint"] and source_before["replay_files"] == source_after["replay_files"]})
    _write_json(out / "invariance.json", {"actor_state_sha256_before": actor_sha, "actor_state_sha256_after": _module_sha(source_learner.actor), "bc_reference_sha256_before": bc_sha, "bc_reference_sha256_after": _module_sha(source_learner.bc_reference), "actor_optimizer_steps": 0, "actor_updates": 0, "replay_unchanged": source_before["replay_files"] == source_after["replay_files"], "production_gate_modified": False, "production_defaults_changed": False})
    manifest = {"diagnostic_id": "critic_td_horizon_factorial", "run_id": out.name, "software_version": SOFTWARE_VERSION, "python": sys.executable, "python_version": platform.python_version(), "torch_version": torch.__version__, "device": str(device), "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "checkpoint": source_before["checkpoint"], "gate_checkpoint": source_before["gate_checkpoint"], "bc_checkpoint": source_before["bc_checkpoint"], "replay": {"path": str(replay_path), "size": int(replay.size), "metadata_sha256": file_sha256(replay_path / "metadata.json")}, "holdout_rows": len(records), "holdout_record_sha256": holdout_hash, "effective_horizon_T": 45, "timeout_semantics": evidence["timeout_semantics"], "sequence_sidecar_sha256": sidecar_sha, "fixed_batch_indices_sha256": fixed_sha, "fixed_batch_indices_shape": list(fixed_batches.shape), "batch_size": 128, "updates_per_branch": 3000, "total_critic_optimizer_steps": 12000, "new_environment_steps": 0, "actor_optimizer_steps": 0, "dev100_evaluations": 0, "final300_used": False, "cql_weight_all_branches": 0.0, "branch_contract": {name: {"n_step": n, "use_budget": budget, "target": "frozen_BC_masked_categorical_T1_expected_min", "critic_vector_dim": 128} for name, n, budget in branch_specs}, "td1_audit": td1_report, "initial_equivalence": equivalence, "paired_bootstrap": bootstrap, "historical_artifacts_unchanged": source_before["replay_files"] == source_after["replay_files"]}
    _write_json(out / "manifest.json", manifest)
    (out / "report_zh.md").write_text("# TD bootstrap × 有限时域预算 2×2 Critic-only 诊断\n\n本报告由固定 generation-829、只读 calibration Replay/holdout 和复用的固定 batch 索引生成。四个分支各执行 3,000 次 Critic 更新，共 12,000 次；没有环境步、Actor 更新、Dev100 或 Final300。\n\n生产 Actor/BC 仍为 127 维；诊断 Critic 仅追加零初始化预算列。`steps_to_recorded_episode_end` 只作为事后审计字段，预算输入使用 `T-t`。timeout 是任务内终止，目标不 bootstrap。请以 `branch_metrics.json`、`paired_mission_bootstrap.json` 和逐行 `bootstrap_decomposition.csv` 为证据，不将诊断 rho 写成生产 calibration PASS。\n", encoding="utf-8")
    _write_json(out / "source_identity.json", {"generation_829_sha256": source_before["checkpoint"]["sha256"], "generation_828_gate_sha256": source_before["gate_checkpoint"]["sha256"], "bc_sha256": source_before["bc_checkpoint"]["sha256"], "holdout_sha256": holdout_hash, "effective_T": 45})
    replay.close()
    print("DIAGNOSTIC_OUTPUT", out)
    print("TD1_MAX_ABS_DIFF", td1_report["max_abs_diff"])
    for name, values in endpoints.items():
        print(name, "rho", _metric(values, holdout_returns)["rho"], "mse", _metric(values, holdout_returns)["mse"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
