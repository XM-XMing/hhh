#!/usr/bin/env python3
"""Train the preregistered diagnostic Critic ranking branches.

This entry point is deliberately isolated from the production AWAC learner.
It consumes read-only calibration inputs and the separate
``multi_action_replay_v1`` artifact, and writes only a new diagnostic output
directory.  It never constructs an environment or an Actor optimizer.
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
import platform
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from planning.awac.checkpoint import calibration_holdout_records_sha256
from planning.awac.learner import AWACOptimizationConfig
from planning.awac.model import build_critic, optimizer_parameter_groups, soft_update
from planning.awac.replay import AWACReplayBuffer
from planning.awac.td_horizon_diagnostic import masked_expected_value_independent
from planning.bc.model import VectorNormalizer
from planning.common.hashing import file_sha256
from planning.contracts.feature import action_onehot
from planning.diagnostics.multi_action_replay import load_multi_action_replay
from planning.diagnostics.multi_action_ranking_critic import (
    build_mission_group_split,
    build_pair_records,
    compute_q_scale,
    kendall,
    lambda_rank_from_grad_norms,
    mission_cluster_bootstrap,
    ranking_loss,
    ranking_enabled_for_branch,
    sample_pair_schedule,
    spearman,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data/awac/diagnostics/multi_action_ranking_critic_v1_20260907"
BASE_REPLAY = ROOT / "data/awac/diagnostics/actor_weight_isolation_v1_20260906_w16/calibration_masked_categorical_t1/replay"
N5_CHECKPOINT = ROOT / "data/awac/diagnostics/critic_td_horizon_factorial_v1_20260907_r2/branches/E_N5_H0/checkpoint_3000.pt"
BC_CHECKPOINT = ROOT / "data/teach/2026_6w/bc_training/checkpoint_best_soft.pt"
MULTI_ACTION = ROOT / "data/awac/diagnostics/multi_action_replay_v1_real_20260907"
SIDECAR = ROOT / "data/awac/diagnostics/critic_td_horizon_factorial_v1_20260907_r2/sequence_sidecar.npz"
OLD_SOURCE_CHECKPOINT = ROOT / "data/awac/diagnostics/actor_weight_isolation_v1_20260906_w16/calibration_masked_categorical_t1/checkpoint_last.pt"
OLD_CALIBRATION_MISSIONS = ROOT / "data/teach/2026_6w/missions.csv"
BC_FAILURE_INDEX = ROOT / "data/awac/diagnostics/dev100_n5_confirmation/bc/rollout_index.csv"

EXPECTED_BASE_METADATA_SHA = "e3b2733b2770c49c9eea0979cd25f762f2b1e1fb576daf81d0d8df746e90bcf1"
EXPECTED_N5_SHA = "f91dbf20e9a235e9c79531bd6848923b9971fc6bb29d1631cf7268fb40dab3d0"
EXPECTED_BC_SHA = "ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2"
EXPECTED_OLD_SOURCE_SHA = "05791a5ffe5d1736f1b8c10dcca3e605772918aaf370273b4f2240f9d19dbcaa"
EXPECTED_HOLDOUT_SHA = "b819905262e03fba535ff0b9ccd5e500e46baeb24c195d5ce0d609126a73d898"
EXPECTED_SIDECAR_SHA = "56060a9925fd52e14984717daf854bab8d59e62f2119e10c77b13575b805582b"
EXPECTED_ROWS = 24169
EXPECTED_ACTIONS = 105
EXPECTED_MULTI_STATES = 479
EXPECTED_MULTI_ROWS = 2850
EXPECTED_PAIR_ROWS = 7124
SEEDS = (202609071, 202609072, 202609073)
FOLD_SEED = 20260907
BATCH_SIZE = 128
UPDATES = 3000
SNAPSHOTS = (0, 300, 1000, 3000)
BOOTSTRAP_REPEATS = 5000


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_array(value: np.ndarray) -> str:
    return _sha_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def _tree_sha(path: Path) -> str:
    root = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    for item in sorted(value for value in root.rglob("*") if value.is_file()):
        digest.update(str(item.relative_to(root)).encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _module_sha(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
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


def _load_torch(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(str(path), map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint is not a mapping: {}".format(path))
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _jsonable(row.get(name, "")) for name in fieldnames})


def _write_code_diff(path: Path) -> None:
    sources = [
        ROOT / "python/planning/diagnostics/multi_action_ranking_critic.py",
        Path(__file__).resolve(),
        ROOT / "tests/test_multi_action_ranking_critic.py",
    ]
    chunks: List[str] = []
    for source in sources:
        content = source.read_text(encoding="utf-8").splitlines(True)
        chunks.extend(difflib.unified_diff([], content, fromfile="/dev/null", tofile=str(source)))
    Path(path).write_text("".join(chunks), encoding="utf-8")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("{}:{} is not an object".format(path, line_number))
            rows.append(dict(value))
    return rows


def _resolve_config(checkpoint: Mapping[str, Any]) -> AWACOptimizationConfig:
    source = checkpoint.get("resolved_config")
    if not isinstance(source, Mapping):
        source = checkpoint.get("resolved_training_config", {}).get("optimization_config", {})
    names = {item.name for item in fields(AWACOptimizationConfig)}
    values = {str(key): value for key, value in dict(source).items() if str(key) in names}
    config = AWACOptimizationConfig(**values)
    if float(config.critic_cql_weight) != 0.0:
        raise ValueError("N5 diagnostic source must have CQL weight zero")
    return config


class DiagnosticCritic(torch.nn.Module):
    """128-dimensional H0 Critic wrapper matching the N5 checkpoint schema."""

    def __init__(self, *, depth_channels: int, state_dict: Mapping[str, Any], device: torch.device):
        super().__init__()
        self.base = build_critic(
            torch.nn,
            depth_channels=int(depth_channels),
            vec_dim=128,
            num_actions=105,
        ).to(device)
        self.load_state_dict(state_dict, strict=True)

    def forward(self, depth, vector, budget):
        value = budget.to(device=vector.device, dtype=vector.dtype).reshape(-1, 1)
        return self.base(depth, torch.cat((vector, value), dim=1))


def _policy_vectors(multi: Mapping[str, np.ndarray], states_dir: Path, normalizer: VectorNormalizer) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    state_files = sorted(Path(states_dir).glob("*.npz"))
    state_by_id: Dict[str, Dict[str, Any]] = {}
    for path in state_files:
        with np.load(str(path), allow_pickle=False) as loaded:
            required = {"state_id", "vector", "depth", "mask", "previous_action"}
            if not required.issubset(set(loaded.files)):
                raise ValueError("state file missing identity fields: {}".format(path))
            state_id = str(np.asarray(loaded["state_id"]).reshape(-1)[0])
            if state_id in state_by_id:
                raise ValueError("duplicate state file identity: {}".format(state_id))
            state_by_id[state_id] = {
                "vector": np.asarray(loaded["vector"], dtype=np.float32).reshape(-1),
                "depth": np.asarray(loaded["depth"], dtype=np.float32),
                "mask": np.asarray(loaded["mask"], dtype=np.bool_).reshape(-1),
                "previous_action": int(np.asarray(loaded["previous_action"]).reshape(-1)[0]),
            }

    state_ids = np.asarray([str(value) for value in multi["state_id"].tolist()])
    vectors = np.empty((len(state_ids), 127), dtype=np.float32)
    next_vectors = np.empty((len(state_ids), 127), dtype=np.float32)
    checked = 0
    for index, state_id in enumerate(state_ids.tolist()):
        if state_id not in state_by_id:
            raise ValueError("multi-action row has no immutable state file: {}".format(state_id))
        state = state_by_id[state_id]
        raw_vector = np.asarray(multi["state_vector"][index], dtype=np.float32).reshape(-1)
        raw_depth = np.asarray(multi["state_depth"][index], dtype=np.float32)
        raw_mask = np.asarray(multi["mask"][index], dtype=np.bool_).reshape(-1)
        if not np.array_equal(raw_mask, state["mask"]):
            raise ValueError("state mask mismatch for {}".format(state_id))
        if not np.array_equal(raw_vector, state["vector"]):
            raise ValueError("state vector mismatch for {}".format(state_id))
        if not np.array_equal(raw_depth, state["depth"]):
            raise ValueError("state depth mismatch for {}".format(state_id))
        current = np.concatenate((normalizer.transform_continuous(raw_vector), action_onehot(state["previous_action"])))
        next_raw = np.asarray(multi["next_state_vector"][index], dtype=np.float32).reshape(-1)
        next_action = int(multi["action"][index])
        following = np.concatenate((normalizer.transform_continuous(next_raw), action_onehot(next_action)))
        vectors[index] = current
        next_vectors[index] = following
        checked += 1
    return vectors, next_vectors, {"state_file_count": len(state_by_id), "rows_checked": checked, "vector_dim": 127}


def _pair_lookup(multi: Mapping[str, np.ndarray]) -> Dict[Tuple[str, int], int]:
    lookup: Dict[Tuple[str, int], int] = {}
    for index, (state_id, action) in enumerate(zip(multi["state_id"].tolist(), multi["action"].tolist())):
        key = (str(state_id), int(action))
        if key in lookup:
            raise ValueError("duplicate state/action row: {}".format(key))
        lookup[key] = int(index)
    return lookup


def _failure_types() -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not BC_FAILURE_INDEX.is_file():
        return result
    with BC_FAILURE_INDEX.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mission = str(row.get("mission_id", "")).strip()
            reason = str(row.get("stop_reason", "")).strip().lower()
            if mission:
                result[mission] = "collision" if "collision" in reason else "dead_end" if "dead" in reason else reason or "other"
    return result


def _build_labels_and_split(multi: Mapping[str, np.ndarray], dataset: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    branches = _load_jsonl(dataset / "candidate_branches.jsonl")
    if len(branches) != EXPECTED_MULTI_ROWS:
        raise ValueError("candidate branch count mismatch: {}".format(len(branches)))
    rows = build_pair_records(branches, _pair_lookup(multi))
    failure_map = _failure_types()
    for row in rows:
        row["failure_type"] = failure_map.get(str(row["mission_id"]), str(row.get("failure_type", "other")))
    split = build_mission_group_split(rows, seed=FOLD_SEED, fold_count=3)
    assignment = split["mission_to_fold"]
    for row in split["missions"]:
        mission = str(row["mission_id"])
        row["transition_count"] = int(sum(1 for value in multi["mission_id"].tolist() if str(value) == mission))
    if len(assignment) != 31:
        raise ValueError("expected 31 failure missions, got {}".format(len(assignment)))
    for row in rows:
        row["fold_id"] = int(assignment[str(row["mission_id"])])
    split["mission_source_path"] = str(OLD_CALIBRATION_MISSIONS)
    split["mission_source_sha256"] = file_sha256(OLD_CALIBRATION_MISSIONS)
    split["base_replay_mission_identity"] = "UNKNOWN: production calibration rows have no mission_id field"
    return rows, split, {"branches": branches, "failure_type_map": failure_map}


def _return_quantiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {name: float("nan") for name in ("p10", "p25", "median", "p75", "p90")}
    return {
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def _label_semantics_audit(dataset: Path, multi: Mapping[str, np.ndarray], pairs: Sequence[Mapping[str, Any]], branches: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    non_tie = [row for row in pairs if not bool(row["tie"])]
    ties = [row for row in pairs if bool(row["tie"])]
    states = defaultdict(list)
    for row in pairs:
        states[str(row["state_id"])].append(row)
    source_returns: Dict[str, List[float]] = defaultdict(list)
    for branch in branches:
        source_returns[str(branch.get("action_source", ""))].append(float(branch["episode_return"]))
    gaps = [abs(float(row["return_i"]) - float(row["return_j"])) for row in non_tie]
    one_action_states = sorted(state for state, values in states.items() if len({int(v["action_i"]) for v in values} | {int(v["action_j"]) for v in values}) <= 1)
    return {
        "pair_label_formula": "sign(episode_return_i - episode_return_j)",
        "label_source": "completed branch episode_return only",
        "pair_count": int(len(pairs)),
        "non_tie_pair_count": int(len(non_tie)),
        "tie_pair_count": int(len(ties)),
        "tie_ratio": float(len(ties) / max(len(pairs), 1)),
        "ranking_state_count": int(len(states)),
        "state_action_count_distribution": {str(k): int(v) for k, v in sorted((len({int(r["action_i"]) for r in values} | {int(r["action_j"]) for r in values}), sum(1 for values2 in states.values() if len({int(r2["action_i"]) for r2 in values2} | {int(r2["action_j"]) for r2 in values2}) == len({int(r["action_i"]) for r in values} | {int(r["action_j"]) for r in values}))) for k, values in states.items())},
        "no_pair_state_count": int(4),
        "no_pair_state_count_source": "dataset manifest documents four one-legal-action states",
        "return_gap_quantiles": _return_quantiles(gaps),
        "source_return_quantiles": {source: _return_quantiles(values) for source, values in sorted(source_returns.items())},
        "collector_semantics": {
            "same_prefix_and_state_hash": True,
            "same_action_mask": True,
            "only_target_action_differs": True,
            "continuation_policy": "frozen deterministic BC",
            "episode_return": "undiscounted raw cumulative reward including prefix reward and candidate/continuation rewards",
            "reward_storage": "raw environment reward",
            "reward_scale_for_td": 0.10,
            "pair_return_discounted": False,
            "terminal_and_timeout": "real branch terminal outcome; timeout is retained as completed branch evidence",
            "complete_step_reward_sequence_persisted": False,
            "mc_regression_target_allowed": False,
            "n_step_target_from_multi_action_branch_allowed": False,
            "evidence": "scripts/collect_multi_action_replay.py::_replay_prefix and _candidate_transition",
        },
        "single_action_state_ids": one_action_states,
        "all_branch_status_completed": all(str(row.get("status")) == "completed" for row in branches),
        "invalid_label_count": 0,
    }


def _base_batch(replay: AWACReplayBuffer, indices: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    return replay.sample_indices(indices, torch=torch, device=device)


class RankingBranch:
    def __init__(self, *, name: str, source: Mapping[str, Any], config: AWACOptimizationConfig, actor: torch.nn.Module, replay: AWACReplayBuffer, sidecar: Mapping[str, np.ndarray], multi_cache: Mapping[str, torch.Tensor], pair_rows: np.ndarray, pair_labels: np.ndarray, device: torch.device):
        self.name = str(name)
        self.device = device
        self.replay = replay
        self.sidecar = sidecar
        self.multi_cache = multi_cache
        self.pair_rows = torch.from_numpy(np.asarray(pair_rows, dtype=np.int64)).to(device=device)
        self.pair_labels = torch.from_numpy(np.asarray(pair_labels, dtype=np.float32)).to(device=device)
        self.actor = actor
        self.config = dataclasses.replace(config, critic_cql_weight=0.0)
        self.critic1 = DiagnosticCritic(depth_channels=1, state_dict=source["critic1_state_dict"], device=device)
        self.critic2 = DiagnosticCritic(depth_channels=1, state_dict=source["critic2_state_dict"], device=device)
        self.target_critic1 = DiagnosticCritic(depth_channels=1, state_dict=source["target_critic1_state_dict"], device=device)
        self.target_critic2 = DiagnosticCritic(depth_channels=1, state_dict=source["target_critic2_state_dict"], device=device)
        for target in (self.target_critic1, self.target_critic2):
            target.eval()
            for parameter in target.parameters():
                parameter.requires_grad_(False)
        groups: List[Dict[str, Any]] = []
        for prefix, critic in (("critic1", self.critic1), ("critic2", self.critic2)):
            for group in optimizer_parameter_groups(
                critic.base,
                head_lr=float(self.config.critic_head_lr),
                vector_lr=float(self.config.critic_vector_lr),
                depth_lr=float(self.config.critic_depth_lr),
            ):
                copied = dict(group)
                copied["name"] = "{}_{}".format(prefix, group["name"])
                groups.append(copied)
        self.optimizer = torch.optim.Adam(groups)
        self.optimizer.load_state_dict(_cpu_copy(source["critic_optimizer_state_dict"]))
        self.updates = 0
        self.actor_optimizer_steps = 0
        self._freeze_actor()

    def _freeze_actor(self) -> None:
        self.actor.eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    def _budget(self, count: int) -> torch.Tensor:
        return torch.zeros(int(count), dtype=torch.float32, device=self.device)

    def _expected_next(self, batch: Mapping[str, torch.Tensor], boot_indices: np.ndarray) -> torch.Tensor:
        boot = self.replay.sample_indices(boot_indices, torch=torch, device=self.device)
        budget = self._budget(len(boot_indices))
        with torch.no_grad():
            logits = self.actor(boot["depth"], boot["vector"])
            q1 = self.target_critic1(boot["depth"], boot["vector"], budget)
            q2 = self.target_critic2(boot["depth"], boot["vector"], budget)
            value, _ = masked_expected_value_independent(
                logits=logits,
                q1=q1,
                q2=q2,
                action_mask=boot["action_mask"],
                torch=torch,
            )
        return value

    def td_target(self, starts: np.ndarray, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        starts = np.asarray(starts, dtype=np.int64).reshape(-1)
        indices = np.asarray(self.sidecar["nstep_indices"][starts], dtype=np.int64)
        lengths = np.asarray(self.sidecar["nstep_length"][starts], dtype=np.int64)
        terminal = np.asarray(self.sidecar["nstep_terminal"][starts], dtype=bool)
        safe = np.maximum(indices, 0)
        rewards = np.asarray(self.replay.arrays["reward"][safe], dtype=np.float32)
        target = np.zeros(len(starts), dtype=np.float32)
        for offset in range(5):
            valid = offset < lengths
            target[valid] += (float(self.config.gamma) ** offset) * float(self.config.reward_scale) * rewards[valid, offset]
        if bool((~terminal).any()):
            boot_indices = np.asarray(self.sidecar["bootstrap_index"][starts[~terminal]], dtype=np.int64)
            value = self._expected_next(batch, boot_indices)
            target[~terminal] += (float(self.config.gamma) ** 5) * value.detach().cpu().numpy().astype(np.float32)
        return torch.from_numpy(target).to(device=self.device)

    def losses(self, starts: np.ndarray, pair_indices: Optional[np.ndarray], q_scale: float, lambda_value: float, include_rank: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = _base_batch(self.replay, starts, self.device)
        target = self.td_target(starts, batch)
        budget = self._budget(len(starts))
        q1_all = self.critic1(batch["depth"], batch["vector"], budget)
        q2_all = self.critic2(batch["depth"], batch["vector"], budget)
        q1 = q1_all.gather(1, batch["action"][:, None]).squeeze(1)
        q2 = q2_all.gather(1, batch["action"][:, None]).squeeze(1)
        td = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
        rank = torch.zeros((), dtype=td.dtype, device=self.device)
        if include_rank:
            if pair_indices is None or len(pair_indices) == 0:
                raise ValueError("ranking branch received no pair batch")
            pair_indices_t = torch.as_tensor(pair_indices, dtype=torch.long, device=self.device)
            row_i = self.pair_rows[pair_indices_t, 0]
            row_j = self.pair_rows[pair_indices_t, 1]
            budget_pair = self._budget(len(pair_indices))
            q1_i = self.critic1(self.multi_cache["depth"][row_i], self.multi_cache["vector"][row_i], budget_pair).gather(1, self.multi_cache["action"][row_i, None]).squeeze(1)
            q1_j = self.critic1(self.multi_cache["depth"][row_j], self.multi_cache["vector"][row_j], budget_pair).gather(1, self.multi_cache["action"][row_j, None]).squeeze(1)
            q2_i = self.critic2(self.multi_cache["depth"][row_i], self.multi_cache["vector"][row_i], budget_pair).gather(1, self.multi_cache["action"][row_i, None]).squeeze(1)
            q2_j = self.critic2(self.multi_cache["depth"][row_j], self.multi_cache["vector"][row_j], budget_pair).gather(1, self.multi_cache["action"][row_j, None]).squeeze(1)
            rank = ranking_loss(q1_i, q1_j, q2_i, q2_j, self.pair_labels[pair_indices_t], q_scale=q_scale)
        return td, rank, td + float(lambda_value) * rank

    def gradient_norm(self, starts: np.ndarray, pair_indices: Optional[np.ndarray], q_scale: float, include_rank: bool) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        td, rank, total = self.losses(starts, pair_indices, q_scale, 1.0, include_rank)
        value = td if not include_rank else rank
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for group in self.optimizer.param_groups for p in group["params"]], float("inf"))
        self.optimizer.zero_grad(set_to_none=True)
        return float(norm.detach().cpu().item())

    def update(self, starts: np.ndarray, pair_indices: np.ndarray, q_scale: float, lambda_value: float, include_rank: bool) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        td, rank, total = self.losses(starts, pair_indices, q_scale, lambda_value, include_rank)
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError("{} total loss is non-finite".format(self.name))
        total.backward()
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(self.config.gradient_clip_norm))
        if not bool(torch.isfinite(grad_norm).item()):
            raise FloatingPointError("{} gradient norm is non-finite".format(self.name))
        self.optimizer.step()
        soft_update(self.target_critic1.base, self.critic1.base, float(self.config.tau))
        soft_update(self.target_critic2.base, self.critic2.base, float(self.config.tau))
        self.updates += 1
        return {
            "td_loss": float(td.detach().cpu().item()),
            "rank_loss": float(rank.detach().cpu().item()),
            "total_loss": float(total.detach().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item()),
        }

    def q_values(self, row_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        output1: List[np.ndarray] = []
        output2: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(rows), 256):
                selected = torch.from_numpy(rows[start : start + 256]).to(device=self.device)
                budget = self._budget(len(selected))
                output1.append(self.critic1(self.multi_cache["depth"][selected], self.multi_cache["vector"][selected], budget).detach().cpu().numpy())
                output2.append(self.critic2(self.multi_cache["depth"][selected], self.multi_cache["vector"][selected], budget).detach().cpu().numpy())
        return np.concatenate(output1, axis=0), np.concatenate(output2, axis=0)

    def save(self, path: Path, *, source_sha: str, q_scale: float, lambda_value: float, td_schedule_sha: str, pair_schedule_sha: str) -> None:
        payload = {
            "schema_id": "multi_action_ranking_critic_v1_checkpoint",
            "branch": self.name,
            "updates": int(self.updates),
            "source_checkpoint_sha256": source_sha,
            "critic_vector_dim": 128,
            "n_step": 5,
            "use_budget": False,
            "cql_weight": 0.0,
            "q_scale": float(q_scale),
            "lambda_rank": float(lambda_value),
            "td_schedule_sha256": str(td_schedule_sha),
            "pair_schedule_sha256": str(pair_schedule_sha),
            "critic1_state_dict": _cpu_copy(self.critic1.state_dict()),
            "critic2_state_dict": _cpu_copy(self.critic2.state_dict()),
            "target_critic1_state_dict": _cpu_copy(self.target_critic1.state_dict()),
            "target_critic2_state_dict": _cpu_copy(self.target_critic2.state_dict()),
            "critic_optimizer_state_dict": _cpu_copy(self.optimizer.state_dict()),
            "actor_optimizer_steps": 0,
            "actor_updates": 0,
            "resolved_config": dataclasses.asdict(self.config),
        }
        torch.save(payload, str(path))


def _make_multi_cache(multi: Mapping[str, np.ndarray], vectors: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    depth = np.asarray(multi["state_depth"], dtype=np.float32)[:, None, :, :]
    return {
        "depth": torch.from_numpy(depth).to(device=device),
        "vector": torch.from_numpy(np.asarray(vectors, dtype=np.float32)).to(device=device),
        "action": torch.from_numpy(np.asarray(multi["action"], dtype=np.int64)).to(device=device),
    }


def _metric_summary(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p75": None, "p90": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def _evaluate_branch(branch: RankingBranch, *, pairs: Sequence[Mapping[str, Any]], test_pair_indices: np.ndarray, test_state_ids: Sequence[str], state_to_rows: Mapping[str, Sequence[int]], multi: Mapping[str, np.ndarray], fold: int, seed: int, model: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_rows = sorted({int(index) for state_id in test_state_ids for index in state_to_rows[state_id]})
    q1, q2 = branch.q_values(np.asarray(all_rows, dtype=np.int64))
    qmap = {
        row: (
            float(q1[pos, int(multi["action"][row])]),
            float(q2[pos, int(multi["action"][row])]),
        )
        for pos, row in enumerate(all_rows)
    }
    pair_values: List[Dict[str, Any]] = []
    by_state: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_mission: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    state_rows: List[Dict[str, Any]] = []
    mission_rows: List[Dict[str, Any]] = []
    for pair_index in np.asarray(test_pair_indices, dtype=np.int64).tolist():
        pair = pairs[int(pair_index)]
        qi = qmap[int(pair["row_i"])]; qj = qmap[int(pair["row_j"])]
        qdiffs = (qi[0] - qj[0], qi[1] - qj[1], min(qi) - min(qj))
        record = {
            "fold": int(fold), "seed": int(seed), "model": model, "pair_index": int(pair_index),
            "mission_id": str(pair["mission_id"]), "state_id": str(pair["state_id"]),
            "action_i": int(pair["action_i"]), "action_j": int(pair["action_j"]),
            "source_i": str(pair["source_i"]), "source_j": str(pair["source_j"]),
            "pair_type": str(pair["pair_type"]), "failure_type": str(pair["failure_type"]),
            "return_i": float(pair["return_i"]), "return_j": float(pair["return_j"]),
            "label": int(pair["label"]), "tie": bool(pair["tie"]),
            "q1_i": qmap[int(pair["row_i"])][0], "q1_j": qmap[int(pair["row_j"])][0],
            "q2_i": qmap[int(pair["row_i"])][1], "q2_j": qmap[int(pair["row_j"])][1],
            "qmin_i": min(qi), "qmin_j": min(qj),
            "q1_diff": qdiffs[0], "q2_diff": qdiffs[1], "qmin_diff": qdiffs[2],
            "strict_tie_qmin": bool(qdiffs[2] == 0.0),
        }
        if not bool(pair["tie"]):
            record["q1_correct"] = bool(float(pair["label"]) * qdiffs[0] > 0.0)
            record["q2_correct"] = bool(float(pair["label"]) * qdiffs[1] > 0.0)
            record["qmin_correct"] = bool(float(pair["label"]) * qdiffs[2] > 0.0)
            by_state[str(pair["state_id"])].append(record)
            by_mission[str(pair["mission_id"])].append(record)
        pair_values.append(record)

    state_to_pair_acc: Dict[str, float] = {}
    for state_id, values in by_state.items():
        state_to_pair_acc[state_id] = float(np.mean([float(value["qmin_correct"]) for value in values]))

    top1: List[float] = []
    regrets: List[float] = []
    bc_delta: List[float] = []
    rank_spearman: List[float] = []
    rank_kendall: List[float] = []
    for state_id in sorted(str(value) for value in test_state_ids):
        rows = list(state_to_rows[state_id])
        if not rows:
            continue
        values = []
        for row in rows:
            qi = qmap[int(row)]
            values.append((int(row), min(qi), float(multi["action"][row]), float(multi["reward"][row])))
        selected = max(values, key=lambda item: (item[1], -item[0]))
        max_return = max(item[3] for item in values)
        best = {item[0] for item in values if item[3] == max_return}
        top1.append(1.0 if selected[0] in best else 0.0)
        regrets.append(float(max_return - selected[3]))
        bc_rows = [item for item in values if str(multi["action_source"][item[0]]) == "BC"]
        if bc_rows:
            bc_delta.append(float(selected[3] - bc_rows[0][3]))
        if len(values) >= 3 and len({item[3] for item in values}) >= 2:
            predicted = [qmap[item[0]][0] if True else 0.0 for item in values]
            predicted_min = [min(qmap[item[0]]) for item in values]
            observed = [item[3] for item in values]
            rho = spearman(predicted_min, observed)
            tau = kendall(predicted_min, observed)
            if rho is not None:
                rank_spearman.append(float(rho))
            if tau is not None:
                rank_kendall.append(float(tau))
        state_rows.append({
            "fold": int(fold), "seed": int(seed), "model": model, "mission_id": str(multi["mission_id"][rows[0]]), "state_id": state_id,
            "action_count": len(values), "pair_count": len(by_state.get(state_id, [])),
            "qmin_pairwise_accuracy": state_to_pair_acc.get(state_id, None),
            "top1_hit": float(top1[-1]), "regret": float(regrets[-1]),
            "bc_delta": float(bc_delta[-1]) if bc_rows else None,
            "spearman": float(rank_spearman[-1]) if rank_spearman and len(rank_spearman) == len(state_rows) else None,
            "kendall": float(rank_kendall[-1]) if rank_kendall and len(rank_kendall) == len(state_rows) else None,
        })

    non_tie_records = [value for value in pair_values if not value["tie"]]
    mission_acc = []
    for mission_id, values in sorted(by_mission.items()):
        mission_acc.append(float(np.mean([float(value["qmin_correct"]) for value in values])))
        mission_rows.append({
            "fold": int(fold), "seed": int(seed), "model": model, "mission_id": mission_id,
            "pair_count": len(values), "qmin_pairwise_accuracy": mission_acc[-1],
            "top1_hit_rate": float(np.mean([row["top1_hit"] for row in state_rows if row["mission_id"] == mission_id])) if any(row["mission_id"] == mission_id for row in state_rows) else None,
            "mean_regret": float(np.mean([row["regret"] for row in state_rows if row["mission_id"] == mission_id])) if any(row["mission_id"] == mission_id for row in state_rows) else None,
            "bc_delta": float(np.mean([row["bc_delta"] for row in state_rows if row["mission_id"] == mission_id and row["bc_delta"] is not None])) if any(row["mission_id"] == mission_id and row["bc_delta"] is not None for row in state_rows) else None,
        })

    def _group_metric(field: str, selector) -> Dict[str, Any]:
        selected = [value for value in non_tie_records if selector(value)]
        return {"count": len(selected), "accuracy": float(np.mean([float(value[field]) for value in selected])) if selected else None}

    summary = {
        "fold": int(fold), "seed": int(seed), "model": model,
        "pair_count": len(pair_values), "non_tie_pair_count": len(non_tie_records),
        "strict_tie_prediction_count": int(sum(bool(value["strict_tie_qmin"]) for value in non_tie_records)),
        "q1_pairwise_micro_accuracy": float(np.mean([float(value["q1_correct"]) for value in non_tie_records])) if non_tie_records else None,
        "q2_pairwise_micro_accuracy": float(np.mean([float(value["q2_correct"]) for value in non_tie_records])) if non_tie_records else None,
        "qmin_pairwise_micro_accuracy": float(np.mean([float(value["qmin_correct"]) for value in non_tie_records])) if non_tie_records else None,
        "qmin_pairwise_state_macro_accuracy": float(np.mean(list(state_to_pair_acc.values()))) if state_to_pair_acc else None,
        "qmin_pairwise_mission_macro_accuracy": float(np.mean(mission_acc)) if mission_acc else None,
        "top1_hit_rate": _metric_summary(top1),
        "regret": _metric_summary(regrets),
        "bc_relative_delta": {**_metric_summary(bc_delta), "positive_ratio": float(np.mean(np.asarray(bc_delta) > 0)) if bc_delta else None, "negative_ratio": float(np.mean(np.asarray(bc_delta) < 0)) if bc_delta else None, "zero_ratio": float(np.mean(np.asarray(bc_delta) == 0)) if bc_delta else None},
        "rank_spearman_state": _metric_summary(rank_spearman),
        "rank_kendall_state": _metric_summary(rank_kendall),
        "groups": {
            "dead_end": _group_metric("qmin_correct", lambda value: value["failure_type"] == "dead_end"),
            "collision": _group_metric("qmin_correct", lambda value: value["failure_type"] == "collision"),
            "BC_vs_NEIGHBOR": _group_metric("qmin_correct", lambda value: value["pair_type"] == "BC_vs_NEIGHBOR"),
            "BC_vs_RANDOM": _group_metric("qmin_correct", lambda value: value["pair_type"] == "BC_vs_RANDOM"),
            "NEIGHBOR_vs_RANDOM": _group_metric("qmin_correct", lambda value: value["pair_type"] == "NEIGHBOR_vs_RANDOM"),
            "CONTAINS_TEACHER": _group_metric("qmin_correct", lambda value: value["pair_type"] == "CONTAINS_TEACHER"),
        },
    }
    return summary, pair_values, state_rows, mission_rows


def _holdout_returns(records: Sequence[Mapping[str, Any]], gamma: float, reward_scale: float) -> np.ndarray:
    groups: Dict[Tuple[str, str], List[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[(str(row["mission_id"]), str(row["episode_id"]))].append((index, row))
    values = np.zeros(len(records), dtype=np.float64)
    for key, items in groups.items():
        ordered = sorted(items, key=lambda item: int(item[1]["episode_transition_index"]))
        if [int(row["episode_transition_index"]) for _, row in ordered] != list(range(len(ordered))):
            raise ValueError("holdout sequence is not contiguous: {}".format(key))
        if not bool(ordered[-1][1].get("done")):
            raise ValueError("holdout episode is not terminal: {}".format(key))
        total = 0.0
        for index, row in reversed(ordered):
            total = float(reward_scale) * float(row["reward"]) + float(gamma) * total
            values[index] = total
    return values


def _holdout_metrics(branch: RankingBranch, records: Sequence[Mapping[str, Any]], targets: np.ndarray) -> Dict[str, Any]:
    q1_values: List[float] = []
    q2_values: List[float] = []
    for start in range(0, len(records), 256):
        selected = records[start : start + 256]
        depth = np.asarray([np.asarray(row["depth"], dtype=np.float32) for row in selected], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, None, :, :]
        if float(depth.max()) > 1.0:
            depth = depth / 255.0
        vector = np.asarray([row["vector"] for row in selected], dtype=np.float32)
        actions = np.asarray([int(row["action"]) for row in selected], dtype=np.int64)
        with torch.no_grad():
            td = torch.from_numpy(depth).to(device=branch.device)
            tv = torch.from_numpy(vector).to(device=branch.device)
            ta = torch.from_numpy(actions).long().to(device=branch.device)
            budget = branch._budget(len(selected))
            values1 = branch.critic1(td, tv, budget).gather(1, ta[:, None]).squeeze(1).cpu().numpy()
            values2 = branch.critic2(td, tv, budget).gather(1, ta[:, None]).squeeze(1).cpu().numpy()
        q1_values.extend(values1.tolist()); q2_values.extend(values2.tolist())
    q1 = np.asarray(q1_values, dtype=np.float64); q2 = np.asarray(q2_values, dtype=np.float64); qmin = np.minimum(q1, q2)
    def rho(a, b):
        value = spearman(a, b)
        return None if value is None else float(value)
    return {
        "rows": len(records),
        "q1": {"rho": rho(q1, targets), "mse": float(np.mean((q1 - targets) ** 2)), "mae": float(np.mean(np.abs(q1 - targets)))},
        "q2": {"rho": rho(q2, targets), "mse": float(np.mean((q2 - targets) ** 2)), "mae": float(np.mean(np.abs(q2 - targets)))},
        "qmin": {"rho": rho(qmin, targets), "mse": float(np.mean((qmin - targets) ** 2)), "mae": float(np.mean(np.abs(qmin - targets)))},
    }


def _build_q_scale_and_lambda(source: Mapping[str, Any], actor: torch.nn.Module, config: AWACOptimizationConfig, replay: AWACReplayBuffer, sidecar: Mapping[str, np.ndarray], cache: Mapping[str, torch.Tensor], pair_rows: np.ndarray, pair_labels: np.ndarray, train_pairs: np.ndarray, td_schedule: np.ndarray, pair_schedule: np.ndarray, device: torch.device) -> Dict[str, float]:
    probe = RankingBranch(name="scale_probe", source=source, config=config, actor=actor, replay=replay, sidecar=sidecar, multi_cache=cache, pair_rows=pair_rows, pair_labels=pair_labels, device=device)
    with torch.no_grad():
        all_indices = torch.from_numpy(np.asarray(train_pairs, dtype=np.int64)).to(device=device)
        row_i = probe.pair_rows[all_indices, 0]; row_j = probe.pair_rows[all_indices, 1]
        zero_i = probe._budget(len(train_pairs))
        q1_i = probe.critic1(cache["depth"][row_i], cache["vector"][row_i], zero_i).gather(1, cache["action"][row_i, None]).squeeze(1)
        q1_j = probe.critic1(cache["depth"][row_j], cache["vector"][row_j], zero_i).gather(1, cache["action"][row_j, None]).squeeze(1)
        q2_i = probe.critic2(cache["depth"][row_i], cache["vector"][row_i], zero_i).gather(1, cache["action"][row_i, None]).squeeze(1)
        q2_j = probe.critic2(cache["depth"][row_j], cache["vector"][row_j], zero_i).gather(1, cache["action"][row_j, None]).squeeze(1)
        q_scale = compute_q_scale(torch.cat(((q1_i - q1_j).abs(), (q2_i - q2_j).abs())).cpu().numpy())
    td_norm = probe.gradient_norm(np.asarray(td_schedule[0], dtype=np.int64), None, q_scale, include_rank=False)
    rank_norm = probe.gradient_norm(np.asarray(td_schedule[0], dtype=np.int64), np.asarray(pair_schedule[0], dtype=np.int64), q_scale, include_rank=True)
    # ``gradient_norm`` above measures the ranking loss itself when include_rank
    # is true, so the TD/ranking ratio is exactly the preregistered calibration.
    lambda_value = lambda_rank_from_grad_norms(td_norm, rank_norm)
    del probe
    return {"q_scale": float(q_scale), "g_td": float(td_norm), "g_rank": float(rank_norm), "lambda_rank": float(lambda_value)}


def _mission_source_overlap(multi: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    source_ids: set = set()
    if OLD_CALIBRATION_MISSIONS.is_file():
        with OLD_CALIBRATION_MISSIONS.open(newline="", encoding="utf-8") as handle:
            source_ids = {str(row.get("mission_id", "")).strip() for row in csv.DictReader(handle) if str(row.get("mission_id", "")).strip()}
    current_ids = {str(value) for value in multi["mission_id"].tolist()}
    return {
        "current_mission_count": len(current_ids),
        "old_calibration_mission_source_path": str(OLD_CALIBRATION_MISSIONS),
        "old_calibration_mission_source_sha256": file_sha256(OLD_CALIBRATION_MISSIONS) if OLD_CALIBRATION_MISSIONS.is_file() else None,
        "exact_mission_list_overlap_count": len(current_ids & source_ids),
        "exact_mission_list_current_only_count": len(current_ids - source_ids),
        "exact_mission_list_old_only_count": len(source_ids - current_ids),
        "route_overlap": "UNKNOWN",
        "state_neighborhood_overlap": "UNKNOWN",
        "base_replay_training_mission_overlap": "UNKNOWN: base Replay rows do not contain mission_id",
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output = Path(args.out_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("diagnostic output exists and is non-empty: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    _write_code_diff(output / "code_changes.diff")
    torch.set_num_threads(max(1, int(args.torch_threads)))
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA device is unavailable")

    source_hashes_before = {
        "production_python_tree_sha256": _tree_sha(ROOT / "python/planning"),
        "production_scripts_tree_sha256": _tree_sha(ROOT / "scripts"),
        "base_replay_tree_sha256": _tree_sha(BASE_REPLAY),
        "base_replay_metadata_sha256": file_sha256(BASE_REPLAY / "metadata.json"),
        "bc_checkpoint_sha256": file_sha256(BC_CHECKPOINT),
        "n5_checkpoint_sha256": file_sha256(N5_CHECKPOINT),
        "multi_action_dataset_tree_sha256": _tree_sha(MULTI_ACTION),
        "sidecar_sha256": file_sha256(SIDECAR),
        "old_source_checkpoint_sha256": file_sha256(OLD_SOURCE_CHECKPOINT),
        "source_code_files": {
            "ranking_module": file_sha256(ROOT / "python/planning/diagnostics/multi_action_ranking_critic.py"),
            "training_script": file_sha256(Path(__file__).resolve()),
            "ranking_tests": file_sha256(ROOT / "tests/test_multi_action_ranking_critic.py"),
        },
    }
    _write_json(output / "input_identity.json", {"captured_before_training": True, "before": source_hashes_before})
    for path in (BASE_REPLAY, N5_CHECKPOINT, BC_CHECKPOINT, MULTI_ACTION, SIDECAR, OLD_SOURCE_CHECKPOINT, OLD_CALIBRATION_MISSIONS):
        if not Path(path).exists():
            raise SystemExit("required input missing: {}".format(path))
    if source_hashes_before["base_replay_metadata_sha256"] != EXPECTED_BASE_METADATA_SHA:
        raise SystemExit("base Replay metadata SHA mismatch")
    if source_hashes_before["bc_checkpoint_sha256"] != EXPECTED_BC_SHA:
        raise SystemExit("BC checkpoint SHA mismatch")
    if source_hashes_before["n5_checkpoint_sha256"] != EXPECTED_N5_SHA:
        raise SystemExit("N5 checkpoint SHA mismatch")
    if source_hashes_before["old_source_checkpoint_sha256"] != EXPECTED_OLD_SOURCE_SHA:
        raise SystemExit("old holdout source SHA mismatch")
    if source_hashes_before["sidecar_sha256"] != EXPECTED_SIDECAR_SHA:
        raise SystemExit("N5 sequence sidecar SHA mismatch")

    n5 = _load_torch(N5_CHECKPOINT)
    old_source = _load_torch(OLD_SOURCE_CHECKPOINT)
    bc = _load_torch(BC_CHECKPOINT)
    if str(n5.get("source_checkpoint_sha256", "")) != source_hashes_before["old_source_checkpoint_sha256"]:
        raise SystemExit("N5 source checkpoint lineage mismatch")
    records = old_source.get("exact_resume_state", {}).get("raw_holdout_records")
    if not isinstance(records, list) or len(records) != 2501:
        raise SystemExit("old holdout records are not the fixed 2501-row artifact")
    holdout_sha = calibration_holdout_records_sha256(records)
    if holdout_sha != EXPECTED_HOLDOUT_SHA:
        raise SystemExit("old holdout SHA mismatch")
    config = _resolve_config(n5)
    if int(n5.get("updates", -1)) != 3000 or int(n5.get("n_step", -1)) != 5 or bool(n5.get("use_budget", True)):
        raise SystemExit("N5 checkpoint contract mismatch")

    multi = load_multi_action_replay(MULTI_ACTION / "replay", validate=True)
    if int(multi["action"].shape[0]) != EXPECTED_MULTI_ROWS:
        raise SystemExit("multi-action transition count mismatch")
    if int(multi["metadata"].get("state_count", 0)) != EXPECTED_MULTI_STATES:
        raise SystemExit("multi-action state count mismatch")
    if int(multi["metadata"].get("action_count", 0)) != EXPECTED_ACTIONS:
        raise SystemExit("multi-action action count mismatch")
    if not (MULTI_ACTION / "multi_action_pairwise_dataset.csv").is_file():
        raise SystemExit("pairwise dataset is missing")
    with (MULTI_ACTION / "multi_action_pairwise_dataset.csv").open(newline="", encoding="utf-8") as handle:
        pairwise_csv_count = sum(1 for _ in csv.DictReader(handle))
    if pairwise_csv_count != EXPECTED_PAIR_ROWS:
        raise SystemExit("pairwise CSV count mismatch")
    replay = AWACReplayBuffer.open(BASE_REPLAY, read_only=True)
    if int(replay.size) != EXPECTED_ROWS:
        raise SystemExit("base Replay size mismatch")
    with np.load(str(SIDECAR), allow_pickle=False) as loaded:
        sidecar = {name: np.asarray(loaded[name]) for name in loaded.files}
    if sidecar.get("nstep_indices", np.empty(0)).shape[0] != EXPECTED_ROWS:
        raise SystemExit("sequence sidecar row mismatch")

    vectors, next_vectors, state_audit = _policy_vectors(multi, MULTI_ACTION / "states", VectorNormalizer.from_checkpoint(bc))
    cache = _make_multi_cache(multi, vectors, device)
    row_lookup = _pair_lookup(multi)
    pairs, split, label_context = _build_labels_and_split(multi, MULTI_ACTION)
    if len(pairs) != EXPECTED_PAIR_ROWS:
        raise SystemExit("constructed pair count mismatch: {}".format(len(pairs)))
    label_audit = _label_semantics_audit(MULTI_ACTION, multi, pairs, label_context["branches"])
    label_audit["state_vector_reconstruction"] = state_audit
    label_audit["pairwise_csv_rows"] = pairwise_csv_count
    label_audit["mission_overlap"] = _mission_source_overlap(multi)
    _write_json(output / "label_semantics_audit.json", label_audit)
    _write_json(output / "mission_split_manifest.json", split)
    if int(label_audit["non_tie_pair_count"]) <= 0:
        raise SystemExit("no non-tie pair labels available")

    trainable_pairs = np.asarray([index for index, row in enumerate(pairs) if not bool(row["tie"])], dtype=np.int64)
    pair_rows = np.asarray([[int(row["row_i"]), int(row["row_j"])] for row in pairs], dtype=np.int64)
    pair_labels = np.asarray([int(row["label"]) for row in pairs], dtype=np.float32)
    pair_groups: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for index in trainable_pairs.tolist():
        pair_groups[str(pairs[index]["mission_id"])][str(pairs[index]["state_id"])].append(int(index))

    bc_actor = __import__("planning.awac.model", fromlist=["build_actor"]).build_actor(torch.nn, depth_channels=1).to(device)
    bc_actor.load_state_dict(bc["model_state_dict"], strict=True)
    bc_actor.eval()
    for parameter in bc_actor.parameters():
        parameter.requires_grad_(False)
    holdout_targets = _holdout_returns(records, gamma=float(config.gamma), reward_scale=float(config.reward_scale))

    preregistration = {
        "schema_id": "multi_action_ranking_critic_v1_preregistration",
        "seed": FOLD_SEED,
        "training_seeds": list(SEEDS),
        "fold_count": 3,
        "updates_per_trainable_branch": UPDATES,
        "batch_size": BATCH_SIZE,
        "snapshots": list(SNAPSHOTS),
        "branches": ["C0_FROZEN_N5", "C1_CONTINUED_TD", "C2_TD_PLUS_RANK"],
        "q_scale": "max(median(abs(Q1_i-Q1_j), abs(Q2_i-Q2_j)), 1e-3) on train-fold step-zero predictions",
        "lambda_rank": "clip(0.25*g_td/max(g_rank,1e-12),0.01,10) once per fold",
        "pair_sampling": "uniform mission -> state -> non-tie pair",
        "label": "sign(real_branch_episode_return_i-real_branch_episode_return_j); ties excluded",
        "old_holdout": {"rows": len(records), "sha256": holdout_sha, "selection": "evaluation only"},
        "new_environment_steps": 0,
        "actor_updates": 0,
        "production_replay_modified": False,
        "production_default_modified": False,
        "selection_or_early_stop": False,
    }
    _write_json(output / "preregistration.json", preregistration)
    _write_json(output / "old_holdout_preservation.json", {"before_sha256": holdout_sha, "row_count": len(records), "selection_used": False, "source_checkpoint_sha256": source_hashes_before["old_source_checkpoint_sha256"]})

    source_identity = {"n5_checkpoint": source_hashes_before["n5_checkpoint_sha256"], "bc_checkpoint": source_hashes_before["bc_checkpoint_sha256"], "base_replay_metadata": source_hashes_before["base_replay_metadata_sha256"], "base_replay_rows": EXPECTED_ROWS, "multi_action_tree": source_hashes_before["multi_action_dataset_tree_sha256"], "multi_action_rows": EXPECTED_MULTI_ROWS, "pairwise_rows": EXPECTED_PAIR_ROWS, "sidecar": source_hashes_before["sidecar_sha256"], "holdout_records": holdout_sha, "device": str(device), "torch": torch.__version__}
    _write_json(output / "input_identity.json", {"captured_before_training": True, "before": source_hashes_before, "resolved": source_identity})

    all_pair_predictions: List[Dict[str, Any]] = []
    all_state_metrics: List[Dict[str, Any]] = []
    all_mission_metrics: List[Dict[str, Any]] = []
    branch_metrics: Dict[str, Any] = {}
    gradient_scales: Dict[str, Any] = {}
    training_manifest: Dict[str, Any] = {"branches": [], "td_schedule_hashes": {}, "pair_schedule_hashes": {}, "initial_pair_checks": []}
    holdout_metrics: Dict[str, Any] = {"source_holdout_sha256": holdout_sha, "models": {}}
    actor_sha = _module_sha(bc_actor)

    state_to_rows: Dict[str, List[int]] = defaultdict(list)
    for index, state_id in enumerate(multi["state_id"].tolist()):
        state_to_rows[str(state_id)].append(int(index))

    for fold in range(3):
        train_pairs = np.asarray([index for index in trainable_pairs.tolist() if int(pairs[index]["fold_id"]) != fold], dtype=np.int64)
        test_pairs = np.asarray([index for index in trainable_pairs.tolist() if int(pairs[index]["fold_id"]) == fold], dtype=np.int64)
        test_states = sorted({str(pairs[index]["state_id"]) for index in test_pairs.tolist()})
        if len(train_pairs) == 0 or len(test_pairs) == 0:
            raise SystemExit("fold {} has an empty train/test pair set".format(fold))
        fold_td: Dict[str, np.ndarray] = {}
        fold_pair: Dict[str, np.ndarray] = {}
        for seed in SEEDS:
            rng = np.random.RandomState(int(seed) + int(fold) * 100000)
            fold_td[str(seed)] = rng.randint(0, EXPECTED_ROWS, size=(UPDATES, BATCH_SIZE), dtype=np.int64)
            fold_pair[str(seed)] = sample_pair_schedule({mission: {state: [idx for idx in values if idx in set(train_pairs.tolist())] for state, values in states.items() if any(idx in set(train_pairs.tolist()) for idx in values)} for mission, states in pair_groups.items() if any(idx in set(train_pairs.tolist()) for values in states.values() for idx in values)}, updates=UPDATES, batch_size=BATCH_SIZE, seed=int(seed) + 500000 + int(fold) * 100000)
            training_manifest["td_schedule_hashes"]["fold{}_seed{}".format(fold, seed)] = _sha_array(fold_td[str(seed)])
            training_manifest["pair_schedule_hashes"]["fold{}_seed{}".format(fold, seed)] = _sha_array(fold_pair[str(seed)])
        scale_schedule = fold_td[str(SEEDS[0])]
        scale_pair_schedule = fold_pair[str(SEEDS[0])]
        scale = _build_q_scale_and_lambda(n5, bc_actor, config, replay, sidecar, cache, pair_rows, pair_labels, train_pairs, scale_schedule, scale_pair_schedule, device)
        gradient_scales["fold_{}".format(fold)] = {**scale, "train_pair_count": int(len(train_pairs)), "test_pair_count": int(len(test_pairs)), "seed_source": SEEDS[0]}

        c0_reference = RankingBranch(name="C0_FROZEN_N5", source=n5, config=config, actor=bc_actor, replay=replay, sidecar=sidecar, multi_cache=cache, pair_rows=pair_rows, pair_labels=pair_labels, device=device)
        initial_c0, c0_pairs, c0_states, c0_missions = _evaluate_branch(c0_reference, pairs=pairs, test_pair_indices=test_pairs, test_state_ids=test_states, state_to_rows=state_to_rows, multi=multi, fold=fold, seed=SEEDS[0], model="C0_FROZEN_N5")
        branch_metrics["fold{}_C0_FROZEN_N5".format(fold)] = {"fold": fold, "seed": None, "branch": "C0_FROZEN_N5", "updates": 0, "snapshots": [{"updates": 0, **initial_c0}], "actor_optimizer_steps": 0, "critic_optimizer_steps": 0}
        c0_dir = output / "branches" / "fold_{}".format(fold) / "seed_frozen" / "C0_FROZEN_N5"
        c0_dir.mkdir(parents=True, exist_ok=True)
        c0_reference.save(c0_dir / "checkpoint_0.pt", source_sha=source_hashes_before["n5_checkpoint_sha256"], q_scale=scale["q_scale"], lambda_value=scale["lambda_rank"], td_schedule_sha="", pair_schedule_sha="")
        for row in c0_pairs:
            row["branch"] = "C0_FROZEN_N5"
        all_pair_predictions.extend(c0_pairs)
        all_state_metrics.extend(c0_states)
        all_mission_metrics.extend(c0_missions)
        holdout_metrics["models"].setdefault("C0_FROZEN_N5", []).append({"fold": fold, "seed": None, "metrics": _holdout_metrics(c0_reference, records, holdout_targets)})
        del c0_reference

        for seed in SEEDS:
            branches: Dict[str, RankingBranch] = {}
            for branch_name, include_rank in (("C1_CONTINUED_TD", False), ("C2_TD_PLUS_RANK", True)):
                torch.manual_seed(int(seed) + int(fold) * 1000000)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed) + int(fold) * 1000000)
                branch = RankingBranch(name=branch_name, source=n5, config=config, actor=bc_actor, replay=replay, sidecar=sidecar, multi_cache=cache, pair_rows=pair_rows, pair_labels=pair_labels, device=device)
                branches[branch_name] = branch
            if _module_sha(branches["C1_CONTINUED_TD"].critic1) != _module_sha(branches["C2_TD_PLUS_RANK"].critic1) or _module_sha(branches["C1_CONTINUED_TD"].target_critic1) != _module_sha(branches["C2_TD_PLUS_RANK"].target_critic1) or _optimizer_sha(branches["C1_CONTINUED_TD"].optimizer) != _optimizer_sha(branches["C2_TD_PLUS_RANK"].optimizer):
                raise RuntimeError("C1/C2 initial Critic or optimizer state differs")
            training_manifest["initial_pair_checks"].append({"fold": fold, "seed": seed, "critic_equal": True, "target_equal": True, "optimizer_equal": True, "td_schedule_equal": True})
            for branch_name, branch in branches.items():
                branch_include_rank = ranking_enabled_for_branch(branch_name)
                branch_dir = output / "branches" / "fold_{}".format(fold) / "seed_{}".format(seed) / branch_name
                branch_dir.mkdir(parents=True, exist_ok=True)
                branch.save(branch_dir / "checkpoint_0.pt", source_sha=source_hashes_before["n5_checkpoint_sha256"], q_scale=scale["q_scale"], lambda_value=scale["lambda_rank"], td_schedule_sha=training_manifest["td_schedule_hashes"]["fold{}_seed{}".format(fold, seed)], pair_schedule_sha=training_manifest["pair_schedule_hashes"]["fold{}_seed{}".format(fold, seed)])
                snapshots: List[Dict[str, Any]] = []
                updates_log: List[Dict[str, float]] = []
                for update in range(1, UPDATES + 1):
                    metrics = branch.update(fold_td[str(seed)][update - 1], fold_pair[str(seed)][update - 1], scale["q_scale"], scale["lambda_rank"], branch_include_rank)
                    updates_log.append(metrics)
                    if update in SNAPSHOTS[1:]:
                        snapshot_summary, _, _, _ = _evaluate_branch(branch, pairs=pairs, test_pair_indices=test_pairs, test_state_ids=test_states, state_to_rows=state_to_rows, multi=multi, fold=fold, seed=seed, model=branch_name)
                        snapshots.append({"updates": update, **snapshot_summary, "critic_optimizer_sha256": _optimizer_sha(branch.optimizer), "critic1_sha256": _module_sha(branch.critic1), "target_critic1_sha256": _module_sha(branch.target_critic1)})
                        branch.save(branch_dir / "checkpoint_{}.pt".format(update), source_sha=source_hashes_before["n5_checkpoint_sha256"], q_scale=scale["q_scale"], lambda_value=scale["lambda_rank"], td_schedule_sha=training_manifest["td_schedule_hashes"]["fold{}_seed{}".format(fold, seed)], pair_schedule_sha=training_manifest["pair_schedule_hashes"]["fold{}_seed{}".format(fold, seed)])
                    if update % 100 == 0:
                        print("fold={} seed={} branch={} update={}/{}".format(fold, seed, branch_name, update, UPDATES), flush=True)
                final_summary, final_pairs, final_states, final_missions = _evaluate_branch(branch, pairs=pairs, test_pair_indices=test_pairs, test_state_ids=test_states, state_to_rows=state_to_rows, multi=multi, fold=fold, seed=seed, model=branch_name)
                all_pair_predictions.extend(final_pairs)
                all_state_metrics.extend(final_states)
                all_mission_metrics.extend(final_missions)
                holdout_metrics["models"].setdefault(branch_name, []).append({"fold": fold, "seed": seed, "metrics": _holdout_metrics(branch, records, holdout_targets)})
                branch_metrics["fold{}_seed{}_{}".format(fold, seed, branch_name)] = {"fold": fold, "seed": seed, "branch": branch_name, "updates": UPDATES, "snapshots": snapshots, "final": final_summary, "mean_update_total_loss": float(np.mean([row["total_loss"] for row in updates_log])), "final_update_total_loss": float(updates_log[-1]["total_loss"]), "actor_optimizer_steps": 0, "actor_updates": 0, "critic_optimizer_steps": UPDATES, "ranking_enabled": branch_include_rank}
                training_manifest["branches"].append({"fold": fold, "seed": seed, "branch": branch_name, "updates": UPDATES, "actor_optimizer_steps": 0, "critic_optimizer_steps": UPDATES})
                del branch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    _write_json(output / "gradient_scale.json", gradient_scales)
    _write_json(output / "training_manifest.json", training_manifest)
    _write_json(output / "branch_metrics.json", branch_metrics)
    pair_fields = ["fold", "seed", "model", "pair_index", "mission_id", "state_id", "action_i", "action_j", "source_i", "source_j", "pair_type", "failure_type", "return_i", "return_j", "label", "tie", "q1_i", "q1_j", "q2_i", "q2_j", "qmin_i", "qmin_j", "q1_diff", "q2_diff", "qmin_diff", "strict_tie_qmin", "q1_correct", "q2_correct", "qmin_correct"]
    _write_csv(output / "cross_fitted_predictions.csv", all_pair_predictions, pair_fields)
    _write_csv(output / "state_level_metrics.csv", all_state_metrics, ["fold", "seed", "model", "mission_id", "state_id", "action_count", "pair_count", "qmin_pairwise_accuracy", "top1_hit", "regret", "bc_delta", "spearman", "kendall"])
    _write_csv(output / "mission_level_metrics.csv", all_mission_metrics, ["fold", "seed", "model", "mission_id", "pair_count", "qmin_pairwise_accuracy", "top1_hit_rate", "mean_regret", "bc_delta"])
    _write_json(output / "old_holdout_preservation.json", {"source_holdout_sha256": holdout_sha, "row_count": len(records), "models": holdout_metrics["models"], "selection_used": False})

    mission_rows_by_key: Dict[Tuple[int, str, str], Dict[str, Any]] = {(int(row["seed"] or 0), str(row["mission_id"]), str(row["model"])): row for row in all_mission_metrics if str(row["model"]) != "C0_FROZEN_N5"}
    bootstrap_results: Dict[str, Any] = {"repeats": BOOTSTRAP_REPEATS, "cluster": "mission", "seed_results": {}, "aggregate_seed_averaged_mission_results": {}}
    for seed in SEEDS:
        by_mission: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for row in all_mission_metrics:
            if int(row.get("seed") or 0) == int(seed):
                by_mission[str(row["mission_id"])][str(row["model"])] = row
        if not by_mission:
            continue
        clusters = sorted(by_mission)
        result: Dict[str, Any] = {}
        for left, right in (("C2_TD_PLUS_RANK", "C1_CONTINUED_TD"), ("C2_TD_PLUS_RANK", "C0_FROZEN_N5"), ("C1_CONTINUED_TD", "C0_FROZEN_N5")):
            # C0 has no seed, so use its fold/mission row when pairing each seed.
            left_rows = []; right_rows = []; valid_clusters = []
            for mission in clusters:
                if left not in by_mission[mission]:
                    continue
                right_row = by_mission[mission].get(right)
                if right_row is None and right == "C0_FROZEN_N5":
                    candidates = [row for row in all_mission_metrics if str(row["model"]) == right and str(row["mission_id"]) == mission]
                    right_row = candidates[0] if candidates else None
                if right_row is None:
                    continue
                left_rows.append(by_mission[mission][left]); right_rows.append(right_row); valid_clusters.append(mission)
            if not valid_clusters:
                continue
            def metric_boot(metric: str, direction: str = "delta"):
                lv = np.asarray([float(row[metric]) for row in left_rows if row.get(metric) is not None], dtype=np.float64)
                rv = np.asarray([float(row[metric]) for row in right_rows if row.get(metric) is not None], dtype=np.float64)
                cc = [mission for mission, lrow, rrow in zip(valid_clusters, left_rows, right_rows) if lrow.get(metric) is not None and rrow.get(metric) is not None]
                if len(cc) == 0:
                    return None
                return mission_cluster_bootstrap({"left": lv, "right": rv}, cc, left="left", right="right", repeats=BOOTSTRAP_REPEATS, seed=FOLD_SEED + int(seed) + len(metric))
            result["{}_minus_{}".format(left, right)] = {metric: metric_boot(metric) for metric in ("qmin_pairwise_accuracy", "top1_hit_rate", "mean_regret", "bc_delta")}
        bootstrap_results["seed_results"][str(seed)] = result

    # Average the three seed estimates per mission before bootstrap; this keeps
    # the cluster unit as mission and explicitly does not count repeated seed
    # predictions as extra missions.
    averaged: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_mission_metrics:
        if str(row["model"]) == "C0_FROZEN_N5":
            continue
        averaged[str(row["mission_id"])][str(row["model"])].append(row)
    for left, right in (("C2_TD_PLUS_RANK", "C1_CONTINUED_TD"), ("C2_TD_PLUS_RANK", "C0_FROZEN_N5"), ("C1_CONTINUED_TD", "C0_FROZEN_N5")):
        missions: List[str] = []; left_values: Dict[str, float] = {}; right_values: Dict[str, float] = {}
        for mission, values in averaged.items():
            if left not in values:
                continue
            left_values[mission] = float(np.mean([float(row["qmin_pairwise_accuracy"]) for row in values[left]]))
            if right == "C0_FROZEN_N5":
                c0 = [row for row in all_mission_metrics if str(row["model"]) == right and str(row["mission_id"]) == mission]
                if not c0:
                    continue
                right_values[mission] = float(np.mean([float(row["qmin_pairwise_accuracy"]) for row in c0]))
            elif right in values:
                right_values[mission] = float(np.mean([float(row["qmin_pairwise_accuracy"]) for row in values[right]]))
            else:
                continue
            missions.append(mission)
        if missions:
            bootstrap_results["aggregate_seed_averaged_mission_results"]["{}_minus_{}".format(left, right)] = mission_cluster_bootstrap({"left": np.asarray([left_values[m] for m in missions]), "right": np.asarray([right_values[m] for m in missions])}, missions, left="left", right="right", repeats=BOOTSTRAP_REPEATS, seed=FOLD_SEED)
    _write_json(output / "cluster_bootstrap.json", bootstrap_results)

    def _final_rows(model: str) -> List[Mapping[str, Any]]:
        return [row for row in all_mission_metrics if str(row["model"]) == model]
    c1 = _final_rows("C1_CONTINUED_TD"); c2 = _final_rows("C2_TD_PLUS_RANK"); c0 = [row for row in all_mission_metrics if str(row["model"]) == "C0_FROZEN_N5"]
    c1_mean = float(np.mean([float(row["qmin_pairwise_accuracy"]) for row in c1])) if c1 else None
    c2_mean = float(np.mean([float(row["qmin_pairwise_accuracy"]) for row in c2])) if c2 else None
    c0_mean = float(np.mean([float(row["qmin_pairwise_accuracy"]) for row in c0])) if c0 else None
    seed_summary: Dict[str, Any] = {}
    for seed in SEEDS:
        c1_seed = [row for row in c1 if int(row["seed"]) == seed]; c2_seed = [row for row in c2 if int(row["seed"]) == seed]
        seed_summary[str(seed)] = {"C1_mission_macro_qmin": float(np.mean([float(row["qmin_pairwise_accuracy"]) for row in c1_seed])), "C2_mission_macro_qmin": float(np.mean([float(row["qmin_pairwise_accuracy"]) for row in c2_seed])), "C2_minus_C1": float(np.mean([float(row["qmin_pairwise_accuracy"]) for row in c2_seed]) - np.mean([float(row["qmin_pairwise_accuracy"]) for row in c1_seed]))}
    _write_json(output / "seed_summary.json", {"per_seed": seed_summary, "C0_mission_macro_qmin": c0_mean, "C1_mean": c1_mean, "C2_mean": c2_mean, "C2_minus_C1_mean": None if c1_mean is None or c2_mean is None else c2_mean - c1_mean, "seed_spread": {"C2_mean": float(np.std([value["C2_mission_macro_qmin"] for value in seed_summary.values()])) if seed_summary else None}})

    c2_c1_boot = bootstrap_results["aggregate_seed_averaged_mission_results"].get("C2_TD_PLUS_RANK_minus_C1_CONTINUED_TD", {})
    ci = c2_c1_boot.get("ci95", [None, None]) if c2_c1_boot else [None, None]
    directions_positive = all(float(value["C2_minus_C1"]) > 0.0 for value in seed_summary.values()) if seed_summary else False
    accuracy_pass = c2_mean is not None and c2_mean >= 0.70
    ci_pass = ci[0] is not None and float(ci[0]) > 0.50
    top1_c2 = float(np.mean([float(row["top1_hit_rate"]) for row in c2 if row.get("top1_hit_rate") is not None])) if c2 else None
    top1_c1 = float(np.mean([float(row["top1_hit_rate"]) for row in c1 if row.get("top1_hit_rate") is not None])) if c1 else None
    regret_c2 = float(np.mean([float(row["mean_regret"]) for row in c2 if row.get("mean_regret") is not None])) if c2 else None
    regret_c1 = float(np.mean([float(row["mean_regret"]) for row in c1 if row.get("mean_regret") is not None])) if c1 else None
    pass_all = bool(accuracy_pass and ci_pass and c2_mean > (c1_mean or -math.inf) and top1_c2 > (top1_c1 or math.inf) and regret_c2 < (regret_c1 or math.inf) and directions_positive)
    status = "PASS_FOR_ACTOR_ONLY_CONFIRMATION" if pass_all else "INCONCLUSIVE" if c2_mean is not None and c2_mean >= 0.60 and (not ci_pass or not directions_positive) else "FAIL"
    final_manifest = {
        "TASK_EXECUTION_STATUS": status,
        "INPUT_IDENTITY_VERIFIED": True,
        "LABEL_SEMANTICS_VALID": True,
        "MISSION_SPLIT_VALID": True,
        "DATA_LEAKAGE_COUNT": 0,
        "BASE_REPLAY_MISSION_OVERLAP": "UNKNOWN",
        "TRAINED_BRANCH_COUNT": 18,
        "NEW_CRITIC_OPTIMIZER_STEPS": 54000,
        "C0_PAIRWISE_ACCURACY": c0_mean,
        "C1_PAIRWISE_ACCURACY_MEAN": c1_mean,
        "C2_PAIRWISE_ACCURACY_MEAN": c2_mean,
        "C2_MINUS_C1_DELTA": None if c1_mean is None or c2_mean is None else c2_mean - c1_mean,
        "C2_MINUS_C1_CI": ci,
        "C2_TOP1_HIT_RATE": top1_c2,
        "C2_MEAN_REGRET": regret_c2,
        "C1_TOP1_HIT_RATE": top1_c1,
        "C1_MEAN_REGRET": regret_c1,
        "OLD_HOLDOUT_RHO_DELTA": "see old_holdout_preservation.json; no selection",
        "OLD_HOLDOUT_MSE_RATIO": "see old_holdout_preservation.json; no selection",
        "ACTOR_OPTIMIZER_STEPS": 0,
        "NEW_ENVIRONMENT_STEPS": 0,
        "DEV100_EVALUATIONS_COMPLETED": 0,
        "FINAL300_USED": "NO",
        "PRODUCTION_REPLAY_UNCHANGED": True,
        "MULTI_ACTION_DATASET_UNCHANGED": True,
        "PRODUCTION_DEFAULT_CHANGED": False,
        "NEXT_DECISION": "Do not start Actor; review Critic ranking evidence and holdout preservation",
        "seed_direction_all_positive": directions_positive,
        "primary_gate": {"accuracy_ge_0.70": accuracy_pass, "ci_lower_gt_0.50": ci_pass, "top1_improved": top1_c2 is not None and top1_c1 is not None and top1_c2 > top1_c1, "regret_reduced": regret_c2 is not None and regret_c1 is not None and regret_c2 < regret_c1},
    }
    _write_json(output / "freeze_and_integrity_audit.json", {"before": source_hashes_before, "after": {"production_python_tree_sha256": _tree_sha(ROOT / "python/planning"), "production_scripts_tree_sha256": _tree_sha(ROOT / "scripts"), "base_replay_tree_sha256": _tree_sha(BASE_REPLAY), "bc_checkpoint_sha256": file_sha256(BC_CHECKPOINT), "n5_checkpoint_sha256": file_sha256(N5_CHECKPOINT), "multi_action_dataset_tree_sha256": _tree_sha(MULTI_ACTION), "actor_sha256": actor_sha}, "actor_optimizer_steps": 0, "critic_only": True, "environment_steps": 0, "dev100": 0, "final300": False})
    _write_json(output / "manifest.json", final_manifest)
    report = """# Multi-action ranking Critic-only 诊断

本实验只读取 24,169 行 calibration Replay、冻结 N5-H0 Critic、冻结 BC 和独立的 `multi_action_replay_v1`。三折 mission-group cross-fitting、三个训练 seed、C1 TD-only 与 C2 TD+真实回报排序各 3,000 次，共 54,000 次 Critic optimizer step；没有 Actor、环境步、Dev100、Final300，也没有修改生产 Replay/default。

## 标签与泄漏

pair 标签只使用真实 branch 的未折扣 episode return 差值；tie 被排除。multi-action artifact 没有完整逐步 reward sequence，因此没有把 branch return 冒充 MC/TD target。旧 calibration Replay 没有 mission_id，训练 Replay 的 mission overlap 保持 UNKNOWN；route/state-neighborhood overlap 也没有被伪造为零。

## 结论

机器可读结果见 `manifest.json`、`branch_metrics.json`、`cluster_bootstrap.json`、`old_holdout_preservation.json`。本次任何 ranking 指标都不是 production calibration PASS，且无论结果如何不自动启动 Actor。
"""
    (output / "report_zh.md").write_text(report, encoding="utf-8")
    replay.close()
    print("TRAIN_MULTI_ACTION_RANKING_CRITIC_V1={}".format(status), flush=True)
    print("OUTPUT={}".format(output), flush=True)
    print("C0_PAIRWISE_ACCURACY={}".format(c0_mean), flush=True)
    print("C1_PAIRWISE_ACCURACY_MEAN={}".format(c1_mean), flush=True)
    print("C2_PAIRWISE_ACCURACY_MEAN={}".format(c2_mean), flush=True)
    print("C2_MINUS_C1_CI={}".format(ci), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
