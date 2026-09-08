#!/usr/bin/env python
"""Run the offline fixed-Critic TD1 versus N5 Actor-only comparison.

This diagnostic never starts Unity/ROS, never writes the production config or
Replay, and never runs a Dev100 evaluation.  The two branches differ only in
the frozen H0 diagnostic Critic loaded from their requested 3000-update
endpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from planning.awac.critic_actor_only import adapt_h0_diagnostic_state_dict
from planning.awac.learner import AWACOptimizationConfig, awac_advantage_weights
from planning.awac.model import build_critic, masked_policy
from planning.awac.replay import AWACReplayBuffer
from planning.awac.td_horizon_diagnostic import BudgetCritic
from planning.awac.trainer import build_learner, validate_bc_checkpoint_for_awac
from planning.common.hashing import file_sha256
from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM


EXPECTED_TD1_SHA = "0ddb2d61af4f4a49ad85f2e65d0f2242c1853bd320b1808ed3736099c0af19c9"
EXPECTED_N5_SHA = "f91dbf20e9a235e9c79531bd6848923b9971fc6bb29d1631cf7268fb40dab3d0"
EXPECTED_BC_SHA = "ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2"
EXPECTED_REPLAY_SIZE = 24169
EXPECTED_SOURCE_CHECKPOINT_SHA = "05791a5ffe5d1736f1b8c10dcca3e605772918aaf370273b4f2240f9d19dbcaa"
SNAPSHOT_STEPS = (0, 25, 50, 100)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _hash_value(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda entry: str(entry)):
                digest.update(str(key).encode("utf-8"))
                digest.update(b"\0")
                visit(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            digest.update(str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
            return
        digest.update(type(item).__name__.encode("ascii"))
        digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def _state_sha(module_or_state: Any) -> str:
    state = (
        module_or_state.state_dict()
        if hasattr(module_or_state, "state_dict")
        else module_or_state
    )
    return _hash_value(state)


def _copy_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _copy_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_cpu(item) for item in value)
    return copy.deepcopy(value)


def _capture_rng() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].clone())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([state.clone() for state in state["torch_cuda"]])


def _rng_identity(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "python_sha256": _hash_value(state["python"]),
        "numpy_sha256": _hash_value(state["numpy"]),
        "torch_cpu_sha256": _hash_value(state["torch_cpu"]),
        "torch_cuda_sha256": _hash_value(state.get("torch_cuda", [])),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _fresh_output(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.exists() or not any(root.iterdir()):
        root.mkdir(parents=True, exist_ok=True)
        return root
    suffix = 2
    while True:
        candidate = Path("{}_r{}".format(root, suffix))
        if not candidate.exists() or not any(candidate.iterdir()):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        suffix += 1


def _hash_tree(path: Path) -> Dict[str, str]:
    root = path.expanduser().resolve()
    result = {}
    for item in sorted(item for item in root.rglob("*") if item.is_file()):
        result[str(item.relative_to(root))] = file_sha256(item)
    return result


def _save_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp".format(path.name))
    try:
        torch.save(dict(payload), str(temporary))
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_payload(path: Path, *, expected_sha: str, expected_branch: str) -> Dict[str, Any]:
    actual = file_sha256(path)
    if actual != expected_sha:
        raise RuntimeError(
            "{} checkpoint SHA mismatch: expected {}, got {}".format(
                expected_branch, expected_sha, actual
            )
        )
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("{} checkpoint is not a mapping".format(expected_branch))
    payload = dict(payload)
    if str(payload.get("branch", "")) != expected_branch:
        raise ValueError("{} checkpoint branch identity mismatch".format(expected_branch))
    if int(payload.get("updates", -1)) != 3000:
        raise ValueError("{} endpoint is not the fixed 3000-update checkpoint".format(expected_branch))
    if int(payload.get("critic_vector_dim", -1)) != int(POLICY_VECTOR_DIM + 1):
        raise ValueError("{} checkpoint is not the 128-dimensional diagnostic Critic".format(expected_branch))
    if bool(payload.get("use_budget", True)):
        raise ValueError("{} must be H0, not budget-input H1".format(expected_branch))
    if str(payload.get("source_checkpoint_sha256", "")) != EXPECTED_SOURCE_CHECKPOINT_SHA:
        raise ValueError("{} does not use the recorded common factorial initialization".format(expected_branch))
    config = payload.get("resolved_config")
    if not isinstance(config, Mapping):
        raise ValueError("{} resolved config is missing".format(expected_branch))
    if float(config.get("critic_cql_weight", float("nan"))) != 0.0:
        raise ValueError("{} must use CQL=0".format(expected_branch))
    for flag in (
        "enable_twin_q_confidence",
        "enable_adaptive_bc_kl",
        "enable_primitive_neighbor_exploration",
    ):
        if bool(config.get(flag, True)):
            raise ValueError("{} has an enabled innovation module: {}".format(expected_branch, flag))
    for name in (
        "critic1_state_dict",
        "critic2_state_dict",
        "target_critic1_state_dict",
        "target_critic2_state_dict",
    ):
        if not isinstance(payload.get(name), Mapping):
            raise ValueError("{} lacks {}".format(expected_branch, name))
    return payload


def _load_bc(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    actual = file_sha256(path)
    if actual != EXPECTED_BC_SHA:
        raise RuntimeError("BC checkpoint SHA mismatch: expected {}, got {}".format(EXPECTED_BC_SHA, actual))
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("BC checkpoint is not a mapping")
    payload = dict(payload)
    fields = validate_bc_checkpoint_for_awac(
        payload, mpl_contract_sha256=str(payload.get("mpl_contract_sha256", ""))
    )
    if int(payload.get("vec_dim", -1)) != int(POLICY_VECTOR_DIM):
        raise ValueError("BC vector dimension mismatch")
    if int(payload.get("num_actions", -1)) != int(NUM_ACTIONS):
        raise ValueError("BC action dimension mismatch")
    return payload, fields


def _config_from_endpoint(payload: Mapping[str, Any]) -> AWACOptimizationConfig:
    raw = payload["resolved_config"]
    allowed = set(AWACOptimizationConfig.__dataclass_fields__)
    values = {name: raw[name] for name in raw if name in allowed}
    config = AWACOptimizationConfig(**values)
    if float(config.critic_cql_weight) != 0.0:
        raise ValueError("Actor-only diagnostic requires CQL=0")
    if bool(config.enable_twin_q_confidence) or bool(config.enable_adaptive_bc_kl) or bool(config.enable_primitive_neighbor_exploration):
        raise ValueError("Actor-only diagnostic requires confidence, adaptive KL and exploration disabled")
    return config


def _fixed_state(learner) -> Dict[str, Any]:
    return {
        "critic1": _state_sha(learner.critic1),
        "critic2": _state_sha(learner.critic2),
        "target_critic1": _state_sha(learner.target_critic1),
        "target_critic2": _state_sha(learner.target_critic2),
        "bc_reference": _state_sha(learner.bc_reference),
        "critic_optimizer": _state_sha(learner.critic_optimizer),
        "critic_update_count": int(learner.critic_update_count),
        "update_step": int(learner.update_step),
    }


def _assert_fixed(before: Mapping[str, Any], learner, branch: str, proposal: int) -> None:
    after = _fixed_state(learner)
    if dict(before) != after:
        raise RuntimeError(
            "{} proposal {} changed fixed Critic state: {}".format(
                branch,
                int(proposal),
                json.dumps(_jsonable(after), sort_keys=True),
            )
        )


def _branch_initial_state(learner) -> Dict[str, Any]:
    return {
        "actor": _state_sha(learner.actor),
        "actor_optimizer": _state_sha(learner.actor_optimizer),
        "bc_reference": _state_sha(learner.bc_reference),
        "fixed": _fixed_state(learner),
        "optimizer_lr": learner.optimizer_lr_inventory(),
    }


def _build_branch(
    *,
    branch: str,
    endpoint: Mapping[str, Any],
    bc: Mapping[str, Any],
    config: AWACOptimizationConfig,
    device,
    seed: int,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    _seed_all(seed)
    learner = build_learner(
        torch=torch,
        nn=torch.nn,
        device=device,
        bc_checkpoint=bc,
        config=config,
    )
    critic_states = {}
    adapter_reports = {}
    for name in ("critic1", "critic2", "target_critic1", "target_critic2"):
        adapted, report = adapt_h0_diagnostic_state_dict(
            endpoint["{}_state_dict".format(name)],
            torch=torch,
            policy_vector_dim=int(POLICY_VECTOR_DIM),
        )
        critic_states[name] = adapted
        adapter_reports[name] = report
    learner.critic1.load_state_dict(critic_states["critic1"], strict=True)
    learner.critic2.load_state_dict(critic_states["critic2"], strict=True)
    learner.target_critic1.load_state_dict(critic_states["target_critic1"], strict=True)
    learner.target_critic2.load_state_dict(critic_states["target_critic2"], strict=True)
    learner.actor_optimizer.zero_grad(set_to_none=True)
    build_rng = _capture_rng()
    initial = _branch_initial_state(learner)
    return learner, {"adapter_reports": adapter_reports, "build_rng": build_rng}, initial


def _make_sequences(size: int, *, proposals: int, batch_size: int, seed: int) -> Dict[str, np.ndarray]:
    if int(size) < int(batch_size):
        raise ValueError("Replay is smaller than the requested batch")
    rng = np.random.RandomState(int(seed))
    result = {
        "train_indices": rng.randint(0, int(size), size=(int(proposals), int(batch_size))).astype(np.int64),
        "trust_indices": rng.randint(0, int(size), size=(int(proposals), int(batch_size))).astype(np.int64),
        "validation_indices": rng.choice(int(size), size=min(512, int(size)), replace=False).astype(np.int64),
    }
    return result


def _sequence_sha(sequences: Mapping[str, np.ndarray]) -> str:
    return _hash_value({name: sequences[name] for name in sorted(sequences)})


def _trust_batch(batch: Mapping[str, Any]) -> Dict[str, Any]:
    return {name: batch[name] for name in ("depth", "vector", "action_mask")}


def _policy_shift(learner, batch: Mapping[str, Any]) -> Dict[str, float]:
    with torch.no_grad():
        actor_logits = learner.actor(batch["depth"], batch["vector"])
        bc_logits = learner.bc_reference(batch["depth"], batch["vector"])
        actor_prob, actor_log, _ = masked_policy(actor_logits, batch["action_mask"], torch)
        bc_prob, bc_log, _ = masked_policy(bc_logits, batch["action_mask"], torch)
        kl = (bc_prob * (bc_log - actor_log)).sum(dim=1)
        tv = 0.5 * (bc_prob - actor_prob).abs().sum(dim=1)
        flips = actor_prob.argmax(dim=1) != bc_prob.argmax(dim=1)
    return {
        "bc_to_actor_kl_mean": float(kl.mean().cpu().item()),
        "bc_to_actor_kl_p95": float(torch.quantile(kl, torch.tensor(0.95, device=kl.device)).cpu().item()),
        "bc_to_actor_kl_max": float(kl.max().cpu().item()),
        "total_variation_mean": float(tv.mean().cpu().item()),
        "total_variation_p95": float(torch.quantile(tv, torch.tensor(0.95, device=tv.device)).cpu().item()),
        "argmax_flip_rate": float(flips.float().mean().cpu().item()),
        "argmax_flip_count": float(flips.sum().cpu().item()),
        "sample_count": float(kl.numel()),
    }


def _distribution_summary(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach()
    return {
        "mean": float(values.mean().cpu().item()),
        "p05": float(torch.quantile(values, torch.tensor(0.05, device=values.device)).cpu().item()),
        "p50": float(torch.quantile(values, torch.tensor(0.50, device=values.device)).cpu().item()),
        "p95": float(torch.quantile(values, torch.tensor(0.95, device=values.device)).cpu().item()),
        "min": float(values.min().cpu().item()),
        "max": float(values.max().cpu().item()),
    }


def _offline_arrays(learner, batch: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        actor_logits = learner.actor(batch["depth"], batch["vector"])
        bc_logits = learner.bc_reference(batch["depth"], batch["vector"])
        actor_prob, _, _ = masked_policy(actor_logits, batch["action_mask"], torch)
        bc_prob, _, _ = masked_policy(bc_logits, batch["action_mask"], torch)
        q1 = learner.critic1(batch["depth"], batch["vector"])
        q2 = learner.critic2(batch["depth"], batch["vector"])
        qmin = torch.minimum(q1, q2)
        data_q = qmin.gather(1, batch["action"][:, None]).squeeze(1)
        state_value = (actor_prob * qmin).sum(dim=1)
        advantage = data_q - state_value
        weights = awac_advantage_weights(
            advantage,
            temperature=float(learner.config.awac_temperature),
            weight_max=float(learner.config.awac_weight_max),
            torch=torch,
        )
    return {
        "q1": q1.detach().cpu().numpy(),
        "q2": q2.detach().cpu().numpy(),
        "qmin": qmin.detach().cpu().numpy(),
        "data_q": data_q.detach().cpu().numpy(),
        "state_value": state_value.detach().cpu().numpy(),
        "advantage": advantage.detach().cpu().numpy(),
        "awac_raw_weight": weights["raw"].detach().cpu().numpy(),
        "awac_weight": weights["normalized"].detach().cpu().numpy(),
        "actor_probabilities": actor_prob.detach().cpu().numpy(),
        "bc_probabilities": bc_prob.detach().cpu().numpy(),
        "action": batch["action"].detach().cpu().numpy(),
    }


def _preflight(
    *,
    learner,
    replay: AWACReplayBuffer,
    validation_indices: np.ndarray,
    trust_indices: np.ndarray,
    fixed_before: Mapping[str, Any],
    branch: str,
    proposal: int,
    output: Path,
) -> Dict[str, Any]:
    batch = replay.sample_indices(validation_indices, torch=torch, device=learner.device)
    trust = _trust_batch(replay.sample_indices(trust_indices, torch=torch, device=learner.device))
    metrics = learner.actor_only_update(
        batch,
        actor_trust_batch=trust,
        weight_mode="awac",
        update_actor=False,
    )
    _assert_fixed(fixed_before, learner, branch, proposal)
    learner.actor_optimizer.zero_grad(set_to_none=True)
    arrays = _offline_arrays(learner, batch)
    array_path = output / "preflight" / "{}_proposal_{:03d}.npz".format(branch, int(proposal))
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(array_path), **arrays)
    policy = _policy_shift(learner, batch)
    return {
        "branch": branch,
        "proposal": int(proposal),
        "validation_indices_sha256": _hash_value(validation_indices),
        "trust_indices_sha256": _hash_value(trust_indices),
        "metrics": _jsonable(metrics),
        "policy_shift": policy,
        "advantage_distribution": _distribution_summary(torch.from_numpy(arrays["advantage"])),
        "awac_weight_distribution": _distribution_summary(torch.from_numpy(arrays["awac_weight"])),
        "raw_awac_weight_distribution": _distribution_summary(torch.from_numpy(arrays["awac_raw_weight"])),
        "array_path": str(array_path.resolve()),
        "array_sha256": file_sha256(array_path),
        "actor_and_optimizer_unchanged": True,
    }


def _snapshot(
    *,
    output: Path,
    branch: str,
    proposal: int,
    learner,
    critic_sha: str,
    bc_sha: str,
    last_metrics: Mapping[str, Any],
    validation: Mapping[str, Any],
    fixed_before: Mapping[str, Any],
) -> Dict[str, Any]:
    checkpoint_path = output / "checkpoints" / "{}_proposal_{:03d}.pt".format(branch, int(proposal))
    payload = {
        "schema_id": "critic_actor_only_snapshot",
        "evaluation_only": True,
        "resume_allowed": False,
        "branch": branch,
        "proposal": int(proposal),
        "critic_checkpoint_sha256": critic_sha,
        "bc_checkpoint_sha256": bc_sha,
        "actor_state_dict": _copy_cpu(learner.actor.state_dict()),
        "actor_optimizer_state_dict": _copy_cpu(learner.actor_optimizer.state_dict()),
        "actor_state_sha256": _state_sha(learner.actor),
        "actor_optimizer_sha256": _state_sha(learner.actor_optimizer),
        "actor_update_count": int(learner.actor_update_count),
        "actor_awac_update_count": int(learner.actor_awac_update_count),
        "actor_recovery_update_count": int(learner.actor_recovery_update_count),
        "actor_rejection_count": int(learner.actor_trust_region_rejection_count),
        "actor_optimizer_step_count": int(learner.actor_optimizer_step_count),
        "critic_update_count": int(learner.critic_update_count),
        "fixed_critic_state": dict(fixed_before),
        "last_update_metrics": _jsonable(last_metrics),
        "validation_diagnostics": _jsonable(validation),
    }
    _save_torch(checkpoint_path, payload)
    summary = {
        "proposal": int(proposal),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "actor_state_sha256": payload["actor_state_sha256"],
        "actor_optimizer_sha256": payload["actor_optimizer_sha256"],
        "actor_kl": validation["policy_shift"]["bc_to_actor_kl_mean"],
        "actor_kl_max": validation["policy_shift"]["bc_to_actor_kl_max"],
        "argmax_flip_rate": validation["policy_shift"]["argmax_flip_rate"],
        "accepted_update_count": int(learner.actor_update_count),
        "awac_accepted_update_count": int(learner.actor_awac_update_count),
        "recovery_update_count": int(learner.actor_recovery_update_count),
        "rejected_update_count": int(learner.actor_trust_region_rejection_count),
        "critic_update_count": int(learner.critic_update_count),
        "awac_metrics": {
            key: value
            for key, value in _jsonable(last_metrics).items()
            if "weight" in key or "ess" in key or "advantage" in key or key in ("actor_loss", "awac_actor_loss", "bc_kl")
        },
        "policy_shift": _jsonable(validation["policy_shift"]),
    }
    _write_json(output / "snapshots" / "{}_proposal_{:03d}.json".format(branch, int(proposal)), summary)
    return summary


def _run_branch(
    *,
    output: Path,
    branch: str,
    critic_sha: str,
    learner,
    build_rng: Mapping[str, Any],
    replay: AWACReplayBuffer,
    sequences: Mapping[str, np.ndarray],
    bc_sha: str,
) -> Dict[str, Any]:
    fixed_before = _fixed_state(learner)
    _restore_rng(build_rng)
    rng_before_updates = _capture_rng()
    preflight = _preflight(
        learner=learner,
        replay=replay,
        validation_indices=sequences["validation_indices"],
        trust_indices=sequences["trust_indices"][0],
        fixed_before=fixed_before,
        branch=branch,
        proposal=0,
        output=output,
    )
    snapshots = {}
    snapshots["0"] = _snapshot(
        output=output,
        branch=branch,
        proposal=0,
        learner=learner,
        critic_sha=critic_sha,
        bc_sha=bc_sha,
        last_metrics=preflight["metrics"],
        validation=preflight,
        fixed_before=fixed_before,
    )
    rows = []
    last_metrics = preflight["metrics"]
    for proposal in range(1, 101):
        batch = replay.sample_indices(
            sequences["train_indices"][proposal - 1],
            torch=torch,
            device=learner.device,
        )
        trust = _trust_batch(
            replay.sample_indices(
                sequences["trust_indices"][proposal - 1],
                torch=torch,
                device=learner.device,
            )
        )
        metrics = learner.actor_only_update(
            batch,
            actor_trust_batch=trust,
            weight_mode="awac",
            update_actor=True,
        )
        _assert_fixed(fixed_before, learner, branch, proposal)
        last_metrics = _jsonable(metrics)
        row = {"proposal": int(proposal), "branch": branch}
        row.update(last_metrics)
        rows.append(row)
        if proposal in SNAPSHOT_STEPS:
            validation = _preflight(
                learner=learner,
                replay=replay,
                validation_indices=sequences["validation_indices"],
                trust_indices=sequences["trust_indices"][0],
                fixed_before=fixed_before,
                branch=branch,
                proposal=proposal,
                output=output,
            )
            snapshots[str(proposal)] = _snapshot(
                output=output,
                branch=branch,
                proposal=proposal,
                learner=learner,
                critic_sha=critic_sha,
                bc_sha=bc_sha,
                last_metrics=last_metrics,
                validation=validation,
                fixed_before=fixed_before,
            )
    fixed_after = _fixed_state(learner)
    if fixed_after != fixed_before:
        raise RuntimeError("{} fixed Critic state changed at branch end".format(branch))
    rng_after = _capture_rng()
    updates_path = output / "updates" / "{}.json".format(branch)
    _write_json(
        updates_path,
        {
            "branch": branch,
            "critic_checkpoint_sha256": critic_sha,
            "proposal_count": 100,
            "weight_mode": "awac",
            "fixed_state_before": fixed_before,
            "fixed_state_after": fixed_after,
            "preflight": preflight,
            "snapshots": snapshots,
            "rows": rows,
            "rng_before_updates": _rng_identity(rng_before_updates),
            "rng_after_updates": _rng_identity(rng_after),
            "critic_optimizer_step_count": 0,
            "target_soft_update_count": 0,
            "replay_read_only": True,
        },
    )
    return {
        "branch": branch,
        "critic_checkpoint_sha256": critic_sha,
        "fixed_state_before": fixed_before,
        "fixed_state_after": fixed_after,
        "preflight": preflight,
        "snapshots": snapshots,
        "updates_path": str(updates_path.resolve()),
        "updates_sha256": file_sha256(updates_path),
        "final_actor_state_sha256": _state_sha(learner.actor),
        "final_actor_optimizer_sha256": _state_sha(learner.actor_optimizer),
        "final_actor_update_count": int(learner.actor_update_count),
        "final_awac_update_count": int(learner.actor_awac_update_count),
        "final_recovery_update_count": int(learner.actor_recovery_update_count),
        "final_rejection_count": int(learner.actor_trust_region_rejection_count),
        "critic_optimizer_step_count": 0,
        "target_soft_update_count": 0,
        "rng_before_updates": _rng_identity(rng_before_updates),
        "rng_after_updates": _rng_identity(rng_after),
    }


def _equivalence(
    *,
    learner,
    endpoint: Mapping[str, Any],
    adapted_states: Mapping[str, Mapping[str, Any]],
    replay: AWACReplayBuffer,
    indices: np.ndarray,
) -> Dict[str, float]:
    batch = replay.sample_indices(indices, torch=torch, device=learner.device)
    results = {}
    for name in ("critic1", "critic2", "target_critic1", "target_critic2"):
        production = getattr(learner, name)
        diagnostic = BudgetCritic(
            nn=torch.nn,
            depth_channels=int(learner.depth_channels),
            source_state_dict=adapted_states[name],
            device=learner.device,
        )
        diagnostic.load_state_dict(endpoint["{}_state_dict".format(name)], strict=True)
        diagnostic.eval()
        production.eval()
        with torch.no_grad():
            zero_budget = torch.zeros((int(batch["vector"].shape[0]),), device=learner.device)
            production_q = production(batch["depth"], batch["vector"])
            diagnostic_q = diagnostic(batch["depth"], batch["vector"], zero_budget)
        difference = (production_q - diagnostic_q).abs()
        max_abs = float(difference.max().cpu().item())
        results["{}_max_abs".format(name)] = max_abs
        results["{}_allclose_atol_1e-5_rtol_1e-5".format(name)] = bool(
            torch.allclose(production_q, diagnostic_q, atol=1.0e-5, rtol=1.0e-5)
        )
        if not results["{}_allclose_atol_1e-5_rtol_1e-5".format(name)]:
            raise RuntimeError("{} H0 adapter numeric equivalence failed".format(name))
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--td1-checkpoint", required=True)
    parser.add_argument("--n5-checkpoint", required=True)
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--proposals", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if int(args.proposals) != 100:
        raise ValueError("this confirmation is fixed at exactly 100 proposals")
    if int(args.batch_size) <= 0:
        raise ValueError("batch size must be positive")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    device = torch.device(args.device)
    td1_path = Path(args.td1_checkpoint).expanduser().resolve()
    n5_path = Path(args.n5_checkpoint).expanduser().resolve()
    bc_path = Path(args.bc_checkpoint).expanduser().resolve()
    replay_path = Path(args.replay_dir).expanduser().resolve()
    output = _fresh_output(Path(args.out_dir))

    td1 = _load_payload(td1_path, expected_sha=EXPECTED_TD1_SHA, expected_branch="E_N1_H0")
    n5 = _load_payload(n5_path, expected_sha=EXPECTED_N5_SHA, expected_branch="E_N5_H0")
    bc, bc_fields = _load_bc(bc_path)
    config = _config_from_endpoint(td1)
    if dict(config.__dict__) != dict(_config_from_endpoint(n5).__dict__):
        raise ValueError("TD1 and N5 resolved optimization configs differ")
    replay = AWACReplayBuffer.open(replay_path, read_only=True)
    if int(replay.size) != EXPECTED_REPLAY_SIZE:
        raise ValueError("calibration Replay size mismatch")
    for key, expected in {
        "bc_checkpoint_sha256": EXPECTED_BC_SHA,
        "vector_dim": int(POLICY_VECTOR_DIM),
        "action_dim": int(NUM_ACTIONS),
        "legacy_replay_transition_count": 0,
        "observation_contract": "reliable_exact_endpoint_snapshot",
        "observation_source": "reliable_exact_endpoint_snapshot",
    }.items():
        if replay.metadata.get(key) != expected:
            raise ValueError("Replay metadata mismatch for {}".format(key))
    replay_hash_before = _hash_tree(replay_path)
    sequences = _make_sequences(
        replay.size,
        proposals=int(args.proposals),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    sequence_path = output / "fixed_batch_sequences.npz"
    np.savez_compressed(str(sequence_path), **sequences)
    sequence_sha = file_sha256(sequence_path)

    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "python/planning/awac/critic_actor_only.py",
        Path(__file__).resolve().parents[1] / "tests/test_critic_actor_only_adapter.py",
        Path(__file__).resolve().parents[1] / "python/planning/awac/learner.py",
    ]
    source_identity = {
        str(path): file_sha256(path)
        for path in source_paths
        if path.is_file()
    }
    preregistration = {
        "protocol_written_before_actor_updates": True,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "diagnostic": "N5_H0_frozen_Critic_vs_TD1_H0_frozen_Critic_Actor_only",
        "no_environment_steps": True,
        "unity_started": False,
        "bridge_started": False,
        "dev100_executed": False,
        "actor_update_proposals_per_branch": int(args.proposals),
        "snapshot_proposals": list(SNAPSHOT_STEPS),
        "weight_mode": "awac",
        "actor_initialization": "same_BC_checkpoint",
        "fixed_critic": True,
        "critic_optimizer_steps": 0,
        "target_soft_updates": 0,
        "replay_read_only": True,
        "batch_size": int(args.batch_size),
        "sequence_seed": int(args.seed),
        "sequence_sha256": sequence_sha,
        "sequence_shapes": {name: list(value.shape) for name, value in sequences.items()},
        "td1_checkpoint": {"path": str(td1_path), "sha256": EXPECTED_TD1_SHA, "branch": "E_N1_H0", "updates": 3000},
        "n5_checkpoint": {"path": str(n5_path), "sha256": EXPECTED_N5_SHA, "branch": "E_N5_H0", "updates": 3000},
        "bc_checkpoint": {"path": str(bc_path), "sha256": EXPECTED_BC_SHA},
        "replay": {"path": str(replay_path), "size": int(replay.size), "tree_sha256": replay_hash_before},
        "resolved_optimization_config": dict(config.__dict__),
        "actual_optimizer_lr_source": "captured_from_each_built_learner",
        "comparison": "offline Actor-only metrics only; no policy success claim without Dev100",
        "source_identity_before": source_identity,
        "bc_validation_fields": bc_fields,
        "output_dir": str(output),
    }
    _write_json(output / "preregistration.json", preregistration)

    adapted_states = {}
    adapter_reports = {}
    for branch, endpoint in (("TD1_A", td1), ("N5_B", n5)):
        adapted_states[branch], adapter_reports[branch] = {}, {}
        for name in ("critic1", "critic2", "target_critic1", "target_critic2"):
            adapted_states[branch][name], adapter_reports[branch][name] = adapt_h0_diagnostic_state_dict(
                endpoint["{}_state_dict".format(name)],
                torch=torch,
                policy_vector_dim=int(POLICY_VECTOR_DIM),
            )

    learner_a, build_a, initial_a = _build_branch(
        branch="TD1_A",
        endpoint=td1,
        bc=bc,
        config=config,
        device=device,
        seed=int(args.seed),
    )
    equivalence_a = _equivalence(
        learner=learner_a,
        endpoint=td1,
        adapted_states=adapted_states["TD1_A"],
        replay=replay,
        indices=sequences["validation_indices"][: int(args.batch_size)],
    )
    result_a = _run_branch(
        output=output,
        branch="TD1_A",
        critic_sha=EXPECTED_TD1_SHA,
        learner=learner_a,
        build_rng=build_a["build_rng"],
        replay=replay,
        sequences=sequences,
        bc_sha=EXPECTED_BC_SHA,
    )
    del learner_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    learner_b, build_b, initial_b = _build_branch(
        branch="N5_B",
        endpoint=n5,
        bc=bc,
        config=config,
        device=device,
        seed=int(args.seed),
    )
    if initial_a["actor"] != initial_b["actor"]:
        raise RuntimeError("TD1/N5 Actor initial state differs")
    if initial_a["actor_optimizer"] != initial_b["actor_optimizer"]:
        raise RuntimeError("TD1/N5 Actor optimizer initial state differs")
    if initial_a["bc_reference"] != initial_b["bc_reference"]:
        raise RuntimeError("TD1/N5 BC reference differs")
    if _rng_identity(build_a["build_rng"]) != _rng_identity(build_b["build_rng"]):
        raise RuntimeError("TD1/N5 post-construction RNG state differs")
    equivalence_b = _equivalence(
        learner=learner_b,
        endpoint=n5,
        adapted_states=adapted_states["N5_B"],
        replay=replay,
        indices=sequences["validation_indices"][: int(args.batch_size)],
    )
    result_b = _run_branch(
        output=output,
        branch="N5_B",
        critic_sha=EXPECTED_N5_SHA,
        learner=learner_b,
        build_rng=build_b["build_rng"],
        replay=replay,
        sequences=sequences,
        bc_sha=EXPECTED_BC_SHA,
    )
    if result_a["rng_before_updates"] != result_b["rng_before_updates"]:
        raise RuntimeError("TD1/N5 RNG state before proposals differs")
    del learner_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    replay_hash_after = _hash_tree(replay_path)
    if replay_hash_before != replay_hash_after:
        raise RuntimeError("read-only Replay content changed during Actor-only experiment")
    initial_identity = {
        "TD1_A": initial_a,
        "N5_B": initial_b,
        "same_actor": True,
        "same_actor_optimizer": True,
        "same_bc_reference": True,
        "same_post_build_rng": True,
    }
    _write_json(output / "initial_identity.json", initial_identity)
    _write_json(
        output / "adapter_equivalence.json",
        {
            "TD1_A": equivalence_a,
            "N5_B": equivalence_b,
            "allclose_atol_1e-5_rtol_1e-5": all(
                bool(value)
                for report in (equivalence_a, equivalence_b)
                for key, value in report.items()
                if key.endswith("allclose_atol_1e-5_rtol_1e-5")
            ),
            "gpu_zero_budget_column_forward_can_differ_by_float_rounding": True,
            "adapter_reports": adapter_reports,
        },
    )
    _write_json(
        output / "replay_immutability.json",
        {
            "read_only": True,
            "before": replay_hash_before,
            "after": replay_hash_after,
            "unchanged": replay_hash_before == replay_hash_after,
        },
    )
    comparison = {
        "TD1_A": result_a,
        "N5_B": result_b,
        "final_offline_metric_difference_N5_minus_TD1": {
            "final_awac_update_count": result_b["final_awac_update_count"] - result_a["final_awac_update_count"],
            "final_recovery_update_count": result_b["final_recovery_update_count"] - result_a["final_recovery_update_count"],
            "final_rejection_count": result_b["final_rejection_count"] - result_a["final_rejection_count"],
            "final_actor_state_hash_equal": result_b["final_actor_state_sha256"] == result_a["final_actor_state_sha256"],
            "proposal_100_actor_kl": result_b["snapshots"]["100"]["actor_kl"] - result_a["snapshots"]["100"]["actor_kl"],
            "proposal_100_argmax_flip_rate": result_b["snapshots"]["100"]["argmax_flip_rate"] - result_a["snapshots"]["100"]["argmax_flip_rate"],
        },
        "policy_success_comparison": "NOT_EXECUTED_NO_DEV100",
        "multi_seed_follow_up": "NOT_DECIDED_WITHOUT_DEV100",
    }
    _write_json(output / "comparison.json", comparison)
    source_identity_after = {str(path): file_sha256(path) for path in source_paths if path.is_file()}
    _write_json(
        output / "source_identity.json",
        {
            "before": source_identity,
            "after": source_identity_after,
            "existing_learner_unchanged": source_identity.get(str(source_paths[-1])) == source_identity_after.get(str(source_paths[-1])),
            "modified_files_for_this_diagnostic": [str(source_paths[0]), str(source_paths[1]), str(source_paths[2])],
        },
    )
    report = "# N5 frozen-Critic Actor-only confirmation\n\n"
    report += "本实验仅执行离线 Actor-only proposal；没有启动 Unity/Bridge，没有环境步，没有 Dev100。\n\n"
    report += "- TD1_A: frozen E_N1_H0 Critic, 100 AWAC proposals\n"
    report += "- N5_B: frozen E_N5_H0 Critic, 100 AWAC proposals\n"
    report += "- 两分支共享 BC、batch index、RNG、Actor optimizer 初态；Critic/target/Replay 不变。\n"
    report += "- H0 128→127 adapter 与诊断模型零预算输入逐项 exact parity。\n"
    report += "- Dev100: NOT EXECUTED（按本轮禁止环境/Dev100约束）。\n"
    (output / "report_zh.md").write_text(report, encoding="utf-8")
    print("OUTPUT_DIR={}".format(output))
    print("TD1_ENDPOINT_SHA256={}".format(EXPECTED_TD1_SHA))
    print("N5_ENDPOINT_SHA256={}".format(EXPECTED_N5_SHA))
    print("BC_SHA256={}".format(EXPECTED_BC_SHA))
    print("REPLAY_SIZE={}".format(replay.size))
    print("TD1_A_FINAL_AWAC_UPDATES={}".format(result_a["final_awac_update_count"]))
    print("N5_B_FINAL_AWAC_UPDATES={}".format(result_b["final_awac_update_count"]))
    print("TD1_A_FINAL_RECOVERY_UPDATES={}".format(result_a["final_recovery_update_count"]))
    print("N5_B_FINAL_RECOVERY_UPDATES={}".format(result_b["final_recovery_update_count"]))
    print("TD1_A_FINAL_REJECTIONS={}".format(result_a["final_rejection_count"]))
    print("N5_B_FINAL_REJECTIONS={}".format(result_b["final_rejection_count"]))
    print("REPLAY_UNCHANGED=YES")
    print("DEV100_EXECUTED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
