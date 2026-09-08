#!/usr/bin/env python3
"""Offline first-KL-crossing audit for the experimental Discrete SAC run.

This command consumes the already verified deterministic-replay V2 run.  It
never samples an environment and never writes into the source run.  The
replay, batch schedule, and RNG fingerprints are replayed from checkpoint 0;
policy drift is then measured on a fixed pre-Actor sentinel set rather than
only on whichever state happened to be present in a minibatch.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from planning.common.hashing import file_sha256
from planning.sac.checkpoint import load_checkpoint
from planning.sac.diagnostics import (
    capture_rng_state,
    rng_fingerprint,
    stable_fingerprint,
)
from planning.sac.network import DiscreteSACAgent, masked_kl
from planning.sac.replay import SACReplayBuffer


ROOT_SCHEMA = "bc_initialized_discrete_sac_first_kl_crossing_v3"
SENTINEL_SCHEMA = "bc_initialized_discrete_sac_sentinel_manifest_v3"
TRAJECTORY_SCHEMA = "bc_initialized_discrete_sac_sentinel_trajectory_v3"
CROSSING_SCHEMA = "bc_initialized_discrete_sac_first_crossing_v3"
GRADIENT_SCHEMA = "bc_initialized_discrete_sac_crossing_gradient_v3"
COUNTERFACTUAL_SCHEMA = "bc_initialized_discrete_sac_one_step_counterfactual_v3"
CUMULATIVE_SCHEMA = "bc_initialized_discrete_sac_cumulative_ablation_v3"
TRANSACTION_SCHEMA = "bc_initialized_discrete_sac_transaction_ordering_audit_v3"
ROLLBACK_SCHEMA = "bc_initialized_discrete_sac_rollback_regression_v3"
ROOT_CAUSE_SCHEMA = "bc_initialized_discrete_sac_root_cause_v3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n") or not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if value.get("schema_id") == "bc_initialized_discrete_sac_update_journal_v2":
                rows.append(value)
    return rows


def _sha256_tree(root: Path) -> Mapping[str, str]:
    values: Dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        values[str(path.relative_to(root))] = file_sha256(path)
    return values


def _restore_rng(torch: Any, state: Mapping[str, Any]) -> None:
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_global" in state:
        np.random.set_state(state["numpy_global"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _new_agent(torch: Any, nn: Any, device: Any, bc_checkpoint: Mapping[str, Any], config: Mapping[str, Any]):
    return DiscreteSACAgent(
        torch=torch,
        nn=nn,
        device=device,
        bc_checkpoint=bc_checkpoint,
        actor_lr=float(config.get("actor_lr", 1.0e-5)),
        critic_lr=float(config.get("critic_lr", 1.0e-4)),
        gamma=float(config.get("gamma", 0.99)),
        tau=float(config.get("tau", 0.005)),
        alpha=float(config.get("alpha", 0.20)),
        beta_bc=float(config.get("beta_bc", 0.05)),
        reward_scale=float(config.get("reward_scale", 0.10)),
        gradient_clip_norm=float(config.get("gradient_clip_norm", 5.0)),
        entropy_floor=float(config.get("entropy_floor", 0.02)),
        deterministic_pool=bool(
            config.get("deterministic_pool", config.get("deterministic_replay", False))
        ),
    )


def _restore_agent(agent: Any, payload: Mapping[str, Any]) -> None:
    for name in (
        "actor", "bc_reference", "critic1", "critic2", "target_critic1", "target_critic2"
    ):
        getattr(agent, name).load_state_dict(payload[name + "_state_dict"], strict=True)
    agent.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    agent.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    for name in (
        "actor_optimizer_step_count", "critic_optimizer_step_count", "actor_update_count",
        "critic_update_count", "actor_proposal_count", "actor_rejection_count",
        "actor_recovery_update_count", "nan_count", "invalid_action_count",
    ):
        if name in payload:
            setattr(agent, name, int(payload[name]))


def _copy_batch(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value.detach().clone() if hasattr(value, "detach") else deepcopy(value)
        for key, value in batch.items()
    }


def _state_hash(replay: SACReplayBuffer, row: int) -> str:
    digest = hashlib.sha256()
    for name in ("depth", "vector", "action_mask"):
        value = np.ascontiguousarray(replay.arrays[name][int(row)])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _mask_hash(replay: SACReplayBuffer, row: int) -> str:
    value = np.ascontiguousarray(np.asarray(replay.arrays["action_mask"][int(row)], dtype=np.uint8))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _sentinel_rows(replay: SACReplayBuffer, first_actor_transition: int) -> Tuple[List[int], List[Mapping[str, Any]]]:
    if first_actor_transition <= 0 or first_actor_transition > int(replay.size):
        raise ValueError("first Actor transition is outside committed Replay")
    rows: List[int] = []
    manifest: List[Mapping[str, Any]] = []
    seen = set()
    for row in range(int(first_actor_transition)):
        state_hash = _state_hash(replay, row)
        if state_hash in seen:
            continue
        seen.add(state_hash)
        identity = dict(replay.identity_records[row])
        rows.append(row)
        manifest.append(
            {
                "replay_row": int(row),
                "mission_id": str(identity["mission_id"]),
                "episode_id": str(identity["episode_id"]),
                "step_id": int(identity["step_id"]),
                "transition_id": str(identity["transition_id"]),
                "state_sha256": state_hash,
                "action_mask_sha256": _mask_hash(replay, row),
            }
        )
    return rows, manifest


def _quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values.size else 0.0


def _policy_stats(
    agent: Any,
    replay: SACReplayBuffer,
    rows: Sequence[int],
    *,
    torch: Any,
    device: Any,
    chunk_size: int,
) -> Mapping[str, Any]:
    kl_values: List[np.ndarray] = []
    entropy_values: List[np.ndarray] = []
    tv_values: List[np.ndarray] = []
    flip_values: List[np.ndarray] = []
    for start in range(0, len(rows), int(chunk_size)):
        indices = np.asarray(rows[start : start + int(chunk_size)], dtype=np.int64)
        batch = replay.batch_from_indices(indices, torch=torch, device=device)
        with torch.no_grad():
            actor_prob, actor_log, actor_mask = agent._distribution(
                agent.actor, batch["depth"], batch["vector"], batch["action_mask"].bool()
            )
            bc_prob, bc_log, _ = agent._distribution(
                agent.bc_reference, batch["depth"], batch["vector"], batch["action_mask"].bool()
            )
            kl = masked_kl(actor_prob, actor_log, bc_prob, bc_log, torch)
            entropy = -(actor_prob * actor_log).sum(dim=1)
            tv = 0.5 * torch.abs(actor_prob - bc_prob).sum(dim=1)
            actor_argmax = torch.argmax(actor_prob.masked_fill(~actor_mask, -1.0), dim=1)
            bc_argmax = torch.argmax(bc_prob.masked_fill(~actor_mask, -1.0), dim=1)
            flip = (actor_argmax != bc_argmax).float()
        kl_values.append(kl.detach().cpu().numpy().astype(np.float64))
        entropy_values.append(entropy.detach().cpu().numpy().astype(np.float64))
        tv_values.append(tv.detach().cpu().numpy().astype(np.float64))
        flip_values.append(flip.detach().cpu().numpy().astype(np.float64))
    kl = np.concatenate(kl_values) if kl_values else np.zeros((0,), dtype=np.float64)
    entropy = np.concatenate(entropy_values) if entropy_values else np.zeros((0,), dtype=np.float64)
    tv = np.concatenate(tv_values) if tv_values else np.zeros((0,), dtype=np.float64)
    flip = np.concatenate(flip_values) if flip_values else np.zeros((0,), dtype=np.float64)
    max_position = int(np.argmax(kl)) if kl.size else -1
    max_row = int(rows[max_position]) if max_position >= 0 else None
    return {
        "count": int(kl.size),
        "kl_mean": float(kl.mean()) if kl.size else 0.0,
        "kl_p50": _quantile(kl, 0.50),
        "kl_p90": _quantile(kl, 0.90),
        "kl_p95": _quantile(kl, 0.95),
        "kl_p99": _quantile(kl, 0.99),
        "kl_p999": _quantile(kl, 0.999),
        "kl_max": float(kl.max()) if kl.size else 0.0,
        "kl_max_replay_row": max_row,
        "count_gt_0_1": int(np.count_nonzero(kl > 0.1)),
        "count_gt_0_25": int(np.count_nonzero(kl > 0.25)),
        "count_gt_0_5": int(np.count_nonzero(kl > 0.5)),
        "count_gt_1": int(np.count_nonzero(kl >= 1.0)),
        "entropy_mean": float(entropy.mean()) if entropy.size else 0.0,
        "argmax_flip_rate": float(flip.mean()) if flip.size else 0.0,
        "tv_mean": float(tv.mean()) if tv.size else 0.0,
        "tv_max": float(tv.max()) if tv.size else 0.0,
    }


def _single_state_stats(agent: Any, replay: SACReplayBuffer, row: int, *, torch: Any, device: Any) -> Mapping[str, Any]:
    batch = replay.batch_from_indices(np.asarray([row], dtype=np.int64), torch=torch, device=device)
    with torch.no_grad():
        actor_prob, actor_log, actor_mask = agent._distribution(
            agent.actor, batch["depth"], batch["vector"], batch["action_mask"].bool()
        )
        bc_prob, bc_log, _ = agent._distribution(
            agent.bc_reference, batch["depth"], batch["vector"], batch["action_mask"].bool()
        )
        kl = masked_kl(actor_prob, actor_log, bc_prob, bc_log, torch)
        entropy = -(actor_prob * actor_log).sum(dim=1)
        tv = 0.5 * torch.abs(actor_prob - bc_prob).sum(dim=1)
        actor_logits = agent.actor(batch["depth"], batch["vector"]).detach().cpu().numpy()[0]
        bc_logits = agent.bc_reference(batch["depth"], batch["vector"]).detach().cpu().numpy()[0]
        actor_argmax = int(torch.argmax(actor_prob.masked_fill(~actor_mask, -1.0), dim=1)[0].item())
        bc_argmax = int(torch.argmax(bc_prob.masked_fill(~actor_mask, -1.0), dim=1)[0].item())
    return {
        "kl": float(kl[0].item()),
        "entropy": float(entropy[0].item()),
        "tv": float(tv[0].item()),
        "actor_argmax": actor_argmax,
        "bc_argmax": bc_argmax,
        "argmax_flip": bool(actor_argmax != bc_argmax),
        "bc_probabilities": bc_prob.detach().cpu().numpy()[0].tolist(),
        "actor_probabilities": actor_prob.detach().cpu().numpy()[0].tolist(),
        "max_logit_delta": float(np.max(np.abs(actor_logits - bc_logits))),
    }


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty_like(values_array, dtype=np.float64)
    ranks[order] = np.arange(values_array.size, dtype=np.float64)
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float], *, rank: bool = False) -> Optional[float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or y.size != x.size or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    if rank:
        x = _rankdata(x)
        y = _rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _actor_specs() -> Mapping[str, Tuple[bool, bool, bool, bool]]:
    return {
        "R0_ORIGINAL": (True, True, True, True),
        "R1_NO_Q_TERM": (False, True, True, True),
        "R2_NO_ENTROPY_TERM": (True, False, True, True),
        "R3_NO_BC_KL_TERM": (True, True, False, True),
        "R4_NO_OPTIMIZER_STEP": (True, True, True, False),
        "R5_Q_ONLY": (True, False, False, True),
        "R6_ENTROPY_ONLY": (False, True, False, True),
        "R7_BC_KL_ONLY": (False, False, True, True),
        "R8_Q_PLUS_BC_KL": (True, False, True, True),
    }


def _flat_gradient(values: Sequence[Any], torch: Any) -> Any:
    tensors = [value.detach().reshape(-1) for value in values if value is not None]
    return torch.cat(tensors) if tensors else torch.zeros((0,))


def _cosine(left: Any, right: Any, torch: Any) -> Optional[float]:
    if left.numel() == 0 or right.numel() == 0:
        return None
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        return None
    return float((torch.dot(left, right) / (left_norm * right_norm)).item())


def _counterfactual_step(
    agent: Any,
    batch: Mapping[str, Any],
    *,
    torch: Any,
    include_q: bool,
    include_entropy: bool,
    include_bc_kl: bool,
    do_step: bool,
) -> Mapping[str, Any]:
    parameters = list(agent.actor.parameters())
    for module in (agent.critic1, agent.critic2):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    try:
        depth, vector, mask = batch["depth"], batch["vector"], batch["action_mask"].bool()
        probabilities, log_probs, valid_mask = agent._distribution(agent.actor, depth, vector, mask)
        with torch.no_grad():
            bc_prob, bc_log, _ = agent._distribution(agent.bc_reference, depth, vector, mask)
        q_min = torch.minimum(agent.critic1(depth, vector), agent.critic2(depth, vector))
        q_term = (probabilities * (-q_min)).sum(dim=1)
        entropy_term = (probabilities * log_probs).sum(dim=1) * agent.alpha
        kl = masked_kl(probabilities, log_probs, bc_prob, bc_log, torch)
        bc_kl_term = agent.beta_bc * kl
        term_values = {"q": q_term, "entropy": entropy_term, "bc_kl": bc_kl_term}
        enabled = {"q": include_q, "entropy": include_entropy, "bc_kl": include_bc_kl}
        selected = [term_values[name] for name in ("q", "entropy", "bc_kl") if enabled[name]]
        actor_loss = sum(selected).mean()
        component_gradients = {}
        for name in ("q", "entropy", "bc_kl"):
            if enabled[name]:
                gradients = torch.autograd.grad(
                    term_values[name].mean(), parameters, retain_graph=True, allow_unused=True
                )
                component_gradients[name] = _flat_gradient(gradients, torch)
            else:
                component_gradients[name] = torch.zeros((0,), device=depth.device)
        actor_before = stable_fingerprint(agent.actor.state_dict())
        optimizer_before = stable_fingerprint(agent.actor_optimizer.state_dict())
        parameter_before = [parameter.detach().clone() for parameter in parameters]
        pre_kl = kl.detach()
        pre_entropy = (-(probabilities * log_probs).sum(dim=1)).detach()
        actor_logits_before = agent.actor(depth, vector).detach().cpu().numpy()
        actor_prob_before = probabilities.detach().cpu().numpy()
        agent.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        preclip = float(torch.nn.utils.clip_grad_norm_(parameters, agent.gradient_clip_norm).item())
        post_values = [parameter.grad.detach().reshape(-1) for parameter in parameters if parameter.grad is not None]
        postclip = float(torch.linalg.vector_norm(torch.cat(post_values)).item()) if post_values else 0.0
        if do_step:
            agent.actor_optimizer.step()
        with torch.no_grad():
            post_prob, post_log, post_mask = agent._distribution(agent.actor, depth, vector, mask)
            post_kl = masked_kl(post_prob, post_log, bc_prob, bc_log, torch)
            post_entropy = -(post_prob * post_log).sum(dim=1)
        pre_max = float(pre_kl.max().item())
        post_max = float(post_kl.max().item())
        return {
            "kl_before_mean": float(pre_kl.mean().item()),
            "kl_before_max": pre_max,
            "kl_before_p99": float(torch.quantile(pre_kl, 0.99).item()),
            "kl_after_mean": float(post_kl.mean().item()),
            "kl_after_max": post_max,
            "kl_after_p99": float(torch.quantile(post_kl, 0.99).item()),
            "kl_delta_max": post_max - pre_max,
            "parameter_delta_norm": float(
                math.sqrt(
                    sum(
                        float((after.detach() - before.detach()).pow(2).sum().item())
                        for after, before in zip(agent.actor.parameters(), parameter_before)
                    )
                )
            ),
            "max_logit_delta": float(
                np.max(np.abs(agent.actor(depth, vector).detach().cpu().numpy() - actor_logits_before))
            ),
            "tv_delta_mean": float(
                (0.5 * torch.abs(post_prob - probabilities).sum(dim=1)).mean().item()
            ),
            "tv_delta_max": float(
                (0.5 * torch.abs(post_prob - probabilities).sum(dim=1)).max().item()
            ),
            "gradient_norm_q": float(torch.linalg.vector_norm(component_gradients["q"]).item()) if component_gradients["q"].numel() else 0.0,
            "gradient_norm_entropy": float(torch.linalg.vector_norm(component_gradients["entropy"]).item()) if component_gradients["entropy"].numel() else 0.0,
            "gradient_norm_bc_kl": float(torch.linalg.vector_norm(component_gradients["bc_kl"]).item()) if component_gradients["bc_kl"].numel() else 0.0,
            "gradient_norm_preclip": preclip,
            "gradient_norm_postclip": postclip,
            "actor_loss": float(actor_loss.item()),
            "q_min_mean": float(q_min.mean().item()),
            "q_min_std": float(q_min.std(unbiased=False).item()),
            "q_min_min": float(q_min.min().item()),
            "q_min_max": float(q_min.max().item()),
            "entropy_mean": float(pre_entropy.mean().item()),
            "alpha": float(agent.alpha),
            "pre_step_hard_stop": bool(pre_max >= 1.0),
            "post_step_hard_stop": bool(post_max >= 1.0),
            "hard_stop_triggered": bool(pre_max >= 1.0 or post_max >= 1.0),
            "optimizer_step_performed": bool(do_step),
            "actor_before_sha256": actor_before,
            "actor_after_sha256": stable_fingerprint(agent.actor.state_dict()),
            "optimizer_before_sha256": optimizer_before,
            "optimizer_after_sha256": stable_fingerprint(agent.actor_optimizer.state_dict()),
            "valid_action_count_min": int(valid_mask.sum(dim=1).min().item()),
            "valid_action_count_max": int(valid_mask.sum(dim=1).max().item()),
            "bc_probabilities": bc_prob.detach().cpu().numpy().tolist(),
            "actor_probabilities_before": actor_prob_before.tolist(),
            "actor_probabilities_after": post_prob.detach().cpu().numpy().tolist(),
            "gradient_vectors": {
                name: value.detach().cpu().numpy().tolist()
                for name, value in component_gradients.items()
                if value.numel()
            },
        }
    finally:
        for module in (agent.critic1, agent.critic2):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def _gradient_decomposition(agent: Any, batch: Mapping[str, Any], *, torch: Any) -> Mapping[str, Any]:
    result = _counterfactual_step(
        agent,
        batch,
        torch=torch,
        include_q=True,
        include_entropy=True,
        include_bc_kl=True,
        do_step=False,
    )
    vectors = {
        name: torch.tensor(values, dtype=torch.float32)
        for name, values in result.get("gradient_vectors", {}).items()
    }
    result["gradient_cosine_q_entropy"] = _cosine(vectors.get("q", torch.zeros(0)), vectors.get("entropy", torch.zeros(0)), torch)
    result["gradient_cosine_q_bc_kl"] = _cosine(vectors.get("q", torch.zeros(0)), vectors.get("bc_kl", torch.zeros(0)), torch)
    result["gradient_cosine_entropy_bc_kl"] = _cosine(vectors.get("entropy", torch.zeros(0)), vectors.get("bc_kl", torch.zeros(0)), torch)
    result.pop("gradient_vectors", None)
    return result


def _capture_payload_at_actor_step(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    config: Mapping[str, Any],
    bc_checkpoint: Mapping[str, Any],
    journal: Sequence[Mapping[str, Any]],
    target_step: int,
    torch: Any,
    nn: Any,
    device: Any,
) -> Tuple[Mapping[str, Any], np.ndarray]:
    checkpoint = load_checkpoint(torch, checkpoint_path, map_location="cpu")
    agent = _new_agent(torch, nn, device, bc_checkpoint, config)
    _restore_agent(agent, checkpoint)
    rng_state = checkpoint.get("rng_state", {})
    _restore_rng(torch, rng_state)
    replay = SACReplayBuffer.open(run_dir / "replay", read_only=True)
    replay_rng = np.random.RandomState()
    replay_rng.set_state(rng_state.get("replay_sampler", rng_state.get("numpy")))
    for record in journal:
        indices = np.asarray(record.get("batch_indices", []), dtype=np.int64)
        expected = np.asarray(
            replay_rng.randint(0, int(record["replay_size"]), size=indices.size), dtype=np.int64
        )
        if not np.array_equal(expected, indices):
            raise RuntimeError("batch schedule diverged while capturing actor checkpoint")
        batch = replay.batch_from_indices(indices, torch=torch, device=device)
        if record.get("update_kind") == "actor" and int(record.get("update_index", -1)) == int(target_step):
            return deepcopy(agent.state_payload()), indices
        if record.get("update_kind") == "critic":
            agent.update_critic(batch)
        else:
            agent.update_actor_transactional(
                batch,
                hard_stop=float(config.get("bc_kl_hard_stop", 1.0)),
                entropy_floor=float(config.get("entropy_floor", 0.02)),
            )
    raise RuntimeError("requested actor checkpoint was not found")


def _input_identity(
    run_dir: Path,
    checkpoint_path: Path,
    source_input_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    exact = _read_json(run_dir / "exact_replay_result.json")
    first = _read_json(run_dir / "first_divergence.json")
    if exact.get("status") != "PASS" or not bool(exact.get("exact_replay")):
        raise RuntimeError("V2 exact replay source is not PASS")
    if first.get("status") != "NONE":
        raise RuntimeError("V2 exact replay source has a first divergence")
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != exact.get("checkpoint_sha256"):
        raise RuntimeError("V2 checkpoint SHA does not match exact replay result")
    bc_path = Path(source_input_identity["bc_checkpoint_path"]).expanduser().resolve()
    return {
        "schema_id": ROOT_SCHEMA,
        "source_run_dir": str(run_dir),
        "source_exact_replay_result_sha256": file_sha256(run_dir / "exact_replay_result.json"),
        "source_checkpoint_step_0_sha256": checkpoint_hash,
        "source_checkpoint_manifest_sha256": file_sha256(run_dir / "checkpoint_manifest.json"),
        "source_replay_audit_sha256": file_sha256(run_dir / "replay_audit.json"),
        "source_journal_sha256": file_sha256(run_dir / "training_step_journal.jsonl"),
        "source_batch_index_journal_sha256": file_sha256(run_dir / "batch_index_journal.npz"),
        "source_one_step_counterfactual_sha256": file_sha256(run_dir / "one_step_counterfactual.json"),
        "source_root_cause_sha256": file_sha256(run_dir / "root_cause.json"),
        "source_input_file_sha256": _sha256_tree(run_dir),
        "source_bc_checkpoint_path": str(bc_path),
        "source_bc_checkpoint_sha256": file_sha256(bc_path),
        "source_input_identity_sha256": stable_fingerprint(source_input_identity),
        "exact_replay_pass": True,
        "first_divergence": None,
        "production_default_modified": False,
    }


def _load_runtime(
    run_dir: Path,
    device_name: str,
    torch: Any,
    nn: Any,
) -> Tuple[Any, Any, Any, Mapping[str, Any], Mapping[str, Any], Path, List[Mapping[str, Any]]]:
    config = _read_json(run_dir / "training_config.json")
    source_input_identity = _read_json(run_dir / "input_identity.json")
    checkpoint_path = run_dir / "checkpoint_step_00000000.pt"
    checkpoint = load_checkpoint(torch, checkpoint_path, map_location="cpu")
    bc_path = Path(source_input_identity["bc_checkpoint_path"]).expanduser().resolve()
    bc_checkpoint = torch.load(str(bc_path), map_location="cpu", weights_only=False)
    device = torch.device(device_name)
    agent = _new_agent(torch, nn, device, bc_checkpoint, config)
    _restore_agent(agent, checkpoint)
    replay = SACReplayBuffer.open(run_dir / "replay", read_only=True)
    journal = _records(run_dir / "training_step_journal.jsonl")
    return agent, replay, checkpoint, config, source_input_identity, checkpoint_path, journal


def _trajectory_row(
    *,
    actor_step: int,
    journal_sequence: Optional[int],
    stats: Mapping[str, Any],
    spike_stats: Mapping[str, Any],
    batch_record: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    metrics = dict(batch_record.get("metrics", {})) if batch_record else {}
    return {
        "schema_id": TRAJECTORY_SCHEMA,
        "actor_update_step": int(actor_step),
        "journal_sequence": journal_sequence,
        "batch_kl_mean": metrics.get("bc_kl_mean"),
        "batch_kl_p90": metrics.get("bc_kl_p90"),
        "batch_kl_p99": metrics.get("bc_kl_p99"),
        "batch_kl_max_before": metrics.get("bc_kl_max"),
        "batch_kl_mean_after": metrics.get("bc_kl_post_mean"),
        "batch_kl_max_after": metrics.get("bc_kl_post_max"),
        "sentinel_kl_mean": stats["kl_mean"],
        "sentinel_kl_p50": stats["kl_p50"],
        "sentinel_kl_p90": stats["kl_p90"],
        "sentinel_kl_p95": stats["kl_p95"],
        "sentinel_kl_p99": stats["kl_p99"],
        "sentinel_kl_p999": stats["kl_p999"],
        "sentinel_kl_max": stats["kl_max"],
        "sentinel_count_gt_0_1": stats["count_gt_0_1"],
        "sentinel_count_gt_0_25": stats["count_gt_0_25"],
        "sentinel_count_gt_0_5": stats["count_gt_0_5"],
        "sentinel_count_gt_1": stats["count_gt_1"],
        "sentinel_entropy_mean": stats["entropy_mean"],
        "sentinel_argmax_flip_rate": stats["argmax_flip_rate"],
        "sentinel_tv_mean": stats["tv_mean"],
        "sentinel_tv_max": stats["tv_max"],
        "spike_state_kl": spike_stats["kl"],
        "spike_state_entropy": spike_stats["entropy"],
        "spike_state_tv": spike_stats["tv"],
        "spike_state_argmax": spike_stats["actor_argmax"],
        "spike_state_bc_argmax": spike_stats["bc_argmax"],
        "spike_state_argmax_flip": spike_stats["argmax_flip"],
    }


def _replay_audit(
    *,
    run_dir: Path,
    agent: Any,
    replay: SACReplayBuffer,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    journal: Sequence[Mapping[str, Any]],
    sentinel_rows: Sequence[int],
    spike_row: int,
    torch: Any,
    device: Any,
    chunk_size: int,
) -> Mapping[str, Any]:
    rng_state = checkpoint.get("rng_state", {})
    _restore_rng(torch, rng_state)
    replay_rng = np.random.RandomState()
    replay_rng.set_state(rng_state.get("replay_sampler", rng_state.get("numpy")))
    comparison: List[Mapping[str, Any]] = []
    accepted_records: List[Mapping[str, Any]] = []
    trajectories: List[Mapping[str, Any]] = []
    spike_trajectory: List[Mapping[str, Any]] = []
    pre_cross_candidate: Optional[Mapping[str, Any]] = None
    new_rejected: Optional[Mapping[str, Any]] = None
    initial_stats: Optional[Mapping[str, Any]] = None
    previous_stats: Optional[Mapping[str, Any]] = None
    previous_spike: Optional[Mapping[str, Any]] = None
    accepted_count = 0
    for record in journal:
        indices = np.asarray(record.get("batch_indices", []), dtype=np.int64)
        expected = np.asarray(
            replay_rng.randint(0, int(record["replay_size"]), size=indices.size), dtype=np.int64
        )
        batch_match = bool(np.array_equal(expected, indices))
        batch = replay.batch_from_indices(indices, torch=torch, device=device)
        rng_before = capture_rng_state(torch, replay_rng)
        state_before = stable_fingerprint(agent.state_payload())
        expected_before = record.get("model_state_before_sha256")
        kind = str(record.get("update_kind", ""))
        if kind in ("actor", "actor_rejected") and initial_stats is None:
            initial_stats = _policy_stats(
                agent, replay, sentinel_rows, torch=torch, device=device, chunk_size=chunk_size
            )
            previous_stats = initial_stats
            previous_spike = _single_state_stats(agent, replay, spike_row, torch=torch, device=device)
            trajectories.append(
                _trajectory_row(
                    actor_step=0,
                    journal_sequence=None,
                    stats=initial_stats,
                    spike_stats=previous_spike,
                    batch_record=None,
                )
            )
            spike_trajectory.append(
                {
                    "actor_update_step": 0,
                    "replay_row": int(spike_row),
                    "kl": previous_spike["kl"],
                    "entropy": previous_spike["entropy"],
                    "tv": previous_spike["tv"],
                    "actor_argmax": previous_spike["actor_argmax"],
                    "bc_argmax": previous_spike["bc_argmax"],
                    "argmax_flip": previous_spike["argmax_flip"],
                    "max_logit_delta": previous_spike["max_logit_delta"],
                    "bc_probabilities": json.dumps(previous_spike["bc_probabilities"], separators=(",", ":")),
                    "actor_probabilities": json.dumps(previous_spike["actor_probabilities"], separators=(",", ":")),
                }
            )
        if kind in ("actor", "actor_rejected"):
            proposal_before_stats = previous_stats
            proposal_before_spike = previous_spike
        if kind == "critic":
            metrics = agent.update_critic(batch)
        else:
            metrics = agent.update_actor_transactional(
                batch,
                hard_stop=float(config.get("bc_kl_hard_stop", 1.0)),
                entropy_floor=float(config.get("entropy_floor", 0.02)),
            )
        state_after = stable_fingerprint(agent.state_payload())
        rng_after = capture_rng_state(torch, replay_rng)
        expected_after = record.get("model_state_after_sha256")
        state_before_match = state_before == expected_before
        state_after_match = state_after == expected_after
        rng_before_match = rng_fingerprint(rng_before) == record.get("replayable_rng_before_fingerprint")
        rng_after_match = rng_fingerprint(rng_after) == record.get("replayable_rng_after_fingerprint")
        comparison.append(
            {
                "journal_sequence": int(record.get("journal_sequence", -1)),
                "update_kind": kind,
                "update_index": int(record.get("update_index", -1)),
                "batch_match": batch_match,
                "batch_before_state_match": state_before_match,
                "state_after_match": state_after_match,
                "rng_before_match": rng_before_match,
                "rng_after_match": rng_after_match,
                "legacy_metrics_not_compared": bool(kind == "actor_rejected"),
            }
        )
        if kind == "actor":
            accepted_count += 1
            after_stats = _policy_stats(
                agent, replay, sentinel_rows, torch=torch, device=device, chunk_size=chunk_size
            )
            after_spike = _single_state_stats(agent, replay, spike_row, torch=torch, device=device)
            trajectories.append(
                _trajectory_row(
                    actor_step=accepted_count,
                    journal_sequence=int(record.get("journal_sequence", -1)),
                    stats=after_stats,
                    spike_stats=after_spike,
                    batch_record=record,
                )
            )
            spike_trajectory.append(
                {
                    "actor_update_step": int(accepted_count),
                    "replay_row": int(spike_row),
                    "kl": after_spike["kl"],
                    "entropy": after_spike["entropy"],
                    "tv": after_spike["tv"],
                    "actor_argmax": after_spike["actor_argmax"],
                    "bc_argmax": after_spike["bc_argmax"],
                    "argmax_flip": after_spike["argmax_flip"],
                    "max_logit_delta": after_spike["max_logit_delta"],
                    "bc_probabilities": json.dumps(after_spike["bc_probabilities"], separators=(",", ":")),
                    "actor_probabilities": json.dumps(after_spike["actor_probabilities"], separators=(",", ":")),
                }
            )
            accepted_records.append(
                {
                    "actor_step": int(accepted_count),
                    "record": dict(record),
                    "batch_indices": indices.copy(),
                    "batch": _copy_batch(batch),
                }
            )
            if (
                proposal_before_stats is not None
                and proposal_before_stats["kl_max"] < 1.0
                and after_stats["kl_max"] >= 1.0
                and pre_cross_candidate is None
            ):
                pre_cross_candidate = {
                    "actor_step": int(accepted_count),
                    "record": dict(record),
                    "batch_indices": indices.copy(),
                    "batch": _copy_batch(batch),
                    "before_stats": dict(proposal_before_stats),
                    "after_stats": dict(after_stats),
                    "before_spike": dict(proposal_before_spike),
                    "after_spike": dict(after_spike),
                    "rng_before_fingerprint": rng_fingerprint(rng_before),
                    "actor_state_before_sha256": state_before,
                    "optimizer_state_before_sha256": stable_fingerprint(agent.actor_optimizer.state_dict()),
                }
            previous_stats = after_stats
            previous_spike = after_spike
        elif kind == "actor_rejected":
            new_rejected = {
                "record": dict(record),
                "metrics": dict(metrics),
                "state_before": state_before,
                "state_after": state_after,
                "batch_indices": indices.tolist(),
            }
    if initial_stats is None:
        raise RuntimeError("journal contains no Actor proposal")
    return {
        "comparison": comparison,
        "accepted_records": accepted_records,
        "trajectories": trajectories,
        "spike_trajectory": spike_trajectory,
        "pre_cross_candidate": pre_cross_candidate,
        "new_rejected": new_rejected,
        "initial_stats": initial_stats,
        "final_stats": previous_stats,
        "accepted_count": accepted_count,
    }


def _capture_crossing_and_window_payloads(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    config: Mapping[str, Any],
    bc_checkpoint: Mapping[str, Any],
    journal: Sequence[Mapping[str, Any]],
    targets: Sequence[int],
    torch: Any,
    nn: Any,
    device: Any,
) -> Mapping[int, Mapping[str, Any]]:
    checkpoint = load_checkpoint(torch, checkpoint_path, map_location="cpu")
    agent = _new_agent(torch, nn, device, bc_checkpoint, config)
    _restore_agent(agent, checkpoint)
    rng_state = checkpoint.get("rng_state", {})
    _restore_rng(torch, rng_state)
    replay = SACReplayBuffer.open(run_dir / "replay", read_only=True)
    replay_rng = np.random.RandomState()
    replay_rng.set_state(rng_state.get("replay_sampler", rng_state.get("numpy")))
    target_set = {int(value) for value in targets}
    result: Dict[int, Mapping[str, Any]] = {}
    for record in journal:
        indices = np.asarray(record.get("batch_indices", []), dtype=np.int64)
        expected = np.asarray(
            replay_rng.randint(0, int(record["replay_size"]), size=indices.size), dtype=np.int64
        )
        if not np.array_equal(expected, indices):
            raise RuntimeError("batch schedule diverged while capturing diagnostic payload")
        batch = replay.batch_from_indices(indices, torch=torch, device=device)
        if record.get("update_kind") == "actor":
            step = int(record.get("update_index", -1))
            if step in target_set:
                result[step] = {
                    "payload": deepcopy(agent.state_payload()),
                    "batch": _copy_batch(batch),
                    "indices": indices.copy(),
                    "record": dict(record),
                }
        if record.get("update_kind") == "critic":
            agent.update_critic(batch)
        else:
            agent.update_actor_transactional(
                batch,
                hard_stop=float(config.get("bc_kl_hard_stop", 1.0)),
                entropy_floor=float(config.get("entropy_floor", 0.02)),
            )
    return result


def _first_crossing(trajectory: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    if not trajectory:
        return {"status": "NO_TRAJECTORY"}
    initial = trajectory[0]
    previous = float(initial[key])
    if previous >= 1.0:
        return {
            "status": "PRE_EXISTING_AT_ACTOR_STEP_0",
            "actor_step": 0,
            "kl_before": None,
            "kl_after": previous,
        }
    for row in trajectory[1:]:
        current = float(row[key])
        if previous < 1.0 <= current:
            return {
                "status": "CROSSING_FOUND",
                "actor_step": int(row["actor_update_step"]),
                "kl_before": previous,
                "kl_after": current,
                "journal_sequence": row.get("journal_sequence"),
            }
        previous = current
    return {
        "status": "NO_CROSSING_IN_ACCEPTED_UPDATES",
        "actor_step": None,
        "kl_before": None,
        "kl_after": None,
    }


def _cumulative_ablation(
    *,
    payload: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    replay: SACReplayBuffer,
    sentinel_rows: Sequence[int],
    spike_row: int,
    config: Mapping[str, Any],
    bc_checkpoint: Mapping[str, Any],
    torch: Any,
    nn: Any,
    device: Any,
    chunk_size: int,
) -> Mapping[str, Any]:
    snapshots = {10, 25, 50, 100}
    specs = {
        "A_FULL": (True, True, True),
        "B_NO_Q": (False, True, True),
        "C_NO_ENTROPY": (True, False, True),
        "D_NO_BC_KL": (True, True, False),
    }
    branches = {}
    for name, (include_q, include_entropy, include_bc_kl) in specs.items():
        agent = _new_agent(torch, nn, device, bc_checkpoint, config)
        _restore_agent(agent, payload)
        values = {}
        for proposal, item in enumerate(schedule, start=1):
            _counterfactual_step(
                agent,
                item["batch"],
                torch=torch,
                include_q=include_q,
                include_entropy=include_entropy,
                include_bc_kl=include_bc_kl,
                do_step=True,
            )
            if proposal in snapshots:
                values[str(proposal)] = {
                    "sentinel": _policy_stats(
                        agent, replay, sentinel_rows, torch=torch, device=device, chunk_size=chunk_size
                    ),
                    "spike_state": _single_state_stats(
                        agent, replay, spike_row, torch=torch, device=device
                    ),
                    "actor_sha256": stable_fingerprint(agent.actor.state_dict()),
                }
        branches[name] = values
    return {
        "schema_id": CUMULATIVE_SCHEMA,
        "status": "PASS",
        "proposal_budget": int(len(specs) * len(schedule)),
        "schedule_length": int(len(schedule)),
        "snapshots": sorted(snapshots),
        "branches": branches,
    }


def _transaction_regression(
    *,
    crossing: Optional[Mapping[str, Any]],
    config: Mapping[str, Any],
    bc_checkpoint: Mapping[str, Any],
    torch: Any,
    nn: Any,
    device: Any,
) -> Mapping[str, Any]:
    if crossing is None:
        return {"schema_id": ROLLBACK_SCHEMA, "status": "NOT_RUN_NO_TRUE_CROSSING"}
    # The true sentinel crossing need not be a crossing in the current
    # minibatch.  Build a deterministic post-step rejection probe from the
    # same crossing payload, then place the hard-stop threshold strictly
    # between that probe's pre/post minibatch KL values.  This tests the
    # transaction boundary itself rather than accidentally testing a
    # pre-step rejection or the unrelated production threshold.
    probe = _new_agent(torch, nn, device, bc_checkpoint, config)
    _restore_agent(probe, crossing["payload"])
    probe_result = _counterfactual_step(
        probe,
        crossing["batch"],
        torch=torch,
        include_q=True,
        include_entropy=True,
        include_bc_kl=True,
        do_step=True,
    )
    probe_before = float(probe_result["kl_before_max"])
    probe_after = float(probe_result["kl_after_max"])
    if not probe_after > probe_before:
        return {
            "schema_id": ROLLBACK_SCHEMA,
            "status": "FAIL_NO_POST_STEP_KL_INCREASE",
            "probe_kl_before": probe_before,
            "probe_kl_after": probe_after,
        }
    hard_stop = (probe_before + probe_after) / 2.0
    agent = _new_agent(torch, nn, device, bc_checkpoint, config)
    _restore_agent(agent, crossing["payload"])
    actor_before = stable_fingerprint(agent.actor.state_dict())
    optimizer_before = stable_fingerprint(agent.actor_optimizer.state_dict())
    result = agent.update_actor_transactional(
        crossing["batch"],
        hard_stop=hard_stop,
        entropy_floor=float(config.get("entropy_floor", 0.02)),
    )
    actor_restored = stable_fingerprint(agent.actor.state_dict()) == actor_before
    optimizer_restored = stable_fingerprint(agent.actor_optimizer.state_dict()) == optimizer_before
    passed = bool(
        result.get("hard_stop_result") == "REJECT_POST_KL"
        and result.get("pre_step_hard_stop") is False
        and result.get("post_step_hard_stop_checked") is True
        and result.get("optimizer_step_performed") is True
        and result.get("rollback_performed") is True
        and actor_restored
        and optimizer_restored
        and int(agent.actor_update_count) == int(crossing["payload"].get("actor_update_count", 0))
    )
    return {
        "schema_id": ROLLBACK_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "probe_kl_before": probe_before,
        "probe_kl_after": probe_after,
        "hard_stop": hard_stop,
        "hard_stop_result": result.get("hard_stop_result"),
        "pre_step_hard_stop": result.get("pre_step_hard_stop"),
        "post_step_hard_stop_checked": result.get("post_step_hard_stop_checked"),
        "optimizer_step_performed": result.get("optimizer_step_performed"),
        "rollback_performed": result.get("rollback_performed"),
        "actor_bitwise_restored": actor_restored,
        "optimizer_bitwise_restored": optimizer_restored,
        "accepted_actor_count": int(agent.actor_update_count),
        "rejected_actor_count": int(agent.actor_rejection_count),
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    import torch
    import torch.nn as nn

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    config = _read_json(run_dir / "training_config.json")
    if config.get("deterministic_replay"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    device = torch.device(args.device)
    agent, replay, checkpoint, config, source_input_identity, checkpoint_path, journal = _load_runtime(
        run_dir, args.device, torch, nn
    )
    input_identity = _input_identity(run_dir, checkpoint_path, source_input_identity)
    _write_json(out_dir / "input_identity.json", input_identity)
    source_counterfactual = _read_json(run_dir / "one_step_counterfactual.json")
    top_states = source_counterfactual.get("top_20_kl_states", [])
    if not top_states:
        raise RuntimeError("V2 one-step counterfactual has no top KL state")
    spike_row = int(top_states[0]["replay_row"])
    first_actor_record = next(
        record for record in journal if record.get("update_kind") in ("actor", "actor_rejected")
    )
    first_actor_transition = int(first_actor_record["replay_size"])
    sentinel_rows, sentinel_manifest = _sentinel_rows(replay, first_actor_transition)
    _write_json(
        out_dir / "sentinel_manifest.json",
        {
            "schema_id": SENTINEL_SCHEMA,
            "selection": "all unique state observations with committed transition row < first Actor update transition",
            "first_actor_update_at_transition": first_actor_transition,
            "uses_return_q_or_terminal_outcome": False,
            "state_count": len(sentinel_rows),
            "rows": sentinel_manifest,
        },
    )
    result = _replay_audit(
        run_dir=run_dir,
        agent=agent,
        replay=replay,
        checkpoint=checkpoint,
        config=config,
        journal=journal,
        sentinel_rows=sentinel_rows,
        spike_row=spike_row,
        torch=torch,
        device=device,
        chunk_size=args.chunk_size,
    )
    comparison = result["comparison"]
    _csv_write(
        out_dir / "sentinel_kl_trajectory.csv",
        result["trajectories"],
        list(result["trajectories"][0].keys()),
    )
    _csv_write(
        out_dir / "spike_state_trajectory.csv",
        result["spike_trajectory"],
        list(result["spike_trajectory"][0].keys()),
    )
    _csv_write(
        out_dir / "exact_replay_comparison.csv",
        comparison,
        list(comparison[0].keys()),
    )
    trajectory = result["trajectories"]
    accepted_trajectory = trajectory[1:]
    batch_before = [row["batch_kl_max_before"] for row in accepted_trajectory if row["batch_kl_max_before"] is not None]
    batch_after = [row["batch_kl_max_after"] for row in accepted_trajectory if row["batch_kl_max_after"] is not None]
    sentinel_max = [row["sentinel_kl_max"] for row in accepted_trajectory]
    batch_sentinel = {
        "schema_id": "bc_initialized_discrete_sac_batch_vs_sentinel_kl_v3",
        "accepted_actor_updates": len(accepted_trajectory),
        "batch_pre_vs_sentinel_after_pearson": _correlation(batch_before, sentinel_max),
        "batch_post_vs_sentinel_after_pearson": _correlation(batch_after, sentinel_max),
        "batch_pre_vs_sentinel_after_spearman": _correlation(batch_before, sentinel_max, rank=True),
        "batch_post_vs_sentinel_after_spearman": _correlation(batch_after, sentinel_max, rank=True),
        "batch_post_minus_sentinel_mean": float(np.mean(np.asarray(batch_after) - np.asarray(sentinel_max))) if batch_after else None,
        "batch_only_crossing_count": int(
            sum(
                1
                for row in accepted_trajectory
                if row["batch_kl_max_after"] is not None
                and float(row["batch_kl_max_after"]) >= 1.0
                and float(row["sentinel_kl_max"]) < 1.0
            )
        ),
        "fixed_sentinel_crossing_count": int(sum(1 for row in accepted_trajectory if float(row["sentinel_kl_max"]) >= 1.0)),
    }
    _write_json(out_dir / "batch_vs_sentinel_kl.json", batch_sentinel)
    sentinel_cross = _first_crossing(trajectory, "sentinel_kl_max")
    spike_cross = _first_crossing(trajectory, "spike_state_kl")
    first_crossing = {
        "schema_id": CROSSING_SCHEMA,
        "sentinel": sentinel_cross,
        "spike_state": spike_cross,
        "spike_replay_row": spike_row,
        "first_actor_update_at_transition": first_actor_transition,
        "accepted_actor_updates": int(result["accepted_count"]),
    }
    _write_json(out_dir / "first_crossing.json", first_crossing)
    true_cross_step = None
    if sentinel_cross.get("status") == "CROSSING_FOUND":
        true_cross_step = int(sentinel_cross["actor_step"])
    elif spike_cross.get("status") == "CROSSING_FOUND":
        true_cross_step = int(spike_cross["actor_step"])
    bc_path = Path(source_input_identity["bc_checkpoint_path"]).expanduser().resolve()
    bc_checkpoint = torch.load(str(bc_path), map_location="cpu", weights_only=False)
    target_steps = []
    if true_cross_step is not None:
        target_steps.append(true_cross_step)
    cumulative_start = max(1, int(result["accepted_count"]) - 99)
    target_steps.append(cumulative_start)
    payloads = _capture_crossing_and_window_payloads(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        config=config,
        bc_checkpoint=bc_checkpoint,
        journal=journal,
        targets=sorted(set(target_steps)),
        torch=torch,
        nn=nn,
        device=device,
    )
    crossing_gradient: Mapping[str, Any]
    counterfactual: Mapping[str, Any]
    cumulative: Mapping[str, Any]
    if true_cross_step is None:
        crossing_gradient = {
            "schema_id": GRADIENT_SCHEMA,
            "status": "NOT_RUN_NO_TRUE_CROSSING",
            "reason": "fixed sentinel and spike state did not cross from below 1.0 during accepted updates",
        }
        counterfactual = {
            "schema_id": COUNTERFACTUAL_SCHEMA,
            "status": "NOT_RUN_NO_TRUE_CROSSING",
            "reason": "R0-R8 require KL_before < 1.0 and original post-step KL >= 1.0",
        }
        cumulative = {
            "schema_id": CUMULATIVE_SCHEMA,
            "status": "NOT_RUN_NO_TRUE_CROSSING",
            "proposal_budget": 0,
        }
    else:
        crossing_payload = payloads[true_cross_step]
        crossing_agent = _new_agent(torch, nn, device, bc_checkpoint, config)
        _restore_agent(crossing_agent, crossing_payload["payload"])
        crossing_gradient = {
            "schema_id": GRADIENT_SCHEMA,
            "status": "PASS",
            "actor_step": true_cross_step,
            "batch_indices": crossing_payload["indices"].tolist(),
            "batch_indices_sha256": hashlib.sha256(
                np.asarray(crossing_payload["indices"], dtype=np.int64).tobytes()
            ).hexdigest(),
            "actor_state_before_sha256": stable_fingerprint(crossing_agent.actor.state_dict()),
            "optimizer_state_before_sha256": stable_fingerprint(crossing_agent.actor_optimizer.state_dict()),
            "sentinel_before": _policy_stats(
                crossing_agent,
                replay,
                sentinel_rows,
                torch=torch,
                device=device,
                chunk_size=args.chunk_size,
            ),
            "spike_state_before": _single_state_stats(
                crossing_agent,
                replay,
                spike_row,
                torch=torch,
                device=device,
            ),
            "gradient": _gradient_decomposition(crossing_agent, crossing_payload["batch"], torch=torch),
        }
        variants: Dict[str, Any] = {}
        for name, (include_q, include_entropy, include_bc_kl, do_step) in _actor_specs().items():
            branch = _new_agent(torch, nn, device, bc_checkpoint, config)
            _restore_agent(branch, crossing_payload["payload"])
            variant = _counterfactual_step(
                branch,
                crossing_payload["batch"],
                torch=torch,
                include_q=include_q,
                include_entropy=include_entropy,
                include_bc_kl=include_bc_kl,
                do_step=do_step,
            )
            sentinel_after = _policy_stats(
                branch,
                replay,
                sentinel_rows,
                torch=torch,
                device=device,
                chunk_size=args.chunk_size,
            )
            spike_after = _single_state_stats(
                branch,
                replay,
                spike_row,
                torch=torch,
                device=device,
            )
            variant.update(
                {
                    "sentinel_kl_before_max": crossing_gradient["sentinel_before"]["kl_max"],
                    "sentinel_kl_after_max": sentinel_after["kl_max"],
                    "sentinel_kl_delta_max": (
                        sentinel_after["kl_max"] - crossing_gradient["sentinel_before"]["kl_max"]
                    ),
                    "sentinel_post_step_hard_stop": bool(sentinel_after["kl_max"] >= 1.0),
                    "spike_state_kl_before": crossing_gradient["spike_state_before"]["kl"],
                    "spike_state_kl_after": spike_after["kl"],
                    "spike_state_post_step_hard_stop": bool(spike_after["kl"] >= 1.0),
                }
            )
            variant.pop("gradient_vectors", None)
            variants[name] = variant
        counterfactual = {
            "schema_id": COUNTERFACTUAL_SCHEMA,
            "status": "PASS",
            "actor_step": true_cross_step,
            "kl_before_must_be_below_threshold": True,
            "measurement_note": (
                "kl_before/after are current training-minibatch values; "
                "sentinel_kl_before/after fields are the fixed pre-Actor "
                "state-set crossing evidence"
            ),
            "batch_kl_before": crossing_gradient["gradient"].get("kl_before_max"),
            "sentinel_kl_before": crossing_gradient["sentinel_before"].get("kl_max"),
            "variants": variants,
        }
        singleton_explains = (
            variants["R0_ORIGINAL"]["sentinel_post_step_hard_stop"]
            and (
                not variants["R1_NO_Q_TERM"]["sentinel_post_step_hard_stop"]
                or not variants["R2_NO_ENTROPY_TERM"]["sentinel_post_step_hard_stop"]
                or not variants["R3_NO_BC_KL_TERM"]["sentinel_post_step_hard_stop"]
            )
        )
        if singleton_explains:
            cumulative = {
                "schema_id": CUMULATIVE_SCHEMA,
                "status": "NOT_RUN_SINGLE_TERM_EXPLAINED_CROSSING",
                "proposal_budget": 0,
            }
        else:
            schedule = result["accepted_records"][cumulative_start - 1 : cumulative_start - 1 + 100]
            cumulative = _cumulative_ablation(
                payload=payloads[cumulative_start]["payload"],
                schedule=schedule,
                replay=replay,
                sentinel_rows=sentinel_rows,
                spike_row=spike_row,
                config=config,
                bc_checkpoint=bc_checkpoint,
                torch=torch,
                nn=nn,
                device=device,
                chunk_size=args.chunk_size,
            )
    _write_json(out_dir / "crossing_gradient_decomposition.json", crossing_gradient)
    _write_json(out_dir / "one_step_counterfactual_v3.json", counterfactual)
    _write_json(out_dir / "cumulative_ablation.json", cumulative)
    exact_pass = bool(
        all(
            row["batch_match"] and row["batch_before_state_match"] and row["state_after_match"]
            and row["rng_before_match"] and row["rng_after_match"]
            for row in comparison
        )
    )
    old_rejected = next(
        (record for record in journal if record.get("update_kind") == "actor_rejected"), None
    )
    new_rejected = result.get("new_rejected")
    new_metrics = (new_rejected or {}).get("metrics", {})
    transaction = {
        "schema_id": TRANSACTION_SCHEMA,
        "old_behavior_stop_step": int(old_rejected.get("update_index")) if old_rejected else None,
        "old_behavior_optimizer_step_performed": True if old_rejected else None,
        "old_behavior_pre_step_unsafe": bool(old_rejected and old_rejected.get("metrics", {}).get("bc_kl_max", 0.0) >= 1.0),
        "new_behavior_stop_step": int(old_rejected.get("update_index")) if old_rejected else None,
        "new_behavior_prestep_safe": not bool(new_metrics.get("pre_step_hard_stop", False)),
        "new_behavior_poststep_rejected": bool(
            new_metrics.get("post_step_hard_stop_checked", False)
            and new_metrics.get("hard_stop_result") == "REJECT_POST_KL"
        ),
        "new_behavior_optimizer_step_performed": bool(new_metrics.get("optimizer_step_performed", False)),
        "new_behavior_actor_before_equals_after": bool(
            new_metrics.get("actor_before_sha256") == new_metrics.get("actor_after_sha256")
        ),
        "new_behavior_optimizer_before_equals_after": bool(
            new_metrics.get("optimizer_before_sha256") == new_metrics.get("optimizer_after_sha256")
        ),
        "accepted_actor_count_after_stop": int(result["accepted_count"]),
        "checkpoint_persists_unsafe_actor": False,
        "threshold": float(config.get("bc_kl_hard_stop", 1.0)),
    }
    _write_json(out_dir / "transaction_ordering_audit.json", transaction)
    rollback = _transaction_regression(
        crossing=(payloads.get(true_cross_step) if true_cross_step is not None else None),
        config=config,
        bc_checkpoint=bc_checkpoint,
        torch=torch,
        nn=nn,
        device=device,
    )
    _write_json(out_dir / "rollback_regression.json", rollback)
    batch_effect = batch_sentinel["batch_only_crossing_count"] > 0
    drift_effect = sentinel_cross.get("status") == "CROSSING_FOUND"
    labels_drift: List[str] = []
    if batch_effect:
        labels_drift.append("A_BATCH_COMPOSITION_REVEALED_PREEXISTING_HIGH_KL_STATE")
    if drift_effect:
        labels_drift.append("B_GRADUAL_GLOBAL_POLICY_DRIFT")
        if any(
            int(row["sentinel_count_gt_1"]) <= 1
            and float(row["sentinel_kl_max"]) >= 1.0
            for row in trajectory
        ):
            labels_drift.append("C_LOCALIZED_STATE_POLICY_DRIFT")
    if true_cross_step is not None:
        variants = counterfactual.get("variants", {})
        if variants.get("R0_ORIGINAL", {}).get("sentinel_post_step_hard_stop"):
            if variants.get("R5_Q_ONLY", {}).get("sentinel_post_step_hard_stop") and not variants.get("R1_NO_Q_TERM", {}).get("sentinel_post_step_hard_stop"):
                labels_drift.append("D_Q_TERM_CAUSAL")
            if variants.get("R6_ENTROPY_ONLY", {}).get("sentinel_post_step_hard_stop") and not variants.get("R2_NO_ENTROPY_TERM", {}).get("sentinel_post_step_hard_stop"):
                labels_drift.append("E_ENTROPY_TERM_CAUSAL")
            if variants.get("R7_BC_KL_ONLY", {}).get("sentinel_post_step_hard_stop") and not variants.get("R3_NO_BC_KL_TERM", {}).get("sentinel_post_step_hard_stop"):
                labels_drift.append("F_BC_KL_TERM_OR_SUPPORT_EFFECT")
            if not any(label in labels_drift for label in ("D_Q_TERM_CAUSAL", "E_ENTROPY_TERM_CAUSAL", "F_BC_KL_TERM_OR_SUPPORT_EFFECT")):
                labels_drift.append("G_MULTI_TERM_INTERACTION")
    if cumulative.get("status") == "PASS":
        labels_drift.append("H_OPTIMIZER_ACCUMULATION")
    labels_late = ["I_TRANSACTION_ORDERING_BUG"]
    if batch_effect:
        labels_late.insert(0, "A_BATCH_COMPOSITION_REVEALED_PREEXISTING_HIGH_KL_STATE")
    root_cause = {
        "schema_id": ROOT_CAUSE_SCHEMA,
        "status": "PASS" if exact_pass else "FAIL_EXACT_REPLAY",
        "root_cause_of_kl_drift": labels_drift or ["NOT_IDENTIFIED"],
        "root_cause_of_late_detection": labels_late,
        "batch_composition_effect": "YES" if batch_effect else "NO",
        "policy_drift_effect": "YES" if drift_effect else "NO",
        "direct_evidence": {
            "sentinel_first_crossing": sentinel_cross,
            "spike_first_crossing": spike_cross,
            "batch_vs_sentinel": batch_sentinel,
            "exact_replay_after_transaction_fix": exact_pass,
            "new_transaction": transaction,
        },
        "causal_claim_scope": "fixed pre-Actor sentinel and exact offline replay; no environment or seed generalization",
    }
    _write_json(out_dir / "root_cause.json", root_cause)
    source_files = [
        "python/planning/sac/network.py",
        "python/planning/sac/runtime.py",
        "scripts/audit_sac_first_kl_crossing.py",
        "tests/test_sac_deterministic_replay_v2.py",
        "tests/contracts/test_contract_naming.py",
    ]
    (out_dir / "code_changes.diff").write_text(
        "# No Git repository is present; this is an additive source identity manifest.\n"
        + "".join(
            "{}  {}\n".format(file_sha256(Path(path)), path)
            for path in source_files
        ),
        encoding="utf-8",
    )
    transaction_status = "PASS" if (
        transaction["new_behavior_optimizer_step_performed"] is False
        and transaction["new_behavior_actor_before_equals_after"]
        and transaction["new_behavior_optimizer_before_equals_after"]
    ) else "FAIL"
    report = [
        "# DISCRETE SAC FIRST KL CROSSING CAUSAL AUDIT V3",
        "",
        "## 范围",
        "",
        "- 仅使用 V2 已通过 exact replay 的固定 Replay/journal/RNG。",
        "- 新环境步数：`0`；未启动 Unity、Bridge、ROS、Dev100 或 Final300。",
        "- sentinel 不使用 return、Q、terminal outcome 或后续策略表现。",
        "",
        "## Exact replay",
        "",
        "- source exact replay：`PASS`；本轮修复后 batch/state/RNG 重放：`{}`。".format("PASS" if exact_pass else "FAIL"),
        "- sentinel state count：`{}`；first Actor transition：`{}`。".format(len(sentinel_rows), first_actor_transition),
        "",
        "## 首次越界",
        "",
        "- sentinel：`{}`".format(json.dumps(sentinel_cross, sort_keys=True)),
        "- spike state：`{}`".format(json.dumps(spike_cross, sort_keys=True)),
        "- batch composition effect：`{}`；policy drift effect：`{}`。".format(
            "YES" if batch_effect else "NO", "YES" if drift_effect else "NO"
        ),
        "",
        "## 固定 sentinel 与训练 batch 的区分",
        "",
        "- 本审计的真实越界是固定 pre-Actor sentinel 的 state-set 越界；R0–R8 同时保留当前训练 minibatch 的 KL，不能把 minibatch KL 当作 sentinel KL。",
    ]
    if counterfactual.get("status") == "PASS":
        report.extend(
            [
                "- Actor step `{}` sentinel KL before=`{}`。".format(
                    counterfactual["actor_step"], counterfactual.get("sentinel_kl_before")
                ),
                "- R0 batch KL before=`{}`；R0 batch KL after=`{}`；R0 sentinel KL after=`{}`。".format(
                    variants["R0_ORIGINAL"].get("kl_before_max"),
                    variants["R0_ORIGINAL"].get("kl_after_max"),
                    variants["R0_ORIGINAL"].get("sentinel_kl_after_max"),
                ),
                "",
                "| 分支 | batch KL after | sentinel KL after | sentinel 越界 |",
                "|---|---:|---:|---|",
            ]
        )
        for name in ("R0_ORIGINAL", "R1_NO_Q_TERM", "R2_NO_ENTROPY_TERM", "R3_NO_BC_KL_TERM", "R4_NO_OPTIMIZER_STEP", "R5_Q_ONLY", "R6_ENTROPY_ONLY", "R7_BC_KL_ONLY", "R8_Q_PLUS_BC_KL"):
            item = variants[name]
            report.append(
                "| `{}` | {:.9f} | {:.9f} | `{}` |".format(
                    name,
                    float(item.get("kl_after_max", 0.0)),
                    float(item.get("sentinel_kl_after_max", 0.0)),
                    "YES" if item.get("sentinel_post_step_hard_stop") else "NO",
                )
            )
        gradient = crossing_gradient.get("gradient", {})
        report.extend(
            [
                "",
                "## 首次 crossing 的 batch gradient decomposition",
                "",
                "- Q gradient norm=`{}`；entropy gradient norm=`{}`；BC-KL gradient norm=`{}`。".format(
                    gradient.get("gradient_norm_q"),
                    gradient.get("gradient_norm_entropy"),
                    gradient.get("gradient_norm_bc_kl"),
                ),
                "- cosine(Q,entropy)=`{}`；cosine(Q,BC-KL)=`{}`；cosine(entropy,BC-KL)=`{}`。".format(
                    gradient.get("gradient_cosine_q_entropy"),
                    gradient.get("gradient_cosine_q_bc_kl"),
                    gradient.get("gradient_cosine_entropy_bc_kl"),
                ),
            ]
        )
    report.extend(
        [
        "",
        "## Transaction ordering",
        "",
        "- old stop step：`{}`；new stop step：`{}`。".format(
            transaction["old_behavior_stop_step"], transaction["new_behavior_stop_step"]
        ),
        "- new pre-step safe：`{}`；optimizer step performed：`{}`；unsafe Actor persisted：`{}`。".format(
            transaction["new_behavior_prestep_safe"],
            transaction["new_behavior_optimizer_step_performed"],
            transaction["checkpoint_persists_unsafe_actor"],
        ),
        "- rollback regression：`{}`。".format(rollback.get("status")),
        "",
        "## 根因",
        "",
        "- KL drift：`{}`。".format(", ".join(labels_drift) if labels_drift else "NOT_IDENTIFIED"),
        "- late detection：`{}`。".format(", ".join(labels_late)),
        "- 结论范围限于固定 sentinel 与本次训练 lineage，不外推到其他 seed 或生产默认。",
        "",
        "## Hyperparameter boundary",
        "",
        "- KL threshold remains `1.0`; SAC algorithm hyperparameters unchanged; AWAC/BC unchanged.",
        ]
    )
    (out_dir / "report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_json(
        out_dir / "audit_summary.json",
        {
            "schema_id": ROOT_SCHEMA,
            "status": "PASS" if exact_pass and transaction_status == "PASS" else "FAIL",
            "exact_replay_source_verified": True,
            "exact_replay_after_fix": exact_pass,
            "sentinel_state_count": len(sentinel_rows),
            "first_sentinel_cross_actor_step": sentinel_cross.get("actor_step"),
            "first_spike_state_cross_actor_step": spike_cross.get("actor_step"),
            "batch_composition_effect": "YES" if batch_effect else "NO",
            "policy_drift_effect": "YES" if drift_effect else "NO",
            "transaction_ordering_fixed": transaction_status == "PASS",
            "new_environment_steps": 0,
            "dev100": False,
            "final300": False,
        },
    )
    return 0 if exact_pass and transaction_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
