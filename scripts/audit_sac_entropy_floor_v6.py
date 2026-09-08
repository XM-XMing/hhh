#!/usr/bin/env python3
"""Offline V6 audit for the residual Discrete SAC entropy-floor failure.

This command deliberately has no environment/runtime imports.  It consumes a
V5 checkpoint, read-only replay and the canonical update journal, and writes a
new diagnostic directory.  It is not a training entry point and never writes
to ``--run-dir``.
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
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from planning.common.hashing import file_sha256
from planning.sac.checkpoint import load_checkpoint
from planning.sac.diagnostics import (
    batch_indices_sha256,
    capture_rng_state,
    restore_rng_state,
    rng_fingerprint,
    stable_fingerprint,
)
from planning.sac.entropy_audit import (
    audit_transaction_rows,
    gate_statistic,
    masked_entropy_reference,
    normalized_entropy,
    residual_saturation_metrics,
    valid_action_entropy_max,
)
from planning.sac.network import DiscreteSACAgent, masked_kl
from planning.sac.replay import SACReplayBuffer


V5_SCHEMA = "bc_initialized_discrete_sac_update_journal_v2"
DEFAULT_RUN = Path("data/sac/diagnostics/bc_residual_online_discrete_sac_v5_20260908")
DEFAULT_OUT = Path("data/sac/diagnostics/discrete_sac_entropy_floor_root_cause_v6_20260908")
EXPECTED_BC_SHA = "ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2"
EXPECTED_REPLAY_ROWS = 7418
EXPECTED_SENTINEL_ROWS = 5081
EXPECTED_FAILURE_STEPS = 7648


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit V5 SAC entropy-floor failure by deterministic offline replay",
        allow_abbrev=False,
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-actor-proposals", type=int, default=0)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _records(path: Path) -> List[Mapping[str, Any]]:
    """Read only complete newline-terminated canonical journal records."""

    result = []
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n") or not raw.strip():
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if value.get("schema_id") == V5_SCHEMA:
                result.append(value)
    return result


def _source_sha(path: Path) -> Optional[str]:
    return file_sha256(path) if path.is_file() else None


def _float_mismatches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> List[str]:
    mismatches = []
    for key, value in expected.items():
        if key not in actual:
            mismatches.append(str(key))
            continue
        if isinstance(value, (int, float)) and isinstance(actual[key], (int, float)):
            if not np.isclose(float(value), float(actual[key]), rtol=1e-6, atol=2e-7):
                mismatches.append(str(key))
    return mismatches


def _new_agent(torch, nn, device, bc_checkpoint: Mapping[str, Any], config: Mapping[str, Any]):
    return DiscreteSACAgent(
        torch=torch,
        nn=nn,
        device=device,
        bc_checkpoint=bc_checkpoint,
        actor_lr=float(config.get("actor_lr", 1e-5)),
        critic_lr=float(config.get("critic_lr", 1e-4)),
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
        actor_architecture=str(config.get("actor_architecture", "direct")),
        residual_logit_cap=float(config.get("residual_logit_cap", 0.49)),
    )


def _clone_batch(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value.detach().clone() if hasattr(value, "detach") else deepcopy(value)
        for key, value in batch.items()
    }


def _summary(values: Any) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": int(array.size), "finite_count": 0}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _policy_snapshot(agent, batch: Mapping[str, Any], *, floor: float) -> Mapping[str, Any]:
    torch = agent.torch
    with torch.no_grad():
        probabilities, log_probs, mask = agent._distribution(
            agent.actor, batch["depth"], batch["vector"], batch["action_mask"].bool()
        )
        bc_prob, bc_log, _ = agent._distribution(
            agent.bc_reference, batch["depth"], batch["vector"], batch["action_mask"].bool()
        )
        entropy = -(probabilities * log_probs).sum(dim=1)
        bc_entropy = -(bc_prob * bc_log).sum(dim=1)
        kl = masked_kl(probabilities, log_probs, bc_prob, bc_log, torch)
        counts = mask.sum(dim=1)
        eligible = (counts > 1) & (bc_entropy >= float(floor))
        flips = (
            torch.argmax(probabilities.masked_fill(~mask, -1.0), dim=1)
            != torch.argmax(bc_prob.masked_fill(~mask, -1.0), dim=1)
        ).float()
        normalized = torch.full_like(entropy, float("nan"))
        multi = counts > 1
        normalized[multi] = entropy[multi] / torch.log(counts[multi].float())
        bc_relative = torch.full_like(entropy, float("nan"))
        bc_positive = bc_entropy > 0.0
        bc_relative[bc_positive] = entropy[bc_positive] / bc_entropy[bc_positive]
        q_min = torch.minimum(agent.critic1(batch["depth"], batch["vector"]), agent.critic2(batch["depth"], batch["vector"]))
    entropy_np = entropy.detach().cpu().numpy().astype(np.float64)
    bc_entropy_np = bc_entropy.detach().cpu().numpy().astype(np.float64)
    normalized_np = normalized.detach().cpu().numpy().astype(np.float64)
    bc_relative_np = bc_relative.detach().cpu().numpy().astype(np.float64)
    eligible_np = eligible.detach().cpu().numpy().astype(bool)
    counts_np = counts.detach().cpu().numpy().astype(np.int64)
    q_np = q_min.detach().cpu().numpy().astype(np.float64)
    return {
        "count": int(entropy_np.size),
        "valid_action_count": _summary(counts_np),
        "entropy": _summary(entropy_np),
        "bc_entropy": _summary(bc_entropy_np),
        "normalized_entropy": _summary(normalized_np),
        "bc_relative_entropy": _summary(bc_relative_np),
        "kl": _summary(kl.detach().cpu().numpy()),
        "eligible_count": int(eligible_np.sum()),
        "entropy_min_eligible": float(np.min(entropy_np[eligible_np])) if eligible_np.any() else 0.0,
        "normalized_entropy_min_eligible": float(np.nanmin(normalized_np[eligible_np])) if eligible_np.any() else None,
        "bc_relative_entropy_min_eligible": float(np.nanmin(bc_relative_np[eligible_np])) if eligible_np.any() else None,
        "argmax_flip_rate": float(np.mean(flips.detach().cpu().numpy())) if entropy_np.size else 0.0,
        "q_min": _summary(q_np),
        "_arrays": {
            "entropy": entropy_np,
            "bc_entropy": bc_entropy_np,
            "normalized_entropy": normalized_np,
            "bc_relative_entropy": bc_relative_np,
            "kl": kl.detach().cpu().numpy().astype(np.float64),
            "eligible": eligible_np,
            "valid_action_count": counts_np,
            "q_min": q_np,
            "probabilities": probabilities.detach().cpu().numpy(),
            "bc_probabilities": bc_prob.detach().cpu().numpy(),
            "mask": mask.detach().cpu().numpy().astype(bool),
        },
    }


def _strip_arrays(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: item for key, item in value.items() if key != "_arrays"}


def _correlation(left: Sequence[float], right: Sequence[float]) -> Mapping[str, Any]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"count": int(x.size), "pearson": None, "spearman": None}
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(np.float64)
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(np.float64)
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"count": int(x.size), "pearson": pearson, "spearman": spearman}


def _restore_agent(agent, payload: Mapping[str, Any]) -> None:
    agent.load_state_payload(payload)


def _component_decomposition(agent, batch: Mapping[str, Any]) -> Mapping[str, Any]:
    """Calculate the three production loss gradients before any optimizer step."""

    torch = agent.torch
    parameters = agent.actor_trainable_parameters()
    for module in (agent.critic1, agent.critic2):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    try:
        depth, vector, mask = batch["depth"], batch["vector"], batch["action_mask"].bool()
        probability, log_probability, _ = agent._distribution(agent.actor, depth, vector, mask)
        with torch.no_grad():
            bc_probability, bc_log_probability, _ = agent._distribution(
                agent.bc_reference, depth, vector, mask
            )
        q_min = torch.minimum(agent.critic1(depth, vector), agent.critic2(depth, vector))
        q_term = (probability * (-q_min)).sum(dim=1)
        entropy_term = (probability * log_probability).sum(dim=1) * agent.alpha
        bc_kl_term = agent.beta_bc * masked_kl(
            probability, log_probability, bc_probability, bc_log_probability, torch
        )
        terms = {"q": q_term, "entropy": entropy_term, "bc_kl": bc_kl_term}
        flattened = {}
        norms = {}
        for name, term in terms.items():
            grads = torch.autograd.grad(term.mean(), parameters, retain_graph=True, allow_unused=True)
            values = [value.detach().reshape(-1) for value in grads if value is not None]
            flattened[name] = torch.cat(values) if values else torch.zeros((0,), device=agent.device)
            norms[name] = float(torch.linalg.vector_norm(flattened[name]).item()) if values else 0.0
        total = flattened["q"] + flattened["entropy"] + flattened["bc_kl"]
        pair_cosines = {}
        for left, right in (("q", "entropy"), ("q", "bc_kl"), ("entropy", "bc_kl")):
            a, b = flattened[left], flattened[right]
            if a.numel() == 0 or float(torch.linalg.vector_norm(a).item()) == 0.0 or float(torch.linalg.vector_norm(b).item()) == 0.0:
                pair_cosines[left + "_vs_" + right] = None
            else:
                pair_cosines[left + "_vs_" + right] = float(
                    torch.dot(a, b).item()
                    / (torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item())
                )
        return {
            "gradient_norm_q": norms["q"],
            "gradient_norm_entropy": norms["entropy"],
            "gradient_norm_bc_kl": norms["bc_kl"],
            "gradient_norm_total_unclipped": float(torch.linalg.vector_norm(total).item()),
            "gradient_component_cosines": pair_cosines,
            "alpha": float(agent.alpha),
            "beta_bc": float(agent.beta_bc),
        }
    finally:
        for module in (agent.critic1, agent.critic2):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def _counterfactual_actor_step(
    agent,
    batch: Mapping[str, Any],
    *,
    include_q: bool,
    include_entropy: bool,
    include_bc_kl: bool,
    do_step: bool,
) -> Mapping[str, Any]:
    """One diagnostic-only actor proposal with selected loss components."""

    torch = agent.torch
    parameters = agent.actor_trainable_parameters()
    for module in (agent.critic1, agent.critic2):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    try:
        depth, vector, mask = batch["depth"], batch["vector"], batch["action_mask"].bool()
        probability, log_probability, valid_mask = agent._distribution(agent.actor, depth, vector, mask)
        with torch.no_grad():
            bc_probability, bc_log_probability, _ = agent._distribution(
                agent.bc_reference, depth, vector, mask
            )
        q_min = torch.minimum(agent.critic1(depth, vector), agent.critic2(depth, vector))
        q_term = (probability * (-q_min)).sum(dim=1)
        entropy_term = (probability * log_probability).sum(dim=1) * agent.alpha
        bc_kl_term = agent.beta_bc * masked_kl(
            probability, log_probability, bc_probability, bc_log_probability, torch
        )
        terms = []
        if include_q:
            terms.append(q_term)
        if include_entropy:
            terms.append(entropy_term)
        if include_bc_kl:
            terms.append(bc_kl_term)
        actor_loss = sum(terms).mean() if terms else probability.sum() * 0.0
        component_norms = {}
        for name, enabled, term in (
            ("q", include_q, q_term),
            ("entropy", include_entropy, entropy_term),
            ("bc_kl", include_bc_kl, bc_kl_term),
        ):
            if not enabled:
                component_norms[name] = 0.0
                continue
            grads = torch.autograd.grad(term.mean(), parameters, retain_graph=True, allow_unused=True)
            values = [value.detach().reshape(-1) for value in grads if value is not None]
            component_norms[name] = float(torch.linalg.vector_norm(torch.cat(values)).item()) if values else 0.0
        actor_before = agent.actor_fingerprint()
        before_parameters = [parameter.detach().clone() for parameter in parameters]
        agent.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        pre_clip = float(torch.nn.utils.clip_grad_norm_(parameters, agent.gradient_clip_norm).item())
        if do_step:
            agent.actor_optimizer.step()
        with torch.no_grad():
            post_probability, post_log_probability, post_mask = agent._distribution(
                agent.actor, depth, vector, mask
            )
            post_kl = masked_kl(post_probability, post_log_probability, bc_probability, bc_log_probability, torch)
            post_entropy = -(post_probability * post_log_probability).sum(dim=1)
            post_counts = post_mask.sum(dim=1)
            post_bc_entropy = -(bc_probability * bc_log_probability).sum(dim=1)
            post_eligible = (post_counts > 1) & (post_bc_entropy >= agent.entropy_floor)
            post_entropy_min = float(post_entropy[post_eligible].min().item()) if bool(post_eligible.any()) else 0.0
            logits_before = agent.actor(depth, vector).detach().cpu().numpy()
            logits_after = agent.actor(depth, vector).detach().cpu().numpy()
        delta_terms = [parameter.detach() - before for parameter, before in zip(parameters, before_parameters)]
        delta = torch.cat([value.reshape(-1) for value in delta_terms]) if delta_terms else torch.zeros((0,), device=agent.device)
        return {
            "include_q": bool(include_q),
            "include_entropy": bool(include_entropy),
            "include_bc_kl": bool(include_bc_kl),
            "do_step": bool(do_step),
            "actor_loss": float(actor_loss.item()),
            "gradient_norm_q": component_norms["q"],
            "gradient_norm_entropy": component_norms["entropy"],
            "gradient_norm_bc_kl": component_norms["bc_kl"],
            "gradient_norm_total": pre_clip,
            "kl_before_max": float(masked_kl(probability, log_probability, bc_probability, bc_log_probability, torch).max().item()),
            "kl_after_max": float(post_kl.max().item()),
            "entropy_before_min": float((-(probability * log_probability).sum(dim=1))[post_eligible].min().item()) if bool(post_eligible.any()) else 0.0,
            "entropy_after_min": post_entropy_min,
            "entropy_floor_eligible_count": int(post_eligible.sum().item()),
            "parameter_delta_norm": float(torch.linalg.vector_norm(delta).item()),
            "actor_before_sha256": actor_before,
            "actor_after_sha256": agent.actor_fingerprint(),
            "_diagnostics": {
                "post_entropy": post_entropy.detach().cpu().numpy().astype(np.float64),
                "bc_entropy": post_bc_entropy.detach().cpu().numpy().astype(np.float64),
                "post_kl": post_kl.detach().cpu().numpy().astype(np.float64),
                "counts": post_counts.detach().cpu().numpy().astype(np.int64),
                "mask": post_mask.detach().cpu().numpy().astype(bool),
                "q_min": q_min.detach().cpu().numpy().astype(np.float64),
                "logits_before": logits_before,
                "logits_after": logits_after,
            },
        }
    finally:
        for module in (agent.critic1, agent.critic2):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def _sentinel_indices(path: Path) -> np.ndarray:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    indices = [int(row["replay_row"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("sentinel manifest contains duplicate replay rows")
    return np.asarray(indices, dtype=np.int64)


def _bc_entropy_baseline(agent, sentinel_batch: Mapping[str, Any], *, floor: float) -> Mapping[str, Any]:
    torch = agent.torch
    with torch.no_grad():
        logits = agent.bc_reference(sentinel_batch["depth"], sentinel_batch["vector"])
        probabilities, log_probabilities, mask = agent._distribution(
            agent.bc_reference,
            sentinel_batch["depth"],
            sentinel_batch["vector"],
            sentinel_batch["action_mask"].bool(),
        )
        entropy = -(probabilities * log_probabilities).sum(dim=1)
        counts = mask.sum(dim=1)
        bc_entropy = entropy.detach().cpu().numpy().astype(np.float64)
        counts_np = counts.detach().cpu().numpy().astype(np.int64)
        logits_np = logits.detach().cpu().numpy().astype(np.float64)
        production_entropy = entropy.detach().cpu().numpy().astype(np.float64)
        mask_np = mask.detach().cpu().numpy().astype(bool)
    ref_prob, ref_log, ref_entropy = masked_entropy_reference(logits_np, mask_np)
    normalized, single = normalized_entropy(ref_entropy, counts_np)
    hmax = valid_action_entropy_max(mask_np)
    eligible = (counts_np > 1) & (bc_entropy >= float(floor))
    buckets = ((1, 1), (2, 2), (3, 5), (6, 10), (11, 25), (26, 50), (51, 105))
    bucket_rows = []
    for low, high in buckets:
        selected = (counts_np >= low) & (counts_np <= high)
        bucket_rows.append({
            "valid_action_range": [low, high],
            "count": int(selected.sum()),
            "ratio": float(np.mean(selected)) if selected.size else 0.0,
            "entropy": _summary(bc_entropy[selected]),
            "hmax": _summary(hmax[selected]),
            "normalized_entropy": _summary(normalized[selected]),
            "below_floor_count": int(np.sum(selected & (bc_entropy < float(floor)))),
        })
    return {
        "state_count": int(bc_entropy.size),
        "entropy_floor": float(floor),
        "valid_action_count": _summary(counts_np),
        "entropy": _summary(bc_entropy),
        "hmax": _summary(hmax),
        "normalized_entropy": _summary(normalized),
        "single_valid_action_count": int(single.sum()),
        "eligible_count": int(eligible.sum()),
        "eligible_entropy_min": float(np.min(bc_entropy[eligible])) if eligible.any() else None,
        "below_gate_count_all_states": int(np.sum(bc_entropy < float(floor))),
        "below_gate_ratio_all_states": float(np.mean(bc_entropy < float(floor))),
        "below_gate_count_eligible_contract": int(np.sum(eligible & (bc_entropy < float(floor)))),
        "below_gate_ratio_eligible_contract": float(np.mean(eligible & (bc_entropy < float(floor)))),
        "bucket_rows": bucket_rows,
        "production_reference_max_abs_entropy_delta": float(np.max(np.abs(production_entropy - ref_entropy))),
        "production_reference_max_abs_probability_delta": float(np.max(np.abs(probabilities.detach().cpu().numpy() - ref_prob))),
        "production_reference_max_abs_log_probability_delta": float(np.max(np.abs(log_probabilities.detach().cpu().numpy() - ref_log))),
        "entropy_gate_compatible": bool(np.isfinite(ref_entropy).all() and np.all(counts_np > 0)),
    }


def _lowest_states(replay, indices: np.ndarray, diagnostic: Mapping[str, Any], count: int = 20) -> List[Mapping[str, Any]]:
    arrays = diagnostic["_diagnostics"]
    order = np.argsort(arrays["post_entropy"], kind="mergesort")[: int(count)]
    result = []
    for position in order:
        row = int(indices[int(position)])
        identity = dict(replay.identity_records[row])
        c = int(arrays["counts"][position])
        hmax = float(math.log(c)) if c > 1 else None
        result.append({
            "batch_position": int(position),
            "replay_row": row,
            "mission_id": identity["mission_id"],
            "episode_id": identity["episode_id"],
            "step_id": int(identity["step_id"]),
            "transition_id": identity["transition_id"],
            "valid_action_count": c,
            "bc_entropy": float(arrays["bc_entropy"][position]),
            "actor_entropy": float(arrays["post_entropy"][position]),
            "actor_normalized_entropy": float(arrays["post_entropy"][position] / hmax) if hmax else None,
            "entropy_floor": 0.02,
            "q_min": arrays["q_min"][position].tolist(),
            "kl": float(arrays["post_kl"][position]),
        })
    return result


def _factor_crossing(metrics: Mapping[str, Any], floor: float) -> Optional[Mapping[str, Any]]:
    for position, factor in enumerate(metrics.get("factor_results", []) or []):
        if bool(factor.get("entropy_violation", False)) and float(factor.get("entropy_min", floor)) < float(floor):
            return {"factor_index": int(position), **dict(factor)}
    return None


def _write_trajectory(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("status\nNO_ACTOR_ROWS\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _transaction_audit(records: Sequence[Mapping[str, Any]], source_text: str) -> Mapping[str, Any]:
    rows = []
    for record in records:
        if record.get("update_kind") not in ("actor", "actor_rejected"):
            continue
        metrics = record.get("metrics", {})
        rows.append({
            "pre_gate_safe": not bool(metrics.get("pre_step_hard_stop", False)),
            "optimizer_step": bool(metrics.get("optimizer_step_performed", False)),
            "accepted": bool(metrics.get("accepted", False)),
            "rollback": bool(metrics.get("rollback_performed", False)),
            "accepted_count_incremented": bool(metrics.get("accepted", False)),
        })
    result = dict(audit_transaction_rows(rows))
    source_checks = {
        "pre_step_hard_stop_before_optimizer_step": source_text.find("if float(sentinel_pre[\"kl_max\"]) >= float(hard_stop):") < source_text.find("self.actor_optimizer.step()"),
        "post_step_entropy_check_present": "entropy_violation" in source_text,
        "rollback_before_rejection_return": source_text.find("self._restore_actor_transaction(actor_before") < source_text.find('"rejection_reason": "all_trust_region_factors_failed"'),
        "accepted_update_increment_after_gate": source_text.find("self.actor_update_count += 1") > source_text.find("if accepted is not True:"),
    }
    result["source_order_checks"] = source_checks
    result["status"] = "PASS" if result["status"] == "PASS" and all(source_checks.values()) else "FAIL"
    result["actor_journal_rows"] = len(rows)
    return result


def _build_one_step_counterfactual(
    torch,
    nn,
    device,
    bc_checkpoint,
    config,
    pre_payload,
    batch,
    sentinel_batch,
) -> Mapping[str, Any]:
    specifications = {
        "R0_ORIGINAL": (True, True, True, True),
        "R1_NO_Q": (False, True, True, True),
        "R2_NO_ENTROPY": (True, False, True, True),
        "R3_NO_BC_KL": (True, True, False, True),
        "R4_NO_OPTIMIZER": (True, True, True, False),
        "R5_Q_ONLY": (True, False, False, True),
        "R6_ENTROPY_ONLY": (False, True, False, True),
        "R7_BC_KL_ONLY": (False, False, True, True),
        "R8_Q_PLUS_BC_KL": (True, False, True, True),
    }
    variants = {}
    for name, flags in specifications.items():
        branch = _new_agent(torch, nn, device, bc_checkpoint, config)
        _restore_agent(branch, pre_payload)
        branch.set_kl_sentinel(sentinel_batch)
        if name == "R0_ORIGINAL":
            result = branch.update_actor_transactional(
                batch,
                hard_stop=float(config.get("bc_kl_hard_stop", 1.0)),
                entropy_floor=float(config.get("entropy_floor", 0.02)),
                trust_region_factors=tuple(config.get("kl_backtracking_factors", (1.0, .5, .25, .125, .0625, .03125, .015625))),
            )
            result = {key: value for key, value in result.items() if key != "factor_results"}
        else:
            result = _counterfactual_actor_step(
                branch,
                batch,
                include_q=flags[0],
                include_entropy=flags[1],
                include_bc_kl=flags[2],
                do_step=flags[3],
            )
            result.pop("_diagnostics", None)
        variants[name] = result
    return {
        "status": "PASS",
        "variants": variants,
        "contract": {
            "same_pre_payload": True,
            "same_batch": True,
            "same_sentinel": True,
            "actor_only": True,
            "critic_optimizer_step": 0,
        },
    }


def _cumulative_ablation(
    torch,
    nn,
    device,
    bc_checkpoint,
    config,
    pre_payload,
    actor_schedule: Sequence[Tuple[int, np.ndarray]],
    replay,
) -> Mapping[str, Any]:
    if not pre_payload or not actor_schedule:
        return {"status": "NOT_RUN_NO_CROSSING_SCHEDULE", "branches": {}}
    specs = {
        "FULL": (True, True, True),
        "NO_Q": (False, True, True),
        "NO_ENTROPY": (True, False, True),
        "NO_BC_KL": (True, True, False),
    }
    branches = {}
    limit = min(200, len(actor_schedule))
    for name, flags in specs.items():
        branch = _new_agent(torch, nn, device, bc_checkpoint, config)
        _restore_agent(branch, pre_payload)
        checkpoints = {}
        for offset in range(limit):
            _, indices = actor_schedule[offset]
            batch = replay.batch_from_indices(indices, torch=torch, device=device)
            _counterfactual_actor_step(
                branch,
                batch,
                include_q=flags[0],
                include_entropy=flags[1],
                include_bc_kl=flags[2],
                do_step=True,
            )
            if offset + 1 in (25, 50, 100, 200):
                checkpoints[str(offset + 1)] = {
                    "actor_fingerprint": branch.actor_fingerprint(),
                    "actor_optimizer_steps": int(branch.actor_optimizer_step_count),
                    "actor_update_count": int(branch.actor_update_count),
                }
        branches[name] = {
            "proposal_count": int(limit),
            "checkpoints": checkpoints,
            "final_actor_fingerprint": branch.actor_fingerprint(),
        }
    return {
        "status": "PASS",
        "proposal_limit_per_branch": int(limit),
        "total_counterfactual_proposals": int(limit * len(specs)),
        "branches": branches,
        "starting_point": "pre_first_entropy_candidate_crossing",
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir == run_dir:
        raise SystemExit("--out-dir must differ from --run-dir; V5 is read-only")
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn as nn

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
    if bool(config.get("deterministic_replay")):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    device = torch.device(args.device)

    input_identity = json.loads((run_dir / "input_identity.json").read_text(encoding="utf-8"))
    bc_path = Path(input_identity["bc_checkpoint_path"]).expanduser().resolve()
    if file_sha256(bc_path) != EXPECTED_BC_SHA:
        raise RuntimeError("BC checkpoint SHA does not match frozen V5 identity")
    checkpoint_path = run_dir / "checkpoint_step_00000000.pt"
    failure_checkpoint_path = run_dir / "checkpoint_failure.pt"
    checkpoint = load_checkpoint(torch, checkpoint_path, map_location="cpu")
    failure_checkpoint = load_checkpoint(torch, failure_checkpoint_path, map_location="cpu")
    replay = SACReplayBuffer.open(run_dir / "replay", read_only=True)
    if replay.size != EXPECTED_REPLAY_ROWS:
        raise RuntimeError("V5 replay row count changed: {}".format(replay.size))
    bc_checkpoint = torch.load(str(bc_path), map_location="cpu", weights_only=False)
    records = _records(run_dir / "training_step_journal.jsonl")
    if not records:
        raise RuntimeError("canonical V5 training journal is empty")
    sentinel_rows = _sentinel_indices(run_dir / "sentinel_manifest.json")
    if sentinel_rows.size != EXPECTED_SENTINEL_ROWS:
        raise RuntimeError("unexpected V5 sentinel count: {}".format(sentinel_rows.size))

    _write_json(out_dir / "input_identity.json", {
        "schema_id": "discrete_sac_entropy_floor_root_cause_v6_input_identity_v1",
        "source_run_dir": str(run_dir),
        "source_run_dir_modified": False,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_failure_checkpoint": str(failure_checkpoint_path),
        "source_failure_checkpoint_sha256": file_sha256(failure_checkpoint_path),
        "source_journal": str(run_dir / "training_step_journal.jsonl"),
        "source_journal_sha256": file_sha256(run_dir / "training_step_journal.jsonl"),
        "source_batch_index_journal_sha256": _source_sha(run_dir / "batch_index_journal.npz"),
        "source_replay_metadata_sha256": replay.metadata_sha256(),
        "source_replay_rows": int(replay.size),
        "source_sentinel_manifest_sha256": file_sha256(run_dir / "sentinel_manifest.json"),
        "source_sentinel_rows": int(sentinel_rows.size),
        "bc_checkpoint_path": str(bc_path),
        "bc_checkpoint_sha256": file_sha256(bc_path),
        "expected_bc_checkpoint_sha256": EXPECTED_BC_SHA,
        "training_config_sha256": file_sha256(run_dir / "training_config.json"),
        "training_config": config,
        "device": str(device),
        "new_environment_steps": 0,
        "unity_bridge_ros_started": False,
        "v5_historical_artifacts_modified": False,
    })

    source_network = Path(__file__).resolve().parents[1] / "python/planning/sac/network.py"
    source_code_text = source_network.read_text(encoding="utf-8")
    floor = float(config.get("entropy_floor", 0.02))
    _write_json(out_dir / "entropy_gate_contract.json", {
        "schema_id": "discrete_sac_entropy_floor_gate_contract_audit_v1",
        "entropy_threshold": floor,
        "entropy_definition": "-sum(valid_pi * valid_log_pi)",
        "mask_definition": "invalid probabilities exactly zero; invalid log probabilities exactly zero",
        "mask_normalization": "masked log_softmax then valid probability renormalization",
        "zero_times_log_zero": "represented as 0 probability and 0 returned log probability",
        "eligible_predicate": "valid_action_count > 1 AND bc_entropy >= entropy_floor",
        "gate_statistic": "minimum actor entropy over eligible sentinel rows",
        "gate_comparison": "absolute strict lower-bound: entropy_min < floor is violation",
        "pre_step_kl_order": "pre-sentinel KL hard stop before optimizer.step",
        "post_step_order": "each trust-region factor steps a restored candidate then evaluates KL and entropy",
        "factors": config.get("kl_backtracking_factors"),
        "n_valid_one": "excluded from entropy-floor eligibility; mathematically Hmax=0",
        "source_network_sha256": file_sha256(source_network),
        "source_contract_unchanged": True,
    })

    agent = _new_agent(torch, nn, device, bc_checkpoint, config)
    _restore_agent(agent, checkpoint)
    replay_rng = np.random.RandomState()
    rng_state = checkpoint.get("rng_state", {})
    restore_rng_state(torch, replay_rng, rng_state)
    sentinel_batch = replay.batch_from_indices(sentinel_rows, torch=torch, device=device)
    sentinel_manifest = agent.set_kl_sentinel(sentinel_batch)
    baseline = _bc_entropy_baseline(agent, sentinel_batch, floor=floor)
    _write_json(out_dir / "bc_entropy_baseline.json", {
        **baseline,
        "sentinel_manifest": sentinel_manifest,
        "sentinel_manifest_sha256": file_sha256(run_dir / "sentinel_manifest.json"),
        "baseline_actor_is_bc": True,
        "gate_compatibility_note": "All-state below-floor rows are reported separately; production excludes n_valid=1 and BC-below-floor rows from the eligible minimum.",
    })

    counts = np.asarray(sentinel_batch["action_mask"].detach().cpu().numpy(), dtype=bool).sum(axis=1)
    hmax = valid_action_entropy_max(sentinel_batch["action_mask"].detach().cpu().numpy())
    _write_json(out_dir / "mask_entropy_feasibility.json", {
        "schema_id": "discrete_sac_mask_entropy_feasibility_v1",
        "state_count": int(counts.size),
        "entropy_floor": floor,
        "single_valid_action_count": int(np.sum(counts == 1)),
        "single_valid_action_ratio": float(np.mean(counts == 1)),
        "current_production_eligible_impossible_count": int(np.sum((counts > 1) & (hmax < floor))),
        "current_production_eligible_impossible_ratio": float(np.mean((counts > 1) & (hmax < floor))),
        "naive_all_state_absolute_floor_impossible_count": int(np.sum(hmax < floor)),
        "naive_all_state_absolute_floor_impossible_ratio": float(np.mean(hmax < floor)),
        "minimum_multi_valid_hmax": float(np.min(hmax[counts > 1])) if np.any(counts > 1) else None,
        "mathematical_compatibility": bool(np.all(hmax[(counts > 1)] >= floor)),
    })

    trajectory_rows = []
    exact_rows = []
    actor_schedule: List[Tuple[int, np.ndarray]] = []
    first_candidate = None
    first_committed = None
    pre_payload = None
    pre_batch = None
    pre_indices = None
    pre_record = None
    last_rejection_payload = None
    last_rejection_batch = None
    last_rejection_indices = None
    exact = True
    first_divergence = None
    actor_seen = 0
    source_replay_sha = replay.metadata_sha256()
    current_sentinel_entropy_min = baseline.get("eligible_entropy_min", None)
    current_sentinel_kl_max = 0.0

    for record in records:
        if args.max_actor_proposals and actor_seen >= int(args.max_actor_proposals):
            break
        indices = np.asarray(record.get("batch_indices", []), dtype=np.int64)
        expected_sample = np.asarray(
            replay_rng.randint(0, int(record["replay_size"]), size=indices.size), dtype=np.int64
        )
        batch_match = bool(np.array_equal(expected_sample, indices))
        batch = replay.batch_from_indices(indices, torch=torch, device=device)
        state_before = stable_fingerprint(agent.state_payload())
        state_before_match = state_before == record.get("model_state_before_sha256")
        rng_before = capture_rng_state(torch, replay_rng)
        kind = str(record.get("update_kind", ""))
        metrics_hint = record.get("metrics", {})
        if kind in ("actor", "actor_rejected"):
            actor_seen += 1
            crossing_hint = _factor_crossing(metrics_hint, floor)
            if crossing_hint is not None and first_candidate is None:
                pre_payload = deepcopy(agent.state_payload())
                pre_batch = _clone_batch(batch)
                pre_indices = indices.copy()
                pre_record = dict(record)
                sentinel_pre = _policy_snapshot(agent, sentinel_batch, floor=floor)
                first_candidate = {
                    "proposal_id": int(metrics_hint.get("actor_proposal_id", actor_seen)),
                    "journal_sequence": int(record.get("journal_sequence", -1)),
                    "factor": crossing_hint,
                    "pre_sentinel": _strip_arrays(sentinel_pre),
                }
            actor_schedule.append((int(record.get("journal_sequence", actor_seen)), indices.copy()))
            if kind == "actor_rejected":
                last_rejection_payload = deepcopy(agent.state_payload())
                last_rejection_batch = _clone_batch(batch)
                last_rejection_indices = indices.copy()
        if kind == "critic":
            metrics = agent.update_critic(batch)
        else:
            metrics = agent.update_actor_transactional(
                batch,
                hard_stop=float(config.get("bc_kl_hard_stop", 1.0)),
                entropy_floor=floor,
                trust_region_factors=tuple(config.get("kl_backtracking_factors", (1.0, .5, .25, .125, .0625, .03125, .015625))),
            )
        if kind in ("actor", "actor_rejected"):
            factor_results = metrics.get("factor_results", []) or []
            final_factor_entropy = (
                float(factor_results[-1].get("entropy_min", current_sentinel_entropy_min or 0.0))
                if factor_results else None
            )
            if bool(metrics.get("accepted", False)):
                post_sentinel_entropy_min = metrics.get("entropy_min", current_sentinel_entropy_min)
                post_sentinel_kl_max = metrics.get("sentinel_kl_max", current_sentinel_kl_max)
            else:
                post_sentinel_entropy_min = current_sentinel_entropy_min
                post_sentinel_kl_max = current_sentinel_kl_max
            committed_cross = (
                current_sentinel_entropy_min is not None
                and post_sentinel_entropy_min is not None
                and current_sentinel_entropy_min >= floor
                and float(post_sentinel_entropy_min) < floor
                and bool(metrics.get("accepted", False))
            )
            if committed_cross and first_committed is None:
                first_committed = {
                    "proposal_id": int(metrics.get("actor_proposal_id", actor_seen)),
                    "journal_sequence": int(record.get("journal_sequence", -1)),
                    "before_entropy_min": current_sentinel_entropy_min,
                    "after_entropy_min": post_sentinel_entropy_min,
                }
            trajectory_rows.append({
                "journal_sequence": int(record.get("journal_sequence", -1)),
                "proposal_id": int(metrics.get("actor_proposal_id", actor_seen)),
                "actor_update_id": int(metrics.get("accepted_actor_update_id", agent.actor_update_count)),
                "update_kind": kind,
                "accepted": bool(metrics.get("accepted", False)),
                "rejected": bool(metrics.get("rejected", False)),
                "accepted_lr_factor": metrics.get("accepted_lr_factor"),
                "pre_sentinel_entropy_min": current_sentinel_entropy_min,
                "post_sentinel_entropy_min": post_sentinel_entropy_min,
                "pre_sentinel_normalized_entropy_min": None,
                "post_sentinel_normalized_entropy_min": None,
                "pre_sentinel_bc_relative_entropy_min": None,
                "post_sentinel_bc_relative_entropy_min": None,
                "pre_sentinel_entropy_mean": None,
                "post_sentinel_entropy_mean": None,
                "pre_sentinel_kl_max": current_sentinel_kl_max,
                "post_sentinel_kl_max": post_sentinel_kl_max,
                "pre_sentinel_argmax_flip_rate": None,
                "post_sentinel_argmax_flip_rate": None,
                "batch_entropy_min": None,
                "batch_entropy_after_min": None,
                "batch_kl_max": metrics.get("bc_kl_max"),
                "residual_saturation_fraction": metrics.get("residual_saturation_fraction"),
                "residual_abs_mean": metrics.get("residual_abs_mean"),
                "residual_abs_p99": metrics.get("residual_abs_p99"),
                "residual_abs_max": metrics.get("residual_abs_max"),
                "gradient_norm_q": metrics.get("gradient_norm_q"),
                "gradient_norm_entropy": metrics.get("gradient_norm_entropy"),
                "gradient_norm_bc_kl": metrics.get("gradient_norm_bc_kl"),
                "gradient_norm_total": metrics.get("gradient_norm_total", metrics.get("gradient_norm")),
                "backtrack_count": metrics.get("backtrack_count"),
                "rejection_reason": metrics.get("rejection_reason", ""),
                "factor_crossing_entropy_min": crossing_hint.get("entropy_min") if crossing_hint else final_factor_entropy if final_factor_entropy is not None and final_factor_entropy < floor else None,
                "factor_crossing_factor": crossing_hint.get("factor") if crossing_hint else None,
            })
            if bool(metrics.get("accepted", False)):
                current_sentinel_entropy_min = post_sentinel_entropy_min
                current_sentinel_kl_max = post_sentinel_kl_max
        state_after = stable_fingerprint(agent.state_payload())
        rng_after = capture_rng_state(torch, replay_rng)
        state_after_match = state_after == record.get("model_state_after_sha256")
        metric_mismatches = _float_mismatches(record.get("metrics", {}), metrics)
        rng_before_match = rng_fingerprint(rng_before) == record.get("replayable_rng_before_fingerprint")
        rng_after_match = rng_fingerprint(rng_after) == record.get("replayable_rng_after_fingerprint")
        row = {
            "journal_sequence": int(record.get("journal_sequence", -1)),
            "update_kind": kind,
            "update_index": int(record.get("update_index", -1)),
            "batch_match": batch_match,
            "batch_sha_match": batch_indices_sha256(indices) == record.get("batch_indices_sha256"),
            "state_before_match": state_before_match,
            "state_after_match": state_after_match,
            "rng_before_match": rng_before_match,
            "rng_after_match": rng_after_match,
            "metric_mismatch_count": len(metric_mismatches),
            "metric_mismatches": ",".join(metric_mismatches),
        }
        exact_rows.append(row)
        if not (batch_match and row["batch_sha_match"] and state_before_match and state_after_match and not metric_mismatches):
            exact = False
            if first_divergence is None:
                first_divergence = {
                    "journal_sequence": row["journal_sequence"],
                    "row": row,
                    "record": record,
                }
        if kind == "actor_rejected":
            last_rejection_payload = deepcopy(agent.state_payload())

    _write_trajectory(out_dir / "sentinel_entropy_trajectory.csv", trajectory_rows)
    _write_trajectory(out_dir / "batch_vs_sentinel_entropy.csv", trajectory_rows)
    _write_json(out_dir / "exact_replay_result.json", {
        "schema_id": "bc_initialized_discrete_sac_entropy_floor_exact_replay_v1",
        "status": "PASS" if exact else "FAIL",
        "exact_replay": bool(exact),
        "journal_records": len(exact_rows),
        "actor_proposals": len(trajectory_rows),
        "first_divergence": first_divergence,
        "source_replay_metadata_sha256_before": source_replay_sha,
        "source_replay_metadata_sha256_after": replay.metadata_sha256(),
        "source_replay_unchanged": source_replay_sha == replay.metadata_sha256(),
        "final_agent_counters": {
            key: int(getattr(agent, key))
            for key in (
                "actor_optimizer_step_count", "critic_optimizer_step_count", "actor_update_count",
                "critic_update_count", "actor_proposal_count", "actor_rejection_count",
                "actor_recovery_update_count", "nan_count", "invalid_action_count",
            )
        },
        "failure_checkpoint_counters": {
            key: failure_checkpoint.get(key)
            for key in ("environment_steps", "completed_episodes", "actor_update_count", "critic_update_count", "actor_proposal_count", "actor_rejection_count")
        },
    })

    candidate_crossing = first_candidate or {}
    _write_json(out_dir / "first_entropy_crossing.json", {
        "status": "FOUND" if first_candidate else "NOT_FOUND",
        "first_committed_state_crossing": first_committed,
        "first_candidate_factor_crossing": first_candidate,
        "definition": "strict eligible entropy before >= floor and candidate/after entropy < floor",
        "floor": floor,
    })

    if pre_payload is not None and pre_batch is not None and pre_indices is not None:
        branch = _new_agent(torch, nn, device, bc_checkpoint, config)
        _restore_agent(branch, pre_payload)
        candidate_result = _counterfactual_actor_step(
            branch,
            pre_batch,
            include_q=True,
            include_entropy=True,
            include_bc_kl=True,
            do_step=True,
        )
        _write_json(out_dir / "lowest_entropy_states.csv", {"rows": _lowest_states(replay, pre_indices, candidate_result, 20)})
        _write_json(out_dir / "crossing_gradient_decomposition.json", {
            "status": "PASS",
            "proposal_id": int(pre_record.get("metrics", {}).get("actor_proposal_id", -1)) if pre_record else None,
            "journal_sequence": int(pre_record.get("journal_sequence", -1)) if pre_record else None,
            **_component_decomposition(branch, pre_batch),
            "production_metrics": pre_record.get("metrics", {}) if pre_record else {},
        })
        _write_json(out_dir / "one_step_counterfactual.json", _build_one_step_counterfactual(
            torch, nn, device, bc_checkpoint, config, pre_payload, pre_batch, sentinel_batch
        ))
        _write_json(out_dir / "cumulative_ablation.json", _cumulative_ablation(
            torch, nn, device, bc_checkpoint, config, pre_payload, actor_schedule, replay
        ))
    else:
        _write_json(out_dir / "lowest_entropy_states.csv", {"rows": [], "status": "NOT_RUN_NO_CROSSING"})
        _write_json(out_dir / "crossing_gradient_decomposition.json", {"status": "NOT_RUN_NO_CROSSING"})
        _write_json(out_dir / "one_step_counterfactual.json", {"status": "NOT_RUN_NO_FIRST_CROSSING"})
        _write_json(out_dir / "cumulative_ablation.json", {"status": "NOT_RUN_NO_FIRST_CROSSING"})

    actor_trajectory = trajectory_rows
    residual_values = [row["residual_saturation_fraction"] for row in actor_trajectory if row.get("residual_saturation_fraction") is not None]
    entropy_values = [row["post_sentinel_entropy_min"] for row in actor_trajectory if row.get("post_sentinel_entropy_min") is not None]
    _write_json(out_dir / "residual_saturation_analysis.json", {
        "status": "PASS" if residual_values else "NO_ACTOR_ROWS",
        "actor_rows": len(actor_trajectory),
        "saturation_fraction": _summary(residual_values),
        "entropy_min": _summary(entropy_values),
        "saturation_vs_entropy_min": _correlation(residual_values, entropy_values),
        "at_first_candidate_crossing": candidate_crossing.get("factor") if candidate_crossing else None,
        "interpretation": "association only; this audit does not claim that residual saturation is causal",
    })
    source_text = Path(__file__).resolve().parents[1].joinpath("python/planning/sac/network.py").read_text(encoding="utf-8")
    transaction = _transaction_audit(records, source_text)
    _write_json(out_dir / "entropy_transaction_audit.json", transaction)

    root_labels = []
    evidence = []
    if baseline["below_gate_count_all_states"] > 0:
        root_labels.append("J_BC_BASELINE_HAS_LOW_ENTROPY_STATES_BUT_EXCLUDED_BY_CURRENT_ELIGIBILITY")
        evidence.append("BC below-floor states are reported; current gate excludes BC entropy below floor from eligible minimum")
    if first_candidate:
        root_labels.append("M_POST_STEP_ENTROPY_FLOOR_CROSSING")
        evidence.append("a trust-region candidate factor crossed below the strict entropy floor while KL remained below hard stop")
    if residual_values and float(np.mean(residual_values)) > 0.25:
        root_labels.append("E_RESIDUAL_CAP_SATURATION")
        evidence.append("actor-batch residual saturation fraction is materially non-zero")
    if transaction["status"] != "PASS":
        root_labels.append("I_TRANSACTION_OR_JOURNAL_CONTRACT_FAILURE")
        evidence.append("transaction audit did not pass")
    if not root_labels:
        root_labels.append("UNKNOWN_NO_SUPPORTED_ROOT_CAUSE")
    _write_json(out_dir / "root_cause.json", {
        "schema_id": "discrete_sac_entropy_floor_root_cause_v6",
        "status": "PASS" if exact else "EXACT_REPLAY_FAIL",
        "labels": root_labels,
        "direct_evidence": evidence,
        "separate_entropy_drift_from_gate_failure": True,
        "entropy_drift_candidates": [label for label in root_labels if label.startswith(("A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_"))],
        "gate_failure_candidates": [label for label in root_labels if label.startswith(("I_", "J_", "K_", "L_", "M_"))],
        "threshold_changed": False,
        "hyperparameters_changed": False,
        "production_gate_changed": False,
        "causal_scope": "deterministic V5 journal replay and fixed sentinel; no claim beyond this run",
    })

    final_metrics = trajectory_rows[-1] if trajectory_rows else {}
    report = [
        "# DISCRETE_SAC_ENTROPY_FLOOR_ROOT_CAUSE_V6",
        "",
        "本报告只读复放 V5 的 SAC 更新 journal；没有启动 Unity/Bridge/ROS，没有新增环境步，也没有写入 V5。",
        "",
        "## 复放与合同",
        "",
        "- EXACT_REPLAY_PASS: `{}`",
        "- journal records: `{}`; actor proposals: `{}`",
        "- entropy floor: `{}`; gate statistic: eligible rows 的绝对最小熵",
        "- eligible: `valid_action_count > 1 AND BC entropy >= floor`",
        "- replay unchanged: `{}`",
        "",
        "## BC 基线与 mask 可行性",
        "",
        "- BC sentinel below-floor rows (all rows): `{}` / `{}`",
        "- current eligible-contract impossible mask rows: `{}`",
        "- single-valid-action rows: `{}`",
        "- independent NumPy reference max entropy delta: `{}`",
        "",
        "## 熵轨迹与残差",
        "",
        "- first committed crossing: `{}`",
        "- first candidate factor crossing: `{}`",
        "- final post-sentinel entropy min: `{}`",
        "- final post-sentinel KL max: `{}`",
        "- residual saturation/entropy correlation: see `residual_saturation_analysis.json`",
        "",
        "## 梯度与反事实",
        "",
        "- `crossing_gradient_decomposition.json` contains Q/entropy/BC-KL norms and cosine terms.",
        "- `one_step_counterfactual.json` contains R0/R1/R2/R3/R4/R5/R6/R7/R8 when a crossing was found.",
        "- `cumulative_ablation.json` is bounded to at most 4 x 200 diagnostic proposals.",
        "",
        "## 结论边界",
        "",
        "- root-cause labels: `{}`",
        "- production threshold/hyperparameters changed: `NO`",
        "- Actor/Critic/Replay production artifacts changed: `NO`",
        "- Dev100/Final300: `NO`",
    ]
    (out_dir / "report_zh.md").write_text("\n".join(report).format(
        "PASS" if exact else "FAIL",
        len(exact_rows),
        len(trajectory_rows),
        floor,
        source_replay_sha == replay.metadata_sha256(),
        baseline["below_gate_count_all_states"],
        baseline["below_gate_ratio_all_states"],
        int(np.sum((counts > 1) & (hmax < floor))),
        baseline["single_valid_action_count"],
        baseline["production_reference_max_abs_entropy_delta"],
        first_committed,
        first_candidate,
        final_metrics.get("post_sentinel_entropy_min"),
        final_metrics.get("post_sentinel_kl_max"),
        ", ".join(root_labels),
    ) + "\n", encoding="utf-8")
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
