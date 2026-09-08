#!/usr/bin/env python3
"""Run a bounded offline sanity check for AWAC Actor depth learning rates.

The script uses one fixed replay batch and a fresh learner per candidate.  It
never writes a formal checkpoint and never starts Unity or Bridge.  The
existing learner update is used unchanged so the measurement is about the
optimizer group only, not a second Actor objective.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from planning.awac.learner import AWACOptimizationConfig
from planning.awac.replay import AWACReplayBuffer
from planning.awac.trainer import build_learner, validate_bc_checkpoint_for_awac
from planning.bc.model import require_torch
from planning.common.checkpoint import load_torch
from planning.common.hashing import file_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded offline AWAC Actor depth-learning-rate sanity benchmark"
    )
    parser.add_argument("--bc-checkpoint", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4026)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--actor-head-lr", type=float, default=1.0e-5)
    parser.add_argument("--actor-vector-lr", type=float, default=3.0e-6)
    parser.add_argument("--critic-head-lr", type=float, default=1.0e-4)
    parser.add_argument("--critic-vector-lr", type=float, default=1.0e-5)
    parser.add_argument("--critic-depth-lr", type=float, default=1.0e-5)
    parser.add_argument("--candidate", dest="candidates", action="append", type=float)
    parser.add_argument("--bc-kl-hard-budget", type=float, default=0.10)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 20 <= int(args.steps) <= 100:
        raise ValueError("steps must be in [20,100]")
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    if int(args.cpu_threads) <= 0:
        raise ValueError("cpu-threads must be positive")
    if float(args.actor_head_lr) <= 0.0 or float(args.actor_vector_lr) < 0.0:
        raise ValueError("Actor learning rates are invalid")
    if float(args.critic_head_lr) <= 0.0 or float(args.critic_vector_lr) < 0.0:
        raise ValueError("Critic learning rates are invalid")
    if float(args.critic_depth_lr) <= 0.0:
        raise ValueError("critic-depth-lr must be positive")
    if float(args.bc_kl_hard_budget) <= 0.0:
        raise ValueError("bc-kl-hard-budget must be positive")
    candidates = args.candidates or [
        0.10 * float(args.actor_head_lr),
        0.25 * float(args.actor_head_lr),
        0.50 * float(args.actor_head_lr),
    ]
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in candidates):
        raise ValueError("all Actor depth-LR candidates must be finite and positive")
    if any(float(value) > float(args.actor_head_lr) for value in candidates):
        raise ValueError("Actor depth-LR candidates may not exceed Actor head LR")


def _clone_batch(batch: Mapping[str, Any], *, torch) -> Dict[str, Any]:
    return {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def _group_state(module: Any, group_name: str) -> Dict[str, Any]:
    group = getattr(module, group_name)
    return {
        name: parameter.detach().clone()
        for name, parameter in group.named_parameters()
    }


def _parameter_delta(before: Mapping[str, Any], module: Any, group_name: str, *, torch) -> float:
    after = _group_state(module, group_name)
    total = torch.tensor(0.0, dtype=torch.float64)
    for name, value in before.items():
        delta = after[name].to(dtype=torch.float64) - value.to(dtype=torch.float64)
        total = total + (delta * delta).sum()
    return float(torch.sqrt(total).item())


def _gradient_norm(module: Any, group_name: str, *, torch) -> float:
    total = torch.tensor(0.0, dtype=torch.float64)
    for parameter in getattr(module, group_name).parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().to(dtype=torch.float64)
        total = total + (gradient * gradient).sum()
    return float(torch.sqrt(total).item())


def _all_finite(metrics: Mapping[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
            return False
    return True


def _run_candidate(
    *,
    torch,
    nn,
    bc_checkpoint: Mapping[str, Any],
    batch: Mapping[str, Any],
    candidate: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    torch.manual_seed(int(args.seed))
    config = AWACOptimizationConfig(
        actor_head_lr=float(args.actor_head_lr),
        actor_vector_lr=float(args.actor_vector_lr),
        actor_depth_lr=float(candidate),
        critic_head_lr=float(args.critic_head_lr),
        critic_vector_lr=float(args.critic_vector_lr),
        critic_depth_lr=float(args.critic_depth_lr),
        bc_kl_hard_budget=float(args.bc_kl_hard_budget),
    )
    learner = build_learner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_checkpoint=bc_checkpoint,
        config=config,
    )
    rows = []
    finite = True
    for step in range(int(args.steps)):
        depth_before = _group_state(learner.actor, "depth_encoder")
        head_before = _group_state(learner.actor, "head")
        metrics = learner.update(
            _clone_batch(batch, torch=torch),
            update_actor=True,
        )
        depth_gradient_norm = _gradient_norm(learner.actor, "depth_encoder", torch=torch)
        head_gradient_norm = _gradient_norm(learner.actor, "head", torch=torch)
        depth_delta = _parameter_delta(
            depth_before, learner.actor, "depth_encoder", torch=torch
        )
        head_delta = _parameter_delta(head_before, learner.actor, "head", torch=torch)
        row = {
            "step": step + 1,
            "actor_update_step": float(metrics.get("actor_optimizer_step", 0.0)),
            "actor_update_count": float(metrics.get("actor_update_count", 0.0)),
            "actor_loss": float(metrics["actor_loss"]),
            "bc_kl": float(metrics["bc_kl"]),
            "bc_kl_max_after_update": float(metrics["bc_kl_max_after_update"]),
            "depth_gradient_norm": depth_gradient_norm,
            "head_gradient_norm": head_gradient_norm,
            "depth_parameter_delta_norm": depth_delta,
            "head_parameter_delta_norm": head_delta,
            "finite": bool(_all_finite(metrics)),
        }
        rows.append(row)
        finite = finite and bool(row["finite"])

    max_kl = max(float(row["bc_kl_max_after_update"]) for row in rows)
    max_depth_gradient = max(float(row["depth_gradient_norm"]) for row in rows)
    total_depth_delta = sum(float(row["depth_parameter_delta_norm"]) for row in rows)
    stable = bool(
        finite
        and max_kl <= float(args.bc_kl_hard_budget) + 1.0e-8
        and max_depth_gradient > 0.0
        and total_depth_delta > 0.0
        and int(learner.actor_update_count) > 0
    )
    return {
        "actor_depth_lr": float(candidate),
        "steps": int(args.steps),
        "stable": stable,
        "finite": finite,
        "actor_update_count": int(learner.actor_update_count),
        "actor_optimizer_step_count": int(learner.actor_optimizer_step_count),
        "max_bc_kl": max_kl,
        "max_depth_gradient_norm": max_depth_gradient,
        "total_depth_parameter_delta_norm": total_depth_delta,
        "max_head_gradient_norm": max(float(row["head_gradient_norm"]) for row in rows),
        "total_head_parameter_delta_norm": sum(
            float(row["head_parameter_delta_norm"]) for row in rows
        ),
        "rows": rows,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    torch, nn, _, _, _ = require_torch()
    torch.set_num_threads(int(args.cpu_threads))
    np.random.seed(int(args.seed))
    bc_path = Path(args.bc_checkpoint).expanduser().resolve()
    replay_path = Path(args.replay_dir).expanduser().resolve()
    if not bc_path.is_file():
        raise FileNotFoundError("BC checkpoint does not exist: {}".format(bc_path))
    checkpoint = load_torch(bc_path, torch=torch, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("BC checkpoint must be a mapping")
    validate_bc_checkpoint_for_awac(
        checkpoint,
        mpl_contract_sha256=str(checkpoint.get("mpl_contract_sha256", "")),
    )
    replay = AWACReplayBuffer.open(replay_path)
    if str(replay.metadata.get("behavior_source_phase", "")) != "critic_calibration":
        raise ValueError("sanity benchmark requires the offline BC_CALIBRATION replay")
    if int(replay.size) < int(args.batch_size):
        raise ValueError("replay is smaller than batch-size")
    batch = replay.sample(
        int(args.batch_size),
        rng=np.random.RandomState(int(args.seed) + 1),
        torch=torch,
        device=torch.device("cpu"),
    )
    candidates = sorted(
        set(float(value) for value in (args.candidates or [
            0.10 * float(args.actor_head_lr),
            0.25 * float(args.actor_head_lr),
            0.50 * float(args.actor_head_lr),
        ]))
    )
    results = [
        _run_candidate(
            torch=torch,
            nn=nn,
            bc_checkpoint=checkpoint,
            batch=batch,
            candidate=candidate,
            args=args,
        )
        for candidate in candidates
    ]
    stable = [row for row in results if row["stable"]]
    output = {
        "contract": "awac_actor_depth_lr_sanity_v1",
        "runtime_mode": "offline_fixed_replay_batch",
        "bc_checkpoint": str(bc_path),
        "bc_checkpoint_sha256": file_sha256(bc_path),
        "replay_dir": str(replay_path),
        "replay_size": int(replay.size),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "actor_head_lr": float(args.actor_head_lr),
        "actor_vector_lr": float(args.actor_vector_lr),
        "critic_depth_lr": float(args.critic_depth_lr),
        "candidates": results,
        "selected_actor_depth_lr": min(
            (float(row["actor_depth_lr"]) for row in stable), default=None
        ),
        "selected_rule": "smallest_stable_nonzero_candidate",
        "actor_update_executed_formal": False,
    }
    replay.close()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "actor_depth_lr_sanity.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = run(args)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["selected_actor_depth_lr"] is not None else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
