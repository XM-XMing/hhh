#!/usr/bin/env python3
"""Run the preregistered BC-initialized online discrete SAC mainline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from planning.common.atomic import write_json_atomic
from planning.common.hashing import file_sha256
from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM, policy_input_contract_sha256
from planning.contracts.offpolicy import algorithm_source_sha256
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.reward import REWARD_CONTRACT_ID, reward_contract_sha256
from planning.contracts.task import TASK_CONTRACT_ID, task_contract_sha256
from planning.evaluation.policy_evaluator import resolve_policy_checkpoint_task_contract
from planning.sac.checkpoint import (
    build_checkpoint_payload,
    save_checkpoint,
    write_checkpoint_manifest,
)
from planning.sac.contract import (
    SAC_ALGORITHM_ID,
    SAC_BC_REFERENCE_ID,
    SAC_CHECKPOINT_CONTRACT_ID,
    SAC_DEFAULTS,
    SAC_KL_BACKTRACKING_FACTORS,
    SAC_MODEL_TYPE,
    SAC_POLICY_ID,
    SAC_RESIDUAL_LOGIT_CAP,
    SAC_RESIDUAL_POLICY_ID,
    SAC_REPLAY_CONTRACT_ID,
    SAC_TRAINING_CONFIG_CONTRACT_ID,
    training_config_sha256,
)
from planning.sac.network import DiscreteSACAgent, masked_categorical
from planning.sac.replay import SACReplayBuffer
from planning.sac.runtime import SACOnlineRunner
from planning.sac.diagnostics import (
    SACUpdateJournal,
    capture_rng_state,
    cuda_determinism_metadata,
    write_batch_index_npz,
)


EXPECTED_BC_SHA256 = "ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2"
EXPECTED_TASK_SHA256 = "2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df"
EXPECTED_MPL_SHA256 = "f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d"
EXPECTED_OBSERVATION_CONTRACT = "reliable_exact_endpoint_snapshot"
SNAPSHOT_TARGETS = (0, 5000, 7500, 10000, 12500, 15000)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_discrete_sac.py",
        description="BC-initialized online masked-categorical discrete SAC",
        allow_abbrev=False,
    )
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--worker-spec-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--actor-architecture",
        choices=("direct", "residual"),
        default="direct",
        help="V4 direct Actor or V5 frozen-BC bounded residual Actor",
    )
    parser.add_argument("--env-workers", type=int, default=SAC_DEFAULTS["max_workers"])
    parser.add_argument("--max-env-steps", type=int, default=SAC_DEFAULTS["max_env_steps"])
    parser.add_argument("--max-episodes", type=int, default=SAC_DEFAULTS["max_episodes"])
    parser.add_argument("--learning-starts", type=int, default=SAC_DEFAULTS["learning_starts"])
    parser.add_argument("--batch-size", type=int, default=SAC_DEFAULTS["batch_size"])
    parser.add_argument("--gamma", type=float, default=SAC_DEFAULTS["gamma"])
    parser.add_argument("--tau", type=float, default=SAC_DEFAULTS["tau"])
    parser.add_argument("--alpha", type=float, default=SAC_DEFAULTS["alpha"])
    parser.add_argument("--beta-bc", type=float, default=SAC_DEFAULTS["beta_bc"])
    parser.add_argument("--reward-scale", type=float, default=SAC_DEFAULTS["reward_scale"])
    parser.add_argument("--actor-lr", type=float, default=1.0e-5)
    parser.add_argument("--critic-lr", type=float, default=1.0e-4)
    parser.add_argument(
        "--critic-updates-per-transition",
        type=float,
        default=SAC_DEFAULTS["critic_updates_per_transition"],
    )
    parser.add_argument(
        "--actor-updates-per-transition",
        type=float,
        default=SAC_DEFAULTS["actor_updates_per_transition"],
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=SAC_DEFAULTS["gradient_clip_norm"])
    parser.add_argument("--bc-kl-hard-stop", type=float, default=SAC_DEFAULTS["bc_kl_hard_stop"])
    parser.add_argument("--entropy-floor", type=float, default=SAC_DEFAULTS["entropy_floor"])
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--deterministic-replay",
        action="store_true",
        help="enable deterministic kernels and durable SAC update replay evidence",
    )
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
        return float(value)
    if isinstance(value, (Path,)):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _json_safe(payload), trailing_newline=True)


def _hash_paths(root: Path, relative_paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(str(item) for item in relative_paths):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(str(path))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_torch_checkpoint(torch, path: Path) -> Mapping[str, Any]:
    value = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("BC checkpoint must be a mapping")
    return value


def _resolve_device(torch, requested: str):
    value = str(requested)
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(value)


def _residual_kl_bound_probe(torch, agent, device: Any) -> Dict[str, Any]:
    """Run the V5 numeric bound and exact step-zero checks before Unity."""

    if agent.actor_architecture != "residual":
        return {"status": "NOT_APPLICABLE", "max_numeric_kl": 0.0}
    torch.manual_seed(20260908)
    depth = torch.zeros((32, 1, 90, 160), device=device, dtype=torch.float32)
    vector = torch.zeros((32, POLICY_VECTOR_DIM), device=device, dtype=torch.float32)
    mask = torch.ones((32, NUM_ACTIONS), device=device, dtype=torch.bool)
    mask[::3, -7:] = False
    with torch.no_grad():
        actor_logits = agent.actor(depth, vector)
        bc_logits = agent.bc_reference(depth, vector)
        actor_prob, actor_log, actor_mask = masked_categorical(actor_logits, mask, torch)
        bc_prob, bc_log, _ = masked_categorical(bc_logits, mask, torch)
        step_zero_kl = (actor_prob * (actor_log - bc_log)).sum(dim=1)
        step_zero_argmax = torch.argmax(
            actor_prob.masked_fill(~actor_mask, -1.0), dim=1
        ) == torch.argmax(bc_prob.masked_fill(~actor_mask, -1.0), dim=1)
        base = torch.randn((64, NUM_ACTIONS), device=device, dtype=torch.float32)
        delta = agent.residual_logit_cap * (2.0 * torch.rand_like(base) - 1.0)
        random_mask = torch.rand((64, NUM_ACTIONS), device=device) > 0.15
        random_mask[:, 0] = True
        candidate_prob, candidate_log, _ = masked_categorical(base + delta, random_mask, torch)
        reference_prob, reference_log, _ = masked_categorical(base, random_mask, torch)
        numeric_kl = (candidate_prob * (candidate_log - reference_log)).sum(dim=1)
    max_numeric_kl = float(numeric_kl.max().item())
    theoretical = float(2.0 * agent.residual_logit_cap)
    result = {
        "status": "PASS"
        if max_numeric_kl <= theoretical + 1.0e-6
        and float(step_zero_kl.abs().max().item()) <= 1.0e-7
        and bool(step_zero_argmax.all().item())
        else "FAIL",
        "step0_policy_max_abs_logit_diff": float((actor_logits - bc_logits).abs().max().item()),
        "step0_kl_max": float(step_zero_kl.abs().max().item()),
        "step0_argmax_match_fraction": float(step_zero_argmax.float().mean().item()),
        "max_numeric_kl": max_numeric_kl,
        "theoretical_kl_bound": theoretical,
        "tolerance": 1.0e-6,
        "mask_cases": 96,
    }
    if result["status"] != "PASS":
        raise RuntimeError("V5 residual KL/step-zero preflight failed: {}".format(result))
    return result


def _worker_identity(path: Path, count: int) -> Dict[str, Any]:
    from planning.runtime.worker import load_worker_runtime_spec_file

    specs = load_worker_runtime_spec_file(path)
    if len(specs) < int(count):
        raise ValueError("worker spec file has fewer workers than requested")
    selected = tuple(specs[: int(count)])
    runtime_ids = [str(spec.runtime_instance_id) for spec in selected]
    if [int(spec.worker_id) for spec in selected] != list(range(int(count))):
        raise ValueError("worker specs are not contiguous from worker 0")
    if len(set(runtime_ids)) != len(runtime_ids) or any(not value for value in runtime_ids):
        raise ValueError("worker runtime identities are not unique")
    return {
        "worker_spec_path": str(path.resolve()),
        "worker_spec_sha256": file_sha256(path),
        "worker_count": int(count),
        "runtime_instance_ids": runtime_ids,
        "runtime_launch_nonces": [str(spec.runtime_launch_nonce) for spec in selected],
        "training_run_ids": [str(spec.training_run_id) for spec in selected],
    }


def _resolved_config(args: argparse.Namespace, *, bc_sha: str, mission_sha: str, source_code_sha: str) -> Dict[str, Any]:
    config = {
        "contract_id": SAC_TRAINING_CONFIG_CONTRACT_ID,
        "algorithm_id": SAC_ALGORITHM_ID,
        "model_type": SAC_MODEL_TYPE,
        "policy_id": SAC_RESIDUAL_POLICY_ID if args.actor_architecture == "residual" else SAC_POLICY_ID,
        "bc_reference_id": SAC_BC_REFERENCE_ID,
        "source_bc_checkpoint_sha256": bc_sha,
        "mission_source_sha256": mission_sha,
        "source_code_sha256": source_code_sha,
        "observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        "observation_source": EXPECTED_OBSERVATION_CONTRACT,
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(int(args.max_steps)),
        "max_primitive_steps": int(args.max_steps),
        "max_steps": int(args.max_steps),
        "feature_contract_id": "depth_goal_state_prev_action",
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "vec_dim": POLICY_VECTOR_DIM,
        "num_actions": NUM_ACTIONS,
        "depth_history_frames": 1,
        "initial_prev_action": -1,
        "reward_contract_id": REWARD_CONTRACT_ID,
        "reward_contract_sha256": reward_contract_sha256(),
        "mask_contract": "depth_action_mask",
        "execution_mode": "continuous",
        "device": str(args.device),
        "seed": int(args.seed),
        "env_workers": int(args.env_workers),
        "max_env_steps": int(args.max_env_steps),
        "max_episodes": int(args.max_episodes),
        "learning_starts": int(args.learning_starts),
        "batch_size": int(args.batch_size),
        "gamma": float(args.gamma),
        "tau": float(args.tau),
        "alpha": float(args.alpha),
        "alpha_mode": "fixed_preregistered",
        "beta_bc": float(args.beta_bc),
        "reward_scale": float(args.reward_scale),
        "actor_lr": float(args.actor_lr),
        "critic_lr": float(args.critic_lr),
        "critic_updates_per_transition": float(args.critic_updates_per_transition),
        "actor_updates_per_transition": float(args.actor_updates_per_transition),
        "gradient_clip_norm": float(args.gradient_clip_norm),
        "bc_kl_hard_stop": float(args.bc_kl_hard_stop),
        "entropy_floor": float(args.entropy_floor),
        "online_replay_contract_id": SAC_REPLAY_CONTRACT_ID,
        "learning_starts_actor_frozen": True,
        "critic_target_update": "polyak",
        "terminal_bootstrap": False,
        "cql": False,
        "awac": False,
        "deterministic_replay": bool(args.deterministic_replay),
        "deterministic_pool": bool(args.deterministic_replay),
        "actor_kl_trust_region": True,
        "kl_backtracking_factors": list(SAC_KL_BACKTRACKING_FACTORS),
        "kl_sentinel_contract": "pre_actor_unique_state_observation_v4",
    }
    if args.actor_architecture == "residual":
        config.update(
            {
                "actor_architecture": "residual",
                "residual_logit_cap": float(SAC_RESIDUAL_LOGIT_CAP),
                "theoretical_kl_bound": float(2.0 * SAC_RESIDUAL_LOGIT_CAP),
                "bc_reference_fully_frozen": True,
                "residual_optimizer_only": True,
            }
        )
    return config


def _validate_inputs(checkpoint: Mapping[str, Any], *, path: Path, args: argparse.Namespace) -> None:
    if file_sha256(path) != EXPECTED_BC_SHA256:
        raise ValueError("BC checkpoint SHA256 does not match the frozen input")
    expected = {
        "vec_dim": POLICY_VECTOR_DIM,
        "num_actions": NUM_ACTIONS,
        "task_contract_sha256": EXPECTED_TASK_SHA256,
        "mpl_contract_sha256": EXPECTED_MPL_SHA256,
        "observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        "observation_source": EXPECTED_OBSERVATION_CONTRACT,
        "max_primitive_steps": int(args.max_steps),
    }
    for field, expected_value in expected.items():
        if checkpoint.get(field) != expected_value:
            raise ValueError(
                "BC checkpoint {} mismatch: {!r} != {!r}".format(
                    field, checkpoint.get(field), expected_value
                )
            )
    if str(checkpoint.get("model_type", "")) not in {"bc_soft", "bc_hard"}:
        raise ValueError("frozen input is not a BC checkpoint")
    if "model_state_dict" not in checkpoint:
        raise ValueError("BC model_state_dict is missing")


def _rng_state(torch, rng: np.random.RandomState, runner: Optional[SACOnlineRunner] = None) -> Dict[str, Any]:
    return dict(
        capture_rng_state(
            torch,
            rng,
            worker_state=runner.worker_state_snapshot() if runner is not None else None,
        )
    )


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if left.size < 2 or right.size != left.size:
        return None
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    left_rank = np.empty(left.size, dtype=np.float64)
    right_rank = np.empty(right.size, dtype=np.float64)
    left_rank[np.argsort(left, kind="mergesort")] = np.arange(left.size)
    right_rank[np.argsort(right, kind="mergesort")] = np.arange(right.size)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _critic_return_rho(agent: Any, replay: SACReplayBuffer, torch, device: Any) -> Optional[float]:
    if replay.size <= 0:
        return None
    q_values = []
    realized = np.asarray(
        [float(record["realized_return"]) for record in replay.identity_records],
        dtype=np.float64,
    )
    with torch.no_grad():
        for start in range(0, replay.size, 128):
            end = min(replay.size, start + 128)
            depth = torch.from_numpy(np.asarray(replay.arrays["depth"][start:end])).to(device=device, dtype=torch.float32)
            vector = torch.from_numpy(np.asarray(replay.arrays["vector"][start:end])).to(device=device, dtype=torch.float32)
            action = torch.from_numpy(np.asarray(replay.arrays["action"][start:end])).to(device=device, dtype=torch.long)
            q = torch.minimum(agent.critic1(depth, vector), agent.critic2(depth, vector))
            q_values.append(q.gather(1, action[:, None]).squeeze(1).detach().cpu().numpy())
    predicted = np.concatenate(q_values).astype(np.float64)
    return _rank_correlation(predicted, realized)


def _write_training_metrics(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    fields = sorted({str(key) for row in values for key in row})
    if not fields:
        fields = ["environment_steps", "replay_size", "update_kind"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(_json_safe(dict(row)), sort_keys=True, allow_nan=False) + "\n"
            )


def _write_report(out_dir: Path, summary: Mapping[str, Any], *, error: str = "") -> None:
    residual = str(summary.get("actor_architecture", "direct")) == "residual"
    lines = [
        "# BC Residual Online Discrete SAC V5" if residual else "# BC-initialized Online Discrete SAC V1",
        "",
        "本目录只属于新的 SAC 主线；既有 AWAC、校准 Replay 与 checkpoint 不作为 SAC 输入。",
        "",
        "## 状态",
        "",
        "- TASK_EXECUTION_STATUS: `{}`".format(summary.get("status", "UNKNOWN")),
        "- bounded environment steps: `{}`".format(summary.get("online_env_steps", 0)),
        "- completed episodes: `{}`".format(summary.get("online_episodes", 0)),
        "- replay rows: `{}`".format(summary.get("replay_size", 0)),
        "- actor optimizer steps: `{}`".format(summary.get("actor_optimizer_steps", 0)),
        "- critic optimizer steps: `{}`".format(summary.get("critic_optimizer_steps", 0)),
        "- first actor update transition: `{}`".format(summary.get("first_actor_update_at_transition")),
        "",
        "## 约束",
        "",
        "- fresh twin-Q；不加载 AWAC/N5/TD1/ranking Critic。",
        "- masked categorical 105-action policy；无效 action 概率为零。",
        "- learning_starts 前 Actor optimizer step 必须为 0。",
        "- 本次未修改生产 AWAC、BC checkpoint 或历史 Replay。",
    ]
    if residual:
        lines.extend(
            [
                "- residual logit cap: `0.49`",
                "- theoretical masked KL bound: `0.98`",
                "- BC reference is frozen and excluded from the Actor optimizer.",
                "- Actor behavior before learning_starts is exactly frozen BC.",
            ]
        )
    if error:
        lines.extend(["", "## 阻塞/异常", "", "```text", error, "```"])
    (out_dir / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> int:
    if int(args.env_workers) <= 0 or int(args.env_workers) > 16:
        raise ValueError("env-workers must be in 1..16 for this preregistered run")
    if int(args.max_env_steps) <= 0 or int(args.max_episodes) <= 0:
        raise ValueError("online budgets must be positive")
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        existing = {item.name for item in out_dir.iterdir()}
        allowed = {"bc_dev100", "v3_crossing_regression.json"}
        unexpected = sorted(existing - allowed)
        if unexpected:
            raise FileExistsError(
                "SAC output directory already contains training artifacts: {} ({})".format(
                    out_dir, ", ".join(unexpected)
                )
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    bc_path = Path(args.bc_checkpoint).expanduser().resolve()
    mission_path = Path(args.train_index).expanduser().resolve()
    worker_path = Path(args.worker_spec_file).expanduser().resolve()
    for path in (bc_path, mission_path, worker_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    if bool(args.deterministic_replay):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    import torch.nn as nn

    if bool(args.deterministic_replay):
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if int(args.cpu_threads) > 0:
        torch.set_num_threads(int(args.cpu_threads))
    device = _resolve_device(torch, str(args.device))
    determinism = dict(cuda_determinism_metadata(torch))
    _write_json(
        out_dir / "runtime_environment.json",
        {
            **determinism,
            "python": sys.version,
            "device_requested": str(args.device),
            "device_resolved": str(device),
            "cpu_threads": int(args.cpu_threads),
            "seed": int(args.seed),
        },
    )
    _write_json(
        out_dir / "determinism_contract.json",
        {
            "schema_id": "bc_initialized_discrete_sac_determinism_contract_v2",
            "enabled": bool(args.deterministic_replay),
            "global_python_numpy_torch_cuda_rng_captured": bool(args.deterministic_replay),
            "full_batch_indices_journaled": bool(args.deterministic_replay),
            "algorithm_source_not_changed_by_flag": True,
        },
    )
    checkpoint = _load_torch_checkpoint(torch, bc_path)
    _validate_inputs(checkpoint, path=bc_path, args=args)
    from planning.bc.model import VectorNormalizer
    from planning.awac.calibration_runtime import (
        calibration_mission_source_identity,
        load_calibration_missions,
    )
    from planning.awac.trainer import start_managed_calibration_runtime

    missions = load_calibration_missions(mission_path, max_steps=int(args.max_steps))
    mission_identity = calibration_mission_source_identity(mission_path, missions)
    worker_identity = _worker_identity(worker_path, int(args.env_workers))
    package_root = Path(__file__).resolve().parents[1]
    source_code_sha = algorithm_source_sha256(package_root, SAC_ALGORITHM_ID)
    config = _resolved_config(
        args,
        bc_sha=file_sha256(bc_path),
        mission_sha=str(mission_identity["sha256"]),
        source_code_sha=source_code_sha,
    )
    input_identity = {
        "bc_checkpoint_path": str(bc_path),
        "bc_checkpoint_sha256": file_sha256(bc_path),
        "bc_model_type": str(checkpoint.get("model_type", "")),
        "train_index_path": str(mission_path),
        "train_index_sha256": str(mission_identity["sha256"]),
        "train_index_row_count": len(missions),
        "worker_spec": worker_identity,
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": EXPECTED_TASK_SHA256,
        "mpl_contract_sha256": EXPECTED_MPL_SHA256,
        "observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "excluded_inputs": [
            "data/awac/",
            "data/awac/diagnostics/",
            "data/teach/2026_6w/bc_training/replay",
            "Teacher recovery replay",
            "multi_action replay",
        ],
    }
    _write_json(out_dir / "input_identity.json", input_identity)
    _write_json(
        out_dir / "training_config.json",
        {
            **config,
            "training_config_sha256": training_config_sha256(config),
        },
    )
    _write_json(
        out_dir / "preregistration.json",
        {
            "schema_id": "bc_initialized_discrete_sac_preregistration_v1",
            "hypothesis": "online discrete SAC initialized from frozen BC can improve the same formal Dev100 policy",
            "actor_architecture": str(args.actor_architecture),
            "residual_logit_cap": float(SAC_RESIDUAL_LOGIT_CAP)
            if args.actor_architecture == "residual"
            else None,
            "primary_endpoint": "same-task deterministic Dev100 paired gain/loss/net_gain",
            "max_environment_steps": int(args.max_env_steps),
            "max_complete_episodes": int(args.max_episodes),
            "worker_cap": 16,
            "learning_starts": int(args.learning_starts),
            "actor_frozen_before_learning_starts": True,
            "no_hyperparameter_sweep": True,
            "no_old_replay": True,
            "no_awac_or_cql": True,
            "actor_kl_trust_region": True,
            "kl_hard_stop": float(args.bc_kl_hard_stop),
            "kl_backtracking_factors": list(SAC_KL_BACKTRACKING_FACTORS),
            "sentinel_contract": "pre_actor_unique_state_observation_v4",
            "unsafe_proposal_checkpointing": False,
            "resolved_training_config_sha256": training_config_sha256(config),
            "seed": int(args.seed),
        },
    )
    _write_json(
        out_dir / "implementation_audit.json",
        {
            "algorithm_id": SAC_ALGORITHM_ID,
            "actor_architecture": str(args.actor_architecture),
            "residual_logit_cap": float(SAC_RESIDUAL_LOGIT_CAP)
            if args.actor_architecture == "residual"
            else None,
            "algorithm_owner": "planning.sac",
            "runtime_owner": "planning.runtime.managed_runtime + planning.awac.calibration_runtime lifecycle only",
            "fresh_twin_q": True,
            "old_critic_checkpoint_loaded": False,
            "old_replay_loaded": False,
            "actor_initialization": "frozen BC model_state_dict",
            "bc_reference_frozen": True,
            "invalid_action_probability": 0.0,
            "source_code_sha256": source_code_sha,
        },
    )
    if args.actor_architecture == "residual":
        _write_json(
            out_dir / "residual_actor_contract.json",
            {
                "schema_id": "bc_residual_online_discrete_sac_actor_contract_v5",
                "actor_architecture": "frozen_bc_plus_bounded_residual_logits",
                "base_policy": "frozen_bc_logits",
                "residual_policy_id": SAC_RESIDUAL_POLICY_ID,
                "residual_logit_cap": float(SAC_RESIDUAL_LOGIT_CAP),
                "residual_parameter_initialization": "copied_bc_encoder_and_zero_final_head",
                "residual_raw_initial_value": 0.0,
                "optimizer_parameters": "residual_model_only",
                "bc_reference_frozen": True,
                "bc_reference_excluded_from_optimizer": True,
                "mask_applied_after_logit_sum": True,
            },
        )
        _write_json(
            out_dir / "kl_bound_proof.json",
            {
                "schema_id": "bc_residual_online_discrete_sac_kl_bound_v5",
                "residual_logit_cap": float(SAC_RESIDUAL_LOGIT_CAP),
                "theoretical_kl_bound": float(2.0 * SAC_RESIDUAL_LOGIT_CAP),
                "hard_stop_threshold": float(args.bc_kl_hard_stop),
                "proof": "bounded per-action logit delta has range at most 2c; masked categorical KL is bounded by that range",
                "numeric_masked_random_check": "PENDING_AGENT_PREFLIGHT",
            },
        )
    _write_json(
        out_dir / "trust_region_contract.json",
        {
            "schema_id": "bc_initialized_discrete_sac_kl_trust_region_contract_v4",
            "hard_stop_threshold": float(args.bc_kl_hard_stop),
            "base_actor_lr": float(args.actor_lr),
            "factors": list(SAC_KL_BACKTRACKING_FACTORS),
            "gradient_recomputed_per_factor": False,
            "batch_resampled_per_factor": False,
            "sentinel_selection": "all_unique_pre_actor_committed_observation_states",
            "sentinel_selection_uses_return_or_q": False,
            "sentinel_fixed_after_first_actor": True,
            "unsafe_proposal_checkpointed": False,
        },
    )
    regression_path = out_dir / "v3_crossing_regression.json"
    if regression_path.exists():
        regression = json.loads(regression_path.read_text(encoding="utf-8"))
        if regression.get("status") != "PASS":
            raise RuntimeError("V4 runtime requires a passing offline V3 crossing regression")
    else:
        if args.actor_architecture == "residual":
            source_v3 = (
                package_root
                / "data/sac/diagnostics/discrete_sac_first_kl_crossing_v3_20260908/first_crossing.json"
            )
            if not source_v3.is_file():
                raise FileNotFoundError(str(source_v3))
            source_crossing = json.loads(source_v3.read_text(encoding="utf-8"))
            sentinel = source_crossing.get("sentinel", {})
            if (
                sentinel.get("status") != "CROSSING_FOUND"
                or int(sentinel.get("actor_step", -1)) != 143
                or abs(float(sentinel.get("kl_before", 0.0)) - 0.9461058378219604) > 1.0e-6
                or abs(float(sentinel.get("kl_after", 0.0)) - 1.0449196100234985) > 1.0e-6
            ):
                raise RuntimeError("V3 source crossing evidence is not the frozen expected result")
            regression = {
                "status": "PASS",
                "source": str(source_v3),
                "source_sha256": file_sha256(source_v3),
                "sentinel": sentinel,
                "not_recomputed_in_v5": True,
            }
            _write_json(regression_path, regression)
        else:
            _write_json(
                regression_path,
                {
                    "status": "PENDING_OFFLINE_REGRESSION",
                    "source": "discrete_sac_first_kl_crossing_v3_20260908",
                },
            )
    if bool(args.preflight_only):
        _write_json(out_dir / "preflight.json", {"status": "PASS", "device": str(device)})
        _write_report(out_dir, {"status": "PREFLIGHT_PASS"})
        return 0

    normalizer = VectorNormalizer.from_checkpoint(dict(checkpoint))
    agent = DiscreteSACAgent(
        torch=torch,
        nn=nn,
        device=device,
        bc_checkpoint=checkpoint,
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        gamma=float(args.gamma),
        tau=float(args.tau),
        alpha=float(args.alpha),
        beta_bc=float(args.beta_bc),
        reward_scale=float(args.reward_scale),
        gradient_clip_norm=float(args.gradient_clip_norm),
        entropy_floor=float(args.entropy_floor),
        deterministic_pool=bool(args.deterministic_replay),
        actor_architecture=str(args.actor_architecture),
        residual_logit_cap=float(SAC_RESIDUAL_LOGIT_CAP),
    )
    if args.actor_architecture == "residual":
        bound_probe = _residual_kl_bound_probe(torch, agent, device)
        _write_json(
            out_dir / "kl_bound_proof.json",
            {
                "schema_id": "bc_residual_online_discrete_sac_kl_bound_v5",
                "residual_logit_cap": float(SAC_RESIDUAL_LOGIT_CAP),
                "theoretical_kl_bound": float(2.0 * SAC_RESIDUAL_LOGIT_CAP),
                "hard_stop_threshold": float(args.bc_kl_hard_stop),
                "proof": "bounded per-action logit delta has range at most 2c; masked categorical KL is bounded by that range",
                "numeric_masked_random_check": bound_probe,
            },
        )
    replay = SACReplayBuffer.create(
        out_dir / "replay",
        capacity=int(args.max_env_steps),
        depth_shape=(90, 160),
        vector_dim=POLICY_VECTOR_DIM,
        action_dim=NUM_ACTIONS,
        provenance={
            "source_bc_checkpoint_sha256": file_sha256(bc_path),
            "mission_source_sha256": str(mission_identity["sha256"]),
            "task_contract_sha256": EXPECTED_TASK_SHA256,
            "mpl_contract_sha256": EXPECTED_MPL_SHA256,
            "observation_contract": EXPECTED_OBSERVATION_CONTRACT,
            "behavior_policy_id": SAC_RESIDUAL_POLICY_ID
            if args.actor_architecture == "residual"
            else SAC_POLICY_ID,
            "old_replay_used": False,
        },
    )
    journal = SACUpdateJournal(out_dir / "training_step_journal.jsonl")
    _write_json(
        out_dir / "checkpoint_state_contract.json",
        {
            "schema_id": "bc_initialized_discrete_sac_checkpoint_state_v2",
            "model_state": "actor_bc_critics_targets",
            "optimizer_state": "actor_critic",
            "rng_state": "python_numpy_global_replay_sampler_torch_cpu_cuda_workers",
            "transition_order": "committed_identity_order",
            "journal": journal.schema_id,
        },
    )
    checkpoint_records = []
    saved_targets = set()

    def save_at(path: Path, *, environment_steps: int, completed_episodes: int) -> str:
        payload = build_checkpoint_payload(
            agent=agent,
            torch=torch,
            bc_checkpoint=checkpoint,
            bc_checkpoint_sha256=file_sha256(bc_path),
            resolved_training_config=config,
            replay=replay,
            environment_steps=int(environment_steps),
            completed_episodes=int(completed_episodes),
            first_actor_update_at_transition=runner.first_actor_update_at_transition if runner is not None else None,
            training_metrics={
                "actor_update_count": int(agent.actor_update_count),
                "critic_update_count": int(agent.critic_update_count),
                "last_update": runner.training_rows[-1] if runner is not None and runner.training_rows else {},
            },
            mission_source_sha256=str(mission_identity["sha256"]),
            source_code_sha256=source_code_sha,
            rng_state=_rng_state(
                torch,
                runner.rng if runner is not None else np.random.RandomState(int(args.seed)),
                runner,
            ),
            runtime_state=runner.runtime_state_snapshot() if runner is not None else {},
            journal_state=journal.checkpoint_state(),
            transition_order=runner.transition_order_snapshot() if runner is not None else {},
            cuda_determinism=determinism,
        )
        sha = save_checkpoint(torch, payload, path)
        checkpoint_records.append(
            {
                "path": str(path.relative_to(out_dir)),
                "sha256": sha,
                "environment_steps": int(environment_steps),
                "completed_episodes": int(completed_episodes),
                "replay_size": int(replay.size),
                "actor_update_count": int(agent.actor_update_count),
                "critic_update_count": int(agent.critic_update_count),
            }
        )
        return sha

    runner: Optional[SACOnlineRunner] = None
    managed_runtime = None
    pool = None
    result: Dict[str, Any] = {}
    runtime_error = ""
    try:
        save_at(out_dir / "checkpoint_step_00000000.pt", environment_steps=0, completed_episodes=0)
        saved_targets.add(0)
        managed_runtime, pool, managed_identity = start_managed_calibration_runtime(
            worker_path,
            worker_count=int(args.env_workers),
            output_dir=out_dir,
            task_contract_sha256=EXPECTED_TASK_SHA256,
            mpl_contract_sha256=EXPECTED_MPL_SHA256,
            max_steps=int(args.max_steps),
            startup_timeout_s=float(args.startup_timeout),
            request_timeout_s=float(args.request_timeout),
        )

        def snapshot_callback(owner: SACOnlineRunner) -> None:
            current = int(owner._environment_step_count)
            for target in SNAPSHOT_TARGETS[1:]:
                if target in saved_targets or current < target:
                    continue
                path = out_dir / "checkpoint_step_{:08d}.pt".format(target)
                save_at(path, environment_steps=current, completed_episodes=owner._completed_episode_count)
                saved_targets.add(target)

        runner = SACOnlineRunner(
            missions=missions,
            pool=pool,
            replay=replay,
            agent=agent,
            normalizer=normalizer,
            torch=torch,
            device=device,
            seed=int(args.seed),
            batch_size=int(args.batch_size),
            learning_starts=int(args.learning_starts),
            critic_updates_per_transition=float(args.critic_updates_per_transition),
            actor_updates_per_transition=float(args.actor_updates_per_transition),
            max_env_steps=int(args.max_env_steps),
            max_episodes=int(args.max_episodes),
            expected_runtime_ids={
                index: value for index, value in enumerate(worker_identity["runtime_instance_ids"])
            },
            output_dir=out_dir,
            snapshot_callback=snapshot_callback,
            bc_kl_hard_stop=float(args.bc_kl_hard_stop),
            entropy_floor=float(args.entropy_floor),
            journal=journal,
        )
        runner.mission_source_identity = dict(mission_identity)
        runner._runtime_ids = {}
        result = runner.run()
        save_at(
            out_dir / "checkpoint_final.pt",
            environment_steps=int(runner._environment_step_count),
            completed_episodes=int(runner._completed_episode_count),
        )
    except Exception as error:
        runtime_error = "{}: {}".format(type(error).__name__, error)
        traceback.print_exc()
        if runner is not None:
            result = {
                "status": "RUNTIME_FAILED",
                "environment_step_count": int(runner._environment_step_count),
                "completed_episode_count": int(runner._completed_episode_count),
                "stop_reason": runtime_error,
            }
            try:
                save_at(
                    out_dir / "checkpoint_failure.pt",
                    environment_steps=int(runner._environment_step_count),
                    completed_episodes=int(runner._completed_episode_count),
                )
            except Exception:
                pass
        else:
            result = {"status": "RUNTIME_START_FAILED", "environment_step_count": 0, "completed_episode_count": 0}
    finally:
        if managed_runtime is not None:
            close = getattr(managed_runtime, "close", None)
            if callable(close):
                close()
        if journal is not None:
            journal.close()
            write_batch_index_npz(
                out_dir / "training_step_journal.jsonl",
                out_dir / "batch_index_journal.npz",
            )

    audit = replay.audit()
    q_return_rho = _critic_return_rho(agent, replay, torch, device) if not runtime_error else None
    drift_records = replay.identity_records
    drift = {
        "rows": len(drift_records),
        "mean_bc_kl": float(np.mean([item["bc_kl"] for item in drift_records])) if drift_records else None,
        "max_bc_kl": float(np.max([item["bc_kl"] for item in drift_records])) if drift_records else None,
        "mean_policy_entropy": float(np.mean([item["policy_entropy"] for item in drift_records])) if drift_records else None,
        "action_flip_rate": float(np.mean([float(item["argmax_flip"]) for item in drift_records])) if drift_records else None,
        "warmup_rows": sum("warmup" in item["behavior_policy_version"] for item in drift_records),
    }
    terminal_counts = {
        "success": sum(bool(item.get("success")) for item in runner.episode_summaries) if runner else 0,
        "collision": sum(bool(item.get("collision")) for item in runner.episode_summaries) if runner else 0,
        "dead_end": sum(bool(item.get("dead_end")) for item in runner.episode_summaries) if runner else 0,
        "timeout": sum(bool(item.get("timeout")) for item in runner.episode_summaries) if runner else 0,
    }
    summary = {
        "schema_id": "bc_initialized_discrete_sac_rollout_summary_v1",
        "actor_architecture": str(args.actor_architecture),
        "actor_policy_id": str(agent.actor_policy_id),
        "residual_logit_cap": float(SAC_RESIDUAL_LOGIT_CAP)
        if args.actor_architecture == "residual"
        else None,
        "status": "RUNTIME_FAILED" if runtime_error else "BOUNDED_COMPLETE",
        "runtime_error": runtime_error,
        "online_env_steps": int(result.get("environment_step_count", 0)),
        "online_episodes": int(result.get("completed_episode_count", 0)),
        "replay_size": int(replay.size),
        "actor_optimizer_steps": int(agent.actor_optimizer_step_count),
        "critic_optimizer_steps": int(agent.critic_optimizer_step_count),
        "actor_proposal_count": int(agent.actor_proposal_count),
        "actor_rejection_count": int(agent.actor_rejection_count),
        "actor_recovery_update_count": int(agent.actor_recovery_update_count),
        "first_actor_update_at_transition": runner.first_actor_update_at_transition if runner else None,
        "terminal_counts": terminal_counts,
        "managed_runtime_identity": _json_safe(locals().get("managed_identity", {})),
        "worker_identity": worker_identity,
        "behavior_policy": runner.behavior_policy if runner else {},
        "observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        "task_contract_sha256": EXPECTED_TASK_SHA256,
        "mpl_contract_sha256": EXPECTED_MPL_SHA256,
        "nan_count": int(agent.nan_count),
        "invalid_action_count": int(agent.invalid_action_count),
        "accepted_actor_updates": int(agent.actor_update_count),
        "rejected_actor_proposals": int(agent.actor_rejection_count),
        "max_sentinel_kl": float(runner.max_sentinel_kl) if runner else 0.0,
        "sentinel_frozen": bool(runner.sentinel_frozen) if runner else False,
        "sentinel_manifest_sha256": str(runner.sentinel_manifest_sha256) if runner else "",
        "sentinel_state_count": int(runner.sentinel_manifest.get("state_count", 0)) if runner else 0,
        "trust_region_factor_counts": dict(runner.trust_region_factor_counts) if runner else {},
    }
    _write_json(out_dir / "rollout_summary.json", summary)
    _write_json(out_dir / "replay_audit.json", audit)
    _write_json(out_dir / "critic_health.json", {"q_return_spearman": q_return_rho, "q_return_rho_status": "MEASURED" if q_return_rho is not None else "UNKNOWN"})
    _write_json(out_dir / "bc_policy_drift.json", drift)
    actor_rows = [
        row for row in (runner.training_rows if runner else [])
        if str(row.get("update_kind", "")).startswith("actor")
    ]
    if args.actor_architecture == "residual":
        residual_rows = [row for row in actor_rows if row.get("architecture") == "residual"]
        residual_summary = {
            "schema_id": "bc_residual_online_discrete_sac_residual_statistics_v5",
            "proposal_rows": len(residual_rows),
            "residual_abs_mean": float(
                np.mean([float(row.get("residual_abs_mean", 0.0)) for row in residual_rows])
            )
            if residual_rows
            else 0.0,
            "residual_abs_p99": float(
                np.max([float(row.get("residual_abs_p99", 0.0)) for row in residual_rows])
            )
            if residual_rows
            else 0.0,
            "residual_abs_max": float(
                np.max([float(row.get("residual_abs_max", 0.0)) for row in residual_rows])
            )
            if residual_rows
            else 0.0,
            "residual_saturation_fraction": float(
                np.mean(
                    [
                        float(row.get("residual_saturation_fraction", 0.0))
                        for row in residual_rows
                    ]
                )
            )
            if residual_rows
            else 0.0,
            "rows": residual_rows,
        }
        _write_json(out_dir / "residual_statistics.json", residual_summary)
        summary.update(
            {
                "residual_abs_mean": residual_summary["residual_abs_mean"],
                "residual_abs_p99": residual_summary["residual_abs_p99"],
                "residual_saturation_fraction": residual_summary[
                    "residual_saturation_fraction"
                ],
            }
        )
        _write_json(out_dir / "rollout_summary.json", summary)
    _write_jsonl(out_dir / "actor_proposal_journal.jsonl", actor_rows)
    factor_counts = dict(runner.trust_region_factor_counts) if runner else {
        str(factor): 0 for factor in SAC_KL_BACKTRACKING_FACTORS
    }
    _write_json(
        out_dir / "backtracking_statistics.json",
        {
            "schema_id": "bc_initialized_discrete_sac_backtracking_statistics_v4",
            "factor_attempt_counts": factor_counts,
            "actor_proposals": int(agent.actor_proposal_count),
            "accepted_actor_updates": int(agent.actor_update_count),
            "rejected_actor_proposals": int(agent.actor_rejection_count),
            "max_sentinel_kl": float(runner.max_sentinel_kl) if runner else 0.0,
        },
    )
    _write_json(
        out_dir / "optimizer_momentum_metrics.json",
        {
            "schema_id": "bc_initialized_discrete_sac_optimizer_momentum_metrics_v4",
            "rows": [
                {
                    "actor_proposal_id": row.get("actor_proposal_id"),
                    "accepted_actor_update_id": row.get("accepted_actor_update_id"),
                    "accepted_lr_factor": row.get("accepted_lr_factor"),
                    "optimizer_exp_avg_norm": row.get("optimizer_exp_avg_norm"),
                    "optimizer_exp_avg_sq_norm": row.get("optimizer_exp_avg_sq_norm"),
                    "gradient_adam_exp_avg_cosine": row.get("gradient_adam_exp_avg_cosine"),
                    "accepted": row.get("accepted", False),
                }
                for row in actor_rows
            ],
        },
    )
    _write_json(
        out_dir / "integrity_audit.json",
        {
            "bc_checkpoint_sha256": file_sha256(bc_path),
            "bc_checkpoint_unchanged": file_sha256(bc_path) == EXPECTED_BC_SHA256,
            "production_awac_modified": False,
            "old_replay_loaded": False,
            "sentinel_frozen": bool(runner.sentinel_frozen) if runner else False,
            "sentinel_manifest_sha256": str(runner.sentinel_manifest_sha256) if runner else "",
            "unsafe_proposal_checkpointed": False,
            "replay_unchanged_by_actor": all(
                bool(row.get("replay_unchanged", True)) for row in actor_rows
            ),
        },
    )
    _write_json(
        out_dir / "dev100_comparison.json",
        {"status": "NOT_RUN_UNTIL_BOUNDED_SAFETY_PASS", "bc": "NOT_RUN", "sac_final": "NOT_RUN"},
    )
    _write_json(out_dir / "new_run_summary.json", summary)
    _write_json(
        out_dir / "rng_state_manifest.json",
        {
            "schema_id": "bc_initialized_discrete_sac_rng_state_manifest_v2",
            "checkpoint_count": len(checkpoint_records),
            "journal_record_count": journal.record_count,
            "captured": bool(args.deterministic_replay),
        },
    )
    _write_json(
        out_dir / "rollback_audit.json",
        {
            "schema_id": "bc_initialized_discrete_sac_rollback_audit_v2",
            "journaled_rejected_proposals": True,
            "accepted_actor_updates": int(agent.actor_update_count),
            "rejected_actor_proposals": int(agent.actor_rejection_count),
            "critic_optimizer_step_count": int(agent.critic_optimizer_step_count),
            "original_exception_preserved": True,
        },
    )
    _write_json(
        out_dir / "kl_spike_analysis.json",
        {
            "status": "PENDING_EXACT_REPLAY",
            "source": "training_step_journal.jsonl",
            "hard_stop_threshold": float(args.bc_kl_hard_stop),
        },
    )
    _write_json(out_dir / "one_step_counterfactual.json", {"status": "NOT_RUN_UNTIL_EXACT_REPLAY_PASS"})
    _write_json(out_dir / "root_cause.json", {"status": "PENDING_DETERMINISTIC_REPLAY_AUDIT"})
    _write_training_metrics(out_dir / "training_metrics.csv", runner.training_rows if runner else [])
    write_checkpoint_manifest(out_dir, checkpoint_records)
    changed_files = [
        "python/planning/sac/contract.py",
        "python/planning/sac/network.py",
        "python/planning/sac/replay.py",
        "python/planning/sac/checkpoint.py",
        "python/planning/sac/runtime.py",
        "python/planning/contracts/offpolicy.py",
        "scripts/train_discrete_sac.py",
    ]
    if args.actor_architecture == "residual":
        changed_files.extend(
            [
                "tests/test_sac_residual_v5.py",
            ]
        )
    (out_dir / "code_changes.diff").write_text(
        "# No Git repository is present; this is an additive source identity manifest.\n"
        + "\n".join("{}  {}".format(file_sha256(package_root / path), path) for path in changed_files)
        + "\n",
        encoding="utf-8",
    )
    _write_report(out_dir, summary, error=runtime_error)
    replay.close()
    return 1 if runtime_error else 0


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
