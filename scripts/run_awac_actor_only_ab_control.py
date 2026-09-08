#!/usr/bin/env python3
"""Run the fixed-Critic A/B/C AWAC Actor-only control experiment.

This owner deliberately has no Unity/ROS runtime path.  It consumes a newly
certified calibration replay read-only, creates non-resumable evaluation-only
Actor snapshots, and emits the exact Dev100/paired-comparison plan.  Formal
Dev100 remains owned by ``evaluate_policy_unity_managed.sh``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from planning.awac.calibration_runtime import (
    calibration_behavior_policy_identity,
)
from planning.awac.checkpoint import (
    calibration_replay_identity,
    load_calibration_pass_checkpoint,
)
from planning.awac.learner import AWACOptimizationConfig, resolve_torch_device
from planning.awac.replay import AWACReplayBuffer
from planning.awac.trainer import build_learner, validate_bc_checkpoint_for_awac
from planning.bc.model import require_torch
from planning.common.checkpoint import load_torch
from planning.common.hashing import file_sha256
from planning.contracts.offpolicy import validate_policy_checkpoint_algorithm
from planning.contracts.policy_checkpoint_fingerprint import actor_state_sha256


EXPERIMENT_SCHEMA_ID = "awac_fixed_critic_actor_only_ab_control_v1"
EVALUATION_CHECKPOINT_SCHEMA_ID = "awac_actor_only_evaluation_checkpoint_v1"
CONDITIONS = ("A", "B", "C")
SNAPSHOT_STEPS = (0, 50, 100)
COMPARISONS = (
    ("A", "B50"),
    ("A", "B100"),
    ("A", "C50"),
    ("A", "C100"),
    ("B50", "C50"),
    ("B100", "C100"),
)
OUTCOMES = ("success", "collision", "dead_end", "timeout", "far", "hard_altitude")
MANIFEST_NAME = "manifest.json"
TERMINAL_REASON_PROVENANCE = "not_stored_in_frozen_awac_replay_layout"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp".format(path.name))
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _hash_value(digest, value: Any) -> None:
    """Hash nested optimizer/model state without relying on pickle bytes."""

    if hasattr(value, "detach") and hasattr(value, "dtype"):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"sequence\0")
        digest.update(str(len(value)).encode("ascii"))
        digest.update(b"\0")
        for item in value:
            _hash_value(digest, item)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(value).encode("utf-8"))
    digest.update(b"\0")


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"awac_actor_only_control_state_v1\0")
    _hash_value(digest, value)
    return digest.hexdigest()


def replay_content_sha256(replay: AWACReplayBuffer) -> str:
    """Hash every committed replay row while retaining the read-only owner."""

    digest = hashlib.sha256()
    digest.update(b"awac_actor_only_control_replay_v1\0")
    digest.update(json.dumps(replay.metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    row_count = int(replay.size)
    for name in sorted(replay.arrays):
        array = replay.arrays[name]
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape[1:]), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        for offset in range(0, row_count, 128):
            value = np.ascontiguousarray(array[offset : min(row_count, offset + 128)])
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_fixed_batch_sequences(
    *, replay_size: int, proposal_count: int, batch_size: int, seed: int
) -> Dict[str, np.ndarray]:
    """Create the exact train/trust index sequences shared by B and C."""

    if int(replay_size) <= 0:
        raise ValueError("replay_size must be positive")
    if int(proposal_count) != 100:
        raise ValueError("the controlled A/B/C contract requires exactly 100 proposals")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    rng = np.random.RandomState(int(seed))
    return {
        "train_indices": rng.randint(
            0, int(replay_size), size=(int(proposal_count), int(batch_size))
        ).astype(np.int64),
        "trust_indices": rng.randint(
            0, int(replay_size), size=(int(proposal_count), int(batch_size))
        ).astype(np.int64),
        # This third, fixed batch is not part of either branch's optimization
        # sequence.  It gives the 0/50/100 snapshots an apples-to-apples
        # diagnostic view without leaking holdout rows into a trust decision.
        "validation_indices": rng.randint(
            0, int(replay_size), size=(int(batch_size),)
        ).astype(np.int64),
    }


def _scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Persist every finite learner scalar, rather than a curated subset."""

    result = {}
    for name, value in metrics.items():
        number = float(value)
        if not math.isfinite(number):
            raise FloatingPointError(
                "Actor-only metric {} is non-finite".format(name)
            )
        result[str(name)] = number
    return result


def _ordered_values_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"awac_actor_only_ordered_values_v1\0")
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _working_tree_identity() -> Dict[str, Any]:
    """Record VCS truth without treating a non-Git deployment as clean."""

    root = Path(__file__).resolve().parents[1]
    result: Dict[str, Any] = {
        "repository_root": str(root),
        "source_files": {
            str(relative): file_sha256(root / relative)
            for relative in (
                "python/planning/awac/calibration_runtime.py",
                "python/planning/awac/trainer.py",
                "python/planning/awac/learner.py",
                "scripts/run_awac_actor_only_ab_control.py",
            )
        },
    }
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        result.update(
            {
                "vcs_status": "git_unavailable",
                "vcs_error": "{}: {}".format(type(error).__name__, error),
            }
        )
        return result
    if commit.returncode != 0:
        result.update(
            {
                "vcs_status": "not_a_git_repository",
                "git_commit": "NOT_AVAILABLE",
                "working_tree_diff_sha256": "NOT_AVAILABLE",
                "working_tree_diff": "NOT_AVAILABLE",
            }
        )
        return result
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
        check=False,
        capture_output=True,
    )
    if diff.returncode != 0:
        raise RuntimeError("git diff failed while creating experiment identity")
    result.update(
        {
            "vcs_status": "git_repository",
            "git_commit": commit.stdout.strip(),
            "working_tree_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
            "working_tree_diff_bytes": int(len(diff.stdout)),
        }
    )
    return result


def _capture_rng_state(torch) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda": [],
    }
    if result["cuda_available"]:
        result["cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return result


def _restore_rng_state(torch, state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].clone())
    if bool(state["cuda_available"]):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG identity changed during controlled experiment")
        torch.cuda.set_rng_state_all([item.clone() for item in state["cuda"]])


def _rng_identity(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "python_sha256": state_sha256(state["python"]),
        "numpy_sha256": state_sha256(state["numpy"]),
        "torch_cpu_sha256": state_sha256(state["torch_cpu"]),
        "cuda_available": bool(state["cuda_available"]),
        "cuda_sha256": state_sha256(state["cuda"]),
    }


def _learner_fixed_state(learner) -> Dict[str, Any]:
    return {
        "critic1": state_sha256(learner.critic1.state_dict()),
        "critic2": state_sha256(learner.critic2.state_dict()),
        "target_critic1": state_sha256(learner.target_critic1.state_dict()),
        "target_critic2": state_sha256(learner.target_critic2.state_dict()),
        "bc_reference": state_sha256(learner.bc_reference.state_dict()),
        "critic_optimizer": state_sha256(learner.critic_optimizer.state_dict()),
        "critic_update_count": int(learner.critic_update_count),
        "update_step": int(learner.update_step),
    }


def _learner_branch_initial_state(learner) -> Dict[str, Any]:
    """Fingerprint every state B/C must share before their sole intervention."""

    return {
        "actor": state_sha256(learner.actor.state_dict()),
        "actor_optimizer": state_sha256(learner.actor_optimizer.state_dict()),
        "fixed_critic_state": _learner_fixed_state(learner),
    }


def _assert_fixed_state_unchanged(
    *, before: Mapping[str, Any], learner, condition: str, proposal: int
) -> None:
    after = _learner_fixed_state(learner)
    if dict(before) != after:
        changed = {
            key: {"before": before.get(key), "after": after.get(key)}
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        }
        raise RuntimeError(
            "Actor-only {} proposal {} changed fixed-Critic state: {}".format(
                condition, int(proposal), json.dumps(changed, sort_keys=True)
            )
        )


def _optimization_config_from_calibration(payload: Mapping[str, Any]) -> AWACOptimizationConfig:
    resolved = payload.get("resolved_training_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("calibration checkpoint resolved training config is missing")
    raw = resolved.get("optimization_config", {})
    if not isinstance(raw, Mapping):
        raise ValueError("calibration optimization config is missing")
    allowed = set(AWACOptimizationConfig.__dataclass_fields__)
    values = {name: raw[name] for name in raw if name in allowed}
    for field in (
        "enable_twin_q_confidence",
        "enable_adaptive_bc_kl",
        "enable_primitive_neighbor_exploration",
    ):
        if bool(values.get(field, False)):
            raise ValueError(
                "fixed-Critic A/B/C requires {}=False in the calibration source".format(
                    field
                )
            )
        values[field] = False
    config = AWACOptimizationConfig(**values)
    if (
        bool(config.enable_twin_q_confidence)
        or bool(config.enable_adaptive_bc_kl)
        or bool(config.enable_primitive_neighbor_exploration)
    ):
        raise AssertionError("controlled innovation modules must be disabled")
    return config


def _validate_calibration_behavior_policy(
    *, calibration: Mapping[str, Any], bc_checkpoint_sha256: str
) -> Dict[str, Any]:
    metrics = calibration.get("calibration_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("calibration metrics are missing")
    if str(metrics.get("state", calibration.get("calibration_gate_state", ""))) != "PASS":
        raise ValueError("Actor-only experiment requires a calibration gate PASS")
    policy = metrics.get("behavior_policy")
    if not isinstance(policy, Mapping):
        raise ValueError(
            "calibration checkpoint lacks the masked-categorical behavior policy identity"
        )
    expected = calibration_behavior_policy_identity(
        checkpoint_sha256=str(bc_checkpoint_sha256),
        rng_seed=policy.get("rng_seed"),
    )
    if dict(policy) != expected:
        raise ValueError("calibration behavior policy is not frozen BC masked categorical T=1")
    progress = calibration.get("mission_progress", {})
    if isinstance(progress, Mapping) and progress.get("behavior_policy") is not None:
        if dict(progress["behavior_policy"]) != expected:
            raise ValueError("calibration mission progress behavior policy mismatch")
    return expected


def _copy_actor_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state.items()
    }


def _controlled_source_code_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/run_awac_actor_only_ab_control.py",
        "python/planning/awac/learner.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _evaluation_checkpoint_payload(
    *,
    calibration: Mapping[str, Any],
    learner,
    condition: str,
    proposal: int,
    calibration_checkpoint_sha256: str,
    fixed_state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a minimal evaluator-valid, explicitly non-resumable snapshot."""

    root_fields = (
        "awac_checkpoint_schema_id",
        "algorithm_id",
        "awac_checkpoint_contract_id",
        "model_type",
        "training_config_contract_id",
        "software_version",
        "feature_contract_id",
        "policy_input_contract_sha256",
        "policy_runtime_contract_id",
        "task_contract_id",
        "task_contract_schema_version",
        "task_contract_sha256",
        "max_primitive_steps",
        "reward_contract_id",
        "reward_contract_sha256",
        "observation_contract",
        "observation_source",
        "vec_dim",
        "num_actions",
        "depth_history_frames",
        "initial_prev_action",
        "mpl_contract_sha256",
        "resolved_training_config",
        "resolved_training_config_sha256",
        "training_contract",
        "training_contract_sha256",
        "source_code_sha256",
        "split_manifest_sha256",
        "source_bc_checkpoint_sha256",
        "bc_checkpoint_sha256",
        "bc_reference_fingerprint",
    )
    missing = [field for field in root_fields if field not in calibration]
    if missing:
        raise ValueError("calibration checkpoint lacks evaluator identity: {}".format(", ".join(missing)))
    payload = {field: copy.deepcopy(calibration[field]) for field in root_fields}
    payload.update(
        {
            "evaluation_checkpoint_schema_id": EVALUATION_CHECKPOINT_SCHEMA_ID,
            "evaluation_only": True,
            "resume_allowed": False,
            "phase": "actor_only_control",
            "actor_update_enabled": True,
            "actor_frozen_for_calibration": False,
            "actor_state_dict": _copy_actor_state(learner.actor.state_dict()),
            "actor_optimizer_state_dict": copy.deepcopy(
                learner.actor_optimizer.state_dict()
            ),
            "actor_update_count": int(learner.actor_update_count),
            "actor_awac_update_count": int(learner.actor_awac_update_count),
            "actor_recovery_update_count": int(
                learner.actor_recovery_update_count
            ),
            "actor_trust_region_rejection_count": int(
                learner.actor_trust_region_rejection_count
            ),
            "actor_optimizer_step_count": int(learner.actor_optimizer_step_count),
            "source_calibration_checkpoint_sha256": str(
                calibration_checkpoint_sha256
            ),
            "controlled_actor_only_experiment": {
                "schema_id": EXPERIMENT_SCHEMA_ID,
                "condition": str(condition),
                "proposal": int(proposal),
                "critic_optimizer_step": False,
                "target_soft_update": False,
                "fixed_state": dict(fixed_state),
            },
            "controlled_actor_only_source_code_sha256": _controlled_source_code_sha256(),
        }
    )
    payload["actor_state_sha256"] = actor_state_sha256(payload)
    validate_policy_checkpoint_algorithm(payload, "actor_state_dict")
    return payload


def _save_torch_atomic(*, torch, path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp".format(path.name))
    try:
        torch.save(dict(payload), str(temporary))
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _snapshot_path(out_dir: Path, condition: str, proposal: int) -> Path:
    return out_dir / "checkpoints" / "{}_proposal_{:03d}.pt".format(condition, proposal)


def _save_snapshot(
    *,
    out_dir: Path,
    condition: str,
    proposal: int,
    calibration: Mapping[str, Any],
    calibration_checkpoint_sha256: str,
    learner,
    fixed_state: Mapping[str, Any],
    torch,
) -> Dict[str, Any]:
    path = _snapshot_path(out_dir, condition, proposal)
    if path.exists():
        raise FileExistsError("Actor-only snapshot already exists: {}".format(path))
    payload = _evaluation_checkpoint_payload(
        calibration=calibration,
        learner=learner,
        condition=condition,
        proposal=proposal,
        calibration_checkpoint_sha256=calibration_checkpoint_sha256,
        fixed_state=fixed_state,
    )
    _save_torch_atomic(torch=torch, path=path, payload=payload)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "actor_state_sha256": str(payload["actor_state_sha256"]),
        "actor_update_count": int(payload["actor_update_count"]),
        "actor_awac_update_count": int(payload["actor_awac_update_count"]),
        "actor_recovery_update_count": int(
            payload["actor_recovery_update_count"]
        ),
        "actor_trust_region_rejection_count": int(
            payload["actor_trust_region_rejection_count"]
        ),
    }


def _trust_batch(batch: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        name: batch[name]
        for name in ("depth", "vector", "action_mask")
    }


def _fixed_validation_diagnostics(
    *,
    learner,
    replay: AWACReplayBuffer,
    validation_indices: np.ndarray,
    trust_indices: np.ndarray,
    weight_mode: str,
    torch,
    device,
    fixed_state: Mapping[str, Any],
    condition: str,
    proposal: int,
) -> Dict[str, Any]:
    """Measure one persisted validation batch without changing any state."""

    validation_batch = replay.sample_indices(
        validation_indices, torch=torch, device=device
    )
    trust_batch = _trust_batch(
        replay.sample_indices(trust_indices, torch=torch, device=device)
    )
    actor_before = state_sha256(learner.actor.state_dict())
    optimizer_before = state_sha256(learner.actor_optimizer.state_dict())
    metrics = learner.actor_only_update(
        validation_batch,
        actor_trust_batch=trust_batch,
        weight_mode=weight_mode,
        update_actor=False,
    )
    _assert_fixed_state_unchanged(
        before=fixed_state,
        learner=learner,
        condition=condition,
        proposal=proposal,
    )
    actor_after = state_sha256(learner.actor.state_dict())
    optimizer_after = state_sha256(learner.actor_optimizer.state_dict())
    if actor_before != actor_after or optimizer_before != optimizer_after:
        raise RuntimeError(
            "fixed validation diagnostics changed Actor state for {} proposal {}".format(
                condition, int(proposal)
            )
        )
    return {
        "proposal": int(proposal),
        "weight_mode": str(weight_mode),
        "sample_count": int(len(validation_indices)),
        "validation_indices_sha256": state_sha256(validation_indices),
        "trust_indices_sha256": state_sha256(trust_indices),
        "terminal_reason_provenance": TERMINAL_REASON_PROVENANCE,
        "actor_and_optimizer_unchanged": True,
        "metrics": _scalar_metrics(metrics),
    }


def _build_control_learner(
    *, torch, nn, device, bc_checkpoint: Mapping[str, Any], calibration: Mapping[str, Any]
):
    config = _optimization_config_from_calibration(calibration)
    learner = build_learner(
        torch=torch,
        nn=nn,
        device=device,
        bc_checkpoint=bc_checkpoint,
        config=config,
    )
    learner.load_state_dict(calibration)
    learner.enable_actor_after_calibration_pass(
        {"handoff_allowed": True, "actor_update_enabled": True}
    )
    return learner, config


def _run_condition(
    *,
    condition: str,
    weight_mode: str,
    replay: AWACReplayBuffer,
    sequences: Mapping[str, np.ndarray],
    torch,
    nn,
    device,
    bc_checkpoint: Mapping[str, Any],
    calibration: Mapping[str, Any],
    calibration_checkpoint_sha256: str,
    out_dir: Path,
) -> Dict[str, Any]:
    learner, config = _build_control_learner(
        torch=torch,
        nn=nn,
        device=device,
        bc_checkpoint=bc_checkpoint,
        calibration=calibration,
    )
    fixed_before = _learner_fixed_state(learner)
    branch_initial_state = _learner_branch_initial_state(learner)
    snapshots = {
        "0": _save_snapshot(
            out_dir=out_dir,
            condition=condition,
            proposal=0,
            calibration=calibration,
            calibration_checkpoint_sha256=calibration_checkpoint_sha256,
            learner=learner,
            fixed_state=fixed_before,
            torch=torch,
        )
    }
    validation_diagnostics = {
        "0": _fixed_validation_diagnostics(
            learner=learner,
            replay=replay,
            validation_indices=sequences["validation_indices"],
            trust_indices=sequences["trust_indices"][0],
            weight_mode=weight_mode,
            torch=torch,
            device=device,
            fixed_state=fixed_before,
            condition=condition,
            proposal=0,
        )
    }
    rows = []
    for proposal in range(1, 101):
        train_batch = replay.sample_indices(
            sequences["train_indices"][proposal - 1], torch=torch, device=device
        )
        trust_batch = _trust_batch(
            replay.sample_indices(
                sequences["trust_indices"][proposal - 1], torch=torch, device=device
            )
        )
        metrics = learner.actor_only_update(
            train_batch,
            actor_trust_batch=trust_batch,
            weight_mode=weight_mode,
        )
        _assert_fixed_state_unchanged(
            before=fixed_before,
            learner=learner,
            condition=condition,
            proposal=proposal,
        )
        row = {
            "proposal": int(proposal),
            "weight_mode": str(weight_mode),
            "terminal_reason_provenance": TERMINAL_REASON_PROVENANCE,
        }
        row.update(_scalar_metrics(metrics))
        row["argmax_flip_rate"] = float(
            metrics["bc_awac_top1_disagreement_rate"]
        )
        rows.append(row)
        if proposal in SNAPSHOT_STEPS:
            snapshots[str(proposal)] = _save_snapshot(
                out_dir=out_dir,
                condition=condition,
                proposal=proposal,
                calibration=calibration,
                calibration_checkpoint_sha256=calibration_checkpoint_sha256,
                learner=learner,
                fixed_state=fixed_before,
                torch=torch,
            )
            validation_diagnostics[str(proposal)] = _fixed_validation_diagnostics(
                learner=learner,
                replay=replay,
                validation_indices=sequences["validation_indices"],
                trust_indices=sequences["trust_indices"][0],
                weight_mode=weight_mode,
                torch=torch,
                device=device,
                fixed_state=fixed_before,
                condition=condition,
                proposal=proposal,
            )
    path = out_dir / "updates" / "{}.json".format(condition)
    fixed_after = _learner_fixed_state(learner)
    if fixed_before != fixed_after:
        raise RuntimeError(
            "Actor-only {} changed fixed-Critic state after all proposals".format(
                condition
            )
        )
    _write_json(
        path,
        {
            "condition": condition,
            "weight_mode": weight_mode,
            "proposal_count": 100,
            "optimization_config": vars(config),
            "fixed_state_before": fixed_before,
            "fixed_state_after": fixed_after,
            "branch_initial_state": branch_initial_state,
            "snapshots": snapshots,
            "validation_diagnostics": validation_diagnostics,
            "rows": rows,
        },
    )
    return {
        "condition": condition,
        "weight_mode": weight_mode,
        "updates_path": str(path.resolve()),
        "updates_sha256": file_sha256(path),
        "fixed_state": fixed_before,
        "fixed_state_after": fixed_after,
        "branch_initial_state": branch_initial_state,
        "snapshots": snapshots,
        "actor_optimizer_steps": int(
            sum(row["actor_optimizer_step"] for row in rows)
        ),
        "actor_awac_updates": int(
            sum(row["actor_awac_optimizer_step"] for row in rows)
        ),
        "actor_recovery_updates": int(
            sum(row["actor_recovery_optimizer_step"] for row in rows)
        ),
        "actor_rejections": int(
            sum(row["actor_trust_region_rejection"] for row in rows)
        ),
    }


def _evaluate_plan(
    *,
    root: Path,
    bc_checkpoint: Path,
    snapshots: Mapping[str, Any],
    dev100: Mapping[str, Any],
) -> Dict[str, Any]:
    evaluation_root = root / "dev100"
    values = {
        "A": {
            "checkpoint": str(bc_checkpoint.resolve()),
            "normalizer_checkpoint": "",
            "mission_index": str(dev100["path"]),
            "episodes": 100,
            "policy_action_selection": "deterministic_argmax_temperature_0",
            "out_dir": str((evaluation_root / "A_bc_frozen").resolve()),
        },
        "B50": {
            "checkpoint": snapshots["B"]["50"]["path"],
            "normalizer_checkpoint": str(bc_checkpoint.resolve()),
            "mission_index": str(dev100["path"]),
            "episodes": 100,
            "policy_action_selection": "deterministic_argmax_temperature_0",
            "out_dir": str((evaluation_root / "B_uniform_050").resolve()),
        },
        "B100": {
            "checkpoint": snapshots["B"]["100"]["path"],
            "normalizer_checkpoint": str(bc_checkpoint.resolve()),
            "mission_index": str(dev100["path"]),
            "episodes": 100,
            "policy_action_selection": "deterministic_argmax_temperature_0",
            "out_dir": str((evaluation_root / "B_uniform_100").resolve()),
        },
        "C50": {
            "checkpoint": snapshots["C"]["50"]["path"],
            "normalizer_checkpoint": str(bc_checkpoint.resolve()),
            "mission_index": str(dev100["path"]),
            "episodes": 100,
            "policy_action_selection": "deterministic_argmax_temperature_0",
            "out_dir": str((evaluation_root / "C_awac_050").resolve()),
        },
        "C100": {
            "checkpoint": snapshots["C"]["100"]["path"],
            "normalizer_checkpoint": str(bc_checkpoint.resolve()),
            "mission_index": str(dev100["path"]),
            "episodes": 100,
            "policy_action_selection": "deterministic_argmax_temperature_0",
            "out_dir": str((evaluation_root / "C_awac_100").resolve()),
        },
    }
    return values


def _optimizer_group_inventory(learner) -> Dict[str, list[Dict[str, Any]]]:
    """Record loaded optimizer param groups, not merely CLI defaults."""

    result = {}
    for name, optimizer in (
        ("actor", learner.actor_optimizer),
        ("critic", learner.critic_optimizer),
    ):
        groups = []
        for index, group in enumerate(optimizer.param_groups):
            groups.append(
                {
                    "index": int(index),
                    "name": str(group.get("name", "group_{}".format(index))),
                    "lr": float(group["lr"]),
                    "parameter_count": int(len(group.get("params", ()))),
                }
            )
        result[name] = groups
    return result


def _calibration_split_identity(calibration: Mapping[str, Any]) -> Dict[str, Any]:
    split = calibration.get("calibration_split")
    if not isinstance(split, Mapping):
        raise ValueError("calibration checkpoint split is missing")
    train_ids = [str(value) for value in split.get("train_episode_ids", ())]
    holdout_ids = [str(value) for value in split.get("holdout_episode_ids", ())]
    if not train_ids or not holdout_ids:
        raise ValueError("calibration split must contain independent train and holdout episodes")
    if len(set(train_ids)) != len(train_ids) or len(set(holdout_ids)) != len(holdout_ids):
        raise ValueError("calibration split contains duplicate episode IDs")
    overlap = sorted(set(train_ids).intersection(holdout_ids))
    if overlap:
        raise ValueError("calibration train/holdout episode IDs overlap")
    source = calibration.get("mission_source_identity")
    if not isinstance(source, Mapping):
        raise ValueError("calibration mission source identity is missing")
    return {
        "contract_id": str(split.get("contract_id", "")),
        "selection_method": str(split.get("selection_method", "")),
        "seed": split.get("seed"),
        "train_episode_count": int(len(train_ids)),
        "holdout_episode_count": int(len(holdout_ids)),
        "train_episode_ids_sha256": _ordered_values_sha256(train_ids),
        "holdout_episode_ids_sha256": _ordered_values_sha256(holdout_ids),
        "overlap_count": int(len(overlap)),
        "mission_source_identity": dict(source),
        "split_checkpoint_sha256": state_sha256(dict(split)),
    }


def _dev100_identity(index_path: Path) -> Dict[str, Any]:
    index = Path(index_path).expanduser().resolve()
    if not index.is_file():
        raise FileNotFoundError("Dev100 mission index is missing: {}".format(index))
    manifest_path = index.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Dev100 manifest is missing: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("role", "")).upper() != "DEV":
        raise ValueError("Dev100 manifest role must be DEV")
    dev = manifest.get("dev")
    if not isinstance(dev, Mapping) or int(dev.get("mission_count", -1)) != 100:
        raise ValueError("Dev100 manifest does not describe exactly 100 missions")
    actual_sha256 = file_sha256(index)
    if str(dev.get("file_sha256", "")) != actual_sha256:
        raise ValueError("Dev100 mission index SHA256 does not match its manifest")
    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100:
        raise ValueError("Dev100 mission index must contain exactly 100 rows")
    mission_ids = [str(row.get("mission_id", "")) for row in rows]
    episode_ids = [str(row.get("episode_id", "")) for row in rows]
    if not all(mission_ids) or not all(episode_ids):
        raise ValueError("Dev100 mission identity is incomplete")
    if len(set(mission_ids)) != len(mission_ids) or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Dev100 mission identity is not unique")
    return {
        "path": str(index),
        "sha256": actual_sha256,
        "row_count": int(len(rows)),
        "seed": manifest.get("seed"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "ordered_mission_ids_sha256": _ordered_values_sha256(mission_ids),
        "ordered_episode_ids_sha256": _ordered_values_sha256(episode_ids),
        "task_contract_id": str(rows[0].get("task_contract_id", "")),
        "task_contract_sha256": str(rows[0].get("task_contract_sha256", "")),
        "max_primitive_steps": int(rows[0].get("max_primitive_steps", "-1")),
    }


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError("controlled experiment output already exists: {}".format(out_dir))
    bc_path = Path(args.bc_checkpoint).expanduser().resolve()
    calibration_path = Path(args.calibration_checkpoint).expanduser().resolve()
    replay_path = Path(args.replay_dir).expanduser().resolve()
    if not bc_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError("BC or calibration checkpoint is missing")
    if not replay_path.is_dir():
        raise FileNotFoundError("calibration replay directory is missing: {}".format(replay_path))
    dev100 = _dev100_identity(Path(args.dev100_index))

    torch, nn, _, _, _ = require_torch()
    torch.set_num_threads(int(args.cpu_threads))
    device = resolve_torch_device(
        torch,
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        ),
    )
    bc_checkpoint = load_torch(bc_path, torch=torch, map_location="cpu")
    if not isinstance(bc_checkpoint, Mapping):
        raise ValueError("BC checkpoint must be a mapping")
    validate_bc_checkpoint_for_awac(
        bc_checkpoint,
        mpl_contract_sha256=str(bc_checkpoint.get("mpl_contract_sha256", "")),
    )
    calibration = load_calibration_pass_checkpoint(
        calibration_path, torch=torch, map_location="cpu"
    )
    bc_sha256 = file_sha256(bc_path)
    if str(calibration.get("source_bc_checkpoint_sha256", "")) != bc_sha256:
        raise ValueError("calibration checkpoint BC identity does not match --bc-checkpoint")
    behavior_policy = _validate_calibration_behavior_policy(
        calibration=calibration, bc_checkpoint_sha256=bc_sha256
    )
    calibration_sha256 = file_sha256(calibration_path)
    split_identity = _calibration_split_identity(calibration)
    for field in (
        "task_contract_id",
        "task_contract_sha256",
        "max_primitive_steps",
        "mpl_contract_sha256",
        "observation_contract",
        "num_actions",
    ):
        bc_value = bc_checkpoint.get(field)
        calibration_value = calibration.get(field)
        if bc_value != calibration_value:
            raise ValueError(
                "BC/calibration {} identity mismatch: {} != {}".format(
                    field, bc_value, calibration_value
                )
            )
    if str(dev100["task_contract_id"]) != str(bc_checkpoint.get("task_contract_id", "")):
        raise ValueError("Dev100 and BC task contract ID mismatch")
    if str(dev100["task_contract_sha256"]) != str(bc_checkpoint.get("task_contract_sha256", "")):
        raise ValueError("Dev100 and BC task contract SHA256 mismatch")
    if int(dev100["max_primitive_steps"]) != int(
        bc_checkpoint.get("max_primitive_steps", -1)
    ):
        raise ValueError("Dev100 and BC max primitive steps mismatch")
    if int(dev100["seed"]) != int(args.dev100_seed):
        raise ValueError("Dev100 manifest seed does not match --dev100-seed")

    replay = AWACReplayBuffer.open(replay_path, read_only=True)
    try:
        actual_replay_identity = calibration_replay_identity(replay)
        expected_replay_identity = calibration.get("replay_identity", {})
        if actual_replay_identity != expected_replay_identity:
            raise ValueError("calibration replay identity does not match calibration PASS")
        if str(replay.metadata.get("behavior_source_phase", "")) != "critic_calibration":
            raise ValueError("Actor-only control requires calibration train replay")
        sequences = build_fixed_batch_sequences(
            replay_size=int(replay.size),
            proposal_count=int(args.proposals),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        )
        replay_before = replay_content_sha256(replay)
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        initial_rng = _capture_rng_state(torch)

        out_dir.mkdir(parents=True, exist_ok=False)
        batches_path = out_dir / "fixed_batch_indices.npz"
        np.savez_compressed(
            batches_path,
            train_indices=sequences["train_indices"],
            trust_indices=sequences["trust_indices"],
            validation_indices=sequences["validation_indices"],
        )
        batch_manifest = {
            "schema_id": "awac_actor_only_fixed_batch_indices_v1",
            "seed": int(args.seed),
            "proposal_count": int(args.proposals),
            "batch_size": int(args.batch_size),
            "replay_size": int(replay.size),
            "path": str(batches_path.resolve()),
            "sha256": file_sha256(batches_path),
            "train_indices_sha256": state_sha256(sequences["train_indices"]),
            "trust_indices_sha256": state_sha256(sequences["trust_indices"]),
            "validation_indices_sha256": state_sha256(
                sequences["validation_indices"]
            ),
        }
        _write_json(out_dir / "fixed_batch_manifest.json", batch_manifest)

        _restore_rng_state(torch, initial_rng)
        branch_b = _run_condition(
            condition="B",
            weight_mode="uniform",
            replay=replay,
            sequences=sequences,
            torch=torch,
            nn=nn,
            device=device,
            bc_checkpoint=bc_checkpoint,
            calibration=calibration,
            calibration_checkpoint_sha256=calibration_sha256,
            out_dir=out_dir,
        )
        _restore_rng_state(torch, initial_rng)
        branch_c = _run_condition(
            condition="C",
            weight_mode="awac",
            replay=replay,
            sequences=sequences,
            torch=torch,
            nn=nn,
            device=device,
            bc_checkpoint=bc_checkpoint,
            calibration=calibration,
            calibration_checkpoint_sha256=calibration_sha256,
            out_dir=out_dir,
        )
        if branch_b["branch_initial_state"] != branch_c["branch_initial_state"]:
            raise RuntimeError(
                "B/C did not start from identical Actor, optimizer, and fixed-Critic state"
            )
        control_probe, _ = _build_control_learner(
            torch=torch,
            nn=nn,
            device=device,
            bc_checkpoint=bc_checkpoint,
            calibration=calibration,
        )
        optimizer_groups = _optimizer_group_inventory(control_probe)
        replay_after = replay_content_sha256(replay)
        if replay_before != replay_after:
            raise RuntimeError("Actor-only control changed the read-only calibration replay")
    finally:
        replay.close()

    baseline_actor_sha256 = actor_state_sha256(bc_checkpoint)
    if actor_state_sha256(calibration) != baseline_actor_sha256:
        raise RuntimeError("calibration Actor is not the frozen BC initialization")
    snapshots = {"B": branch_b["snapshots"], "C": branch_c["snapshots"]}
    for condition in ("B", "C"):
        if snapshots[condition]["0"]["actor_state_sha256"] != baseline_actor_sha256:
            raise RuntimeError(
                "{} proposal-0 snapshot is not the frozen BC baseline".format(
                    condition
                )
            )
    evaluation_plan = _evaluate_plan(
        root=out_dir,
        bc_checkpoint=bc_path,
        snapshots=snapshots,
        dev100=dev100,
    )
    calibration_metrics = calibration.get("calibration_metrics", {})
    mission_progress = calibration.get("mission_progress", {})
    gate_history = []
    if isinstance(mission_progress, Mapping):
        gate_history = list(mission_progress.get("gate_history", ()))
    if not gate_history and isinstance(calibration_metrics, Mapping):
        gate_history = list(calibration_metrics.get("gate_history", ()))
    if not gate_history:
        raise ValueError("calibration PASS checkpoint has no gate-history evidence")
    latest_gate = gate_history[-1]
    alignment_status = str(
        latest_gate.get(
            "value_policy_alignment",
            calibration_metrics.get("value_policy_alignment", "NOT_PERSISTED")
            if isinstance(calibration_metrics, Mapping)
            else "NOT_PERSISTED",
        )
    )
    if alignment_status != "MATCH":
        raise ValueError(
            "calibration PASS lacks matched behavior/target/return policy evidence: {}".format(
                alignment_status
            )
        )
    policy_alignment = {
        "status": "MATCH",
        "calibration_checkpoint": str(calibration_path),
        "calibration_checkpoint_sha256": calibration_sha256,
        "frozen_bc_checkpoint": str(bc_path),
        "frozen_bc_checkpoint_sha256": bc_sha256,
        "policy_identity": behavior_policy,
        "trajectory_action_selection_mode": "masked_categorical",
        "trajectory_temperature": 1.0,
        "holdout_action_selection_mode": "masked_categorical",
        "holdout_temperature": 1.0,
        "bellman_value_evaluation_mode": "masked_softmax_expectation",
        "bellman_temperature": 1.0,
        "formal_dev100_action_selection_mode": "deterministic_argmax",
        "formal_dev100_temperature": 0.0,
        "gate_value_policy_alignment": alignment_status,
    }
    invariance = {
        "status": "PASS",
        "branch_initial_state_identical": True,
        "baseline_actor_state_sha256": baseline_actor_sha256,
        "branches": {
            name: {
                "fixed_state_before": branch["fixed_state"],
                "fixed_state_after": branch["fixed_state_after"],
                "fixed_state_unchanged": branch["fixed_state"]
                == branch["fixed_state_after"],
                "actor_optimizer_steps": branch["actor_optimizer_steps"],
                "actor_awac_updates": branch["actor_awac_updates"],
                "actor_recovery_updates": branch["actor_recovery_updates"],
                "actor_rejections": branch["actor_rejections"],
                "updates_path": branch["updates_path"],
                "updates_sha256": branch["updates_sha256"],
            }
            for name, branch in (("B", branch_b), ("C", branch_c))
        },
        "replay_content_sha256_before": replay_before,
        "replay_content_sha256_after": replay_after,
        "replay_unchanged": replay_before == replay_after,
        "critic_update_count_delta": {"B": 0, "C": 0},
        "target_soft_update": False,
        "critic_optimizer_step": False,
        "bc_reference_updated": False,
    }
    _write_json(out_dir / "calibration_policy_alignment.json", policy_alignment)
    _write_json(
        out_dir / "calibration_gate_history.json",
        {
            "calibration_checkpoint": str(calibration_path),
            "calibration_checkpoint_sha256": calibration_sha256,
            "calibration_gate_state": str(
                calibration.get("calibration_gate_state", "")
            ),
            "gate_history": gate_history,
            "split": split_identity,
            "replay_identity": actual_replay_identity,
        },
    )
    _write_json(out_dir / "actor_only_invariance.json", invariance)
    result = {
        "schema_id": EXPERIMENT_SCHEMA_ID,
        "status": "PASS",
        "runtime_mode": "offline_fixed_read_only_calibration_replay",
        "command": {
            "python_executable": sys.executable,
            "argv": list(sys.argv),
            "working_tree": _working_tree_identity(),
        },
        "runtime": {
            "python_version": sys.version,
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": str(torch.version.cuda),
            "device": str(device),
        },
        "calibration_checkpoint": str(calibration_path),
        "calibration_checkpoint_sha256": calibration_sha256,
        "bc_checkpoint": str(bc_path),
        "bc_checkpoint_sha256": bc_sha256,
        "bc_actor_state_sha256": baseline_actor_sha256,
        "bc_contract": {
            key: bc_checkpoint.get(key)
            for key in (
                "feature_contract_id",
                "policy_input_contract_sha256",
                "task_contract_id",
                "task_contract_sha256",
                "max_primitive_steps",
                "mpl_contract_sha256",
                "observation_contract",
                "observation_source",
                "depth_history_frames",
                "num_actions",
                "vec_dim",
            )
        },
        "calibration_behavior_policy": behavior_policy,
        "calibration_split": split_identity,
        "calibration_gate": {
            "state": str(calibration.get("calibration_gate_state", "")),
            "latest": latest_gate,
            "history_count": int(len(gate_history)),
        },
        "replay": {
            "path": str(replay_path),
            "read_only": True,
            "identity": actual_replay_identity,
            "content_sha256_before": replay_before,
            "content_sha256_after": replay_after,
            "unchanged": replay_before == replay_after,
        },
        "fixed_batches": batch_manifest,
        "rng_initial_identity": _rng_identity(initial_rng),
        "proposals": int(args.proposals),
        "batch_size": int(args.batch_size),
        "loaded_optimizer_param_groups": optimizer_groups,
        "dev100": dev100,
        "innovations": {
            "twin_q_confidence": False,
            "adaptive_bc_kl": False,
            "primitive_neighbor_exploration": False,
        },
        "conditions": {"A": {"weight_mode": "frozen_bc_no_update"}, "B": branch_b, "C": branch_c},
        "evaluation_plan": evaluation_plan,
        "comparison_plan": [
            {"left": left, "right": right, "out_dir": str((out_dir / "paired" / (left + "_vs_" + right)).resolve())}
            for left, right in COMPARISONS
        ],
        "artifacts": {
            "manifest": str((out_dir / MANIFEST_NAME).resolve()),
            "calibration_policy_alignment": str(
                (out_dir / "calibration_policy_alignment.json").resolve()
            ),
            "calibration_gate_history": str(
                (out_dir / "calibration_gate_history.json").resolve()
            ),
            "actor_only_invariance": str(
                (out_dir / "actor_only_invariance.json").resolve()
            ),
        },
    }
    _write_json(out_dir / MANIFEST_NAME, result)
    return result


def _read_rows(path: Path) -> Tuple[Dict[str, Any], list[Dict[str, str]]]:
    summary_path = path / "summary.json"
    index_path = path / "rollout_index.csv"
    if not summary_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("completed Dev100 result is missing: {}".format(path))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100 or int(summary.get("episodes", -1)) != 100:
        raise ValueError("controlled Dev100 must contain exactly 100 episodes: {}".format(path))
    return summary, rows


def _row_outcome(row: Mapping[str, Any]) -> str:
    active = [
        outcome
        for outcome in OUTCOMES
        if str(row.get(outcome, "")).strip().lower() in {"1", "true", "yes"}
    ]
    if len(active) != 1:
        raise ValueError("evaluation row does not have one terminal outcome")
    return active[0]


def terminal_transition_table(
    left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, int]]:
    if len(left_rows) != len(right_rows):
        raise ValueError("paired rows have different lengths")
    result = {left: {right: 0 for right in OUTCOMES} for left in OUTCOMES}
    for left, right in zip(left_rows, right_rows):
        identity_left = (str(left.get("episode_id")), str(left.get("mission_id")))
        identity_right = (str(right.get("episode_id")), str(right.get("mission_id")))
        if identity_left != identity_right:
            raise ValueError("paired evaluation episode/mission order mismatch")
        result[_row_outcome(left)][_row_outcome(right)] += 1
    return result


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.out_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        # Read-only compatibility for an early diagnostic run created before
        # the task's required canonical manifest name was introduced.
        manifest_path = root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("actor-only experiment manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = manifest.get("evaluation_plan", {})
    if set(plan) != {"A", "B50", "B100", "C50", "C100"}:
        raise ValueError("actor-only evaluation plan is invalid")
    dev100 = manifest.get("dev100", {})
    expected_mission_sha256 = str(dev100.get("sha256", ""))
    if not expected_mission_sha256:
        raise ValueError("actor-only manifest lacks Dev100 mission identity")
    # Reuse the repository's formal paired statistical owner.  It validates
    # task/observation/MPL identity before computing exact McNemar evidence.
    from scripts.compare_paired_policy_evaluations import build_comparison

    records = {}
    table_rows = []
    for left, right in COMPARISONS:
        left_dir = Path(plan[left]["out_dir"])
        right_dir = Path(plan[right]["out_dir"])
        result, paired_rows = build_comparison(left_dir, right_dir)
        left_summary, left_rows = _read_rows(left_dir)
        right_summary, right_rows = _read_rows(right_dir)
        if (
            str(left_summary.get("mission_index_sha256", ""))
            != expected_mission_sha256
            or str(right_summary.get("mission_index_sha256", ""))
            != expected_mission_sha256
        ):
            raise ValueError("controlled Dev100 result does not use the manifest mission index")
        transition = terminal_transition_table(left_rows, right_rows)
        pair_name = "{}_vs_{}".format(left, right)
        record = {
            "left": left,
            "right": right,
            "comparison": result,
            "terminal_transition_matrix": transition,
            "per_mission_rows": paired_rows,
        }
        pair_dir = root / "paired" / pair_name
        _write_json(pair_dir / "comparison.json", record)
        with (pair_dir / "per_mission_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0].keys()))
            writer.writeheader()
            writer.writerows(paired_rows)
        paired = result["paired"]
        table_rows.append({
            "comparison": pair_name,
            "left_success": int(result["aggregate"]["bc"]["success_count"]),
            "right_success": int(result["aggregate"]["awac"]["success_count"]),
            "gain": int(paired["recovered"]),
            "loss": int(paired["regressed"]),
            "net_gain": int(paired["net_success_change"]),
            "mcnemar_p_value": float(result["mcnemar"]["p_value"]),
        })
        records[pair_name] = record
    result = {
        "schema_id": "awac_fixed_critic_actor_only_dev100_paired_v1",
        "status": "PASS",
        "evaluation_plan": plan,
        "comparisons": records,
        "comparison_table": table_rows,
    }
    _write_json(root / "paired_dev100_summary.json", result)
    _write_json(root / "comparison_summary.json", result)
    with (root / "paired_dev100_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    with (root / "paired_dev100.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    _write_chinese_conclusion(root / "中文结论.md", result)
    return result


def _write_chinese_conclusion(path: Path, result: Mapping[str, Any]) -> None:
    """Write a factual close-out, never a convergence claim from one seed."""

    table = {str(row["comparison"]): row for row in result["comparison_table"]}
    lines = [
        "# 固定 Critic Actor 权重对照结论",
        "",
        "本报告只描述本次新校准、固定数据、固定 Critic、单一训练 seed 的 Dev100 结果；它不构成在线 AWAC 收敛或泛化结论。",
        "",
        "## 配对结果",
        "",
        "| 比较 | 左策略成功 | 右策略成功 | gain | loss | net gain | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["comparison_table"]:
        lines.append(
            "| {comparison} | {left_success} | {right_success} | {gain} | {loss} | {net_gain} | {mcnemar_p_value:.8f} |".format(
                **row
            )
        )
    primary = table.get("A_vs_C100")
    uniform = table.get("A_vs_B100")
    weighted_vs_uniform = table.get("B100_vs_C100")
    lines.extend(
        [
            "",
            "## 中文结论",
            "",
            "- 已执行的唯一干预是 B 的等权样本与 C 的生产 AWAC 优势权重；两支共享初始 Actor、Actor optimizer、Critic、Replay、RNG 初态和固定 batch 索引。",
            "- 0/50/100 快照、每次提案的接受/恢复/拒绝、权重分位数、ESS、KL 与参数组变化见 `updates/B.json`、`updates/C.json`；固定 Critic 和 Replay 指纹见 `actor_only_invariance.json`。",
            "- 训练 Replay 不存储逐 transition 的真实终止原因，因此该字段没有被伪造；真实 Dev100 终止类型及逐任务记录见各 `dev100/*/rollout_index.csv` 和 `paired/*/per_mission_results.csv`。",
        ]
    )
    if primary is not None:
        lines.append(
            "- 预先指定主终点 C100 相对 A：gain={gain}，loss={loss}，net gain={net_gain}。".format(
                **primary
            )
        )
    if uniform is not None:
        lines.append(
            "- B100 相对 A：gain={gain}，loss={loss}，net gain={net_gain}。".format(
                **uniform
            )
        )
    if weighted_vs_uniform is not None:
        lines.append(
            "- C100 相对 B100：gain={gain}，loss={loss}，net gain={net_gain}。".format(
                **weighted_vs_uniform
            )
        )
    lines.extend(
        [
            "- 尚未证实的事项：跨 seed 的稳定性、在线训练收益以及 Final300 泛化表现。本轮没有运行在线 AWAC、没有使用 Final300，也没有自动扩大 Actor 提案预算。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="run offline fixed-Critic A/B/C")
    prepare_parser.add_argument("--bc-checkpoint", required=True, type=Path)
    prepare_parser.add_argument("--calibration-checkpoint", required=True, type=Path)
    prepare_parser.add_argument("--replay-dir", required=True, type=Path)
    prepare_parser.add_argument("--dev100-index", required=True, type=Path)
    prepare_parser.add_argument("--dev100-seed", type=int, default=4026)
    prepare_parser.add_argument("--out-dir", required=True, type=Path)
    prepare_parser.add_argument("--proposals", type=int, default=100)
    prepare_parser.add_argument("--batch-size", type=int, default=128)
    prepare_parser.add_argument("--seed", type=int, default=20260906)
    prepare_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    prepare_parser.add_argument("--cpu-threads", type=int, default=1)
    summary_parser = subparsers.add_parser("summarize", help="write paired Dev100 tables after all five evaluations")
    summary_parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        output = prepare(args)
        print("AWAC_ACTOR_ONLY_AB_CONTROL=PASS")
        print("EXPERIMENT_MANIFEST={}".format(Path(args.out_dir).resolve() / MANIFEST_NAME))
        print("REPLAY_UNCHANGED={}".format(output["replay"]["unchanged"]))
    else:
        output = summarize(args)
        print("AWAC_ACTOR_ONLY_DEV100_PAIRED=PASS")
        print("PAIRED_SUMMARY={}".format(Path(args.out_dir).resolve() / "paired_dev100_summary.json"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
