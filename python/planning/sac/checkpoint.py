"""Transactional SAC checkpoint persistence and identity validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from planning.common.atomic import write_json_atomic
from planning.common.hashing import file_sha256
from planning.contracts.reward import REWARD_CONTRACT_ID, reward_contract_sha256
from planning.version import SOFTWARE_VERSION
from planning.sac.contract import (
    SAC_ALGORITHM_ID,
    SAC_CHECKPOINT_CONTRACT_ID,
    SAC_MODEL_TYPE,
    SAC_RESIDUAL_LOGIT_CAP,
    SAC_TRAINING_CONFIG_CONTRACT_ID,
    canonical_sha256,
    training_config_sha256,
)
from planning.sac.network import state_dict_sha256


def _torch_save_atomic(torch, payload: Mapping[str, Any], path: Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(dict(payload), str(temporary))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(destination))


def build_checkpoint_payload(
    *,
    agent,
    torch,
    bc_checkpoint: Mapping[str, Any],
    bc_checkpoint_sha256: str,
    resolved_training_config: Mapping[str, Any],
    replay,
    environment_steps: int,
    completed_episodes: int,
    first_actor_update_at_transition: Optional[int],
    training_metrics: Mapping[str, Any],
    mission_source_sha256: str,
    source_code_sha256: str,
    rng_state: Mapping[str, Any],
    runtime_state: Optional[Mapping[str, Any]] = None,
    journal_state: Optional[Mapping[str, Any]] = None,
    transition_order: Optional[Mapping[str, Any]] = None,
    cuda_determinism: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    config = dict(resolved_training_config)
    payload = {
        "algorithm_id": SAC_ALGORITHM_ID,
        "sac_checkpoint_contract_id": SAC_CHECKPOINT_CONTRACT_ID,
        "training_config_contract_id": SAC_TRAINING_CONFIG_CONTRACT_ID,
        "model_type": SAC_MODEL_TYPE,
        # The BC artifact predates the off-policy checkpoint contract and may
        # not carry these fields.  SAC checkpoints must bind to the current
        # canonical runtime contracts, never to an empty inherited value.
        "software_version": SOFTWARE_VERSION,
        "feature_contract_id": str(bc_checkpoint["feature_contract_id"]),
        "policy_input_contract": dict(bc_checkpoint.get("policy_input_contract", {})),
        "policy_input_contract_sha256": str(bc_checkpoint["policy_input_contract_sha256"]),
        "policy_runtime_contract_id": str(bc_checkpoint["policy_runtime_contract_id"]),
        "observation_contract": str(bc_checkpoint["observation_contract"]),
        "observation_source": str(bc_checkpoint["observation_source"]),
        "task_contract_id": str(bc_checkpoint["task_contract_id"]),
        "task_contract_schema_version": int(bc_checkpoint["task_contract_schema_version"]),
        "task_contract_sha256": str(bc_checkpoint["task_contract_sha256"]),
        "max_primitive_steps": int(bc_checkpoint["max_primitive_steps"]),
        "max_steps": int(bc_checkpoint.get("max_steps", bc_checkpoint["max_primitive_steps"])),
        "mpl_contract_sha256": str(bc_checkpoint["mpl_contract_sha256"]),
        "reward_contract_id": REWARD_CONTRACT_ID,
        "reward_contract_sha256": reward_contract_sha256(),
        "safety_mask": str(bc_checkpoint.get("safety_mask", "depth")),
        "execution_mode": str(bc_checkpoint.get("execution_mode", "continuous")),
        "vec_dim": int(bc_checkpoint["vec_dim"]),
        "num_actions": int(bc_checkpoint["num_actions"]),
        "depth_history_frames": int(bc_checkpoint.get("depth_history_frames", 1)),
        "initial_prev_action": int(bc_checkpoint["initial_prev_action"]),
        "feature_mean": bc_checkpoint["feature_mean"],
        "feature_std": bc_checkpoint["feature_std"],
        "normalizer_sha256": str(bc_checkpoint.get("normalizer_sha256", "")),
        "source_bc_checkpoint_sha256": str(bc_checkpoint_sha256),
        "bc_checkpoint_sha256": str(bc_checkpoint_sha256),
        "bc_reference_id": agent.bc_reference_id,
        "bc_reference_fingerprint": agent.bc_reference_fingerprint(),
        "actor_fingerprint": agent.actor_fingerprint(),
        "privileged_runtime_inputs": [],
        "resolved_training_config": config,
        "resolved_training_config_sha256": training_config_sha256(config),
        "source_code_sha256": str(source_code_sha256),
        "split_manifest_sha256": str(mission_source_sha256),
        "mission_source_sha256": str(mission_source_sha256),
        "replay_identity": {
            "contract_id": str(replay.metadata["contract_id"]),
            "metadata_sha256": replay.metadata_sha256(),
            "rows": int(replay.size),
        },
        "environment_steps": int(environment_steps),
        "completed_episodes": int(completed_episodes),
        "first_actor_update_at_transition": first_actor_update_at_transition,
        "training_metrics": dict(training_metrics),
        "rng_state": dict(rng_state),
        "checkpoint_state_contract_id": "bc_initialized_discrete_sac_checkpoint_state_v2",
        "runtime_state": dict(runtime_state or {}),
        "journal_state": dict(journal_state or {}),
        "transition_order": dict(transition_order or {}),
        "cuda_determinism": dict(cuda_determinism or {}),
        **dict(agent.state_payload()),
    }
    return payload


def save_checkpoint(torch, payload: Mapping[str, Any], path: Path) -> str:
    _torch_save_atomic(torch, payload, Path(path))
    return file_sha256(Path(path))


def load_checkpoint(torch, path: Path, *, map_location="cpu") -> Mapping[str, Any]:
    value = torch.load(str(Path(path).expanduser().resolve()), map_location=map_location, weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("SAC checkpoint must be a mapping")
    validate_checkpoint_payload(value)
    return value


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    required = (
        "algorithm_id",
        "sac_checkpoint_contract_id",
        "training_config_contract_id",
        "actor_state_dict",
        "bc_reference_state_dict",
        "critic1_state_dict",
        "critic2_state_dict",
        "target_critic1_state_dict",
        "target_critic2_state_dict",
        "actor_optimizer_state_dict",
        "critic_optimizer_state_dict",
        "resolved_training_config",
        "resolved_training_config_sha256",
        "source_bc_checkpoint_sha256",
        "bc_reference_fingerprint",
        "actor_fingerprint",
        "replay_identity",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError("SAC checkpoint missing fields: {}".format(", ".join(missing)))
    if payload.get("algorithm_id") != SAC_ALGORITHM_ID:
        raise ValueError("SAC algorithm identity mismatch")
    if payload.get("sac_checkpoint_contract_id") != SAC_CHECKPOINT_CONTRACT_ID:
        raise ValueError("SAC checkpoint contract mismatch")
    if payload.get("training_config_contract_id") != SAC_TRAINING_CONFIG_CONTRACT_ID:
        raise ValueError("SAC training config contract mismatch")
    config = payload["resolved_training_config"]
    if not isinstance(config, Mapping) or training_config_sha256(config) != payload["resolved_training_config_sha256"]:
        raise ValueError("SAC resolved training config hash mismatch")
    if payload.get("bc_reference_fingerprint") != state_dict_sha256(payload["bc_reference_state_dict"]):
        raise ValueError("SAC BC reference fingerprint mismatch")
    if payload.get("actor_fingerprint") != state_dict_sha256(payload["actor_state_dict"]):
        raise ValueError("SAC Actor fingerprint mismatch")
    architecture = str(payload.get("actor_architecture", "direct"))
    if architecture not in {"direct", "residual"}:
        raise ValueError("SAC Actor architecture is invalid")
    if architecture == "residual":
        cap = float(payload.get("residual_logit_cap", -1.0))
        if cap != float(SAC_RESIDUAL_LOGIT_CAP):
            raise ValueError("SAC residual logit cap mismatch")
        if float(payload.get("theoretical_kl_bound", -1.0)) != float(2.0 * cap):
            raise ValueError("SAC residual theoretical KL bound mismatch")
        if str(payload.get("actor_policy_id", "")) != "sac_masked_categorical_residual_v5":
            raise ValueError("SAC residual Actor policy identity mismatch")
    if not isinstance(payload["replay_identity"], Mapping):
        raise ValueError("SAC replay identity is invalid")
    if "checkpoint_state_contract_id" in payload and payload["checkpoint_state_contract_id"] != "bc_initialized_discrete_sac_checkpoint_state_v2":
        raise ValueError("SAC checkpoint state contract mismatch")
    for field in ("runtime_state", "journal_state", "transition_order", "cuda_determinism"):
        if field in payload and not isinstance(payload[field], Mapping):
            raise ValueError("SAC checkpoint {} is invalid".format(field))


def write_checkpoint_manifest(output_dir: Path, records) -> None:
    write_json_atomic(
        Path(output_dir) / "checkpoint_manifest.json",
        {
            "schema_id": "bc_initialized_discrete_sac_checkpoint_manifest_v1",
            "checkpoints": [dict(record) for record in records],
        },
        trailing_newline=True,
    )


__all__ = [
    "build_checkpoint_payload",
    "load_checkpoint",
    "save_checkpoint",
    "validate_checkpoint_payload",
    "write_checkpoint_manifest",
]
