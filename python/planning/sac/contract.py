"""Stable identities and preregistered defaults for the SAC mainline."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SAC_ALGORITHM_ID = "bc_initialized_online_discrete_sac_v1"
SAC_CHECKPOINT_CONTRACT_ID = "bc_initialized_discrete_sac_checkpoint_v1"
SAC_REPLAY_CONTRACT_ID = "bc_initialized_discrete_sac_online_replay_v1"
SAC_TRAINING_CONFIG_CONTRACT_ID = "bc_initialized_discrete_sac_training_config_v1"
SAC_MODEL_TYPE = "sac_discrete"
SAC_POLICY_ID = "sac_masked_categorical_v1"
SAC_RESIDUAL_POLICY_ID = "sac_masked_categorical_residual_v5"
SAC_BC_REFERENCE_ID = "frozen_bc_reference_v1"
SAC_KL_BACKTRACKING_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
SAC_RESIDUAL_LOGIT_CAP = 0.49

SAC_DEFAULTS = {
    "batch_size": 128,
    "learning_starts": 5000,
    "gamma": 0.99,
    "tau": 0.005,
    "reward_scale": 0.10,
    "alpha": 0.20,
    "alpha_mode": "fixed_preregistered",
    "beta_bc": 0.05,
    "critic_updates_per_transition": 1.0,
    "actor_updates_per_transition": 1.0,
    "max_env_steps": 25000,
    "max_episodes": 1000,
    "max_workers": 16,
    "gradient_clip_norm": 5.0,
    "bc_kl_hard_stop": 1.0,
    "entropy_floor": 0.02,
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def training_config_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(config))


__all__ = [
    "SAC_ALGORITHM_ID",
    "SAC_CHECKPOINT_CONTRACT_ID",
    "SAC_REPLAY_CONTRACT_ID",
    "SAC_TRAINING_CONFIG_CONTRACT_ID",
    "SAC_MODEL_TYPE",
    "SAC_POLICY_ID",
    "SAC_RESIDUAL_POLICY_ID",
    "SAC_BC_REFERENCE_ID",
    "SAC_KL_BACKTRACKING_FACTORS",
    "SAC_RESIDUAL_LOGIT_CAP",
    "SAC_DEFAULTS",
    "canonical_sha256",
    "training_config_sha256",
]
