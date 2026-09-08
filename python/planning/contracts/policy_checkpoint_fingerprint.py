"""Canonical fingerprints for the deployable policy in BC/AWAC checkpoints.

Whole-checkpoint hashes intentionally differ after Critic or optimizer updates.
The guarded policy runner needs a narrower identity: whether the Actor that will
actually control Unity changed.  This module hashes only the ordered Actor
state tensors and deliberately treats ``model_state_dict`` (BC) and
``actor_state_dict`` (AWAC) as the same policy representation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Mapping

import numpy as np

from planning.contracts.feature import INITIAL_PREV_ACTION
from planning.contracts.policy_runtime import (
    checkpoint_depth_mask_numeric_config,
    checkpoint_runtime_contract,
)


POLICY_ACTOR_FINGERPRINT_CONTRACT_ID = "policy_actor_state_sha256"
DEPLOYABLE_POLICY_FINGERPRINT_CONTRACT_ID = "deployable_policy_sha256"


def _update_bytes(digest, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _actor_state_dict(checkpoint: Mapping) -> Mapping:
    if "actor_state_dict" in checkpoint:
        state = checkpoint["actor_state_dict"]
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint actor_state_dict must be a mapping")
        return state
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint model_state_dict must be a mapping")
        return state
    raise ValueError(
        "checkpoint contains neither actor_state_dict nor model_state_dict"
    )


def actor_state_sha256(checkpoint: Mapping) -> str:
    """Return a deterministic SHA256 of the checkpoint's deployable Actor."""

    state_dict = _actor_state_dict(checkpoint)
    if not state_dict:
        raise ValueError("actor state dictionary is empty")
    digest = hashlib.sha256()
    _update_bytes(
        digest, POLICY_ACTOR_FINGERPRINT_CONTRACT_ID.encode("utf-8")
    )
    if not all(isinstance(name, str) for name in state_dict):
        raise TypeError("actor state dictionary keys must be strings")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not hasattr(tensor, "detach"):
            raise TypeError("actor state entry is not a tensor: {}".format(name))
        value = tensor.detach().cpu().contiguous()
        if str(getattr(value, "layout", "")) != "torch.strided":
            raise TypeError("actor state entry is not strided: {}".format(name))
        metadata = {
            "dtype": str(value.dtype),
            "name": name,
            "shape": [int(dimension) for dimension in value.shape],
        }
        _update_bytes(
            digest,
            json.dumps(
                metadata, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        )
        # Viewing the contiguous tensor as bytes avoids dtype-dependent NumPy
        # conversion (notably bfloat16) while preserving exact parameter bits.
        # ``view(torch.uint8)`` requires torch without importing it globally;
        # obtain the dtype from the tensor module so this module stays lazy.
        torch_module = __import__("torch")
        raw = value.reshape(-1).view(torch_module.uint8)
        _update_bytes(digest, raw.numpy().tobytes(order="C"))
    return digest.hexdigest()


def deployable_policy_sha256(checkpoint: Mapping) -> str:
    """Hash every checkpoint field that changes deterministic deployment.

    Critic/optimizer/training metadata are intentionally excluded. Runtime and
    depth-mask fields are normalized through the same compatibility readers as
    Unity evaluation, so an older BC checkpoint and a current AWAC checkpoint
    with the same resolved deployment semantics compare equal.
    """

    if not isinstance(checkpoint, Mapping):
        raise TypeError("policy checkpoint must contain a mapping")
    safety_mask, execution_mode = checkpoint_runtime_contract(dict(checkpoint))
    depth_mask = checkpoint_depth_mask_numeric_config(checkpoint)
    required = (
        "feature_contract_id",
        "policy_input_contract_sha256",
        "task_contract_id",
        "task_contract_sha256",
        "mpl_contract_sha256",
        "vec_dim",
        "num_actions",
        "feature_mean",
        "feature_std",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(
            "policy checkpoint lacks deployable fields: {}".format(
                ", ".join(missing)
            )
        )
    metadata = {
        "actor_state_sha256": actor_state_sha256(checkpoint),
        "depth_history_frames": int(checkpoint.get("depth_history_frames", 1)),
        "depth_mask_config": depth_mask,
        "execution_mode": execution_mode,
        "feature_contract_id": str(checkpoint["feature_contract_id"]),
        "initial_prev_action": int(
            checkpoint.get("initial_prev_action", INITIAL_PREV_ACTION)
        ),
        "mpl_contract_sha256": str(checkpoint["mpl_contract_sha256"]),
        "num_actions": int(checkpoint["num_actions"]),
        "policy_input_contract_sha256": str(
            checkpoint["policy_input_contract_sha256"]
        ),
        "privileged_runtime_inputs": list(
            checkpoint.get("privileged_runtime_inputs", [])
        ),
        "safety_mask": safety_mask,
        "task_contract_id": str(checkpoint["task_contract_id"]),
        "task_contract_sha256": str(checkpoint["task_contract_sha256"]),
        "vec_dim": int(checkpoint["vec_dim"]),
    }
    digest = hashlib.sha256()
    _update_bytes(
        digest, DEPLOYABLE_POLICY_FINGERPRINT_CONTRACT_ID.encode("utf-8")
    )
    _update_bytes(
        digest,
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        ),
    )
    for name in ("feature_mean", "feature_std"):
        values = np.asarray(checkpoint[name], dtype=np.float32)
        array_metadata = {
            "dtype": "float32",
            "name": name,
            "shape": [int(dimension) for dimension in values.shape],
        }
        _update_bytes(
            digest,
            json.dumps(
                array_metadata, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        )
        _update_bytes(
            digest,
            np.ascontiguousarray(values).tobytes(order="C"),
        )
    return digest.hexdigest()


def checkpoint_actor_sha256(path: Path) -> str:
    """Load a BC/AWAC checkpoint on CPU and fingerprint its deployable Actor."""

    torch = __import__("torch")
    resolved = Path(path).expanduser().resolve()
    try:
        checkpoint = torch.load(str(resolved), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(str(resolved), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("policy checkpoint must contain a mapping")
    return actor_state_sha256(checkpoint)


def checkpoint_deployable_policy_sha256(path: Path) -> str:
    """Load a checkpoint and fingerprint its deterministic deployed policy."""

    torch = __import__("torch")
    resolved = Path(path).expanduser().resolve()
    try:
        checkpoint = torch.load(str(resolved), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(str(resolved), map_location="cpu")
    return deployable_policy_sha256(checkpoint)


__all__ = [
    "POLICY_ACTOR_FINGERPRINT_CONTRACT_ID",
    "DEPLOYABLE_POLICY_FINGERPRINT_CONTRACT_ID",
    "actor_state_sha256",
    "checkpoint_actor_sha256",
    "checkpoint_deployable_policy_sha256",
    "deployable_policy_sha256",
]
