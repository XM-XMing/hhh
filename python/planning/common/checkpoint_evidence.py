"""Verifiable P3 frozen-checkpoint evidence.

The helpers in this module operate on persisted checkpoint payloads.  They do
not participate in action selection, replay, or optimizer scheduling.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Mapping, Optional

from planning.contracts.policy_checkpoint_fingerprint import actor_state_sha256


P3_FROZEN_CHECKPOINT_EVIDENCE_SCHEMA = "p3_frozen_checkpoint_evidence_v1"


class CheckpointEvidenceError(ValueError):
    """A checkpoint exists but cannot support the P3-D evidence contract."""


def frozen_checkpoint_filename(global_step: int) -> str:
    """Return the immutable P3 artifact name for one exact global step."""

    if int(global_step) < 0:
        raise CheckpointEvidenceError("global_step must be non-negative")
    return "checkpoint_{}.pt".format(int(global_step))


def _update_bytes(digest, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _tensor_mapping_sha256(*, contract: str, mappings: Mapping[str, Mapping]) -> str:
    """Hash named tensor mappings without importing torch at module import time."""

    digest = hashlib.sha256()
    _update_bytes(digest, contract.encode("utf-8"))
    torch_module = __import__("torch")
    for mapping_name in sorted(mappings):
        state = mappings[mapping_name]
        if not isinstance(state, Mapping) or not state:
            raise CheckpointEvidenceError("{} must be a non-empty mapping".format(mapping_name))
        for tensor_name in sorted(state):
            tensor = state[tensor_name]
            if not hasattr(tensor, "detach"):
                raise CheckpointEvidenceError(
                    "{} entry is not a tensor: {}".format(mapping_name, tensor_name)
                )
            value = tensor.detach().cpu().contiguous()
            metadata = {
                "mapping": str(mapping_name),
                "name": str(tensor_name),
                "dtype": str(value.dtype),
                "shape": [int(dimension) for dimension in value.shape],
            }
            _update_bytes(
                digest,
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                ),
            )
            raw = value.reshape(-1).view(torch_module.uint8)
            _update_bytes(digest, raw.numpy().tobytes(order="C"))
    return digest.hexdigest()


def critic_state_sha256(checkpoint: Mapping) -> str:
    """Return a deterministic identity for both online and target Critics."""

    required = (
        "critic1_state_dict",
        "critic2_state_dict",
        "target_critic1_state_dict",
        "target_critic2_state_dict",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise CheckpointEvidenceError(
            "checkpoint lacks critic state: {}".format(", ".join(missing))
        )
    return _tensor_mapping_sha256(
        contract="p3_critic_state_sha256",
        mappings={name: checkpoint[name] for name in required},
    )


def validate_frozen_checkpoint_payload(
    checkpoint: Mapping,
    *,
    expected_global_step: Optional[int] = None,
    require_runtime_provenance: bool = False,
) -> dict:
    """Validate the public metadata required to evaluate a frozen P3 checkpoint."""

    if not isinstance(checkpoint, Mapping):
        raise CheckpointEvidenceError("checkpoint must be a mapping")
    if expected_global_step is not None and int(checkpoint.get("global_step", -1)) != int(
        expected_global_step
    ):
        raise CheckpointEvidenceError("global_step does not match frozen checkpoint target")
    for name in (
        "global_step",
        "replay_size",
        "update_step",
        "actor_update_count",
        "bc_reference_fingerprint",
        "actor_optimizer_state_dict",
        "critic_optimizer_state_dict",
    ):
        if name not in checkpoint or checkpoint[name] in (None, ""):
            raise CheckpointEvidenceError("checkpoint lacks {}".format(name))
    evidence = {
        "schema": P3_FROZEN_CHECKPOINT_EVIDENCE_SCHEMA,
        "global_step": int(checkpoint["global_step"]),
        "replay_size": int(checkpoint["replay_size"]),
        "critic_update_step": int(checkpoint["update_step"]),
        "actor_update_step": int(checkpoint["actor_update_count"]),
        "actor_fingerprint": actor_state_sha256(checkpoint),
        "critic_fingerprint": critic_state_sha256(checkpoint),
        "bc_reference_fingerprint": str(checkpoint["bc_reference_fingerprint"]),
    }
    provenance_names = (
        "training_run_id",
        "runtime_instance_lifecycle",
        "runtime_instance_ledger_validation",
        "worker_runtime_provenance_summary",
    )
    has_provenance = any(name in checkpoint for name in provenance_names)
    if require_runtime_provenance or has_provenance:
        missing = [name for name in provenance_names if name not in checkpoint]
        if missing:
            raise CheckpointEvidenceError(
                "checkpoint lacks runtime provenance: {}".format(", ".join(missing))
            )
        lifecycle = checkpoint["runtime_instance_lifecycle"]
        validation = checkpoint["runtime_instance_ledger_validation"]
        if not isinstance(lifecycle, Mapping) or not isinstance(validation, Mapping):
            raise CheckpointEvidenceError("checkpoint runtime provenance must be mappings")
        training_run_id = str(checkpoint["training_run_id"])
        if not training_run_id or str(lifecycle.get("training_run_id", "")) != training_run_id:
            raise CheckpointEvidenceError("training_run_id does not match runtime lifecycle")
        runtime_ids = sorted({
            str(worker["runtime_instance_id"])
            for segment in lifecycle.get("segments", [])
            for worker in segment.get("workers", [])
            if str(worker.get("runtime_instance_id", ""))
        })
        expected_summary = {
            "runtime_instance_count": int(validation["runtime_instance_count"]),
            "runtime_segments_per_worker": dict(
                validation["runtime_segments_per_worker"]
            ),
            "runtime_instance_ids": runtime_ids,
        }
        if dict(checkpoint["worker_runtime_provenance_summary"]) != expected_summary:
            raise CheckpointEvidenceError("worker_runtime_provenance_summary mismatch")
        evidence.update({
            "training_run_id": training_run_id,
            "runtime_instance_ledger_validation": dict(validation),
            "worker_runtime_provenance_summary": expected_summary,
        })
    return evidence


def save_frozen_checkpoint(path: Path, *, torch, checkpoint: Mapping) -> dict:
    """Atomically persist a checkpoint bound to its model-state fingerprints."""

    payload = dict(checkpoint)
    evidence = validate_frozen_checkpoint_payload(
        payload, expected_global_step=int(payload["global_step"])
    )
    payload["p3_frozen_checkpoint_evidence"] = evidence
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(destination))
    return evidence


def load_verified_frozen_checkpoint(
    path: Path,
    *,
    torch,
    expected_global_step: Optional[int] = None,
    require_runtime_provenance: bool = False,
) -> dict:
    """Load a frozen checkpoint only when saved and recomputed evidence agree."""

    try:
        payload = torch.load(str(Path(path)), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(str(Path(path)), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise CheckpointEvidenceError("checkpoint must be a mapping")
    evidence = validate_frozen_checkpoint_payload(
        payload,
        expected_global_step=expected_global_step,
        require_runtime_provenance=require_runtime_provenance,
    )
    saved = payload.get("p3_frozen_checkpoint_evidence")
    if not isinstance(saved, Mapping):
        raise CheckpointEvidenceError("checkpoint lacks p3_frozen_checkpoint_evidence")
    for name, value in evidence.items():
        if saved.get(name) != value:
            raise CheckpointEvidenceError("{} mismatch".format(name))
    return dict(payload)


def attach_runtime_provenance(
    path: Path,
    *,
    torch,
    runtime_instance_lifecycle: Mapping,
    runtime_instance_ledger_validation: Mapping,
) -> dict:
    """Bind post-run reliable-v4 lifecycle evidence to one frozen checkpoint."""

    destination = Path(path)
    try:
        payload = torch.load(str(destination), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(str(destination), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise CheckpointEvidenceError("checkpoint must be a mapping")
    payload = dict(payload)
    lifecycle = dict(runtime_instance_lifecycle)
    validation = dict(runtime_instance_ledger_validation)
    training_run_id = str(lifecycle.get("training_run_id", ""))
    if not training_run_id:
        raise CheckpointEvidenceError("runtime lifecycle lacks training_run_id")
    runtime_ids = sorted({
        str(worker["runtime_instance_id"])
        for segment in lifecycle.get("segments", [])
        for worker in segment.get("workers", [])
        if str(worker.get("runtime_instance_id", ""))
    })
    payload["training_run_id"] = training_run_id
    payload["runtime_instance_lifecycle"] = lifecycle
    payload["runtime_instance_ledger_validation"] = validation
    payload["worker_runtime_provenance_summary"] = {
        "runtime_instance_count": int(validation["runtime_instance_count"]),
        "runtime_segments_per_worker": dict(validation["runtime_segments_per_worker"]),
        "runtime_instance_ids": runtime_ids,
    }
    payload["p3_frozen_checkpoint_evidence"] = validate_frozen_checkpoint_payload(
        payload, expected_global_step=int(payload["global_step"]), require_runtime_provenance=True
    )
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(destination))
    return dict(payload["p3_frozen_checkpoint_evidence"])


__all__ = [
    "P3_FROZEN_CHECKPOINT_EVIDENCE_SCHEMA",
    "CheckpointEvidenceError",
    "attach_runtime_provenance",
    "critic_state_sha256",
    "frozen_checkpoint_filename",
    "load_verified_frozen_checkpoint",
    "save_frozen_checkpoint",
    "validate_frozen_checkpoint_payload",
]
