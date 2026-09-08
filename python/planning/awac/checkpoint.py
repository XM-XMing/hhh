"""Formal discrete AWAC checkpoint schema and atomic persistence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from planning.awac.contract import (
    AWAC_ALGORITHM_ID,
    AWAC_CHECKPOINT_CONTRACT_ID,
    AWAC_MODEL_TYPE,
    AWAC_TRAINING_CONFIG_CONTRACT_ID,
    AWAC_PHASE_CRITIC_CALIBRATION,
    AWAC_REWARD_SCALE,
    AWAC_REWARD_SCALE_OWNER,
    AWAC_PHASE_STANDARD_TRAINING,
    replay_contract_sha256,
    validate_awac_checkpoint_identity,
)
from planning.awac.phase1 import (
    PHASE_ACTOR_ENABLED_STANDARD_AWAC,
    PHASE_CRITIC_CALIBRATION,
    Phase1ReadinessError,
    Phase1StateMachine,
    require_calibration_pass,
    validate_phase1_depth_learning_rates,
)
from planning.common import canonical_json_sha256
from planning.common.checkpoint import load_torch, save_torch_atomic
from planning.common.hashing import file_sha256
from planning.awac.storage import garbage_collect_checkpoint_generations
from planning.contracts.feature import (
    FEATURE_CONTRACT_ID,
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    policy_input_contract_sha256,
)
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.policy_runtime import POLICY_RUNTIME_CONTRACT_ID
from planning.contracts.policy_checkpoint_fingerprint import actor_state_sha256
from planning.contracts.reward import REWARD_CONTRACT_ID, reward_contract_sha256
from planning.contracts.task import (
    TASK_CONTRACT_ID,
    TASK_CONTRACT_SCHEMA_VERSION,
    task_contract_sha256,
)
from planning.version import SOFTWARE_VERSION


AWAC_CHECKPOINT_SCHEMA_ID = "awac_checkpoint_schema_v3"

CALIBRATION_EXACT_RESUME_STATE_SCHEMA_ID = (
    "awac_calibration_exact_resume_state_v1"
)
CALIBRATION_CHECKPOINT_TRANSACTION_SCHEMA_ID = (
    "awac_calibration_checkpoint_transaction_v1"
)
CALIBRATION_CHECKPOINT_TRANSACTION_DIAGNOSTICS_SCHEMA_ID = (
    "awac_calibration_checkpoint_transaction_diagnostics_v1"
)
CALIBRATION_REPLAY_PENDING_GENERATION_SCHEMA_ID = (
    "awac_calibration_replay_pending_generation_v1"
)
STANDARD_ONLINE_EXACT_RESUME_STATE_SCHEMA_ID = (
    "awac_standard_online_exact_resume_state_v1"
)
STANDARD_ONLINE_RUNTIME_SCHEMA_ID = "awac_standard_online_runtime_v1"

# The standard phase reuses the calibration transaction machinery, but has a
# different exact-resume payload.  Keeping the state-owner list explicit makes
# a missing online cursor or schedule counter fail closed at the checkpoint
# boundary instead of being silently reconstructed from a seed.
STANDARD_ONLINE_EXACT_RESUME_REQUIRED_STATES = (
    "learner_parameters_and_optimizer_state",
    "learner_counters",
    "replay_identity",
    "producer_replay_sampler_random_state",
    "python_global_random_state",
    "numpy_global_random_state",
    "torch_cpu_random_state",
    "torch_cuda_random_state",
    "mission_source_identity_and_ordered_ids",
    "mission_cursor_and_completed_ids",
    "environment_and_online_counters",
    "online_schedule_counters",
    "phase1_state",
    "runtime_identity",
)

# These are logical state owners, not another serialized training schema.  The
# model, replay, mission, split, and runtime entries are already carried by
# the formal checkpoint.  The final six entries are captured below because a
# seed alone cannot reproduce their future sequence after a resume.
CALIBRATION_EXACT_RESUME_REQUIRED_STATES = (
    "actor_parameters",
    "critic1_parameters",
    "critic2_parameters",
    "target_critic1_parameters",
    "target_critic2_parameters",
    "actor_optimizer_state",
    "critic_optimizer_state",
    "learner_counters_and_frozen_actor_state",
    "replay_arrays_cursor_and_metadata_identity",
    "mission_source_identity_and_ordered_ids",
    "train_holdout_split_identity",
    "managed_worker_and_runtime_identity",
    "completed_mission_cursor_and_ids",
    "environment_and_episode_counters",
    "gate_history_and_controller_state",
    "completed_holdout_episode_ids",
    "runtime_configuration_and_phase_identity",
    "raw_completed_holdout_transition_records",
    "producer_replay_sampler_random_state",
    "python_global_random_state",
    "numpy_global_random_state",
    "torch_cpu_random_state",
    "torch_cuda_random_state",
)

_RAW_HOLDOUT_RECORD_REQUIRED_FIELDS = (
    "episode_id",
    "mission_id",
    "episode_transition_index",
    "holdout_record_index",
    "depth",
    "vector",
    "action_mask",
    "action",
    "reward",
    "next_depth",
    "next_vector",
    "next_action_mask",
    "done",
    "behavior_source",
    "terminal_reason",
)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(
        ".{}.{}.tmp".format(destination.name, uuid.uuid4().hex)
    )


def write_calibration_transaction_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably publish a small calibration transaction record."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    encoded = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(destination))
        _fsync_directory(destination.parent)
    except BaseException:
        # A leftover temporary file is deliberately not a commit marker.
        raise


def remove_calibration_transaction_file(path: Path) -> None:
    """Durably remove a stale, non-authoritative transaction sidecar."""

    target = Path(path).expanduser().resolve()
    try:
        target.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(target.parent)


def _save_torch_durable_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    torch,
    failpoint: Optional[Callable[[str], None]] = None,
) -> None:
    """Write one checkpoint file durably without making it a generation commit."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    if failpoint is not None:
        failpoint("before_checkpoint_temp_write")
    try:
        torch.save(dict(payload), str(temporary))
        if failpoint is not None:
            # This is deliberately before fsync/rename: tests use it to
            # model a partially durable temporary checkpoint.
            failpoint("checkpoint_temp_write")
        _fsync_file(temporary)
        if failpoint is not None:
            failpoint("checkpoint_temp_complete")
        os.replace(str(temporary), str(destination))
        _fsync_directory(destination.parent)
        if failpoint is not None:
            failpoint("after_checkpoint_rename")
    except BaseException:
        # Do not promote, load, or resume an uncommitted temporary file.
        raise


def _resume_mapping_key_bytes(key: Any) -> bytes:
    """Encode one non-string mapping key with an explicit type identity."""

    if isinstance(key, bool):
        return b"bool\0" + (b"true" if key else b"false")
    if isinstance(key, str):
        return b"str\0" + key.encode("utf-8")
    if isinstance(key, int):
        return b"int\0" + str(key).encode("ascii")
    if isinstance(key, float):
        return b"float\0" + json.dumps(
            key, allow_nan=True, separators=(",", ":")
        ).encode("ascii")
    if key is None:
        return b"none"
    if isinstance(key, bytes):
        return b"bytes\0" + key
    if isinstance(key, np.generic):
        scalar = np.asarray(key)
        if scalar.dtype.hasobject:
            raise TypeError(
                "unsupported exact-resume mapping key: numpy object scalar"
            )
        return (
            b"numpy_scalar\0"
            + str(scalar.dtype).encode("ascii")
            + b"\0"
            + scalar.tobytes(order="C")
        )
    if isinstance(key, tuple):
        encoded_items = []
        for item in key:
            encoded = _resume_mapping_key_bytes(item)
            encoded_items.append(
                str(len(encoded)).encode("ascii") + b":" + encoded
            )
        return b"tuple\0" + b"".join(encoded_items)
    raise TypeError(
        "unsupported exact-resume mapping key: {}".format(type(key).__name__)
    )


def _hash_resume_value(digest, value: Any) -> None:
    """Hash nested resume values with stable, type-aware map keys.

    Existing calibration holdout records use string-key mappings.  Their
    historical byte encoding is retained exactly for checkpoint compatibility.
    A mapping containing any non-string key uses the typed encoding so integer
    optimizer parameter IDs cannot be coerced into string lookups or collide
    with a real string key.
    """

    if isinstance(value, Mapping):
        items = list(value.items())
        if all(isinstance(key, str) for key, _ in items):
            # Preserve the pre-existing calibration holdout fingerprint for
            # all-string mappings already committed in V6 artifacts.
            digest.update(b"mapping\0")
            for key in sorted(str(name) for name, _ in items):
                digest.update(key.encode("utf-8"))
                digest.update(b"\0")
                _hash_resume_value(digest, value[key])
            return

        digest.update(b"mapping_typed\0")
        encoded_items = [
            (_resume_mapping_key_bytes(key), item) for key, item in items
        ]
        for key_bytes, item in sorted(encoded_items, key=lambda pair: pair[0]):
            digest.update(b"key\0")
            digest.update(str(len(key_bytes)).encode("ascii"))
            digest.update(b"\0")
            digest.update(key_bytes)
            digest.update(b"\0")
            _hash_resume_value(digest, item)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
        return
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        if scalar.dtype.hasobject:
            raise TypeError(
                "unsupported exact-resume holdout value: numpy object scalar"
            )
        digest.update(b"numpy_scalar\0")
        digest.update(str(scalar.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(scalar.tobytes(order="C"))
        return
    detach = getattr(value, "detach", None)
    if callable(detach):
        array = value.detach().cpu().contiguous().numpy()
        _hash_resume_value(digest, array)
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_resume_value(digest, item)
            digest.update(b"\0")
        return
    if value is None:
        digest.update(b"none")
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0")
        digest.update(value)
        return
    if isinstance(value, (str, bool, int, float)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        return
    raise TypeError(
        "unsupported exact-resume holdout value: {}".format(type(value).__name__)
    )


def calibration_holdout_records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint the existing raw holdout record dictionaries in order."""

    rows = list(records)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("calibration holdout records must be mappings")
    for index, row in enumerate(rows):
        missing = [
            name for name in _RAW_HOLDOUT_RECORD_REQUIRED_FIELDS if name not in row
        ]
        if missing:
            raise ValueError(
                "calibration holdout record {} is missing: {}".format(
                    index, ", ".join(missing)
                )
            )
        if not str(row["episode_id"]).strip() or not str(
            row["mission_id"]
        ).strip():
            raise ValueError("calibration holdout record identity is missing")
        if int(row["episode_transition_index"]) < 0 or int(
            row["holdout_record_index"]
        ) != index:
            raise ValueError("calibration holdout record ordering is invalid")
        if not str(row["terminal_reason"]).strip():
            raise ValueError("calibration holdout record terminal reason is missing")
    digest = hashlib.sha256()
    digest.update(b"awac_calibration_holdout_records_v1\0")
    _hash_resume_value(digest, rows)
    return digest.hexdigest()


def _clone_torch_rng_state(state):
    clone = getattr(state, "clone", None)
    return clone() if callable(clone) else copy.deepcopy(state)


def build_calibration_exact_resume_state(
    *,
    raw_holdout_records: Sequence[Mapping[str, Any]],
    producer_rng: np.random.RandomState,
    torch,
) -> Dict[str, Any]:
    """Capture the non-seed state needed to continue calibration exactly."""

    if not isinstance(producer_rng, np.random.RandomState):
        raise TypeError("calibration producer RNG must be numpy RandomState")
    records = copy.deepcopy(list(raw_holdout_records))
    record_sha256 = calibration_holdout_records_sha256(records)
    torch_cpu_rng_state = None
    torch_cuda_rng_state_all = []
    torch_cuda_available = False
    if torch is not None:
        torch_cpu_rng_state = _clone_torch_rng_state(torch.get_rng_state())
        cuda = getattr(torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        torch_cuda_available = bool(callable(is_available) and is_available())
        if torch_cuda_available:
            torch_cuda_rng_state_all = [
                _clone_torch_rng_state(value)
                for value in cuda.get_rng_state_all()
            ]
    return {
        "schema_id": CALIBRATION_EXACT_RESUME_STATE_SCHEMA_ID,
        "required_state_names": list(CALIBRATION_EXACT_RESUME_REQUIRED_STATES),
        "required_state_count": len(CALIBRATION_EXACT_RESUME_REQUIRED_STATES),
        "raw_holdout_records": records,
        "raw_holdout_records_sha256": record_sha256,
        "producer_rng_state": copy.deepcopy(producer_rng.get_state()),
        "python_rng_state": copy.deepcopy(random.getstate()),
        "numpy_rng_state": copy.deepcopy(np.random.get_state()),
        "torch_cpu_rng_state": torch_cpu_rng_state,
        "torch_cuda_rng_available": bool(torch_cuda_available),
        "torch_cuda_rng_state_all": torch_cuda_rng_state_all,
    }


def validate_calibration_exact_resume_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Reject a checkpoint that cannot reconstruct the exact producer state."""

    if not isinstance(state, Mapping):
        raise ValueError("calibration exact resume state is missing")
    if state.get("schema_id") != CALIBRATION_EXACT_RESUME_STATE_SCHEMA_ID:
        raise ValueError("calibration exact resume state schema mismatch")
    names = state.get("required_state_names")
    if list(names or ()) != list(CALIBRATION_EXACT_RESUME_REQUIRED_STATES):
        raise ValueError("calibration exact resume required-state identity mismatch")
    if int(state.get("required_state_count", -1)) != len(
        CALIBRATION_EXACT_RESUME_REQUIRED_STATES
    ):
        raise ValueError("calibration exact resume required-state count mismatch")
    required = (
        "raw_holdout_records",
        "raw_holdout_records_sha256",
        "producer_rng_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_available",
        "torch_cuda_rng_state_all",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(
            "calibration exact resume state missing: {}".format(", ".join(missing))
        )
    records = state["raw_holdout_records"]
    if not isinstance(records, (list, tuple)) or not all(
        isinstance(row, Mapping) for row in records
    ):
        raise ValueError("calibration exact resume holdout records are invalid")
    if state["raw_holdout_records_sha256"] != calibration_holdout_records_sha256(
        records
    ):
        raise ValueError("calibration exact resume holdout records SHA mismatch")
    if not isinstance(state["torch_cuda_rng_state_all"], (list, tuple)):
        raise ValueError("calibration exact resume CUDA RNG state is invalid")
    return dict(state)


def restore_calibration_exact_resume_state(
    state: Mapping[str, Any],
    *,
    producer_rng: np.random.RandomState,
    torch,
) -> Dict[str, Any]:
    """Restore raw holdout evidence and every RNG owner captured at save."""

    validated = validate_calibration_exact_resume_state(state)
    if not isinstance(producer_rng, np.random.RandomState):
        raise TypeError("calibration producer RNG must be numpy RandomState")
    producer_rng.set_state(copy.deepcopy(validated["producer_rng_state"]))
    random.setstate(copy.deepcopy(validated["python_rng_state"]))
    np.random.set_state(copy.deepcopy(validated["numpy_rng_state"]))

    if torch is None:
        if validated["torch_cpu_rng_state"] is not None:
            raise ValueError("calibration exact resume Torch CPU RNG cannot be restored")
        if validated["torch_cuda_rng_available"]:
            raise ValueError("calibration exact resume CUDA RNG cannot be restored")
    else:
        if validated["torch_cpu_rng_state"] is None:
            raise ValueError("calibration exact resume Torch CPU RNG state is missing")
        torch.set_rng_state(_clone_torch_rng_state(validated["torch_cpu_rng_state"]))
        cuda = getattr(torch, "cuda", None)
        available = bool(
            callable(getattr(cuda, "is_available", None)) and cuda.is_available()
        )
        if available != bool(validated["torch_cuda_rng_available"]):
            raise ValueError("calibration exact resume CUDA capability mismatch")
        if available:
            states = [
                _clone_torch_rng_state(value)
                for value in validated["torch_cuda_rng_state_all"]
            ]
            if not states:
                raise ValueError("calibration exact resume CUDA RNG state is missing")
            cuda.set_rng_state_all(states)

    return {
        "raw_holdout_records": copy.deepcopy(
            list(validated["raw_holdout_records"])
        ),
        "raw_holdout_records_sha256": str(
            validated["raw_holdout_records_sha256"]
        ),
        "required_state_count": int(validated["required_state_count"]),
    }


def _resume_state_sha256(value: Any) -> str:
    """Hash a checkpoint-owned nested state without serializing it twice."""

    digest = hashlib.sha256()
    digest.update(b"awac_standard_online_resume_value_v1\0")
    _hash_resume_value(digest, value)
    return digest.hexdigest()


def _learner_state_from_checkpoint(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Project the learner fields from a complete checkpoint payload."""

    fields = (
        "awac_learner_state_schema_id",
        "actor_state_dict",
        "critic1_state_dict",
        "critic2_state_dict",
        "target_critic1_state_dict",
        "target_critic2_state_dict",
        "actor_optimizer_state_dict",
        "critic_optimizer_state_dict",
        "update_step",
        "actor_update_count",
        "actor_awac_update_count",
        "actor_recovery_update_count",
        "actor_trust_region_rejection_count",
        "actor_optimizer_step_count",
        "critic_update_count",
        "actor_frozen_for_calibration",
    )
    missing = [name for name in fields if name not in payload]
    if missing:
        raise ValueError(
            "standard AWAC checkpoint learner state is missing: {}".format(
                ", ".join(missing)
            )
        )
    state = {name: payload[name] for name in fields}
    # New checkpoints persist the LR provenance inventory as part of the
    # learner state fingerprint.  Older committed Standard checkpoints were
    # written before that optional inventory existed; omitting it here keeps
    # their exact-resume fingerprint backward-compatible.
    if "optimizer_lr_provenance" in payload:
        state["optimizer_lr_provenance"] = payload["optimizer_lr_provenance"]
    return state


def build_standard_online_exact_resume_state(
    *,
    learner_state: Mapping[str, Any],
    producer_rng: np.random.RandomState,
    torch,
    mission_source_identity: Mapping[str, Any],
    mission_progress: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    phase1_state: Mapping[str, Any],
    replay_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Capture every state owner needed by a Standard AWAC resume.

    Model and optimizer tensors remain in the normal checkpoint fields.  This
    object stores their deterministic fingerprint together with the mutable
    online/runtime cursors and RNG owners, avoiding a second multi-megabyte
    copy while still making omissions fail closed.
    """

    if not isinstance(learner_state, Mapping) or not learner_state:
        raise TypeError("standard online learner state must be a mapping")
    if not isinstance(producer_rng, np.random.RandomState):
        raise TypeError("standard online producer RNG must be numpy RandomState")
    for name, value in (
        ("mission_source_identity", mission_source_identity),
        ("mission_progress", mission_progress),
        ("runtime_identity", runtime_identity),
        ("phase1_state", phase1_state),
    ):
        if not isinstance(value, Mapping):
            raise TypeError("standard online {} must be a mapping".format(name))

    torch_cpu_rng_state = None
    torch_cuda_rng_state_all = []
    torch_cuda_available = False
    if torch is not None:
        torch_cpu_rng_state = _clone_torch_rng_state(torch.get_rng_state())
        cuda = getattr(torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        torch_cuda_available = bool(callable(is_available) and is_available())
        if torch_cuda_available:
            torch_cuda_rng_state_all = [
                _clone_torch_rng_state(value) for value in cuda.get_rng_state_all()
            ]

    progress = dict(mission_progress)
    if "online_transitions_committed" not in progress:
        raise ValueError(
            "standard online resume progress is missing online_transitions_committed"
        )
    counter_names = (
        "environment_step_count",
        "online_env_steps",
        "online_transitions_committed",
        "completed_episode_count",
        "next_mission_index",
        "critic_update_count",
        "actor_update_count",
        "actor_optimizer_step_count",
        "update_step",
    )
    counters = {
        name: int(progress[name])
        if name == "online_transitions_committed"
        else int(progress.get(name, 0))
        for name in counter_names
    }
    if any(value < 0 for value in counters.values()):
        raise ValueError("standard online resume counters must be non-negative")
    if counters["online_transitions_committed"] > counters["online_env_steps"]:
        raise ValueError(
            "standard online committed transitions exceed environment steps"
        )
    learner_counters = {
        name: int(learner_state.get(name, 0))
        for name in (
            "update_step",
            "actor_update_count",
            "actor_awac_update_count",
            "actor_recovery_update_count",
            "actor_trust_region_rejection_count",
            "actor_optimizer_step_count",
            "critic_update_count",
        )
    }
    if any(value < 0 for value in learner_counters.values()):
        raise ValueError("standard online learner counters must be non-negative")
    return {
        "schema_id": STANDARD_ONLINE_EXACT_RESUME_STATE_SCHEMA_ID,
        "required_state_names": list(STANDARD_ONLINE_EXACT_RESUME_REQUIRED_STATES),
        "required_state_count": len(STANDARD_ONLINE_EXACT_RESUME_REQUIRED_STATES),
        "learner_state_sha256": _resume_state_sha256(dict(learner_state)),
        "learner_state_field_names": sorted(str(name) for name in learner_state),
        "learner_counters": learner_counters,
        "replay_identity": dict(replay_identity or {}),
        "mission_source_identity": dict(mission_source_identity),
        "mission_progress": progress,
        "mission_ordered_ids": list(progress.get("ordered_mission_ids", [])),
        "completed_mission_ids": list(progress.get("completed_mission_ids", [])),
        "environment_and_online_counters": counters,
        "online_schedule_counters": {
            name: counters[name]
            for name in (
                "critic_update_count",
                "actor_update_count",
                "actor_optimizer_step_count",
                "update_step",
            )
        },
        "phase1_state": dict(phase1_state),
        "runtime_identity": dict(runtime_identity),
        "producer_rng_state": copy.deepcopy(producer_rng.get_state()),
        "python_rng_state": copy.deepcopy(random.getstate()),
        "numpy_rng_state": copy.deepcopy(np.random.get_state()),
        "torch_cpu_rng_state": torch_cpu_rng_state,
        "torch_cuda_rng_available": bool(torch_cuda_available),
        "torch_cuda_rng_state_all": torch_cuda_rng_state_all,
    }


def validate_standard_online_exact_resume_state(
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the complete Standard online resume-owner inventory."""

    if not isinstance(state, Mapping):
        raise ValueError("standard online exact resume state is missing")
    if state.get("schema_id") != STANDARD_ONLINE_EXACT_RESUME_STATE_SCHEMA_ID:
        raise ValueError("standard online exact resume state schema mismatch")
    if list(state.get("required_state_names", ())) != list(
        STANDARD_ONLINE_EXACT_RESUME_REQUIRED_STATES
    ):
        raise ValueError("standard online exact resume required-state identity mismatch")
    if int(state.get("required_state_count", -1)) != len(
        STANDARD_ONLINE_EXACT_RESUME_REQUIRED_STATES
    ):
        raise ValueError("standard online exact resume required-state count mismatch")
    required = (
        "learner_state_sha256",
        "learner_state_field_names",
        "learner_counters",
        "replay_identity",
        "mission_source_identity",
        "mission_progress",
        "mission_ordered_ids",
        "completed_mission_ids",
        "environment_and_online_counters",
        "online_schedule_counters",
        "phase1_state",
        "runtime_identity",
        "producer_rng_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_available",
        "torch_cuda_rng_state_all",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(
            "standard online exact resume state missing: {}".format(
                ", ".join(missing)
            )
        )
    if len(str(state["learner_state_sha256"])) != 64:
        raise ValueError("standard online learner state SHA is invalid")
    if not isinstance(state["learner_state_field_names"], (list, tuple)):
        raise ValueError("standard online learner state field names are invalid")
    for name in (
        "learner_counters",
        "environment_and_online_counters",
        "online_schedule_counters",
    ):
        if not isinstance(state[name], Mapping):
            raise ValueError("standard online resume {} is invalid".format(name))
    expected_learner_counters = {
        "update_step",
        "actor_update_count",
        "actor_awac_update_count",
        "actor_recovery_update_count",
        "actor_trust_region_rejection_count",
        "actor_optimizer_step_count",
        "critic_update_count",
    }
    if set(state["learner_counters"]) != expected_learner_counters:
        raise ValueError("standard online learner counter inventory mismatch")
    expected_environment_counters = {
        "environment_step_count",
        "online_env_steps",
        "online_transitions_committed",
        "completed_episode_count",
        "next_mission_index",
        "critic_update_count",
        "actor_update_count",
        "actor_optimizer_step_count",
        "update_step",
    }
    if set(state["environment_and_online_counters"]) != expected_environment_counters:
        raise ValueError("standard online environment counter inventory mismatch")
    if set(state["online_schedule_counters"]) != {
        "critic_update_count",
        "actor_update_count",
        "actor_optimizer_step_count",
        "update_step",
    }:
        raise ValueError("standard online schedule counter inventory mismatch")
    for name in (
        "replay_identity",
        "mission_source_identity",
        "mission_progress",
        "phase1_state",
        "runtime_identity",
    ):
        if not isinstance(state[name], Mapping):
            raise ValueError("standard online resume {} is invalid".format(name))
    if not state["replay_identity"] or not state["runtime_identity"]:
        raise ValueError("standard online resume identity is missing")
    for name in (
        "mission_ordered_ids",
        "completed_mission_ids",
        "torch_cuda_rng_state_all",
    ):
        if not isinstance(state[name], (list, tuple)):
            raise ValueError("standard online resume {} is invalid".format(name))
    for group_name in ("learner_counters", "environment_and_online_counters"):
        for name, value in state[group_name].items():
            if int(value) < 0:
                raise ValueError(
                    "standard online resume counter {} is negative".format(name)
                )
    if list(state["mission_ordered_ids"]) != list(
        state["mission_progress"].get("ordered_mission_ids", [])
    ):
        raise ValueError("standard online resume mission ordering is inconsistent")
    if list(state["completed_mission_ids"]) != list(
        state["mission_progress"].get("completed_mission_ids", [])
    ):
        raise ValueError("standard online resume completed mission identity is inconsistent")
    if dict(state["mission_source_identity"]) != dict(
        state["mission_progress"].get("mission_source_identity", {})
    ):
        raise ValueError("standard online resume mission source identity is inconsistent")
    for name in (
        "environment_step_count",
        "online_env_steps",
        "online_transitions_committed",
        "completed_episode_count",
        "next_mission_index",
    ):
        if int(state["mission_progress"].get(name, -1)) != int(
            state["environment_and_online_counters"][name]
        ):
            raise ValueError(
                "standard online resume mission progress {} mismatch".format(name)
            )
    for name in (
        "update_step",
        "actor_update_count",
        "actor_optimizer_step_count",
        "critic_update_count",
    ):
        if int(state["online_schedule_counters"][name]) != int(
            state["environment_and_online_counters"][name]
        ):
            raise ValueError("standard online resume schedule counter mismatch")
    return dict(state)


def restore_standard_online_exact_resume_state(
    state: Mapping[str, Any],
    *,
    producer_rng: np.random.RandomState,
    torch,
) -> Dict[str, Any]:
    """Restore Standard online RNG owners after learner state is loaded."""

    validated = validate_standard_online_exact_resume_state(state)
    if not isinstance(producer_rng, np.random.RandomState):
        raise TypeError("standard online producer RNG must be numpy RandomState")
    producer_rng.set_state(copy.deepcopy(validated["producer_rng_state"]))
    random.setstate(copy.deepcopy(validated["python_rng_state"]))
    np.random.set_state(copy.deepcopy(validated["numpy_rng_state"]))

    if torch is None:
        if validated["torch_cpu_rng_state"] is not None:
            raise ValueError("standard online Torch CPU RNG cannot be restored")
        if validated["torch_cuda_rng_available"]:
            raise ValueError("standard online CUDA RNG cannot be restored")
    else:
        if validated["torch_cpu_rng_state"] is None:
            raise ValueError("standard online Torch CPU RNG state is missing")
        torch.set_rng_state(_clone_torch_rng_state(validated["torch_cpu_rng_state"]))
        cuda = getattr(torch, "cuda", None)
        available = bool(
            callable(getattr(cuda, "is_available", None)) and cuda.is_available()
        )
        if available != bool(validated["torch_cuda_rng_available"]):
            raise ValueError("standard online CUDA capability mismatch")
        if available:
            states = [
                _clone_torch_rng_state(value)
                for value in validated["torch_cuda_rng_state_all"]
            ]
            if not states:
                raise ValueError("standard online CUDA RNG state is missing")
            cuda.set_rng_state_all(states)
    return {
        "learner_state_sha256": str(validated["learner_state_sha256"]),
        "mission_progress": copy.deepcopy(validated["mission_progress"]),
        "runtime_identity": copy.deepcopy(validated["runtime_identity"]),
        "required_state_count": int(validated["required_state_count"]),
    }


def build_awac_checkpoint_identity(*, max_primitive_steps: int) -> Dict:
    """Build the invariant formal identity shared by every AWAC checkpoint."""

    max_steps = int(max_primitive_steps)
    if max_steps <= 0:
        raise ValueError("max_primitive_steps must be positive")
    return {
        "awac_checkpoint_schema_id": AWAC_CHECKPOINT_SCHEMA_ID,
        "algorithm_id": AWAC_ALGORITHM_ID,
        "awac_checkpoint_contract_id": AWAC_CHECKPOINT_CONTRACT_ID,
        "model_type": AWAC_MODEL_TYPE,
        "training_config_contract_id": AWAC_TRAINING_CONFIG_CONTRACT_ID,
        "software_version": SOFTWARE_VERSION,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_schema_version": TASK_CONTRACT_SCHEMA_VERSION,
        "task_contract_sha256": task_contract_sha256(max_steps),
        "max_primitive_steps": max_steps,
        "reward_contract_id": REWARD_CONTRACT_ID,
        "reward_contract_sha256": reward_contract_sha256(),
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "vec_dim": POLICY_VECTOR_DIM,
        "num_actions": NUM_ACTIONS,
        "depth_history_frames": 1,
        "initial_prev_action": INITIAL_PREV_ACTION,
    }


def validate_awac_checkpoint_payload(payload: Mapping) -> Dict:
    """Validate state and identity required for AWAC resume/evaluation."""
    if not isinstance(payload, Mapping):
        raise TypeError("AWAC checkpoint payload must be a mapping")
    validate_awac_checkpoint_identity(payload)
    required = {
        "actor_state_dict",
        "critic1_state_dict",
        "critic2_state_dict",
        "target_critic1_state_dict",
        "target_critic2_state_dict",
        "actor_optimizer_state_dict",
        "critic_optimizer_state_dict",
        "global_step",
        "replay_size",
        "update_step",
        "observation_contract",
        "observation_source",
        "mpl_contract_sha256",
        "resolved_training_config",
        "resolved_training_config_sha256",
    }
    missing = sorted(name for name in required if name not in payload)
    if missing:
        raise ValueError("AWAC checkpoint missing fields: {}".format(", ".join(missing)))
    if str(payload.get("awac_checkpoint_schema_id", "")) != AWAC_CHECKPOINT_SCHEMA_ID:
        raise ValueError("AWAC checkpoint schema mismatch")
    if payload.get("observation_contract") != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("AWAC checkpoint observation contract mismatch")
    if payload.get("observation_source") != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("AWAC checkpoint observation source mismatch")
    for name in ("global_step", "replay_size", "update_step"):
        if int(payload[name]) < 0:
            raise ValueError("AWAC checkpoint {} must be non-negative".format(name))
    for name in (
        "actor_state_dict", "critic1_state_dict", "critic2_state_dict",
        "target_critic1_state_dict", "target_critic2_state_dict",
        "actor_optimizer_state_dict", "critic_optimizer_state_dict",
    ):
        if not isinstance(payload[name], Mapping):
            raise ValueError("AWAC checkpoint {} is missing or empty".format(name))
        if name != "actor_optimizer_state_dict" and not payload[name]:
            raise ValueError("AWAC checkpoint {} is missing or empty".format(name))
    forbidden = {
        key for key in payload
        if key in {"alpha", "log_alpha", "alpha_optimizer_state_dict", "target_entropy"}
    }
    if forbidden:
        raise ValueError("AWAC checkpoint contains unsupported state: {}".format(sorted(forbidden)))
    return dict(payload)


def validate_calibration_checkpoint_payload(payload: Mapping) -> Dict:
    """Validate the stricter phase/identity contract for Critic Calibration."""

    validated = validate_awac_checkpoint_payload(payload)
    required = {
        "phase",
        "actor_update_enabled",
        "actor_optimizer_status",
        "actor_optimizer_step_count",
        "critic_update_count",
        "environment_step_count",
        "calibration_split",
        "calibration_split_sha256",
        "training_contract",
        "training_contract_sha256",
        "replay_identity",
        "replay_contract_sha256",
        "calibration_metrics",
        "calibration_gate_state",
        "reward_contract_sha256",
        "reward_scale",
        "reward_scale_owner",
        "actor_state_sha256",
        "critic1_state_sha256",
        "critic2_state_sha256",
    }
    missing = sorted(required.difference(validated))
    if missing:
        raise ValueError(
            "calibration checkpoint missing fields: {}".format(", ".join(missing))
        )
    if validated["phase"] != AWAC_PHASE_CRITIC_CALIBRATION:
        raise ValueError("calibration checkpoint phase mismatch")
    if validated["actor_update_enabled"] is not False:
        raise ValueError("calibration checkpoint Actor updates must be disabled")
    if validated.get("actor_frozen_for_calibration") is not True:
        raise ValueError("calibration checkpoint Actor is not frozen")
    if validated["actor_optimizer_status"] != "unused_frozen":
        raise ValueError("calibration checkpoint Actor optimizer status mismatch")
    if int(validated["actor_optimizer_step_count"]) != 0:
        raise ValueError("calibration checkpoint Actor optimizer was used")
    for name in ("critic_update_count", "environment_step_count"):
        if int(validated[name]) < 0:
            raise ValueError("calibration checkpoint {} is invalid".format(name))
    split = validated["calibration_split"]
    if not isinstance(split, Mapping) or not split:
        raise ValueError("calibration checkpoint split identity is missing")
    if validated["calibration_split_sha256"] != canonical_json_sha256(dict(split)):
        raise ValueError("calibration checkpoint split identity mismatch")
    training_contract = validated["training_contract"]
    if not isinstance(training_contract, Mapping) or not training_contract:
        raise ValueError("calibration checkpoint training contract is missing")
    if validated["training_contract_sha256"] != canonical_json_sha256(dict(training_contract)):
        raise ValueError("calibration checkpoint training contract SHA mismatch")
    if training_contract.get("phase") != AWAC_PHASE_CRITIC_CALIBRATION:
        raise ValueError("calibration checkpoint training contract phase mismatch")
    if training_contract.get("actor_update_enabled") is not False:
        raise ValueError("calibration checkpoint training contract enables Actor updates")
    replay_identity = validated["replay_identity"]
    if not isinstance(replay_identity, Mapping) or not replay_identity:
        raise ValueError("calibration checkpoint replay identity is missing")
    if validated["replay_contract_sha256"] != replay_contract_sha256():
        raise ValueError("calibration checkpoint replay contract mismatch")
    if validated["reward_contract_sha256"] != reward_contract_sha256():
        raise ValueError("calibration checkpoint reward contract mismatch")
    if validated["reward_scale_owner"] != AWAC_REWARD_SCALE_OWNER:
        raise ValueError("calibration checkpoint reward-scale owner mismatch")
    if float(validated["reward_scale"]) != float(AWAC_REWARD_SCALE):
        raise ValueError("calibration checkpoint reward scale mismatch")
    if training_contract.get("replay_contract_sha256") != replay_contract_sha256():
        raise ValueError("calibration training contract replay identity mismatch")
    if training_contract.get("reward_contract_sha256") != reward_contract_sha256():
        raise ValueError("calibration training contract reward identity mismatch")
    if float(training_contract.get("reward_scale", -1.0)) != float(AWAC_REWARD_SCALE):
        raise ValueError("calibration training contract reward scale mismatch")
    if training_contract.get("reward_scale_owner") != AWAC_REWARD_SCALE_OWNER:
        raise ValueError("calibration training contract reward-scale owner mismatch")
    if str(validated["calibration_gate_state"]) not in (
        "PENDING",
        "PASS",
        "FAIL_DIVERGED",
    ):
        raise ValueError("calibration checkpoint gate state mismatch")
    for name in ("actor_state_sha256", "critic1_state_sha256", "critic2_state_sha256"):
        if len(str(validated[name])) != 64:
            raise ValueError("calibration checkpoint {} is invalid".format(name))
    if not isinstance(validated["calibration_metrics"], Mapping):
        raise ValueError("calibration checkpoint metrics are missing")
    runtime_schema = validated.get("calibration_runtime_schema_id")
    if runtime_schema is not None and runtime_schema != "awac_formal_calibration_runtime_v1":
        raise ValueError("calibration checkpoint runtime schema mismatch")
    for name in ("mission_source_identity", "mission_progress", "runtime_identity"):
        value = validated.get(name)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError("calibration checkpoint {} must be a mapping".format(name))
    if "completed_episode_count" in validated and int(validated["completed_episode_count"]) < 0:
        raise ValueError("calibration checkpoint completed episode count is invalid")
    exact_resume_state = validated.get("exact_resume_state")
    if exact_resume_state not in (None, {}):
        validate_calibration_exact_resume_state(exact_resume_state)
    return validated


def validate_exact_calibration_resume_payload(payload: Mapping) -> Dict:
    """Validate the extra state required for a bit-exact calibration resume."""

    validated = validate_calibration_checkpoint_payload(payload)
    state = validated.get("exact_resume_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("calibration exact resume state is missing")
    validate_calibration_exact_resume_state(state)
    progress = validated.get("mission_progress")
    if not isinstance(progress, Mapping) or not progress:
        raise ValueError("calibration exact resume mission progress is missing")
    for field in (
        "environment_step_count",
        "completed_episode_count",
        "critic_update_count",
    ):
        try:
            checkpoint_value = int(validated[field])
            progress_value = int(progress[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "calibration exact resume progress {} is missing".format(field)
            ) from exc
        if checkpoint_value != progress_value:
            raise ValueError(
                "calibration exact resume progress {} mismatch".format(field)
            )
    records = state["raw_holdout_records"]
    record_episode_ids = {
        str(record.get("episode_id", ""))
        for record in records
        if str(record.get("episode_id", ""))
    }
    completed_holdout_ids = {
        str(value) for value in progress.get("holdout_completed_ids", [])
    }
    if not record_episode_ids.issubset(completed_holdout_ids):
        raise ValueError(
            "calibration exact resume holdout records exceed completed episodes"
        )
    result = dict(validated)
    result["resume_required_state_count"] = len(
        CALIBRATION_EXACT_RESUME_REQUIRED_STATES
    )
    result["resume_restored_state_count"] = len(
        CALIBRATION_EXACT_RESUME_REQUIRED_STATES
    )
    return result


def validate_calibration_pass_checkpoint_payload(payload: Mapping) -> Dict:
    """Validate the Phase-1 artifact that is allowed to enable the Actor."""

    validated = validate_calibration_checkpoint_payload(payload)
    try:
        require_calibration_pass(validated.get("calibration_gate_state", ""))
        validate_phase1_depth_learning_rates(
            actor_depth_lr=validated["phase1_actor_depth_lr"],
            critic_depth_lr=validated["phase1_critic_depth_lr"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, Phase1ReadinessError):
            raise
        raise ValueError("calibration pass checkpoint Phase-1 contract is invalid") from exc
    required = {
        "calibration_pass_checkpoint",
        "phase1_state",
        "phase1_training_contract",
        "phase1_training_contract_sha256",
        "phase1_actor_depth_lr",
        "phase1_critic_depth_lr",
        "calibration_actor_unchanged",
        "calibration_actor_update_count",
    }
    missing = sorted(required.difference(validated))
    if missing:
        raise ValueError(
            "calibration pass checkpoint missing fields: {}".format(", ".join(missing))
        )
    if validated["calibration_pass_checkpoint"] is not True:
        raise ValueError("calibration pass checkpoint marker is missing")
    if validated["phase1_state"] != PHASE_CRITIC_CALIBRATION:
        raise ValueError("calibration pass checkpoint phase1 state mismatch")
    if validated["calibration_actor_unchanged"] is not True:
        raise ValueError("calibration pass checkpoint Actor changed")
    if int(validated["calibration_actor_update_count"]) != 0:
        raise ValueError("calibration pass checkpoint Actor update count is non-zero")
    if int(validated.get("actor_update_count", 0)) != 0:
        raise ValueError("calibration pass checkpoint learner Actor count is non-zero")
    if int(validated.get("actor_optimizer_step_count", 0)) != 0:
        raise ValueError("calibration pass checkpoint Actor optimizer was used")
    if int(validated.get("critic_update_count", 0)) <= 0:
        raise ValueError("calibration pass checkpoint has no Critic updates")
    phase1_contract = validated["phase1_training_contract"]
    if not isinstance(phase1_contract, Mapping) or not phase1_contract:
        raise ValueError("calibration pass checkpoint Phase-1 contract is missing")
    if phase1_contract.get("actor_depth_lr") != validated["phase1_actor_depth_lr"]:
        raise ValueError("calibration pass checkpoint Actor depth LR mismatch")
    if phase1_contract.get("critic_depth_lr") != validated["phase1_critic_depth_lr"]:
        raise ValueError("calibration pass checkpoint Critic depth LR mismatch")
    if validated["phase1_training_contract_sha256"] != phase1_contract.get(
        "contract_sha256"
    ):
        raise ValueError("calibration pass checkpoint Phase-1 contract SHA mismatch")
    return validated


def validate_standard_awac_checkpoint_payload(payload: Mapping) -> Dict[str, Any]:
    """Validate a committed Standard AWAC online checkpoint payload."""

    validated = validate_awac_checkpoint_payload(payload)
    required = {
        "phase",
        "actor_update_enabled",
        "actor_optimizer_status",
        "actor_optimizer_step_count",
        "actor_update_count",
        "critic_update_count",
        "environment_step_count",
        "online_env_steps",
        "online_env_steps_budget",
        "online_transitions_committed",
        "online_transition_count",
        "starting_replay_size",
        "starting_replay_total_added",
        "replay_identity",
        "source_replay_identity",
        "source_calibration_checkpoint_sha256",
        "standard_online_runtime_schema_id",
        "standard_online_progress",
        "standard_online_resume_state",
        "phase1_state",
        "runtime_identity",
        "training_contract",
        "training_contract_sha256",
        "actor_state_sha256",
        "critic1_state_sha256",
        "critic2_state_sha256",
    }
    missing = sorted(required.difference(validated))
    if missing:
        raise ValueError(
            "standard AWAC checkpoint missing fields: {}".format(
                ", ".join(missing)
            )
        )
    if validated["phase"] != AWAC_PHASE_STANDARD_TRAINING:
        raise ValueError("standard AWAC checkpoint phase mismatch")
    if validated["actor_update_enabled"] is not True:
        raise ValueError("standard AWAC checkpoint Actor updates are disabled")
    if validated.get("actor_frozen_for_calibration") is not False:
        raise ValueError("standard AWAC checkpoint Actor remains frozen")
    if validated["actor_optimizer_status"] != "active":
        raise ValueError("standard AWAC checkpoint Actor optimizer status mismatch")

    counters = (
        "actor_optimizer_step_count",
        "actor_update_count",
        "critic_update_count",
        "environment_step_count",
        "online_env_steps",
        "online_transitions_committed",
        "online_transition_count",
        "starting_replay_size",
        "starting_replay_total_added",
    )
    for name in counters:
        if int(validated[name]) < 0:
            raise ValueError("standard AWAC checkpoint {} is invalid".format(name))
    if int(validated["online_env_steps"]) != int(
        validated["environment_step_count"]
    ):
        raise ValueError("standard AWAC online environment-step counters mismatch")
    if int(validated["online_env_steps_budget"]) <= 0:
        raise ValueError("standard AWAC online environment-step budget is invalid")
    if int(validated["online_env_steps"]) > int(
        validated["online_env_steps_budget"]
    ):
        raise ValueError("standard AWAC online environment steps exceed budget")
    if int(validated["online_transition_count"]) > int(
        validated["online_env_steps"]
    ):
        raise ValueError("standard AWAC online transitions exceed environment steps")
    if int(validated["online_transitions_committed"]) != int(
        validated["online_transition_count"]
    ):
        raise ValueError(
            "standard AWAC committed transition field mismatch"
        )
    if int(validated["actor_update_count"]) > int(
        validated["actor_optimizer_step_count"]
    ):
        raise ValueError("standard AWAC Actor counters are inconsistent")

    for name in ("replay_identity", "source_replay_identity", "phase1_state", "runtime_identity"):
        if not isinstance(validated[name], Mapping) or not validated[name]:
            raise ValueError("standard AWAC checkpoint {} is missing".format(name))
    source_sha = str(validated["source_calibration_checkpoint_sha256"])
    if len(source_sha) != 64:
        raise ValueError("standard AWAC source calibration checkpoint SHA is invalid")
    if validated["standard_online_runtime_schema_id"] != "awac_standard_online_runtime_v1":
        raise ValueError("standard AWAC runtime schema mismatch")
    if validated["phase1_state"].get("actor_update_enabled") is not True:
        raise ValueError("standard AWAC Phase-1 state does not enable Actor")

    contract = validated["training_contract"]
    if not isinstance(contract, Mapping) or not contract:
        raise ValueError("standard AWAC training contract is missing")
    if contract.get("phase") != AWAC_PHASE_STANDARD_TRAINING:
        raise ValueError("standard AWAC training contract phase mismatch")
    if contract.get("actor_update_enabled") is not True:
        raise ValueError("standard AWAC training contract disables Actor")
    if validated["training_contract_sha256"] != canonical_json_sha256(dict(contract)):
        raise ValueError("standard AWAC training contract SHA mismatch")
    if contract.get("replay_contract_sha256") != replay_contract_sha256():
        raise ValueError("standard AWAC replay contract mismatch")
    if contract.get("reward_contract_sha256") != reward_contract_sha256():
        raise ValueError("standard AWAC reward contract mismatch")
    if contract.get("observation_contract") != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("standard AWAC observation contract mismatch")
    if contract.get("observation_source") != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("standard AWAC observation source mismatch")

    resolved_config = validated.get("resolved_training_config")
    if not isinstance(resolved_config, Mapping) or not resolved_config:
        raise ValueError("standard AWAC resolved training config is missing")
    if validated["resolved_training_config_sha256"] != canonical_json_sha256(
        dict(resolved_config)
    ):
        raise ValueError("standard AWAC resolved training config SHA mismatch")
    if resolved_config.get("phase") != AWAC_PHASE_STANDARD_TRAINING:
        raise ValueError("standard AWAC resolved training config phase mismatch")
    for field, expected in (
        ("observation_contract", EXACT_ENDPOINT_OBSERVATION_CONTRACT),
        ("observation_source", EXACT_ENDPOINT_OBSERVATION_CONTRACT),
        ("task_contract_sha256", validated["task_contract_sha256"]),
        ("mpl_contract_sha256", validated["mpl_contract_sha256"]),
        ("reward_contract_sha256", reward_contract_sha256()),
        ("online_budget_counter_owner", "phase_local_online_environment_steps"),
        ("online_env_steps_budget", int(validated["online_env_steps_budget"])),
    ):
        if resolved_config.get(field) != expected:
            raise ValueError(
                "standard AWAC resolved config {} mismatch".format(field)
            )
    if validated["source_bc_checkpoint_sha256"] != resolved_config.get(
        "source_bc_checkpoint_sha256"
    ):
        raise ValueError("standard AWAC source BC checkpoint identity mismatch")

    replay_identity = validated["replay_identity"]
    source_replay_identity = validated["source_replay_identity"]
    # The exact-resume state is checked below; keeping the comparison here
    # makes a checkpoint with a separately edited replay identity fail closed.
    if not isinstance(replay_identity, Mapping) or not isinstance(
        source_replay_identity, Mapping
    ):
        raise ValueError("standard AWAC replay identities are invalid")
    if str(replay_identity.get("behavior_source_phase", "")) != "awac_training":
        raise ValueError("standard AWAC replay phase identity mismatch")
    if str(source_replay_identity.get("behavior_source_phase", "")) != (
        "critic_calibration"
    ):
        raise ValueError("standard AWAC source replay phase identity mismatch")
    if int(replay_identity.get("replay_size", -1)) != int(
        validated.get("replay_size", -2)
    ):
        raise ValueError("standard AWAC replay size identity mismatch")
    if int(replay_identity.get("replay_total_added", -1)) < int(
        validated["starting_replay_total_added"]
    ):
        raise ValueError("standard AWAC replay total-added identity regressed")

    resume_state = validated["standard_online_resume_state"]
    validate_standard_online_exact_resume_state(resume_state)
    if dict(resume_state["replay_identity"]) != dict(replay_identity):
        raise ValueError("standard AWAC replay identity/resume state mismatch")
    learner_state = _learner_state_from_checkpoint(validated)
    if resume_state["learner_state_sha256"] != _resume_state_sha256(learner_state):
        raise ValueError("standard AWAC learner state SHA mismatch")
    progress = validated["standard_online_progress"]
    if not isinstance(progress, Mapping) or not progress:
        raise ValueError("standard AWAC online progress is missing")
    if progress != resume_state["mission_progress"]:
        raise ValueError("standard AWAC online progress/resume state mismatch")
    for name in (
        "environment_step_count",
        "online_env_steps",
        "critic_update_count",
        "actor_update_count",
        "actor_optimizer_step_count",
    ):
        if int(progress.get(name, -1)) != int(validated[name]):
            raise ValueError("standard AWAC progress {} mismatch".format(name))
    if int(progress.get("online_transitions_committed", -1)) != int(
        validated["online_transition_count"]
    ):
        raise ValueError("standard AWAC progress online transition count mismatch")
    environment_counters = resume_state["environment_and_online_counters"]
    for name, expected in (
        ("environment_step_count", validated["environment_step_count"]),
        ("online_env_steps", validated["online_env_steps"]),
        (
            "online_transitions_committed",
            validated["online_transitions_committed"],
        ),
        ("critic_update_count", validated["critic_update_count"]),
        ("actor_update_count", validated["actor_update_count"]),
        ("actor_optimizer_step_count", validated["actor_optimizer_step_count"]),
        ("update_step", validated["update_step"]),
    ):
        if int(environment_counters.get(name, -1)) != int(expected):
            raise ValueError("standard AWAC exact environment counter {} mismatch".format(name))
    learner_counters = resume_state["learner_counters"]
    for name, expected in (
        ("update_step", validated["update_step"]),
        ("actor_update_count", validated["actor_update_count"]),
        ("actor_optimizer_step_count", validated["actor_optimizer_step_count"]),
        ("critic_update_count", validated["critic_update_count"]),
    ):
        if int(learner_counters.get(name, -1)) != int(expected):
            raise ValueError("standard AWAC exact learner counter {} mismatch".format(name))
    for name, state_name in (
        ("actor_state_sha256", "actor_state_dict"),
        ("critic1_state_sha256", "critic1_state_dict"),
        ("critic2_state_sha256", "critic2_state_dict"),
    ):
        if len(str(validated[name])) != 64:
            raise ValueError("standard AWAC {} is invalid".format(name))
        expected = actor_state_sha256({"actor_state_dict": validated[state_name]})
        if str(validated[name]) != expected:
            raise ValueError("standard AWAC {} does not match model state".format(name))
    return validated


def validate_standard_awac_handoff_payload(
    payload: Mapping,
    *,
    expected_bc_checkpoint_sha256: str,
    expected_bc_reference_fingerprint: str,
    expected_mpl_contract_sha256: str,
    expected_task_contract_sha256: str,
    expected_replay_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Authorize the sole Calibration PASS to Standard AWAC transition.

    This is a construction-time gate.  It restores no optimizer state and
    performs no learner update; the trainer must call the learner's explicit
    enable method only after this validator and the live replay identity both
    pass.
    """

    validated = validate_exact_calibration_resume_payload(
        validate_calibration_pass_checkpoint_payload(payload)
    )
    transaction = validated.get("checkpoint_transaction")
    if not isinstance(transaction, Mapping):
        raise Phase1ReadinessError(
            "Standard AWAC handoff requires a committed calibration transaction"
        )
    if transaction.get("transaction_state") != "COMMITTED":
        raise Phase1ReadinessError(
            "Standard AWAC handoff requires a committed calibration transaction"
        )
    if transaction.get("checkpoint_kind") != "checkpoint_calibration_pass":
        raise Phase1ReadinessError(
            "Standard AWAC handoff requires checkpoint_calibration_pass"
        )

    checks = {
        "source_bc_checkpoint_sha256": str(expected_bc_checkpoint_sha256),
        "bc_checkpoint_sha256": str(expected_bc_checkpoint_sha256),
        "bc_reference_fingerprint": str(expected_bc_reference_fingerprint),
        "mpl_contract_sha256": str(expected_mpl_contract_sha256),
        "task_contract_sha256": str(expected_task_contract_sha256),
        "observation_contract": "reliable_exact_endpoint_snapshot",
        "observation_source": "reliable_exact_endpoint_snapshot",
    }
    for field, expected in checks.items():
        if validated.get(field) != expected:
            raise Phase1ReadinessError(
                "Standard AWAC handoff {} mismatch: received={} expected={}".format(
                    field, validated.get(field), expected
                )
            )
    if validated.get("actor_state_sha256") != validated.get(
        "bc_reference_fingerprint"
    ):
        raise Phase1ReadinessError(
            "Standard AWAC handoff Actor state differs from BC reference"
        )
    if expected_replay_identity is not None:
        if not isinstance(expected_replay_identity, Mapping):
            raise TypeError("expected_replay_identity must be a mapping")
        received_replay_identity = validated.get("replay_identity")
        if not isinstance(received_replay_identity, Mapping):
            raise Phase1ReadinessError(
                "Standard AWAC handoff replay identity is missing"
            )
        for field, expected in expected_replay_identity.items():
            if received_replay_identity.get(field) != expected:
                raise Phase1ReadinessError(
                    "Standard AWAC handoff replay identity {} mismatch: received={} expected={}".format(
                        field, received_replay_identity.get(field), expected
                    )
                )

    phase1 = Phase1StateMachine()
    transition = phase1.transition_from_calibration(
        gate_state=validated["calibration_gate_state"],
        actor_depth_lr=float(validated["phase1_actor_depth_lr"]),
        critic_depth_lr=float(validated["phase1_critic_depth_lr"]),
        env_step=int(validated.get("environment_step_count", 0)),
        replay_size=int(validated.get("replay_size", 0)),
        critic_update_count=int(validated.get("critic_update_count", 0)),
        actor_update_count=int(validated.get("actor_update_count", 0)),
        source_checkpoint=str(transaction.get("checkpoint_filename", "")),
    )
    if not transition or phase1.phase != PHASE_ACTOR_ENABLED_STANDARD_AWAC:
        raise Phase1ReadinessError(
            "Standard AWAC handoff did not enter Actor-enabled phase"
        )
    return {
        "handoff_allowed": True,
        "from_phase": PHASE_CRITIC_CALIBRATION,
        "to_phase": PHASE_ACTOR_ENABLED_STANDARD_AWAC,
        "actor_update_enabled": True,
        "source_checkpoint": str(transaction["checkpoint_filename"]),
        "transition": transition,
        "replay_identity": dict(validated["replay_identity"]),
    }


def save_awac_checkpoint(path, payload: Mapping, *, torch) -> None:
    validated = validate_awac_checkpoint_payload(payload)
    save_torch_atomic(path, validated, torch=torch)


def load_awac_checkpoint(path, *, torch, map_location="cpu") -> Dict:
    return validate_awac_checkpoint_payload(load_torch(path, torch=torch, map_location=map_location))


def save_calibration_checkpoint(path, payload: Mapping, *, torch) -> None:
    validated = validate_calibration_checkpoint_payload(payload)
    _save_torch_durable_atomic(path, validated, torch=torch)


def save_calibration_pass_checkpoint(path, payload: Mapping, *, torch) -> None:
    """Atomically save only a validated calibration-gate PASS artifact."""

    validated = validate_calibration_pass_checkpoint_payload(payload)
    _save_torch_durable_atomic(path, validated, torch=torch)


def load_calibration_checkpoint(path, *, torch, map_location="cpu") -> Dict:
    payload = load_torch(path, torch=torch, map_location=map_location)
    return validate_calibration_checkpoint_payload(payload)


def load_calibration_pass_checkpoint(path, *, torch, map_location="cpu") -> Dict:
    """Load the pass artifact without performing or scheduling an update."""

    payload = load_torch(path, torch=torch, map_location=map_location)
    return validate_calibration_pass_checkpoint_payload(payload)


def calibration_checkpoint_transaction_manifest_path(
    output_dir: Path, checkpoint_name: str
) -> Path:
    """Return the sole commit marker for one named calibration checkpoint."""

    name = str(checkpoint_name).strip()
    if not name or "/" in name or "\\" in name:
        raise ValueError("calibration checkpoint name is invalid")
    return (
        Path(output_dir).expanduser().resolve()
        / "{}.transaction.json".format(name)
    )


def calibration_checkpoint_transaction_diagnostics_path(
    output_dir: Path, checkpoint_name: str
) -> Path:
    """Return the non-authoritative diagnostic record for one checkpoint.

    This sidecar is never a resume marker.  It remains after an interrupted
    save so a partial generation is distinguishable from the last complete
    generation selected by ``*.transaction.json``.
    """

    name = str(checkpoint_name).strip()
    if not name or "/" in name or "\\" in name:
        raise ValueError("calibration checkpoint name is invalid")
    return (
        Path(output_dir).expanduser().resolve()
        / "{}.transaction_attempt.json".format(name)
    )


def _persist_transaction_diagnostics(
    path: Path,
    payload: Mapping[str, Any],
    *,
    best_effort: bool,
) -> None:
    """Persist a diagnostic sidecar without making it a commit marker."""

    try:
        write_calibration_transaction_json(path, payload)
    except BaseException:
        if not best_effort:
            raise


def _load_transaction_manifest(path: Path) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            "calibration checkpoint transaction manifest is missing: {}".format(
                source
            )
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("calibration checkpoint transaction manifest is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("calibration checkpoint transaction manifest is invalid")
    required = (
        "schema_id",
        "generation",
        "checkpoint_name",
        "checkpoint_filename",
        "checkpoint_sha256",
        "checkpoint_kind",
        "replay_committed_state",
        "transaction_state",
        "checkpoint_final_commit",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(
            "calibration checkpoint transaction manifest missing: {}".format(
                ", ".join(missing)
            )
        )
    if payload["schema_id"] != CALIBRATION_CHECKPOINT_TRANSACTION_SCHEMA_ID:
        raise ValueError("calibration checkpoint transaction manifest schema mismatch")
    if int(payload["generation"]) <= 0:
        raise ValueError("calibration checkpoint transaction generation is invalid")
    if payload["transaction_state"] != "COMMITTED":
        raise ValueError("calibration checkpoint transaction is not committed")
    if payload["checkpoint_final_commit"] != "PASS":
        raise ValueError("calibration checkpoint transaction final commit failed")
    if not isinstance(payload["replay_committed_state"], Mapping):
        raise ValueError("calibration checkpoint replay committed state is invalid")
    return dict(payload)


def _replay_committed_identity(replay) -> Dict[str, Any]:
    metadata_path = (
        Path(replay.directory).expanduser().resolve() / "metadata.json"
    )
    metadata = dict(replay.metadata)
    if not metadata_path.is_file():
        raise FileNotFoundError("calibration replay metadata is missing")
    return {
        "run_identity": str(metadata.get("run_identity", "")),
        "mission_source_sha256": str(metadata.get("mission_source_sha256", "")),
        "mission_index_sha256": str(metadata.get("mission_index_sha256", "")),
        "replay_size": int(getattr(replay, "size", 0)),
        "replay_position": int(getattr(replay, "position", 0)),
        "replay_total_added": int(getattr(replay, "total_added", 0)),
        "replay_metadata_sha256": file_sha256(metadata_path),
        "replay_contract_sha256": str(
            metadata.get("replay_contract_sha256", "")
        ),
        "behavior_source_phase": str(
            metadata.get("behavior_source_phase", "")
        ),
    }


def calibration_replay_identity(replay) -> Dict[str, Any]:
    """Return the durable replay identity used by handoff validation."""

    return _replay_committed_identity(replay)


def _validate_payload_replay_identity(
    payload: Mapping[str, Any], replay_identity: Mapping[str, Any]
) -> None:
    received = payload.get("replay_identity")
    if not isinstance(received, Mapping):
        raise ValueError("calibration checkpoint replay identity is missing")
    for field in (
        "run_identity",
        "mission_source_sha256",
        "mission_index_sha256",
        "replay_size",
        "replay_position",
        "replay_total_added",
        "replay_metadata_sha256",
        "replay_contract_sha256",
        "behavior_source_phase",
    ):
        if received.get(field) != replay_identity.get(field):
            raise ValueError(
                "calibration checkpoint replay identity {} mismatch".format(field)
            )


def _publish_checkpoint_alias(
    *,
    artifact: Path,
    alias: Path,
) -> None:
    """Update the conventional checkpoint name only after manifest commit."""

    destination = Path(alias).expanduser().resolve()
    temporary = _temporary_path(destination)
    try:
        os.link(str(artifact), str(temporary))
        _fsync_directory(destination.parent)
        os.replace(str(temporary), str(destination))
        _fsync_directory(destination.parent)
    except BaseException:
        # The immutable generation and commit marker remain authoritative.
        # A stale or absent convenience alias is never accepted for resume.
        raise


def commit_calibration_checkpoint_transaction(
    *,
    output_dir: Path,
    checkpoint_name: str,
    replay,
    build_payload: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    torch,
    save_reason: str,
    checkpoint_kind: str = "checkpoint_last",
    failpoint: Optional[Callable[[str], None]] = None,
    recovery_diagnostics: Optional[Mapping[str, Any]] = None,
    payload_validator: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
    checkpoint_loader: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Commit one replay-bound checkpoint generation with a final marker.

    The replay owner supplies durable metadata and a bounded rollback window.
    The checkpoint generation is immutable; the transaction manifest is the
    only resume authority and is written last.
    """

    payload_validator = payload_validator or validate_exact_calibration_resume_payload
    checkpoint_loader = checkpoint_loader or load_calibration_checkpoint
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = calibration_checkpoint_transaction_manifest_path(
        root, checkpoint_name
    )
    diagnostics_path = calibration_checkpoint_transaction_diagnostics_path(
        root, checkpoint_name
    )
    previous_generation = 0
    if manifest_path.exists():
        previous_generation = int(_load_transaction_manifest(manifest_path)["generation"])
    generation = previous_generation + 1
    if recovery_diagnostics is not None and not isinstance(
        recovery_diagnostics, Mapping
    ):
        raise TypeError("calibration recovery diagnostics must be a mapping")
    diagnostics = {
        "schema_id": CALIBRATION_CHECKPOINT_TRANSACTION_DIAGNOSTICS_SCHEMA_ID,
        "checkpoint_name": str(checkpoint_name),
        "checkpoint_kind": str(checkpoint_kind),
        "checkpoint_generation": int(generation),
        "checkpoint_save_reason": str(save_reason),
        "transaction_state": "PENDING",
        "replay_final_flush": "NOT_ATTEMPTED",
        "checkpoint_final_commit": "NOT_ATTEMPTED",
        "checkpoint_alias_status": "NOT_ATTEMPTED",
        "interrupted_save_traceback": "",
        "runtime_failure_summary": "",
        **dict(recovery_diagnostics or {}),
    }
    _persist_transaction_diagnostics(
        diagnostics_path, diagnostics, best_effort=False
    )

    committed = False
    artifact = root / "{}.generation-{:08d}.pt".format(
        checkpoint_name, generation
    )
    try:
        if failpoint is not None:
            failpoint("before_replay_flush")
        flush = getattr(replay, "flush", None)
        if not callable(flush):
            raise TypeError("calibration transaction replay has no flush")
        flush()
        diagnostics["replay_final_flush"] = "PASS"
        diagnostics["transaction_state"] = "REPLAY_FLUSHED"
        _persist_transaction_diagnostics(
            diagnostics_path, diagnostics, best_effort=False
        )
        if failpoint is not None:
            failpoint("after_replay_flush")
        replay_identity = _replay_committed_identity(replay)
        replay_state = {
            "metadata": dict(replay.metadata),
            "identity": dict(replay_identity),
        }
        transaction = {
            "schema_id": CALIBRATION_CHECKPOINT_TRANSACTION_SCHEMA_ID,
            "generation": int(generation),
            "checkpoint_name": str(checkpoint_name),
            "checkpoint_kind": str(checkpoint_kind),
            "checkpoint_filename": artifact.name,
            "replay_committed_state": replay_state,
            "transaction_state": "COMMITTED",
            "checkpoint_save_reason": str(save_reason),
        }
        payload = dict(build_payload(dict(replay_identity)))
        _validate_payload_replay_identity(payload, replay_identity)
        payload["checkpoint_transaction"] = dict(transaction)
        payload["checkpoint_recovery_diagnostics"] = {
            **dict(payload.get("checkpoint_recovery_diagnostics", {})),
            **dict(recovery_diagnostics or {}),
            "checkpoint_save_reason": str(save_reason),
            "checkpoint_generation": int(generation),
            "replay_final_flush": "PASS",
            "checkpoint_final_commit": "COMMIT_MARKER_REQUIRED",
        }
        payload_validator(payload)

        _save_torch_durable_atomic(
            artifact, payload, torch=torch, failpoint=failpoint
        )
        # A complete file is still uncommitted until the manifest below exists.
        validated = checkpoint_loader(
            artifact, torch=torch, map_location="cpu"
        )
        payload_validator(validated)
        if failpoint is not None:
            failpoint("before_transaction_manifest_commit")
        manifest = {
            **transaction,
            "checkpoint_sha256": file_sha256(artifact),
            "checkpoint_final_commit": "PASS",
        }
        write_calibration_transaction_json(manifest_path, manifest)
        committed = True
        diagnostics.update(
            {
                "transaction_state": "COMMITTED",
                "checkpoint_final_commit": "PASS",
                "checkpoint_filename": artifact.name,
                "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
            }
        )
        _persist_transaction_diagnostics(
            diagnostics_path, diagnostics, best_effort=False
        )
        if failpoint is not None:
            failpoint("after_transaction_manifest_commit")

        activate = getattr(replay, "activate_calibration_checkpoint_generation", None)
        if not callable(activate):
            raise TypeError("calibration transaction replay has no generation owner")
        activate(
            generation=int(generation),
            committed_metadata=dict(replay.metadata),
        )
        diagnostics["replay_generation_activation"] = "PASS"
        try:
            _publish_checkpoint_alias(
                artifact=artifact,
                alias=root / "{}.pt".format(checkpoint_name),
            )
            diagnostics["checkpoint_alias_status"] = "PASS"
        except BaseException as alias_error:
            # The immutable generation and manifest remain valid.  The alias
            # is a convenience path only and can be reconstructed safely.
            diagnostics["checkpoint_alias_status"] = "FAIL"
            diagnostics["checkpoint_alias_failure"] = "{}: {}".format(
                type(alias_error).__name__, str(alias_error)
            )
        if str(checkpoint_name) == "checkpoint_last":
            # The transaction marker and validated artifact are authoritative.
            # Retention is deliberately best-effort after commit: inability to
            # reclaim old generations must never turn a valid checkpoint into
            # a failed transaction.
            try:
                retention = garbage_collect_checkpoint_generations(
                    root,
                    checkpoint_name="checkpoint_last",
                )
                diagnostics["checkpoint_retention_status"] = str(
                    retention.get("status", "WARNING")
                )
                diagnostics["checkpoint_retention"] = dict(retention)
            except BaseException as retention_error:
                diagnostics["checkpoint_retention_status"] = "WARNING"
                diagnostics["checkpoint_retention_warning"] = "{}: {}".format(
                    type(retention_error).__name__, str(retention_error)
                )
        else:
            diagnostics["checkpoint_retention_status"] = "NOT_APPLICABLE_PINNED"
        _persist_transaction_diagnostics(
            diagnostics_path, diagnostics, best_effort=True
        )
        return {
            "generation": int(generation),
            "checkpoint_path": str(artifact),
            "checkpoint_alias": str(root / "{}.pt".format(checkpoint_name)),
            "manifest_path": str(manifest_path),
            "diagnostics_path": str(diagnostics_path),
            "replay_identity": dict(replay_identity),
        }
    except BaseException as error:
        diagnostics.update(
            {
                "transaction_state": (
                    "COMMITTED" if committed else "FAILED_INCOMPLETE"
                ),
                "checkpoint_final_commit": "PASS" if committed else "FAIL",
                "interrupted_save_traceback": traceback.format_exc(),
                "runtime_failure_summary": "{}: {}".format(
                    type(error).__name__, str(error)
                ),
            }
        )
        if committed:
            diagnostics["post_commit_failure"] = diagnostics[
                "runtime_failure_summary"
            ]
        _persist_transaction_diagnostics(
            diagnostics_path, diagnostics, best_effort=True
        )
        raise


def validate_standard_online_replay_counter(
    payload: Mapping[str, Any], replay: Any
) -> Dict[str, int]:
    """Cross-check the canonical Standard counter against committed replay rows.

    ``online_transitions_committed`` remains the phase counter owner.  Replay
    storage is an independent checkpoint-boundary audit: it may reject a
    payload, but it never reconstructs or overwrites the counter.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("standard online checkpoint payload must be a mapping")
    required = (
        "online_transitions_committed",
        "online_transition_count",
        "starting_replay_total_added",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(
            "standard online replay cross-check missing: {}".format(
                ", ".join(missing)
            )
        )
    committed = int(payload["online_transitions_committed"])
    legacy_alias = int(payload["online_transition_count"])
    starting_total = int(payload["starting_replay_total_added"])
    if committed < 0 or legacy_alias < 0 or starting_total < 0:
        raise ValueError("standard online replay cross-check counters are invalid")
    if committed != legacy_alias:
        raise ValueError(
            "standard online replay AWAC_ONLINE counter fields mismatch"
        )

    arrays = getattr(replay, "arrays", None)
    if not isinstance(arrays, Mapping) or "behavior_source" not in arrays:
        raise ValueError(
            "standard online replay AWAC_ONLINE rows are unavailable"
        )
    size = int(getattr(replay, "size", -1))
    capacity = int(getattr(replay, "capacity", size))
    total_added = int(getattr(replay, "total_added", size))
    if size < 0 or capacity <= 0 or size > capacity or total_added < 0:
        raise ValueError("standard online replay storage counters are invalid")
    values = np.asarray(arrays["behavior_source"][:size], dtype=np.int64).reshape(-1)
    if values.size != size:
        raise ValueError("standard online replay behavior-source rows are truncated")
    from planning.awac.interaction import BehaviorSource

    stored_online_rows = int(
        np.count_nonzero(values == int(BehaviorSource.AWAC_ONLINE))
    )
    if total_added < starting_total:
        raise ValueError("standard online replay total-added counter regressed")
    appended_rows = int(total_added - starting_total)
    if appended_rows != committed:
        raise ValueError(
            "standard online replay AWAC_ONLINE count mismatch: "
            "committed={} replay_appended={}".format(committed, appended_rows)
        )

    # Before ring wrap, the persisted source column is a complete row count.
    # After wrap, total_added-starting_total is the only complete replay audit
    # available; the canonical phase counter is still never re-derived.
    if total_added <= capacity and stored_online_rows != committed:
        raise ValueError(
            "standard online replay AWAC_ONLINE row count mismatch: "
            "committed={} stored={}".format(committed, stored_online_rows)
        )
    if total_added > capacity and stored_online_rows > committed:
        raise ValueError(
            "standard online replay AWAC_ONLINE stored rows exceed committed count"
        )
    return {
        "online_transitions_committed": committed,
        "stored_awac_online_rows": stored_online_rows,
        "replay_appended_rows": appended_rows,
    }


def commit_standard_awac_checkpoint_transaction(
    *,
    output_dir: Path,
    checkpoint_name: str,
    replay,
    build_payload: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    torch,
    save_reason: str,
    checkpoint_kind: str = "checkpoint_last",
    failpoint: Optional[Callable[[str], None]] = None,
    recovery_diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Commit a Standard AWAC checkpoint using the proven transaction owner."""

    def build_and_crosscheck(replay_identity: Mapping[str, Any]):
        payload = dict(build_payload(dict(replay_identity)))
        validate_standard_online_replay_counter(payload, replay)
        return payload

    return commit_calibration_checkpoint_transaction(
        output_dir=output_dir,
        checkpoint_name=checkpoint_name,
        replay=replay,
        build_payload=build_and_crosscheck,
        torch=torch,
        save_reason=save_reason,
        checkpoint_kind=checkpoint_kind,
        failpoint=failpoint,
        recovery_diagnostics=recovery_diagnostics,
        payload_validator=validate_standard_awac_checkpoint_payload,
        checkpoint_loader=load_awac_checkpoint,
    )


def _committed_checkpoint_manifest_for_path(path: Path) -> Dict[str, Any]:
    requested = Path(path).expanduser().resolve()
    if requested.suffix != ".pt":
        raise ValueError("calibration resume checkpoint must be a canonical .pt path")
    return _load_transaction_manifest(
        calibration_checkpoint_transaction_manifest_path(
            requested.parent, requested.stem
        )
    )


def committed_checkpoint_kind(path: Path) -> str:
    """Return the committed checkpoint kind without loading model tensors."""

    return str(_committed_checkpoint_manifest_for_path(path)["checkpoint_kind"])


def load_committed_calibration_resume_checkpoint(
    path: Path, *, torch, map_location="cpu"
) -> Dict[str, Any]:
    """Load only the checkpoint selected by a committed generation marker."""

    manifest = _committed_checkpoint_manifest_for_path(path)
    artifact = (
        Path(path).expanduser().resolve().parent
        / str(manifest["checkpoint_filename"])
    )
    if not artifact.is_file():
        raise ValueError("committed calibration checkpoint artifact is missing")
    if file_sha256(artifact) != str(manifest["checkpoint_sha256"]):
        raise ValueError("committed calibration checkpoint SHA mismatch")
    if str(manifest["checkpoint_kind"]) == "checkpoint_calibration_pass":
        payload = load_calibration_pass_checkpoint(
            artifact, torch=torch, map_location=map_location
        )
    else:
        payload = load_calibration_checkpoint(
            artifact, torch=torch, map_location=map_location
        )
    exact = validate_exact_calibration_resume_payload(payload)
    transaction = exact.get("checkpoint_transaction")
    if not isinstance(transaction, Mapping):
        raise ValueError("committed calibration checkpoint transaction is missing")
    for field in (
        "schema_id",
        "generation",
        "checkpoint_name",
        "checkpoint_kind",
        "checkpoint_filename",
        "transaction_state",
    ):
        if transaction.get(field) != manifest.get(field):
            raise ValueError(
                "committed calibration checkpoint transaction {} mismatch".format(
                    field
                )
            )
    if transaction.get("replay_committed_state") != manifest.get(
        "replay_committed_state"
    ):
        raise ValueError(
            "committed calibration checkpoint replay committed state mismatch"
        )
    return exact


def load_committed_standard_resume_checkpoint(
    path: Path, *, torch, map_location="cpu"
) -> Dict[str, Any]:
    """Load only a Standard AWAC checkpoint selected by a commit marker."""

    manifest = _committed_checkpoint_manifest_for_path(path)
    artifact = (
        Path(path).expanduser().resolve().parent
        / str(manifest["checkpoint_filename"])
    )
    if not artifact.is_file():
        raise ValueError("committed Standard AWAC checkpoint artifact is missing")
    if file_sha256(artifact) != str(manifest["checkpoint_sha256"]):
        raise ValueError("committed Standard AWAC checkpoint SHA mismatch")
    payload = load_awac_checkpoint(artifact, torch=torch, map_location=map_location)
    validated = validate_standard_awac_checkpoint_payload(payload)
    transaction = validated.get("checkpoint_transaction")
    if not isinstance(transaction, Mapping):
        raise ValueError("committed Standard AWAC checkpoint transaction is missing")
    for field in (
        "schema_id",
        "generation",
        "checkpoint_name",
        "checkpoint_kind",
        "checkpoint_filename",
        "transaction_state",
    ):
        if transaction.get(field) != manifest.get(field):
            raise ValueError(
                "committed Standard AWAC checkpoint transaction {} mismatch".format(
                    field
                )
            )
    if transaction.get("replay_committed_state") != manifest.get(
        "replay_committed_state"
    ):
        raise ValueError(
            "committed Standard AWAC checkpoint replay committed state mismatch"
        )
    return validated


def recover_calibration_replay_from_checkpoint_transaction(
    replay, checkpoint_path: Path
) -> Dict[str, Any]:
    """Restore replay to the last committed generation before exact resume."""

    manifest = _committed_checkpoint_manifest_for_path(checkpoint_path)
    restore = getattr(replay, "restore_calibration_checkpoint_generation", None)
    if not callable(restore):
        raise TypeError("calibration transaction replay has no recovery owner")
    restore(
        generation=int(manifest["generation"]),
        committed_metadata=dict(manifest["replay_committed_state"]["metadata"]),
        committed_metadata_sha256=str(
            manifest["replay_committed_state"]["identity"][
                "replay_metadata_sha256"
            ]
        ),
    )
    return dict(manifest)


__all__ = [
    "AWAC_ALGORITHM_ID",
    "AWAC_CHECKPOINT_CONTRACT_ID",
    "AWAC_CHECKPOINT_SCHEMA_ID",
    "AWAC_MODEL_TYPE",
    "AWAC_TRAINING_CONFIG_CONTRACT_ID",
    "CALIBRATION_CHECKPOINT_TRANSACTION_DIAGNOSTICS_SCHEMA_ID",
    "CALIBRATION_CHECKPOINT_TRANSACTION_SCHEMA_ID",
    "CALIBRATION_EXACT_RESUME_REQUIRED_STATES",
    "CALIBRATION_EXACT_RESUME_STATE_SCHEMA_ID",
    "CALIBRATION_REPLAY_PENDING_GENERATION_SCHEMA_ID",
    "STANDARD_ONLINE_EXACT_RESUME_REQUIRED_STATES",
    "STANDARD_ONLINE_EXACT_RESUME_STATE_SCHEMA_ID",
    "STANDARD_ONLINE_RUNTIME_SCHEMA_ID",
    "build_awac_checkpoint_identity",
    "build_calibration_exact_resume_state",
    "build_standard_online_exact_resume_state",
    "calibration_replay_identity",
    "calibration_checkpoint_transaction_diagnostics_path",
    "calibration_checkpoint_transaction_manifest_path",
    "calibration_holdout_records_sha256",
    "commit_calibration_checkpoint_transaction",
    "commit_standard_awac_checkpoint_transaction",
    "committed_checkpoint_kind",
    "load_awac_checkpoint",
    "load_calibration_checkpoint",
    "load_calibration_pass_checkpoint",
    "load_committed_calibration_resume_checkpoint",
    "load_committed_standard_resume_checkpoint",
    "recover_calibration_replay_from_checkpoint_transaction",
    "remove_calibration_transaction_file",
    "restore_calibration_exact_resume_state",
    "restore_standard_online_exact_resume_state",
    "save_calibration_checkpoint",
    "save_calibration_pass_checkpoint",
    "save_awac_checkpoint",
    "validate_calibration_exact_resume_state",
    "validate_standard_online_exact_resume_state",
    "validate_standard_online_replay_counter",
    "validate_exact_calibration_resume_payload",
    "validate_calibration_checkpoint_payload",
    "validate_calibration_pass_checkpoint_payload",
    "validate_standard_awac_handoff_payload",
    "validate_standard_awac_checkpoint_payload",
    "validate_awac_checkpoint_payload",
    "write_calibration_transaction_json",
]
