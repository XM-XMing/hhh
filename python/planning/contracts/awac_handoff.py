"""Fail-closed provenance seam between BC artifacts and AWAC.

This module owns identity validation and artifact projections only.  It does
not construct a learner or contain update math.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from planning.contracts.feature import (
    CONTINUOUS_DIM,
    FEATURE_CONTRACT_ID,
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    policy_input_contract_sha256,
)
from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    observation_provenance,
)
from planning.contracts.policy_runtime import POLICY_RUNTIME_CONTRACT_ID
from planning.contracts.task import TASK_CONTRACT_ID, task_contract_sha256


AWAC_HANDOFF_CONTRACT_ID = "awac_bc_to_rl_handoff_provenance_v1"
ACTION_ORDERING = "motion_primitive_index_ascending"
EXPECTED_OBSERVATION_CONTRACT = EXACT_ENDPOINT_OBSERVATION_CONTRACT
EXPECTED_OBSERVATION_SOURCE = EXACT_ENDPOINT_OBSERVATION_CONTRACT


def _same(value: Any, expected: Any, *, field: str, path: str) -> None:
    if value != expected:
        raise ValueError(
            "{} {} mismatch: received={} expected={}".format(
                path, field, value, expected
            )
        )


def _required_mapping(value: Any, *, field: str, path: str) -> Mapping:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("{} {} is missing or not a mapping".format(path, field))
    return value


def _array_field(
    checkpoint: Mapping[str, Any], *, field: str, shape: tuple[int, ...]
) -> np.ndarray:
    if field not in checkpoint:
        raise ValueError("BC checkpoint missing {}".format(field))
    try:
        value = np.asarray(checkpoint[field], dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("BC checkpoint {} is not numeric".format(field)) from error
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            "BC checkpoint {} shape {} != {}".format(field, value.shape, shape)
        )
    if not np.isfinite(value).all():
        raise ValueError("BC checkpoint {} contains non-finite values".format(field))
    return value


def validate_awac_bc_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_mpl_contract_sha256: str,
    expected_observation_contract: str = EXPECTED_OBSERVATION_CONTRACT,
    expected_observation_source: str = EXPECTED_OBSERVATION_SOURCE,
    expected_feature_contract_id: str = FEATURE_CONTRACT_ID,
    expected_task_contract_id: str = TASK_CONTRACT_ID,
    expected_task_contract_sha256: str = task_contract_sha256(),
    expected_action_ordering: str = ACTION_ORDERING,
    path: str = "BC checkpoint",
) -> Dict[str, Any]:
    """Validate the complete formal BC input boundary for an AWAC run."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("{} must be a mapping".format(path))
    provenance = observation_provenance(checkpoint, path=path)
    _same(
        provenance["observation_contract"],
        str(expected_observation_contract),
        field="observation contract",
        path=path,
    )
    _same(
        provenance["observation_source"],
        str(expected_observation_source),
        field="observation source",
        path=path,
    )
    _required_mapping(checkpoint.get("model_state_dict"), field="model_state_dict", path=path)

    _same(checkpoint.get("feature_contract_id", ""), expected_feature_contract_id,
          field="feature contract", path=path)
    _same(
        checkpoint.get("policy_input_contract_sha256", ""),
        policy_input_contract_sha256(),
        field="policy input contract hash",
        path=path,
    )
    _same(checkpoint.get("task_contract_id", ""), expected_task_contract_id,
          field="task contract", path=path)
    _same(checkpoint.get("task_contract_sha256", ""), expected_task_contract_sha256,
          field="task contract hash", path=path)
    _same(
        checkpoint.get("mpl_contract_sha256", ""),
        str(expected_mpl_contract_sha256),
        field="motion primitive contract hash",
        path=path,
    )
    _same(checkpoint.get("vec_dim", -1), POLICY_VECTOR_DIM,
          field="vector dimension", path=path)
    _same(checkpoint.get("num_actions", -1), NUM_ACTIONS,
          field="action count", path=path)
    depth_history_frames = int(checkpoint.get("depth_history_frames", 0))
    if depth_history_frames <= 0:
        raise ValueError("{} depth history frames are missing or invalid".format(path))
    _same(checkpoint.get("initial_prev_action", None), INITIAL_PREV_ACTION,
          field="initial previous action", path=path)

    # These fields were not present in every historical BC artifact.  When
    # absent, their values are derived from the immutable feature contract;
    # when present, a conflicting declaration is rejected.
    _same(
        checkpoint.get("state_feature_dim", CONTINUOUS_DIM),
        CONTINUOUS_DIM,
        field="state feature dimension",
        path=path,
    )
    _same(
        checkpoint.get("previous_action_dim", NUM_ACTIONS),
        NUM_ACTIONS,
        field="previous-action dimension",
        path=path,
    )
    _same(
        checkpoint.get("action_ordering", ACTION_ORDERING),
        expected_action_ordering,
        field="action ordering",
        path=path,
    )
    _array_field(checkpoint, field="feature_mean", shape=(CONTINUOUS_DIM,))
    _array_field(checkpoint, field="feature_std", shape=(CONTINUOUS_DIM,))
    if str(checkpoint.get("policy_runtime_contract_id", "")) != POLICY_RUNTIME_CONTRACT_ID:
        raise ValueError("{} policy runtime contract mismatch".format(path))
    if "reliable_execution" in checkpoint and checkpoint["reliable_execution"] is not True:
        raise ValueError("{} reliable execution must be enabled".format(path))
    if "legacy_rows" in checkpoint and int(checkpoint["legacy_rows"]) != 0:
        raise ValueError("{} contains legacy rows".format(path))

    return {
        **provenance,
        "feature_contract_id": str(checkpoint["feature_contract_id"]),
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "task_contract_id": str(checkpoint["task_contract_id"]),
        "task_contract_sha256": str(checkpoint["task_contract_sha256"]),
        "mpl_contract_sha256": str(checkpoint["mpl_contract_sha256"]),
        "vec_dim": POLICY_VECTOR_DIM,
        "state_feature_dim": CONTINUOUS_DIM,
        "previous_action_dim": NUM_ACTIONS,
        "action_ordering": str(
            checkpoint.get("action_ordering", expected_action_ordering)
        ),
        "num_actions": NUM_ACTIONS,
        "depth_history_frames": depth_history_frames,
        "initial_prev_action": INITIAL_PREV_ACTION,
    }


def validate_awac_runtime_provenance(
    *,
    observation_contract: str,
    observation_source: str,
    reliable_execution_enabled: bool,
    expected_observation_contract: str = EXPECTED_OBSERVATION_CONTRACT,
    expected_observation_source: str = EXPECTED_OBSERVATION_SOURCE,
) -> Dict[str, Any]:
    """Validate the runtime side of the exact endpoint observation seam."""

    if str(observation_contract) != str(expected_observation_contract) or str(
        observation_source
    ) != str(expected_observation_source):
        raise ValueError(
            "runtime observation contract/source mismatch: received={}/{} expected={}/{}".format(
                observation_contract,
                observation_source,
                expected_observation_contract,
                expected_observation_source,
            )
        )
    if not bool(reliable_execution_enabled):
        raise ValueError("runtime observation requires reliable exact endpoint execution")
    return {
        "observation_contract": str(observation_contract),
        "observation_source": str(observation_source),
        "reliable_execution_enabled": True,
    }


def _expected_replay_fields(expected: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "bc_checkpoint_sha256": str(expected.get("bc_checkpoint_sha256", "")),
        "task_contract_id": str(expected.get("task_contract_id", TASK_CONTRACT_ID)),
        "task_contract_sha256": str(
            expected.get("task_contract_sha256", task_contract_sha256())
        ),
        "mpl_contract_sha256": str(expected.get("mpl_contract_sha256", "")),
        "run_identity": str(expected.get("run_identity", "")),
        "run_contract_sha256": str(expected.get("run_contract_sha256", "")),
    }


def validate_awac_replay_provenance(
    metadata: Mapping[str, Any], *, expected: Mapping[str, Any], path: str = "AWAC replay"
) -> Dict[str, Any]:
    """Validate replay metadata before it can be sampled by AWAC."""

    if not isinstance(metadata, Mapping):
        raise TypeError("{} metadata must be a mapping".format(path))
    provenance = observation_provenance(metadata, path=path)
    _same(
        provenance["observation_contract"],
        EXPECTED_OBSERVATION_CONTRACT,
        field="observation contract",
        path=path,
    )
    _same(
        provenance["observation_source"],
        EXPECTED_OBSERVATION_SOURCE,
        field="observation source",
        path=path,
    )
    if "observation_semantics" in metadata:
        _same(
            metadata.get("observation_semantics"),
            EXPECTED_OBSERVATION_CONTRACT,
            field="observation semantics",
            path=path,
        )
    expected_fields = _expected_replay_fields(expected)
    for field, value in expected_fields.items():
        if not value:
            raise ValueError("{} expected {} is missing".format(path, field))
        if field not in metadata:
            raise ValueError("{} missing {}".format(path, field))
        _same(metadata.get(field), value, field=field, path=path)
    legacy = int(metadata.get("legacy_replay_transition_count", -1))
    reliable = int(metadata.get("reliable_v4_transition_count", -1))
    if legacy < 0 or reliable < 0:
        raise ValueError("{} replay provenance counts are missing or invalid".format(path))
    if legacy != 0:
        raise ValueError("{} contains legacy transitions".format(path))
    return {
        **provenance,
        **expected_fields,
        "reliable_transition_count": reliable,
        "legacy_transition_count": legacy,
    }


def validate_awac_resume_provenance(
    resume: Mapping[str, Any],
    replay_metadata: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reject resume artifacts that cross BC, runtime, or replay identities."""

    if not isinstance(resume, Mapping):
        raise TypeError("AWAC resume checkpoint must be a mapping")
    expected_fields = {
        **_expected_replay_fields(expected),
        "bc_observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        "bc_observation_source": EXPECTED_OBSERVATION_SOURCE,
        "runtime_observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        "runtime_observation_source": EXPECTED_OBSERVATION_SOURCE,
        "expected_observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        "feature_contract_id": str(expected.get("feature_contract_id", FEATURE_CONTRACT_ID)),
        "policy_input_contract_sha256": str(
            expected.get("policy_input_contract_sha256", policy_input_contract_sha256())
        ),
        "task_contract_id": str(expected.get("task_contract_id", TASK_CONTRACT_ID)),
        "task_contract_sha256": str(
            expected.get("task_contract_sha256", task_contract_sha256())
        ),
        "mpl_contract_sha256": str(expected.get("mpl_contract_sha256", "")),
        "run_identity": str(expected.get("run_identity", "")),
        "run_contract_sha256": str(expected.get("run_contract_sha256", "")),
    }
    resume_bc_sha = str(
        resume.get("bc_checkpoint_sha256", resume.get("source_bc_checkpoint_sha256", ""))
    )
    if resume_bc_sha != expected_fields["bc_checkpoint_sha256"]:
        raise ValueError("resume BC checkpoint hash mismatch")
    if "source_bc_checkpoint_sha256" in resume and str(
        resume["source_bc_checkpoint_sha256"]
    ) != expected_fields["bc_checkpoint_sha256"]:
        raise ValueError("resume BC checkpoint hash mismatch")
    for field, value in expected_fields.items():
        if not value:
            raise ValueError("resume expected {} is missing".format(field))
        if field not in resume:
            raise ValueError("resume missing {}".format(field))
        _same(resume.get(field), value, field=field, path="AWAC resume")
    replay = validate_awac_replay_provenance(replay_metadata, expected=expected)
    for field in ("observation_contract", "observation_source"):
        _same(
            resume.get("replay_{}".format(field)),
            replay[field],
            field="replay_{}".format(field),
            path="AWAC resume",
        )
    return replay


def build_awac_provenance(
    *,
    bc_checkpoint_path: str,
    bc_checkpoint_sha256: str,
    bc: Mapping[str, Any],
    runtime: Mapping[str, Any],
    replay: Mapping[str, Any],
    run_identity: str,
    run_contract_sha256: str,
) -> Dict[str, Any]:
    """Project one canonical set of business provenance fields."""

    bc_fields = validate_awac_bc_checkpoint(
        bc,
        expected_mpl_contract_sha256=str(bc.get("mpl_contract_sha256", "")),
    )
    runtime_fields = validate_awac_runtime_provenance(
        observation_contract=str(runtime.get("observation_contract", "")),
        observation_source=str(runtime.get("observation_source", "")),
        reliable_execution_enabled=bool(runtime.get("reliable_execution_enabled", False)),
    )
    expected = {
        "bc_checkpoint_sha256": str(bc_checkpoint_sha256),
        "task_contract_id": bc_fields["task_contract_id"],
        "task_contract_sha256": bc_fields["task_contract_sha256"],
        "mpl_contract_sha256": bc_fields["mpl_contract_sha256"],
        "run_identity": str(run_identity),
        "run_contract_sha256": str(run_contract_sha256),
    }
    replay_fields = validate_awac_replay_provenance(replay, expected=expected)
    return {
        "awac_handoff_contract_id": AWAC_HANDOFF_CONTRACT_ID,
        "bc_checkpoint_path": str(bc_checkpoint_path),
        "bc_checkpoint_sha256": str(bc_checkpoint_sha256),
        "source_bc_checkpoint_sha256": str(bc_checkpoint_sha256),
        "bc_observation_contract": bc_fields["observation_contract"],
        "bc_observation_source": bc_fields["observation_source"],
        "runtime_observation_contract": runtime_fields["observation_contract"],
        "runtime_observation_source": runtime_fields["observation_source"],
        "expected_observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        "reliable_execution_enabled": True,
        "replay_observation_contract": replay_fields["observation_contract"],
        "replay_observation_source": replay_fields["observation_source"],
        "reliable_transition_count": replay_fields["reliable_transition_count"],
        "legacy_transition_count": replay_fields["legacy_transition_count"],
        "feature_contract_id": bc_fields["feature_contract_id"],
        "policy_input_contract_sha256": bc_fields["policy_input_contract_sha256"],
        "task_contract_id": bc_fields["task_contract_id"],
        "task_contract_sha256": bc_fields["task_contract_sha256"],
        "mpl_contract_sha256": bc_fields["mpl_contract_sha256"],
        "num_actions": NUM_ACTIONS,
        "state_feature_dim": CONTINUOUS_DIM,
        "previous_action_dim": NUM_ACTIONS,
        "action_ordering": ACTION_ORDERING,
        "run_identity": str(run_identity),
        "run_contract_sha256": str(run_contract_sha256),
    }


__all__ = [
    "ACTION_ORDERING",
    "AWAC_HANDOFF_CONTRACT_ID",
    "EXPECTED_OBSERVATION_CONTRACT",
    "EXPECTED_OBSERVATION_SOURCE",
    "build_awac_provenance",
    "validate_awac_bc_checkpoint",
    "validate_awac_replay_provenance",
    "validate_awac_resume_provenance",
    "validate_awac_runtime_provenance",
]
