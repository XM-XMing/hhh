"""Exact endpoint session seam for formal Teacher rollout collection.

The session owns provenance and transition identity only.  Teacher scoring,
feature construction, reward, and termination remain in their existing
owners.  A caller supplies the observation returned by the reliable runtime
after reset or after one committed primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    validate_reliable_exact_metadata,
)


ExactIdentity = Tuple[str, str, str, int, str, int]


@dataclass
class ExactCollectionCounters:
    """Counters written to a worker summary and checked by merge."""

    reliable_rows: int = 0
    legacy_rows: int = 0
    telemetry_lookup_count: int = 0
    snapshot_missing_count: int = 0
    state_depth_skew_max_ns: int = 0
    frame_contract_failures: int = 0
    endpoint_identity_chain_valid: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reliable_rows": int(self.reliable_rows),
            "legacy_rows": int(self.legacy_rows),
            "telemetry_lookup_count": int(self.telemetry_lookup_count),
            "snapshot_missing_count": int(self.snapshot_missing_count),
            "state_depth_skew_max_ns": int(self.state_depth_skew_max_ns),
            "exact_primitive_frame_count_failures": int(self.frame_contract_failures),
            "frame_contract_failures": int(self.frame_contract_failures),
            "endpoint_identity_chain_valid": bool(self.endpoint_identity_chain_valid),
        }


@dataclass(frozen=True)
class ExactTransitionRecord:
    """The provenance identity for one action-level rollout row."""

    transition_id: str
    execution_id: int
    runtime_instance_id: str
    action_id: int
    previous_action: int
    before_identity: ExactIdentity
    after_identity: ExactIdentity
    requested_frame_count: int
    applied_frame_count: int


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a mapping".format(name))
    return value


def _non_empty(value: Any, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("{} is missing".format(name))
    return text


def _integer(value: Any, *, name: str, non_negative: bool = True) -> int:
    if isinstance(value, bool):
        raise ValueError("{} must be an integer".format(name))
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be an integer".format(name))
    if non_negative and parsed < 0:
        raise ValueError("{} must be non-negative".format(name))
    return parsed


def _identity_from_endpoint(
    observation: Mapping[str, Any],
    *,
    expected_shape: Tuple[int, int],
    expected_runtime_instance_id: Optional[str] = None,
    expected_episode_id: Optional[str] = None,
    expected_reset_id: Optional[str] = None,
    path: str = "<observation>",
) -> ExactIdentity:
    """Validate one exact state/depth pair and return its full identity."""

    validate_reliable_exact_metadata(observation, path=path)
    state = _mapping(observation.get("state"), name="{}.state".format(path))
    sensor_time = _mapping(
        observation.get("sensor_time"), name="{}.sensor_time".format(path)
    )
    state_id = _integer(state.get("state_id"), name="{}.state_id".format(path))
    sim_time_ns = _integer(
        state.get("sim_time_ns"), name="{}.sim_time_ns".format(path)
    )
    state_stamp_ns = _integer(
        sensor_time.get("state_stamp_ns"),
        name="{}.state_stamp_ns".format(path),
    )
    depth_stamp_ns = _integer(
        sensor_time.get("depth_stamp_ns"),
        name="{}.depth_stamp_ns".format(path),
    )
    skew_ns = _integer(sensor_time.get("skew_ns"), name="{}.skew_ns".format(path))
    if state_stamp_ns != sim_time_ns or depth_stamp_ns != sim_time_ns or skew_ns != 0:
        raise ValueError(
            "{} exact observation is not co-timestamped: state={} depth={} sim={} skew={}".format(
                path, state_stamp_ns, depth_stamp_ns, sim_time_ns, skew_ns
            )
        )

    depth = np.asarray(observation.get("depth"), dtype=np.float32)
    if tuple(depth.shape) != tuple(expected_shape):
        raise ValueError(
            "{} exact depth shape {} != {}".format(path, depth.shape, expected_shape)
        )
    if not np.isfinite(depth).all():
        raise ValueError("{} exact depth contains non-finite values".format(path))

    endpoint = _mapping(
        observation.get("endpoint_identity"),
        name="{}.endpoint_identity".format(path),
    )
    runtime_instance_id = _non_empty(
        endpoint.get("runtime_instance_id"),
        name="{}.endpoint_identity.runtime_instance_id".format(path),
    )
    episode_id = _non_empty(
        endpoint.get("episode_id", observation.get("episode_id")),
        name="{}.endpoint_identity.episode_id".format(path),
    )
    reset_id = _non_empty(
        endpoint.get("reset_id", observation.get("reset_id")),
        name="{}.endpoint_identity.reset_id".format(path),
    )
    endpoint_state_id = _integer(
        endpoint.get("state_id"),
        name="{}.endpoint_identity.state_id".format(path),
    )
    depth_id = _non_empty(
        endpoint.get("depth_id"),
        name="{}.endpoint_identity.depth_id".format(path),
    )
    endpoint_sim_time_ns = _integer(
        endpoint.get("sim_time_ns"),
        name="{}.endpoint_identity.sim_time_ns".format(path),
    )
    if (endpoint_state_id, endpoint_sim_time_ns) != (state_id, sim_time_ns):
        raise ValueError("{} endpoint identity does not match state".format(path))
    if str(observation.get("episode_id", episode_id)) != episode_id:
        raise ValueError("{} episode identity mismatch".format(path))
    if str(observation.get("reset_id", reset_id)) != reset_id:
        raise ValueError("{} reset identity mismatch".format(path))
    if expected_runtime_instance_id is not None and runtime_instance_id != str(
        expected_runtime_instance_id
    ):
        raise ValueError(
            "{} runtime identity mismatch: {} != {}".format(
                path, runtime_instance_id, expected_runtime_instance_id
            )
        )
    if expected_episode_id is not None and episode_id != str(expected_episode_id):
        raise ValueError("{} episode identity mismatch".format(path))
    if expected_reset_id is not None and reset_id != str(expected_reset_id):
        raise ValueError("{} reset identity mismatch".format(path))
    return (
        runtime_instance_id,
        episode_id,
        reset_id,
        state_id,
        depth_id,
        sim_time_ns,
    )


def validate_exact_observation(
    observation: Mapping[str, Any],
    *,
    expected_shape: Tuple[int, int] = (90, 160),
    expected_runtime_instance_id: Optional[str] = None,
    expected_episode_id: Optional[str] = None,
    expected_reset_id: Optional[str] = None,
    path: str = "<observation>",
) -> ExactIdentity:
    """Public validator used by reset/step collection and focused tests."""

    return _identity_from_endpoint(
        _mapping(observation, name=path),
        expected_shape=(int(expected_shape[0]), int(expected_shape[1])),
        expected_runtime_instance_id=expected_runtime_instance_id,
        expected_episode_id=expected_episode_id,
        expected_reset_id=expected_reset_id,
        path=path,
    )


class ReliableExactRolloutSession:
    """Track the exact endpoint chain for one worker's active collection run."""

    observation_contract = EXACT_ENDPOINT_OBSERVATION_CONTRACT

    def __init__(
        self,
        *,
        runtime_instance_id: str,
        depth_shape: Tuple[int, int] = (90, 160),
        transition_id_offset: int = 0,
    ) -> None:
        self.runtime_instance_id = _non_empty(
            runtime_instance_id, name="runtime_instance_id"
        )
        self.depth_shape = (int(depth_shape[0]), int(depth_shape[1]))
        if self.depth_shape[0] <= 0 or self.depth_shape[1] <= 0:
            raise ValueError("depth_shape must be positive")
        if int(transition_id_offset) < 0:
            raise ValueError("transition_id_offset must be non-negative")
        self.counters = ExactCollectionCounters()
        self._episode_id: Optional[str] = None
        self._reset_id: Optional[str] = None
        self._previous_identity: Optional[ExactIdentity] = None
        self._execution_ids = set()
        self._transition_id_offset = int(transition_id_offset)
        self._issued_transition_ids = set()

    def reset(
        self,
        observation: Mapping[str, Any],
        *,
        episode_id: Optional[str] = None,
        reset_id: Optional[str] = None,
    ) -> ExactIdentity:
        identity = validate_exact_observation(
            observation,
            expected_shape=self.depth_shape,
            expected_runtime_instance_id=self.runtime_instance_id,
            expected_episode_id=episode_id,
            expected_reset_id=reset_id,
            path="reset",
        )
        self._episode_id = identity[1]
        self._reset_id = identity[2]
        self._previous_identity = identity
        self._execution_ids.clear()
        return identity

    def _reject_telemetry(self, info: Mapping[str, Any]) -> None:
        lookup_count = int(info.get("telemetry_lookup_count", 0) or 0)
        self.counters.telemetry_lookup_count += lookup_count
        if lookup_count != 0 or bool(info.get("telemetry_observation", False)):
            raise ValueError("telemetry observation/lookup is forbidden for formal collection")
        if bool(info.get("legacy_observation", False)):
            self.counters.legacy_rows += 1
            raise ValueError("legacy observation is forbidden for formal collection")

    @staticmethod
    def _reference_identity(reference: Mapping[str, Any]) -> ExactIdentity:
        ref = _mapping(reference, name="endpoint_observation_ref")
        return (
            _non_empty(ref.get("runtime_instance_id"), name="ref.runtime_instance_id"),
            _non_empty(ref.get("episode_id"), name="ref.episode_id"),
            _non_empty(ref.get("reset_id"), name="ref.reset_id"),
            _integer(ref.get("state_id"), name="ref.state_id"),
            _non_empty(ref.get("depth_id"), name="ref.depth_id"),
            _integer(ref.get("sim_time_ns"), name="ref.sim_time_ns"),
        )

    def record_transition(
        self,
        *,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        info: Mapping[str, Any],
        action_id: int,
        previous_action: int,
        command_frame_count: int,
    ) -> ExactTransitionRecord:
        if self._previous_identity is None:
            raise ValueError("exact session must be reset before recording a transition")
        if after is None:
            self.counters.snapshot_missing_count += 1
            raise ValueError("authoritative endpoint snapshot is missing")
        info_mapping = _mapping(info, name="transition.info")
        self._reject_telemetry(info_mapping)
        if info_mapping.get("reliable_v4") is not True and info_mapping.get(
            "reliable_execution"
        ) is not True:
            raise ValueError("formal transition is missing reliable execution identity")

        before_identity = validate_exact_observation(
            before,
            expected_shape=self.depth_shape,
            expected_runtime_instance_id=self.runtime_instance_id,
            expected_episode_id=self._episode_id,
            expected_reset_id=self._reset_id,
            path="transition.before",
        )
        after_identity = validate_exact_observation(
            after,
            expected_shape=self.depth_shape,
            expected_runtime_instance_id=self.runtime_instance_id,
            expected_episode_id=self._episode_id,
            expected_reset_id=self._reset_id,
            path="transition.after",
        )
        if before_identity != self._previous_identity:
            self.counters.endpoint_identity_chain_valid = False
            raise ValueError(
                "endpoint identity chain is not continuous: before={} previous={}".format(
                    before_identity, self._previous_identity
                )
            )

        execution = info_mapping.get("primitive_execution")
        execution_mapping = _mapping(execution, name="primitive_execution")
        execution_id = _integer(
            info_mapping.get(
                "reliable_v4_execution_id", execution_mapping.get("execution_id")
            ),
            name="execution_id",
        )
        if execution_id in self._execution_ids:
            raise ValueError("duplicate execution_id {}".format(execution_id))
        runtime_id = _non_empty(
            info_mapping.get(
                "reliable_v4_runtime_instance_id",
                execution_mapping.get("runtime_instance_id"),
            ),
            name="execution.runtime_instance_id",
        )
        if runtime_id != self.runtime_instance_id:
            raise ValueError("execution runtime identity mismatch")
        reference = info_mapping.get(
            "reliable_v4_observation_ref",
            execution_mapping.get("endpoint_observation_ref"),
        )
        if self._reference_identity(reference) != after_identity:
            self.counters.endpoint_identity_chain_valid = False
            raise ValueError(
                "endpoint ObservationRef does not match after observation"
            )

        requested = _integer(
            execution_mapping.get("requested_frame_count"),
            name="requested_frame_count",
        )
        applied = _integer(
            execution_mapping.get("applied_frame_count"),
            name="applied_frame_count",
        )
        expected_frames = _integer(
            command_frame_count, name="command_frame_count"
        )
        if requested != expected_frames or applied != expected_frames:
            self.counters.frame_contract_failures += 1
            raise ValueError(
                "exact primitive frame count mismatch: requested={} applied={} expected={}".format(
                    requested, applied, expected_frames
                )
                )

        transition_id = "{}:{}".format(
            self.runtime_instance_id, execution_id + self._transition_id_offset
        )
        while transition_id in self._issued_transition_ids:
            self._transition_id_offset += 1
            transition_id = "{}:{}".format(
                self.runtime_instance_id, execution_id + self._transition_id_offset
            )

        self._execution_ids.add(execution_id)
        self._issued_transition_ids.add(transition_id)
        self._previous_identity = after_identity
        self.counters.reliable_rows += 1
        self.counters.state_depth_skew_max_ns = max(
            self.counters.state_depth_skew_max_ns,
            int(after.get("sensor_time", {}).get("skew_ns", 0)),
        )
        return ExactTransitionRecord(
            transition_id=transition_id,
            execution_id=execution_id,
            runtime_instance_id=self.runtime_instance_id,
            action_id=int(action_id),
            previous_action=int(previous_action),
            before_identity=before_identity,
            after_identity=after_identity,
            requested_frame_count=requested,
            applied_frame_count=applied,
        )

    def provenance_metadata(self) -> Dict[str, Any]:
        from planning.contracts.observation import exact_endpoint_metadata

        metadata = exact_endpoint_metadata()
        metadata.update(self.counters.as_dict())
        return metadata


__all__ = [
    "ExactCollectionCounters",
    "ExactIdentity",
    "ExactTransitionRecord",
    "ReliableExactRolloutSession",
    "validate_exact_observation",
]
