"""Exact state/depth index for v4 endpoint observation binding.

The store never selects a latest observation or advances to a future frame.
State and depth records are accepted only for the configured runtime,
episode/reset context and are returned together only when their explicit
identity and physics timestamp match.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from planning.protocol.primitive_execution_result_consumer import EndpointObservation


class EndpointObservationStoreError(ValueError):
    """Raised when an indexed observation violates its binding contract."""


@dataclass(frozen=True)
class _IndexedRecord:
    observation_id: int
    state_id: int
    physics_time_ns: int
    capture_time_ns: int
    depth_id: Any
    runtime_instance_id: str
    episode_id: str
    reset_id: str
    value: Any


_MISSING = object()


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        result = value.get(name, default)
    else:
        result = getattr(value, name, default)
    if result is _MISSING:
        raise EndpointObservationStoreError(
            "observation is missing required field {}".format(name)
        )
    return result


def _required_non_empty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EndpointObservationStoreError(
            "{} must be a non-empty string".format(name)
        )
    return value


def _required_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EndpointObservationStoreError(
            "{} must be a non-negative integer".format(name)
        )
    return value


def _required_depth_id(value: Any, name: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise EndpointObservationStoreError(
            "{} must be an integer or non-empty string".format(name)
        )
    if isinstance(value, int) and value < 0:
        raise EndpointObservationStoreError(
            "{} must be non-negative".format(name)
        )
    if isinstance(value, str) and not value:
        raise EndpointObservationStoreError(
            "{} must be non-empty".format(name)
        )
    return value


class IndexedEndpointObservationStore:
    """Store exact state/depth pairs for one episode and reset boundary.

    ``put_depth(depth_id, depth)`` requires the depth payload's ``state_id``
    when present; otherwise ``depth_id`` is its state binding.  This keeps the
    public API usable for both indexed mappings and message-like objects while
    still making the state/depth pairing explicit.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        reset_id: str,
        runtime_instance_id: str = "",
        diagnostic_sink: Optional[Callable[..., None]] = None,
    ) -> None:
        self._runtime_instance_id = _required_non_empty_text(
            runtime_instance_id, "runtime_instance_id"
        ) if runtime_instance_id else ""
        self._episode_id = _required_non_empty_text(episode_id, "episode_id")
        self._reset_id = _required_non_empty_text(reset_id, "reset_id")
        self._states: Dict[int, _IndexedRecord] = {}
        self._depths: Dict[Any, _IndexedRecord] = {}
        self._diagnostic_sink = diagnostic_sink

    def put_state(self, state_id: int, state: Any) -> None:
        """Index one state without replacing its declared identity."""

        record_id = _required_non_negative_int(state_id, "state_id")
        try:
            record = self._record(
                observation_id=record_id,
                state_id=record_id,
                value=state,
                kind="state",
            )
            outcome = self._put_once(self._states, record, "state")
        except EndpointObservationStoreError as error:
            self._emit("STATE_REJECTED", state_id=record_id, reason=str(error))
            raise
        self._emit(
            "STATE_INSERT",
            state_id=record.state_id,
            physics_time_ns=record.physics_time_ns,
            insert_outcome=outcome,
        )

    def put_depth(self, depth_id: int, depth: Any) -> None:
        """Index one depth capture by the state it explicitly represents."""

        record_id = _required_depth_id(depth_id, "depth_id")
        declared_state_id = _field(depth, "state_id")
        state_id = _required_non_negative_int(declared_state_id, "state_id")
        declared_depth_id = _field(depth, "depth_id", record_id)
        declared_depth_id = _required_depth_id(declared_depth_id, "depth_id")
        if declared_depth_id != record_id:
            raise EndpointObservationStoreError(
                "depth_id argument does not match depth payload"
            )
        try:
            record = self._record(
                observation_id=record_id,
                state_id=state_id,
                value=depth,
                kind="depth",
            )
            outcome = self._put_once(self._depths, record, "depth", key=record_id)
        except EndpointObservationStoreError as error:
            self._emit("DEPTH_REJECTED", depth_id=record_id, reason=str(error))
            raise
        self._emit(
            "DEPTH_INSERT",
            state_id=record.state_id,
            depth_id=record.depth_id,
            physics_time_ns=record.physics_time_ns,
            capture_time_ns=record.capture_time_ns,
            insert_outcome=outcome,
        )

    def get(
        self,
        endpoint_state_id: int,
        *,
        observation_ref: Optional[Mapping[str, Any]] = None,
    ) -> Optional[EndpointObservation]:
        """Return the exact state/depth pair for ``endpoint_state_id``.

        With an observation ref, depth identity, physics time, capture time,
        and episode/reset identity are all checked. Without one, the legacy
        API is accepted only when exactly one indexed depth represents the
        requested state; no latest/nearest selection is performed.
        """

        state_id = _required_non_negative_int(endpoint_state_id, "endpoint_state_id")
        state = self._states.get(state_id)
        if state is None:
            self._emit(
                "LOOKUP_MISS",
                endpoint_state_id=state_id,
                depth_id=self._requested_depth_id(observation_ref),
                reason="STATE_MISSING",
            )
            return None
        depth, miss_reason = self._find_depth(state, observation_ref)
        if depth is None:
            self._emit(
                "LOOKUP_MISS",
                endpoint_state_id=state_id,
                depth_id=self._requested_depth_id(observation_ref),
                reason=miss_reason,
            )
            return None
        if (
            state.state_id != depth.state_id
            or state.physics_time_ns != depth.physics_time_ns
            or state.episode_id != depth.episode_id
            or state.reset_id != depth.reset_id
            or state.runtime_instance_id != depth.runtime_instance_id
        ):
            self._emit(
                "LOOKUP_MISS",
                endpoint_state_id=state_id,
                depth_id=depth.depth_id,
                reason="STATE_DEPTH_IDENTITY_MISMATCH",
            )
            return None
        self._emit(
            "LOOKUP_HIT",
            endpoint_state_id=state_id,
            depth_id=depth.depth_id,
        )
        return EndpointObservation(
            runtime_instance_id=state.runtime_instance_id,
            execution_id=-1,
            state_id=state.state_id,
            sim_time_ns=state.physics_time_ns,
            state=state.value,
            depth=depth.value,
            episode_id=state.episode_id,
            reset_id=state.reset_id,
            depth_id=depth.depth_id,
            physics_time_ns=state.physics_time_ns,
            capture_time_ns=depth.capture_time_ns,
        )

    def lookup(self, result: Any) -> Optional[EndpointObservation]:
        """Adapt the store to the consumer's exact endpoint-provider seam."""

        endpoint_state_id = _field(result, "endpoint_state_id", None)
        if endpoint_state_id is None:
            self._emit("LOOKUP_MISS", endpoint_state_id=None, depth_id=None,
                       reason="RESULT_ENDPOINT_STATE_ID_MISSING")
            return None
        observation_ref = _field(result, "endpoint_observation_ref", None)
        if observation_ref is not None and not isinstance(observation_ref, Mapping):
            self._emit(
                "LOOKUP_MISS",
                endpoint_state_id=endpoint_state_id,
                depth_id=None,
                reason="INVALID_OBSERVATION_REF",
            )
            return None
        strict_identity = observation_ref is not None and any(
            field in observation_ref
            for field in (
                "depth_id",
                "physics_time_ns",
                "capture_time_ns",
                "episode_id",
                "reset_id",
            )
        )
        if (
            observation_ref is not None
            and str(observation_ref.get("capture_id", "")).startswith("depth-")
        ):
            strict_identity = True
        observation = self.get(
            endpoint_state_id,
            observation_ref=observation_ref if strict_identity else None,
        )
        if observation is None:
            return None
        result_runtime = _field(result, "runtime_instance_id", "")
        if self._runtime_instance_id and result_runtime != self._runtime_instance_id:
            self._emit(
                "LOOKUP_MISS",
                endpoint_state_id=endpoint_state_id,
                depth_id=self._requested_depth_id(observation_ref),
                reason="RUNTIME_INSTANCE_MISMATCH",
            )
            return None
        return replace(
            observation,
            runtime_instance_id=str(result_runtime or observation.runtime_instance_id),
            execution_id=int(_field(result, "execution_id", -1)),
        )

    def _record(
        self,
        *,
        observation_id: int,
        state_id: int,
        value: Any,
        kind: str,
    ) -> _IndexedRecord:
        declared_state_id = _field(value, "state_id", state_id)
        if declared_state_id != state_id:
            raise EndpointObservationStoreError(
                "{} state_id does not match its index".format(kind)
            )
        physics_time_value = _field(value, "physics_time_ns", None)
        if self._runtime_instance_id and physics_time_value is None:
            raise EndpointObservationStoreError(
                "{} physics_time_ns is required".format(kind)
            )
        if physics_time_value is None:
            physics_time_value = _field(value, "sim_time_ns")
        physics_time_ns = _required_non_negative_int(
            physics_time_value, "physics_time_ns"
        )
        capture_time_value = _field(value, "capture_time_ns", None)
        if self._runtime_instance_id and kind == "depth" and capture_time_value is None:
            raise EndpointObservationStoreError(
                "{} capture_time_ns is required".format(kind)
            )
        if capture_time_value is None:
            capture_time_value = _field(value, "sim_time_ns")
        capture_time_ns = _required_non_negative_int(
            capture_time_value, "capture_time_ns"
        )
        runtime_instance_id = _field(value, "runtime_instance_id", None)
        if self._runtime_instance_id and runtime_instance_id is None:
            raise EndpointObservationStoreError(
                "{} runtime_instance_id is required".format(kind)
            )
        if runtime_instance_id is None:
            runtime_instance_id = self._runtime_instance_id
        runtime_instance_id = _required_non_empty_text(
            runtime_instance_id, "runtime_instance_id"
        ) if runtime_instance_id else ""
        if self._runtime_instance_id and runtime_instance_id != self._runtime_instance_id:
            raise EndpointObservationStoreError(
                "{} runtime_instance_id does not match the active runtime".format(kind)
            )
        episode_id = _required_non_empty_text(
            _field(value, "episode_id"),
            "episode_id",
        )
        reset_id = _required_non_empty_text(
            _field(value, "reset_id"),
            "reset_id",
        )
        if episode_id != self._episode_id:
            raise EndpointObservationStoreError(
                "{} episode_id does not match the active episode".format(kind)
            )
        if reset_id != self._reset_id:
            raise EndpointObservationStoreError(
                "{} reset_id does not match the active reset".format(kind)
            )
        return _IndexedRecord(
            observation_id=observation_id,
            state_id=state_id,
            physics_time_ns=physics_time_ns,
            capture_time_ns=capture_time_ns,
            depth_id=(
                _required_depth_id(
                    _field(
                        value,
                        "depth_id",
                        observation_id if not self._runtime_instance_id else _MISSING,
                    ),
                    "depth_id",
                )
                if kind == "depth"
                else None
            ),
            runtime_instance_id=runtime_instance_id,
            episode_id=episode_id,
            reset_id=reset_id,
            value=value,
        )

    def _find_depth(
        self,
        state: _IndexedRecord,
        observation_ref: Optional[Mapping[str, Any]],
    ) -> Tuple[Optional[_IndexedRecord], str]:
        if observation_ref is not None:
            ref_state_id = observation_ref.get("state_id")
            if ref_state_id != state.state_id:
                return None, "REFERENCE_STATE_ID_MISMATCH"
            ref_physics_time = observation_ref.get(
                "physics_time_ns", observation_ref.get("sim_time_ns")
            )
            if ref_physics_time != state.physics_time_ns:
                return None, "REFERENCE_PHYSICS_TIME_MISMATCH"
            ref_episode = observation_ref.get("episode_id", state.episode_id)
            ref_reset = observation_ref.get("reset_id", state.reset_id)
            if ref_episode != state.episode_id or ref_reset != state.reset_id:
                return None, "REFERENCE_EPISODE_RESET_MISMATCH"
            ref_depth_id = observation_ref.get("depth_id")
            if ref_depth_id is None:
                ref_depth_id = observation_ref.get("capture_id")
            candidates = [
                record for record in self._depths.values()
                if record.state_id == state.state_id
                and self._same_depth_identity(record, ref_depth_id)
            ]
            if not candidates:
                return None, "DEPTH_MISSING"
            if len(candidates) != 1:
                return None, "AMBIGUOUS_DEPTH"
            depth = candidates[0]
            ref_capture_time = observation_ref.get("capture_time_ns")
            if ref_capture_time is not None and ref_capture_time != depth.capture_time_ns:
                return None, "REFERENCE_CAPTURE_TIME_MISMATCH"
            return depth, ""

        candidates = [
            record for record in self._depths.values()
            if record.state_id == state.state_id
        ]
        if not candidates:
            return None, "DEPTH_MISSING"
        if len(candidates) != 1:
            return None, "AMBIGUOUS_DEPTH"
        return candidates[0], ""

    @staticmethod
    def _requested_depth_id(
        observation_ref: Optional[Mapping[str, Any]],
    ) -> Any:
        if observation_ref is None:
            return None
        return observation_ref.get("depth_id", observation_ref.get("capture_id"))

    def _emit(self, event: str, **fields: Any) -> None:
        if self._diagnostic_sink is not None:
            self._diagnostic_sink(str(event), **fields)

    @staticmethod
    def _same_depth_identity(record: _IndexedRecord, requested: Any) -> bool:
        if requested is None:
            return False
        if record.depth_id == requested:
            return True
        capture_id = _field(record.value, "capture_id", None)
        return capture_id == requested or str(record.depth_id) == str(requested)

    @staticmethod
    def _put_once(
        records: Dict[Any, _IndexedRecord],
        record: _IndexedRecord,
        kind: str,
        *,
        key: Optional[int] = None,
    ) -> str:
        record_key = record.state_id if key is None else key
        previous = records.get(record_key)
        if previous is None:
            records[record_key] = record
            return "INSERTED"
        if previous != record:
            raise EndpointObservationStoreError(
                "conflicting {} record for state_id={}".format(
                    kind, record.state_id
                )
            )
        return "DUPLICATE"


__all__ = [
    "EndpointObservationStoreError",
    "IndexedEndpointObservationStore",
]
