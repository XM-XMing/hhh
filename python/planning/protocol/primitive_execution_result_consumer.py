"""Python-side v4 result consumer with exactly-once local accounting.

This module stops at the application seam used by P0-L4A:

    PrimitiveExecutionResult
        -> schema validation and identity deduplication
        -> endpoint observation provider
        -> transition commit interface
        -> replay append interface

The endpoint provider is intentionally injected.  It may be backed by a fake
in unit tests now and by the real state/depth binding in P0-L4B.  This module
does not open sockets, read ROS messages, or alter RL code.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import re
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, Union

_schema = import_module("planning.protocol.primitive_execution_schema" + "_" + "v4")
PrimitiveExecutionProtocolError = _schema.PrimitiveExecutionProtocolError
PrimitiveExecutionResult = _schema.PrimitiveExecutionResult
STATUS_COMPLETE = _schema.STATUS_COMPLETE
result_identity = _schema.result_identity
validate_execution_result = _schema.validate_execution_result


RESULT_CONSUMER_COMMITTED = "COMMITTED"
RESULT_CONSUMER_DUPLICATE = "DUPLICATE"
RESULT_CONSUMER_PROTOCOL_ERROR = "PROTOCOL_ERROR"
RESULT_CONSUMER_ENDPOINT_UNAVAILABLE = "ENDPOINT_UNAVAILABLE"
RESULT_CONSUMER_ENDPOINT_MISMATCH = "ENDPOINT_MISMATCH"
RESULT_CONSUMER_NO_TRANSITION = "NO_TRANSITION"
RESULT_CONSUMER_NOT_READY = "NOT_READY"

ResultKey = Tuple[str, int]
ResultInput = Union[PrimitiveExecutionResult, Mapping[str, Any]]
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class EndpointObservation:
    """Exact endpoint identity and payload supplied by an observation provider."""

    runtime_instance_id: str
    execution_id: int
    state_id: int
    sim_time_ns: int
    state: Any
    depth: Any = None
    episode_id: Optional[str] = None
    reset_id: Optional[str] = None
    depth_id: Any = None
    physics_time_ns: Optional[int] = None
    capture_time_ns: Optional[int] = None
    snapshot_hash: Optional[str] = None


@dataclass(frozen=True)
class ResultTransition:
    """Immutable transition input passed to commit and replay seams."""

    runtime_instance_id: str
    execution_id: int
    result_payload_hash: str
    result: PrimitiveExecutionResult
    endpoint: EndpointObservation


class EndpointObservationProvider(Protocol):
    def lookup(self, result: PrimitiveExecutionResult) -> Optional[EndpointObservation]:
        """Return the exact endpoint observation, or None if not available."""


class TransitionCommitter(Protocol):
    def commit(self, transition: ResultTransition) -> None:
        """Apply one transition side effect."""


class ReplayAppender(Protocol):
    def append_once(self, transition: ResultTransition) -> None:
        """Append the transition to replay with an idempotent identity."""


@dataclass(frozen=True)
class ResultConsumerOutcome:
    status: str
    runtime_instance_id: Optional[str] = None
    execution_id: Optional[int] = None
    result_payload_hash: Optional[str] = None
    transition_committed: bool = False
    replay_appended: bool = False
    error: Optional[str] = None


@dataclass
class _ResultRecord:
    result: PrimitiveExecutionResult
    result_payload_hash: str
    transition: Optional[ResultTransition] = None
    transition_committed: bool = False
    replay_appended: bool = False
    no_transition: bool = False


class PrimitiveExecutionResultConsumer:
    """Validate and account for v4 results exactly once per worker/execution.

    The identity key is ``(runtime_instance_id, execution_id)``.  A repeated
    result with the same semantic payload is idempotent; a repeated key with a
    different payload identity is a protocol error.  A result is remembered
    before endpoint lookup so a temporary endpoint miss cannot admit a
    conflicting payload later.
    """

    def __init__(
        self,
        *,
        endpoint_provider: EndpointObservationProvider,
        transition_committer: TransitionCommitter,
        replay_appender: ReplayAppender,
    ) -> None:
        self._endpoint_provider = endpoint_provider
        self._transition_committer = transition_committer
        self._replay_appender = replay_appender
        self._records: Dict[ResultKey, _ResultRecord] = {}
        self._transition_commit_count = 0
        self._replay_append_count = 0
        self._protocol_error_count = 0

    @property
    def transition_commit_count(self) -> int:
        return self._transition_commit_count

    @property
    def replay_append_count(self) -> int:
        return self._replay_append_count

    @property
    def protocol_error_count(self) -> int:
        return self._protocol_error_count

    @property
    def pending_count(self) -> int:
        """Number of identities seen but not yet fully accounted."""

        return sum(
            1
            for record in self._records.values()
            if not record.no_transition
            and not (record.transition_committed and record.replay_appended)
        )

    def consume(
        self,
        result: ResultInput,
        *,
        expected_runtime_instance_id: Optional[str] = None,
        expected_execution_id: Optional[int] = None,
        expected_command_sequence_hash: Optional[str] = None,
    ) -> ResultConsumerOutcome:
        """Consume one result mapping through the public accounting seam."""

        try:
            validated = validate_execution_result(
                result,
                expected_execution_id=expected_execution_id,
                expected_command_sequence_hash=expected_command_sequence_hash,
            )
            identity = result_identity(validated)
        except PrimitiveExecutionProtocolError as exc:
            self._protocol_error_count += 1
            return ResultConsumerOutcome(
                status=RESULT_CONSUMER_PROTOCOL_ERROR,
                error=str(exc),
            )

        if (
            expected_runtime_instance_id is not None
            and validated.runtime_instance_id != expected_runtime_instance_id
        ):
            return self._protocol_error(
                "runtime_instance_id mismatch: received={} expected={}".format(
                    validated.runtime_instance_id, expected_runtime_instance_id
                )
            )

        key = (identity.runtime_instance_id, identity.execution_id)
        identity_tuple = (
            identity.runtime_instance_id,
            identity.execution_id,
            identity.result_payload_hash,
        )
        record = self._records.get(key)
        if record is not None:
            if record.result_payload_hash != identity.result_payload_hash:
                return self._protocol_error(
                    "result_payload_hash conflict for runtime_instance_id={} execution_id={}".format(
                        identity.runtime_instance_id, identity.execution_id
                    ),
                    identity=identity_tuple,
                )
            if record.no_transition or (
                record.transition_committed and record.replay_appended
            ):
                return self._outcome(RESULT_CONSUMER_DUPLICATE, identity_tuple)
        else:
            record = _ResultRecord(
                result=validated,
                result_payload_hash=identity.result_payload_hash,
            )
            self._records[key] = record

        if validated.status != STATUS_COMPLETE:
            record.no_transition = True
            return self._outcome(RESULT_CONSUMER_NO_TRANSITION, identity_tuple)

        if record.transition is None:
            try:
                endpoint = self._endpoint_provider.lookup(validated)
            except PrimitiveExecutionProtocolError as exc:
                return self._protocol_error(
                    "endpoint snapshot protocol error: {}".format(exc),
                    identity=identity_tuple,
                )
            if endpoint is None:
                return self._outcome(
                    RESULT_CONSUMER_ENDPOINT_UNAVAILABLE,
                    identity_tuple,
                )
            mismatch = self._endpoint_mismatch(validated, endpoint)
            if mismatch is not None:
                return self._protocol_or_endpoint_mismatch(
                    mismatch,
                    identity_tuple,
                )
            record.transition = ResultTransition(
                runtime_instance_id=identity.runtime_instance_id,
                execution_id=identity.execution_id,
                result_payload_hash=identity.result_payload_hash,
                result=validated,
                endpoint=endpoint,
            )

        self._finalize(record)
        return self._outcome(
            RESULT_CONSUMER_COMMITTED,
            identity_tuple,
            transition_committed=record.transition_committed,
            replay_appended=record.replay_appended,
        )

    def commit_transition(
        self,
        *,
        runtime_instance_id: str,
        execution_id: int,
        result_payload_hash: str,
    ) -> ResultConsumerOutcome:
        """Idempotent explicit commit seam for a previously resolved result.

        This models a duplicate commit request without invoking either side
        effect a second time.  A commit is accepted only for the exact
        immutable identity already observed by :meth:`consume`.
        """

        identity_error = self._validate_commit_identity(
            runtime_instance_id, execution_id, result_payload_hash
        )
        if identity_error is not None:
            return self._protocol_or_endpoint_mismatch(
                identity_error,
                None,
            )
        identity = (runtime_instance_id, execution_id, result_payload_hash)
        record = self._records.get((runtime_instance_id, execution_id))
        if record is None or record.result_payload_hash != result_payload_hash:
            return self._protocol_error(
                "commit identity was not observed",
                identity=identity,
            )
        if record.no_transition:
            return self._outcome(RESULT_CONSUMER_NO_TRANSITION, identity)
        if record.transition is None:
            return self._outcome(RESULT_CONSUMER_NOT_READY, identity)
        if record.transition_committed and record.replay_appended:
            return self._outcome(RESULT_CONSUMER_DUPLICATE, identity)
        self._finalize(record)
        return self._outcome(
            RESULT_CONSUMER_COMMITTED,
            identity,
            transition_committed=record.transition_committed,
            replay_appended=record.replay_appended,
        )

    def _finalize(self, record: _ResultRecord) -> None:
        if record.transition is None:
            raise RuntimeError("cannot finalize a result without an endpoint transition")
        if not record.transition_committed:
            self._transition_committer.commit(record.transition)
            record.transition_committed = True
            self._transition_commit_count += 1
        if not record.replay_appended:
            self._replay_appender.append_once(record.transition)
            record.replay_appended = True
            self._replay_append_count += 1

    @staticmethod
    def _endpoint_mismatch(
        result: PrimitiveExecutionResult,
        endpoint: EndpointObservation,
    ) -> Optional[str]:
        expected = (
            result.runtime_instance_id,
            result.execution_id,
            result.endpoint_state_id,
            result.endpoint_sim_time_ns,
        )
        actual = (
            endpoint.runtime_instance_id,
            endpoint.execution_id,
            endpoint.state_id,
            endpoint.sim_time_ns,
        )
        if actual != expected:
            return "endpoint observation identity mismatch: received={} expected={}".format(
                actual, expected
            )
        observation_ref = result.endpoint_observation_ref
        if observation_ref is not None:
            expected_depth_id = observation_ref.get(
                "depth_id", observation_ref.get("capture_id")
            )
            strict_depth_identity = (
                "depth_id" in observation_ref
                or str(observation_ref.get("capture_id", "")).startswith("depth-")
            )
            if (
                strict_depth_identity
                and expected_depth_id is not None
                and endpoint.depth_id is not None
                and str(endpoint.depth_id) != str(expected_depth_id)
            ):
                return "endpoint depth identity mismatch: received={} expected={}".format(
                    endpoint.depth_id, expected_depth_id
                )
            expected_physics_time = observation_ref.get(
                "physics_time_ns", observation_ref.get("sim_time_ns")
            )
            if (
                expected_physics_time is not None
                and endpoint.physics_time_ns is not None
                and endpoint.physics_time_ns != expected_physics_time
            ):
                return "endpoint physics timestamp mismatch: received={} expected={}".format(
                    endpoint.physics_time_ns, expected_physics_time
                )
            expected_capture_time = observation_ref.get("capture_time_ns")
            if (
                expected_capture_time is not None
                and endpoint.capture_time_ns is not None
                and endpoint.capture_time_ns != expected_capture_time
            ):
                return "endpoint capture timestamp mismatch: received={} expected={}".format(
                    endpoint.capture_time_ns, expected_capture_time
                )
            for field_name, actual_value in (
                ("episode_id", endpoint.episode_id),
                ("reset_id", endpoint.reset_id),
            ):
                expected_value = observation_ref.get(field_name)
                if expected_value is not None and actual_value != expected_value:
                    return "endpoint {} mismatch: received={} expected={}".format(
                        field_name, actual_value, expected_value
                    )
        return None

    @staticmethod
    def _validate_commit_identity(
        runtime_instance_id: str,
        execution_id: int,
        result_payload_hash: str,
    ) -> Optional[str]:
        if not isinstance(runtime_instance_id, str) or not runtime_instance_id:
            return "runtime_instance_id is required"
        if isinstance(execution_id, bool) or not isinstance(execution_id, int) or execution_id < 0:
            return "execution_id must be a non-negative integer"
        if (
            not isinstance(result_payload_hash, str)
            or _SHA256_RE.fullmatch(result_payload_hash) is None
        ):
            return "result_payload_hash must be a lowercase SHA-256 digest"
        return None

    def _protocol_error(
        self,
        error: str,
        *,
        identity: Optional[Tuple[str, int, str]] = None,
    ) -> ResultConsumerOutcome:
        self._protocol_error_count += 1
        if identity is None:
            return ResultConsumerOutcome(
                status=RESULT_CONSUMER_PROTOCOL_ERROR,
                error=error,
            )
        return self._outcome(
            RESULT_CONSUMER_PROTOCOL_ERROR,
            identity,
            error=error,
        )

    def _protocol_or_endpoint_mismatch(
        self,
        error: str,
        identity: Optional[Tuple[str, int, str]],
    ) -> ResultConsumerOutcome:
        if error.startswith("endpoint observation identity mismatch"):
            return self._outcome(
                RESULT_CONSUMER_ENDPOINT_MISMATCH,
                identity,
                error=error,
            )
        return self._protocol_error(error, identity=identity)

    @staticmethod
    def _outcome(
        status: str,
        identity: Optional[Tuple[str, int, str]],
        *,
        transition_committed: bool = False,
        replay_appended: bool = False,
        error: Optional[str] = None,
    ) -> ResultConsumerOutcome:
        if identity is None:
            return ResultConsumerOutcome(
                status=status,
                transition_committed=transition_committed,
                replay_appended=replay_appended,
                error=error,
            )
        return ResultConsumerOutcome(
            status=status,
            runtime_instance_id=identity[0],
            execution_id=identity[1],
            result_payload_hash=identity[2],
            transition_committed=transition_committed,
            replay_appended=replay_appended,
            error=error,
        )


__all__ = [
    "EndpointObservation",
    "EndpointObservationProvider",
    "PrimitiveExecutionResultConsumer",
    "ReplayAppender",
    "RESULT_CONSUMER_COMMITTED",
    "RESULT_CONSUMER_DUPLICATE",
    "RESULT_CONSUMER_ENDPOINT_MISMATCH",
    "RESULT_CONSUMER_ENDPOINT_UNAVAILABLE",
    "RESULT_CONSUMER_NO_TRANSITION",
    "RESULT_CONSUMER_NOT_READY",
    "RESULT_CONSUMER_PROTOCOL_ERROR",
    "ResultConsumerOutcome",
    "ResultKey",
    "ResultTransition",
    "TransitionCommitter",
]
