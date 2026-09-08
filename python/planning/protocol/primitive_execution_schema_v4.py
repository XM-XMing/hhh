"""Pure schema-v4 primitive execution model and identity invariants.

This module deliberately stops at the protocol-input boundary.  It does not
open sockets, acknowledge messages, execute Unity commands, or commit RL
transitions.  The public seam is:

    protocol mapping -> validated result -> canonical identity

Canonical identities use recursively sorted maps and canonical MessagePack.
The result payload hash excludes the ``result_payload_hash`` field itself, so
retransmitting the same semantic result produces the same digest.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from planning.protocol.constants import (
    PRIMITIVE_FRAME_COUNT,
    SCHEMA_VERSION as PROTOCOL_SCHEMA_VERSION,
)
from planning.protocol.validation import (
    require_digest,
    require_field,
    require_int,
    require_mapping,
    require_non_empty_text,
)
from planning.protocol.msgpack import (
    CanonicalMessagePackWriter,
    canonical_messagepack_bytes as _canonical_messagepack_bytes,
)


SCHEMA_VERSION = PROTOCOL_SCHEMA_VERSION
DEFAULT_REQUESTED_FRAME_COUNT = PRIMITIVE_FRAME_COUNT
RESULT_MESSAGE_TYPE = "PrimitiveExecutionResult"

STATUS_COMPLETE = "COMPLETE"
STATUS_REJECTED = "REJECTED"
STATUS_FAILED = "FAILED"
STATUS_CANCELLED = "CANCELLED"
VALID_STATUSES = frozenset(
    {STATUS_COMPLETE, STATUS_REJECTED, STATUS_FAILED, STATUS_CANCELLED}
)

# These are environment outcomes, rather than transport/protocol failures.
# They may carry an exact authoritative terminal observation so the
# environment can materialize one normal terminal transition.
RELIABLE_ENVIRONMENT_FAILURE_REASONS = frozenset({"COLLISION"})

RESULT_IDENTITY_FIRST = "FIRST"
RESULT_IDENTITY_DUPLICATE = "DUPLICATE"

class PrimitiveExecutionProtocolError(ValueError):
    """Raised when a v4 protocol input violates a fail-closed invariant."""


class SchemaMismatchError(PrimitiveExecutionProtocolError):
    """Raised before parsing the rest of a message with the wrong schema."""


def _optional_int(payload: Mapping[str, Any], name: str) -> Optional[int]:
    value = require_field(payload, name)
    if value is None:
        return None
    return require_int(value, name=name)


def canonical_messagepack_bytes(payload: Any) -> bytes:
    """Encode a payload with deterministic recursive map ordering."""

    try:
        return _canonical_messagepack_bytes(payload)
    except (TypeError, ValueError) as error:
        raise PrimitiveExecutionProtocolError(str(error)) from error


def canonical_command_sequence_hash(
    command_frames: Sequence[Mapping[str, Any]],
) -> str:
    """Hash command frames in their submitted order.

    Map key order is not semantic; frame sequence order is semantic.  The
    caller is responsible for supplying the complete command sequence.
    """

    return canonical_v4_command_sequence_hash(command_frames)


def _v4_frame_values(frame: Mapping[str, Any], index: int):
    mapping = require_mapping(frame, name="command_frames[{}]".format(index))
    frame_index = require_int(require_field(mapping, "frame_index"), name="frame_index")
    command_id = require_int(require_field(mapping, "command_id"), name="command_id")
    action = require_field(mapping, "action")
    if isinstance(action, (str, bytes)) or not isinstance(action, SequenceABC):
        raise PrimitiveExecutionProtocolError(
            "command_frames[{}].action must be a sequence".format(index)
        )
    return frame_index, command_id, action


def canonical_v4_command_sequence_bytes(
    command_frames: Sequence[Mapping[str, Any]],
) -> bytes:
    """Encode command frames with the fixed v4 typed MessagePack contract."""

    if isinstance(command_frames, (str, bytes)) or not isinstance(command_frames, SequenceABC):
        raise PrimitiveExecutionProtocolError("command_frames must be a sequence")
    if not command_frames:
        raise PrimitiveExecutionProtocolError("command_frames must not be empty")
    writer = CanonicalMessagePackWriter(error_type=PrimitiveExecutionProtocolError)
    writer.array_header(len(command_frames))
    for index, frame in enumerate(command_frames):
        frame_index, command_id, action = _v4_frame_values(frame, index)
        writer.map_header(3)
        writer.string("action")
        writer.array_header(len(action))
        for value in action:
            writer.float32(value)
        writer.string("command_id")
        writer.int64(command_id)
        writer.string("frame_index")
        writer.uint32(frame_index)
    return writer.bytes()


def canonical_v4_command_sequence_hash(
    command_frames: Sequence[Mapping[str, Any]],
) -> str:
    return hashlib.sha256(canonical_v4_command_sequence_bytes(command_frames)).hexdigest()


@dataclass(frozen=True)
class PrimitiveExecutionResult:
    """Validated v4 final result, independent of any transport implementation."""

    schema_version: int
    message_type: str
    runtime_instance_id: str
    execution_id: int
    status: str
    requested_frame_count: int
    applied_frame_count: int
    first_applied_state_id: Optional[int]
    endpoint_state_id: Optional[int]
    last_applied_frame_index: int
    reason_code: str
    command_sequence_hash: str
    endpoint_sim_time_ns: Optional[int]
    endpoint_observation_ref: Optional[Mapping[str, Any]] = None
    result_generation: int = 0
    result_payload_hash: Optional[str] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PrimitiveExecutionResult":
        """Parse and validate one protocol mapping before identity use."""

        mapping = require_mapping(payload, name="result")
        schema_version = require_int(require_field(mapping, "schema_version"), name="schema_version")
        if schema_version != SCHEMA_VERSION:
            raise SchemaMismatchError(
                "schema_version mismatch: received={} expected={}".format(
                    schema_version, SCHEMA_VERSION
                )
            )

        observation_ref = mapping.get("endpoint_observation_ref")
        if observation_ref is not None:
            observation_ref = dict(
                require_mapping(observation_ref, name="endpoint_observation_ref")
            )

        result = cls(
            schema_version=schema_version,
            message_type=require_non_empty_text(require_field(mapping, "message_type"), name="message_type"),
            runtime_instance_id=require_non_empty_text(require_field(mapping, "runtime_instance_id"), name="runtime_instance_id"),
            execution_id=require_int(require_field(mapping, "execution_id"), name="execution_id"),
            status=require_non_empty_text(require_field(mapping, "status"), name="status"),
            requested_frame_count=require_int(require_field(mapping, "requested_frame_count"), name="requested_frame_count"),
            applied_frame_count=require_int(require_field(mapping, "applied_frame_count"), name="applied_frame_count"),
            first_applied_state_id=_optional_int(mapping, "first_applied_state_id"),
            endpoint_state_id=_optional_int(mapping, "endpoint_state_id"),
            last_applied_frame_index=require_int(require_field(mapping, "last_applied_frame_index"), name="last_applied_frame_index"),
            reason_code=require_non_empty_text(require_field(mapping, "reason_code"), name="reason_code"),
            command_sequence_hash=require_non_empty_text(require_field(mapping, "command_sequence_hash"), name="command_sequence_hash"),
            endpoint_sim_time_ns=_optional_int(mapping, "endpoint_sim_time_ns"),
            endpoint_observation_ref=observation_ref,
            result_generation=mapping.get("result_generation", 0),
            result_payload_hash=mapping.get("result_payload_hash"),
        )
        return result.validate()

    def canonical_payload(self) -> Dict[str, Any]:
        """Return the semantic result map used for result identity."""

        return {
            "schema_version": self.schema_version,
            "message_type": self.message_type,
            "runtime_instance_id": self.runtime_instance_id,
            "execution_id": self.execution_id,
            "status": self.status,
            "requested_frame_count": self.requested_frame_count,
            "applied_frame_count": self.applied_frame_count,
            "first_applied_state_id": self.first_applied_state_id,
            "endpoint_state_id": self.endpoint_state_id,
            "last_applied_frame_index": self.last_applied_frame_index,
            "reason_code": self.reason_code,
            "command_sequence_hash": self.command_sequence_hash,
            "endpoint_sim_time_ns": self.endpoint_sim_time_ns,
            "endpoint_observation_ref": self.endpoint_observation_ref,
            "result_generation": self.result_generation,
        }

    def validate(
        self,
        *,
        expected_execution_id: Optional[int] = None,
        expected_command_sequence_hash: Optional[str] = None,
        expected_schema_version: int = SCHEMA_VERSION,
        expected_frame_count: int = DEFAULT_REQUESTED_FRAME_COUNT,
    ) -> "PrimitiveExecutionResult":
        """Validate schema, identity bindings, and status-specific invariants."""

        if self.schema_version != expected_schema_version:
            if expected_schema_version == SCHEMA_VERSION:
                raise SchemaMismatchError(
                    "schema_version mismatch: received={} expected={}".format(
                        self.schema_version, expected_schema_version
                    )
                )
            raise PrimitiveExecutionProtocolError(
                "schema_version mismatch: received={} expected={}".format(
                    self.schema_version, expected_schema_version
                )
            )
        if self.message_type != RESULT_MESSAGE_TYPE:
            raise PrimitiveExecutionProtocolError(
                "message_type mismatch: received={} expected={}".format(
                    self.message_type, RESULT_MESSAGE_TYPE
                )
            )
        if not self.runtime_instance_id:
            raise PrimitiveExecutionProtocolError("runtime_instance_id is required")
        if self.execution_id < 0:
            raise PrimitiveExecutionProtocolError("execution_id must be non-negative")
        if expected_execution_id is not None and self.execution_id != expected_execution_id:
            raise PrimitiveExecutionProtocolError(
                "execution_id mismatch: received={} expected={}".format(
                    self.execution_id, expected_execution_id
                )
            )
        if self.status not in VALID_STATUSES:
            raise PrimitiveExecutionProtocolError(
                "unknown execution result status: {}".format(self.status)
            )
        if self.requested_frame_count != expected_frame_count:
            raise PrimitiveExecutionProtocolError(
                "requested_frame_count mismatch: received={} expected={}".format(
                    self.requested_frame_count, expected_frame_count
                )
            )
        if self.applied_frame_count < 0 or self.applied_frame_count > expected_frame_count:
            raise PrimitiveExecutionProtocolError(
                "applied_frame_count is outside the requested frame range"
            )
        require_digest(self.command_sequence_hash, name="command_sequence_hash")
        if expected_command_sequence_hash is not None:
            require_digest(
                expected_command_sequence_hash,
                name="expected_command_sequence_hash",
            )
            if self.command_sequence_hash != expected_command_sequence_hash:
                raise PrimitiveExecutionProtocolError(
                    "command_sequence_hash mismatch: received={} expected={}".format(
                        self.command_sequence_hash, expected_command_sequence_hash
                    )
                )
        if isinstance(self.result_generation, bool) or not isinstance(self.result_generation, int):
            raise PrimitiveExecutionProtocolError("result_generation must be uint32")
        if self.result_generation != 0:
            raise PrimitiveExecutionProtocolError(
                "result_generation must remain 0 for immutable v4 retransmission"
            )
        if self.endpoint_sim_time_ns is not None and self.endpoint_sim_time_ns < 0:
            raise PrimitiveExecutionProtocolError("endpoint_sim_time_ns must be non-negative")

        if self.status == STATUS_COMPLETE:
            self._validate_complete(expected_frame_count)
        elif self.status == STATUS_REJECTED:
            self._validate_rejected()
        else:
            self._validate_interrupted(expected_frame_count)

        if self.result_payload_hash is not None:
            supplied_hash = require_digest(
                self.result_payload_hash,
                name="result_payload_hash",
            )
            actual_hash = _hash_result_payload(self)
            if supplied_hash != actual_hash:
                raise PrimitiveExecutionProtocolError(
                    "result_payload_hash does not match the semantic result payload"
                )
        return self

    def _validate_complete(self, expected_frame_count: int) -> None:
        if self.requested_frame_count != expected_frame_count:
            raise PrimitiveExecutionProtocolError(
                "COMPLETE requested_frame_count must equal {}".format(
                    expected_frame_count
                )
            )
        if self.applied_frame_count != expected_frame_count:
            raise PrimitiveExecutionProtocolError(
                "COMPLETE applied_frame_count must equal {}".format(
                    expected_frame_count
                )
            )
        if self.last_applied_frame_index != expected_frame_count - 1:
            raise PrimitiveExecutionProtocolError(
                "COMPLETE last_applied_frame_index must equal {}".format(
                    expected_frame_count - 1
                )
            )
        if self.first_applied_state_id is None or self.first_applied_state_id < 0:
            raise PrimitiveExecutionProtocolError(
                "COMPLETE first_applied_state_id is required"
            )
        expected_endpoint = self.first_applied_state_id + expected_frame_count - 1
        if self.endpoint_state_id != expected_endpoint:
            raise PrimitiveExecutionProtocolError(
                "COMPLETE endpoint_state_id mismatch: received={} expected={}".format(
                    self.endpoint_state_id, expected_endpoint
                )
            )
        if self.endpoint_sim_time_ns is None:
            raise PrimitiveExecutionProtocolError(
                "COMPLETE endpoint_sim_time_ns is required"
            )
        if self.reason_code != "NONE":
            raise PrimitiveExecutionProtocolError(
                "COMPLETE reason_code must be NONE"
            )
        observation_ref = self.endpoint_observation_ref
        if not isinstance(observation_ref, MappingABC):
            raise PrimitiveExecutionProtocolError(
                "COMPLETE endpoint_observation_ref is required"
            )
        required_keys = {
            "schema_version",
            "runtime_instance_id",
            "episode_id",
            "reset_id",
            "state_id",
            "depth_id",
            "sim_time_ns",
        }
        if set(observation_ref.keys()) != required_keys:
            raise PrimitiveExecutionProtocolError(
                "COMPLETE endpoint_observation_ref must contain the exact ObservationRef fields"
            )
        if observation_ref["schema_version"] != SCHEMA_VERSION:
            raise SchemaMismatchError(
                "COMPLETE endpoint observation schema_version mismatch"
            )
        if observation_ref["runtime_instance_id"] != self.runtime_instance_id:
            raise PrimitiveExecutionProtocolError(
                "COMPLETE endpoint observation runtime_instance_id mismatch"
            )
        for field in ("episode_id", "reset_id", "depth_id"):
            if not isinstance(observation_ref[field], str) or not observation_ref[field]:
                raise PrimitiveExecutionProtocolError(
                    "COMPLETE endpoint observation {} is required".format(field)
                )
        if observation_ref.get("state_id") != self.endpoint_state_id:
            raise PrimitiveExecutionProtocolError(
                "COMPLETE endpoint observation state_id mismatch"
            )
        if observation_ref.get("sim_time_ns") != self.endpoint_sim_time_ns:
            raise PrimitiveExecutionProtocolError(
                "COMPLETE endpoint observation sim_time_ns mismatch"
            )

    def _validate_rejected(self) -> None:
        if self.applied_frame_count != 0:
            raise PrimitiveExecutionProtocolError(
                "REJECTED applied_frame_count must equal 0"
            )
        if self.last_applied_frame_index != -1:
            raise PrimitiveExecutionProtocolError(
                "REJECTED last_applied_frame_index must equal -1"
            )
        if self.first_applied_state_id is not None or self.endpoint_state_id is not None:
            raise PrimitiveExecutionProtocolError(
                "REJECTED must not claim an applied state endpoint"
            )
        if self.endpoint_sim_time_ns is not None or self.endpoint_observation_ref is not None:
            raise PrimitiveExecutionProtocolError(
                "REJECTED must not claim an endpoint observation"
            )
        if self.reason_code == "NONE":
            raise PrimitiveExecutionProtocolError(
                "REJECTED reason_code must identify the rejection"
            )

    def _validate_interrupted(self, expected_frame_count: int) -> None:
        if (
            self.status == STATUS_FAILED
            and self.reason_code in RELIABLE_ENVIRONMENT_FAILURE_REASONS
        ):
            self._validate_terminal_observation_failure(expected_frame_count)
            return
        if (
            self.status == STATUS_FAILED
            and self.reason_code == "PHYSICS_COMPLETE_OBSERVATION_FAILED"
        ):
            if self.applied_frame_count != expected_frame_count:
                raise PrimitiveExecutionProtocolError(
                    "PHYSICS_COMPLETE_OBSERVATION_FAILED applied_frame_count must equal {}".format(
                        expected_frame_count
                    )
                )
            if self.last_applied_frame_index != expected_frame_count - 1:
                raise PrimitiveExecutionProtocolError(
                    "PHYSICS_COMPLETE_OBSERVATION_FAILED last_applied_frame_index must equal {}".format(
                        expected_frame_count - 1
                    )
                )
            if self.first_applied_state_id is None or self.first_applied_state_id < 0:
                raise PrimitiveExecutionProtocolError(
                    "PHYSICS_COMPLETE_OBSERVATION_FAILED requires first_applied_state_id"
                )
            if self.endpoint_state_id is not None:
                raise PrimitiveExecutionProtocolError(
                    "PHYSICS_COMPLETE_OBSERVATION_FAILED must not claim endpoint_state_id"
                )
            if self.endpoint_sim_time_ns is not None or self.endpoint_observation_ref is not None:
                raise PrimitiveExecutionProtocolError(
                    "PHYSICS_COMPLETE_OBSERVATION_FAILED must not claim endpoint observation"
                )
            return
        if self.applied_frame_count >= expected_frame_count:
            raise PrimitiveExecutionProtocolError(
                "{} cannot claim a complete applied prefix".format(self.status)
            )
        expected_last = self.applied_frame_count - 1
        if self.last_applied_frame_index != expected_last:
            raise PrimitiveExecutionProtocolError(
                "{} last_applied_frame_index must equal {}".format(
                    self.status, expected_last
                )
            )
        if self.applied_frame_count == 0:
            if self.first_applied_state_id is not None:
                raise PrimitiveExecutionProtocolError(
                    "zero-applied {} must not claim first_applied_state_id".format(
                        self.status
                    )
                )
        elif self.first_applied_state_id is None or self.first_applied_state_id < 0:
            raise PrimitiveExecutionProtocolError(
                "applied {} requires first_applied_state_id".format(self.status)
            )
        if self.endpoint_state_id is not None:
            raise PrimitiveExecutionProtocolError(
                "{} must not claim endpoint_state_id".format(self.status)
            )
        if self.endpoint_sim_time_ns is not None or self.endpoint_observation_ref is not None:
            raise PrimitiveExecutionProtocolError(
                "{} must not claim an endpoint observation".format(self.status)
            )
        if self.reason_code == "NONE":
            raise PrimitiveExecutionProtocolError(
                "{} reason_code must identify the interruption".format(self.status)
            )

    def _validate_terminal_observation_failure(self, expected_frame_count: int) -> None:
        """Validate a legal environment failure with an exact terminal ref."""

        if self.applied_frame_count >= expected_frame_count:
            raise PrimitiveExecutionProtocolError(
                "{} cannot claim a complete applied prefix".format(self.status)
            )
        if self.last_applied_frame_index != self.applied_frame_count - 1:
            raise PrimitiveExecutionProtocolError(
                "{} last_applied_frame_index must equal applied_frame_count - 1".format(
                    self.status
                )
            )
        if self.applied_frame_count == 0:
            if self.first_applied_state_id is not None:
                raise PrimitiveExecutionProtocolError(
                    "zero-applied {} must not claim first_applied_state_id".format(
                        self.status
                    )
                )
        elif self.first_applied_state_id is None or self.first_applied_state_id < 0:
            raise PrimitiveExecutionProtocolError(
                "applied {} requires first_applied_state_id".format(self.status)
            )
        if self.endpoint_state_id is None or self.endpoint_state_id < 0:
            raise PrimitiveExecutionProtocolError(
                "{} terminal endpoint_state_id is required".format(self.status)
            )
        if self.endpoint_sim_time_ns is None:
            raise PrimitiveExecutionProtocolError(
                "{} terminal endpoint_sim_time_ns is required".format(self.status)
            )
        observation_ref = self.endpoint_observation_ref
        if not isinstance(observation_ref, MappingABC):
            raise PrimitiveExecutionProtocolError(
                "{} terminal endpoint_observation_ref is required".format(self.status)
            )
        required_keys = {
            "schema_version",
            "runtime_instance_id",
            "episode_id",
            "reset_id",
            "state_id",
            "depth_id",
            "sim_time_ns",
        }
        if set(observation_ref.keys()) != required_keys:
            raise PrimitiveExecutionProtocolError(
                "{} terminal endpoint_observation_ref must contain the exact ObservationRef fields".format(
                    self.status
                )
            )
        if observation_ref["schema_version"] != SCHEMA_VERSION:
            raise SchemaMismatchError(
                "{} terminal endpoint observation schema_version mismatch".format(
                    self.status
                )
            )
        if observation_ref["runtime_instance_id"] != self.runtime_instance_id:
            raise PrimitiveExecutionProtocolError(
                "{} terminal observation runtime_instance_id mismatch".format(self.status)
            )
        for field in ("episode_id", "reset_id", "depth_id"):
            if not isinstance(observation_ref[field], str) or not observation_ref[field]:
                raise PrimitiveExecutionProtocolError(
                    "{} terminal observation {} is required".format(self.status, field)
                )
        if observation_ref["state_id"] != self.endpoint_state_id:
            raise PrimitiveExecutionProtocolError(
                "{} terminal observation state_id mismatch".format(self.status)
            )
        if observation_ref["sim_time_ns"] != self.endpoint_sim_time_ns:
            raise PrimitiveExecutionProtocolError(
                "{} terminal observation sim_time_ns mismatch".format(self.status)
            )


def _encode_v4_result_payload(result: PrimitiveExecutionResult) -> bytes:
    """Encode the result map without its self-referential payload hash."""

    writer = CanonicalMessagePackWriter(error_type=PrimitiveExecutionProtocolError)
    writer.map_header(15)

    # Keys are emitted in UTF-8 lexicographic order.  The value widths are
    # fixed by the v4 field table, not selected by the MessagePack library.
    writer.string("applied_frame_count")
    writer.uint32(result.applied_frame_count)
    writer.string("command_sequence_hash")
    writer.binary32(bytes.fromhex(require_digest(result.command_sequence_hash, name="command_sequence_hash")))
    writer.string("endpoint_observation_ref")
    if result.endpoint_observation_ref is None:
        writer.nil()
    else:
        observation = result.endpoint_observation_ref
        writer.map_header(7)
        writer.string("depth_id")
        writer.string(observation["depth_id"])
        writer.string("episode_id")
        writer.string(observation["episode_id"])
        writer.string("reset_id")
        writer.string(observation["reset_id"])
        writer.string("runtime_instance_id")
        writer.string(observation["runtime_instance_id"])
        writer.string("schema_version")
        writer.uint32(observation["schema_version"])
        writer.string("sim_time_ns")
        writer.uint64(observation["sim_time_ns"])
        writer.string("state_id")
        writer.int64(observation["state_id"])
    writer.string("endpoint_sim_time_ns")
    if result.endpoint_sim_time_ns is None:
        writer.nil()
    else:
        writer.uint64(result.endpoint_sim_time_ns)
    writer.string("endpoint_state_id")
    if result.endpoint_state_id is None:
        writer.nil()
    else:
        writer.int64(result.endpoint_state_id)
    writer.string("execution_id")
    writer.uint64(result.execution_id)
    writer.string("first_applied_state_id")
    if result.first_applied_state_id is None:
        writer.nil()
    else:
        writer.int64(result.first_applied_state_id)
    writer.string("last_applied_frame_index")
    writer.int32(result.last_applied_frame_index)
    writer.string("message_type")
    writer.string(result.message_type)
    writer.string("reason_code")
    writer.string(result.reason_code)
    writer.string("requested_frame_count")
    writer.uint32(result.requested_frame_count)
    writer.string("result_generation")
    writer.uint32(result.result_generation)
    writer.string("runtime_instance_id")
    writer.string(result.runtime_instance_id)
    writer.string("schema_version")
    writer.uint32(result.schema_version)
    writer.string("status")
    writer.string(result.status)
    return writer.bytes()


def canonical_v4_result_payload_bytes(result: Any) -> bytes:
    """Return canonical v4 result bytes, excluding result_payload_hash."""

    return _encode_v4_result_payload(_coerce_result(result))


def canonical_result_payload_hash(result: Any) -> str:
    """Return the v4 result identity hash over canonical typed bytes."""

    return hashlib.sha256(canonical_v4_result_payload_bytes(result)).hexdigest()


@dataclass(frozen=True)
class PrimitiveExecutionResultIdentity:
    """Deduplication key for one immutable physical execution result."""

    runtime_instance_id: str
    execution_id: int
    result_payload_hash: str


def _coerce_result(result: Any) -> PrimitiveExecutionResult:
    if isinstance(result, PrimitiveExecutionResult):
        return result.validate()
    if isinstance(result, MappingABC):
        return PrimitiveExecutionResult.from_mapping(result)
    raise PrimitiveExecutionProtocolError(
        "result must be a PrimitiveExecutionResult or mapping"
    )


def _hash_result_payload(result: PrimitiveExecutionResult) -> str:
    return hashlib.sha256(_encode_v4_result_payload(result)).hexdigest()


def canonical_result_payload(result: Any) -> Dict[str, Any]:
    """Return the validated semantic payload used for result hashing."""

    return _coerce_result(result).canonical_payload()


def result_identity(result: Any) -> PrimitiveExecutionResultIdentity:
    """Build the runtime/execution/hash identity used for deduplication."""

    validated = _coerce_result(result)
    return PrimitiveExecutionResultIdentity(
        runtime_instance_id=validated.runtime_instance_id,
        execution_id=validated.execution_id,
        result_payload_hash=_hash_result_payload(validated),
    )


def validate_execution_result(
    result: Any,
    *,
    expected_execution_id: Optional[int] = None,
    expected_command_sequence_hash: Optional[str] = None,
    expected_schema_version: int = SCHEMA_VERSION,
    expected_frame_count: int = DEFAULT_REQUESTED_FRAME_COUNT,
) -> PrimitiveExecutionResult:
    """Validate protocol input and return the immutable result model."""

    if isinstance(result, PrimitiveExecutionResult):
        return result.validate(
            expected_execution_id=expected_execution_id,
            expected_command_sequence_hash=expected_command_sequence_hash,
            expected_schema_version=expected_schema_version,
            expected_frame_count=expected_frame_count,
        )
    mapping = require_mapping(result, name="result")
    parsed = PrimitiveExecutionResult.from_mapping(mapping)
    return parsed.validate(
        expected_execution_id=expected_execution_id,
        expected_command_sequence_hash=expected_command_sequence_hash,
        expected_schema_version=expected_schema_version,
        expected_frame_count=expected_frame_count,
    )


class ResultIdentityRegistry:
    """Pure in-memory identity ledger for at-least-once result delivery.

    It models only the identity decision.  It does not ACK, retransmit,
    persist, or commit a transition.
    """

    def __init__(self) -> None:
        self._payload_hashes: Dict[Tuple[str, int], str] = {}

    def observe(self, result: Any) -> str:
        identity = result_identity(result)
        key = (identity.runtime_instance_id, identity.execution_id)
        previous_hash = self._payload_hashes.get(key)
        if previous_hash is None:
            self._payload_hashes[key] = identity.result_payload_hash
            return RESULT_IDENTITY_FIRST
        if previous_hash != identity.result_payload_hash:
            raise PrimitiveExecutionProtocolError(
                "result_payload_hash conflict for runtime_instance_id={} execution_id={}".format(
                    identity.runtime_instance_id, identity.execution_id
                )
            )
        return RESULT_IDENTITY_DUPLICATE


__all__ = [
    "DEFAULT_REQUESTED_FRAME_COUNT",
    "PrimitiveExecutionProtocolError",
    "PrimitiveExecutionResult",
    "PrimitiveExecutionResultIdentity",
    "RESULT_IDENTITY_DUPLICATE",
    "RESULT_IDENTITY_FIRST",
    "RESULT_MESSAGE_TYPE",
    "SCHEMA_VERSION",
    "STATUS_CANCELLED",
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "STATUS_REJECTED",
    "ResultIdentityRegistry",
    "SchemaMismatchError",
    "canonical_command_sequence_hash",
    "canonical_v4_command_sequence_bytes",
    "canonical_v4_command_sequence_hash",
    "canonical_messagepack_bytes",
    "canonical_result_payload",
    "canonical_result_payload_hash",
    "canonical_v4_result_payload_bytes",
    "result_identity",
    "validate_execution_result",
]
