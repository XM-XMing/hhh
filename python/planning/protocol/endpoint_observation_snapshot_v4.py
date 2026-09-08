"""Pure schema-v4 models for exact reliable endpoint snapshot retrieval.

The public seam is deliberately limited to protocol input, canonical bytes,
snapshot identity, and duplicate/conflict decisions.  It does not open a
socket, retain a runtime cache, consume PUB/SUB telemetry, or commit a
transition.

``EndpointObservationSnapshot`` hashes its immutable semantic payload without
the self-referential ``snapshot_hash`` field.  A snapshot identity is the full
``ObservationRef``; a different hash for that same ref is a protocol error.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, Mapping, Tuple

from planning.protocol.primitive_execution_schema_v4 import (
    SCHEMA_VERSION,
    PrimitiveExecutionProtocolError,
    SchemaMismatchError,
)
from planning.protocol.msgpack import CanonicalMessagePackWriter
from planning.protocol.validation import (
    require_bytes,
    require_digest,
    require_field,
    require_mapping,
    require_non_empty_text,
    require_uint64,
)


SNAPSHOT_IDENTITY_FIRST = "FIRST"
SNAPSHOT_IDENTITY_DUPLICATE = "DUPLICATE"


def _int64_non_negative(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0x7FFFFFFFFFFFFFFF:
        raise PrimitiveExecutionProtocolError("{} must be a non-negative int64".format(name))
    return value


@dataclass(frozen=True)
class ObservationRef:
    """Exact endpoint identity; no temporal or latest-observation lookup exists."""

    schema_version: int
    runtime_instance_id: str
    episode_id: str
    reset_id: str
    state_id: int
    depth_id: str
    sim_time_ns: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ObservationRef":
        mapping = require_mapping(payload, name="observation_ref")
        schema_version = require_field(mapping, "schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise PrimitiveExecutionProtocolError("schema_version must be uint32")
        if schema_version != SCHEMA_VERSION:
            raise SchemaMismatchError(
                "schema_version mismatch: received={} expected={}".format(
                    schema_version, SCHEMA_VERSION
                )
            )
        return cls(
            schema_version=schema_version,
            runtime_instance_id=require_non_empty_text(
                require_field(mapping, "runtime_instance_id"), name="runtime_instance_id"
            ),
            episode_id=require_non_empty_text(require_field(mapping, "episode_id"), name="episode_id"),
            reset_id=require_non_empty_text(require_field(mapping, "reset_id"), name="reset_id"),
            state_id=_int64_non_negative(require_field(mapping, "state_id"), name="state_id"),
            depth_id=require_non_empty_text(require_field(mapping, "depth_id"), name="depth_id"),
            sim_time_ns=require_uint64(require_field(mapping, "sim_time_ns"), name="sim_time_ns"),
        )

    def canonical_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_instance_id": self.runtime_instance_id,
            "episode_id": self.episode_id,
            "reset_id": self.reset_id,
            "state_id": self.state_id,
            "depth_id": self.depth_id,
            "sim_time_ns": self.sim_time_ns,
        }


def _append_observation_ref(
    writer: CanonicalMessagePackWriter, observation_ref: ObservationRef
) -> None:
    writer.map_header(7)
    writer.string("depth_id")
    writer.string(observation_ref.depth_id)
    writer.string("episode_id")
    writer.string(observation_ref.episode_id)
    writer.string("reset_id")
    writer.string(observation_ref.reset_id)
    writer.string("runtime_instance_id")
    writer.string(observation_ref.runtime_instance_id)
    writer.string("schema_version")
    writer.uint32(observation_ref.schema_version)
    writer.string("sim_time_ns")
    writer.uint64(observation_ref.sim_time_ns)
    writer.string("state_id")
    writer.int64(observation_ref.state_id)


def canonical_observation_ref_bytes(observation_ref: Any) -> bytes:
    parsed = _coerce_observation_ref(observation_ref)
    writer = CanonicalMessagePackWriter(error_type=PrimitiveExecutionProtocolError)
    _append_observation_ref(writer, parsed)
    return writer.bytes()


@dataclass(frozen=True)
class EndpointObservationSnapshot:
    """Immutable exact state/depth pair, independently retrievable by ref."""

    observation_ref: ObservationRef
    state_bytes: bytes
    depth_bytes: bytes
    snapshot_hash: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EndpointObservationSnapshot":
        mapping = require_mapping(payload, name="snapshot")
        observation_ref = ObservationRef.from_mapping(
            require_mapping(require_field(mapping, "observation_ref"), name="observation_ref")
        )
        state_bytes = require_bytes(require_field(mapping, "state_bytes"), name="state_bytes")
        depth_bytes = require_bytes(require_field(mapping, "depth_bytes"), name="depth_bytes")
        provisional = cls(
            observation_ref=observation_ref,
            state_bytes=state_bytes,
            depth_bytes=depth_bytes,
            snapshot_hash="",
        )
        actual_hash = _snapshot_hash(provisional)
        supplied_hash = mapping.get("snapshot_hash")
        if supplied_hash is not None and require_digest(supplied_hash, name="snapshot_hash") != actual_hash:
            raise PrimitiveExecutionProtocolError(
                "snapshot_hash does not match the semantic snapshot payload"
            )
        return cls(
            observation_ref=observation_ref,
            state_bytes=state_bytes,
            depth_bytes=depth_bytes,
            snapshot_hash=actual_hash,
        )

    def canonical_payload(self) -> Dict[str, Any]:
        return {
            "observation_ref": self.observation_ref.canonical_payload(),
            "state_bytes": self.state_bytes,
            "depth_bytes": self.depth_bytes,
        }


def _encode_snapshot(snapshot: EndpointObservationSnapshot) -> bytes:
    writer = CanonicalMessagePackWriter(error_type=PrimitiveExecutionProtocolError)
    writer.map_header(3)
    writer.string("depth_bytes")
    writer.binary(snapshot.depth_bytes)
    writer.string("observation_ref")
    _append_observation_ref(writer, snapshot.observation_ref)
    writer.string("state_bytes")
    writer.binary(snapshot.state_bytes)
    return writer.bytes()


def canonical_endpoint_observation_snapshot_bytes(snapshot: Any) -> bytes:
    return _encode_snapshot(_coerce_snapshot(snapshot))


def _snapshot_hash(snapshot: EndpointObservationSnapshot) -> str:
    return hashlib.sha256(_encode_snapshot(snapshot)).hexdigest()


def canonical_snapshot_hash(snapshot: Any) -> str:
    """SHA-256 of canonical snapshot bytes, excluding snapshot_hash itself."""

    return _snapshot_hash(_coerce_snapshot(snapshot))


@dataclass(frozen=True)
class SnapshotRequest:
    """Exact snapshot request bound to one immutable execution result."""

    observation_ref: ObservationRef
    execution_id: int
    result_payload_hash: str
    command_sequence_hash: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SnapshotRequest":
        mapping = require_mapping(payload, name="snapshot_request")
        return cls(
            observation_ref=ObservationRef.from_mapping(
                require_mapping(require_field(mapping, "observation_ref"), name="observation_ref")
            ),
            execution_id=require_uint64(require_field(mapping, "execution_id"), name="execution_id"),
            result_payload_hash=require_digest(
                require_field(mapping, "result_payload_hash"), name="result_payload_hash"
            ),
            command_sequence_hash=require_digest(
                require_field(mapping, "command_sequence_hash"), name="command_sequence_hash"
            ),
        )


def canonical_snapshot_request_bytes(request: Any) -> bytes:
    parsed = _coerce_request(request)
    writer = CanonicalMessagePackWriter(error_type=PrimitiveExecutionProtocolError)
    writer.map_header(4)
    writer.string("command_sequence_hash")
    writer.binary32(
        bytes.fromhex(
            require_digest(parsed.command_sequence_hash, name="command_sequence_hash")
        )
    )
    writer.string("execution_id")
    writer.uint64(parsed.execution_id)
    writer.string("observation_ref")
    _append_observation_ref(writer, parsed.observation_ref)
    writer.string("result_payload_hash")
    writer.binary32(
        bytes.fromhex(
            require_digest(parsed.result_payload_hash, name="result_payload_hash")
        )
    )
    return writer.bytes()


@dataclass(frozen=True)
class SnapshotAck:
    """Durable receipt acknowledgement for one immutable snapshot."""

    observation_ref: ObservationRef
    snapshot_hash: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SnapshotAck":
        mapping = require_mapping(payload, name="snapshot_ack")
        return cls(
            observation_ref=ObservationRef.from_mapping(
                require_mapping(require_field(mapping, "observation_ref"), name="observation_ref")
            ),
            snapshot_hash=require_digest(require_field(mapping, "snapshot_hash"), name="snapshot_hash"),
        )


def canonical_snapshot_ack_bytes(ack: Any) -> bytes:
    parsed = _coerce_ack(ack)
    writer = CanonicalMessagePackWriter(error_type=PrimitiveExecutionProtocolError)
    writer.map_header(2)
    writer.string("observation_ref")
    _append_observation_ref(writer, parsed.observation_ref)
    writer.string("snapshot_hash")
    writer.binary32(
        bytes.fromhex(require_digest(parsed.snapshot_hash, name="snapshot_hash"))
    )
    return writer.bytes()


def _coerce_observation_ref(value: Any) -> ObservationRef:
    if isinstance(value, ObservationRef):
        return value
    return ObservationRef.from_mapping(require_mapping(value, name="observation_ref"))


def _coerce_snapshot(value: Any) -> EndpointObservationSnapshot:
    if isinstance(value, EndpointObservationSnapshot):
        return value
    return EndpointObservationSnapshot.from_mapping(require_mapping(value, name="snapshot"))


def _coerce_request(value: Any) -> SnapshotRequest:
    if isinstance(value, SnapshotRequest):
        return value
    return SnapshotRequest.from_mapping(require_mapping(value, name="snapshot_request"))


def _coerce_ack(value: Any) -> SnapshotAck:
    if isinstance(value, SnapshotAck):
        return value
    return SnapshotAck.from_mapping(require_mapping(value, name="snapshot_ack"))


class ObservationSnapshotIdentityRegistry:
    """Pure identity ledger for at-least-once immutable snapshot delivery."""

    def __init__(self) -> None:
        self._snapshot_hashes: Dict[Tuple[Any, ...], str] = {}

    def observe(self, snapshot: Any) -> str:
        parsed = _coerce_snapshot(snapshot)
        key = (
            parsed.observation_ref.schema_version,
            parsed.observation_ref.runtime_instance_id,
            parsed.observation_ref.episode_id,
            parsed.observation_ref.reset_id,
            parsed.observation_ref.state_id,
            parsed.observation_ref.depth_id,
            parsed.observation_ref.sim_time_ns,
        )
        previous_hash = self._snapshot_hashes.get(key)
        if previous_hash is None:
            self._snapshot_hashes[key] = parsed.snapshot_hash
            return SNAPSHOT_IDENTITY_FIRST
        if previous_hash != parsed.snapshot_hash:
            raise PrimitiveExecutionProtocolError(
                "snapshot_hash conflict for observation_ref={}".format(key)
            )
        return SNAPSHOT_IDENTITY_DUPLICATE


__all__ = [
    "EndpointObservationSnapshot",
    "ObservationRef",
    "ObservationSnapshotIdentityRegistry",
    "PrimitiveExecutionProtocolError",
    "SNAPSHOT_IDENTITY_DUPLICATE",
    "SNAPSHOT_IDENTITY_FIRST",
    "SnapshotAck",
    "SnapshotRequest",
    "canonical_endpoint_observation_snapshot_bytes",
    "canonical_observation_ref_bytes",
    "canonical_snapshot_ack_bytes",
    "canonical_snapshot_hash",
    "canonical_snapshot_request_bytes",
]
