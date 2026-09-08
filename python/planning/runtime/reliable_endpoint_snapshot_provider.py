"""Python P0-O5 adapter for exact bridge-owned endpoint snapshots.

The adapter is deliberately transport-free.  Its injected retriever is the
future bridge exposure boundary; it never reads or reconstructs state/depth
from PUB/SUB telemetry.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Union

import zmq

from planning.protocol.constants import PROTOCOL_VERSION
from planning.protocol.msgpack import messagepack_unpack
from planning.protocol.snapshot import build_snapshot_request_wire

from planning.protocol.endpoint_observation_snapshot_v4 import (
    EndpointObservationSnapshot,
    ObservationRef,
    ObservationSnapshotIdentityRegistry,
    PrimitiveExecutionProtocolError,
    SnapshotRequest,
    canonical_snapshot_hash,
)
from planning.protocol.primitive_execution_result_consumer import EndpointObservation
from planning.protocol.primitive_execution_schema_v4 import PrimitiveExecutionResult


SnapshotInput = Union[EndpointObservationSnapshot, Mapping[str, Any]]


class BridgeSnapshotRetriever(Protocol):
    """Returns only the immutable snapshot for one exact request, or ``None``."""

    def retrieve(
        self, request: SnapshotRequest
    ) -> Optional[SnapshotInput]:
        ...


class BridgeSnapshotEndpointProvider:
    """Adapts bridge snapshot retrieval to the consumer endpoint-provider seam."""

    def __init__(self, retriever: BridgeSnapshotRetriever) -> None:
        self._retriever = retriever
        self._identities = ObservationSnapshotIdentityRegistry()

    def lookup(
        self, result: PrimitiveExecutionResult
    ) -> Optional[EndpointObservation]:
        observation_ref = self._observation_ref_for(result)
        request = SnapshotRequest(
            observation_ref=observation_ref,
            execution_id=result.execution_id,
            result_payload_hash=result.result_payload_hash,
            command_sequence_hash=result.command_sequence_hash,
        )
        supplied = self._retriever.retrieve(request)
        if supplied is None:
            return None
        snapshot = self._coerce_snapshot(supplied)
        self._validate_snapshot(request, snapshot)
        self._identities.observe(snapshot)
        return EndpointObservation(
            runtime_instance_id=observation_ref.runtime_instance_id,
            execution_id=result.execution_id,
            state_id=observation_ref.state_id,
            sim_time_ns=observation_ref.sim_time_ns,
            state=snapshot.state_bytes,
            depth=snapshot.depth_bytes,
            episode_id=observation_ref.episode_id,
            reset_id=observation_ref.reset_id,
            depth_id=observation_ref.depth_id,
            physics_time_ns=observation_ref.sim_time_ns,
            snapshot_hash=snapshot.snapshot_hash,
        )

    def lookup_reset(
        self,
        observation_ref: Mapping[str, Any],
        *,
        runtime_instance_id: str,
        episode_id: str,
        reset_id: str,
    ) -> Optional[EndpointObservation]:
        """Retrieve an exact reset snapshot through the same bridge cache.

        Reset has no primitive result identity, so execution/hash fields use
        the reserved zero identity.  The observation reference remains the
        authoritative key and is validated exactly like a primitive endpoint.
        """

        ref = ObservationRef.from_mapping(observation_ref)
        if (
            ref.runtime_instance_id != runtime_instance_id
            or ref.episode_id != episode_id
            or ref.reset_id != reset_id
        ):
            raise PrimitiveExecutionProtocolError(
                "reset ObservationRef identity does not match reset completion"
            )
        request = SnapshotRequest(
            observation_ref=ref,
            execution_id=0,
            result_payload_hash="0" * 64,
            command_sequence_hash="0" * 64,
        )
        supplied = self._retriever.retrieve(request)
        if supplied is None:
            return None
        snapshot = self._coerce_snapshot(supplied)
        self._validate_snapshot(request, snapshot)
        self._identities.observe(snapshot)
        return EndpointObservation(
            runtime_instance_id=ref.runtime_instance_id,
            execution_id=0,
            state_id=ref.state_id,
            sim_time_ns=ref.sim_time_ns,
            state=snapshot.state_bytes,
            depth=snapshot.depth_bytes,
            episode_id=ref.episode_id,
            reset_id=ref.reset_id,
            depth_id=ref.depth_id,
            physics_time_ns=ref.sim_time_ns,
            snapshot_hash=snapshot.snapshot_hash,
        )

    @staticmethod
    def _coerce_snapshot(value: SnapshotInput) -> EndpointObservationSnapshot:
        if isinstance(value, EndpointObservationSnapshot):
            return value
        if not isinstance(value, Mapping):
            raise PrimitiveExecutionProtocolError("bridge snapshot must be a mapping")
        return EndpointObservationSnapshot.from_mapping(value)

    @staticmethod
    def _observation_ref_for(result: PrimitiveExecutionResult) -> ObservationRef:
        if result.endpoint_observation_ref is None:
            raise PrimitiveExecutionProtocolError(
                "{} result is missing endpoint_observation_ref".format(result.status)
            )
        observation_ref = ObservationRef.from_mapping(result.endpoint_observation_ref)
        expected = (
            result.runtime_instance_id,
            result.endpoint_state_id,
            result.endpoint_sim_time_ns,
        )
        actual = (
            observation_ref.runtime_instance_id,
            observation_ref.state_id,
            observation_ref.sim_time_ns,
        )
        if actual != expected:
            raise PrimitiveExecutionProtocolError(
                "ObservationRef does not exactly match result endpoint: "
                "received={} expected={}".format(actual, expected)
            )
        return observation_ref

    @staticmethod
    def _validate_snapshot(
        request: SnapshotRequest, snapshot: EndpointObservationSnapshot
    ) -> None:
        if snapshot.observation_ref != request.observation_ref:
            raise PrimitiveExecutionProtocolError(
                "snapshot ObservationRef does not exactly match SnapshotRequest"
            )
        if canonical_snapshot_hash(snapshot) != snapshot.snapshot_hash:
            raise PrimitiveExecutionProtocolError(
                "snapshot_hash does not match immutable snapshot payload"
            )


class ZmqBridgeSnapshotRetriever:
    """Independent Python client for the bridge's immutable snapshot cache."""

    def __init__(self, context: zmq.Context, endpoint: str, *, timeout_ms: int) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self._socket = context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._timeout_ms = int(timeout_ms)

    def close(self) -> None:
        self._socket.close(0)

    def retrieve(self, request: SnapshotRequest) -> Optional[EndpointObservationSnapshot]:
        self._socket.send(self._serialize_request(request))
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        if self._socket not in dict(poller.poll(self._timeout_ms)):
            return None
        return self._parse_response(self._socket.recv(), request)

    @staticmethod
    def _serialize_request(request: SnapshotRequest) -> bytes:
        return build_snapshot_request_wire(request)

    @staticmethod
    def _parse_response(
        raw: bytes, request: SnapshotRequest
    ) -> Optional[EndpointObservationSnapshot]:
        try:
            payload = messagepack_unpack(raw)
        except Exception as exc:
            raise PrimitiveExecutionProtocolError(
                "bridge snapshot response is not MessagePack: {}".format(exc)
            )
        if not isinstance(payload, Mapping) or payload.get("schema_version") != PROTOCOL_VERSION:
            raise PrimitiveExecutionProtocolError("bridge snapshot response schema mismatch")
        ZmqBridgeSnapshotRetriever._validate_response_request(payload, request)
        message_type = payload.get("message_type")
        if message_type == "BridgeSnapshotMissing":
            return None
        if message_type != "BridgeSnapshotResponse":
            raise PrimitiveExecutionProtocolError("bridge snapshot response type mismatch")
        snapshot_hash = payload.get("snapshot_hash")
        if not isinstance(snapshot_hash, bytes) or len(snapshot_hash) != 32:
            raise PrimitiveExecutionProtocolError("bridge snapshot_hash must be bytes32")
        return EndpointObservationSnapshot.from_mapping(
            {
                "observation_ref": payload.get("observation_ref"),
                "state_bytes": payload.get("state_bytes"),
                "depth_bytes": payload.get("depth_bytes"),
                "snapshot_hash": snapshot_hash.hex(),
            }
        )

    @staticmethod
    def _validate_response_request(payload: Mapping[str, Any], request: SnapshotRequest) -> None:
        try:
            response_request = SnapshotRequest.from_mapping(
                {
                    "observation_ref": payload.get("observation_ref"),
                    "execution_id": payload.get("execution_id"),
                    "result_payload_hash": ZmqBridgeSnapshotRetriever._digest_hex(
                        payload.get("result_payload_hash")
                    ),
                    "command_sequence_hash": ZmqBridgeSnapshotRetriever._digest_hex(
                        payload.get("command_sequence_hash")
                    ),
                }
            )
        except PrimitiveExecutionProtocolError:
            raise
        except Exception as exc:
            raise PrimitiveExecutionProtocolError(
                "bridge snapshot response identity is invalid: {}".format(exc)
            )
        if response_request != request:
            raise PrimitiveExecutionProtocolError(
                "bridge snapshot response does not exactly match SnapshotRequest"
            )

    @staticmethod
    def _digest_hex(value: Any) -> str:
        if not isinstance(value, bytes) or len(value) != 32:
            raise PrimitiveExecutionProtocolError("bridge response digest must be bytes32")
        return value.hex()


__all__ = [
    "BridgeSnapshotEndpointProvider",
    "BridgeSnapshotRetriever",
    "ZmqBridgeSnapshotRetriever",
]
