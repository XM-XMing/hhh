"""Canonical endpoint-snapshot protocol owner."""

from planning.protocol.constants import PROTOCOL_VERSION
from planning.protocol.msgpack import messagepack_bytes

from planning.protocol.endpoint_observation_snapshot_v4 import (
    EndpointObservationSnapshot,
    ObservationRef,
    ObservationSnapshotIdentityRegistry,
    SnapshotAck,
    SnapshotRequest,
    canonical_endpoint_observation_snapshot_bytes,
    canonical_observation_ref_bytes,
    canonical_snapshot_ack_bytes,
    canonical_snapshot_hash,
    canonical_snapshot_request_bytes,
)


def build_snapshot_request_wire(request: SnapshotRequest) -> bytes:
    """Encode the bridge request using the existing ordinary MessagePack order."""

    return messagepack_bytes(
        {
            "schema_version": PROTOCOL_VERSION,
            "message_type": "BridgeSnapshotRequest",
            "observation_ref": request.observation_ref.canonical_payload(),
            "execution_id": request.execution_id,
            "result_payload_hash": bytes.fromhex(request.result_payload_hash),
            "command_sequence_hash": bytes.fromhex(request.command_sequence_hash),
        }
    )

__all__ = [
    "EndpointObservationSnapshot",
    "ObservationRef",
    "ObservationSnapshotIdentityRegistry",
    "SnapshotAck",
    "SnapshotRequest",
    "canonical_endpoint_observation_snapshot_bytes",
    "canonical_observation_ref_bytes",
    "canonical_snapshot_ack_bytes",
    "canonical_snapshot_hash",
    "canonical_snapshot_request_bytes",
    "build_snapshot_request_wire",
]
