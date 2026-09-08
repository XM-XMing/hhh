"""Public protocol seam tests for reliable v4 endpoint snapshots.

These tests cover only semantic protocol mappings, canonical bytes, snapshot
identity, and duplicate/conflict outcomes.  They intentionally do not open a
socket or exercise Unity, bridge, telemetry, or the transition consumer.
"""

from collections import OrderedDict

import pytest

from planning.protocol.endpoint_observation_snapshot_v4 import (
    EndpointObservationSnapshot,
    ObservationRef,
    ObservationSnapshotIdentityRegistry,
    PrimitiveExecutionProtocolError,
    SnapshotAck,
    SnapshotRequest,
    canonical_endpoint_observation_snapshot_bytes,
    canonical_snapshot_ack_bytes,
    canonical_snapshot_hash,
    canonical_snapshot_request_bytes,
)


RUNTIME_INSTANCE_ID = "worker-00-runtime-test"
EXECUTION_ID = 4_400_000_000_0054
STATE_ID = 1514
SIM_TIME_NS = 559_999_987
DEPTH_ID = "depth-1514"
RESULT_HASH = "a" * 64
COMMAND_HASH = "b" * 64


def _observation_ref(**overrides):
    payload = {
        "schema_version": 4,
        "runtime_instance_id": RUNTIME_INSTANCE_ID,
        "episode_id": "episode-54",
        "reset_id": "reset-54",
        "state_id": STATE_ID,
        "depth_id": DEPTH_ID,
        "sim_time_ns": SIM_TIME_NS,
    }
    payload.update(overrides)
    return payload


def _snapshot(**overrides):
    payload = {
        "observation_ref": _observation_ref(),
        "state_bytes": b"state-1514",
        "depth_bytes": b"depth-1514",
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_snapshot_has_stable_canonical_bytes_and_hash_for_reordered_semantics():
    first = EndpointObservationSnapshot.from_mapping(_snapshot())
    reordered = EndpointObservationSnapshot.from_mapping(
        OrderedDict(
            [
                ("depth_bytes", b"depth-1514"),
                (
                    "observation_ref",
                    OrderedDict(
                        [
                            ("depth_id", DEPTH_ID),
                            ("episode_id", "episode-54"),
                            ("reset_id", "reset-54"),
                            ("runtime_instance_id", RUNTIME_INSTANCE_ID),
                            ("schema_version", 4),
                            ("sim_time_ns", SIM_TIME_NS),
                            ("state_id", STATE_ID),
                        ]
                    ),
                ),
                ("state_bytes", b"state-1514"),
            ]
        )
    )

    first_bytes = canonical_endpoint_observation_snapshot_bytes(first)
    assert first_bytes == canonical_endpoint_observation_snapshot_bytes(reordered)
    assert canonical_snapshot_hash(first) == canonical_snapshot_hash(reordered)
    assert [canonical_snapshot_hash(first) for _ in range(100)] == [
        canonical_snapshot_hash(first)
    ] * 100


@pytest.mark.unit
def test_snapshot_rejects_a_supplied_hash_that_does_not_match_its_exact_bytes():
    with pytest.raises(PrimitiveExecutionProtocolError, match="snapshot_hash"):
        EndpointObservationSnapshot.from_mapping(
            _snapshot(snapshot_hash="0" * 64)
        )


@pytest.mark.unit
def test_request_and_ack_bind_the_exact_observation_ref_and_execution_identity():
    snapshot = EndpointObservationSnapshot.from_mapping(_snapshot())
    request = SnapshotRequest.from_mapping(
        {
            "observation_ref": _observation_ref(),
            "execution_id": EXECUTION_ID,
            "result_payload_hash": RESULT_HASH,
            "command_sequence_hash": COMMAND_HASH,
        }
    )
    ack = SnapshotAck.from_mapping(
        {
            "observation_ref": _observation_ref(),
            "snapshot_hash": canonical_snapshot_hash(snapshot),
        }
    )

    assert request.observation_ref == snapshot.observation_ref
    assert request.execution_id == EXECUTION_ID
    assert ack.observation_ref == snapshot.observation_ref
    assert ack.snapshot_hash == canonical_snapshot_hash(snapshot)
    assert canonical_snapshot_request_bytes(request) == canonical_snapshot_request_bytes(
        OrderedDict(
            [
                ("command_sequence_hash", COMMAND_HASH),
                ("result_payload_hash", RESULT_HASH),
                ("execution_id", EXECUTION_ID),
                ("observation_ref", _observation_ref()),
            ]
        )
    )
    assert canonical_snapshot_ack_bytes(ack) == canonical_snapshot_ack_bytes(
        OrderedDict(
            [
                ("snapshot_hash", canonical_snapshot_hash(snapshot)),
                ("observation_ref", _observation_ref()),
            ]
        )
    )


@pytest.mark.unit
def test_duplicate_snapshot_with_the_same_exact_identity_and_hash_is_accepted():
    registry = ObservationSnapshotIdentityRegistry()
    snapshot = EndpointObservationSnapshot.from_mapping(_snapshot())

    assert registry.observe(snapshot) == "FIRST"
    assert registry.observe(snapshot) == "DUPLICATE"


@pytest.mark.unit
def test_same_observation_identity_with_different_snapshot_hash_is_protocol_error():
    registry = ObservationSnapshotIdentityRegistry()
    first = EndpointObservationSnapshot.from_mapping(_snapshot())
    conflicting = EndpointObservationSnapshot.from_mapping(
        _snapshot(depth_bytes=b"different-depth-bytes")
    )

    assert registry.observe(first) == "FIRST"
    with pytest.raises(PrimitiveExecutionProtocolError, match="snapshot_hash conflict"):
        registry.observe(conflicting)


@pytest.mark.unit
def test_observation_ref_schema_mismatch_fails_before_identity_use():
    with pytest.raises(PrimitiveExecutionProtocolError, match="schema_version"):
        ObservationRef.from_mapping(_observation_ref(schema_version=3))
