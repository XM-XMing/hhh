"""Language-neutral golden vectors for v4 exact endpoint snapshots."""

import json
import hashlib
from pathlib import Path

import pytest

from planning.protocol.endpoint_observation_snapshot_v4 import (
    EndpointObservationSnapshot,
    ObservationRef,
    SnapshotAck,
    SnapshotRequest,
    canonical_endpoint_observation_snapshot_bytes,
    canonical_observation_ref_bytes,
    canonical_snapshot_ack_bytes,
    canonical_snapshot_hash,
    canonical_snapshot_request_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests/fixtures/observation_snapshot_v4"


def _fixture(name):
    return json.loads((FIXTURE_DIR / "{}.json".format(name)).read_text(encoding="utf-8"))


def _hex(name):
    return bytes.fromhex((FIXTURE_DIR / name).read_text(encoding="utf-8").strip())


def _observation_ref(payload):
    return ObservationRef.from_mapping(payload)


def _snapshot(payload):
    return EndpointObservationSnapshot.from_mapping(
        {
            "observation_ref": payload["observation_ref"],
            "state_bytes": bytes.fromhex(payload["state_bytes_hex"]),
            "depth_bytes": bytes.fromhex(payload["depth_bytes_hex"]),
            "snapshot_hash": payload["snapshot_hash"],
        }
    )


@pytest.mark.unit
def test_observation_snapshot_v4_golden_bytes_and_hashes_are_language_neutral():
    ref_fixture = _fixture("observation_ref")
    snapshot_fixture = _fixture("snapshot")
    request_fixture = _fixture("snapshot_request")
    ack_fixture = _fixture("snapshot_ack")

    observation_ref = _observation_ref(ref_fixture["observation_ref"])
    snapshot = _snapshot(snapshot_fixture)
    request = SnapshotRequest.from_mapping(request_fixture["snapshot_request"])
    ack = SnapshotAck.from_mapping(ack_fixture["snapshot_ack"])

    ref_bytes = canonical_observation_ref_bytes(observation_ref)
    snapshot_bytes = canonical_endpoint_observation_snapshot_bytes(snapshot)
    request_bytes = canonical_snapshot_request_bytes(request)
    ack_bytes = canonical_snapshot_ack_bytes(ack)

    assert ref_bytes == _hex(ref_fixture["msgpack_hex_file"])
    assert snapshot_bytes == _hex(snapshot_fixture["msgpack_hex_file"])
    assert request_bytes == _hex(request_fixture["msgpack_hex_file"])
    assert ack_bytes == _hex(ack_fixture["msgpack_hex_file"])
    assert hashlib.sha256(ref_bytes).hexdigest() == ref_fixture["payload_sha256"]
    assert hashlib.sha256(snapshot_bytes).hexdigest() == snapshot_fixture["snapshot_hash"]
    assert hashlib.sha256(request_bytes).hexdigest() == request_fixture["payload_sha256"]
    assert hashlib.sha256(ack_bytes).hexdigest() == ack_fixture["payload_sha256"]
    assert canonical_snapshot_hash(snapshot) == snapshot_fixture["snapshot_hash"]


@pytest.mark.unit
def test_snapshot_golden_bytes_and_hash_are_stable_across_100_repetitions():
    fixture = _fixture("snapshot")
    snapshot = _snapshot(fixture)
    expected_bytes = _hex(fixture["msgpack_hex_file"])

    for _ in range(100):
        assert canonical_endpoint_observation_snapshot_bytes(snapshot) == expected_bytes
        assert canonical_snapshot_hash(snapshot) == fixture["snapshot_hash"]
