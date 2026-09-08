"""P0-O5 public seam: v4 result -> exact bridge snapshot -> accounting.

The bridge retriever is injected: this module intentionally opens no socket and
does not read state/depth PUB telemetry.  The consumer remains the public
exactly-once transition/replay seam.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from planning.protocol.endpoint_observation_snapshot_v4 import (
    EndpointObservationSnapshot,
    PrimitiveExecutionProtocolError,
)
from planning.protocol.primitive_execution_result_consumer import (
    PrimitiveExecutionResultConsumer,
)
from planning.protocol.primitive_execution_schema_v4 import validate_execution_result
from planning.runtime.reliable_endpoint_snapshot_provider import (
    BridgeSnapshotEndpointProvider,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "primitive_execution_v4"
    / "complete.json"
)


class SequenceBridgeSnapshotRetriever:
    def __init__(self, replies):
        self._replies = list(replies)
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        if not self._replies:
            return None
        return self._replies.pop(0)


class TransitionCommitter:
    def __init__(self):
        self.transitions = []

    def commit(self, transition):
        self.transitions.append(transition)


class ReplayAppender:
    def __init__(self):
        self.transitions = []

    def append_once(self, transition):
        self.transitions.append(transition)


def _result():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["result"]


def _snapshot(result, *, state_bytes=b"exact-state", depth_bytes=b"exact-depth"):
    return EndpointObservationSnapshot.from_mapping(
        {
            "observation_ref": result["endpoint_observation_ref"],
            "state_bytes": state_bytes,
            "depth_bytes": depth_bytes,
        }
    )


def _consumer(retriever):
    committer = TransitionCommitter()
    replay = ReplayAppender()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=BridgeSnapshotEndpointProvider(retriever),
        transition_committer=committer,
        replay_appender=replay,
    )
    return consumer, committer, replay


@pytest.mark.unit
def test_complete_commits_from_exact_bridge_snapshot_without_telemetry():
    result = _result()
    retriever = SequenceBridgeSnapshotRetriever([_snapshot(result)])
    consumer, committer, replay = _consumer(retriever)

    outcome = consumer.consume(result)

    assert outcome.status == "COMMITTED"
    assert len(retriever.requests) == 1
    assert retriever.requests[0].observation_ref.state_id == result["endpoint_state_id"]
    assert retriever.requests[0].observation_ref.depth_id == result["endpoint_observation_ref"]["depth_id"]
    assert committer.transitions[0].endpoint.state == b"exact-state"
    assert committer.transitions[0].endpoint.depth == b"exact-depth"
    assert len(replay.transitions) == 1


@pytest.mark.unit
def test_missing_snapshot_does_not_commit():
    consumer, committer, replay = _consumer(SequenceBridgeSnapshotRetriever([None]))

    outcome = consumer.consume(_result())

    assert outcome.status == "ENDPOINT_UNAVAILABLE"
    assert committer.transitions == []
    assert replay.transitions == []


@pytest.mark.unit
def test_wrong_snapshot_hash_is_protocol_error_without_commit():
    result = _result()
    snapshot = _snapshot(result)
    bad_hash_snapshot = EndpointObservationSnapshot(
        observation_ref=snapshot.observation_ref,
        state_bytes=snapshot.state_bytes,
        depth_bytes=snapshot.depth_bytes,
        snapshot_hash="0" * 64,
    )
    consumer, committer, replay = _consumer(
        SequenceBridgeSnapshotRetriever([bad_hash_snapshot])
    )

    outcome = consumer.consume(result)

    assert outcome.status == "PROTOCOL_ERROR"
    assert committer.transitions == []
    assert replay.transitions == []


@pytest.mark.unit
def test_wrong_observation_ref_is_protocol_error_without_commit():
    result = _result()
    wrong = dict(result["endpoint_observation_ref"], depth_id="depth-wrong")
    snapshot = EndpointObservationSnapshot.from_mapping(
        {
            "observation_ref": wrong,
            "state_bytes": b"exact-state",
            "depth_bytes": b"exact-depth",
        }
    )
    consumer, committer, replay = _consumer(SequenceBridgeSnapshotRetriever([snapshot]))

    outcome = consumer.consume(result)

    assert outcome.status == "PROTOCOL_ERROR"
    assert committer.transitions == []
    assert replay.transitions == []


@pytest.mark.unit
def test_duplicate_complete_has_one_transition_and_replay_append():
    result = _result()
    retriever = SequenceBridgeSnapshotRetriever([_snapshot(result)])
    consumer, committer, replay = _consumer(retriever)

    first = consumer.consume(result)
    duplicates = [consumer.consume(result) for _ in range(3)]

    assert first.status == "COMMITTED"
    assert [outcome.status for outcome in duplicates] == ["DUPLICATE"] * 3
    assert len(retriever.requests) == 1
    assert len(committer.transitions) == 1
    assert len(replay.transitions) == 1


@pytest.mark.unit
def test_temporary_snapshot_unavailable_retries_same_result_and_commits_once():
    result = _result()
    retriever = SequenceBridgeSnapshotRetriever([None, _snapshot(result)])
    consumer, committer, replay = _consumer(retriever)

    unavailable = consumer.consume(result)
    committed = consumer.consume(result)

    assert unavailable.status == "ENDPOINT_UNAVAILABLE"
    assert committed.status == "COMMITTED"
    assert len(retriever.requests) == 2
    assert len(committer.transitions) == 1
    assert len(replay.transitions) == 1


@pytest.mark.unit
def test_conflicting_snapshot_for_same_ref_is_protocol_error():
    result = _result()
    first = _snapshot(result, state_bytes=b"state-a")
    conflicting = _snapshot(result, state_bytes=b"state-b")
    provider = BridgeSnapshotEndpointProvider(
        SequenceBridgeSnapshotRetriever([first, conflicting])
    )

    validated = validate_execution_result(result)
    provider.lookup(validated)
    with pytest.raises(PrimitiveExecutionProtocolError, match="snapshot_hash conflict"):
        provider.lookup(validated)
