"""P0-L4A public-seam tests for Python result accounting.

The test double boundary is deliberately small:

    PrimitiveExecutionResult -> consumer -> endpoint provider
                                      -> transition committer
                                      -> replay appender

No ROS, sockets, Unity, or RL implementation is involved here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from planning.protocol.primitive_execution_result_consumer import (
    EndpointObservation,
    PrimitiveExecutionResultConsumer,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "primitive_execution_v4"
    / "complete.json"
)


class FakeEndpointProvider:
    def __init__(self, observation):
        self.observation = observation
        self.lookup_count = 0

    def lookup(self, result):
        self.lookup_count += 1
        return self.observation


class FakeTransitionCommitter:
    def __init__(self):
        self.transitions = []

    def commit(self, transition):
        self.transitions.append(transition)


class FakeReplayAppender:
    def __init__(self):
        self.transitions = []

    def append_once(self, transition):
        self.transitions.append(transition)


class DynamicEndpointProvider:
    def __init__(self):
        self.lookup_count = 0

    def lookup(self, result):
        self.lookup_count += 1
        return _matching_endpoint(
            {
                "runtime_instance_id": result.runtime_instance_id,
                "execution_id": result.execution_id,
                "endpoint_state_id": result.endpoint_state_id,
                "endpoint_sim_time_ns": result.endpoint_sim_time_ns,
                "endpoint_observation_ref": result.endpoint_observation_ref,
            }
        )


def _complete_result(**overrides):
    result = json.loads(FIXTURE.read_text(encoding="utf-8"))["result"]
    result.update(overrides)
    return result


def _matching_endpoint(result):
    ref = result["endpoint_observation_ref"]
    return EndpointObservation(
        runtime_instance_id=result["runtime_instance_id"],
        execution_id=result["execution_id"],
        state_id=result["endpoint_state_id"],
        sim_time_ns=result["endpoint_sim_time_ns"],
        state={"state_id": result["endpoint_state_id"]},
        episode_id=ref["episode_id"],
        reset_id=ref["reset_id"],
        depth_id=ref["depth_id"],
        physics_time_ns=ref["sim_time_ns"],
    )


@pytest.mark.unit
def test_first_complete_commits_one_transition_and_appends_replay_once():
    result = _complete_result()
    endpoint = FakeEndpointProvider(_matching_endpoint(result))
    committer = FakeTransitionCommitter()
    replay = FakeReplayAppender()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=endpoint,
        transition_committer=committer,
        replay_appender=replay,
    )

    outcome = consumer.consume(result)

    assert outcome.status == "COMMITTED"
    assert outcome.transition_committed is True
    assert outcome.replay_appended is True
    assert len(committer.transitions) == 1
    assert len(replay.transitions) == 1
    assert consumer.transition_commit_count == 1
    assert consumer.replay_append_count == 1


@pytest.mark.unit
def test_duplicate_complete_100_times_has_one_transition_and_one_replay_append():
    result = _complete_result()
    endpoint = FakeEndpointProvider(_matching_endpoint(result))
    committer = FakeTransitionCommitter()
    replay = FakeReplayAppender()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=endpoint,
        transition_committer=committer,
        replay_appender=replay,
    )

    first = consumer.consume(result)
    duplicates = [consumer.consume(result) for _ in range(100)]

    assert first.status == "COMMITTED"
    assert all(outcome.status == "DUPLICATE" for outcome in duplicates)
    assert consumer.transition_commit_count == 1
    assert consumer.replay_append_count == 1
    assert len(committer.transitions) == 1
    assert len(replay.transitions) == 1
    assert endpoint.lookup_count == 1


@pytest.mark.unit
def test_duplicate_commit_is_idempotent():
    result = _complete_result()
    committer = FakeTransitionCommitter()
    replay = FakeReplayAppender()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=FakeEndpointProvider(_matching_endpoint(result)),
        transition_committer=committer,
        replay_appender=replay,
    )

    first = consumer.consume(result)
    duplicate = consumer.commit_transition(
        runtime_instance_id=result["runtime_instance_id"],
        execution_id=result["execution_id"],
        result_payload_hash=first.result_payload_hash,
    )

    assert duplicate.status == "DUPLICATE"
    assert consumer.transition_commit_count == 1
    assert consumer.replay_append_count == 1


@pytest.mark.unit
def test_wrong_execution_id_does_not_commit():
    result = _complete_result()
    committer = FakeTransitionCommitter()
    replay = FakeReplayAppender()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=FakeEndpointProvider(_matching_endpoint(result)),
        transition_committer=committer,
        replay_appender=replay,
    )

    outcome = consumer.consume(
        dict(result, execution_id=result["execution_id"] + 1),
        expected_execution_id=result["execution_id"],
    )

    assert outcome.status == "PROTOCOL_ERROR"
    assert consumer.transition_commit_count == 0
    assert consumer.replay_append_count == 0


@pytest.mark.unit
def test_wrong_result_payload_hash_does_not_commit():
    result = _complete_result(result_payload_hash="0" * 64)
    committer = FakeTransitionCommitter()
    replay = FakeReplayAppender()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=FakeEndpointProvider(_matching_endpoint(result)),
        transition_committer=committer,
        replay_appender=replay,
    )

    outcome = consumer.consume(result)

    assert outcome.status == "PROTOCOL_ERROR"
    assert consumer.transition_commit_count == 0
    assert consumer.replay_append_count == 0


@pytest.mark.unit
def test_conflicting_duplicate_is_protocol_error_and_does_not_overwrite_first():
    result = _complete_result()
    changed = dict(
        result,
        endpoint_sim_time_ns=result["endpoint_sim_time_ns"] + 1,
        endpoint_observation_ref={
            "schema_version": 4,
            "runtime_instance_id": result["runtime_instance_id"],
            "episode_id": "episode-v4-0001",
            "reset_id": "reset-v4-0001",
            "state_id": result["endpoint_state_id"],
            "depth_id": "depth-v4-conflict",
            "sim_time_ns": result["endpoint_sim_time_ns"] + 1,
        },
    )
    committer = FakeTransitionCommitter()
    replay = FakeReplayAppender()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=FakeEndpointProvider(_matching_endpoint(result)),
        transition_committer=committer,
        replay_appender=replay,
    )

    assert consumer.consume(result).status == "COMMITTED"
    conflict = consumer.consume(changed)

    assert conflict.status == "PROTOCOL_ERROR"
    assert consumer.transition_commit_count == 1
    assert consumer.replay_append_count == 1
    assert len(committer.transitions) == 1
    assert len(replay.transitions) == 1


@pytest.mark.unit
def test_endpoint_unavailable_does_not_create_transition():
    result = _complete_result()
    committer = FakeTransitionCommitter()
    replay = FakeReplayAppender()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=FakeEndpointProvider(None),
        transition_committer=committer,
        replay_appender=replay,
    )

    outcome = consumer.consume(result)

    assert outcome.status == "ENDPOINT_UNAVAILABLE"
    assert outcome.transition_committed is False
    assert outcome.replay_appended is False
    assert consumer.pending_count == 1
    assert consumer.transition_commit_count == 0
    assert consumer.replay_append_count == 0


@pytest.mark.unit
def test_endpoint_mismatch_does_not_create_transition():
    result = _complete_result()
    mismatched = EndpointObservation(
        runtime_instance_id=result["runtime_instance_id"],
        execution_id=result["execution_id"],
        state_id=result["endpoint_state_id"] + 1,
        sim_time_ns=result["endpoint_sim_time_ns"],
        state={"state_id": result["endpoint_state_id"] + 1},
        episode_id=result["endpoint_observation_ref"]["episode_id"],
        reset_id=result["endpoint_observation_ref"]["reset_id"],
        depth_id=result["endpoint_observation_ref"]["depth_id"],
        physics_time_ns=result["endpoint_observation_ref"]["sim_time_ns"],
    )
    committer = FakeTransitionCommitter()
    replay = FakeReplayAppender()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=FakeEndpointProvider(mismatched),
        transition_committer=committer,
        replay_appender=replay,
    )

    outcome = consumer.consume(result)

    assert outcome.status == "ENDPOINT_MISMATCH"
    assert consumer.pending_count == 1
    assert consumer.transition_commit_count == 0
    assert consumer.replay_append_count == 0


@pytest.mark.unit
def test_same_execution_id_isolated_by_runtime_instance_id():
    first = _complete_result()
    second = dict(first, runtime_instance_id="worker-01-runtime-test")
    second["endpoint_observation_ref"] = dict(first["endpoint_observation_ref"])
    second["endpoint_observation_ref"]["runtime_instance_id"] = second[
        "runtime_instance_id"
    ]
    committer = FakeTransitionCommitter()
    replay = FakeReplayAppender()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=DynamicEndpointProvider(),
        transition_committer=committer,
        replay_appender=replay,
    )

    first_outcome = consumer.consume(first)
    second_outcome = consumer.consume(second)

    assert first_outcome.status == "COMMITTED"
    assert second_outcome.status == "COMMITTED"
    assert consumer.transition_commit_count == 2
    assert consumer.replay_append_count == 2
    assert len(committer.transitions) == 2
    assert len(replay.transitions) == 2
