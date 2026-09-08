"""P0-L4B public tests for exact endpoint observation indexing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from planning.runtime.primitive_execution_observation_store import (
    EndpointObservationStoreError,
    IndexedEndpointObservationStore,
)
from planning.protocol.primitive_execution_result_consumer import (
    PrimitiveExecutionResultConsumer,
)


STATE_ID = 4024
SIM_TIME_NS = 5_000_000_000
EPISODE_ID = "episode-17"
RESET_ID = "reset-03"
RUNTIME_ID = "worker-00-runtime-test"
COMPLETE_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "primitive_execution_v4"
    / "complete.json"
)


def _state(**overrides):
    value = {
        "state_id": STATE_ID,
        "sim_time_ns": SIM_TIME_NS,
        "episode_id": EPISODE_ID,
        "reset_id": RESET_ID,
        "position": [1.0, 2.0, 3.0],
    }
    value.update(overrides)
    return value


def _depth(**overrides):
    value = {
        "state_id": STATE_ID,
        "sim_time_ns": SIM_TIME_NS,
        "episode_id": EPISODE_ID,
        "reset_id": RESET_ID,
        "pixels": [0.25, 0.5],
    }
    value.update(overrides)
    return value


def _complete_result():
    result = json.loads(COMPLETE_FIXTURE.read_text(encoding="utf-8"))["result"]
    result["endpoint_observation_ref"] = dict(result["endpoint_observation_ref"])
    result["endpoint_observation_ref"].update(
        runtime_instance_id=RUNTIME_ID,
        episode_id=EPISODE_ID,
        reset_id=RESET_ID,
        depth_id="depth-{}".format(STATE_ID),
    )
    return result


def _bound_state(**overrides):
    return _state(
        runtime_instance_id=RUNTIME_ID,
        physics_time_ns=SIM_TIME_NS,
        **overrides
    )


def _bound_depth(depth_id, **overrides):
    return _depth(
        runtime_instance_id=RUNTIME_ID,
        depth_id=depth_id,
        physics_time_ns=SIM_TIME_NS,
        capture_time_ns=SIM_TIME_NS,
        **overrides
    )


@pytest.mark.unit
def test_exact_state_and_depth_are_retrievable_by_endpoint_state_id():
    store = IndexedEndpointObservationStore(
        episode_id=EPISODE_ID,
        reset_id=RESET_ID,
    )

    store.put_state(STATE_ID, _bound_state())
    store.put_depth(STATE_ID, _bound_depth(STATE_ID))

    observation = store.get(STATE_ID)

    assert observation is not None
    assert observation.state_id == STATE_ID
    assert observation.sim_time_ns == SIM_TIME_NS
    assert observation.episode_id == EPISODE_ID
    assert observation.reset_id == RESET_ID
    assert observation.state["position"] == [1.0, 2.0, 3.0]
    assert observation.depth["pixels"] == [0.25, 0.5]


@pytest.mark.unit
def test_missing_state_is_not_synthesized_from_depth():
    store = IndexedEndpointObservationStore(
        episode_id=EPISODE_ID,
        reset_id=RESET_ID,
    )
    store.put_depth(STATE_ID, _depth())

    assert store.get(STATE_ID) is None


@pytest.mark.unit
def test_missing_depth_is_not_synthesized_from_state():
    store = IndexedEndpointObservationStore(
        episode_id=EPISODE_ID,
        reset_id=RESET_ID,
    )
    store.put_state(STATE_ID, _state())

    assert store.get(STATE_ID) is None


@pytest.mark.unit
def test_lookup_diagnostics_distinguish_state_insert_from_missing_depth():
    events = []
    store = IndexedEndpointObservationStore(
        episode_id=EPISODE_ID,
        reset_id=RESET_ID,
        diagnostic_sink=lambda event, **fields: events.append((event, fields)),
    )
    result = {
        "endpoint_state_id": STATE_ID,
        "runtime_instance_id": "",
        "execution_id": 54,
        "endpoint_observation_ref": {
            "state_id": STATE_ID,
            "capture_id": "depth-4024",
            "sim_time_ns": SIM_TIME_NS,
        },
    }

    store.put_state(STATE_ID, _state())

    assert store.lookup(result) is None
    assert events == [
        (
            "STATE_INSERT",
            {
                "state_id": STATE_ID,
                "physics_time_ns": SIM_TIME_NS,
                "insert_outcome": "INSERTED",
            },
        ),
        (
            "LOOKUP_MISS",
            {
                "endpoint_state_id": STATE_ID,
                "depth_id": "depth-4024",
                "reason": "DEPTH_MISSING",
            },
        ),
    ]


@pytest.mark.unit
def test_state_and_depth_timestamp_mismatch_is_rejected():
    store = IndexedEndpointObservationStore(
        episode_id=EPISODE_ID,
        reset_id=RESET_ID,
    )
    store.put_state(STATE_ID, _state())
    store.put_depth(STATE_ID, _depth(sim_time_ns=SIM_TIME_NS + 1))

    assert store.get(STATE_ID) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value",
    [("episode_id", "episode-18"), ("reset_id", "reset-04")],
)
def test_wrong_episode_or_reset_id_is_rejected_on_insert(field, value):
    store = IndexedEndpointObservationStore(
        episode_id=EPISODE_ID,
        reset_id=RESET_ID,
    )

    with pytest.raises(EndpointObservationStoreError, match=field):
        store.put_state(STATE_ID, _state(**{field: value}))


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value",
    [("episode_id", "episode-18"), ("reset_id", "reset-04")],
)
def test_wrong_episode_or_reset_id_is_rejected_for_depth(field, value):
    store = IndexedEndpointObservationStore(
        episode_id=EPISODE_ID,
        reset_id=RESET_ID,
    )

    with pytest.raises(EndpointObservationStoreError, match=field):
        store.put_depth(STATE_ID, _depth(**{field: value}))


class _Committer:
    def __init__(self):
        self.transitions = []

    def commit(self, transition):
        self.transitions.append(transition)


class _Replay:
    def __init__(self):
        self.transitions = []

    def append_once(self, transition):
        self.transitions.append(transition)


@pytest.mark.unit
def test_complete_result_binds_to_indexed_endpoint_and_commits_once():
    result = _complete_result()
    store = IndexedEndpointObservationStore(
        episode_id=EPISODE_ID,
        reset_id=RESET_ID,
        runtime_instance_id=RUNTIME_ID,
    )
    store.put_state(STATE_ID, _bound_state())
    store.put_depth("depth-{}".format(STATE_ID), _bound_depth(
        "depth-{}".format(STATE_ID)))
    committer = _Committer()
    replay = _Replay()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=store,
        transition_committer=committer,
        replay_appender=replay,
    )

    first = consumer.consume(result)
    duplicate = consumer.consume(result)

    assert first.status == "COMMITTED"
    assert duplicate.status == "DUPLICATE"
    assert consumer.transition_commit_count == 1
    assert consumer.replay_append_count == 1
    assert len(committer.transitions) == 1
    assert len(replay.transitions) == 1
    assert committer.transitions[0].endpoint.state["state_id"] == STATE_ID
    assert committer.transitions[0].endpoint.depth["state_id"] == STATE_ID


@pytest.mark.unit
def test_consumer_does_not_fallback_to_latest_or_next_index():
    result = _complete_result()
    store = IndexedEndpointObservationStore(
        episode_id=EPISODE_ID,
        reset_id=RESET_ID,
        runtime_instance_id=RUNTIME_ID,
    )
    store.put_state(STATE_ID - 1, _bound_state(state_id=STATE_ID - 1))
    store.put_depth("depth-{}".format(STATE_ID - 1), _bound_depth(
        "depth-{}".format(STATE_ID - 1), state_id=STATE_ID - 1))
    store.put_state(STATE_ID + 1, _bound_state(state_id=STATE_ID + 1))
    store.put_depth("depth-{}".format(STATE_ID + 1), _bound_depth(
        "depth-{}".format(STATE_ID + 1), state_id=STATE_ID + 1))
    committer = _Committer()
    replay = _Replay()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=store,
        transition_committer=committer,
        replay_appender=replay,
    )

    outcome = consumer.consume(result)

    assert outcome.status == "ENDPOINT_UNAVAILABLE"
    assert consumer.transition_commit_count == 0
    assert consumer.replay_append_count == 0
    assert committer.transitions == []
    assert replay.transitions == []


@pytest.mark.unit
def test_consumer_does_not_commit_when_result_timestamp_has_no_exact_endpoint():
    result = _complete_result()
    result["endpoint_sim_time_ns"] += 1
    result["endpoint_observation_ref"]["sim_time_ns"] += 1
    store = IndexedEndpointObservationStore(
        episode_id=EPISODE_ID,
        reset_id=RESET_ID,
        runtime_instance_id=RUNTIME_ID,
    )
    store.put_state(STATE_ID, _bound_state())
    store.put_depth("depth-{}".format(STATE_ID), _bound_depth(
        "depth-{}".format(STATE_ID)))
    committer = _Committer()
    replay = _Replay()
    consumer = PrimitiveExecutionResultConsumer(
        endpoint_provider=store,
        transition_committer=committer,
        replay_appender=replay,
    )

    outcome = consumer.consume(result)

    assert outcome.status == "ENDPOINT_UNAVAILABLE"
    assert consumer.transition_commit_count == 0
    assert consumer.replay_append_count == 0
