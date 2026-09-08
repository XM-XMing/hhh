"""P0-M1.5 RED tests for explicit Unity endpoint observation identity."""

from types import SimpleNamespace

import pytest

from planning.runtime.primitive_execution_observation_store import (
    IndexedEndpointObservationStore,
)


RUNTIME_ID = "worker-00-real-m1"
STATE_ID = 4024
DEPTH_ID = 77
PHYSICS_TIME_NS = 559999987
CAPTURE_TIME_NS = 561224825
EPISODE_ID = "m1-episode-00"
RESET_ID = "m1-reset-00"


def _state(**overrides):
    value = {
        "runtime_instance_id": RUNTIME_ID,
        "state_id": STATE_ID,
        "sim_time_ns": PHYSICS_TIME_NS,
        "physics_time_ns": PHYSICS_TIME_NS,
        "episode_id": EPISODE_ID,
        "reset_id": RESET_ID,
        "position": [1.0, 2.0, 3.0],
    }
    value.update(overrides)
    return value


def _depth(**overrides):
    value = {
        "runtime_instance_id": RUNTIME_ID,
        "depth_id": DEPTH_ID,
        "capture_id": "depth-{}".format(DEPTH_ID),
        "state_id": STATE_ID,
        "physics_time_ns": PHYSICS_TIME_NS,
        "capture_time_ns": CAPTURE_TIME_NS,
        "sim_time_ns": CAPTURE_TIME_NS,
        "episode_id": EPISODE_ID,
        "reset_id": RESET_ID,
        "pixels": [0.25, 0.5],
    }
    value.update(overrides)
    return value


def _result(**overrides):
    ref = {
        "capture_id": "depth-{}".format(DEPTH_ID),
        "state_id": STATE_ID,
        "sim_time_ns": PHYSICS_TIME_NS,
        "depth_id": DEPTH_ID,
        "physics_time_ns": PHYSICS_TIME_NS,
        "capture_time_ns": CAPTURE_TIME_NS,
        "episode_id": EPISODE_ID,
        "reset_id": RESET_ID,
    }
    ref.update(overrides.pop("endpoint_observation_ref", {}))
    value = {
        "runtime_instance_id": RUNTIME_ID,
        "execution_id": 4435486932233309751,
        "endpoint_state_id": STATE_ID,
        "endpoint_sim_time_ns": PHYSICS_TIME_NS,
        "endpoint_observation_ref": ref,
    }
    value.update(overrides)
    return SimpleNamespace(**value)


def _store(runtime_id=RUNTIME_ID, episode_id=EPISODE_ID, reset_id=RESET_ID):
    return IndexedEndpointObservationStore(
        runtime_instance_id=runtime_id,
        episode_id=episode_id,
        reset_id=reset_id,
    )


@pytest.mark.unit
def test_matching_state_and_depth_identity_is_bound_without_timestamp_inference():
    store = _store()
    store.put_state(STATE_ID, _state())
    store.put_depth(DEPTH_ID, _depth())

    observation = store.lookup(_result())

    assert observation is not None
    assert observation.runtime_instance_id == RUNTIME_ID
    assert observation.state_id == STATE_ID
    assert observation.depth_id == DEPTH_ID
    assert observation.physics_time_ns == PHYSICS_TIME_NS
    assert observation.capture_time_ns == CAPTURE_TIME_NS
    assert observation.episode_id == EPISODE_ID
    assert observation.reset_id == RESET_ID


@pytest.mark.unit
def test_state_without_depth_is_unavailable():
    store = _store()
    store.put_state(STATE_ID, _state())

    assert store.lookup(_result()) is None


@pytest.mark.unit
def test_depth_with_wrong_state_identity_is_not_rebound_to_endpoint():
    store = _store()
    store.put_state(STATE_ID, _state())
    store.put_depth(DEPTH_ID, _depth(state_id=STATE_ID + 1))

    assert store.lookup(_result()) is None


@pytest.mark.unit
def test_physics_timestamp_mismatch_is_rejected_even_when_capture_is_delayed():
    store = _store()
    store.put_state(STATE_ID, _state())
    store.put_depth(DEPTH_ID, _depth(physics_time_ns=PHYSICS_TIME_NS + 1))

    assert store.lookup(_result()) is None


@pytest.mark.unit
def test_delayed_depth_cannot_bind_to_the_next_state():
    store = _store()
    store.put_state(STATE_ID, _state())
    store.put_depth(
        DEPTH_ID,
        _depth(
            state_id=STATE_ID + 1,
            physics_time_ns=PHYSICS_TIME_NS + 20_000_000,
        ),
    )

    assert store.lookup(_result()) is None


@pytest.mark.unit
def test_duplicate_observation_identity_is_idempotent():
    store = _store()
    state = _state()
    depth = _depth()
    store.put_state(STATE_ID, state)
    store.put_state(STATE_ID, dict(state))
    store.put_depth(DEPTH_ID, depth)
    store.put_depth(DEPTH_ID, dict(depth))

    assert store.lookup(_result()) is not None


@pytest.mark.unit
def test_same_state_id_from_two_workers_is_isolated():
    worker_a = _store(runtime_id="worker-a")
    worker_b = _store(runtime_id="worker-b")
    worker_a.put_state(STATE_ID, _state(runtime_instance_id="worker-a"))
    worker_a.put_depth(DEPTH_ID, _depth(runtime_instance_id="worker-a"))
    worker_b.put_state(STATE_ID, _state(runtime_instance_id="worker-b"))
    worker_b.put_depth(DEPTH_ID, _depth(runtime_instance_id="worker-b"))

    assert worker_a.lookup(_result(runtime_instance_id="worker-a")) is not None
    assert worker_b.lookup(_result(runtime_instance_id="worker-b")) is not None
    assert worker_a.lookup(_result(runtime_instance_id="worker-b")) is None

