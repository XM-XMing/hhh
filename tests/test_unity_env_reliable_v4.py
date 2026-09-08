"""Behavior seam tests for the reliable-v4 UnityForestEnv step path."""

from __future__ import annotations

from dataclasses import dataclass

import msgpack
import numpy as np
import pytest

from planning.runtime.unity_env import EnvConfig, UnityForestEnv
from planning.protocol.primitive_execution_schema_v4 import canonical_result_payload_hash
from planning.protocol.primitive_execution_result_consumer import EndpointObservation
from planning.runtime.reliable_unity_env_backend import (
    ReliableV4BackendError,
    ReliableV4ExecutionBackend,
    ReliableV4ResetResult,
    ReliableV4StepResult,
    _ZmqResultTransport,
)


class _FakeMotionPrimitives:
    num_actions = 105

    def command_sequence(self, action_id):
        return [
            np.asarray([float(action_id), 0.25, -0.5, 0.0], dtype=np.float32)
            for _ in range(25)
        ]

    def endpoint(self, action_id):
        return np.asarray([1.5, 0.0, 0.0], dtype=np.float32)

    def action_metadata(self, action_id):
        return {"action_id": int(action_id)}

    def valid_action_mask(self, z, *, z_min, z_max, margin):
        return np.ones(105, dtype=np.bool_)


def _observation(position, *, state_id, depth_value):
    position = np.asarray(position, dtype=np.float32)
    goal = np.asarray([40.0, 0.0, 2.0], dtype=np.float32)
    relative = goal - position
    distance_xy = float(np.linalg.norm(relative[:2]))
    return {
        "depth": np.full((90, 160), depth_value, dtype=np.float32),
        "depth_m": np.full((90, 160), 2.0, dtype=np.float32),
        "depth_intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 80.0, "cy": 45.0},
        "state": {
            "position": position,
            "velocity": np.zeros(3, dtype=np.float32),
            "acceleration": np.zeros(3, dtype=np.float32),
            "orientation_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "yaw": 0.0,
            "z": float(position[2]),
            "z_to_min": float(position[2] - 1.0),
            "z_to_max": float(3.0 - position[2]),
            "state_id": int(state_id),
            "sim_time_ns": int(state_id) * 20_000_000,
            "flags": 0,
        },
        "goal": {
            "position": goal,
            "relative": relative,
            "direction_xy": np.asarray([1.0, 0.0], dtype=np.float32),
            "direction_body_xy": np.asarray([1.0, 0.0], dtype=np.float32),
            "distance_xy": distance_xy,
            "distance_xy_norm40": distance_xy / 40.0,
            "dz": float(relative[2]),
        },
        "safety": {
            "min_clearance": 1.0,
            "front_clearances": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "collided": False,
            "altitude_violation": False,
        },
        "action_mask": np.ones(105, dtype=np.bool_),
    }


@dataclass
class _FakeReliableBackend:
    endpoint_observation: dict
    calls: list

    def execute(self, *, action_id, primitive_commands):
        self.calls.append((int(action_id), tuple(tuple(frame) for frame in primitive_commands)))
        return ReliableV4StepResult(
            status="COMPLETE",
            execution_id=1234,
            result={
                "execution_id": 1234,
                "status": "COMPLETE",
                "requested_frame_count": 25,
                "applied_frame_count": 25,
                "last_applied_frame_index": 24,
                "endpoint_state_id": 24,
                "endpoint_sim_time_ns": 480_000_000,
            },
            observation=self.endpoint_observation,
        )


class _FakeCommandTransport:
    def __init__(self):
        self.submissions = []

    def submit(self, payload, *, execution_id, command_sequence_hash):
        self.submissions.append(
            (bytes(payload), int(execution_id), str(command_sequence_hash))
        )


class _FakeResultTransport:
    def __init__(self):
        self.receives = []
        self.commits = []

    def receive(self, *, execution_id, command_sequence_hash):
        self.receives.append((int(execution_id), str(command_sequence_hash)))
        result = {
            "schema_version": 4,
            "message_type": "PrimitiveExecutionResult",
            "runtime_instance_id": "worker-00",
            "execution_id": int(execution_id),
            "status": "COMPLETE",
            "requested_frame_count": 25,
            "applied_frame_count": 25,
            "first_applied_state_id": 0,
            "endpoint_state_id": 24,
            "last_applied_frame_index": 24,
            "reason_code": "NONE",
            "command_sequence_hash": str(command_sequence_hash),
            "endpoint_sim_time_ns": 480_000_000,
            "endpoint_observation_ref": {
                "schema_version": 4,
                "runtime_instance_id": "worker-00",
                "episode_id": "episode-0",
                "reset_id": "reset-0",
                "state_id": 24,
                "depth_id": "depth-24",
                "sim_time_ns": 480_000_000,
            },
            "result_generation": 0,
        }
        result["result_payload_hash"] = canonical_result_payload_hash(result)
        return result

    def commit(self, result):
        self.commits.append(dict(result))


class _FakeCollisionResultTransport:
    def __init__(self):
        self.commits = []

    def receive(self, *, execution_id, command_sequence_hash):
        result = {
            "schema_version": 4,
            "message_type": "PrimitiveExecutionResult",
            "runtime_instance_id": "worker-00",
            "execution_id": int(execution_id),
            "status": "FAILED",
            "requested_frame_count": 25,
            "applied_frame_count": 3,
            "first_applied_state_id": 100,
            "endpoint_state_id": 103,
            "last_applied_frame_index": 2,
            "reason_code": "COLLISION",
            "command_sequence_hash": str(command_sequence_hash),
            "endpoint_sim_time_ns": 80_000_000,
            "endpoint_observation_ref": {
                "schema_version": 4,
                "runtime_instance_id": "worker-00",
                "episode_id": "episode-0",
                "reset_id": "reset-0",
                "state_id": 103,
                "depth_id": "depth-terminal-0",
                "sim_time_ns": 80_000_000,
            },
            "result_generation": 0,
        }
        result["result_payload_hash"] = canonical_result_payload_hash(result)
        return result

    def commit(self, result):
        self.commits.append(dict(result))


class _FakeResetTransport:
    def __init__(self, observation, *, fail_receive_once=False):
        self.observation = observation
        self.fail_receive_once = bool(fail_receive_once)
        self.submissions = []
        self.receives = []

    def submit(self, payload, *, episode_id, reset_id):
        self.submissions.append((bytes(payload), episode_id, reset_id))

    def receive(self, *, episode_id, reset_id):
        self.receives.append((episode_id, reset_id))
        if self.fail_receive_once:
            self.fail_receive_once = False
            raise TimeoutError("reset ACK lost")
        return {
            "schema_version": 4,
            "message_type": "PrimitiveResetComplete",
            "runtime_instance_id": "worker-00",
            "episode_id": episode_id,
            "reset_id": reset_id,
            "observation": self.observation,
        }


class _FakeEndpointProvider:
    def __init__(self, observation):
        self.observation = observation
        self.results = []

    def lookup(self, result):
        self.results.append(result)
        return self.observation


class _RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)

    def get_num_connections(self):
        return 1


class _FakeResultSocket:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(bytes(payload))


def test_zmq_result_transport_drops_repeated_committed_result():
    result = _FakeResultTransport().receive(
        execution_id=7001,
        command_sequence_hash="a" * 64,
    )
    ack = {
        "message_type": "PrimitiveExecutionResultCommitAck",
        "runtime_instance_id": "worker-00",
        "execution_id": 7001,
        "command_sequence_hash": result["command_sequence_hash"],
        "result_payload_hash": result["result_payload_hash"],
        "commit_status": "COMMITTED",
    }
    transport = object.__new__(_ZmqResultTransport)
    transport._socket = _FakeResultSocket()
    transport._timeout_s = 1.0
    transport._results = {}
    transport._hashes = {}
    responses = iter((result, ack))
    transport._recv = lambda timeout_s: next(responses)
    transport._send_receipt = lambda candidate: None

    transport.commit(result)

    assert transport._results == {}
    assert transport._hashes == {}


def _make_env(backend):
    env = UnityForestEnv.__new__(UnityForestEnv)
    env.config = EnvConfig()
    env.start = np.asarray([0.0, 0.0, 2.0], dtype=np.float32)
    env.goal = np.asarray([40.0, 0.0, 2.0], dtype=np.float32)
    env.mpl = _FakeMotionPrimitives()
    env.action_space_n = 105
    env._collision_checker = None
    env._last_action_mask_info = {}
    env.episode_step = 0
    env.prev_distance_xy = 40.0
    env.prev_abs_goal_dz = 0.0
    env.prev_action_id = -1
    env._reliable_v4_backend = backend
    env._reliable_transition_cache = {}
    env.telemetry_lookup_count = 0
    env.episode_id = "episode-0"
    env.reset_id = "reset-0"
    env._reset_generation = 0
    env.observation_token = lambda: (0, 0)
    return env


def test_reset_uses_reliable_v4_backend_and_exact_new_identity():
    reset_observation = _observation([4.0, 5.0, 2.0], state_id=11, depth_value=0.40)

    class _ReliableResetBackend:
        def __init__(self):
            self.calls = []

        def reset(self, *, start, goal, episode_id, reset_id):
            self.calls.append((tuple(start), tuple(goal), episode_id, reset_id))
            return ReliableV4ResetResult(
                episode_id=episode_id,
                reset_id=reset_id,
                observation=reset_observation,
            )

    backend = _ReliableResetBackend()
    env = _make_env(backend)
    env.start = np.asarray([4.0, 5.0, 2.0], dtype=np.float32)
    env.goal = np.asarray([40.0, 0.0, 2.0], dtype=np.float32)
    env.prev_distance_xy = 1.0
    env.prev_abs_goal_dz = 1.0
    env.prev_action_id = 7
    env.episode_step = 3

    observation = env.reset(start=env.start, goal=env.goal, settle=0.0)

    assert np.allclose(observation["state"]["position"], reset_observation["state"]["position"])
    assert observation["episode_id"] == "episode-1"
    assert observation["reset_id"] == "reset-1"
    assert backend.calls == [
        ((4.0, 5.0, 2.0), (40.0, 0.0, 2.0), "episode-1", "reset-1")
    ]
    assert env.episode_id == "episode-1"
    assert env.reset_id == "reset-1"
    assert env.episode_step == 0
    assert env.prev_action_id == -1


def test_reliable_v4_reset_and_stop_emit_no_legacy_control_messages():
    reset_observation = _observation([0.0, 0.0, 2.0], state_id=11, depth_value=0.40)

    class _ReliableResetBackend:
        def reset(self, *, start, goal, episode_id, reset_id):
            return ReliableV4ResetResult(
                episode_id=episode_id,
                reset_id=reset_id,
                observation=reset_observation,
            )

    env = _make_env(_ReliableResetBackend())
    env._reset_pub = _RecordingPublisher()
    env._primitive_execution_pub = _RecordingPublisher()
    env._stop_pub = _RecordingPublisher()
    env._cmd_pub = _RecordingPublisher()

    env.reset(start=env.start, goal=env.goal, settle=0.0)
    env.stop()

    assert env._reset_pub.messages == []
    assert env._primitive_execution_pub.messages == []
    assert env._stop_pub.messages == []
    assert env._cmd_pub.messages == []


def test_reliable_v4_step_does_not_publish_legacy_primitive_command():
    before = _observation([0.0, 0.0, 2.0], state_id=0, depth_value=0.25)
    endpoint = _observation([1.5, 0.0, 2.0], state_id=24, depth_value=0.75)
    env = _make_env(_FakeReliableBackend(endpoint_observation=endpoint, calls=[]))
    env._primitive_execution_pub = _RecordingPublisher()
    env._stop_pub = _RecordingPublisher()
    env._cmd_pub = _RecordingPublisher()
    env.get_action_mask = lambda obs, return_info=False: (
        (np.ones(105, dtype=np.bool_), {"dead_end": False})
        if return_info
        else np.ones(105, dtype=np.bool_)
    )

    env.step_primitive(7, obs_before=before)

    assert env._primitive_execution_pub.messages == []
    assert env._stop_pub.messages == []
    assert env._cmd_pub.messages == []


def test_reset_materializes_exact_snapshot_endpoint_into_existing_observation_shape():
    """The real bridge provider returns EndpointObservation, not a mapping."""
    state_id = 11
    sim_time_ns = 220_000_000
    state_values = [
        4,
        state_id,
        sim_time_ns,
        0,
        1.0,
        [4.0, 5.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        0.0,
        0.0,
        0.0,
        0.0,
        "episode-1",
        "reset-1",
        "worker-00",
    ]
    endpoint = EndpointObservation(
        runtime_instance_id="worker-00",
        execution_id=0,
        state_id=state_id,
        sim_time_ns=sim_time_ns,
        state=msgpack.packb(state_values, use_bin_type=True),
        depth=(2000).to_bytes(2, "little") * (848 * 480),
        episode_id="episode-1",
        reset_id="reset-1",
        depth_id="depth-11",
        physics_time_ns=sim_time_ns,
    )

    class _ReliableResetBackend:
        def reset(self, *, start, goal, episode_id, reset_id):
            return ReliableV4ResetResult(
                episode_id=episode_id,
                reset_id=reset_id,
                observation=endpoint,
            )

    env = _make_env(_ReliableResetBackend())
    observation = env.reset(start=env.start, goal=env.goal, settle=0.0)

    assert observation["state"]["state_id"] == state_id
    assert observation["episode_id"] == "episode-1"
    assert observation["reset_id"] == "reset-1"
    assert observation["depth"].shape == (90, 160)


def test_step_primitive_uses_exact_reliable_v4_endpoint_and_existing_reward_path():
    before = _observation([0.0, 0.0, 2.0], state_id=0, depth_value=0.25)
    endpoint = _observation([1.5, 0.0, 2.0], state_id=24, depth_value=0.75)
    backend = _FakeReliableBackend(endpoint_observation=endpoint, calls=[])
    env = _make_env(backend)
    env.get_action_mask = lambda obs, return_info=False: (
        (np.ones(105, dtype=np.bool_), {"dead_end": False})
        if return_info
        else np.ones(105, dtype=np.bool_)
    )

    obs, reward, done, info = env.step_primitive(7, obs_before=before)

    assert len(backend.calls) == 1
    assert backend.calls[0][0] == 7
    assert len(backend.calls[0][1]) == 25
    assert obs is endpoint
    assert np.all(obs["depth"] == 0.75)
    assert env.prev_action_id == 7
    assert env.episode_step == 1
    assert done is False
    assert info["primitive_execution"]["status"] == "COMPLETE"
    assert reward == 2.98
    assert env.telemetry_lookup_count == 0


def test_reliable_backend_preserves_command_identity_and_commits_once():
    command_transport = _FakeCommandTransport()
    result_transport = _FakeResultTransport()
    endpoint = object()
    provider = _FakeEndpointProvider(endpoint)
    backend = ReliableV4ExecutionBackend(
        runtime_instance_id="worker-00",
        command_transport=command_transport,
        result_transport=result_transport,
        endpoint_provider=provider,
        execution_id_factory=iter([9001, 9001]).__next__,
    )
    commands = [np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for _ in range(25)]

    first = backend.execute(action_id=11, primitive_commands=commands)
    duplicate = backend.execute(action_id=11, primitive_commands=commands)

    assert first is duplicate
    assert first.status == "COMPLETE"
    assert first.observation is endpoint
    assert len(command_transport.submissions) == 1
    assert len(result_transport.receives) == 1
    assert len(result_transport.commits) == 1
    assert len(provider.results) == 1
    assert result_transport.commits[0]["execution_id"] == 9001


def test_reliable_backend_completion_cache_is_bounded_across_transactions():
    command_transport = _FakeCommandTransport()
    result_transport = _FakeResultTransport()
    backend = ReliableV4ExecutionBackend(
        runtime_instance_id="worker-00",
        command_transport=command_transport,
        result_transport=result_transport,
        endpoint_provider=_FakeEndpointProvider(object()),
        execution_id_factory=iter([9101, 9102, 9103]).__next__,
    )
    commands = [
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for _ in range(25)
    ]

    backend.execute(action_id=11, primitive_commands=commands)
    assert len(backend._completed) == 1
    assert 9101 in backend._completed
    backend.execute(action_id=11, primitive_commands=commands)
    assert len(backend._completed) == 1
    assert 9102 in backend._completed
    assert 9101 not in backend._completed
    backend.execute(action_id=11, primitive_commands=commands)
    assert len(backend._completed) == 1
    assert 9103 in backend._completed


def test_reliable_backend_retrieves_exact_observation_for_failed_collision():
    command_transport = _FakeCommandTransport()
    result_transport = _FakeCollisionResultTransport()
    endpoint = object()
    provider = _FakeEndpointProvider(endpoint)
    backend = ReliableV4ExecutionBackend(
        runtime_instance_id="worker-00",
        command_transport=command_transport,
        result_transport=result_transport,
        endpoint_provider=provider,
        execution_id_factory=iter([9004]).__next__,
    )
    commands = [np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for _ in range(25)]

    step = backend.execute(action_id=11, primitive_commands=commands)

    assert step.status == "FAILED"
    assert step.result["reason_code"] == "COLLISION"
    assert step.observation is endpoint
    assert provider.results[0].status == "FAILED"
    assert len(result_transport.commits) == 1


def test_reliable_reset_duplicate_is_idempotent_and_conflict_is_rejected():
    observation = _observation([4.0, 5.0, 2.0], state_id=11, depth_value=0.40)
    transport = _FakeResetTransport(observation)
    backend = ReliableV4ExecutionBackend(
        runtime_instance_id="worker-00",
        command_transport=_FakeCommandTransport(),
        result_transport=_FakeResultTransport(),
        endpoint_provider=object(),
        reset_transport=transport,
    )

    first = backend.reset(
        start=[4.0, 5.0, 2.0],
        goal=[40.0, 0.0, 2.0],
        episode_id="episode-1",
        reset_id="reset-1",
    )
    duplicate = backend.reset(
        start=[4.0, 5.0, 2.0],
        goal=[40.0, 0.0, 2.0],
        episode_id="episode-1",
        reset_id="reset-1",
    )

    assert duplicate is first
    assert len(transport.submissions) == 1
    assert len(transport.receives) == 1
    assert len(backend._completed_resets) <= 1
    assert len(backend._reset_payloads) <= 1
    try:
        backend.reset(
            start=[9.0, 5.0, 2.0],
            goal=[40.0, 0.0, 2.0],
            episode_id="episode-1",
            reset_id="reset-1",
        )
    except ReliableV4BackendError as error:
        assert "conflicting reset payload" in str(error)
    else:
        raise AssertionError("conflicting reset identity must fail closed")


def test_reliable_reset_ack_loss_retries_exact_request_without_local_duplicate():
    observation = _observation([4.0, 5.0, 2.0], state_id=11, depth_value=0.40)
    transport = _FakeResetTransport(observation, fail_receive_once=True)
    backend = ReliableV4ExecutionBackend(
        runtime_instance_id="worker-00",
        command_transport=_FakeCommandTransport(),
        result_transport=_FakeResultTransport(),
        endpoint_provider=object(),
        reset_transport=transport,
    )
    kwargs = {
        "start": [4.0, 5.0, 2.0],
        "goal": [40.0, 0.0, 2.0],
        "episode_id": "episode-2",
        "reset_id": "reset-2",
    }
    try:
        backend.reset(**kwargs)
    except TimeoutError:
        pass
    else:
        raise AssertionError("lost reset ACK must remain observable to the caller")
    completion = backend.reset(**kwargs)
    assert completion.observation is observation
    assert len(transport.submissions) == 2
    assert transport.submissions[0][0] == transport.submissions[1][0]


def test_stale_pre_reset_observation_is_rejected():
    stale = _observation([4.0, 5.0, 2.0], state_id=11, depth_value=0.40)
    stale["episode_id"] = "episode-old"
    stale["reset_id"] = "reset-old"

    class _StaleBackend:
        def reset(self, *, start, goal, episode_id, reset_id):
            return ReliableV4ResetResult(
                episode_id=episode_id,
                reset_id=reset_id,
                observation=stale,
            )

    env = _make_env(_StaleBackend())
    try:
        env.reset(start=[4.0, 5.0, 2.0], goal=[40.0, 0.0, 2.0], settle=0.0)
    except RuntimeError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("stale pre-reset observation must be rejected")
    assert env.episode_id == "episode-0"
    assert env.reset_id == "reset-0"


def test_failed_reliable_reset_reserves_identity_before_next_mission():
    reset_observation = _observation([4.0, 5.0, 2.0], state_id=11, depth_value=0.40)

    class _FailOnceResetBackend:
        def __init__(self):
            self.calls = []
            self.fail_once = True

        def reset(self, *, start, goal, episode_id, reset_id):
            self.calls.append((tuple(start), tuple(goal), episode_id, reset_id))
            if self.fail_once:
                self.fail_once = False
                raise TimeoutError("reset completion timeout")
            return ReliableV4ResetResult(
                episode_id=episode_id,
                reset_id=reset_id,
                observation=reset_observation,
            )

    backend = _FailOnceResetBackend()
    env = _make_env(backend)
    with pytest.raises(TimeoutError, match="reset completion"):
        env.reset(start=[4.0, 5.0, 2.0], goal=[40.0, 0.0, 2.0], settle=0.0)

    env.reset(start=[9.0, 5.0, 2.0], goal=[40.0, 0.0, 2.0], settle=0.0)

    assert backend.calls == [
        ((4.0, 5.0, 2.0), (40.0, 0.0, 2.0), "episode-1", "reset-1"),
        ((9.0, 5.0, 2.0), (40.0, 0.0, 2.0), "episode-2", "reset-2"),
    ]


def test_reset_then_reliable_step_advances_once_from_new_identity():
    reset_observation = _observation([4.0, 5.0, 2.0], state_id=11, depth_value=0.40)
    next_observation = _observation([5.5, 5.0, 2.0], state_id=36, depth_value=0.60)

    class _ResetAndStepBackend:
        def __init__(self):
            self.reset_calls = []
            self.step_calls = []

        def reset(self, *, start, goal, episode_id, reset_id):
            self.reset_calls.append((episode_id, reset_id))
            return ReliableV4ResetResult(
                episode_id=episode_id,
                reset_id=reset_id,
                observation=reset_observation,
            )

        def execute(self, *, action_id, primitive_commands):
            self.step_calls.append((action_id, len(primitive_commands)))
            return ReliableV4StepResult(
                status="COMPLETE",
                execution_id=77,
                result={"status": "COMPLETE"},
                observation=next_observation,
            )

    backend = _ResetAndStepBackend()
    env = _make_env(backend)
    env.reset(start=[4.0, 5.0, 2.0], goal=[40.0, 0.0, 2.0], settle=0.0)
    env.get_action_mask = lambda obs, return_info=False: (
        (np.ones(105, dtype=np.bool_), {"dead_end": False})
        if return_info
        else np.ones(105, dtype=np.bool_)
    )

    obs, _, done, _ = env.step_primitive(
        7, obs_before=reset_observation
    )

    assert backend.reset_calls == [("episode-1", "reset-1")]
    assert backend.step_calls == [(7, 25)]
    assert obs is next_observation
    assert done is False
    assert env.episode_step == 1
    assert env.prev_action_id == 7


def test_reliable_v4_reset_after_45_steps_stays_exclusive_and_advances_once():
    reset_observation = _observation([0.0, 0.0, 2.0], state_id=11, depth_value=0.40)
    step_observation = _observation([1.5, 0.0, 2.0], state_id=36, depth_value=0.60)

    class _TwoResetBackend:
        def __init__(self):
            self.reset_calls = []
            self.step_calls = 0

        def reset(self, *, start, goal, episode_id, reset_id):
            self.reset_calls.append((episode_id, reset_id))
            return ReliableV4ResetResult(
                episode_id=episode_id,
                reset_id=reset_id,
                observation=reset_observation,
            )

        def execute(self, *, action_id, primitive_commands):
            self.step_calls += 1
            assert int(action_id) == 7
            assert len(primitive_commands) == 25
            return ReliableV4StepResult(
                status="COMPLETE",
                execution_id=1000 + self.step_calls,
                result={"status": "COMPLETE"},
                observation=step_observation,
            )

    backend = _TwoResetBackend()
    env = _make_env(backend)
    env._reset_pub = _RecordingPublisher()
    env._primitive_execution_pub = _RecordingPublisher()
    env._stop_pub = _RecordingPublisher()
    env._cmd_pub = _RecordingPublisher()
    env.get_action_mask = lambda obs, return_info=False: (
        (np.ones(105, dtype=np.bool_), {"dead_end": False})
        if return_info
        else np.ones(105, dtype=np.bool_)
    )

    observation = env.reset(start=env.start, goal=env.goal, settle=0.0)
    for _ in range(45):
        observation, _, done, _ = env.step_primitive(7, obs_before=observation)
        assert done is False
    env.reset(start=env.start, goal=env.goal, settle=0.0)

    assert backend.reset_calls == [
        ("episode-1", "reset-1"),
        ("episode-2", "reset-2"),
    ]
    assert backend.step_calls == 45
    assert env.episode_id == "episode-2"
    assert env.reset_id == "reset-2"
    assert env.episode_step == 0
    assert env._reset_pub.messages == []
    assert env._primitive_execution_pub.messages == []
    assert env._stop_pub.messages == []
    assert env._cmd_pub.messages == []


def test_reliable_snapshot_bytes_become_the_next_policy_observation():
    state_values = [
        3, 24, 480_000_000, 0, 1.0,
        [1.5, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
        9001, 24, 900124, 2, "episode-0", "reset-0", "worker-00",
    ]
    endpoint = EndpointObservation(
        runtime_instance_id="worker-00",
        execution_id=9001,
        state_id=24,
        sim_time_ns=480_000_000,
        state=__import__("msgpack").packb(state_values, use_bin_type=True),
        depth=(2000).to_bytes(2, "little") * (90 * 160),
        episode_id="episode-0",
        reset_id="reset-0",
        depth_id="depth-24",
        physics_time_ns=480_000_000,
    )
    backend = _FakeReliableBackend(endpoint_observation=endpoint, calls=[])
    env = _make_env(backend)
    env.get_action_mask = lambda obs, return_info=False: (
        (np.ones(105, dtype=np.bool_), {"dead_end": False})
        if return_info
        else np.ones(105, dtype=np.bool_)
    )

    obs, _, _, _ = env.step_primitive(7, obs_before=_observation([0.0, 0.0, 2.0], state_id=0, depth_value=0.25))

    assert obs["state"]["state_id"] == 24
    assert np.allclose(obs["state"]["position"], [1.5, 0.0, 2.0])
    assert obs["depth"].shape == (90, 160)
    assert np.allclose(obs["depth_m"], 2.0)


def test_missing_reliable_snapshot_does_not_advance_environment_step():
    class _MissingBackend:
        def execute(self, *, action_id, primitive_commands):
            raise ReliableV4BackendError("authoritative endpoint snapshot unavailable")

    env = _make_env(_MissingBackend())
    env.get_action_mask = lambda obs, return_info=False: (
        (np.ones(105, dtype=np.bool_), {"dead_end": False})
        if return_info
        else np.ones(105, dtype=np.bool_)
    )

    try:
        env.step_primitive(7, obs_before=_observation([0.0, 0.0, 2.0], state_id=0, depth_value=0.25))
    except ReliableV4BackendError:
        pass
    else:
        raise AssertionError("missing snapshot must fail the step")
    assert env.prev_action_id == -1
    assert env.episode_step == 0


def test_failed_cancelled_and_rejected_results_do_not_advance_environment_step():
    for status in ("FAILED", "CANCELLED", "REJECTED"):
        class _TerminalBackend:
            def execute(self, *, action_id, primitive_commands):
                return ReliableV4StepResult(
                    status=status,
                    execution_id=99,
                    result={"status": status},
                    observation=None,
                )

        env = _make_env(_TerminalBackend())
        env.get_action_mask = lambda obs, return_info=False: (
            (np.ones(105, dtype=np.bool_), {"dead_end": False})
            if return_info
            else np.ones(105, dtype=np.bool_)
        )
        try:
            env.step_primitive(7, obs_before=_observation([0.0, 0.0, 2.0], state_id=0, depth_value=0.25))
        except Exception as error:
            assert "not a legal environment transition" in str(error)
        else:
            raise AssertionError("{} must not be treated as COMPLETE".format(status))
        assert env.prev_action_id == -1
        assert env.episode_step == 0


def test_failed_collision_with_exact_terminal_observation_is_a_normal_transition():
    before = _observation([0.0, 0.0, 2.0], state_id=0, depth_value=0.25)
    terminal = _observation([0.6, 0.0, 2.0], state_id=3, depth_value=0.75)
    terminal["safety"]["collided"] = True

    class _CollisionBackend:
        def __init__(self):
            self.calls = 0

        def execute(self, *, action_id, primitive_commands):
            self.calls += 1
            return ReliableV4StepResult(
                status="FAILED",
                execution_id=9001,
                result={
                    "status": "FAILED",
                    "reason_code": "COLLISION",
                    "execution_id": 9001,
                },
                observation=terminal,
            )

    backend = _CollisionBackend()
    env = _make_env(backend)
    env.get_action_mask = lambda obs, return_info=False: (
        (np.ones(105, dtype=np.bool_), {"dead_end": False})
        if return_info
        else np.ones(105, dtype=np.bool_)
    )

    obs, reward, done, info = env.step_primitive(7, obs_before=before)

    assert obs is terminal
    assert done is True
    assert info["collided"] is True
    assert info["done_reason"] == "collision"
    assert info["terminal_result_status"] == "FAILED"
    assert info["primitive_completed"] is False
    expected_reward = (
        env.config.reward_progress_scale * info["progress"]
        + env.config.reward_goal_z_progress_scale * info["goal_z_progress"]
        + env.config.reward_step
        + env.config.reward_clearance_scale * max(
            0.0, env.config.clearance_margin_m - info["min_clearance"]
        )
        + env.config.reward_collision
    )
    assert reward == expected_reward
    assert env.episode_step == 1
    assert env.prev_action_id == 7
    assert env.telemetry_lookup_count == 0
    assert backend.calls == 1


def test_failed_collision_without_exact_terminal_observation_fails_closed():
    class _MissingTerminalObservationBackend:
        def execute(self, *, action_id, primitive_commands):
            return ReliableV4StepResult(
                status="FAILED",
                execution_id=9002,
                result={
                    "status": "FAILED",
                    "reason_code": "COLLISION",
                    "execution_id": 9002,
                },
                observation=None,
            )

    env = _make_env(_MissingTerminalObservationBackend())
    env.get_action_mask = lambda obs, return_info=False: (
        (np.ones(105, dtype=np.bool_), {"dead_end": False})
        if return_info
        else np.ones(105, dtype=np.bool_)
    )

    try:
        env.step_primitive(
            7,
            obs_before=_observation([0.0, 0.0, 2.0], state_id=0, depth_value=0.25),
        )
    except ReliableV4BackendError as error:
        assert "exact terminal observation" in str(error)
    else:
        raise AssertionError("missing terminal observation must fail closed")
    assert env.episode_step == 0
    assert env.prev_action_id == -1
    assert env.telemetry_lookup_count == 0


def test_duplicate_failed_collision_does_not_advance_environment_twice():
    before = _observation([0.0, 0.0, 2.0], state_id=0, depth_value=0.25)
    terminal = _observation([0.6, 0.0, 2.0], state_id=3, depth_value=0.75)
    terminal["safety"]["collided"] = True

    class _DuplicateCollisionBackend:
        def __init__(self):
            self.calls = 0

        def execute(self, *, action_id, primitive_commands):
            self.calls += 1
            return ReliableV4StepResult(
                status="FAILED",
                execution_id=9003,
                result={"status": "FAILED", "reason_code": "COLLISION", "execution_id": 9003},
                observation=terminal,
            )

    backend = _DuplicateCollisionBackend()
    env = _make_env(backend)
    env.get_action_mask = lambda obs, return_info=False: (
        (np.ones(105, dtype=np.bool_), {"dead_end": False})
        if return_info
        else np.ones(105, dtype=np.bool_)
    )

    first = env.step_primitive(7, obs_before=before)
    second = env.step_primitive(7, obs_before=before)

    assert first[1:] == second[1:]
    assert env.episode_step == 1
    assert env.prev_action_id == 7
    assert backend.calls == 2
