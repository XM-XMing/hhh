"""P0-M1 real Unity -> bridge -> Python result-path integration.

This test starts the actual Unity Player and actual bridge process.  The only
test doubles are the final transition/replay sinks at the Python application
boundary; Unity, result transport, bridge receipt, state telemetry, and depth
telemetry are real processes/sockets.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import msgpack
import pytest
import zmq
from zmq.utils.monitor import recv_monitor_message

from planning.runtime.primitive_execution_observation_store import (
    IndexedEndpointObservationStore,
)
from planning.protocol.primitive_execution_result_consumer import (
    PrimitiveExecutionResultConsumer,
)
from planning.protocol.primitive_execution_schema_v4 import (
    canonical_command_sequence_hash,
)
from planning.runtime.ports import DirectRuntimePortProfile


PLANNING_DIR = Path(__file__).resolve().parents[1]
UNITY_BINARY = Path(
    os.environ.get(
        "PLANNING_UNITY_BINARY",
        "/home/xm/XM/xm_ws/src/unity/XMflight.x86_64",
    )
)
BRIDGE_BINARY = Path(
    os.environ.get(
        "PLANNING_BRIDGE_BINARY",
        str(PLANNING_DIR / "devel/lib/planning/unity_bridge_node"),
    )
)
RUNTIME_ID = "worker-00-real-m1"
EXECUTION_ID = 4435486932233309751
EPISODE_ID = "m1-episode-00"
RESET_ID = "m1-reset-00"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_tcp(port: int, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            continue
    raise AssertionError("TCP endpoint did not become ready: {}".format(port))


def _wait_for_zmq_connected(socket_obj, timeout_s: float = 10.0) -> None:
    monitor = socket_obj.get_monitor_socket()
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            poller = zmq.Poller()
            poller.register(monitor, zmq.POLLIN)
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            if monitor not in dict(poller.poll(min(remaining_ms, 100))):
                continue
            event = recv_monitor_message(monitor, flags=zmq.NOBLOCK)
            if event["event"] == zmq.EVENT_CONNECTED:
                return
            if event["event"] == zmq.EVENT_CONNECT_DELAYED:
                continue
        raise AssertionError("ZMQ socket did not connect before timeout")
    finally:
        monitor.close(0)


def _terminate_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8)


def _command_wire() -> bytes:
    frames = []
    canonical_frames = []
    for frame_index in range(25):
        command_id = 10_000 + frame_index
        action = [0.0, 0.0, 0.0, 0.0]
        frames.append([frame_index, command_id, action])
        canonical_frames.append({
            "frame_index": frame_index,
            "command_id": command_id,
            "action": action,
        })
    assert canonical_command_sequence_hash(canonical_frames)
    return msgpack.packb(
        [
            4,
            4,
            None,
            None,
            0,
            -1,
            EXECUTION_ID,
            -1,
            25,
            frames,
        ],
        use_bin_type=True,
    )


def _result_mapping(raw: bytes):
    result = msgpack.unpackb(raw, raw=False)
    result["command_sequence_hash"] = result["command_sequence_hash"].hex()
    result["result_payload_hash"] = result["result_payload_hash"].hex()
    return result


def _commit_wire(result) -> bytes:
    return msgpack.packb(
        {
            "schema_version": 4,
            "message_type": "PrimitiveExecutionResultCommit",
            "commit_status": "COMMITTED",
            "runtime_instance_id": result["runtime_instance_id"],
            "execution_id": result["execution_id"],
            "command_sequence_hash": bytes.fromhex(result["command_sequence_hash"]),
            "result_payload_hash": bytes.fromhex(result["result_payload_hash"]),
        },
        use_bin_type=True,
    )


def _receipt_ack_wire(result) -> bytes:
    return msgpack.packb(
        {
            "schema_version": 4,
            "message_type": "PrimitiveExecutionResultReceiptAck",
            "runtime_instance_id": result["runtime_instance_id"],
            "execution_id": result["execution_id"],
            "receipt_status": "RECEIVED",
            "result_payload_hash": bytes.fromhex(result["result_payload_hash"]),
            "command_sequence_hash": bytes.fromhex(result["command_sequence_hash"]),
        },
        use_bin_type=True,
    )


def _ready_wire() -> bytes:
    return msgpack.packb(
        {"schema_version": 4, "message_type": "PrimitiveExecutionResultReady"},
        use_bin_type=True,
    )


def _recv_one(socket_obj, timeout_ms=1000):
    poller = zmq.Poller()
    poller.register(socket_obj, zmq.POLLIN)
    events = dict(poller.poll(timeout_ms))
    if socket_obj not in events:
        return None
    return socket_obj.recv()


@contextmanager
def _running_real_runtime(tmp_path, *, observation_snapshot_port=None):
    assert UNITY_BINARY.is_file(), "Unity Player is missing: {}".format(UNITY_BINARY)
    assert BRIDGE_BINARY.is_file(), "bridge binary is missing: {}".format(BRIDGE_BINARY)

    profile = DirectRuntimePortProfile.allocate(runtime_instance_id=RUNTIME_ID)
    if observation_snapshot_port is not None:
        profile = replace(profile, unity_snapshot_port=int(observation_snapshot_port))
    master_port = profile.master_port
    cmd_port = profile.command_port
    state_port = profile.state_port
    depth_port = profile.depth_port
    result_port = profile.unity_result_port
    python_port = profile.python_result_port
    snapshot_port = profile.unity_snapshot_port
    env = dict(os.environ)
    env.update({
        "ROS_MASTER_URI": "http://127.0.0.1:{}".format(master_port),
        "ROS_LOG_DIR": str(tmp_path / "ros_logs"),
    })
    (tmp_path / "ros_logs").mkdir()
    unity_log = (tmp_path / "unity.log").open("w", encoding="utf-8")
    bridge_log = (tmp_path / "bridge.log").open("w", encoding="utf-8")
    roscore_log = (tmp_path / "roscore.log").open("w", encoding="utf-8")
    roscore = subprocess.Popen(
        ["roscore", "-p", str(master_port)],
        cwd=str(PLANNING_DIR),
        env=env,
        stdout=roscore_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    bridge = None
    unity = None
    context = zmq.Context()
    command = context.socket(zmq.PUB)
    state = context.socket(zmq.SUB)
    depth = context.socket(zmq.SUB)
    result_receiver = context.socket(zmq.DEALER)
    for socket_obj in (command, state, depth, result_receiver):
        socket_obj.setsockopt(zmq.LINGER, 0)
    state.setsockopt(zmq.SUBSCRIBE, b"")
    depth.setsockopt(zmq.SUBSCRIBE, b"")
    metrics_path = tmp_path / "bridge_result_metrics.json"
    try:
        _wait_for_tcp(master_port)
        bridge = subprocess.Popen(
            [
                str(BRIDGE_BINARY),
                "_unity_host:=127.0.0.1",
                *(
                    "_{}:={}".format(name, value)
                    for name, value in profile.bridge_launch_args().items()
                ),
                "_python_result_bind_host:=127.0.0.1",
                "_python_snapshot_bind_host:=127.0.0.1",
                "_execution_result_metrics_path:={}".format(metrics_path),
                "_spin_hz:=200",
            ],
            cwd=str(PLANNING_DIR),
            env=env,
            stdout=bridge_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state.connect("tcp://127.0.0.1:{}".format(state_port))
        depth.connect("tcp://127.0.0.1:{}".format(depth_port))
        command.connect("tcp://127.0.0.1:{}".format(cmd_port))
        result_receiver.setsockopt(zmq.IDENTITY, b"python-m1-consumer")
        result_receiver.connect("tcp://127.0.0.1:{}".format(python_port))
        _wait_for_zmq_connected(result_receiver)
        result_receiver.send(_ready_wire())

        unity = subprocess.Popen(
            [
                str(UNITY_BINARY),
                "-screen-width", "320",
                "-screen-height", "240",
                "-screen-fullscreen", "0",
                *profile.unity_launch_argv(),
                "-primitiveResultSchema", "4",
                "-endpointEpisodeId", EPISODE_ID,
                "-endpointResetId", RESET_ID,
                "-logFile", str(tmp_path / "unity-player.log"),
            ],
            cwd=str(UNITY_BINARY.parent),
            env=env,
            stdout=unity_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        yield {
            "context": context,
            "command": command,
            "state": state,
            "depth": depth,
            "result_receiver": result_receiver,
            "bridge": bridge,
            "unity": unity,
            "roscore": roscore,
            "metrics_path": metrics_path,
            "endpoint_identity_path": tmp_path / "endpoint_observation_identity.json",
            "log_path": tmp_path / "unity-player.log",
        }
    finally:
        for socket_obj in (command, state, depth, result_receiver):
            socket_obj.close(0)
        context.term()
        _terminate_process(unity)
        _terminate_process(bridge)
        _terminate_process(roscore)
        unity_log.close()
        bridge_log.close()
        roscore_log.close()


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


@pytest.mark.unity
def test_real_unity_complete_reaches_bridge_python_consumer_once(tmp_path):
    state_records = {}
    depth_records = {}
    applied_frames = []
    result = None

    with _running_real_runtime(tmp_path) as runtime:
        initial_state = _recv_one(runtime["state"], timeout_ms=5000)
        assert initial_state is not None, "real Unity did not publish an initial state"
        depth_poller = zmq.Poller()
        depth_poller.register(runtime["depth"], zmq.POLLIN)
        depth_events = dict(depth_poller.poll(5000))
        assert runtime["depth"] in depth_events, (
            "real Unity did not publish an initial depth frame; log={}".format(
                runtime["log_path"]
            )
        )
        initial_depth = runtime["depth"].recv_multipart()
        assert len(initial_depth) == 2, "real Unity depth frame is not multipart"
        runtime["command"].send(_command_wire())
        poller = zmq.Poller()
        poller.register(runtime["state"], zmq.POLLIN)
        poller.register(runtime["depth"], zmq.POLLIN)
        poller.register(runtime["result_receiver"], zmq.POLLIN)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            endpoint_time = (
                int(result["endpoint_sim_time_ns"])
                if result is not None
                else None
            )
            endpoint_capture_id = (
                str(result["endpoint_observation_ref"]["depth_id"])
                if result is not None
                else None
            )
            if result is not None and endpoint_time in state_records and \
                    endpoint_capture_id in depth_records:
                break
            events = dict(poller.poll(100))
            if runtime["state"] in events:
                raw = runtime["state"].recv()
                message = msgpack.unpackb(raw, raw=False)
                if len(message) >= 14:
                    state_id = int(message[1])
                    sim_time_ns = int(message[2])
                    execution_id = int(message[10])
                    frame_index = int(message[11])
                    if execution_id == EXECUTION_ID and frame_index >= 0:
                        applied_frames.append((frame_index, state_id, sim_time_ns))
                    state_records[sim_time_ns] = {
                        "state_id": state_id,
                        "sim_time_ns": sim_time_ns,
                        "physics_time_ns": sim_time_ns,
                        "episode_id": str(message[14]) if len(message) >= 17 else EPISODE_ID,
                        "reset_id": str(message[15]) if len(message) >= 17 else RESET_ID,
                        "runtime_instance_id": (
                            str(message[16]) if len(message) >= 17 else RUNTIME_ID
                        ),
                        "message": message,
                    }
            if runtime["depth"] in events:
                parts = runtime["depth"].recv_multipart()
                if len(parts) == 2:
                    message = msgpack.unpackb(parts[0], raw=False)
                    if len(message) >= 3:
                            sim_time_ns = int(message[2])
                            capture_id = "depth-{}".format(message[1])
                            depth_records[capture_id] = {
                            "sim_time_ns": sim_time_ns,
                            "capture_id": capture_id,
                            "message": message,
                            "data": parts[1],
                        }
            if runtime["result_receiver"] in events:
                result = _result_mapping(runtime["result_receiver"].recv())
                runtime["result_receiver"].send(_receipt_ack_wire(result))

        assert result is not None, "real Unity did not produce a result; log={}".format(
            runtime["log_path"]
        )
        assert result["status"] == "COMPLETE"
        assert int(result["requested_frame_count"]) == 25
        assert int(result["applied_frame_count"]) == 25
        assert int(result["last_applied_frame_index"]) == 24
        assert sorted(set(frame[0] for frame in applied_frames)) == list(range(25))

        endpoint_state_id = int(result["endpoint_state_id"])
        endpoint_time = int(result["endpoint_sim_time_ns"])
        endpoint_state = state_records.get(endpoint_time)
        endpoint_capture_id = str(
            result["endpoint_observation_ref"]["depth_id"]
        )
        endpoint_depth = depth_records.get(endpoint_capture_id)
        assert endpoint_state is not None, "endpoint state timestamp was not observed"
        assert endpoint_depth is not None, (
            "endpoint depth identity was not observed: capture_id={} "
            "depth_count={} known={}".format(
                endpoint_capture_id,
                len(depth_records),
                sorted(depth_records),
            )
        )
        assert endpoint_state["state_id"] == endpoint_state_id
        depth_message = endpoint_depth["message"]
        assert len(depth_message) >= 19
        assert int(depth_message[13]) == endpoint_state_id
        assert int(depth_message[14]) == endpoint_time
        assert int(depth_message[15]) >= int(depth_message[14])
        assert str(depth_message[16]) == EPISODE_ID
        assert str(depth_message[17]) == RESET_ID
        assert str(depth_message[18]) == RUNTIME_ID
        runtime["endpoint_identity_path"].write_text(
            json.dumps(
                {
                    "endpoint_state_id": endpoint_state_id,
                    "physics_time_ns": int(depth_message[14]),
                    "depth_id": endpoint_capture_id,
                    "capture_time_ns": int(depth_message[15]),
                    "episode_id": str(depth_message[16]),
                    "reset_id": str(depth_message[17]),
                    "runtime_instance_id": str(depth_message[18]),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        store = IndexedEndpointObservationStore(
            runtime_instance_id=RUNTIME_ID,
            episode_id=EPISODE_ID,
            reset_id=RESET_ID,
        )
        store.put_state(endpoint_state_id, endpoint_state)
        store.put_depth(
            endpoint_capture_id,
            {
                "runtime_instance_id": RUNTIME_ID,
                "depth_id": endpoint_capture_id,
                "state_id": endpoint_state_id,
                "physics_time_ns": int(depth_message[14]),
                "capture_time_ns": int(depth_message[15]),
                "sim_time_ns": int(depth_message[15]),
                "episode_id": EPISODE_ID,
                "reset_id": RESET_ID,
                "capture_id": endpoint_capture_id,
                "data": endpoint_depth["data"],
            },
        )
        committer = _Committer()
        replay = _Replay()
        consumer = PrimitiveExecutionResultConsumer(
            endpoint_provider=store,
            transition_committer=committer,
            replay_appender=replay,
        )
        outcome = consumer.consume(result)
        assert outcome.status == "COMMITTED"
        assert len(committer.transitions) == 1
        assert len(replay.transitions) == 1

        runtime["result_receiver"].send(_commit_wire(result))
        commit_ack = msgpack.unpackb(
            _recv_one(runtime["result_receiver"], timeout_ms=5000),
            raw=False,
        )
        assert commit_ack["commit_status"] == "COMMITTED"

    assert len(applied_frames) >= 25
    frame_indices = [frame[0] for frame in applied_frames]
    assert sorted(set(frame_indices)) == list(range(25))
    assert len(frame_indices) == len(set(frame_indices))
    metrics = json.loads(runtime["metrics_path"].read_text(encoding="utf-8"))
    assert metrics["accepted"] == 1
    assert metrics["ack"] == 1
    assert metrics["python_relay"] == 1
    assert metrics["commit"] == 1
    assert metrics["pending"] == 0
    assert metrics["receipt"] == 1
    assert metrics["pending_receipt"] == 0


@pytest.mark.unity
def test_real_unity_player_retrieves_exact_terminal_snapshot_once(tmp_path):
    """P0-O4.5: Player cache, bridge request, response, exact-hash validation."""
    result = None
    metrics = None

    with _running_real_runtime(tmp_path) as runtime:
        assert _recv_one(runtime["state"], timeout_ms=5000) is not None, (
            "real Unity did not publish its initial state"
        )
        depth_poller = zmq.Poller()
        depth_poller.register(runtime["depth"], zmq.POLLIN)
        assert runtime["depth"] in dict(depth_poller.poll(5000)), (
            "real Unity did not publish its initial depth"
        )
        runtime["depth"].recv_multipart()

        runtime["command"].send(_command_wire())
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if result is None:
                raw = _recv_one(runtime["result_receiver"], timeout_ms=100)
                if raw is not None:
                    result = _result_mapping(raw)
                    runtime["result_receiver"].send(_receipt_ack_wire(result))

            if runtime["metrics_path"].is_file():
                candidate = json.loads(runtime["metrics_path"].read_text(encoding="utf-8"))
                if candidate.get("snapshot_response", 0) >= 1:
                    metrics = candidate
                    break

        assert result is not None, "real Unity did not return COMPLETE"
        assert result["status"] == "COMPLETE"
        assert result["endpoint_observation_ref"] == {
            "schema_version": 4,
            "runtime_instance_id": RUNTIME_ID,
            "episode_id": EPISODE_ID,
            "reset_id": RESET_ID,
            "state_id": int(result["endpoint_state_id"]),
            "depth_id": result["endpoint_observation_ref"]["depth_id"],
            "sim_time_ns": int(result["endpoint_sim_time_ns"]),
        }
        assert metrics is not None, (
            "bridge did not record a snapshot response from the real Unity Player"
        )
        assert metrics["snapshot_request"] == 1
        assert metrics["snapshot_response"] == 1
        assert metrics["snapshot_hash_match"] == 1
        assert metrics["snapshot_cache_hit"] == 1
        assert metrics["snapshot_protocol_error"] == 0
        assert metrics["protocol_error"] == 0
