"""Real-process bridge integration for the v4 execution-result path."""

import os
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import socket
import signal
import subprocess
import time
from types import SimpleNamespace

import msgpack
import pytest
import zmq

from planning.protocol.primitive_execution_schema_v4 import canonical_result_payload_hash
from planning.runtime.ports import DirectRuntimePortProfile


PLANNING_DIR = Path(__file__).resolve().parents[1]
BRIDGE_BINARY = Path(
    os.environ.get(
        "PLANNING_BRIDGE_BINARY",
        str(PLANNING_DIR / "devel/lib/planning/unity_bridge_node"),
    )
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_tcp(port: int, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.02)
    raise AssertionError("roscore did not become ready on port {}".format(port))


def _assert_core_metrics(metrics, expected):
    """Keep the historical result counters strict while allowing shared extras."""
    assert {key: metrics.get(key) for key in expected} == expected


def _complete_result_mapping():
    return {
        "schema_version": 4,
        "message_type": "PrimitiveExecutionResult",
        "runtime_instance_id": "worker-00-runtime-test",
        "execution_id": 72623859790382856,
        "status": "COMPLETE",
        "requested_frame_count": 25,
        "applied_frame_count": 25,
        "first_applied_state_id": 4000,
        "endpoint_state_id": 4024,
        "last_applied_frame_index": 24,
        "reason_code": "NONE",
        "command_sequence_hash": bytes.fromhex(
            "82e7f39c8f9bb8b6ea5a42cee108cf1e33a3ee872ba47a51167c4ffd25bed62c"
        ),
        "endpoint_sim_time_ns": 5000000000,
        "endpoint_observation_ref": {
            "schema_version": 4,
            "runtime_instance_id": "worker-00-runtime-test",
            "episode_id": "episode-v4-0001",
            "reset_id": "reset-v4-0001",
            "state_id": 4024,
            "depth_id": "depth-v4-0001",
            "sim_time_ns": 5000000000,
        },
        "result_generation": 0,
        "result_payload_hash": bytes.fromhex(
            "8527192b92ff82f9c702811d632ea672670ce81f99ec7cb5a5bec073c7376e09"
        ),
    }


def _complete_result_wire() -> bytes:
    result = _complete_result_mapping()
    return msgpack.packb(result, use_bin_type=True)


def _rejected_result_wire() -> bytes:
    result = _complete_result_mapping()
    result.update(
        {
            "execution_id": 0,
            "status": "REJECTED",
            "applied_frame_count": 0,
            "first_applied_state_id": None,
            "endpoint_state_id": None,
            "last_applied_frame_index": -1,
            "reason_code": "SCHEMA_MISMATCH",
            "endpoint_sim_time_ns": None,
            "endpoint_observation_ref": None,
        }
    )
    semantic = dict(result)
    semantic["command_sequence_hash"] = semantic["command_sequence_hash"].hex()
    semantic.pop("result_payload_hash")
    result["result_payload_hash"] = bytes.fromhex(
        canonical_result_payload_hash(semantic)
    )
    return _result_wire_for_mapping(result)


def _result_wire_for_mapping(result_mapping) -> bytes:
    return msgpack.packb(result_mapping, use_bin_type=True)


def _result_wire_for_runtime(runtime_instance_id: str) -> bytes:
    result = _complete_result_mapping()
    result["runtime_instance_id"] = runtime_instance_id
    result["endpoint_observation_ref"] = dict(result["endpoint_observation_ref"])
    result["endpoint_observation_ref"]["runtime_instance_id"] = runtime_instance_id
    semantic = dict(result)
    semantic["command_sequence_hash"] = semantic["command_sequence_hash"].hex()
    semantic.pop("result_payload_hash")
    result["result_payload_hash"] = bytes.fromhex(
        canonical_result_payload_hash(semantic)
    )
    return _result_wire_for_mapping(result)


def _conflicting_result_wire() -> bytes:
    result = _complete_result_mapping()
    changed_command_hash = bytearray(result["command_sequence_hash"])
    changed_command_hash[0] ^= 1
    result["command_sequence_hash"] = bytes(changed_command_hash)
    semantic = dict(result)
    semantic["command_sequence_hash"] = semantic["command_sequence_hash"].hex()
    semantic.pop("result_payload_hash")
    result["result_payload_hash"] = bytes.fromhex(
        canonical_result_payload_hash(semantic)
    )
    return _result_wire_for_mapping(result)


def _recv(socket_obj, timeout_ms=3000):
    poller = zmq.Poller()
    poller.register(socket_obj, zmq.POLLIN)
    events = dict(poller.poll(timeout_ms))
    assert socket_obj in events, "timed out waiting for ZMQ message"
    return socket_obj.recv_multipart()


def _ready_wire() -> bytes:
    return msgpack.packb(
        {"schema_version": 4, "message_type": "PrimitiveExecutionResultReady"},
        use_bin_type=True,
    )


def _commit_wire(
    result_mapping,
    result_hash=None,
    command_sequence_hash=None,
) -> bytes:
    return msgpack.packb(
        {
            "schema_version": 4,
            "message_type": "PrimitiveExecutionResultCommit",
            "commit_status": "COMMITTED",
            "runtime_instance_id": result_mapping["runtime_instance_id"],
            "execution_id": result_mapping["execution_id"],
            "command_sequence_hash": command_sequence_hash
            if command_sequence_hash is not None
            else result_mapping["command_sequence_hash"],
            "result_payload_hash": result_hash
            if result_hash is not None
            else result_mapping["result_payload_hash"],
        },
        use_bin_type=True,
    )


def _receipt_ack_wire(result_mapping, *, result_hash=None, command_hash=None) -> bytes:
    return msgpack.packb(
        {
            "schema_version": 4,
            "message_type": "PrimitiveExecutionResultReceiptAck",
            "runtime_instance_id": result_mapping["runtime_instance_id"],
            "execution_id": result_mapping["execution_id"],
            "receipt_status": "RECEIVED",
            "result_payload_hash": result_hash
            if result_hash is not None
            else result_mapping["result_payload_hash"],
            "command_sequence_hash": command_hash
            if command_hash is not None
            else result_mapping["command_sequence_hash"],
        },
        use_bin_type=True,
    )


def _connect_python_receiver(runtime, identity=b"python-test-00"):
    python = runtime.context.socket(zmq.DEALER)
    python.setsockopt(zmq.IDENTITY, identity)
    python.setsockopt(zmq.LINGER, 0)
    python.connect("tcp://127.0.0.1:{}".format(runtime.python_port))
    python.send(_ready_wire())
    return python


def _wait_for_bridge_diagnostic_event(
    path, *, event, timeout_s=3.0, execution_id=None, monitor_event=None
):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            for row in rows:
                if row.get("event") != event:
                    continue
                if execution_id is not None and row.get("execution_id") != execution_id:
                    continue
                if monitor_event is not None and row.get("monitor_event") != monitor_event:
                    continue
                return row
        time.sleep(0.01)
    raise AssertionError(
        "timed out waiting for bridge diagnostic event={} execution_id={}".format(
            event, execution_id
        )
    )


def _next_complete_result(result_mapping):
    result = dict(result_mapping)
    result["execution_id"] = int(result["execution_id"]) + 1
    semantic = dict(result)
    semantic["command_sequence_hash"] = semantic["command_sequence_hash"].hex()
    semantic.pop("result_payload_hash")
    result["result_payload_hash"] = bytes.fromhex(
        canonical_result_payload_hash(semantic)
    )
    return result


@contextmanager
def _running_bridge(
    tmp_path,
    *,
    python_diagnostics_path=None,
    python_retry_interval_s=None,
    python_stale_retry_threshold=None,
    observation_snapshot_port=None,
    python_snapshot_port=None,
):
    assert BRIDGE_BINARY.is_file(), "bridge binary is missing: {}".format(BRIDGE_BINARY)
    profile = DirectRuntimePortProfile.allocate(runtime_instance_id="bridge-test-00")
    if observation_snapshot_port is not None:
        profile = replace(profile, unity_snapshot_port=int(observation_snapshot_port))
    if python_snapshot_port is not None:
        profile = replace(profile, python_snapshot_port=int(python_snapshot_port))
    master_port = profile.master_port
    cmd_port = profile.command_port
    state_port = profile.state_port
    depth_port = profile.depth_port
    result_port = profile.unity_result_port
    python_port = profile.python_result_port
    snapshot_port = profile.unity_snapshot_port
    python_snapshot_port = profile.python_snapshot_port
    metrics_path = tmp_path / "bridge_result_metrics.json"
    env = dict(os.environ)
    env.update(
        {
            "ROS_MASTER_URI": "http://127.0.0.1:{}".format(master_port),
            "ROS_LOG_DIR": str(tmp_path / "ros_logs"),
        }
    )
    (tmp_path / "ros_logs").mkdir()
    roscore = subprocess.Popen(
        ["roscore", "-p", str(master_port)],
        cwd=str(PLANNING_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    bridge = None
    context = zmq.Context()
    unity = context.socket(zmq.DEALER)
    unity.setsockopt(zmq.IDENTITY, b"unity-test-00")
    unity.setsockopt(zmq.LINGER, 0)
    try:
        _wait_for_tcp(master_port)
        bridge_args = [str(BRIDGE_BINARY), "_unity_host:=127.0.0.1"]
        bridge_args.extend(
            "_{}:={}".format(name, value)
            for name, value in profile.bridge_launch_args().items()
        )
        bridge_args.extend([
            "_python_result_bind_host:=127.0.0.1",
            "_python_snapshot_bind_host:=127.0.0.1",
            "_execution_result_metrics_path:={}".format(metrics_path),
            "_spin_hz:=200",
        ])
        if python_diagnostics_path is not None:
            bridge_args.append(
                "_python_result_diagnostics_path:={}".format(
                    python_diagnostics_path
                )
            )
        if python_retry_interval_s is not None:
            bridge_args.append(
                "_python_result_retry_interval_s:={}".format(
                    python_retry_interval_s
                )
            )
        if python_stale_retry_threshold is not None:
            bridge_args.append(
                "_python_result_stale_retry_threshold:={}".format(
                    python_stale_retry_threshold
                )
            )
        bridge = subprocess.Popen(
            bridge_args,
            cwd=str(PLANNING_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        unity.connect("tcp://127.0.0.1:{}".format(result_port))
        yield SimpleNamespace(
            context=context,
            unity=unity,
            bridge=bridge,
            roscore=roscore,
            result_port=result_port,
            python_port=python_port,
            observation_snapshot_port=snapshot_port,
            python_snapshot_port=python_snapshot_port,
            metrics_path=metrics_path,
        )
    finally:
        unity.close(0)
        context.term()
        if bridge is not None:
            if bridge.poll() is None:
                bridge.send_signal(signal.SIGINT)
                try:
                    bridge.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    bridge.kill()
                    bridge.wait(timeout=5)
        if roscore.poll() is None:
            roscore.send_signal(signal.SIGINT)
            try:
                roscore.wait(timeout=5)
            except subprocess.TimeoutExpired:
                roscore.kill()
                roscore.wait(timeout=5)


def _snapshot_ready_wire(runtime_instance_id):
    return msgpack.packb(
        {
            "schema_version": 4,
            "message_type": "EndpointObservationSnapshotReady",
            "runtime_instance_id": runtime_instance_id,
        },
        use_bin_type=True,
    )


def _snapshot_response_wire(request, *, observation_ref=None, snapshot_hash=None):
    observation_ref = observation_ref or request["observation_ref"]
    payload = {
        "schema_version": 4,
        "message_type": "EndpointObservationSnapshot",
        "observation_ref": observation_ref,
        "state_bytes": b"state-v4-endpoint",
        "depth_bytes": b"depth-v4-endpoint",
        "execution_id": request["execution_id"],
        "result_payload_hash": request["result_payload_hash"],
        "command_sequence_hash": request["command_sequence_hash"],
    }
    from planning.protocol.endpoint_observation_snapshot_v4 import (
        EndpointObservationSnapshot,
        canonical_snapshot_hash,
    )

    snapshot = EndpointObservationSnapshot.from_mapping(
        {
            "observation_ref": observation_ref,
            "state_bytes": payload["state_bytes"],
            "depth_bytes": payload["depth_bytes"],
        }
    )
    payload["snapshot_hash"] = (
        snapshot_hash if snapshot_hash is not None else bytes.fromhex(canonical_snapshot_hash(snapshot))
    )
    return msgpack.packb(payload, use_bin_type=True)


def _snapshot_missing_wire(request):
    return msgpack.packb(
        {
            "schema_version": 4,
            "message_type": "SnapshotMissing",
            "observation_ref": request["observation_ref"],
            "execution_id": request["execution_id"],
            "result_payload_hash": request["result_payload_hash"],
            "command_sequence_hash": request["command_sequence_hash"],
        },
        use_bin_type=True,
    )


def _connect_snapshot_service(runtime, identity=b"unity-snapshot-test-00"):
    snapshot = runtime.context.socket(zmq.DEALER)
    snapshot.setsockopt(zmq.IDENTITY, identity)
    snapshot.setsockopt(zmq.LINGER, 0)
    snapshot.connect("tcp://127.0.0.1:{}".format(runtime.observation_snapshot_port))
    snapshot.send(_snapshot_ready_wire("worker-00-runtime-test"))
    return snapshot


@pytest.mark.ros
def test_real_bridge_process_retrieves_exact_endpoint_snapshot_after_complete(tmp_path):
    """Bridge process must request, validate, cache, then ACK one exact snapshot."""
    with _running_bridge(tmp_path) as runtime:
        snapshot_service = _connect_snapshot_service(runtime)
        try:
            result = _complete_result_mapping()
            runtime.unity.send(_result_wire_for_mapping(result))
            unity_ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
            assert unity_ack["ack_status"] == "DURABLE_RECEIVED"

            request = msgpack.unpackb(_recv(snapshot_service)[0], raw=False)
            assert request["message_type"] == "SnapshotRequest"
            assert request["observation_ref"] == result["endpoint_observation_ref"]
            assert request["execution_id"] == result["execution_id"]
            assert request["result_payload_hash"] == result["result_payload_hash"]
            assert request["command_sequence_hash"] == result["command_sequence_hash"]

            snapshot_service.send(_snapshot_response_wire(request))
            snapshot_ack = msgpack.unpackb(_recv(snapshot_service)[0], raw=False)
            assert snapshot_ack["message_type"] == "SnapshotAck"
            assert snapshot_ack["observation_ref"] == request["observation_ref"]
        finally:
            snapshot_service.close(0)


@pytest.mark.ros
def test_real_bridge_defers_python_snapshot_response_until_unity_snapshot_arrives(tmp_path):
    """A Python exact lookup must wait for the matching Unity snapshot.

    This is the reset/endpoint race seam: a Python request can arrive after
    Unity has accepted the request but before Unity has returned the snapshot.
    Returning ``BridgeSnapshotMissing`` at that point loses a valid snapshot.
    """
    with _running_bridge(tmp_path) as runtime:
        snapshot_service = _connect_snapshot_service(runtime)
        python_snapshot = runtime.context.socket(zmq.DEALER)
        python_snapshot.setsockopt(zmq.IDENTITY, b"python-snapshot-test-00")
        python_snapshot.setsockopt(zmq.LINGER, 0)
        python_snapshot.connect(
            "tcp://127.0.0.1:{}".format(runtime.python_snapshot_port)
        )
        try:
            result = _complete_result_mapping()
            runtime.unity.send(_result_wire_for_mapping(result))
            _recv(runtime.unity)
            request = msgpack.unpackb(_recv(snapshot_service)[0], raw=False)

            python_request = msgpack.packb(
                {
                    "schema_version": 4,
                    "message_type": "BridgeSnapshotRequest",
                    "observation_ref": request["observation_ref"],
                    "execution_id": request["execution_id"],
                    "result_payload_hash": request["result_payload_hash"],
                    "command_sequence_hash": request["command_sequence_hash"],
                },
                use_bin_type=True,
            )
            python_snapshot.send(python_request)

            # The request is pending in the bridge until Unity answers; an
            # immediate response here would be an incorrect missing result.
            poller = zmq.Poller()
            poller.register(python_snapshot, zmq.POLLIN)
            assert not dict(poller.poll(200))

            snapshot_service.send(_snapshot_response_wire(request))
            response = msgpack.unpackb(_recv(python_snapshot)[0], raw=False)
            assert response["message_type"] == "BridgeSnapshotResponse"
            assert response["observation_ref"] == request["observation_ref"]
            assert response["execution_id"] == request["execution_id"]
        finally:
            python_snapshot.close(0)
            snapshot_service.close(0)


@pytest.mark.ros
def test_real_bridge_exposes_cached_snapshot_to_python_consumer_once(tmp_path):
    """P0-O5.5: Python queries bridge cache, never telemetry, then commits once."""
    from planning.runtime.reliable_endpoint_snapshot_provider import (
        BridgeSnapshotEndpointProvider,
        ZmqBridgeSnapshotRetriever,
    )
    from planning.protocol.primitive_execution_result_consumer import PrimitiveExecutionResultConsumer

    class Committer:
        def __init__(self):
            self.transitions = []
        def commit(self, transition):
            self.transitions.append(transition)

    class Replay:
        def __init__(self):
            self.transitions = []
        def append_once(self, transition):
            self.transitions.append(transition)

    with _running_bridge(tmp_path) as runtime:
        snapshot_service = _connect_snapshot_service(runtime)
        client = None
        python_result_socket = None
        try:
            result = _complete_result_mapping()
            runtime.unity.send(_result_wire_for_mapping(result))
            _recv(runtime.unity)
            request = msgpack.unpackb(_recv(snapshot_service)[0], raw=False)
            snapshot_service.send(_snapshot_response_wire(request))
            _recv(snapshot_service)

            client = ZmqBridgeSnapshotRetriever(
                runtime.context,
                "tcp://127.0.0.1:{}".format(runtime.python_snapshot_port),
                timeout_ms=1000,
            )
            committer, replay = Committer(), Replay()
            consumer = PrimitiveExecutionResultConsumer(
                endpoint_provider=BridgeSnapshotEndpointProvider(client),
                transition_committer=committer,
                replay_appender=replay,
            )
            python_result = msgpack.unpackb(_result_wire_for_mapping(result), raw=False)
            python_result["command_sequence_hash"] = python_result[
                "command_sequence_hash"
            ].hex()
            python_result["result_payload_hash"] = python_result[
                "result_payload_hash"
            ].hex()
            outcome = consumer.consume(python_result)

            assert outcome.status == "COMMITTED", outcome.error
            assert len(committer.transitions) == 1
            assert len(replay.transitions) == 1
            assert committer.transitions[0].endpoint.state == b"state-v4-endpoint"

            python_result_socket = _connect_python_receiver(runtime)
            relayed = msgpack.unpackb(_recv(python_result_socket)[0], raw=False)
            assert relayed["execution_id"] == result["execution_id"]
            python_result_socket.send(_receipt_ack_wire(result))
            python_result_socket.send(_commit_wire(result))
            commit_ack = msgpack.unpackb(_recv(python_result_socket)[0], raw=False)
            assert commit_ack["commit_status"] == "COMMITTED"

        finally:
            if client is not None:
                client.close()
            if python_result_socket is not None:
                python_result_socket.close(0)
            snapshot_service.close(0)

        metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
        assert metrics["result_cache_current_entries"] == 0
        assert metrics["snapshot_cache_current_entries"] == 0
        assert metrics["command_cache_current_entries"] == 0
        assert metrics["pending"] == 0
        assert metrics["pending_snapshot"] == 0


@pytest.mark.ros
def test_real_bridge_process_keeps_missing_snapshot_pending_and_retries(tmp_path):
    with _running_bridge(tmp_path) as runtime:
        snapshot_service = _connect_snapshot_service(runtime)
        try:
            runtime.unity.send(_complete_result_wire())
            _recv(runtime.unity)
            request = msgpack.unpackb(_recv(snapshot_service)[0], raw=False)
            snapshot_service.send(_snapshot_missing_wire(request))
            assert not dict(zmq.Poller().poll(1))
            retry = msgpack.unpackb(_recv(snapshot_service, timeout_ms=1500)[0], raw=False)
            assert retry == request
        finally:
            snapshot_service.close(0)


@pytest.mark.ros
@pytest.mark.parametrize("variant", ["wrong_ref", "wrong_hash"])
def test_real_bridge_process_rejects_wrong_snapshot_identity_or_hash(tmp_path, variant):
    with _running_bridge(tmp_path) as runtime:
        snapshot_service = _connect_snapshot_service(runtime)
        try:
            runtime.unity.send(_complete_result_wire())
            _recv(runtime.unity)
            request = msgpack.unpackb(_recv(snapshot_service)[0], raw=False)
            if variant == "wrong_ref":
                wrong_ref = dict(request["observation_ref"])
                wrong_ref["depth_id"] = "depth-v4-wrong"
                response = _snapshot_response_wire(request, observation_ref=wrong_ref)
            else:
                response = _snapshot_response_wire(request, snapshot_hash=bytes(32))
            snapshot_service.send(response)
            retry = msgpack.unpackb(_recv(snapshot_service, timeout_ms=1500)[0], raw=False)
            assert retry["message_type"] == "SnapshotRequest", "invalid snapshot must not be ACKed"
            assert retry == request
        finally:
            snapshot_service.close(0)


@pytest.mark.ros
def test_real_bridge_process_accepts_duplicate_snapshot_without_overwrite(tmp_path):
    with _running_bridge(tmp_path) as runtime:
        snapshot_service = _connect_snapshot_service(runtime)
        try:
            runtime.unity.send(_complete_result_wire())
            _recv(runtime.unity)
            request = msgpack.unpackb(_recv(snapshot_service)[0], raw=False)
            response = _snapshot_response_wire(request)
            snapshot_service.send(response)
            first_ack = msgpack.unpackb(_recv(snapshot_service)[0], raw=False)
            snapshot_service.send(response)
            duplicate_ack = msgpack.unpackb(_recv(snapshot_service)[0], raw=False)
            assert duplicate_ack == first_ack
        finally:
            snapshot_service.close(0)


@pytest.mark.ros
def test_real_bridge_process_resends_pending_snapshot_after_disconnect_and_ready(tmp_path):
    identity = b"unity-snapshot-reconnect-00"
    with _running_bridge(tmp_path) as runtime:
        first = _connect_snapshot_service(runtime, identity=identity)
        try:
            runtime.unity.send(_complete_result_wire())
            _recv(runtime.unity)
            request = msgpack.unpackb(_recv(first)[0], raw=False)
        finally:
            first.close(0)

        recovered = _connect_snapshot_service(runtime, identity=identity)
        try:
            resent = msgpack.unpackb(_recv(recovered, timeout_ms=1500)[0], raw=False)
            assert resent == request
            recovered.send(_snapshot_response_wire(resent))
            ack = msgpack.unpackb(_recv(recovered)[0], raw=False)
            assert ack["message_type"] == "SnapshotAck"
        finally:
            recovered.close(0)


@pytest.mark.ros
def test_real_bridge_process_accepts_v4_result_and_acks(tmp_path):
    assert BRIDGE_BINARY.is_file(), "bridge binary is missing: {}".format(BRIDGE_BINARY)

    profile = DirectRuntimePortProfile.allocate(
        runtime_instance_id="bridge-result-test-00"
    )
    master_port = profile.master_port
    cmd_port = profile.command_port
    state_port = profile.state_port
    depth_port = profile.depth_port
    result_port = profile.unity_result_port
    python_port = profile.python_result_port
    metrics_path = tmp_path / "bridge_result_metrics.json"

    env = dict(os.environ)
    env.update(
        {
            "ROS_MASTER_URI": "http://127.0.0.1:{}".format(master_port),
            "ROS_LOG_DIR": str(tmp_path / "ros_logs"),
        }
    )
    (tmp_path / "ros_logs").mkdir()
    roscore = subprocess.Popen(
        ["roscore", "-p", str(master_port)],
        cwd=str(PLANNING_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    bridge = None
    context = zmq.Context()
    unity = context.socket(zmq.DEALER)
    unity.setsockopt(zmq.IDENTITY, b"unity-test-00")
    unity.setsockopt(zmq.LINGER, 0)
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
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        unity.connect("tcp://127.0.0.1:{}".format(result_port))
        unity.send(_complete_result_wire())

        poller = zmq.Poller()
        poller.register(unity, zmq.POLLIN)
        events = dict(poller.poll(3000))
        assert unity in events, "bridge did not return an ACK"
        ack = msgpack.unpackb(unity.recv(), raw=False)
        assert ack["schema_version"] == 4
        assert ack["message_type"] == "PrimitiveExecutionResultAck"
        assert ack["ack_status"] == "DURABLE_RECEIVED"
    finally:
        unity.close(0)
        context.term()
        if bridge is not None:
            bridge.send_signal(signal.SIGINT)
            try:
                bridge.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bridge.kill()
                bridge.wait(timeout=5)
        roscore.send_signal(signal.SIGINT)
        try:
            roscore.wait(timeout=5)
        except subprocess.TimeoutExpired:
            roscore.kill()
            roscore.wait(timeout=5)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _assert_core_metrics(metrics, {
        "accepted": 1,
        "duplicate": 0,
        "conflict": 0,
        "pending": 1,
        "pending_receipt": 1,
        "commit": 0,
        "ack": 1,
        "python_relay": 0,
        "receipt": 0,
        "duplicate_receipt": 0,
        "protocol_error": 0,
    })


@pytest.mark.ros
def test_real_bridge_does_not_retrieve_snapshot_for_rejected_result(tmp_path):
    """Only COMPLETE results require an endpoint snapshot retrieval request."""
    with _running_bridge(tmp_path) as runtime:
        runtime.unity.send(_rejected_result_wire())
        ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
        assert ack["message_type"] == "PrimitiveExecutionResultAck"
        assert ack["ack_status"] == "DURABLE_RECEIVED"
    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    assert metrics["protocol_error"] == 0


@pytest.mark.ros
def test_real_bridge_duplicate_result_is_acked_without_second_store_or_relay(tmp_path):
    result_wire = _complete_result_wire()
    with _running_bridge(tmp_path) as runtime:
        runtime.unity.send(result_wire)
        first_ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
        runtime.unity.send(result_wire)
        duplicate_ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
        assert first_ack["ack_status"] == "DURABLE_RECEIVED"
        assert duplicate_ack == first_ack

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    _assert_core_metrics(metrics, {
        "accepted": 1,
        "duplicate": 1,
        "conflict": 0,
        "pending": 1,
        "pending_receipt": 1,
        "commit": 0,
        "ack": 2,
        "python_relay": 0,
        "receipt": 0,
        "duplicate_receipt": 0,
        "protocol_error": 0,
    })


@pytest.mark.ros
def test_real_bridge_conflicting_duplicate_is_protocol_error(tmp_path):
    with _running_bridge(tmp_path) as runtime:
        runtime.unity.send(_complete_result_wire())
        first_ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
        assert first_ack["ack_status"] == "DURABLE_RECEIVED"

        runtime.unity.send(_conflicting_result_wire())
        poller = zmq.Poller()
        poller.register(runtime.unity, zmq.POLLIN)
        assert not dict(poller.poll(500)), "conflicting result must not be ACKed"

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    _assert_core_metrics(metrics, {
        "accepted": 1,
        "duplicate": 0,
        "conflict": 1,
        "pending": 1,
        "pending_receipt": 1,
        "commit": 0,
        "ack": 1,
        "python_relay": 0,
        "receipt": 0,
        "duplicate_receipt": 0,
        "protocol_error": 1,
    })


@pytest.mark.ros
def test_real_bridge_python_reconnect_and_commit_are_exactly_once(tmp_path):
    result = _complete_result_mapping()
    with _running_bridge(tmp_path) as runtime:
        runtime.unity.send(_result_wire_for_mapping(result))
        unity_ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
        assert unity_ack["ack_status"] == "DURABLE_RECEIVED"

        python = _connect_python_receiver(runtime)
        try:
            relayed = msgpack.unpackb(_recv(python)[0], raw=False)
            assert relayed["execution_id"] == result["execution_id"]
            assert relayed["result_payload_hash"] == result["result_payload_hash"]
            python.send(_receipt_ack_wire(result))

            bad_hash = bytes(32)
            python.send(_commit_wire(result, result_hash=bad_hash))
            bad_commit_ack = msgpack.unpackb(_recv(python)[0], raw=False)
            assert bad_commit_ack["commit_status"] == "PROTOCOL_ERROR"

            python.send(_commit_wire(result, command_sequence_hash=bytes(32)))
            bad_command_ack = msgpack.unpackb(_recv(python)[0], raw=False)
            assert bad_command_ack["commit_status"] == "PROTOCOL_ERROR"

            wrong_execution = dict(result)
            wrong_execution["execution_id"] += 1
            python.send(_commit_wire(wrong_execution))
            wrong_execution_ack = msgpack.unpackb(_recv(python)[0], raw=False)
            assert wrong_execution_ack["commit_status"] == "PROTOCOL_ERROR"

            python.send(_commit_wire(result))
            commit_ack = msgpack.unpackb(_recv(python)[0], raw=False)
            assert commit_ack["commit_status"] == "COMMITTED"

            python.send(_commit_wire(result))
            duplicate_commit_ack = msgpack.unpackb(_recv(python)[0], raw=False)
            assert duplicate_commit_ack["commit_status"] == "DUPLICATE"
        finally:
            python.close(0)

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    _assert_core_metrics(metrics, {
        "accepted": 1,
        "duplicate": 0,
        "conflict": 0,
        "pending": 0,
        "pending_receipt": 0,
        "commit": 1,
        "ack": 1,
        "python_relay": 1,
        "receipt": 1,
        "duplicate_receipt": 0,
        "protocol_error": 3,
    })


@pytest.mark.ros
def test_real_bridge_receipt_ack_clears_python_relay_pending(tmp_path):
    result = _complete_result_mapping()
    with _running_bridge(tmp_path) as runtime:
        runtime.unity.send(_result_wire_for_mapping(result))
        unity_ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
        assert unity_ack["ack_status"] == "DURABLE_RECEIVED"

        python = _connect_python_receiver(runtime)
        try:
            relayed = msgpack.unpackb(_recv(python)[0], raw=False)
            assert relayed["execution_id"] == result["execution_id"]
            python.send(_receipt_ack_wire(result))
        finally:
            python.close(0)

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    assert metrics["python_relay"] == 1
    assert metrics["receipt"] == 1
    assert metrics["pending_receipt"] == 0


@pytest.mark.ros
def test_real_bridge_reconnect_retransmits_until_receipt_ack(tmp_path):
    result = _complete_result_mapping()
    with _running_bridge(tmp_path) as runtime:
        runtime.unity.send(_result_wire_for_mapping(result))
        unity_ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
        assert unity_ack["ack_status"] == "DURABLE_RECEIVED"

        first_python = _connect_python_receiver(runtime, b"python-test-00")
        first_relay = msgpack.unpackb(_recv(first_python)[0], raw=False)
        assert first_relay["execution_id"] == result["execution_id"]
        first_python.close(0)

        second_python = _connect_python_receiver(runtime, b"python-test-01")
        try:
            second_relay = msgpack.unpackb(_recv(second_python)[0], raw=False)
            assert second_relay == first_relay
            second_python.send(_receipt_ack_wire(result))
            second_python.send(_receipt_ack_wire(result))
        finally:
            second_python.close(0)

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    assert metrics["python_relay"] == 2
    assert metrics["receipt"] == 1
    assert metrics["duplicate_receipt"] == 1
    assert metrics["pending_receipt"] == 0


@pytest.mark.ros
def test_real_bridge_same_identity_reregistration_recovers_pending_result(tmp_path):
    diagnostics_path = tmp_path / "python_router_diagnostics.jsonl"
    result_a = _complete_result_mapping()
    result_b = _next_complete_result(result_a)
    python_identity = b"python-test-00"

    with _running_bridge(
        tmp_path, python_diagnostics_path=diagnostics_path
    ) as runtime:
        first_python = _connect_python_receiver(runtime, python_identity)
        try:
            runtime.unity.send(_result_wire_for_mapping(result_a))
            assert msgpack.unpackb(_recv(runtime.unity)[0], raw=False)[
                "ack_status"
            ] == "DURABLE_RECEIVED"
            assert msgpack.unpackb(_recv(first_python)[0], raw=False)[
                "execution_id"
            ] == result_a["execution_id"]
            first_python.send(_receipt_ack_wire(result_a))
        finally:
            first_python.close(0)

        disconnected = _wait_for_bridge_diagnostic_event(
            diagnostics_path,
            event="python_router_monitor",
            monitor_event="DISCONNECTED",
        )
        assert disconnected["monitor_event"] == "DISCONNECTED"

        runtime.unity.send(_result_wire_for_mapping(result_b))
        assert msgpack.unpackb(_recv(runtime.unity)[0], raw=False)[
            "ack_status"
        ] == "DURABLE_RECEIVED"
        skipped = _wait_for_bridge_diagnostic_event(
            diagnostics_path,
            event="relay_skip",
            execution_id=result_b["execution_id"],
        )
        assert skipped["peer_available"] is False

        second_python = _connect_python_receiver(runtime, python_identity)
        try:
            relayed_b = msgpack.unpackb(_recv(second_python)[0], raw=False)
            assert relayed_b["execution_id"] == result_b["execution_id"]
            assert relayed_b["result_payload_hash"] == result_b["result_payload_hash"]
            second_python.send(_receipt_ack_wire(result_b))
        finally:
            second_python.close(0)

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    assert metrics["accepted"] == 2
    assert metrics["ack"] == 2
    assert metrics["python_relay"] == 2
    assert metrics["receipt"] == 2
    assert metrics["pending_receipt"] == 0


@pytest.mark.ros
def test_real_bridge_same_identity_handover_replaces_live_route(tmp_path):
    """A new DEALER pipe with the same identity must replace the old route.

    This deliberately does not wait for a ROUTER disconnect monitor event: the
    replacement READY is the handover boundary, not delayed disconnect
    detection.
    """
    result = _complete_result_mapping()
    python_identity = b"python-test-00"

    with _running_bridge(tmp_path) as runtime:
        old_python = _connect_python_receiver(runtime, python_identity)
        try:
            runtime.unity.send(_result_wire_for_mapping(result))
            assert msgpack.unpackb(_recv(runtime.unity)[0], raw=False)[
                "ack_status"
            ] == "DURABLE_RECEIVED"
            first_relay = _recv(old_python)[0]
        finally:
            old_python.close(0)

        # No monitor wait here: this is a same-identity pipe handover.
        new_python = _connect_python_receiver(runtime, python_identity)
        try:
            replacement_relay = _recv(new_python, timeout_ms=3000)[0]
            assert replacement_relay == first_relay
            new_python.send(_receipt_ack_wire(result))
            new_python.send(_commit_wire(result))
            committed = msgpack.unpackb(_recv(new_python)[0], raw=False)
            assert committed["commit_status"] == "COMMITTED"
        finally:
            new_python.close(0)

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    assert metrics["accepted"] == 1
    assert metrics["ack"] == 1
    assert metrics["receipt"] == 1
    assert metrics["commit"] == 1
    assert metrics["pending_receipt"] == 0


@pytest.mark.ros
def test_runner_dealer_result_timeout_recovers_same_identity_pending_result(tmp_path):
    """A runner timeout must recreate its DEALER and re-register its identity."""
    from planning.diagnostics.reliable_single_worker import ResultDealerLifecycle

    result = _complete_result_mapping()
    python_identity = b"worker-00-p0-m2"

    with _running_bridge(tmp_path) as runtime:
        # The validation runner gives the result DEALER its own context, so a
        # stale pipe recovery can tear down its I/O state without disturbing
        # command/state/depth telemetry sockets.
        channel = ResultDealerLifecycle(
            zmq.Context(),
            endpoint="tcp://127.0.0.1:{}".format(runtime.python_port),
            dealer_identity=python_identity,
            connect_timeout_s=3.0,
            owns_context=True,
        )
        try:
            channel.connect_ready()
            runtime.unity.send(_result_wire_for_mapping(result))
            assert msgpack.unpackb(_recv(runtime.unity)[0], raw=False)[
                "ack_status"
            ] == "DURABLE_RECEIVED"

            # The first pipe never receipts the result. Recovery must replace
            # it with the same identity and make the bridge resend the exact
            # immutable payload.
            channel.recover_after_result_timeout()
            recovered = msgpack.unpackb(_recv(channel.socket)[0], raw=False)
            assert recovered["execution_id"] == result["execution_id"]
            assert recovered["result_payload_hash"] == result["result_payload_hash"]
            channel.socket.send(_receipt_ack_wire(result))
            channel.socket.send(_commit_wire(result))
            committed = msgpack.unpackb(_recv(channel.socket)[0], raw=False)
            assert committed["commit_status"] == "COMMITTED"
            assert channel.state == "CONNECTED"
            assert channel.state_history == [
                "STALE",
                "RECONNECTING",
                "READY_SENT",
                "CONNECTED",
                "STALE",
                "RECONNECTING",
                "READY_SENT",
                "CONNECTED",
            ]
        finally:
            channel.close()

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    assert metrics["accepted"] == 1
    assert metrics["receipt"] == 1
    assert metrics["commit"] == 1
    assert metrics["pending_receipt"] == 0


@pytest.mark.ros
def test_runner_dealer_recovers_after_bridge_marks_peer_stale(tmp_path):
    """A recreated same-identity DEALER must re-register after bridge STALE.

    This is the P0-M2 execution-54 shape: receipt retries have already marked
    the bridge peer STALE before the runner notices its result timeout and
    recreates the DEALER.
    """
    from planning.diagnostics.reliable_single_worker import ResultDealerLifecycle

    result = _complete_result_mapping()
    python_identity = b"worker-00-p0-m2-stale"

    with _running_bridge(
        tmp_path,
        python_retry_interval_s=0.05,
        python_stale_retry_threshold=2,
    ) as runtime:
        channel = ResultDealerLifecycle(
            runtime.context,
            endpoint="tcp://127.0.0.1:{}".format(runtime.python_port),
            dealer_identity=python_identity,
            connect_timeout_s=3.0,
        )
        try:
            channel.connect_ready()
            runtime.unity.send(_result_wire_for_mapping(result))
            assert msgpack.unpackb(_recv(runtime.unity)[0], raw=False)[
                "ack_status"
            ] == "DURABLE_RECEIVED"

            # Initial delivery plus two receipt retries mark the bridge peer
            # STALE.  Drain them without issuing a receipt ACK.
            relayed = [_recv(channel.socket, timeout_ms=1000)[0] for _ in range(3)]
            assert relayed[0] == relayed[1] == relayed[2]
            poller = zmq.Poller()
            poller.register(channel.socket, zmq.POLLIN)
            assert not dict(poller.poll(250)), "bridge peer must be STALE"

            channel.recover_after_result_timeout()
            recovered = _recv(channel.socket, timeout_ms=1000)[0]
            assert recovered == relayed[0]
            channel.socket.send(_receipt_ack_wire(result))
            channel.socket.send(_commit_wire(result))
            assert msgpack.unpackb(_recv(channel.socket)[0], raw=False)[
                "commit_status"
            ] == "COMMITTED"
        finally:
            channel.close()

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    assert metrics["accepted"] == 1
    assert metrics["receipt"] == 1
    assert metrics["commit"] == 1
    assert metrics["pending_receipt"] == 0


@pytest.mark.ros
def test_real_bridge_receipt_timeout_marks_peer_stale_until_ready_recovers(tmp_path):
    result_a = _complete_result_mapping()
    result_b = _next_complete_result(result_a)

    with _running_bridge(
        tmp_path,
        python_retry_interval_s=0.05,
        python_stale_retry_threshold=2,
    ) as runtime:
        python = _connect_python_receiver(runtime)
        try:
            runtime.unity.send(_result_wire_for_mapping(result_a))
            assert msgpack.unpackb(_recv(runtime.unity)[0], raw=False)[
                "ack_status"
            ] == "DURABLE_RECEIVED"
            assert msgpack.unpackb(_recv(python)[0], raw=False)["execution_id"] == result_a[
                "execution_id"
            ]
            python.send(_receipt_ack_wire(result_a))
            python.send(_commit_wire(result_a))
            assert msgpack.unpackb(_recv(python)[0], raw=False)["commit_status"] == "COMMITTED"

            runtime.unity.send(_result_wire_for_mapping(result_b))
            assert msgpack.unpackb(_recv(runtime.unity)[0], raw=False)[
                "ack_status"
            ] == "DURABLE_RECEIVED"
            relayed_b = [_recv(python, timeout_ms=1000)[0] for _ in range(3)]
            assert relayed_b[0] == relayed_b[1] == relayed_b[2]

            poller = zmq.Poller()
            poller.register(python, zmq.POLLIN)
            assert not dict(poller.poll(250)), "STALE peer must stop relay attempts"

            python.send(_ready_wire())
            recovered_b = _recv(python, timeout_ms=1000)[0]
            assert recovered_b == relayed_b[0]
            python.send(_receipt_ack_wire(result_b))
            python.send(_commit_wire(result_b))
            assert msgpack.unpackb(_recv(python)[0], raw=False)["commit_status"] == "COMMITTED"
        finally:
            python.close(0)

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    assert metrics["accepted"] == 2
    assert metrics["receipt"] == 2
    assert metrics["commit"] == 2
    assert metrics["duplicate_receipt"] == 0
    assert metrics["pending_receipt"] == 0


@pytest.mark.ros
def test_real_bridge_timeout_retransmits_same_payload_without_receipt(tmp_path):
    result = _complete_result_mapping()
    with _running_bridge(tmp_path, python_retry_interval_s=0.05) as runtime:
        runtime.unity.send(_result_wire_for_mapping(result))
        unity_ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
        assert unity_ack["ack_status"] == "DURABLE_RECEIVED"

        python = _connect_python_receiver(runtime)
        try:
            first = _recv(python)[0]
            second = _recv(python, timeout_ms=2000)[0]
            assert second == first
            assert msgpack.unpackb(second, raw=False)["execution_id"] == result[
                "execution_id"
            ]
        finally:
            python.close(0)

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    assert metrics["python_relay"] >= 2
    assert metrics["pending_receipt"] == 1


@pytest.mark.ros
def test_real_bridge_python_router_diagnostics_capture_peer_and_relay_send(tmp_path):
    diagnostics_path = tmp_path / "python_router_diagnostics.jsonl"
    result = _complete_result_mapping()
    relayed_frames = []

    with _running_bridge(
        tmp_path, python_diagnostics_path=diagnostics_path
    ) as runtime:
        runtime.unity.send(_complete_result_wire())
        unity_ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
        assert unity_ack["ack_status"] == "DURABLE_RECEIVED"

        python = _connect_python_receiver(runtime)
        try:
            relayed_frames = _recv(python)
            assert len(relayed_frames) == 1
            relayed = msgpack.unpackb(relayed_frames[0], raw=False)
            assert relayed["execution_id"] == result["execution_id"]
        finally:
            python.close(0)

    rows = [
        json.loads(line)
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        row["event"] == "python_peer_ready"
        and row["peer_identity_hex"] == b"python-test-00".hex()
        for row in rows
    )
    relay_rows = [row for row in rows if row["event"] == "relay_send"]
    assert relay_rows
    assert relay_rows[0]["execution_id"] == result["execution_id"]
    assert relay_rows[0]["result_payload_hash"] == result[
        "result_payload_hash"
    ].hex()
    assert relay_rows[0]["target_router_identity_hex"] == b"python-test-00".hex()
    assert relay_rows[0]["target_router_identity_length"] == len(
        b"python-test-00"
    )
    assert relay_rows[0]["router_send_frame_count"] == 2
    assert relay_rows[0]["router_send_frame_sizes"] == [
        len(b"python-test-00"),
        len(relayed_frames[0]),
    ]
    assert relay_rows[0]["router_send_frame_0_identity_hex"] == b"python-test-00".hex()
    assert relay_rows[0]["router_send_frame_1_payload_hex"] == relayed_frames[0].hex()
    assert relay_rows[0]["router_send_api_accepted"] is True
    assert relay_rows[0]["router_mandatory_available"] is False
    assert relay_rows[0]["socket_send_hwm"] == 4
    assert relay_rows[0]["send_queue_depth_available"] is False
    assert relay_rows[0]["relay_started_monotonic_ns"] > 0
    assert isinstance(relay_rows[0]["zmq_events"], int)
    assert relay_rows[0]["last_python_ready_monotonic_ns"] > 0
    assert relay_rows[0]["identity_send_rc"] >= 0
    assert relay_rows[0]["payload_send_rc"] >= 0
    assert relay_rows[0]["socket_owner_match"] is True
    assert (
        relay_rows[0]["socket_owner_thread_id"]
        == relay_rows[0]["socket_access_thread_id"]
    )
    ready_rows = [row for row in rows if row["event"] == "python_peer_ready"]
    assert ready_rows
    assert ready_rows[0]["peer_identity_length"] == len(b"python-test-00")
    assert (
        ready_rows[0]["runtime_instance_id_from_dealer_identity"]
        == "python-test-00"
    )
    assert ready_rows[0]["ready_monotonic_ns"] > 0
    assert ready_rows[0]["socket_owner_match"] is True
    assert (
        ready_rows[0]["socket_owner_thread_id"]
        == ready_rows[0]["socket_access_thread_id"]
    )
    monitor_events = {
        row["monitor_event"]
        for row in rows
        if row["event"] == "python_router_monitor"
    }
    assert "HANDSHAKE_SUCCEEDED" in monitor_events
    assert any(
        row.get("monitor_event") == "HANDSHAKE_SUCCEEDED"
        and row.get("endpoint") == "tcp://127.0.0.1:{}".format(runtime.python_port)
        for row in rows
    )


@pytest.mark.ros
def test_real_bridge_worker_identity_isolation(tmp_path):
    with _running_bridge(tmp_path) as runtime:
        runtime.unity.send(_complete_result_wire())
        runtime.unity.send(_result_wire_for_runtime("worker-01-runtime-test"))
        first_ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
        second_ack = msgpack.unpackb(_recv(runtime.unity)[0], raw=False)
        assert first_ack["ack_status"] == "DURABLE_RECEIVED"
        assert second_ack["ack_status"] == "DURABLE_RECEIVED"

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    _assert_core_metrics(metrics, {
        "accepted": 2,
        "duplicate": 0,
        "conflict": 0,
        "pending": 2,
        "pending_receipt": 2,
        "commit": 0,
        "ack": 2,
        "python_relay": 0,
        "receipt": 0,
        "duplicate_receipt": 0,
        "protocol_error": 0,
    })


@pytest.mark.ros
def test_real_bridge_schema_mismatch_fails_fast(tmp_path):
    invalid = _complete_result_mapping()
    invalid["schema_version"] = 3
    with _running_bridge(tmp_path) as runtime:
        runtime.unity.send(_result_wire_for_mapping(invalid))
        poller = zmq.Poller()
        poller.register(runtime.unity, zmq.POLLIN)
        assert not dict(poller.poll(500)), "schema mismatch must not be ACKed"

    metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
    _assert_core_metrics(metrics, {
        "accepted": 0,
        "duplicate": 0,
        "conflict": 0,
        "pending": 0,
        "pending_receipt": 0,
        "commit": 0,
        "ack": 0,
        "python_relay": 0,
        "receipt": 0,
        "duplicate_receipt": 0,
        "protocol_error": 1,
    })
