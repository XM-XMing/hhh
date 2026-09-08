"""C6 characterization checks for the split Bridge/protocol/transport tree.

The stress case is opt-in because it starts a ROS master and a real Bridge
process.  It uses only local ZMQ peers; no Unity Player or formal data is
involved.
"""

from __future__ import annotations

import importlib.util
import hashlib
import os
from pathlib import Path
import re
import time

import msgpack
import pytest
import zmq


P = Path(__file__).resolve().parents[1]


def _bridge_test_module():
    path = P / "tests/test_unity_bridge_result_runtime_integration.py"
    spec = importlib.util.spec_from_file_location("c6_bridge_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_split_bridge_has_one_formal_source_owner_per_runtime_target():
    cmake = (P / "CMakeLists.txt").read_text(encoding="utf-8")
    sources = (
        "src/unity_bridge_main.cpp",
        "src/bridge/bridge_node.cpp",
        "src/bridge/command_gateway.cpp",
        "src/bridge/result_gateway.cpp",
        "src/bridge/snapshot_gateway.cpp",
        "src/bridge/reset_gateway.cpp",
        "src/bridge/telemetry_bridge.cpp",
        "src/protocol/endpoint_observation_snapshot_wire.cpp",
        "src/protocol/telemetry_wire.cpp",
        "src/transport/bridge_transport.cpp",
        "src/transport/zmq_socket.cpp",
        "src/transport/command_transport.cpp",
        "src/transport/telemetry_transport.cpp",
        "src/transport/result_transport.cpp",
        "src/transport/snapshot_transport.cpp",
    )

    assert not (P / "src/unity_bridge_node.cpp").exists()
    assert all(cmake.count(source) == 1 for source in sources)
    assert "zmq_ctx_new" in (P / "src/transport/bridge_transport.cpp").read_text(
        encoding="utf-8"
    )
    assert "zmq_ctx_term" in (P / "src/transport/bridge_transport.cpp").read_text(
        encoding="utf-8"
    )
    assert "zmq_ctx_new" not in (P / "src/bridge/bridge_node.cpp").read_text(
        encoding="utf-8"
    )


@pytest.mark.unit
def test_protocol_header_and_language_neutral_fixture_bytes_match_current_contract():
    header = P / "include/planning/protocol/xm_protocol.hpp"
    assert hashlib.sha256(header.read_bytes()).hexdigest() == (
        "e45566e4bec13fb8e1bb4d0e6e7b0162b83cffdec5019aefb447fc66df04d6e3"
    )

    fixture_root = P / "tests/fixtures"
    fixture_files = sorted(
        path.relative_to(fixture_root)
        for path in fixture_root.rglob("*")
        if path.is_file()
    )
    assert len(fixture_files) == 27
    expected_sha256 = {
        Path("observation_snapshot_v4/observation_ref.msgpack.hex"):
            "33e28f0fb6b53de43ba01e012ab160895424ef9db9910299ddb76c9712bff6d3",
        Path("observation_snapshot_v4/snapshot.msgpack.hex"):
            "91b22577a1e0c83bf8a4240b7657090be1b6233e27e435a0801b872e2b171cfd",
        Path("observation_snapshot_v4/snapshot_request.msgpack.hex"):
            "efe9fba8cb15277b83679224b2a43aa8f81bef4f4d342dccb09104c21545ca47",
        Path("primitive_execution_v4/complete.result_msgpack.hex"):
            "998c01a474c8f95a44f0e27df1040d9db01e2e80a3cee9759d9c44e75eeaf6e4",
        Path("primitive_execution_v4/complete.result_payload_hash.sha256"):
            "6905efd66ea66729d6875a303361f9f7e2a4f5c635061d033f334434ef10cfd2",
        Path("primitive_execution_v4/complete.command_sequence_hash.hex"):
            "055322190be0d59e9ba778a16bd07a2d84c805416b0be9f921762e3095b2928f",
    }
    for relative, expected in expected_sha256.items():
        assert hashlib.sha256((fixture_root / relative).read_bytes()).hexdigest() == expected


@pytest.mark.unit
def test_managed_ports_match_and_direct_profile_matches_frozen_defaults():
    from planning.runtime.ports import DirectRuntimePortProfile
    from planning.runtime.worker import WorkerRuntimeSpec

    spec = WorkerRuntimeSpec.from_mapping(
        {
            "worker_id": 0,
            "runtime_instance_id": "worker-00-c6",
            "master_port": 11621,
            "ros_home": "/tmp/xmflight-c6-worker-00",
            "command_port": 10553,
            "state_port": 10554,
            "depth_port": 12554,
            "python_command_port": 10559,
            "unity_command_port": 10560,
            "unity_result_port": 10555,
            "python_result_port": 10556,
            "unity_snapshot_port": 10557,
            "python_snapshot_port": 10558,
        }
    )
    assert spec.bridge_launch_args()["execution_result_port"] == 10555
    assert spec.bridge_launch_args()["observation_snapshot_port"] == 10557
    assert spec.unity_launch_args()["reliableCommandPort"] == 10560
    assert spec.python_backend_config()["snapshot_endpoint"].endswith(":10558")

    p_config = (P / "config/unity.yaml").read_text(encoding="utf-8")
    assert "depth_port: 12254" in p_config
    p_bridge = (P / "src/bridge/bridge_node.cpp").read_text(encoding="utf-8")
    direct_profile = DirectRuntimePortProfile.frozen_defaults()
    expected_direct = {
        "master_port": 11321,
        "command_port": 10253,
        "state_port": 10254,
        "depth_port": 11254,
        "unity_result_port": 11255,
        "unity_snapshot_port": 11256,
        "python_result_port": 11257,
        "python_snapshot_port": 11258,
        "python_command_port": 11259,
        "unity_command_port": 11260,
    }
    p_managed_default = re.search(r"depth_port:\s*(\d+)", p_config)
    assert direct_profile.port_mapping() == expected_direct
    assert '"depth_port", transport_config_.depth_port, 11254' not in p_bridge
    assert p_managed_default and p_managed_default.group(1) == "12254"
    assert str(direct_profile.depth_port) != p_managed_default.group(1)


@pytest.mark.ros
def test_transport_only_1000_interleaved_results_have_exact_accounting(tmp_path):
    if os.environ.get("PLANNING_C6_STRESS") != "1":
        pytest.skip("set PLANNING_C6_STRESS=1 to run the local 1000-row fixture")

    live = _bridge_test_module()
    from planning.protocol.primitive_execution_schema_v4 import (
        canonical_result_payload_hash,
    )

    def rejected_result(runtime_instance_id, execution_id):
        result = live._complete_result_mapping()
        result.update(
            {
                "runtime_instance_id": runtime_instance_id,
                "execution_id": execution_id,
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
        return result

    with live._running_bridge(tmp_path, python_retry_interval_s=60.0) as runtime:
        peers = []
        python = runtime.context.socket(zmq.DEALER)
        python.setsockopt(zmq.IDENTITY, b"python-c6-stress")
        python.setsockopt(zmq.LINGER, 0)
        python.connect("tcp://127.0.0.1:{}".format(runtime.python_port))
        python.send(live._ready_wire())
        try:
            for index in range(4):
                peer = runtime.context.socket(zmq.DEALER)
                peer.setsockopt(
                    zmq.IDENTITY, "unity-c6-{:02d}".format(index).encode("ascii")
                )
                peer.setsockopt(zmq.LINGER, 0)
                peer.connect("tcp://127.0.0.1:{}".format(runtime.result_port))
                peers.append(peer)

            # Exactly one intentional malformed payload establishes the
            # fail-closed error counter without entering a pending transaction.
            peers[0].send(b"c6-intentional-malformed")
            poller = zmq.Poller()
            poller.register(peers[0], zmq.POLLIN)
            assert not dict(poller.poll(250))

            for index in range(1000):
                runtime_id = "worker-{:02d}-c6".format(index % 4)
                result = rejected_result(runtime_id, 100000 + index)
                peer = peers[index % 4]
                peer.send(live._result_wire_for_mapping(result))

                unity_ack = live._recv(peer, timeout_ms=5000)[0]
                ack = msgpack.unpackb(unity_ack, raw=False)
                assert ack["message_type"] == "PrimitiveExecutionResultAck"
                assert ack["ack_status"] == "DURABLE_RECEIVED"
                assert ack["runtime_instance_id"] == runtime_id
                assert ack["execution_id"] == result["execution_id"]

                relayed = msgpack.unpackb(live._recv(python, timeout_ms=5000)[0], raw=False)
                assert relayed["runtime_instance_id"] == runtime_id
                assert relayed["execution_id"] == result["execution_id"]
                assert relayed["result_payload_hash"] == result["result_payload_hash"]
                python.send(live._receipt_ack_wire(result))
                # Let the Bridge consume the receipt before the commit arrives;
                # this keeps the fixture on the same ordered application path
                # as the worker runtime and avoids a retry-shaped extra frame.
                time.sleep(0.02)
                python.send(live._commit_wire(result))
                commit_ack = msgpack.unpackb(
                    live._recv(python, timeout_ms=5000)[0], raw=False
                )
                assert commit_ack.get("commit_status") == "COMMITTED", {
                    "index": index,
                    "message_type": commit_ack.get("message_type"),
                    "message": commit_ack,
                }
        finally:
            for peer in peers:
                peer.close(0)
            python.close(0)

    metrics = __import__("json").loads(
        runtime.metrics_path.read_text(encoding="utf-8")
    )
    assert metrics["accepted"] == 1000
    assert metrics["ack"] == 1000
    assert metrics["python_relay"] == 1000
    assert metrics["receipt"] == 1000
    assert metrics["commit"] == 1000
    assert metrics["pending"] == 0
    assert metrics["pending_receipt"] == 0
    assert metrics["protocol_error"] == 1
