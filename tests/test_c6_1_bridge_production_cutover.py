"""Static C6.1 contracts for the single formal split Bridge owner."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from planning.runtime.bridge_identity import (
    BRIDGE_BINARY_NAME,
    BRIDGE_SOURCE_FILES,
    bridge_identity_manifest,
    bridge_source_manifest_sha256,
    planning_bridge_binary,
)


P = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit


BRIDGE_CPP_SOURCES = (
    "src/unity_bridge_main.cpp",
    "src/bridge/bridge_node.cpp",
    "src/bridge/bridge_util.cpp",
    "src/bridge/command_gateway.cpp",
    "src/bridge/result_gateway.cpp",
    "src/bridge/snapshot_gateway.cpp",
    "src/bridge/reset_gateway.cpp",
    "src/bridge/telemetry_bridge.cpp",
    "src/protocol/endpoint_observation_snapshot_wire.cpp",
    "src/protocol/telemetry_wire.cpp",
    "src/transport/bridge_transport.cpp",
    "src/transport/zmq_option_contract.cpp",
    "src/transport/zmq_socket.cpp",
    "src/transport/command_transport.cpp",
    "src/transport/telemetry_transport.cpp",
    "src/transport/result_transport.cpp",
    "src/transport/snapshot_transport.cpp",
)


def _cmake_bridge_source_block() -> str:
    source = (P / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(
        r"set\(PLANNING_BRIDGE_SOURCES\n(?P<body>.*?)\n\)\n"
        r"add_executable\(unity_bridge_node \$\{PLANNING_BRIDGE_SOURCES\}\)",
        source,
        re.DOTALL,
    )
    assert match is not None
    return match.group("body")


def test_cmake_has_one_canonical_bridge_target_and_no_flat_implementation():
    cmake = (P / "CMakeLists.txt").read_text(encoding="utf-8")
    block = _cmake_bridge_source_block()
    assert not (P / "src/unity_bridge_node.cpp").exists()
    assert "src/unity_bridge_node.cpp" not in cmake
    assert all(block.count(path) == 1 for path in BRIDGE_CPP_SOURCES)
    assert cmake.count("add_executable(unity_bridge_node") == 1
    assert cmake.count("install(TARGETS") == 1


def test_split_owner_definitions_are_unique_and_compatibility_headers_are_thin():
    definitions = {
        "UnityBridgeNode::Init": 0,
        "CommandGateway::PrimitiveExecutionCallback": 0,
        "ResultGateway::PollExecutionResults": 0,
        "SnapshotGateway::PollObservationSnapshots": 0,
        "ResetGateway::ReceivePythonReset": 0,
        "TelemetryBridge::PollState": 0,
        "BridgeTransport::Init": 0,
        "ZmqSocket::Open": 0,
    }
    for path in (P / "src").rglob("*.cpp"):
        text = path.read_text(encoding="utf-8")
        for symbol in definitions:
            definitions[symbol] += text.count(symbol + "(")
    assert definitions == {symbol: 1 for symbol in definitions}

    wrappers = (
        P / "include/planning/xm_protocol.hpp",
        P / "include/planning/primitive_execution_command_wire.hpp",
        P / "include/planning/primitive_execution_result_wire.hpp",
    )
    for wrapper in wrappers:
        body = wrapper.read_text(encoding="utf-8").strip().splitlines()
        assert body[0] == "#pragma once"
        assert len(body) == 2
        assert "#include <planning/protocol/" in body[1]


def test_formal_runtime_wiring_never_defaults_to_the_original_bridge_binary():
    formal_files = (
        P / "python/planning/diagnostics/reliable_single_worker.py",
        P / "python/planning/contracts/offpolicy.py",
        P / "python/planning/runtime/identity.py",
        P / "scripts/train_awac.py",
        P / "launch/unity_bridge.launch",
        P / "CMakeLists.txt",
        P / "package.xml",
    )
    old_binary = "/home/xm/XM/xm_ws/devel/lib/planning/unity_bridge_node"
    old_source = "src/unity_bridge_node.cpp"
    for path in formal_files:
        text = path.read_text(encoding="utf-8")
        assert old_binary not in text
        assert old_source not in text

    assert planning_bridge_binary(P) == (
        P.parent.parent / "devel/lib/planning" / BRIDGE_BINARY_NAME
    )
    launch = (P / "launch/unity_bridge.launch").read_text(encoding="utf-8")
    assert '<node pkg="planning" type="unity_bridge_node" name="unity_bridge_node"' in launch


def test_bridge_source_manifest_is_complete_and_build_metadata_declares_owner():
    source = (P / "CMakeLists.txt").read_text(encoding="utf-8")
    block = _cmake_bridge_source_block()
    assert all(path.is_file() for path in (P / relative for relative in BRIDGE_SOURCE_FILES))
    assert all(relative in source or relative not in BRIDGE_CPP_SOURCES for relative in BRIDGE_SOURCE_FILES)
    assert bridge_source_manifest_sha256(P) == bridge_identity_manifest(P)[
        "source_manifest_sha256"
    ]

    package = (P / "package.xml").read_text(encoding="utf-8")
    assert 'bridge_binary="unity_bridge_node"' in package
    assert 'bridge_owner="BridgeNode"' in package
    assert 'bridge_implementation="split"' in package
    assert 'bridge_protocol_schema="4"' in package
    assert 'bridge_source_manifest="python/planning/runtime/bridge_identity.py"' in package
    assert all(path in block for path in BRIDGE_CPP_SOURCES)


def test_runtime_identity_manifest_records_bridge_owner_protocol_and_source():
    identity_source = (P / "python/planning/runtime/identity.py").read_text(
        encoding="utf-8"
    )
    runner_source = (
        P / "python/planning/diagnostics/reliable_single_worker.py"
    ).read_text(encoding="utf-8")
    assert '"bridge_identity": bridge_identity_manifest()' in identity_source
    assert '"protocol": {' in runner_source
    assert 'from planning.protocol.constants import PRIMITIVE_FRAME_COUNT, PROTOCOL_VERSION' in runner_source
    assert '"schema_version": PROTOCOL_VERSION' in runner_source
    assert "bridge_identity_manifest(planning_root)" in runner_source
