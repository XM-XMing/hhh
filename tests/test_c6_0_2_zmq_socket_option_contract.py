"""C6.0.2 socket-option criticality and failure-policy contracts."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest


P = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.unit


EXPECTED_PRODUCTION_OPTIONS = {
    "ZMQ_LINGER": "REQUIRED_FOR_LIFECYCLE_CLEANUP",
    "ZMQ_SNDHWM": "OPTIONAL_PERFORMANCE",
    "ZMQ_RCVHWM": "OPTIONAL_PERFORMANCE",
    "ZMQ_ROUTER_HANDOVER": "REQUIRED_FOR_EXACTLY_ONCE",
    "ZMQ_SUBSCRIBE": "REQUIRED_FOR_CORRECTNESS",
}


def _transport_sources() -> str:
    return "\n".join(
        (P / "src/transport" / name).read_text(encoding="utf-8")
        for name in (
            "command_transport.cpp",
            "telemetry_transport.cpp",
            "result_transport.cpp",
            "snapshot_transport.cpp",
            # The shared helper is the single implementation owner after
            # C++ transport-option consolidation.
            "zmq_socket.cpp",
        )
    )


def test_every_production_option_has_one_canonical_criticality():
    contract = (P / "src/transport/zmq_option_contract.cpp").read_text(
        encoding="utf-8"
    )
    for option, criticality in EXPECTED_PRODUCTION_OPTIONS.items():
        assert re.search(
            r"\{" + option + r",\s*\"" + option + r"\".*" + criticality,
            contract,
            re.DOTALL,
        )
    assert "ZmqSocketOptionCriticality::UNKNOWN" in contract


def test_all_production_setsockopt_calls_are_classified():
    source = _transport_sources()
    observed = set(
        re.findall(r"(?:SetInt|SetBytes)\(\s*(ZMQ_[A-Z0-9_]+)", source)
    )
    assert observed == set(EXPECTED_PRODUCTION_OPTIONS)


def test_required_failure_is_fail_closed_and_unknown_is_fail_closed():
    socket_source = (P / "src/transport/zmq_socket.cpp").read_text(
        encoding="utf-8"
    )
    bridge_source = (P / "src/transport/bridge_transport.cpp").read_text(
        encoding="utf-8"
    )
    assert "criticality == ZmqSocketOptionCriticality::UNKNOWN" in socket_source
    assert "return false" in socket_source
    assert "if (!command_.Init" in bridge_source
    assert "Close();" in bridge_source


def test_optional_failure_records_degraded_option_instead_of_changing_normal_values():
    socket_source = (P / "src/transport/zmq_socket.cpp").read_text(
        encoding="utf-8"
    )
    node_source = (P / "src/bridge/bridge_node.cpp").read_text(encoding="utf-8")
    metrics_source = (P / "src/bridge/bridge_node.cpp").read_text(
        encoding="utf-8"
    )
    assert "OPTIONAL_PERFORMANCE" in socket_source
    assert "degraded_socket_options" in socket_source
    assert "degraded_socket_options" in node_source
    assert 'stream << ",\\"degraded_socket_options\\":["' in metrics_source
    assert "receive_hwm = 4" in (
        P / "include/planning/transport/bridge_transport.hpp"
    ).read_text(encoding="utf-8")
    assert "send_hwm = 4" in (
        P / "include/planning/transport/bridge_transport.hpp"
    ).read_text(encoding="utf-8")


def test_partial_initialization_and_required_fault_cannot_expose_gateways():
    source = (P / "src/bridge/bridge_node.cpp").read_text(encoding="utf-8")
    transport_source = (P / "src/transport/bridge_transport.cpp").read_text(
        encoding="utf-8"
    )
    assert "if (!transport_.Init" in source
    assert "return false;" in source[source.index("if (!transport_.Init") :]
    assert "Close();" in transport_source
    assert "zmq_ctx_term" in transport_source


def test_failed_required_initialization_does_not_accept_requests_and_releases_ports():
    source = (P / "src/bridge/bridge_node.cpp").read_text(encoding="utf-8")
    assert source.index("if (!transport_.Init") < source.index(
        "telemetry_bridge_ ="
    )
    assert "transport_.Close();" in source
    transport_source = (P / "src/transport/bridge_transport.cpp").read_text(
        encoding="utf-8"
    )
    assert "command_.Close();" in transport_source
    assert "zmq_ctx_term(context_)" in transport_source


def test_degraded_summary_is_explicitly_diagnostic_only():
    source = (P / "src/bridge/bridge_node.cpp").read_text(encoding="utf-8")
    assert 'stream << "{\\\"accepted\\\":"' in source
    assert 'stream << ",\\"degraded_socket_options\\":["' in source
    assert "socket type" not in source.lower()


def test_cpp_unknown_option_fails_closed(tmp_path):
    source = P / "tests/cpp/zmq_socket_option_contract.cpp"
    binary = tmp_path / "zmq_socket_option_contract"
    compile_result = subprocess.run(
        [
            "/usr/bin/g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-I",
            str(P / "include"),
            str(source),
            str(P / "src/transport/zmq_option_contract.cpp"),
            str(P / "src/transport/zmq_socket.cpp"),
            "-lzmq",
            "-o",
            str(binary),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        [str(binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "C++ ZMQ option contract tests: 2 passed" in run_result.stdout
