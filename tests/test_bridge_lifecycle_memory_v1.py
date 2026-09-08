"""Red/green coverage for bounded reliable bridge lifecycle storage."""

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_cpp_bridge_lifecycle_memory_contract(tmp_path):
    source = ROOT / "tests/cpp/bridge_lifecycle_memory_v1.cpp"
    binary = tmp_path / "bridge_lifecycle_memory_v1"
    compile_result = subprocess.run(
        [
            "g++",
            "-std=c++14",
            "-Wall",
            "-Wextra",
            "-I",
            str(ROOT / "include"),
            "-I",
            "/home/xm/XM/xm_ws/devel/include",
            "-I",
            "/opt/ros/noetic/include",
            str(source),
            "-lcrypto",
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
    assert "C++ bridge lifecycle memory tests: 3 passed" in run_result.stdout


@pytest.mark.unit
def test_cpp_reset_gateway_lifecycle_memory_contract(tmp_path):
    source = ROOT / "tests/cpp/reset_gateway_lifecycle_memory_v1.cpp"
    binary = tmp_path / "reset_gateway_lifecycle_memory_v1"
    compile_result = subprocess.run(
        [
            "g++",
            "-std=c++14",
            "-Wall",
            "-Wextra",
            "-I",
            str(ROOT / "include"),
            "-I",
            "/home/xm/XM/xm_ws/devel/include",
            "-I",
            "/opt/ros/noetic/include",
            str(source),
            str(ROOT / "src/bridge/reset_gateway.cpp"),
            str(ROOT / "src/bridge/bridge_util.cpp"),
            str(ROOT / "src/transport/zmq_socket.cpp"),
            str(ROOT / "src/transport/zmq_option_contract.cpp"),
            "-lzmq",
            "-lcrypto",
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
    assert "C++ reset gateway lifecycle memory tests: 1 passed" in run_result.stdout
