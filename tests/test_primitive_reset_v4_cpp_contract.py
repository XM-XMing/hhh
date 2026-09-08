"""Compile and run the C++ reliable reset wire contract."""

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_cpp_reset_wire_contract(tmp_path):
    binary = tmp_path / "primitive_reset_v4_contract"
    compile_result = subprocess.run(
        [
            "/usr/bin/g++",
            "-std=c++14",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            str(ROOT / "include"),
            str(ROOT / "tests/cpp/primitive_reset_v4_contract.cpp"),
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
    assert "C++ reset contract tests: 4 passed" in run_result.stdout
