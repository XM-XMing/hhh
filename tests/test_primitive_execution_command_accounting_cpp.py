"""Focused C++ command accounting/audit contract."""

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_cpp_primitive_execution_command_accounting_contract(tmp_path):
    source = ROOT / "tests/cpp/primitive_execution_command_accounting_contract.cpp"
    binary = tmp_path / "primitive_execution_command_accounting_contract"
    compile_result = subprocess.run(
        [
            "g++",
            "-std=c++14",
            "-Wall",
            "-Wextra",
            "-I",
            str(ROOT / "include"),
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
