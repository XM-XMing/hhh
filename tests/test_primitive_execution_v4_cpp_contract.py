"""Compile and run the public C++ v4 codec contract harness."""

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_cpp_codec_matches_language_neutral_golden_vectors(tmp_path):
    compiler = "/usr/bin/g++"
    source = ROOT / "tests/cpp/primitive_execution_v4_contract.cpp"
    binary = tmp_path / "primitive_execution_v4_contract"
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++14",
            "-O2",
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
        [str(binary), str(ROOT / "tests/fixtures/primitive_execution_v4")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr

