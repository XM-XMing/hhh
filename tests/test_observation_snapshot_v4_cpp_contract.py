"""Compile and run the C++ v4 endpoint snapshot golden-vector harness."""

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_cpp_observation_snapshot_codec_matches_golden_vectors(tmp_path):
    binary = tmp_path / "observation_snapshot_v4_contract"
    compile_result = subprocess.run(
        [
            "/usr/bin/g++",
            "-std=c++14",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            str(ROOT / "include"),
            str(ROOT / "tests/cpp/observation_snapshot_v4_contract.cpp"),
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
        [str(binary), str(ROOT / "tests/fixtures/observation_snapshot_v4")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr
