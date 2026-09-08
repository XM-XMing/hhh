"""Compile and run the C# v4 endpoint snapshot golden-vector harness."""

from pathlib import Path
import subprocess

import pytest

from helpers.unity_final_paths import (
    UNITY_MESSAGEPACK,
    require_protocol_core,
    require_protocol_identity_registry,
)

ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path("/home/xm/XM/XMflight")
MONO_ROOT = Path("/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/bin-linux64")


@pytest.mark.unit
def test_csharp_observation_snapshot_codec_matches_golden_vectors(tmp_path):
    binary = tmp_path / "observation_snapshot_v4_contract.exe"
    protocol_sources = require_protocol_core()
    identity_registry = require_protocol_identity_registry()
    compile_result = subprocess.run(
        [
            str(MONO_ROOT / "mcs"),
            "-langversion:7.2",
            "-r:{}".format(UNITY_MESSAGEPACK),
            "-out:{}".format(binary),
            *(str(path) for path in protocol_sources),
            str(identity_registry),
            str(ROOT / "tests/csharp/ObservationSnapshotV4Contract.cs"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        [
            str(MONO_ROOT / "mono"),
            str(binary),
            str(ROOT / "tests/fixtures/observation_snapshot_v4"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr
