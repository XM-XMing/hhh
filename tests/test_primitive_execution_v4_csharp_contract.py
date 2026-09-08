"""Compile and run the standalone Unity C# v4 codec contract harness."""

from pathlib import Path
import subprocess

import pytest

from helpers.unity_final_paths import UNITY_MESSAGEPACK


ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path("/home/xm/XM/XMflight")
UNITY_PROTOCOL_ROOT = UNITY_ROOT / "Assets/Scripts/Runtime/Protocol"
MONO_ROOT = Path("/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/bin-linux64")


@pytest.mark.unit
def test_csharp_codec_matches_language_neutral_golden_vectors(tmp_path):
    compiler = MONO_ROOT / "mcs"
    runtime = MONO_ROOT / "mono"
    messagepack = UNITY_MESSAGEPACK
    source = ROOT / "tests/csharp/PrimitiveExecutionV4Contract.cs"
    binary = tmp_path / "primitive_execution_v4_contract.exe"
    compile_result = subprocess.run(
        [
            str(compiler),
            "-langversion:7.2",
            "-r:{}".format(messagepack),
            "-out:{}".format(binary),
            str(UNITY_PROTOCOL_ROOT / "ProtocolModels.cs"),
            str(UNITY_PROTOCOL_ROOT / "ProtocolCanonical.cs"),
            str(source),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        [str(runtime), str(binary), str(ROOT / "tests/fixtures/primitive_execution_v4")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr
