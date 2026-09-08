"""Compile and run the Unity v4 command admission RED/GREEN contract."""

from pathlib import Path
import subprocess

import pytest

from helpers.unity_final_paths import (
    UNITY_MESSAGEPACK,
    require_command_admission,
    require_protocol_core,
)

ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path("/home/xm/XM/XMflight")
MONO_ROOT = Path(
    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/bin-linux64"
)


@pytest.mark.unit
def test_csharp_command_admission_contract(tmp_path):
    compiler = MONO_ROOT / "mcs"
    runtime = MONO_ROOT / "mono"
    messagepack = UNITY_MESSAGEPACK
    source = ROOT / "tests/csharp/PrimitiveExecutionCommandAdmissionContract.cs"
    admission = require_command_admission()
    protocol_sources = require_protocol_core()
    binary = tmp_path / "primitive_execution_command_admission_contract.exe"
    compile_result = subprocess.run(
        [
            str(compiler),
            "-langversion:7.2",
            "-out:{}".format(binary),
            "-r:{}".format(messagepack),
            *(str(path) for path in protocol_sources),
            str(admission),
            str(source),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        [str(runtime), str(binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "C# command admission tests: 5 passed" in run_result.stdout
