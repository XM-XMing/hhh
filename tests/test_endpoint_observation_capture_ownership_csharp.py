"""RED contract for Unity endpoint capture callback ownership."""

from pathlib import Path
import subprocess

import pytest

from helpers.unity_final_paths import require_capture_ownership

ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path("/home/xm/XM/XMflight")
MONO_ROOT = Path(
    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/bin-linux64"
)


@pytest.mark.unit
def test_endpoint_capture_callback_ownership_contract(tmp_path):
    compiler = MONO_ROOT / "mcs"
    runtime = MONO_ROOT / "mono"
    source = ROOT / "tests/csharp/EndpointObservationCaptureOwnershipContract.cs"
    ownership = require_capture_ownership()
    binary = tmp_path / "endpoint_observation_capture_ownership_contract.exe"
    compile_result = subprocess.run(
        [
            str(compiler),
            "-langversion:7.2",
            "-out:{}".format(binary),
            str(ownership),
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
    assert "C# endpoint ownership tests: 6 passed" in run_result.stdout
