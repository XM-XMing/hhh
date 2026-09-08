"""Compile and run Unity's pure endpoint snapshot service contract."""

from pathlib import Path
import subprocess

import pytest

from helpers.unity_final_paths import (
    UNITY_MESSAGEPACK,
    require_observation_cache,
    require_protocol_core,
    require_snapshot_service,
)

ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path("/home/xm/XM/XMflight")
MONO_ROOT = Path(
    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/bin-linux64"
)


@pytest.mark.unit
def test_unity_endpoint_observation_snapshot_service_contract(tmp_path):
    binary = tmp_path / "endpoint_observation_snapshot_service_contract.exe"
    protocol_sources = require_protocol_core()
    cache = require_observation_cache()
    service = require_snapshot_service()
    compile_result = subprocess.run(
        [
            str(MONO_ROOT / "mcs"),
            "-langversion:7.2",
            "-r:{}".format(UNITY_MESSAGEPACK),
            "-out:{}".format(binary),
            *(str(path) for path in protocol_sources),
            str(cache),
            str(service),
            str(ROOT / "tests/csharp/EndpointObservationSnapshotServiceContract.cs"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        [str(MONO_ROOT / "mono"), str(binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "C# endpoint snapshot service tests: 4 passed" in run_result.stdout
