"""Compile and run the Unity-only authoritative endpoint cache contract."""

from pathlib import Path
import subprocess

import pytest

from helpers.unity_final_paths import (
    UNITY_MESSAGEPACK,
    require_observation_cache,
    require_protocol_core,
)

ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path("/home/xm/XM/XMflight")
MONO_ROOT = Path("/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/bin-linux64")


@pytest.mark.unit
def test_unity_endpoint_observation_cache_contract(tmp_path):
    binary = tmp_path / "endpoint_observation_cache_contract.exe"
    protocol_sources = require_protocol_core()
    cache = require_observation_cache()
    compile_result = subprocess.run(
        [
            str(MONO_ROOT / "mcs"),
            "-langversion:7.2",
            "-r:{}".format(UNITY_MESSAGEPACK),
            "-out:{}".format(binary),
            *(str(path) for path in protocol_sources),
            str(cache),
            str(ROOT / "tests/csharp/EndpointObservationCacheContract.cs"),
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
