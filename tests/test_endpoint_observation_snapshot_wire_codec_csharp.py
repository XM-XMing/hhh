"""Compile and run the Unity snapshot channel wire codec contract."""

from pathlib import Path
import os
import subprocess

import pytest

from helpers.unity_final_paths import (
    UNITY_MANAGED,
    UNITY_MESSAGEPACK,
    require_protocol_core,
    require_snapshot_wire_codec,
)

ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path("/home/xm/XM/XMflight")
MONO_ROOT = Path(
    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/bin-linux64"
)


@pytest.mark.unit
def test_unity_endpoint_observation_snapshot_wire_codec_contract(tmp_path):
    binary = tmp_path / "endpoint_observation_snapshot_wire_codec_contract.exe"
    protocol_sources = require_protocol_core()
    codec = require_snapshot_wire_codec()
    compile_result = subprocess.run(
        [
            str(MONO_ROOT / "mcs"),
            "-langversion:7.2",
            "-r:{}".format(
                MONO_ROOT.parent / "lib/mono/unityjit-linux/Facades/netstandard.dll"
            ),
            "-r:{}".format(
                MONO_ROOT.parent / "lib/mono/unityjit-linux/Facades/System.Memory.dll"
            ),
            "-r:{}".format(
                MONO_ROOT.parent / "lib/mono/unityjit-linux/Facades/System.Buffers.dll"
            ),
            "-r:{}".format(
                MONO_ROOT.parent / "lib/mono/net_4_x-linux/System.Runtime.CompilerServices.Unsafe.dll"
            ),
            "-r:{}".format(UNITY_MESSAGEPACK),
            "-out:{}".format(binary),
            *(str(path) for path in protocol_sources),
            str(codec),
            str(ROOT / "tests/csharp/EndpointObservationSnapshotWireCodecContract.cs"),
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
        env={
            **os.environ,
            "MONO_PATH": "{}:{}".format(
                UNITY_MANAGED,
                "{}:{}".format(
                    MONO_ROOT.parent / "lib/mono/unityjit-linux/Facades",
                    "{}:{}".format(
                        MONO_ROOT.parent / "lib/mono/net_4_x-linux",
                        "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/"
                        "PlaybackEngines/LinuxStandaloneSupport/Variations/"
                        "linux64_player_nondevelopment_mono/Data/Managed",
                    ),
                ),
            ),
        },
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "C# endpoint snapshot wire codec tests: 2 passed" in run_result.stdout
