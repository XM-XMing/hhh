"""Compile and run the Unity-side reliable reset codec contract."""

from pathlib import Path
import os
import subprocess

import pytest

from helpers.unity_final_paths import UNITY_MANAGED, UNITY_MESSAGEPACK


ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path("/home/xm/XM/XMflight")
UNITY_PROTOCOL_ROOT = UNITY_ROOT / "Assets/Scripts/Runtime/Protocol"
MONO_ROOT = Path(
    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/bin-linux64"
)
NETSTANDARD = (
    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/"
    "lib/mono/net_4_x-linux/Facades/netstandard.dll"
)
UNITY_CORE = Path(
    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/Managed/UnityEngine/"
    "UnityEngine.CoreModule.dll"
)
UNSAFE = Path(
    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/"
    "lib/mono/net_4_x-linux/System.Runtime.CompilerServices.Unsafe.dll"
)


@pytest.mark.unit
def test_csharp_reset_codec_contract(tmp_path):
    binary = tmp_path / "primitive_reset_v4_contract.exe"
    compile_result = subprocess.run(
        [
            str(MONO_ROOT / "mcs"),
            "-langversion:7.2",
            "-r:{}".format(UNITY_MESSAGEPACK),
            "-r:{}".format(NETSTANDARD),
            "-r:{}".format(UNITY_CORE),
            "-r:{}".format(UNSAFE),
            "-out:{}".format(binary),
            str(UNITY_PROTOCOL_ROOT / "ProtocolModels.cs"),
            str(UNITY_PROTOCOL_ROOT / "PrimitiveResetV4.cs"),
            str(ROOT / "tests/csharp/PrimitiveResetV4Contract.cs"),
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
        env=dict(
            os.environ,
            MONO_PATH="{}:{}:{}".format(
                UNITY_MANAGED,
                UNITY_CORE.parent,
                UNSAFE.parent,
            ),
        ),
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "C# reset contract tests: 4 passed" in run_result.stdout
