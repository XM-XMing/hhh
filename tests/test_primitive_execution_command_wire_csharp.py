"""Cross-language command wire decoding contract."""

from pathlib import Path
import os
import subprocess

import pytest

from planning.protocol.primitive_execution_command_v4 import build_command_wire
from helpers.unity_final_paths import (
    UNITY_MANAGED,
    UNITY_MESSAGEPACK,
    require_command_admission,
    require_command_wire_codec,
    require_protocol_core,
)


ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path("/home/xm/XM/XMflight")
MONO_ROOT = Path(
    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/bin-linux64"
)


def _frames():
    return [
        {
            "frame_index": index,
            "command_id": 1000 + index,
            "action": [0.1, 0.2, 0.3, 0.4],
        }
        for index in range(25)
    ]


@pytest.mark.unit
def test_csharp_decodes_python_command_wire(tmp_path):
    _, wire = build_command_wire(
        runtime_instance_id="worker-00",
        execution_id=44000000000001,
        frames=_frames(),
    )
    fixture = tmp_path / "primitive_command.msgpack"
    fixture.write_bytes(wire)

    source = ROOT / "tests/csharp/PrimitiveExecutionCommandWireContract.cs"
    codec = require_command_wire_codec()
    admission = require_command_admission()
    protocol_sources = require_protocol_core()
    binary = tmp_path / "primitive_execution_command_wire_contract.exe"
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
                MONO_ROOT.parent
                / "lib/mono/net_4_x-linux/System.Runtime.CompilerServices.Unsafe.dll"
            ),
            "-out:{}".format(binary),
            "-r:{}".format(UNITY_MESSAGEPACK),
            *(str(path) for path in protocol_sources),
            str(admission),
            str(codec),
            str(source),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(MONO_ROOT / "mono"), str(binary), str(fixture)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "MONO_PATH": ":".join(
                str(path)
                for path in (
                UNITY_MANAGED,
                    MONO_ROOT.parent / "lib/mono/unityjit-linux/Facades",
                    MONO_ROOT.parent / "lib/mono/net_4_x-linux",
                    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/Managed/UnityEngine",
                    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/"
                    "PlaybackEngines/LinuxStandaloneSupport/Variations/"
                    "linux64_player_nondevelopment_mono/Data/Managed",
                )
            ),
        },
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "C# command wire tests: 1 passed" in run_result.stdout
