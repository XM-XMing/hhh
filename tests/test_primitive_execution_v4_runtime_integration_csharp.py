"""Compile and run the first Unity runtime-event integration contract slice."""

from pathlib import Path
import subprocess

import pytest

from helpers.unity_final_paths import (
    UNITY_MESSAGEPACK,
    require_execution_controller,
    require_protocol_constants,
    require_protocol_core,
    require_result_lifecycle,
    require_result_transport_adapter,
    require_runtime_integration,
)

ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path("/home/xm/XM/XMflight")
MONO_ROOT = Path(
    "/home/xm/Unity/Hub/Editor/2022.3.62f2c1/Editor/Data/MonoBleedingEdge/bin-linux64"
)


@pytest.mark.unit
def test_csharp_runtime_event_integration_contract(tmp_path):
    compiler = MONO_ROOT / "mcs"
    runtime = MONO_ROOT / "mono"
    messagepack = UNITY_MESSAGEPACK
    source = ROOT / "tests/csharp/PrimitiveExecutionV4RuntimeIntegrationContract.cs"
    protocol_sources = require_protocol_core()
    protocol_constants = require_protocol_constants()
    execution_controller = require_execution_controller()
    lifecycle = require_result_lifecycle()
    adapter = require_result_transport_adapter()
    integration = require_runtime_integration()
    binary = tmp_path / "primitive_execution_v4_runtime_integration_contract.exe"
    compile_result = subprocess.run(
        [
            str(compiler),
            "-langversion:7.2",
            "-r:{}".format(messagepack),
            "-out:{}".format(binary),
                *(str(path) for path in protocol_sources),
                str(protocol_constants),
                * (str(path) for path in execution_controller),
                str(lifecycle),
                str(adapter),
                str(integration),
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
    assert "C# runtime integration tests: 23 passed" in run_result.stdout
