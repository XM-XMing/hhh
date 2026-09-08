"""Public-seam tests for the C4 depth-safety production contract."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from planning.safety.depth_safety import (
    DepthSafetyConfig,
    local_depth_action_mask,
)
from planning.native.loader import resolve_depth_safety_library


ROOT = Path(__file__).resolve().parents[1]
DEPTH_LIBRARY = Path(resolve_depth_safety_library())


class _SyntheticMPL:
    """Small public MPL substitute with the formal 105-action width."""

    num_actions = 105

    @staticmethod
    def reference_path(action_id: int) -> np.ndarray:
        lateral = (int(action_id) % 15 - 7) * 0.025
        vertical = (int(action_id) // 15 - 3) * 0.025
        path = np.zeros((17, 3), dtype=np.float32)
        path[:, 0] = np.linspace(0.0, 2.0, path.shape[0], dtype=np.float32)
        path[:, 1] = np.linspace(0.0, lateral, path.shape[0], dtype=np.float32)
        path[:, 2] = np.linspace(0.0, vertical, path.shape[0], dtype=np.float32)
        return path


def _subprocess_code() -> str:
    return """
import numpy as np
from planning.safety.depth_safety import DepthSafetyConfig, local_depth_action_mask

class M:
    num_actions = 105
    @staticmethod
    def reference_path(action_id):
        lateral = (int(action_id) % 15 - 7) * 0.025
        vertical = (int(action_id) // 15 - 3) * 0.025
        path = np.zeros((17, 3), dtype=np.float32)
        path[:, 0] = np.linspace(0.0, 2.0, 17, dtype=np.float32)
        path[:, 1] = np.linspace(0.0, lateral, 17, dtype=np.float32)
        path[:, 2] = np.linspace(0.0, vertical, 17, dtype=np.float32)
        return path

local_depth_action_mask(M(), np.full((30, 40), 6.0, dtype=np.float32), DepthSafetyConfig())
print('UNEXPECTED_SUCCESS')
"""


def _run_isolated(extra_env: dict[str, str], code: str | None = None):
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "python"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    environment.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code or _subprocess_code()],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.unit
def test_native_depth_safety_is_the_formal_public_backend(monkeypatch):
    monkeypatch.setenv("PLANNING_DEPTH_SAFETY_BACKEND", "cpp")
    monkeypatch.setenv("PLANNING_DEPTH_SAFETY_LIB", str(DEPTH_LIBRARY))
    mask, info = local_depth_action_mask(
        _SyntheticMPL(), np.full((30, 40), 6.0, dtype=np.float32), DepthSafetyConfig()
    )

    assert mask.shape == (105,)
    assert info["depth_safety_backend_contract_id"] == "cpp_depth_patch"


@pytest.mark.unit
def test_missing_native_library_fails_closed_instead_of_using_python():
    result = _run_isolated(
        {
            "PLANNING_DEPTH_SAFETY_BACKEND": "auto",
            "PLANNING_DEPTH_SAFETY_LIB": "/tmp/xmflight-c4-missing-depth-safety.so",
        }
    )

    assert result.returncode != 0
    assert "DEPTH_SAFETY_NATIVE_UNAVAILABLE" in result.stderr


@pytest.mark.unit
def test_missing_native_symbol_fails_closed():
    code = """
import ctypes
class MissingSymbolLibrary:
    pass
ctypes.CDLL = lambda _path: MissingSymbolLibrary()
""" + _subprocess_code()
    result = _run_isolated(
        {
            "PLANNING_DEPTH_SAFETY_BACKEND": "cpp",
            "PLANNING_DEPTH_SAFETY_LIB": str(DEPTH_LIBRARY),
        },
        code,
    )

    assert result.returncode != 0
    assert "DEPTH_SAFETY_NATIVE_SYMBOL_MISSING" in result.stderr


@pytest.mark.unit
def test_python_backend_requires_explicit_reference_seam(monkeypatch):
    monkeypatch.setenv("PLANNING_DEPTH_SAFETY_BACKEND", "python")
    monkeypatch.delenv("PLANNING_DEPTH_SAFETY_REFERENCE", raising=False)

    with pytest.raises(ValueError, match="PYTHON_REFERENCE_DEPTH_SAFETY_REQUIRES_EXPLICIT_DEBUG"):
        local_depth_action_mask(
            _SyntheticMPL(),
            np.full((30, 40), 6.0, dtype=np.float32),
            DepthSafetyConfig(),
        )


@pytest.mark.unit
def test_explicit_python_reference_is_not_formal_backend(monkeypatch):
    monkeypatch.setenv("PLANNING_DEPTH_SAFETY_BACKEND", "python")
    monkeypatch.setenv("PLANNING_DEPTH_SAFETY_REFERENCE", "1")
    mask, info = local_depth_action_mask(
        _SyntheticMPL(), np.full((30, 40), 6.0, dtype=np.float32), DepthSafetyConfig()
    )

    assert mask.shape == (105,)
    assert info["depth_safety_backend_contract_id"] == "numpy_depth_patch"


@pytest.mark.unit
def test_invalid_depth_shape_is_rejected_at_public_seam(monkeypatch):
    monkeypatch.setenv("PLANNING_DEPTH_SAFETY_BACKEND", "python")
    monkeypatch.setenv("PLANNING_DEPTH_SAFETY_REFERENCE", "1")

    with pytest.raises(ValueError, match="depth_m must have shape"):
        local_depth_action_mask(_SyntheticMPL(), np.zeros((30,), dtype=np.float32))


@pytest.mark.unit
def test_production_source_manifest_has_one_native_depth_owner():
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    package = (ROOT / "package.xml").read_text(encoding="utf-8")

    assert cmake.count("add_library(planning_depth_safety ") == 1
    assert "src/geometry/depth_safety.cpp" in cmake
    assert "planning_depth_safety" in cmake
    assert 'depth_safety_backend="cpp_native"' in package
    assert 'depth_safety_contract_id="cpp_depth_patch"' in package
    assert 'depth_safety_library="libplanning_depth_safety.so"' in package
    assert 'depth_safety_source_id="planning_depth_safety_cpp17"' in package


@pytest.mark.unit
def test_production_callers_do_not_own_native_depth_details():
    production_root = ROOT / "python" / "planning"
    native_detail_paths = []
    for path in production_root.rglob("*.py"):
        if path.name == "depth_safety.py" or "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if "planning_depth_safety_mask" in source or "_NativeDepthSafety" in source:
            native_detail_paths.append(path.relative_to(ROOT).as_posix())

    assert native_detail_paths == []
