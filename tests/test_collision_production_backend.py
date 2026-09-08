"""Contracts for the C2.2 collision production cutover."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

import planning.safety.collision_checker as collision


ROOT = Path(__file__).resolve().parents[1]


def _small_checker() -> collision.VoxelCollisionChecker:
    origin = np.asarray([-2, -2, -2], dtype=np.int64)
    shape = np.asarray([8, 8, 8], dtype=np.int64)
    occupied_ijk = np.asarray([[0, 0, 0]], dtype=np.int64)
    keys = collision.VoxelCollisionChecker._pack_keys_static(
        occupied_ijk, origin, shape
    )
    return collision.VoxelCollisionChecker(
        np.unique(keys), origin, shape, 0.10, 0.35
    )


@pytest.mark.unit
@pytest.mark.parametrize("backend", ("auto", "cpu", "cpp", "cpp_cpu"))
def test_formal_backend_aliases_resolve_to_one_cpp_cpu_backend(backend):
    assert collision.resolve_collision_backend(backend) == collision.FORMAL_COLLISION_BACKEND


@pytest.mark.unit
def test_unsupported_collision_backends_fail_fast():
    for requested in ("cuda", "hybrid", "unknown"):
        with pytest.raises(ValueError, match="UNSUPPORTED_COLLISION_BACKEND"):
            collision.resolve_collision_backend(requested)


@pytest.mark.unit
def test_production_cli_rejects_unsupported_collision_backends():
    for requested in ("cuda", "hybrid", "python"):
        with pytest.raises(argparse.ArgumentTypeError, match="UNSUPPORTED_COLLISION_BACKEND"):
            collision.collision_backend_cli_type(requested)


@pytest.mark.unit
def test_python_reference_requires_explicit_debug_opt_in(monkeypatch):
    monkeypatch.delenv("PLANNING_COLLISION_REFERENCE", raising=False)
    with pytest.raises(ValueError, match="PYTHON_REFERENCE_BACKEND_REQUIRES_EXPLICIT_DEBUG"):
        collision.resolve_collision_backend("python")

    monkeypatch.setenv("PLANNING_COLLISION_REFERENCE", "1")
    assert (
        collision.resolve_collision_backend("python")
        == collision.PYTHON_REFERENCE_COLLISION_BACKEND
    )


@pytest.mark.unit
def test_explicit_python_reference_is_not_a_production_fallback(monkeypatch):
    monkeypatch.setenv("PLANNING_COLLISION_BACKEND", "python")
    monkeypatch.setenv("PLANNING_COLLISION_REFERENCE", "1")
    checker = _small_checker()

    # Keep the historical public diagnostic stable; the canonical internal
    # selection is asserted separately by the contract ID below.
    assert checker.collision_backend == "python"
    assert checker.collision_backend_contract_id == collision.COLLISION_REFERENCE_CONTRACT_ID
    result = checker.check_path(np.asarray([[0.05, 0.05, 0.05]], dtype=np.float32))
    assert result["collision"] is True


@pytest.mark.unit
def test_missing_native_library_fails_closed(monkeypatch):
    monkeypatch.setenv("PLANNING_COLLISION_BACKEND", "cpp_cpu")
    monkeypatch.setattr(
        collision,
        "_find_collision_library",
        lambda: Path("/tmp/xmflight-c2-2-missing-collision.so"),
    )

    with pytest.raises(RuntimeError, match=r"C\+\+ collision backend required"):
        _small_checker()


@pytest.mark.unit
def test_native_runtime_failure_does_not_switch_to_python(monkeypatch):
    class BrokenNativeBackend:
        library_path = "forced-runtime-failure"
        thread_count = 1
        collision_radius = 0.45

        def check_path(self, path, check_step=1):
            raise RuntimeError("forced native runtime failure")

    monkeypatch.setenv("PLANNING_COLLISION_BACKEND", "cpp_cpu")
    monkeypatch.setattr(
        collision, "_FastCollisionBackend", lambda *args, **kwargs: BrokenNativeBackend()
    )
    checker = _small_checker()

    with pytest.raises(RuntimeError, match="Python reference fallback is disabled"):
        checker.check_path(np.zeros((1, 3), dtype=np.float32))


@pytest.mark.unit
def test_production_source_manifest_has_one_cpp_openmp_collision_owner():
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    source = (ROOT / "python/planning/safety/collision_checker.py").read_text(
        encoding="utf-8"
    )

    assert collision.COLLISION_SOURCE_ID == "planning_collision_checker_cpp17_openmp"
    assert "find_package(OpenMP REQUIRED)" in cmake
    assert cmake.count("add_library(planning_collision_checker ") == 1
    assert "planning_collision_checker_cuda" not in cmake
    assert "find_package(CUDA" not in cmake
    assert "cuda_add_library" not in cmake
    assert "_CudaCollisionBackend" not in source
    assert "PLANNING_COLLISION_CUDA_LIBRARY" not in source
    assert "fallback is disabled" in source
