"""Public-domain API contracts for the C3.1 Global A* production cutover."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import planning.mission.global_route as route
from planning.contracts.collection import build_resolved_collection_config


ROOT = Path(__file__).resolve().parents[1]


def _checker(occupied_keys=(), shape=(8, 8, 1)):
    return SimpleNamespace(
        occupied_keys=np.asarray(occupied_keys, dtype=np.int64),
        origin_ijk=np.asarray([0, 0, 0], dtype=np.int64),
        grid_shape=np.asarray(shape, dtype=np.int64),
        voxel_size=1.0,
        collision_radius=0.0,
    )


def _wall_checker():
    shape = (6, 6, 4)
    keys = [
        3 * shape[1] * shape[2] + y * shape[2] + 1
        for y in range(shape[1])
    ]
    checker = _checker(keys, shape=shape)
    checker.collision_radius = 0.35
    return checker


def _missing_library(monkeypatch):
    missing = Path("/tmp/xmflight-c3-1-missing-global-route.so")
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_LIBRARY", str(missing))
    return missing


@pytest.mark.unit
def test_native_library_exists_and_production_planner_is_cpp_native(monkeypatch):
    monkeypatch.delenv("PLANNING_GLOBAL_ROUTE_BACKEND", raising=False)
    monkeypatch.delenv("PLANNING_GLOBAL_ROUTE_REFERENCE", raising=False)
    library = route._find_global_route_library()
    assert library is not None
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_LIBRARY", str(library))

    planner = route.GlobalRoutePlanner2D(_checker())

    assert planner.backend_name == route.FORMAL_GLOBAL_ROUTE_BACKEND


@pytest.mark.unit
def test_missing_native_library_fails_closed(monkeypatch):
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_BACKEND", "cpp")
    _missing_library(monkeypatch)

    with pytest.raises(RuntimeError, match="GLOBAL_ROUTE_NATIVE_LIBRARY_MISSING"):
        route.GlobalRoutePlanner2D(_checker())


@pytest.mark.unit
def test_missing_native_symbol_fails_closed(monkeypatch):
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_BACKEND", "native")
    monkeypatch.setattr(
        route,
        "_find_global_route_library",
        lambda: Path("/tmp/xmflight-c3-1-symbol-fixture.so"),
    )
    monkeypatch.setattr(route.ctypes, "CDLL", lambda _path: object())

    with pytest.raises(RuntimeError, match="GLOBAL_ROUTE_NATIVE_SYMBOL_MISSING"):
        route.GlobalRoutePlanner2D(_checker())


@pytest.mark.unit
def test_unknown_backend_is_rejected_with_contract_error(monkeypatch):
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_BACKEND", "surprise")

    with pytest.raises(ValueError, match="UNSUPPORTED_GLOBAL_ROUTE_BACKEND"):
        route.GlobalRoutePlanner2D(_checker())


@pytest.mark.unit
def test_production_source_manifest_and_package_identity_are_declared():
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    package = (ROOT / "package.xml").read_text(encoding="utf-8")

    assert route.GLOBAL_ROUTE_SOURCE_ID == "planning_global_route_cpp17"
    assert "src/navigation/global_route_planner.cpp" in route.GLOBAL_ROUTE_SOURCE_FILES
    assert "set(PLANNING_GLOBAL_ROUTE_SOURCES" in cmake
    assert "planning_global_route" in cmake
    assert 'global_route_backend="cpp_native"' in package
    assert 'global_route_source_id="planning_global_route_cpp17"' in package


@pytest.mark.unit
def test_global_route_identity_persists_canonical_config_and_source_manifest():
    identity = route.global_route_identity(
        route.GlobalRouteConfig(resolution_m=0.5, tracking_margin_m=0.25)
    )

    assert identity["global_route_backend"] == "cpp_native"
    assert identity["global_route_contract_id"] == route.GLOBAL_ROUTE_CONTRACT_ID
    assert identity["global_route_source_id"] == route.GLOBAL_ROUTE_SOURCE_ID
    assert identity["global_route_source_files"] == list(route.GLOBAL_ROUTE_SOURCE_FILES)
    assert identity["global_route_config"]["resolution_m"] == 0.5
    assert identity["global_route_config"]["tracking_margin_m"] == 0.25


@pytest.mark.unit
def test_production_callers_do_not_import_python_reference_backend():
    production_root = ROOT / "python" / "planning"
    references = []
    for path in production_root.rglob("*.py"):
        if path.name == "global_route.py" or "__pycache__" in path.parts:
            continue
        if "_PythonReferenceGlobalRoutePlanner2D" in path.read_text(encoding="utf-8"):
            references.append(path.relative_to(ROOT).as_posix())

    assert references == []


@pytest.mark.unit
def test_collection_preparation_manifest_records_route_identity():
    args = SimpleNamespace(
        num_workers=2,
        voxel_size=0.10,
        inflate_radius=0.35,
        out_dir="data/teach/flight/rollouts",
    )
    manifest = build_resolved_collection_config(
        args,
        mission_index_sha256="a" * 64,
        collision_cache_sha256="b" * 64,
        code_version_sha256="c" * 64,
        teacher_config={
            "z_min": 1.0,
            "z_max": 3.0,
            "global_route_resolution_m": 0.25,
            "global_route_lookahead_m": 3.0,
            "global_route_tracking_margin_m": 0.0,
        },
        mpl_contract_sha256="d" * 64,
        async_prefetch_enabled=False,
        package_root=ROOT,
    )

    shared = manifest["shared_collection_config"]
    assert shared["global_route_backend"] == "cpp_native"
    assert shared["global_route_source_id"] == route.GLOBAL_ROUTE_SOURCE_ID
    assert shared["global_route_map_identity"]["cache_sha256"] == "b" * 64


@pytest.mark.unit
def test_python_backend_requires_explicit_reference_debug_seam(monkeypatch):
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_BACKEND", "python")
    monkeypatch.delenv("PLANNING_GLOBAL_ROUTE_REFERENCE", raising=False)

    with pytest.raises(ValueError, match="PYTHON_REFERENCE_GLOBAL_ROUTE_REQUIRES_EXPLICIT_DEBUG"):
        route.GlobalRoutePlanner2D(_checker())


@pytest.mark.unit
def test_auto_backend_missing_native_library_does_not_fallback(monkeypatch):
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_BACKEND", "auto")
    _missing_library(monkeypatch)

    with pytest.raises(RuntimeError, match="GLOBAL_ROUTE_NATIVE_LIBRARY_MISSING"):
        route.GlobalRoutePlanner2D(_checker())


@pytest.mark.unit
def test_no_path_is_a_normal_business_result(monkeypatch):
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_BACKEND", "cpp")
    monkeypatch.delenv("PLANNING_GLOBAL_ROUTE_REFERENCE", raising=False)
    library = route._find_global_route_library()
    assert library is not None
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_LIBRARY", str(library))

    planner = route.GlobalRoutePlanner2D(_wall_checker())

    with pytest.raises(route.GlobalRouteUnavailableError):
        planner.plan((0.1, 0.1, 1.5), (5.1, 5.1, 1.5))


@pytest.mark.unit
def test_native_execution_failure_is_explicit_and_not_reference_fallback(monkeypatch):
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_BACKEND", "cpp")
    library = route._find_global_route_library()
    assert library is not None
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_LIBRARY", str(library))
    planner = route.GlobalRoutePlanner2D(_checker())

    def fail_native(_start, _goal):
        raise RuntimeError("forced native route execution failure")

    monkeypatch.setattr(planner._backend, "plan", fail_native)
    with pytest.raises(RuntimeError, match="forced native route execution failure"):
        planner.plan((0.1, 0.1, 1.5), (5.1, 5.1, 1.5))
