"""Contract tests for the process-local Python native geometry owner.

These tests intentionally exercise the public context/factory seam.  They use
only a tiny synthetic cache and never create or modify project data.
"""

from __future__ import annotations

import ctypes.util
import gc
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from planning.mission.global_route import GlobalRouteConfig
from planning.native.geometry import NativeGeometryContext


def _configure_native_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PLANNING_VOXEL_MAP_LIBRARY",
        "PLANNING_COLLISION_LIBRARY",
        "PLANNING_GLOBAL_ROUTE_LIBRARY",
        "PLANNING_DEPTH_SAFETY_LIB",
        "PLANNING_DEPTH_SAFETY_TEST_LIBRARY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PLANNING_COLLISION_BACKEND", "cpp_cpu")
    monkeypatch.setenv("PLANNING_GLOBAL_ROUTE_BACKEND", "cpp_native")


def _write_cache(path: Path, *, extra_key: bool = False) -> Path:
    shape = np.asarray([20, 20, 4], dtype=np.int64)
    keys = [10 * int(shape[1]) * int(shape[2]) + 10 * int(shape[2]) + 1]
    if extra_key:
        keys.append(11 * int(shape[1]) * int(shape[2]) + 10 * int(shape[2]) + 1)
    np.savez(
        str(path),
        occupied_keys=np.asarray(keys, dtype=np.int64),
        origin_ijk=np.asarray([-10, -10, 0], dtype=np.int64),
        grid_shape=shape,
        voxel_size=np.asarray(0.10, dtype=np.float64),
    )
    return path


def _make_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, extra_key: bool = False):
    _configure_native_environment(monkeypatch)
    cache = _write_cache(tmp_path / ("map_b.npz" if extra_key else "map_a.npz"), extra_key=extra_key)
    context = NativeGeometryContext.from_voxel_cache(
        cache,
        voxel_size=0.10,
        route_config=GlobalRouteConfig(),
    )
    return context, cache


def test_collision_and_route_share_one_python_native_voxel_owner(tmp_path, monkeypatch):
    context, _ = _make_context(tmp_path, monkeypatch)
    checker = context.collision_checker(0.35)
    route = context.global_route_planner(checker)

    assert checker.native_map_id == route.native_map_id == context.native_map_id
    assert checker.native_checker_id != route.native_planner_id
    assert context.stats["voxel_map_load_count"] == 1
    assert context.stats["occupied_buffer_allocation_count"] == 1
    context.close()


def test_collision_radii_share_map_but_not_checker(tmp_path, monkeypatch):
    context, _ = _make_context(tmp_path, monkeypatch)
    low = context.collision_checker(0.35)
    high = context.collision_checker(0.40)

    assert low is context.collision_checker(0.35)
    assert low.native_map_id == high.native_map_id == context.native_map_id
    assert low.native_checker_id != high.native_checker_id
    assert np.shares_memory(context.occupied_keys, low.occupied_keys)
    assert np.shares_memory(context.occupied_keys, high.occupied_keys)
    context.close()


def test_context_close_is_fail_closed_and_releases_children(tmp_path, monkeypatch):
    context, _ = _make_context(tmp_path, monkeypatch)
    checker = context.collision_checker(0.35)
    route = context.global_route_planner(checker)
    context.close()

    assert context.closed
    with pytest.raises(RuntimeError, match="closed"):
        checker.check_path(np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        route.plan([0.0, 0.0, 1.0], [0.5, 0.0, 1.0])


def test_consumer_lifetime_retains_map_after_other_consumer_closes(tmp_path, monkeypatch):
    context, _ = _make_context(tmp_path, monkeypatch)
    checker = context.collision_checker(0.35)
    route = context.global_route_planner(checker)
    checker.close()

    result = route.plan([0.0, 0.0, 1.0], [0.5, 0.0, 1.0])
    assert result.shape[1] == 3
    context.close()


def test_closed_checker_cannot_reenter_context_route_cache(tmp_path, monkeypatch):
    context, _ = _make_context(tmp_path, monkeypatch)
    checker = context.collision_checker(0.35)
    checker.close()

    with pytest.raises(RuntimeError, match="closed"):
        context.global_route_planner(checker)
    context.close()


def test_different_artifact_sha_never_shares_native_map(tmp_path, monkeypatch):
    context_a, cache_a = _make_context(tmp_path, monkeypatch)
    context_b, cache_b = _make_context(tmp_path, monkeypatch, extra_key=True)

    assert context_a.map_identity["sha256"] != context_b.map_identity["sha256"]
    assert context_a.map_identity["path"] != context_b.map_identity["path"]
    assert context_a.native_map_id != context_b.native_map_id
    context_a.close()
    context_b.close()
    assert cache_a.exists() and cache_b.exists()


def test_same_context_does_not_reread_or_decode_cache(tmp_path, monkeypatch):
    context, _ = _make_context(tmp_path, monkeypatch)
    checker = context.collision_checker(0.35)

    import planning.native.geometry as geometry

    original_load = geometry.np.load
    calls = []

    def unexpected_load(*args, **kwargs):
        calls.append((args, kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(geometry.np, "load", unexpected_load)
    context.collision_checker(0.40)
    context.global_route_planner(checker)
    assert calls == []
    assert context.stats["voxel_map_load_count"] == 1
    context.close()


def test_native_binding_rejects_wrong_numpy_dtype_and_shape(tmp_path, monkeypatch):
    context, _ = _make_context(tmp_path, monkeypatch)
    checker = context.collision_checker(0.35)

    with pytest.raises((TypeError, ValueError), match="dtype"):
        checker.check_path(np.zeros((1, 3), dtype=np.float64))
    with pytest.raises((TypeError, ValueError), match="shape"):
        checker.check_path(np.zeros((3,), dtype=np.float32))
    route = context.global_route_planner(checker)
    with pytest.raises((TypeError, ValueError), match="dtype"):
        route.plan(np.zeros((3,), dtype=np.float64), np.zeros((3,), dtype=np.float64))
    context.close()


def test_missing_native_symbol_fails_closed(tmp_path, monkeypatch):
    cache = _write_cache(tmp_path / "missing_symbol.npz")
    libc = ctypes.util.find_library("c")
    if not libc:
        pytest.skip("libc is not discoverable")
    monkeypatch.setenv("PLANNING_VOXEL_MAP_LIBRARY", libc)
    monkeypatch.delenv("PLANNING_COLLISION_LIBRARY", raising=False)
    monkeypatch.delenv("PLANNING_GLOBAL_ROUTE_LIBRARY", raising=False)

    with pytest.raises(RuntimeError, match="SYMBOL_MISSING"):
        NativeGeometryContext.from_voxel_cache(cache, voxel_size=0.10)


def test_missing_explicit_native_library_fails_closed(tmp_path, monkeypatch):
    cache = _write_cache(tmp_path / "missing_library.npz")
    monkeypatch.setenv(
        "PLANNING_VOXEL_MAP_LIBRARY", str(tmp_path / "does_not_exist.so")
    )

    with pytest.raises(RuntimeError, match="LIBRARY_MISSING"):
        NativeGeometryContext.from_voxel_cache(cache, voxel_size=0.10)


def _child_context_probe(cache_text: str, result_queue) -> None:
    for name in (
        "PLANNING_VOXEL_MAP_LIBRARY",
        "PLANNING_COLLISION_LIBRARY",
        "PLANNING_GLOBAL_ROUTE_LIBRARY",
        "PLANNING_DEPTH_SAFETY_LIB",
        "PLANNING_DEPTH_SAFETY_TEST_LIBRARY",
    ):
        os.environ.pop(name, None)
    from planning.native.geometry import NativeGeometryContext

    context = NativeGeometryContext.from_voxel_cache(Path(cache_text), voxel_size=0.10)
    checker = context.collision_checker(0.35)
    result_queue.put(
        {
            "pid": os.getpid(),
            "map_id": context.native_map_id,
            "checker_map_id": checker.native_map_id,
            "map_load_count": context.stats["voxel_map_load_count"],
        }
    )
    context.close()


def test_child_process_creates_own_context_and_loads_once(tmp_path, monkeypatch):
    context, cache = _make_context(tmp_path, monkeypatch)
    parent_map_id = context.native_map_id
    spawn = mp.get_context("spawn")
    children = []
    result_queue = spawn.Queue()
    for _ in range(2):
        child = spawn.Process(target=_child_context_probe, args=(str(cache), result_queue))
        child.start()
        children.append(child)
    for child in children:
        child.join(30)
        assert child.exitcode == 0
    results = [result_queue.get(timeout=5) for _ in children]
    assert all(result["map_load_count"] == 1 for result in results)
    assert all(result["map_id"] == result["checker_map_id"] for result in results)
    assert all(result["map_id"] != parent_map_id for result in results)
    assert len({result["map_id"] for result in results}) == 2
    context.close()
    gc.collect()


_INHERITED_CONTEXT = None


def _forked_parent_context_probe(result_queue) -> None:
    try:
        _INHERITED_CONTEXT.collision_checker(0.35)
    except Exception as error:
        result_queue.put(str(error))
        return
    result_queue.put("UNEXPECTED_SUCCESS")


def test_forked_parent_handle_reuse_fails_closed(tmp_path, monkeypatch):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork is not available")
    context, _ = _make_context(tmp_path, monkeypatch)
    global _INHERITED_CONTEXT
    _INHERITED_CONTEXT = context
    fork = mp.get_context("fork")
    result_queue = fork.Queue()
    child = fork.Process(target=_forked_parent_context_probe, args=(result_queue,))
    child.start()
    child.join(30)
    assert child.exitcode == 0
    assert "FORK_REUSE_FORBIDDEN" in result_queue.get(timeout=5)
    context.close()
    _INHERITED_CONTEXT = None
