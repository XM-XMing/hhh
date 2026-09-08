"""Tests for the single deterministic native-library path owner."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from planning.native import loader


LIBRARY_KINDS = (
    "voxel_map",
    "collision",
    "global_route",
    "depth_safety",
)


def _fake_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    for name in (
        "libplanning_voxel_map.so",
        "libplanning_collision_checker.so",
        "libplanning_global_route.so",
        "libplanning_depth_safety.so",
    ):
        path = workspace / "devel" / "lib" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return workspace


def _clear_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PLANNING_VOXEL_MAP_LIBRARY",
        "PLANNING_COLLISION_LIBRARY",
        "PLANNING_GLOBAL_ROUTE_LIBRARY",
        "PLANNING_DEPTH_SAFETY_LIB",
        "PLANNING_DEPTH_SAFETY_TEST_LIBRARY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.unit
@pytest.mark.parametrize("kind", LIBRARY_KINDS)
def test_unset_override_resolves_only_project_derived_devel_library(
    tmp_path, monkeypatch, kind
):
    workspace = _fake_workspace(tmp_path)
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(loader, "planning_workspace_root", lambda: workspace)

    resolved = loader.resolve_native_library_path(kind)

    expected = workspace / "devel" / "lib" / loader.native_library_spec(kind).filename
    assert resolved == expected


@pytest.mark.unit
def test_explicit_override_wins_over_devel(monkeypatch, tmp_path):
    workspace = _fake_workspace(tmp_path)
    override = tmp_path / "special" / "libplanning_collision_checker.so"
    override.parent.mkdir()
    override.touch()
    monkeypatch.setattr(loader, "planning_workspace_root", lambda: workspace)
    _clear_overrides(monkeypatch)
    monkeypatch.setenv("PLANNING_COLLISION_LIBRARY", str(override))

    assert loader.resolve_collision_library() == override.resolve()


@pytest.mark.unit
def test_missing_explicit_override_fails_without_devel_fallback(monkeypatch, tmp_path):
    workspace = _fake_workspace(tmp_path)
    missing = tmp_path / "missing" / "libplanning_voxel_map.so"
    monkeypatch.setattr(loader, "planning_workspace_root", lambda: workspace)
    _clear_overrides(monkeypatch)
    monkeypatch.setenv("PLANNING_VOXEL_MAP_LIBRARY", str(missing))

    with pytest.raises(loader.NativeLibraryError) as error:
        loader.resolve_voxel_map_library()

    message = str(error.value)
    assert "VOXEL_MAP_NATIVE_LIBRARY_MISSING" in message
    assert str(missing) in message
    assert "build workspace with catkin_make" in message


@pytest.mark.unit
def test_missing_devel_does_not_fallback_to_install(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    install = workspace / "install" / "lib" / "libplanning_collision_checker.so"
    install.parent.mkdir(parents=True)
    install.touch()
    monkeypatch.setattr(loader, "planning_workspace_root", lambda: workspace)
    _clear_overrides(monkeypatch)

    with pytest.raises(loader.NativeLibraryError) as error:
        loader.resolve_collision_library()

    message = str(error.value)
    assert "COLLISION_NATIVE_LIBRARY_MISSING" in message
    assert str(workspace / "devel" / "lib") in message
    assert "install" not in message


@pytest.mark.unit
def test_project_derived_workspace_wins_over_other_workspace_candidate(
    monkeypatch, tmp_path
):
    workspace = _fake_workspace(tmp_path)
    other_workspace = tmp_path / "other-workspace"
    other = other_workspace / "devel" / "lib" / "libplanning_collision_checker.so"
    other.parent.mkdir(parents=True)
    other.touch()
    monkeypatch.setattr(loader, "planning_workspace_root", lambda: workspace)
    _clear_overrides(monkeypatch)

    assert loader.resolve_collision_library() == (
        workspace / "devel" / "lib" / "libplanning_collision_checker.so"
    ).resolve()


@pytest.mark.unit
def test_depth_test_override_is_optional_and_canonical(monkeypatch, tmp_path):
    workspace = _fake_workspace(tmp_path)
    test_override = tmp_path / "test-depth.so"
    test_override.touch()
    monkeypatch.setattr(loader, "planning_workspace_root", lambda: workspace)
    _clear_overrides(monkeypatch)
    monkeypatch.setenv("PLANNING_DEPTH_SAFETY_TEST_LIBRARY", str(test_override))

    assert loader.resolve_depth_safety_library() == test_override.resolve()


@pytest.mark.unit
def test_unknown_library_kind_fails_at_public_resolver_seam():
    with pytest.raises(ValueError, match="unknown native library kind"):
        loader.resolve_native_library_path("not-a-library")


@pytest.mark.unit
def test_all_formal_libraries_load_without_path_overrides(monkeypatch):
    _clear_overrides(monkeypatch)

    loaded = {}
    for kind in LIBRARY_KINDS:
        path, library = loader.load_native_library(kind)
        loaded[kind] = (Path(path), library)

    assert set(loaded) == set(LIBRARY_KINDS)
    assert all(path.parent.name == "lib" for path, _library in loaded.values())
    assert all(path.parent.parent.name == "devel" for path, _library in loaded.values())


@pytest.mark.unit
def test_native_path_resolution_has_one_production_owner():
    root = Path(__file__).resolve().parents[1] / "python" / "planning"
    path_environment_names = (
        "PLANNING_VOXEL_MAP_LIBRARY",
        "PLANNING_COLLISION_LIBRARY",
        "PLANNING_GLOBAL_ROUTE_LIBRARY",
        "PLANNING_DEPTH_SAFETY_LIB",
        "PLANNING_DEPTH_SAFETY_TEST_LIBRARY",
    )
    forbidden = re.compile(
        r"(?:{}|devel/lib|install/lib|ctypes\.util\.find_library|"
        r"native_library_candidates)".format("|".join(path_environment_names))
    )
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "loader.py" or "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if forbidden.search(source):
            offenders.append(path.relative_to(root.parent.parent).as_posix())

    assert offenders == []
