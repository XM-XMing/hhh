"""Single owner for native shared-library discovery and symbol lookup."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from planning.common.paths import planning_workspace_root


class NativeLibraryError(RuntimeError):
    """Raised when a required native library cannot be resolved or loaded."""


@dataclass(frozen=True)
class NativeLibrarySpec:
    """Canonical identity and override names for one Planning library."""

    kind: str
    filename: str
    environment_name: str
    component: str
    additional_override_names: Tuple[str, ...] = ()

    @property
    def override_environment_names(self) -> Tuple[str, ...]:
        return self.additional_override_names + (self.environment_name,)


NATIVE_LIBRARY_SPECS: Dict[str, NativeLibrarySpec] = {
    "voxel_map": NativeLibrarySpec(
        "voxel_map",
        "libplanning_voxel_map.so",
        "PLANNING_VOXEL_MAP_LIBRARY",
        "VOXEL_MAP",
    ),
    "collision": NativeLibrarySpec(
        "collision",
        "libplanning_collision_checker.so",
        "PLANNING_COLLISION_LIBRARY",
        "COLLISION",
    ),
    "global_route": NativeLibrarySpec(
        "global_route",
        "libplanning_global_route.so",
        "PLANNING_GLOBAL_ROUTE_LIBRARY",
        "GLOBAL_ROUTE",
    ),
    "depth_safety": NativeLibrarySpec(
        "depth_safety",
        "libplanning_depth_safety.so",
        "PLANNING_DEPTH_SAFETY_LIB",
        "DEPTH_SAFETY",
        additional_override_names=("PLANNING_DEPTH_SAFETY_TEST_LIBRARY",),
    ),
}

NativeLibraryPath = Union[Path, str]


def native_library_spec(kind: str) -> NativeLibrarySpec:
    """Return the canonical specification for a known library kind."""

    try:
        return NATIVE_LIBRARY_SPECS[str(kind).strip().lower()]
    except KeyError as error:
        raise ValueError("unknown native library kind: {}".format(kind)) from error


def _missing_library_error(
    spec: NativeLibrarySpec,
    expected_path: NativeLibraryPath,
    workspace: Path,
    override_name: Optional[str] = None,
) -> NativeLibraryError:
    source = (
        "override={} ".format(override_name) if override_name is not None else ""
    )
    return NativeLibraryError(
        "{component}_NATIVE_LIBRARY_MISSING\n"
        "library_kind={kind}\n"
        "{source}expected_path={expected}\n"
        "workspace={workspace}\n"
        "build workspace with catkin_make".format(
            component=spec.component,
            kind=spec.kind,
            source=source,
            expected=expected_path,
            workspace=workspace,
        )
    )


def _is_dynamic_loader_name(value: str) -> bool:
    """Keep ctypes loader-name overrides usable without discovering paths."""

    raw = str(value).strip()
    return bool(
        raw
        and raw.startswith("lib")
        and "/" not in raw
        and "\\" not in raw
        and not raw.startswith(".")
        and not Path(raw).is_absolute()
    )


def resolve_native_library_path(kind: str) -> NativeLibraryPath:
    """Resolve a native library deterministically for the current workspace.

    Explicit environment values have precedence.  A path-valued override must
    already exist and is never replaced by an automatic candidate.  With no
    override, only the project-derived catkin ``devel/lib`` location is used;
    install trees, package-local libraries, filesystem searches, and
    ``ctypes.util.find_library`` are deliberately outside the formal runtime.
    """

    spec = native_library_spec(kind)
    workspace = planning_workspace_root().resolve()
    for environment_name in spec.override_environment_names:
        raw = os.environ.get(environment_name, "").strip()
        if not raw:
            continue
        if _is_dynamic_loader_name(raw):
            return raw
        explicit_path = Path(raw).expanduser()
        if not explicit_path.is_file():
            raise _missing_library_error(
                spec, explicit_path, workspace, environment_name
            )
        return explicit_path.resolve()

    expected_path = workspace / "devel" / "lib" / spec.filename
    if not expected_path.is_file():
        raise _missing_library_error(spec, expected_path, workspace)
    return expected_path.resolve()


def resolve_voxel_map_library() -> NativeLibraryPath:
    return resolve_native_library_path("voxel_map")


def resolve_collision_library() -> NativeLibraryPath:
    return resolve_native_library_path("collision")


def resolve_global_route_library() -> NativeLibraryPath:
    return resolve_native_library_path("global_route")


def resolve_depth_safety_library() -> NativeLibraryPath:
    return resolve_native_library_path("depth_safety")


def load_native_library(kind: str) -> Tuple[str, ctypes.CDLL]:
    """Resolve and load one canonical Planning native library."""

    spec = native_library_spec(kind)
    path = resolve_native_library_path(spec.kind)
    try:
        return str(path), ctypes.CDLL(str(path))
    except Exception as error:
        raise NativeLibraryError(
            "{}_NATIVE_INITIALIZATION_FAILED: {}".format(
                spec.component, repr(error)
            )
        ) from error


def _legacy_kind(filename: str, environment_name: str) -> str:
    for kind, spec in NATIVE_LIBRARY_SPECS.items():
        if spec.filename == filename and spec.environment_name == environment_name:
            return kind
    raise ValueError(
        "legacy native loader arguments are not a canonical Planning library: "
        "{} {}".format(filename, environment_name)
    )


def find_library(
    filename: str,
    environment_name: str,
    *,
    explicit_only: bool = False,
    allow_missing_explicit: bool = False,
) -> Optional[Path]:
    """Compatibility wrapper delegating to the canonical resolver."""

    del explicit_only, allow_missing_explicit
    kind = _legacy_kind(filename, environment_name)
    try:
        path = resolve_native_library_path(kind)
    except NativeLibraryError:
        return None
    return Path(path) if not isinstance(path, Path) else path


def load_library(
    filename: str,
    environment_name: str,
    component: str,
    *,
    explicit_only: bool = False,
    allow_missing_explicit: bool = False,
) -> Tuple[str, ctypes.CDLL]:
    del component, explicit_only, allow_missing_explicit
    return load_native_library(_legacy_kind(filename, environment_name))


def require_symbol(library, symbol: str, component: str):
    try:
        return getattr(library, symbol)
    except AttributeError as error:
        raise NativeLibraryError(
            "{}_NATIVE_SYMBOL_MISSING: {}".format(component, symbol)
        ) from error


def load_library_path(path: Path, component: str) -> Tuple[str, ctypes.CDLL]:
    """Load an already-resolved path without repeating discovery."""

    try:
        return str(path), ctypes.CDLL(str(path))
    except Exception as error:
        raise NativeLibraryError(
            "{}_NATIVE_INITIALIZATION_FAILED: {}".format(component, repr(error))
        ) from error


__all__ = [
    "NativeLibraryPath",
    "NativeLibrarySpec",
    "NativeLibraryError",
    "NATIVE_LIBRARY_SPECS",
    "find_library",
    "load_native_library",
    "load_library",
    "load_library_path",
    "require_symbol",
    "native_library_spec",
    "resolve_native_library_path",
    "resolve_voxel_map_library",
    "resolve_collision_library",
    "resolve_global_route_library",
    "resolve_depth_safety_library",
]
