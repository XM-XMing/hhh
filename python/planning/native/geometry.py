"""Shared process-local owner for native voxel geometry.

This module is the narrow Python binding seam for the C++17 geometry targets.
It owns one immutable ``VoxelMap`` per artifact identity and creates
collision/route consumers that retain that map through the existing C ABI.
Business code receives the established CollisionChecker and GlobalRoutePlanner
APIs; it never needs to know a library path, ctypes signature, or raw handle.
"""

from __future__ import annotations

import ctypes
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from planning.common import file_sha256
from planning.native.loader import NativeLibraryError, load_native_library, require_symbol


class NativeGeometryError(RuntimeError):
    """A fail-closed native geometry binding or lifecycle error."""


_I64_PTR = ctypes.POINTER(ctypes.c_int64)


def _prepare_native_array(
    value: Any,
    dtype: np.dtype,
    shape: Optional[Tuple[int, ...]],
    name: str,
    context: Optional["NativeGeometryContext"] = None,
) -> np.ndarray:
    """Validate an ABI array and make only an explicit contiguity copy.

    Python sequences are accepted for compatibility and converted to the ABI
    dtype.  A caller-provided NumPy array with a wrong dtype is rejected rather
    than silently force-cast; a non-contiguous array of the right dtype may be
    copied once for the C ABI.
    """
    expected_dtype = np.dtype(dtype)
    raw = np.asarray(value)
    if shape is not None:
        if len(shape) != raw.ndim or any(
            expected != -1 and actual != expected
            for actual, expected in zip(raw.shape, shape)
        ):
            raise ValueError("{} shape {} != {}".format(name, raw.shape, shape))
    is_numpy_input = isinstance(value, np.ndarray)
    dtype_converted = raw.dtype != expected_dtype
    if is_numpy_input and dtype_converted:
        raise TypeError(
            "{} dtype {} != {}".format(name, raw.dtype, expected_dtype)
        )
    prepared = raw if not dtype_converted else np.asarray(value, dtype=expected_dtype)
    copied = False
    if not prepared.flags.c_contiguous:
        prepared = np.ascontiguousarray(prepared, dtype=expected_dtype)
        copied = True
    if context is not None:
        context._record_numpy_input(
            copied=copied,
            dtype_converted=dtype_converted and not is_numpy_input,
        )
    return prepared


def _route_config_identity(config: Any) -> Dict[str, object]:
    """Return the stable route configuration fields used by the owner key."""
    if config is None:
        from planning.mission.global_route import GlobalRouteConfig

        config = GlobalRouteConfig()
    names = (
        "resolution_m",
        "flight_z_min_m",
        "flight_z_max_m",
        "lookahead_m",
        "tracking_margin_m",
        "nearest_free_radius_cells",
    )
    missing = [name for name in names if not hasattr(config, name)]
    if missing:
        raise ValueError("route config missing fields: {}".format(",".join(missing)))
    return {
        "resolution_m": float(config.resolution_m),
        "flight_z_min_m": float(config.flight_z_min_m),
        "flight_z_max_m": float(config.flight_z_max_m),
        "lookahead_m": float(config.lookahead_m),
        "tracking_margin_m": float(config.tracking_margin_m),
        "nearest_free_radius_cells": int(config.nearest_free_radius_cells),
    }


def _strict_cache_array(data: Any, name: str, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
    value = np.asarray(data)
    if value.dtype != np.dtype(dtype):
        raise TypeError(
            "voxel cache {} dtype {} != {}".format(name, value.dtype, np.dtype(dtype))
        )
    if value.shape != shape:
        raise ValueError(
            "voxel cache {} shape {} != {}".format(name, value.shape, shape)
        )
    result = np.array(value, dtype=value.dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


class _NativeVoxelMapOwner:
    """RAII-like ctypes owner for one C++ VoxelMap handle."""

    def __init__(
        self,
        occupied_keys: np.ndarray,
        origin_ijk: np.ndarray,
        grid_shape: np.ndarray,
        voxel_size: float,
    ):
        self._pid = os.getpid()
        try:
            self.library_path, self.lib = load_native_library("voxel_map")
            create = require_symbol(self.lib, "planning_voxel_map_create", "VOXEL_MAP")
            destroy = require_symbol(self.lib, "planning_voxel_map_destroy", "VOXEL_MAP")
        except NativeLibraryError as error:
            raise NativeGeometryError(str(error)) from error
        create.argtypes = [_I64_PTR, ctypes.c_int64, _I64_PTR, _I64_PTR, ctypes.c_double]
        create.restype = ctypes.c_void_p
        destroy.argtypes = [ctypes.c_void_p]
        destroy.restype = None
        self._destroy = destroy
        self.handle = None
        try:
            raw_handle = create(
                occupied_keys.ctypes.data_as(_I64_PTR),
                ctypes.c_int64(int(occupied_keys.size)),
                origin_ijk.ctypes.data_as(_I64_PTR),
                grid_shape.ctypes.data_as(_I64_PTR),
                ctypes.c_double(float(voxel_size)),
            )
        except Exception as error:
            raise NativeGeometryError(
                "VOXEL_MAP_NATIVE_INITIALIZATION_FAILED: {}".format(repr(error))
            ) from error
        if not raw_handle:
            raise NativeGeometryError(
                "VOXEL_MAP_NATIVE_INITIALIZATION_FAILED: planning_voxel_map_create returned null"
            )
        self.handle = ctypes.c_void_p(raw_handle)
        self.native_map_id = int(raw_handle)

    def assert_open(self, owner: str = "native geometry") -> None:
        if os.getpid() != self._pid:
            raise NativeGeometryError(
                "NATIVE_CONTEXT_FORK_REUSE_FORBIDDEN: {} was created in pid {} and used in pid {}".format(
                    owner, self._pid, os.getpid()
                )
            )
        if not self.handle:
            raise NativeGeometryError("{} is closed".format(owner))

    def close(self) -> None:
        if self.handle is None:
            return
        if os.getpid() == self._pid:
            self._destroy(self.handle)
        self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class NativeGeometryContext:
    """One process-local immutable map owner with cached native consumers.

    A context is intentionally not pickleable and never transfers a native
    handle across a process boundary.  Child processes must construct their
    own context from the artifact path.
    """

    def __init__(
        self,
        cache_path: Path,
        artifact_sha256: str,
        occupied_keys: np.ndarray,
        origin_ijk: np.ndarray,
        grid_shape: np.ndarray,
        voxel_size: float,
        route_config: Any,
    ):
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._closed = False
        self._cache_path = Path(cache_path).resolve()
        self._route_config = route_config
        self._occupied_keys = occupied_keys
        self._origin_ijk = origin_ijk
        self._grid_shape = grid_shape
        self._voxel_size = float(voxel_size)
        self._artifact_sha256 = str(artifact_sha256)
        self._checkers: Dict[float, Any] = {}
        self._routes: Dict[Tuple[float, Tuple[Tuple[str, object], ...]], Any] = {}
        self._numpy_input_copy_count = 0
        self._numpy_dtype_conversion_count = 0
        self._numpy_contiguity_conversion_count = 0
        self._numpy_output_array_allocation_count = 0
        self._map = _NativeVoxelMapOwner(
            self._occupied_keys,
            self._origin_ijk,
            self._grid_shape,
            self._voxel_size,
        )

    @classmethod
    def from_voxel_cache(
        cls,
        cache_path: Path,
        *,
        voxel_size: Optional[float] = None,
        route_config: Any = None,
        expected_sha256: Optional[str] = None,
    ) -> "NativeGeometryContext":
        path = Path(cache_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("voxel cache not found: {}".format(path))
        actual_sha256 = file_sha256(path)
        if expected_sha256 is not None and str(expected_sha256) != actual_sha256:
            raise ValueError(
                "voxel cache SHA256 mismatch: expected={} actual={}".format(
                    expected_sha256, actual_sha256
                )
            )
        with np.load(str(path), allow_pickle=False) as data:
            required = ("occupied_keys", "origin_ijk", "grid_shape", "voxel_size")
            missing = [name for name in required if name not in data.files]
            if missing:
                raise ValueError(
                    "voxel cache missing required fields: {}".format(",".join(missing))
                )
            occupied_value = np.asarray(data["occupied_keys"])
            if occupied_value.ndim != 1:
                raise ValueError(
                    "voxel cache occupied_keys shape {} != 1D".format(
                        occupied_value.shape
                    )
                )
            occupied_keys = _strict_cache_array(
                occupied_value,
                "occupied_keys",
                np.dtype(np.int64),
                (int(occupied_value.size),),
            )
            origin_ijk = _strict_cache_array(
                data["origin_ijk"], "origin_ijk", np.dtype(np.int64), (3,)
            )
            grid_shape = _strict_cache_array(
                data["grid_shape"], "grid_shape", np.dtype(np.int64), (3,)
            )
            cached_voxel_size_array = np.asarray(data["voxel_size"])
            if cached_voxel_size_array.shape != ():
                raise ValueError(
                    "voxel cache voxel_size shape {} != ()".format(
                        cached_voxel_size_array.shape
                    )
                )
            if cached_voxel_size_array.dtype not in (
                np.dtype(np.float32),
                np.dtype(np.float64),
            ):
                raise TypeError(
                    "voxel cache voxel_size dtype {} is not float32/float64".format(
                        cached_voxel_size_array.dtype
                    )
                )
            cached_voxel_size = float(cached_voxel_size_array)
        effective_voxel_size = cached_voxel_size
        if voxel_size is not None:
            effective_voxel_size = float(voxel_size)
            if not math.isclose(
                cached_voxel_size, effective_voxel_size, rel_tol=1.0e-6, abs_tol=1.0e-8
            ):
                raise ValueError(
                    "voxel cache size mismatch: cache={} requested={}".format(
                        cached_voxel_size, effective_voxel_size
                    )
                )
        if not math.isfinite(effective_voxel_size) or effective_voxel_size <= 0.0:
            raise ValueError("voxel cache voxel_size must be finite and positive")
        if np.any(grid_shape <= 0):
            raise ValueError("voxel cache grid_shape must be positive")
        route_config = route_config or cls._default_route_config()
        return cls(
            path,
            actual_sha256,
            occupied_keys,
            origin_ijk,
            grid_shape,
            effective_voxel_size,
            route_config,
        )

    @staticmethod
    def _default_route_config():
        from planning.mission.global_route import GlobalRouteConfig

        return GlobalRouteConfig()

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    @property
    def native_map_id(self) -> int:
        self._assert_open()
        return int(self._map.native_map_id)

    @property
    def map_identity(self) -> Dict[str, object]:
        return {
            "path": str(self._cache_path),
            "sha256": self._artifact_sha256,
            "voxel_size": float(self._voxel_size),
            "origin_ijk": [int(value) for value in self._origin_ijk],
            "grid_shape": [int(value) for value in self._grid_shape],
            "route_config": dict(_route_config_identity(self._route_config)),
        }

    @property
    def occupied_keys(self) -> np.ndarray:
        self._assert_open()
        return self._occupied_keys

    @property
    def origin_ijk(self) -> np.ndarray:
        self._assert_open()
        return self._origin_ijk

    @property
    def grid_shape(self) -> np.ndarray:
        self._assert_open()
        return self._grid_shape

    @property
    def voxel_size(self) -> float:
        self._assert_open()
        return float(self._voxel_size)

    @property
    def stats(self) -> Dict[str, object]:
        return {
            "pid": int(self._pid),
            "voxel_map_load_count": 1,
            "occupied_buffer_allocation_count": 1,
            "occupied_voxel_count": int(self._occupied_keys.size),
            "numpy_input_copy_count": int(self._numpy_input_copy_count),
            "numpy_dtype_conversion_count": int(self._numpy_dtype_conversion_count),
            "numpy_contiguity_conversion_count": int(
                self._numpy_contiguity_conversion_count
            ),
            "numpy_output_array_allocation_count": int(
                self._numpy_output_array_allocation_count
            ),
        }

    def _assert_open(self) -> None:
        if os.getpid() != self._pid:
            raise NativeGeometryError(
                "NATIVE_CONTEXT_FORK_REUSE_FORBIDDEN: context created in pid {} used in pid {}".format(
                    self._pid, os.getpid()
                )
            )
        if self._closed:
            raise NativeGeometryError("native geometry context is closed")
        self._map.assert_open("native geometry context")

    def _record_numpy_input(self, *, copied: bool, dtype_converted: bool) -> None:
        with self._lock:
            self._numpy_input_copy_count += int(bool(copied))
            self._numpy_dtype_conversion_count += int(bool(dtype_converted))
            self._numpy_contiguity_conversion_count += int(bool(copied))

    def _record_numpy_output(self, count: int = 1) -> None:
        with self._lock:
            self._numpy_output_array_allocation_count += max(0, int(count))

    def collision_checker(self, inflate_radius: float):
        """Return a cached public collision checker for this shared map."""
        self._assert_open()
        radius = float(inflate_radius)
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("inflate_radius must be finite and non-negative")
        with self._lock:
            existing = self._checkers.get(radius)
            if existing is not None:
                if not getattr(existing, "_closed", False):
                    return existing
                self._checkers.pop(radius, None)
            from planning.safety.collision_checker import VoxelCollisionChecker

            checker = VoxelCollisionChecker(
                self._occupied_keys,
                self._origin_ijk,
                self._grid_shape,
                self._voxel_size,
                radius,
                _native_map_handle=self._map.handle,
                _native_geometry_context=self,
            )
            self._checkers[radius] = checker
            return checker

    def global_route_planner(self, checker, config: Any = None):
        """Return a cached public route planner retaining this map owner."""
        self._assert_open()
        checker_assert_open = getattr(checker, "_assert_open", None)
        if callable(checker_assert_open):
            checker_assert_open()
        if getattr(checker, "native_geometry_context", None) is not self:
            raise ValueError("route checker must belong to this NativeGeometryContext")
        effective_config = config or self._route_config
        expected = _route_config_identity(self._route_config)
        actual = _route_config_identity(effective_config)
        if actual != expected:
            raise ValueError(
                "route config identity differs from NativeGeometryContext owner"
            )
        key = (
            float(checker.collision_radius),
            tuple(sorted(actual.items())),
        )
        with self._lock:
            existing = self._routes.get(key)
            if existing is not None:
                if not getattr(existing, "_closed", False):
                    return existing
                self._routes.pop(key, None)
            from planning.mission.global_route import GlobalRoutePlanner2D

            planner = GlobalRoutePlanner2D(
                checker,
                effective_config,
                _native_map_handle=self._map.handle,
                _native_geometry_context=self,
            )
            self._routes[key] = planner
            return planner

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if os.getpid() != self._pid:
                self._closed = True
                return
            for planner in list(self._routes.values()):
                planner.close()
            for checker in list(self._checkers.values()):
                checker.close()
            self._routes.clear()
            self._checkers.clear()
            self._map.close()
            self._closed = True

    def __enter__(self) -> "NativeGeometryContext":
        self._assert_open()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def __getstate__(self):
        raise TypeError("NativeGeometryContext cannot cross a process boundary")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
