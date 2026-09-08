"""Native ctypes binding for the C++ global-route backend.

The route domain module owns configuration and public semantics; this module
owns native library resolution, ABI signatures, handles, and marshalling.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional, Sequence

import numpy as np

from planning.native.geometry import _prepare_native_array
from planning.native.loader import NativeLibraryError, load_library_path, load_native_library


class _NativeGlobalRouteBackend:
    def __init__(self, checker, config, native_map_handle=None, native_geometry_context=None, library_path=None):
        self._pid = os.getpid()
        self.native_geometry_context = native_geometry_context
        self.native_map_handle = native_map_handle
        self.native_map_id = (
            int(native_map_handle.value)
            if isinstance(native_map_handle, ctypes.c_void_p)
            else (int(native_map_handle) if native_map_handle else None)
        )
        self.handle = None
        try:
            if library_path is None:
                self.library_path, self.lib = load_native_library("global_route")
            else:
                self.library_path, self.lib = load_library_path(
                    library_path, "GLOBAL_ROUTE"
                )
        except NativeLibraryError as error:
            raise RuntimeError(str(error)) from error
        try:
            self._setup_signatures()
        except AttributeError as error:
            raise RuntimeError(
                "GLOBAL_ROUTE_NATIVE_SYMBOL_MISSING: {}".format(error)
            ) from error
        except Exception as error:
            raise RuntimeError(
                "GLOBAL_ROUTE_NATIVE_INITIALIZATION_FAILED: {}".format(repr(error))
            ) from error
        if native_map_handle is None:
            occupied = np.ascontiguousarray(checker.occupied_keys, dtype=np.int64)
            origin = np.ascontiguousarray(checker.origin_ijk, dtype=np.int64).reshape(3)
            grid_shape = np.ascontiguousarray(checker.grid_shape, dtype=np.int64).reshape(3)
            self.handle = self.lib.planning_global_route_create(
                occupied.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.c_int64(occupied.size),
                origin.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                grid_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.c_double(float(checker.voxel_size)),
                ctypes.c_double(float(checker.collision_radius)),
                ctypes.c_double(float(config.resolution_m)),
                ctypes.c_double(float(config.flight_z_min_m)),
                ctypes.c_double(float(config.flight_z_max_m)),
                ctypes.c_double(float(config.tracking_margin_m)),
                ctypes.c_int(int(config.nearest_free_radius_cells)),
            )
        else:
            self.handle = self.lib.planning_global_route_create_from_voxel_map(
                native_map_handle,
                ctypes.c_double(float(checker.collision_radius)),
                ctypes.c_double(float(config.resolution_m)),
                ctypes.c_double(float(config.flight_z_min_m)),
                ctypes.c_double(float(config.flight_z_max_m)),
                ctypes.c_double(float(config.tracking_margin_m)),
                ctypes.c_int(int(config.nearest_free_radius_cells)),
            )
        if not self.handle:
            raise RuntimeError(
                "GLOBAL_ROUTE_NATIVE_INITIALIZATION_FAILED: "
                "planning_global_route_create returned null"
            )
        self.native_planner_id = (
            int(self.handle.value)
            if isinstance(self.handle, ctypes.c_void_p)
            else int(self.handle)
        )
        shape = np.empty((2,), dtype=np.int32)
        origin_xy = np.empty((2,), dtype=np.float64)
        resolution = ctypes.c_double()
        rc = self.lib.planning_global_route_shape(
            self.handle,
            shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            origin_xy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(resolution),
        )
        if rc != 0:
            self.close()
            raise RuntimeError(
                "GLOBAL_ROUTE_NATIVE_INITIALIZATION_FAILED: "
                "planning_global_route_shape failed: {}".format(rc)
            )
        self.shape = shape
        self.origin_xy = origin_xy
        self.resolution_m = float(resolution.value)
        blocked_flat = np.empty((int(shape[0]) * int(shape[1]),), dtype=np.uint8)
        rc = self.lib.planning_global_route_copy_blocked(
            self.handle,
            blocked_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int64(blocked_flat.size),
        )
        if rc != 0:
            self.close()
            raise RuntimeError(
                "GLOBAL_ROUTE_NATIVE_INITIALIZATION_FAILED: "
                "planning_global_route_copy_blocked failed: {}".format(rc)
            )
        self.blocked = blocked_flat.reshape((int(shape[0]), int(shape[1]))).astype(np.bool_)
        if self.native_geometry_context is not None:
            # shape, origin, blocked_flat, and the public bool blocked view.
            self.native_geometry_context._record_numpy_output(4)

    def _setup_signatures(self) -> None:
        self.lib.planning_global_route_create.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]
        self.lib.planning_global_route_create.restype = ctypes.c_void_p
        self.lib.planning_global_route_create_from_voxel_map.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]
        self.lib.planning_global_route_create_from_voxel_map.restype = ctypes.c_void_p
        self.lib.planning_global_route_destroy.argtypes = [ctypes.c_void_p]
        self.lib.planning_global_route_destroy.restype = None
        self.lib.planning_global_route_shape.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.planning_global_route_shape.restype = ctypes.c_int
        self.lib.planning_global_route_copy_blocked.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64,
        ]
        self.lib.planning_global_route_copy_blocked.restype = ctypes.c_int
        self.lib.planning_global_route_plan.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.planning_global_route_plan.restype = ctypes.c_int
        self.lib.planning_global_route_last_point_count.argtypes = [ctypes.c_void_p]
        self.lib.planning_global_route_last_point_count.restype = ctypes.c_int64
        self.lib.planning_global_route_copy_last_route.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        self.lib.planning_global_route_copy_last_route.restype = ctypes.c_int

    def close(self) -> None:
        if getattr(self, "handle", None):
            if os.getpid() == self._pid:
                self.lib.planning_global_route_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def plan(self, start: Sequence[float], goal: Sequence[float]) -> np.ndarray:
        if os.getpid() != getattr(self, "_pid", os.getpid()):
            raise RuntimeError("NATIVE_CONTEXT_FORK_REUSE_FORBIDDEN: route backend")
        if not self.handle:
            raise RuntimeError("global route planner is closed")
        start_array = _prepare_native_array(
            start, np.dtype(np.float32), (3,), "start", self.native_geometry_context
        )
        goal_array = _prepare_native_array(
            goal, np.dtype(np.float32), (3,), "goal", self.native_geometry_context
        )
        # The O Python contract feeds the original scalar z values to
        # np.linspace before storing the result as float32.  The C ABI keeps
        # its established float32 input shape, so restore that exact public
        # output seam after the native planner returns.
        start_z = float(start[2])
        goal_z = float(goal[2])
        try:
            rc = self.lib.planning_global_route_plan(
                self.handle,
                start_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                goal_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        except Exception as error:
            raise RuntimeError(
                "GLOBAL_ROUTE_NATIVE_EXECUTION_FAILED: {}".format(repr(error))
            ) from error
        if rc == 1:
            from planning.mission.global_route import GlobalRouteUnavailableError
            raise GlobalRouteUnavailableError(
                "no global route from {} to {}".format(start_array.tolist(), goal_array.tolist())
            )
        if rc != 0:
            raise RuntimeError(
                "GLOBAL_ROUTE_NATIVE_EXECUTION_FAILED: "
                "planning_global_route_plan failed: {}".format(rc)
            )
        try:
            point_count = int(self.lib.planning_global_route_last_point_count(self.handle))
        except Exception as error:
            raise RuntimeError(
                "GLOBAL_ROUTE_NATIVE_EXECUTION_FAILED: {}".format(repr(error))
            ) from error
        if point_count <= 0:
            raise RuntimeError(
                "GLOBAL_ROUTE_NATIVE_EXECUTION_FAILED: "
                "native global route returned no points"
            )
        route = np.empty((point_count, 3), dtype=np.float32)
        try:
            rc = self.lib.planning_global_route_copy_last_route(
                self.handle,
                route.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int64(point_count),
            )
        except Exception as error:
            raise RuntimeError(
                "GLOBAL_ROUTE_NATIVE_EXECUTION_FAILED: {}".format(repr(error))
            ) from error
        if rc != 0:
            raise RuntimeError(
                "GLOBAL_ROUTE_NATIVE_EXECUTION_FAILED: "
                "planning_global_route_copy_last_route failed: {}".format(rc)
            )
        if point_count > 2:
            route[1:-1, 2] = np.linspace(
                start_z, goal_z, point_count - 2, dtype=np.float32
            )
        if self.native_geometry_context is not None:
            self.native_geometry_context._record_numpy_output(1)
        return route


__all__ = ["_NativeGlobalRouteBackend"]
