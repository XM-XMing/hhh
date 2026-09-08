"""Native ctypes binding for the C++ collision backend.

The public collision domain module owns semantics; this module owns library
resolution, ABI signatures, native handles, and array marshalling.
"""

from __future__ import annotations

import ctypes
import os
from typing import Dict, Optional, Sequence

import numpy as np

from planning.native.geometry import _prepare_native_array
from planning.native.loader import (
    NativeLibraryError,
    load_library_path,
    load_native_library,
)


class _FastCollisionBackend:
    """Thin ctypes binding with array-oriented single- and multi-pose calls."""

    def __init__(
        self,
        occupied_keys: np.ndarray,
        origin_ijk: np.ndarray,
        grid_shape: np.ndarray,
        voxel_size: float,
        inflate_radius: float,
        native_map_handle=None,
        native_geometry_context=None,
        library_path=None,
    ):
        self._pid = os.getpid()
        self.native_geometry_context = native_geometry_context
        try:
            if library_path is None:
                self.library_path, self.lib = load_native_library("collision")
            else:
                self.library_path, self.lib = load_library_path(
                    library_path, "COLLISION"
                )
        except NativeLibraryError as error:
            raise RuntimeError(str(error)) from error
        self._setup_signatures()
        self._occupied_keys = np.ascontiguousarray(occupied_keys, dtype=np.int64)
        self._origin_ijk = np.ascontiguousarray(origin_ijk, dtype=np.int64).reshape(3)
        self._grid_shape = np.ascontiguousarray(grid_shape, dtype=np.int64).reshape(3)
        self.native_map_handle = native_map_handle
        self.native_map_id = (
            int(native_map_handle.value)
            if isinstance(native_map_handle, ctypes.c_void_p)
            else (int(native_map_handle) if native_map_handle else None)
        )
        if native_map_handle is None:
            self.handle = self.lib.planning_collision_create(
                self._occupied_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.c_int64(self._occupied_keys.size),
                self._origin_ijk.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                self._grid_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.c_double(float(voxel_size)),
                ctypes.c_double(float(inflate_radius)),
            )
        else:
            self.handle = self.lib.planning_collision_create_from_voxel_map(
                native_map_handle,
                ctypes.c_double(float(inflate_radius)),
            )
        if not self.handle:
            raise RuntimeError("planning_collision_create returned null")
        self.thread_count = max(1, int(os.environ.get("PLANNING_COLLISION_THREADS", "1")))
        if hasattr(self.lib, "planning_collision_set_threads"):
            self.lib.planning_collision_set_threads(self.handle, ctypes.c_int(self.thread_count))
        self.collision_radius = float(self.lib.planning_collision_radius(self.handle))

    def _setup_signatures(self) -> None:
        self.lib.planning_collision_create.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_double,
            ctypes.c_double,
        ]
        self.lib.planning_collision_create.restype = ctypes.c_void_p
        self.lib.planning_collision_create_from_voxel_map.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
        ]
        self.lib.planning_collision_create_from_voxel_map.restype = ctypes.c_void_p
        self.lib.planning_collision_destroy.argtypes = [ctypes.c_void_p]
        self.lib.planning_collision_destroy.restype = None
        if hasattr(self.lib, "planning_collision_set_threads"):
            self.lib.planning_collision_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.planning_collision_set_threads.restype = None
        self.lib.planning_collision_radius.argtypes = [ctypes.c_void_p]
        self.lib.planning_collision_radius.restype = ctypes.c_double
        self.lib.planning_collision_check_path.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.planning_collision_check_path.restype = ctypes.c_int
        self.lib.planning_collision_check_action.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.planning_collision_check_action.restype = ctypes.c_int
        self.lib.planning_collision_check_actions_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.planning_collision_check_actions_batch.restype = ctypes.c_int
        if hasattr(self.lib, "planning_collision_check_pose_actions_batch"):
            self.lib.planning_collision_check_pose_actions_batch.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_float),
            ]
            self.lib.planning_collision_check_pose_actions_batch.restype = ctypes.c_int

    @property
    def native_checker_id(self) -> Optional[int]:
        handle = getattr(self, "handle", None)
        if not handle:
            return None
        value = getattr(handle, "value", handle)
        return None if value is None else int(value)

    def close(self) -> None:
        if getattr(self, "handle", None):
            if os.getpid() == self._pid:
                self.lib.planning_collision_destroy(self.handle)
            self.handle = None

    def _assert_open(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError(
                "NATIVE_CONTEXT_FORK_REUSE_FORBIDDEN: collision backend"
            )
        if not getattr(self, "handle", None):
            raise RuntimeError("collision backend is closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _pos_ref(self, motion_primitives) -> np.ndarray:
        values = _prepare_native_array(
            motion_primitives.pos_ref,
            np.dtype(np.float32),
            None,
            "pos_ref",
            self.native_geometry_context,
        )
        if values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("pos_ref must have shape [actions, points, 3]")
        return values

    def check_path(self, path_map: np.ndarray, check_step: int = 1) -> Dict:
        path = _prepare_native_array(
            path_map,
            np.dtype(np.float32),
            (-1, 3),
            "path_map",
            self.native_geometry_context,
        )
        collision = ctypes.c_int(0)
        minimum_distance = ctypes.c_double(float("inf"))
        first_index = ctypes.c_int32(-1)
        first_point = np.zeros((3,), dtype=np.float32)
        result = self.lib.planning_collision_check_path(
            self.handle,
            path.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(path.shape[0]),
            ctypes.c_int(max(1, int(check_step))),
            ctypes.byref(collision),
            ctypes.byref(minimum_distance),
            ctypes.byref(first_index),
            first_point.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if result != 0:
            raise RuntimeError(f"planning_collision_check_path failed: {result}")
        hit = bool(collision.value)
        if self.native_geometry_context is not None:
            self.native_geometry_context._record_numpy_output(1 + int(hit))
        return {
            "collision": hit,
            "valid": not hit,
            "min_distance_voxel_center_m": float(minimum_distance.value),
            "collision_radius_m": self.collision_radius,
            "first_collision_index": int(first_index.value),
            "first_collision_point": first_point.copy() if hit else None,
        }

    def check_action(self, motion_primitives, action_id, position, yaw, check_step=1) -> Dict:
        arrays = self.check_actions_array(
            motion_primitives, [int(action_id)], position, yaw, check_step
        )
        hit = bool(arrays["collisions"][0])
        return {
            "action_id": int(action_id),
            "collision": hit,
            "valid": not hit,
            "min_distance_voxel_center_m": float(arrays["minimum_distances"][0]),
            "collision_radius_m": self.collision_radius,
            "first_collision_index": int(arrays["first_collision_indices"][0]),
            "first_collision_point": None,
            "endpoint_map": arrays["endpoints"][0].copy(),
        }

    def check_actions_array(
        self,
        motion_primitives,
        action_ids: Sequence[int],
        position: Sequence[float],
        yaw: float,
        check_step: int = 1,
    ) -> Dict[str, np.ndarray]:
        ids = _prepare_native_array(
            action_ids,
            np.dtype(np.int32),
            (-1,),
            "action_ids",
            self.native_geometry_context,
        )
        pos_ref = self._pos_ref(motion_primitives)
        position_array = _prepare_native_array(
            position,
            np.dtype(np.float32),
            (3,),
            "position",
            self.native_geometry_context,
        )
        collisions = np.empty((ids.size,), dtype=np.int32)
        minimum_distances = np.empty((ids.size,), dtype=np.float64)
        first_indices = np.empty((ids.size,), dtype=np.int32)
        endpoints = np.empty((ids.size, 3), dtype=np.float32)
        result = self.lib.planning_collision_check_actions_batch(
            self.handle,
            pos_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(pos_ref.shape[0]),
            ctypes.c_int64(pos_ref.shape[1]),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int64(ids.size),
            position_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_double(float(yaw)),
            ctypes.c_int(max(1, int(check_step))),
            collisions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            minimum_distances.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            first_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            endpoints.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if result != 0:
            raise RuntimeError(f"planning_collision_check_actions_batch failed: {result}")
        if self.native_geometry_context is not None:
            # Four native output buffers plus the bool view returned to callers.
            self.native_geometry_context._record_numpy_output(5)
        return {
            "action_ids": ids,
            "collisions": collisions.astype(np.bool_),
            "minimum_distances": minimum_distances,
            "first_collision_indices": first_indices,
            "endpoints": endpoints,
        }


    def check_pose_actions_array(
        self,
        motion_primitives,
        positions: np.ndarray,
        yaws: np.ndarray,
        action_ids: Sequence[int],
        check_step: int = 1,
    ) -> Dict[str, np.ndarray]:
        positions_array = _prepare_native_array(
            positions,
            np.dtype(np.float32),
            (-1, 3),
            "positions",
            self.native_geometry_context,
        )
        yaws_array = _prepare_native_array(
            yaws,
            np.dtype(np.float64),
            (-1,),
            "yaws",
            self.native_geometry_context,
        )
        if positions_array.shape[0] != yaws_array.size:
            raise ValueError("positions and yaws must contain the same number of poses")
        ids = _prepare_native_array(
            action_ids,
            np.dtype(np.int32),
            (-1,),
            "action_ids",
            self.native_geometry_context,
        )
        if not hasattr(self.lib, "planning_collision_check_pose_actions_batch"):
            rows = [
                self.check_actions_array(
                    motion_primitives, ids, positions_array[index], yaws_array[index], check_step
                )
                for index in range(positions_array.shape[0])
            ]
            return {
                "action_ids": ids,
                "collisions": np.stack([row["collisions"] for row in rows]),
                "minimum_distances": np.stack([row["minimum_distances"] for row in rows]),
                "first_collision_indices": np.stack(
                    [row["first_collision_indices"] for row in rows]
                ),
                "endpoints": np.stack([row["endpoints"] for row in rows]),
            }
        pos_ref = self._pos_ref(motion_primitives)
        shape = (positions_array.shape[0], ids.size)
        collisions = np.empty(shape, dtype=np.int32)
        minimum_distances = np.empty(shape, dtype=np.float64)
        first_indices = np.empty(shape, dtype=np.int32)
        endpoints = np.empty(shape + (3,), dtype=np.float32)
        result = self.lib.planning_collision_check_pose_actions_batch(
            self.handle,
            pos_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(pos_ref.shape[0]),
            ctypes.c_int64(pos_ref.shape[1]),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int64(ids.size),
            positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            yaws_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(positions_array.shape[0]),
            ctypes.c_int(max(1, int(check_step))),
            collisions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            minimum_distances.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            first_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            endpoints.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if result != 0:
            raise RuntimeError(f"planning_collision_check_pose_actions_batch failed: {result}")
        if self.native_geometry_context is not None:
            # Four native output buffers plus the bool view returned to callers.
            self.native_geometry_context._record_numpy_output(5)
        return {
            "action_ids": ids,
            "collisions": collisions.astype(np.bool_),
            "minimum_distances": minimum_distances,
            "first_collision_indices": first_indices,
            "endpoints": endpoints,
        }


    def check_actions_batch(self, motion_primitives, action_ids, position, yaw, check_step=1):
        arrays = self.check_actions_array(
            motion_primitives, action_ids, position, yaw, check_step
        )
        return [
            {
                "action_id": int(action_id),
                "collision": bool(arrays["collisions"][index]),
                "valid": not bool(arrays["collisions"][index]),
                "min_distance_voxel_center_m": float(
                    arrays["minimum_distances"][index]
                ),
                "collision_radius_m": self.collision_radius,
                "first_collision_index": int(
                    arrays["first_collision_indices"][index]
                ),
                "first_collision_point": None,
                "endpoint_map": arrays["endpoints"][index].copy(),
            }
            for index, action_id in enumerate(arrays["action_ids"])
        ]


__all__ = ["_FastCollisionBackend"]
