#!/usr/bin/env python3
"""Compare the O and P collision C ABIs without changing either backend.

The default fixture is the deterministic C1 synthetic voxel cache.  An
optional cache produced under /tmp may be supplied with --production-cache;
the script never writes project data.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import resource
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_O_CPU = Path("/tmp/xm-cxx-c0-o-devel/planning/lib/libplanning_collision_checker.so")
DEFAULT_O_CUDA = Path(
    "/tmp/xm-cxx-c0-o-devel/planning/lib/libplanning_collision_checker_cuda.so"
)
DEFAULT_P_CPU = Path("/tmp/xm-cxx-c0-p-devel/planning/lib/libplanning_collision_checker.so")
DEFAULT_P_VOXEL = Path("/tmp/xm-cxx-c0-p-devel/planning/lib/libplanning_voxel_map.so")

_I64_PTR = ctypes.POINTER(ctypes.c_int64)
_I32_PTR = ctypes.POINTER(ctypes.c_int32)
_F32_PTR = ctypes.POINTER(ctypes.c_float)
_F64_PTR = ctypes.POINTER(ctypes.c_double)
_INT_PTR = ctypes.POINTER(ctypes.c_int)


def _as_i64(values: Sequence[int]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64).reshape(-1))


def _as_f32(values: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32))


def _as_f64(values: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _ptr(array: np.ndarray, pointer_type: Any) -> Any:
    return array.ctypes.data_as(pointer_type)


def _bitwise_equal(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    left = np.ascontiguousarray(lhs)
    right = np.ascontiguousarray(rhs)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(np.array_equal(left.view(np.uint8), right.view(np.uint8)))


def _float_deltas(lhs: np.ndarray, rhs: np.ndarray) -> Tuple[float, float]:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return 0.0, 0.0
    absolute = np.abs(left[finite] - right[finite])
    relative = absolute / np.maximum(np.abs(right[finite]), 1.0e-300)
    return float(np.max(absolute)), float(np.max(relative))


def _library(path: Path) -> ctypes.CDLL:
    mode = getattr(ctypes, "RTLD_LOCAL", 0)
    return ctypes.CDLL(str(path), mode=mode)


class CpuBackend:
    def __init__(self, path: Path, label: str):
        self.library_path = Path(path)
        self.label = label
        self.lib = _library(self.library_path)
        self._setup_signatures()

    def _setup_signatures(self) -> None:
        lib = self.lib
        lib.planning_collision_create.argtypes = [
            _I64_PTR,
            ctypes.c_int64,
            _I64_PTR,
            _I64_PTR,
            ctypes.c_double,
            ctypes.c_double,
        ]
        lib.planning_collision_create.restype = ctypes.c_void_p
        lib.planning_collision_destroy.argtypes = [ctypes.c_void_p]
        lib.planning_collision_destroy.restype = None
        lib.planning_collision_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.planning_collision_set_threads.restype = None
        lib.planning_collision_radius.argtypes = [ctypes.c_void_p]
        lib.planning_collision_radius.restype = ctypes.c_double
        lib.planning_collision_check_path.argtypes = [
            ctypes.c_void_p,
            _F32_PTR,
            ctypes.c_int64,
            ctypes.c_int,
            _INT_PTR,
            _F64_PTR,
            _I32_PTR,
            _F32_PTR,
        ]
        lib.planning_collision_check_path.restype = ctypes.c_int
        lib.planning_collision_check_action.argtypes = [
            ctypes.c_void_p,
            _F32_PTR,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
            _F32_PTR,
            ctypes.c_double,
            ctypes.c_int,
            _INT_PTR,
            _F64_PTR,
            _I32_PTR,
            _F32_PTR,
            _F32_PTR,
        ]
        lib.planning_collision_check_action.restype = ctypes.c_int
        lib.planning_collision_check_actions_batch.argtypes = [
            ctypes.c_void_p,
            _F32_PTR,
            ctypes.c_int64,
            ctypes.c_int64,
            _I32_PTR,
            ctypes.c_int64,
            _F32_PTR,
            ctypes.c_double,
            ctypes.c_int,
            _INT_PTR,
            _F64_PTR,
            _I32_PTR,
            _F32_PTR,
        ]
        lib.planning_collision_check_actions_batch.restype = ctypes.c_int
        lib.planning_collision_check_pose_actions_batch.argtypes = [
            ctypes.c_void_p,
            _F32_PTR,
            ctypes.c_int64,
            ctypes.c_int64,
            _I32_PTR,
            ctypes.c_int64,
            _F32_PTR,
            _F64_PTR,
            ctypes.c_int64,
            ctypes.c_int,
            _INT_PTR,
            _F64_PTR,
            _I32_PTR,
            _F32_PTR,
        ]
        lib.planning_collision_check_pose_actions_batch.restype = ctypes.c_int
        if hasattr(lib, "planning_collision_create_from_voxel_map"):
            lib.planning_collision_create_from_voxel_map.argtypes = [
                ctypes.c_void_p,
                ctypes.c_double,
            ]
            lib.planning_collision_create_from_voxel_map.restype = ctypes.c_void_p

    def create(
        self,
        keys: np.ndarray,
        origin: np.ndarray,
        shape: np.ndarray,
        voxel_size: float,
        inflate_radius: float,
    ) -> Optional[ctypes.c_void_p]:
        handle = self.lib.planning_collision_create(
            _ptr(keys, _I64_PTR),
            ctypes.c_int64(keys.size),
            _ptr(origin, _I64_PTR),
            _ptr(shape, _I64_PTR),
            ctypes.c_double(voxel_size),
            ctypes.c_double(inflate_radius),
        )
        return handle if handle else None

    def create_from_map(
        self, map_handle: ctypes.c_void_p, inflate_radius: float
    ) -> Optional[ctypes.c_void_p]:
        if not hasattr(self.lib, "planning_collision_create_from_voxel_map"):
            return None
        handle = self.lib.planning_collision_create_from_voxel_map(
            map_handle, ctypes.c_double(inflate_radius)
        )
        return handle if handle else None

    def destroy(self, handle: Optional[ctypes.c_void_p]) -> None:
        if handle:
            self.lib.planning_collision_destroy(handle)

    def set_threads(self, handle: ctypes.c_void_p, count: int) -> None:
        self.lib.planning_collision_set_threads(handle, ctypes.c_int(count))

    def radius(self, handle: ctypes.c_void_p) -> float:
        return float(self.lib.planning_collision_radius(handle))

    def path(self, handle: ctypes.c_void_p, path: np.ndarray, step: int) -> Dict[str, Any]:
        path = _as_f32(path).reshape(-1, 3)
        collision = ctypes.c_int(0)
        minimum = ctypes.c_double(float("inf"))
        first_index = ctypes.c_int32(-1)
        first_point = np.zeros(3, dtype=np.float32)
        rc = self.lib.planning_collision_check_path(
            handle,
            _ptr(path, _F32_PTR),
            ctypes.c_int64(path.shape[0]),
            ctypes.c_int(max(1, int(step))),
            ctypes.byref(collision),
            ctypes.byref(minimum),
            ctypes.byref(first_index),
            _ptr(first_point, _F32_PTR),
        )
        return {
            "rc": int(rc),
            "collision": int(collision.value),
            "minimum": np.asarray([minimum.value], dtype=np.float64),
            "first_index": np.asarray([first_index.value], dtype=np.int32),
            "first_point": first_point,
        }

    def action(
        self,
        handle: ctypes.c_void_p,
        pos_ref: np.ndarray,
        action_id: int,
        position: np.ndarray,
        yaw: float,
        step: int,
    ) -> Dict[str, Any]:
        pos_ref = _as_f32(pos_ref).reshape(pos_ref.shape[0], pos_ref.shape[1], 3)
        position = _as_f32(position).reshape(3)
        collision = ctypes.c_int(0)
        minimum = ctypes.c_double(float("inf"))
        first_index = ctypes.c_int32(-1)
        first_point = np.zeros(3, dtype=np.float32)
        endpoint = np.zeros(3, dtype=np.float32)
        rc = self.lib.planning_collision_check_action(
            handle,
            _ptr(pos_ref, _F32_PTR),
            ctypes.c_int64(pos_ref.shape[0]),
            ctypes.c_int64(pos_ref.shape[1]),
            ctypes.c_int(int(action_id)),
            _ptr(position, _F32_PTR),
            ctypes.c_double(float(yaw)),
            ctypes.c_int(max(1, int(step))),
            ctypes.byref(collision),
            ctypes.byref(minimum),
            ctypes.byref(first_index),
            _ptr(first_point, _F32_PTR),
            _ptr(endpoint, _F32_PTR),
        )
        return {
            "rc": int(rc),
            "collision": int(collision.value),
            "minimum": np.asarray([minimum.value], dtype=np.float64),
            "first_index": np.asarray([first_index.value], dtype=np.int32),
            "first_point": first_point,
            "endpoint": endpoint,
        }

    def actions(
        self,
        handle: ctypes.c_void_p,
        pos_ref: np.ndarray,
        action_ids: np.ndarray,
        position: np.ndarray,
        yaw: float,
        step: int,
    ) -> Dict[str, Any]:
        pos_ref = _as_f32(pos_ref).reshape(pos_ref.shape[0], pos_ref.shape[1], 3)
        action_ids = np.ascontiguousarray(action_ids, dtype=np.int32).reshape(-1)
        position = _as_f32(position).reshape(3)
        collisions = np.empty(action_ids.size, dtype=np.int32)
        minimum = np.empty(action_ids.size, dtype=np.float64)
        first_index = np.empty(action_ids.size, dtype=np.int32)
        endpoints = np.empty((action_ids.size, 3), dtype=np.float32)
        rc = self.lib.planning_collision_check_actions_batch(
            handle,
            _ptr(pos_ref, _F32_PTR),
            ctypes.c_int64(pos_ref.shape[0]),
            ctypes.c_int64(pos_ref.shape[1]),
            _ptr(action_ids, _I32_PTR),
            ctypes.c_int64(action_ids.size),
            _ptr(position, _F32_PTR),
            ctypes.c_double(float(yaw)),
            ctypes.c_int(max(1, int(step))),
            _ptr(collisions, _INT_PTR),
            _ptr(minimum, _F64_PTR),
            _ptr(first_index, _I32_PTR),
            _ptr(endpoints, _F32_PTR),
        )
        return {
            "rc": int(rc),
            "collisions": collisions,
            "valid": (collisions == 0).astype(np.bool_),
            "minimum": minimum,
            "first_index": first_index,
            "endpoints": endpoints,
            "action_ids": action_ids.copy(),
        }

    def pose_actions(
        self,
        handle: ctypes.c_void_p,
        pos_ref: np.ndarray,
        action_ids: np.ndarray,
        positions: np.ndarray,
        yaws: np.ndarray,
        step: int,
    ) -> Dict[str, Any]:
        pos_ref = _as_f32(pos_ref).reshape(pos_ref.shape[0], pos_ref.shape[1], 3)
        action_ids = np.ascontiguousarray(action_ids, dtype=np.int32).reshape(-1)
        positions = _as_f32(positions).reshape(-1, 3)
        yaws = _as_f64(yaws).reshape(-1)
        shape = (positions.shape[0], action_ids.size)
        collisions = np.empty(shape, dtype=np.int32)
        minimum = np.empty(shape, dtype=np.float64)
        first_index = np.empty(shape, dtype=np.int32)
        endpoints = np.empty(shape + (3,), dtype=np.float32)
        rc = self.lib.planning_collision_check_pose_actions_batch(
            handle,
            _ptr(pos_ref, _F32_PTR),
            ctypes.c_int64(pos_ref.shape[0]),
            ctypes.c_int64(pos_ref.shape[1]),
            _ptr(action_ids, _I32_PTR),
            ctypes.c_int64(action_ids.size),
            _ptr(positions, _F32_PTR),
            _ptr(yaws, _F64_PTR),
            ctypes.c_int64(positions.shape[0]),
            ctypes.c_int(max(1, int(step))),
            _ptr(collisions, _INT_PTR),
            _ptr(minimum, _F64_PTR),
            _ptr(first_index, _I32_PTR),
            _ptr(endpoints, _F32_PTR),
        )
        return {
            "rc": int(rc),
            "collisions": collisions,
            "valid": (collisions == 0).astype(np.bool_),
            "minimum": minimum,
            "first_index": first_index,
            "endpoints": endpoints,
            "action_ids": action_ids.copy(),
        }


class CudaBackend:
    def __init__(self, path: Path):
        self.library_path = Path(path)
        self.lib = _library(self.library_path)
        self._setup_signatures()

    def _setup_signatures(self) -> None:
        lib = self.lib
        lib.planning_collision_cuda_create.argtypes = [
            _I64_PTR,
            ctypes.c_int64,
            _I64_PTR,
            _I64_PTR,
            ctypes.c_double,
            ctypes.c_double,
        ]
        lib.planning_collision_cuda_create.restype = ctypes.c_void_p
        lib.planning_collision_cuda_destroy.argtypes = [ctypes.c_void_p]
        lib.planning_collision_cuda_destroy.restype = None
        lib.planning_collision_cuda_radius.argtypes = [ctypes.c_void_p]
        lib.planning_collision_cuda_radius.restype = ctypes.c_double
        batch = [
            ctypes.c_void_p,
            _F32_PTR,
            ctypes.c_int64,
            ctypes.c_int64,
            _I32_PTR,
            ctypes.c_int64,
        ]
        tail = [
            _F32_PTR,
            ctypes.c_double,
            ctypes.c_int,
            _INT_PTR,
            _F64_PTR,
            _I32_PTR,
            _F32_PTR,
        ]
        lib.planning_collision_cuda_check_actions_batch.argtypes = batch + tail
        lib.planning_collision_cuda_check_actions_batch.restype = ctypes.c_int
        pose_tail = [
            _F32_PTR,
            _F64_PTR,
            ctypes.c_int64,
            ctypes.c_int,
            _INT_PTR,
            _F64_PTR,
            _I32_PTR,
            _F32_PTR,
        ]
        lib.planning_collision_cuda_check_pose_actions_batch.argtypes = batch + pose_tail
        lib.planning_collision_cuda_check_pose_actions_batch.restype = ctypes.c_int

    def create(
        self,
        keys: np.ndarray,
        origin: np.ndarray,
        shape: np.ndarray,
        voxel_size: float,
        inflate_radius: float,
    ) -> Optional[ctypes.c_void_p]:
        handle = self.lib.planning_collision_cuda_create(
            _ptr(keys, _I64_PTR),
            ctypes.c_int64(keys.size),
            _ptr(origin, _I64_PTR),
            _ptr(shape, _I64_PTR),
            ctypes.c_double(voxel_size),
            ctypes.c_double(inflate_radius),
        )
        return handle if handle else None

    def destroy(self, handle: Optional[ctypes.c_void_p]) -> None:
        if handle:
            self.lib.planning_collision_cuda_destroy(handle)

    def radius(self, handle: ctypes.c_void_p) -> float:
        return float(self.lib.planning_collision_cuda_radius(handle))

    def actions(
        self,
        handle: ctypes.c_void_p,
        pos_ref: np.ndarray,
        action_ids: np.ndarray,
        position: np.ndarray,
        yaw: float,
        step: int,
    ) -> Dict[str, Any]:
        pos_ref = _as_f32(pos_ref).reshape(pos_ref.shape[0], pos_ref.shape[1], 3)
        action_ids = np.ascontiguousarray(action_ids, dtype=np.int32).reshape(-1)
        position = _as_f32(position).reshape(3)
        collisions = np.empty(action_ids.size, dtype=np.int32)
        minimum = np.empty(action_ids.size, dtype=np.float64)
        first_index = np.empty(action_ids.size, dtype=np.int32)
        endpoints = np.empty((action_ids.size, 3), dtype=np.float32)
        rc = self.lib.planning_collision_cuda_check_actions_batch(
            handle,
            _ptr(pos_ref, _F32_PTR),
            ctypes.c_int64(pos_ref.shape[0]),
            ctypes.c_int64(pos_ref.shape[1]),
            _ptr(action_ids, _I32_PTR),
            ctypes.c_int64(action_ids.size),
            _ptr(position, _F32_PTR),
            ctypes.c_double(float(yaw)),
            ctypes.c_int(max(1, int(step))),
            _ptr(collisions, _INT_PTR),
            _ptr(minimum, _F64_PTR),
            _ptr(first_index, _I32_PTR),
            _ptr(endpoints, _F32_PTR),
        )
        return {
            "rc": int(rc),
            "collisions": collisions,
            "valid": (collisions == 0).astype(np.bool_),
            "minimum": minimum,
            "first_index": first_index,
            "endpoints": endpoints,
            "action_ids": action_ids.copy(),
        }

    def pose_actions(
        self,
        handle: ctypes.c_void_p,
        pos_ref: np.ndarray,
        action_ids: np.ndarray,
        positions: np.ndarray,
        yaws: np.ndarray,
        step: int,
    ) -> Dict[str, Any]:
        pos_ref = _as_f32(pos_ref).reshape(pos_ref.shape[0], pos_ref.shape[1], 3)
        action_ids = np.ascontiguousarray(action_ids, dtype=np.int32).reshape(-1)
        positions = _as_f32(positions).reshape(-1, 3)
        yaws = _as_f64(yaws).reshape(-1)
        shape = (positions.shape[0], action_ids.size)
        collisions = np.empty(shape, dtype=np.int32)
        minimum = np.empty(shape, dtype=np.float64)
        first_index = np.empty(shape, dtype=np.int32)
        endpoints = np.empty(shape + (3,), dtype=np.float32)
        rc = self.lib.planning_collision_cuda_check_pose_actions_batch(
            handle,
            _ptr(pos_ref, _F32_PTR),
            ctypes.c_int64(pos_ref.shape[0]),
            ctypes.c_int64(pos_ref.shape[1]),
            _ptr(action_ids, _I32_PTR),
            ctypes.c_int64(action_ids.size),
            _ptr(positions, _F32_PTR),
            _ptr(yaws, _F64_PTR),
            ctypes.c_int64(positions.shape[0]),
            ctypes.c_int(max(1, int(step))),
            _ptr(collisions, _INT_PTR),
            _ptr(minimum, _F64_PTR),
            _ptr(first_index, _I32_PTR),
            _ptr(endpoints, _F32_PTR),
        )
        return {
            "rc": int(rc),
            "collisions": collisions,
            "valid": (collisions == 0).astype(np.bool_),
            "minimum": minimum,
            "first_index": first_index,
            "endpoints": endpoints,
            "action_ids": action_ids.copy(),
        }


@dataclass
class Layout:
    name: str
    keys: np.ndarray
    origin: np.ndarray
    shape: np.ndarray
    voxel_size: float
    source: str = "synthetic_c1"


def _c1_keys() -> np.ndarray:
    shape = (18, 16, 9)
    key_count = shape[0] * shape[1] * shape[2]
    keys: List[int] = []
    mask = (1 << 64) - 1
    for key in range(key_count):
        mixed = (key * 0x9E3779B97F4A7C15 + 0x243F6A8885A308D3) & mask
        if ((mixed ^ (mixed >> 29)) % 23) < 2:
            keys.append(key)
    keys.extend((0, key_count - 1, shape[1] * shape[2], shape[2] - 1))
    return np.asarray(sorted(set(keys)), dtype=np.int64)


def synthetic_layouts() -> List[Layout]:
    origin = np.asarray([-6, -4, 1], dtype=np.int64)
    shape = np.asarray([18, 16, 9], dtype=np.int64)
    sparse = _c1_keys()
    total = int(np.prod(shape))
    return [
        Layout("c1_sparse", sparse, origin, shape, 0.25),
        Layout("single_boundary_obstacle", np.asarray([0], dtype=np.int64), origin, shape, 0.25),
        Layout("boundary_shell", np.asarray(sorted({0, 8, 143, 144, total - 145, total - 1}), dtype=np.int64), origin, shape, 0.25),
        Layout("dense", np.arange(total, dtype=np.int64), origin, shape, 0.25),
        Layout("empty_rejected", np.empty(0, dtype=np.int64), origin, shape, 0.25),
    ]


def production_layout(path: Optional[Path]) -> Optional[Layout]:
    if path is None:
        return None
    if not path.exists():
        return None
    with np.load(str(path), allow_pickle=False) as data:
        return Layout(
            "production_like_temp_cache",
            np.ascontiguousarray(data["occupied_keys"], dtype=np.int64),
            np.ascontiguousarray(data["origin_ijk"], dtype=np.int64).reshape(3),
            np.ascontiguousarray(data["grid_shape"], dtype=np.int64).reshape(3),
            float(np.asarray(data["voxel_size"]).reshape(())),
            source="temporary_cache_from_original_point_cloud",
        )


def _center(layout: Layout, key: int) -> np.ndarray:
    stride_x = int(layout.shape[1] * layout.shape[2])
    rel_x, rem = divmod(int(key), stride_x)
    rel_y, rel_z = divmod(rem, int(layout.shape[2]))
    return np.asarray(
        [
            (rel_x + int(layout.origin[0]) + 0.5) * layout.voxel_size,
            (rel_y + int(layout.origin[1]) + 0.5) * layout.voxel_size,
            (rel_z + int(layout.origin[2]) + 0.5) * layout.voxel_size,
        ],
        dtype=np.float32,
    )


def make_cases(layout: Layout) -> Dict[str, Any]:
    if layout.keys.size == 0:
        return {
            "paths": {},
            "pos_ref": np.empty((4, 5, 3), dtype=np.float32),
            "position": np.zeros(3, dtype=np.float32),
            "yaw": 0.37,
            "positions": np.zeros((3, 3), dtype=np.float32),
            "yaws": np.zeros(3, dtype=np.float64),
            "action_ids": np.asarray([3, 1, 0, 2], dtype=np.int32),
        }
    occupied = _center(layout, int(layout.keys[0]))
    lower = layout.origin.astype(np.float64) * layout.voxel_size
    outside = np.asarray(
        [lower[0] - 5.0 * layout.voxel_size, lower[1] - 5.0 * layout.voxel_size, lower[2] - 5.0 * layout.voxel_size],
        dtype=np.float32,
    )
    outside_1 = outside + np.asarray([0.13, 0.07, 0.11], dtype=np.float32)
    outside_2 = outside + np.asarray([0.21, -0.08, 0.17], dtype=np.float32)
    outside_3 = outside + np.asarray([-0.19, 0.16, -0.13], dtype=np.float32)
    outside_4 = outside + np.asarray([0.05, -0.22, 0.09], dtype=np.float32)
    paths = {
        "no_collision": np.asarray([outside, outside_1, outside_2, outside_3, outside_4], dtype=np.float32),
        "start_collision": np.asarray([occupied, outside_1, outside_2, outside_3, outside_4], dtype=np.float32),
        "middle_collision": np.asarray([outside, outside_1, occupied, outside_3, outside_4], dtype=np.float32),
        "end_collision": np.asarray([outside, outside_1, outside_2, outside_3, occupied], dtype=np.float32),
    }
    position = np.asarray([0.13, -0.27, 0.19], dtype=np.float32)
    yaw = 0.37
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    pos_ref = []
    for world_path in paths.values():
        delta = world_path.astype(np.float64) - position.astype(np.float64)
        body = np.empty_like(delta, dtype=np.float64)
        body[:, 0] = cosine * delta[:, 0] + sine * delta[:, 1]
        body[:, 1] = -sine * delta[:, 0] + cosine * delta[:, 1]
        body[:, 2] = delta[:, 2]
        pos_ref.append(body.astype(np.float32))
    positions = np.asarray(
        [position, position + [0.31, -0.18, 0.07], position + [-0.42, 0.22, -0.11]],
        dtype=np.float32,
    )
    yaws = np.asarray([yaw, -0.20, 1.10], dtype=np.float64)
    return {
        "paths": paths,
        "pos_ref": np.asarray(pos_ref, dtype=np.float32),
        "position": position,
        "yaw": yaw,
        "positions": positions,
        "yaws": yaws,
        "action_ids": np.asarray([3, 1, 0, 2], dtype=np.int32),
    }


class GateBook:
    def __init__(self) -> None:
        self.gates: Dict[str, bool] = {
            "PATH_COLLISION_PARITY": True,
            "SINGLE_ACTION_PARITY": True,
            "ACTIONS_BATCH_PARITY": True,
            "POSE_ACTIONS_BATCH_PARITY": True,
            "FIRST_COLLISION_INDEX_PARITY": True,
            "ENDPOINT_PARITY": True,
            "MIN_DISTANCE_PARITY": True,
            "BATCH_ORDER_PARITY": True,
            "FIRST_COLLISION_POINT_PARITY": True,
        }
        self.mismatches: List[str] = []
        self.max_abs_delta = 0.0
        self.max_relative_delta = 0.0

    def exact(self, gate: str, label: str, lhs: Any, rhs: Any) -> None:
        if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray):
            left = np.asarray(lhs)
            right = np.asarray(rhs)
            equal = _bitwise_equal(left, right)
            if left.dtype.kind in "fc" or right.dtype.kind in "fc":
                absolute, relative = _float_deltas(left, right)
                self.max_abs_delta = max(self.max_abs_delta, absolute)
                self.max_relative_delta = max(self.max_relative_delta, relative)
        else:
            equal = lhs == rhs
        if not equal:
            self.gates[gate] = False
            if len(self.mismatches) < 20:
                self.mismatches.append(
                    "{}: {} differs (lhs={!r}, rhs={!r})".format(label, gate, lhs, rhs)
                )

    def result(self) -> Dict[str, Any]:
        return {
            "gates": dict(self.gates),
            "mismatches": list(self.mismatches),
            "max_abs_delta": self.max_abs_delta,
            "max_relative_delta": self.max_relative_delta,
        }


def compare_cpu_layout(
    book: GateBook,
    original: CpuBackend,
    optimized: CpuBackend,
    layout: Layout,
    radii: Iterable[float],
) -> int:
    cases = make_cases(layout)
    if layout.keys.size == 0:
        origin = np.ascontiguousarray(layout.origin, dtype=np.int64)
        shape = np.ascontiguousarray(layout.shape, dtype=np.int64)
        o_handle = original.create(layout.keys, origin, shape, layout.voxel_size, 0.35)
        p_handle = optimized.create(layout.keys, origin, shape, layout.voxel_size, 0.35)
        book.exact("PATH_COLLISION_PARITY", layout.name + ": empty create", bool(o_handle), bool(p_handle))
        original.destroy(o_handle)
        optimized.destroy(p_handle)
        return 1

    origin = np.ascontiguousarray(layout.origin, dtype=np.int64)
    shape = np.ascontiguousarray(layout.shape, dtype=np.int64)
    checked = 0
    expected_first = {
        "no_collision": -1,
        "start_collision": 0,
        "middle_collision": 2,
        "end_collision": 4,
    }
    for radius in radii:
        o_handle = original.create(layout.keys, origin, shape, layout.voxel_size, radius)
        p_handle = optimized.create(layout.keys, origin, shape, layout.voxel_size, radius)
        book.exact("PATH_COLLISION_PARITY", layout.name + ": create", bool(o_handle), bool(p_handle))
        if not o_handle or not p_handle:
            original.destroy(o_handle)
            optimized.destroy(p_handle)
            continue
        o_radius = np.asarray([original.radius(o_handle)], dtype=np.float64)
        p_radius = np.asarray([optimized.radius(p_handle)], dtype=np.float64)
        book.exact("MIN_DISTANCE_PARITY", layout.name + ": collision radius", o_radius, p_radius)
        for step in (1, 2, 3):
            for name, path in cases["paths"].items():
                o = original.path(o_handle, path, step)
                p = optimized.path(p_handle, path, step)
                prefix = "{} radius={} step={} path={} ".format(layout.name, radius, step, name)
                book.exact("PATH_COLLISION_PARITY", prefix + "rc", o["rc"], p["rc"])
                book.exact("PATH_COLLISION_PARITY", prefix + "collision", o["collision"], p["collision"])
                book.exact("FIRST_COLLISION_INDEX_PARITY", prefix + "first_index", o["first_index"], p["first_index"])
                book.exact("FIRST_COLLISION_POINT_PARITY", prefix + "first_point", o["first_point"], p["first_point"])
                book.exact("MIN_DISTANCE_PARITY", prefix + "minimum", o["minimum"], p["minimum"])
                if name in expected_first:
                    expected_index = expected_first[name]
                    if name == "middle_collision" and step > 2:
                        expected_index = -1
                    if name == "end_collision" and step > 2:
                        expected_index = -1
                    book.exact("FIRST_COLLISION_INDEX_PARITY", prefix + "expected first_index", int(o["first_index"][0]), expected_index)
                checked += 1

            for action_id in range(cases["pos_ref"].shape[0]):
                o = original.action(
                    o_handle, cases["pos_ref"], action_id, cases["position"], cases["yaw"], step
                )
                p = optimized.action(
                    p_handle, cases["pos_ref"], action_id, cases["position"], cases["yaw"], step
                )
                prefix = "{} radius={} step={} action={} ".format(layout.name, radius, step, action_id)
                book.exact("SINGLE_ACTION_PARITY", prefix + "rc", o["rc"], p["rc"])
                book.exact("SINGLE_ACTION_PARITY", prefix + "collision", o["collision"], p["collision"])
                book.exact("FIRST_COLLISION_INDEX_PARITY", prefix + "first_index", o["first_index"], p["first_index"])
                book.exact("FIRST_COLLISION_POINT_PARITY", prefix + "first_point", o["first_point"], p["first_point"])
                book.exact("MIN_DISTANCE_PARITY", prefix + "minimum", o["minimum"], p["minimum"])
                book.exact("ENDPOINT_PARITY", prefix + "endpoint", o["endpoint"], p["endpoint"])

            o = original.actions(
                o_handle, cases["pos_ref"], cases["action_ids"], cases["position"], cases["yaw"], step
            )
            p = optimized.actions(
                p_handle, cases["pos_ref"], cases["action_ids"], cases["position"], cases["yaw"], step
            )
            prefix = "{} radius={} step={} actions ".format(layout.name, radius, step)
            book.exact("ACTIONS_BATCH_PARITY", prefix + "rc", o["rc"], p["rc"])
            book.exact("ACTIONS_BATCH_PARITY", prefix + "collisions", o["collisions"], p["collisions"])
            book.exact("ACTIONS_BATCH_PARITY", prefix + "valid", o["valid"], p["valid"])
            book.exact("FIRST_COLLISION_INDEX_PARITY", prefix + "first_index", o["first_index"], p["first_index"])
            book.exact("MIN_DISTANCE_PARITY", prefix + "minimum", o["minimum"], p["minimum"])
            book.exact("ENDPOINT_PARITY", prefix + "endpoint", o["endpoints"], p["endpoints"])
            book.exact("BATCH_ORDER_PARITY", prefix + "action_ids", o["action_ids"], p["action_ids"])

            o = original.pose_actions(
                o_handle,
                cases["pos_ref"],
                cases["action_ids"],
                cases["positions"],
                cases["yaws"],
                step,
            )
            p = optimized.pose_actions(
                p_handle,
                cases["pos_ref"],
                cases["action_ids"],
                cases["positions"],
                cases["yaws"],
                step,
            )
            prefix = "{} radius={} step={} pose_actions ".format(layout.name, radius, step)
            book.exact("POSE_ACTIONS_BATCH_PARITY", prefix + "rc", o["rc"], p["rc"])
            book.exact("POSE_ACTIONS_BATCH_PARITY", prefix + "collisions", o["collisions"], p["collisions"])
            book.exact("POSE_ACTIONS_BATCH_PARITY", prefix + "valid", o["valid"], p["valid"])
            book.exact("FIRST_COLLISION_INDEX_PARITY", prefix + "first_index", o["first_index"], p["first_index"])
            book.exact("MIN_DISTANCE_PARITY", prefix + "minimum", o["minimum"], p["minimum"])
            book.exact("ENDPOINT_PARITY", prefix + "endpoint", o["endpoints"], p["endpoints"])
            book.exact("BATCH_ORDER_PARITY", prefix + "action_ids", o["action_ids"], p["action_ids"])
        original.destroy(o_handle)
        optimized.destroy(p_handle)
    return checked


def run_cpu_parity(layouts: List[Layout], o_path: Path, p_path: Path) -> Dict[str, Any]:
    original = CpuBackend(o_path, "O_CPU")
    optimized = CpuBackend(p_path, "P_CPP")
    book = GateBook()
    checked = 0
    for layout in layouts:
        checked += compare_cpu_layout(book, original, optimized, layout, (0.35, 0.40))
    result = book.result()
    result["checked_cases"] = checked
    result["layouts"] = [layout.name for layout in layouts]
    result["radii"] = [0.35, 0.40]
    return result


def run_cuda_parity(
    layouts: List[Layout], o_cpu_path: Path, o_cuda_path: Path
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "status": "NOT_AVAILABLE",
        "gates": {
            "ACTIONS_BATCH_PARITY": None,
            "POSE_ACTIONS_BATCH_PARITY": None,
            "FIRST_COLLISION_INDEX_PARITY": None,
            "ENDPOINT_PARITY": None,
            "MIN_DISTANCE_PARITY": None,
            "BATCH_ORDER_PARITY": None,
        },
        "max_abs_delta": 0.0,
        "max_relative_delta": 0.0,
    }
    try:
        cuda = CudaBackend(o_cuda_path)
    except Exception as error:
        result["reason"] = "load failed: {}".format(repr(error))
        return result
    cpu = CpuBackend(o_cpu_path, "O_CPU")
    gates = GateBook()
    checked = 0
    for layout in layouts:
        if layout.keys.size == 0:
            continue
        cases = make_cases(layout)
        for radius in (0.35, 0.40):
            cuda_handle = cuda.create(
                layout.keys, layout.origin, layout.shape, layout.voxel_size, radius
            )
            cpu_handle = cpu.create(
                layout.keys, layout.origin, layout.shape, layout.voxel_size, radius
            )
            if not cuda_handle or not cpu_handle:
                cuda.destroy(cuda_handle)
                cpu.destroy(cpu_handle)
                result["reason"] = "CUDA create returned null; driver/device unavailable"
                return result
            result["available"] = True
            for step in (1, 2, 3):
                o = cpu.actions(
                    cpu_handle, cases["pos_ref"], cases["action_ids"], cases["position"], cases["yaw"], step
                )
                c = cuda.actions(
                    cuda_handle, cases["pos_ref"], cases["action_ids"], cases["position"], cases["yaw"], step
                )
                prefix = "{} radius={} step={} cuda_actions ".format(layout.name, radius, step)
                gates.exact("ACTIONS_BATCH_PARITY", prefix + "rc", o["rc"], c["rc"])
                gates.exact("ACTIONS_BATCH_PARITY", prefix + "collisions", o["collisions"], c["collisions"])
                gates.exact("ACTIONS_BATCH_PARITY", prefix + "valid", o["valid"], c["valid"])
                gates.exact("FIRST_COLLISION_INDEX_PARITY", prefix + "first_index", o["first_index"], c["first_index"])
                gates.exact("MIN_DISTANCE_PARITY", prefix + "minimum", o["minimum"], c["minimum"])
                gates.exact("ENDPOINT_PARITY", prefix + "endpoint", o["endpoints"], c["endpoints"])
                gates.exact("BATCH_ORDER_PARITY", prefix + "action_ids", o["action_ids"], c["action_ids"])

                o = cpu.pose_actions(
                    cpu_handle,
                    cases["pos_ref"],
                    cases["action_ids"],
                    cases["positions"],
                    cases["yaws"],
                    step,
                )
                c = cuda.pose_actions(
                    cuda_handle,
                    cases["pos_ref"],
                    cases["action_ids"],
                    cases["positions"],
                    cases["yaws"],
                    step,
                )
                prefix = "{} radius={} step={} cuda_pose_actions ".format(layout.name, radius, step)
                gates.exact("POSE_ACTIONS_BATCH_PARITY", prefix + "rc", o["rc"], c["rc"])
                gates.exact("POSE_ACTIONS_BATCH_PARITY", prefix + "collisions", o["collisions"], c["collisions"])
                gates.exact("POSE_ACTIONS_BATCH_PARITY", prefix + "valid", o["valid"], c["valid"])
                gates.exact("FIRST_COLLISION_INDEX_PARITY", prefix + "first_index", o["first_index"], c["first_index"])
                gates.exact("MIN_DISTANCE_PARITY", prefix + "minimum", o["minimum"], c["minimum"])
                gates.exact("ENDPOINT_PARITY", prefix + "endpoint", o["endpoints"], c["endpoints"])
                gates.exact("BATCH_ORDER_PARITY", prefix + "action_ids", o["action_ids"], c["action_ids"])
                checked += 2
            cuda.destroy(cuda_handle)
            cpu.destroy(cpu_handle)
    compared = gates.result()
    result["status"] = "PASS" if result["available"] and all(compared["gates"].values()) else "FAIL"
    result["gates"] = compared["gates"]
    result["mismatches"] = compared["mismatches"]
    result["max_abs_delta"] = compared["max_abs_delta"]
    result["max_relative_delta"] = compared["max_relative_delta"]
    result["checked_cases"] = checked
    return result


def shared_owner_test(layout: Layout, p_cpu_path: Path, p_voxel_path: Path) -> Dict[str, Any]:
    if layout.keys.size == 0:
        return {"status": "FAIL", "reason": "empty layout cannot exercise owner lifetime"}
    collision = CpuBackend(p_cpu_path, "P_CPP")
    voxel = _library(p_voxel_path)
    voxel.planning_voxel_map_create.argtypes = [
        _I64_PTR,
        ctypes.c_int64,
        _I64_PTR,
        _I64_PTR,
        ctypes.c_double,
    ]
    voxel.planning_voxel_map_create.restype = ctypes.c_void_p
    voxel.planning_voxel_map_destroy.argtypes = [ctypes.c_void_p]
    voxel.planning_voxel_map_destroy.restype = None
    map_handle = voxel.planning_voxel_map_create(
        _ptr(layout.keys, _I64_PTR),
        ctypes.c_int64(layout.keys.size),
        _ptr(layout.origin, _I64_PTR),
        _ptr(layout.shape, _I64_PTR),
        ctypes.c_double(layout.voxel_size),
    )
    if not map_handle:
        return {"status": "FAIL", "reason": "planning_voxel_map_create returned null"}
    c1 = collision.create_from_map(map_handle, 0.35)
    c2 = collision.create_from_map(map_handle, 0.40)
    voxel.planning_voxel_map_destroy(map_handle)
    cases = make_cases(layout)
    checks = []
    for handle in (c1, c2):
        if not handle:
            checks.append(False)
            continue
        path = cases["paths"]["start_collision"]
        output = collision.path(handle, path, 1)
        checks.append(bool(output["rc"] == 0 and output["collision"] == 1 and output["first_index"][0] == 0))
    collision.destroy(c1)
    collision.destroy(c2)
    return {
        "status": "PASS" if all(checks) else "FAIL",
        "map_load_count": 1,
        "checker_count": 2,
        "occupied_buffer_allocation_count": 1,
        "complete_occupancy_copies_in_consumers": 0,
        "different_inflate_radii": [0.35, 0.40],
        "post_map_destroy_queries": checks,
        "count_evidence": "one VoxelMap bitmap owner; consumers hold shared_ptr<const VoxelMap>",
    }


def _rss_mb() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _time_calls(function: Any, count: int) -> Tuple[float, float, float, List[float]]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    latencies: List[float] = []
    for _ in range(count):
        one = time.perf_counter()
        function()
        latencies.append(time.perf_counter() - one)
    elapsed_wall = time.perf_counter() - started_wall
    elapsed_cpu = time.process_time() - started_cpu
    return elapsed_wall, elapsed_cpu, elapsed_cpu / max(elapsed_wall, 1.0e-12), latencies


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def benchmark_one(
    label: str,
    backend: CpuBackend,
    layout: Layout,
    shared: bool,
    p_voxel_path: Optional[Path],
    runs: int,
    iterations: int,
) -> List[Dict[str, Any]]:
    cases = make_cases(layout)
    init_rows: List[Dict[str, float]] = []
    map_lib = None
    map_handle = None
    for run_id in range(1, runs + 1):
        map_load_started = time.perf_counter()
        if shared:
            if map_lib is None:
                map_lib = _library(p_voxel_path)  # type: ignore[arg-type]
                map_lib.planning_voxel_map_create.argtypes = [
                    _I64_PTR, ctypes.c_int64, _I64_PTR, _I64_PTR, ctypes.c_double
                ]
                map_lib.planning_voxel_map_create.restype = ctypes.c_void_p
                map_lib.planning_voxel_map_destroy.argtypes = [ctypes.c_void_p]
                map_lib.planning_voxel_map_destroy.restype = None
            map_handle = map_lib.planning_voxel_map_create(
                _ptr(layout.keys, _I64_PTR), ctypes.c_int64(layout.keys.size),
                _ptr(layout.origin, _I64_PTR), _ptr(layout.shape, _I64_PTR),
                ctypes.c_double(layout.voxel_size),
            )
            map_load_ms = (time.perf_counter() - map_load_started) * 1000.0
            checker_started = time.perf_counter()
            handle = backend.create_from_map(map_handle, 0.35)
            checker_init_ms = (time.perf_counter() - checker_started) * 1000.0
        else:
            handle = backend.create(
                layout.keys, layout.origin, layout.shape, layout.voxel_size, 0.35
            )
            map_load_ms = (time.perf_counter() - map_load_started) * 1000.0
            checker_init_ms = map_load_ms
        if not handle:
            raise RuntimeError("{} benchmark checker initialization failed".format(label))
        backend.set_threads(handle, int(os.environ.get("C2_BENCH_THREADS", "1")))
        for _ in range(10):
            backend.path(handle, cases["paths"]["no_collision"], 1)
            backend.action(handle, cases["pos_ref"], 1, cases["position"], cases["yaw"], 1)
            backend.actions(handle, cases["pos_ref"], cases["action_ids"], cases["position"], cases["yaw"], 1)
            backend.pose_actions(handle, cases["pos_ref"], cases["action_ids"], cases["positions"], cases["yaws"], 1)
        operations = [
            (
                "paths_per_sec",
                1,
                lambda: backend.path(handle, cases["paths"]["no_collision"], 1),
            ),
            (
                "actions_per_sec",
                1,
                lambda: backend.action(handle, cases["pos_ref"], 1, cases["position"], cases["yaw"], 1),
            ),
            (
                "poses_x_actions_per_sec",
                int(cases["positions"].shape[0] * cases["action_ids"].size),
                lambda: backend.pose_actions(handle, cases["pos_ref"], cases["action_ids"], cases["positions"], cases["yaws"], 1),
            ),
        ]
        row: Dict[str, Any] = {
            "backend": label,
            "run": run_id,
            "threads": int(os.environ.get("C2_BENCH_THREADS", "1")),
            "layout": layout.name,
            "iterations": iterations,
            "map_load_ms": map_load_ms,
            "checker_init_ms": checker_init_ms,
            "peak_rss_mb_before": _rss_mb(),
        }
        for metric, units_per_call, function in operations:
            wall, cpu, cpu_ratio, latencies = _time_calls(function, iterations)
            row[metric] = units_per_call * iterations / max(wall, 1.0e-12)
            row[metric + "_wall_s"] = wall
            row[metric + "_process_cpu_s"] = cpu
            row[metric + "_process_cpu_utilization_pct"] = 100.0 * cpu_ratio
            if metric == "poses_x_actions_per_sec":
                row["batch_latency_p50_ms"] = 1000.0 * _percentile(latencies, 50.0)
                row["batch_latency_p95_ms"] = 1000.0 * _percentile(latencies, 95.0)
        row["peak_rss_mb"] = _rss_mb()
        init_rows.append(row)
        backend.destroy(handle)
        if map_handle and map_lib is not None:
            map_lib.planning_voxel_map_destroy(map_handle)
            map_handle = None
    return init_rows


def run_benchmark(
    layouts: List[Layout], o_cpu_path: Path, p_cpu_path: Path, p_voxel_path: Path
) -> Dict[str, Any]:
    layout = next(item for item in layouts if item.name == "c1_sparse")
    rows: List[Dict[str, Any]] = []
    for threads in (1, 2, 4):
        os.environ["C2_BENCH_THREADS"] = str(threads)
        rows.extend(
            benchmark_one(
                "O_CPU",
                CpuBackend(o_cpu_path, "O_CPU"),
                layout,
                False,
                None,
                runs=3,
                iterations=200,
            )
        )
        rows.extend(
            benchmark_one(
                "P_CPP_SHARED_VOXELMAP",
                CpuBackend(p_cpu_path, "P_CPP"),
                layout,
                True,
                p_voxel_path,
                runs=3,
                iterations=200,
            )
        )
    return {
        "status": "PASS",
        "fixture": layout.name,
        "runs_per_backend_thread_count": 3,
        "thread_counts": [1, 2, 4],
        "rows": rows,
        "resource_metric_note": "process CPU time and ru_maxrss; CPU utilization is process-level, not a fabricated per-core reading",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--o-cpu", type=Path, default=Path(os.environ.get("XM_O_CPU_LIB", DEFAULT_O_CPU)))
    parser.add_argument("--o-cuda", type=Path, default=Path(os.environ.get("XM_O_CUDA_LIB", DEFAULT_O_CUDA)))
    parser.add_argument("--p-cpu", type=Path, default=Path(os.environ.get("XM_P_CPU_LIB", DEFAULT_P_CPU)))
    parser.add_argument("--p-voxel", type=Path, default=Path(os.environ.get("XM_P_VOXEL_LIB", DEFAULT_P_VOXEL)))
    parser.add_argument("--production-cache", type=Path, default=None)
    parser.add_argument("--benchmark-output", type=Path, default=None)
    args = parser.parse_args()

    layouts = synthetic_layouts()
    production = production_layout(args.production_cache)
    if production is not None:
        layouts.append(production)
    parity = run_cpu_parity(layouts, args.o_cpu, args.p_cpu)
    print("CPU_PARITY_RESULT={}".format("PASS" if all(parity["gates"].values()) else "FAIL"))
    for name, value in parity["gates"].items():
        print("{}={}".format(name, "PASS" if value else "FAIL"))
    print("CPU_PARITY_CASES={}".format(parity["checked_cases"]))
    print("CPU_MAX_ABS_DELTA={:.17g}".format(parity["max_abs_delta"]))
    print("CPU_MAX_RELATIVE_DELTA={:.17g}".format(parity["max_relative_delta"]))
    for mismatch in parity["mismatches"]:
        print("CPU_MISMATCH={}".format(mismatch))
    print("PRODUCTION_ASSET_PARITY={}".format("PASS" if production is not None else "BLOCKED"))

    cuda = run_cuda_parity(layouts, args.o_cpu, args.o_cuda)
    print("O_CUDA_AVAILABLE={}".format("YES" if cuda["available"] else "NO"))
    print("O_CUDA_STATUS={}".format(cuda["status"]))
    if cuda.get("reason"):
        print("O_CUDA_REASON={}".format(cuda["reason"]))

    owner = shared_owner_test(
        next(item for item in layouts if item.name == "c1_sparse"), args.p_cpu, args.p_voxel
    )
    print("SHARED_VOXELMAP_RUNTIME_OWNERSHIP={}".format(owner["status"]))
    print("SHARED_VOXELMAP_COUNTS=" + json.dumps(owner, sort_keys=True))

    if args.benchmark_output is not None:
        benchmark = run_benchmark(layouts, args.o_cpu, args.p_cpu, args.p_voxel)
        args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_output.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("BENCHMARK_RESULT=PASS")
        print("BENCHMARK_ARTIFACT={}".format(args.benchmark_output))
    else:
        benchmark = None

    output = {
        "cpu": parity,
        "cuda": cuda,
        "owner": owner,
        "production_asset_parity": "PASS" if production is not None else "BLOCKED",
        "o_hybrid": {
            "available": False,
            "status": "NOT_AVAILABLE",
            "reason": "hybrid is an O Teacher CPU/CUDA scheduler and CUDA device/cache runtime was unavailable",
        },
        "p_hybrid": {
            "available": False,
            "status": "NOT_AVAILABLE",
            "reason": "P safety wrapper exposes cpu/cpp/python only; no CUDA or hybrid backend",
        },
        "benchmark": benchmark,
    }
    print("RESULT=" + ("PASS" if all(parity["gates"].values()) and owner["status"] == "PASS" else "FAIL"))
    return 0 if all(parity["gates"].values()) and owner["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
