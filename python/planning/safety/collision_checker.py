#!/usr/bin/env python3
"""Voxel collision checking for motion primitives.

The C++17/OpenMP backend is the only production path. The NumPy
implementation remains available only through an explicit test/debug
reference selection; native initialization and runtime failures never fall
back silently.
"""

from __future__ import annotations
import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from planning.primitives.library import resolve_package_path, rotation_z
from planning.common.config import parse_bool
from planning.native.geometry import _prepare_native_array
from planning.native.collision_backend import _FastCollisionBackend
from planning.native.loader import resolve_collision_library
from planning.data.voxel_cache import VoxelCacheConfig, build_voxel_cache


FORMAL_COLLISION_BACKEND = "cpp_cpu"
PYTHON_REFERENCE_COLLISION_BACKEND = "python_reference"
COLLISION_BACKEND_CONTRACT_ID = "cpp_bitmap_openmp"
COLLISION_REFERENCE_CONTRACT_ID = "numpy_spherical_offsets_reference"
COLLISION_SOURCE_ID = "planning_collision_checker_cpp17_openmp"
COLLISION_SOURCE_FILES = (
    "include/planning/geometry/voxel_map.hpp",
    "src/geometry/voxel_map.cpp",
    "include/planning/geometry/collision_checker.hpp",
    "src/geometry/collision_checker.cpp",
    "python/planning/safety/collision_checker.py",
)

_FORMAL_COLLISION_ALIASES = frozenset({"auto", "cpu", "cpp", "cpp_cpu"})
_UNSUPPORTED_COLLISION_BACKENDS = frozenset({"cuda", "hybrid"})


def resolve_collision_backend(requested: Optional[str] = None) -> str:
    """Resolve the collision backend without permitting implicit fallback.

    ``cpu``, ``cpp`` and ``auto`` are compatibility spellings for the single
    formal backend. ``python`` requires an explicit reference-mode opt-in so
    production callers cannot reach the NumPy implementation accidentally.
    CUDA and hybrid are rejected instead of being silently redirected.
    """

    raw = (
        os.environ.get("PLANNING_COLLISION_BACKEND", FORMAL_COLLISION_BACKEND)
        if requested is None
        else requested
    )
    backend = str(raw).strip().lower()
    if not backend:
        backend = FORMAL_COLLISION_BACKEND
    if backend in _UNSUPPORTED_COLLISION_BACKENDS:
        raise ValueError("UNSUPPORTED_COLLISION_BACKEND: {}".format(backend))
    if backend in _FORMAL_COLLISION_ALIASES:
        return FORMAL_COLLISION_BACKEND
    if backend == "python":
        reference_enabled = parse_bool(
            os.environ.get("PLANNING_COLLISION_REFERENCE", "false")
        )
        if not reference_enabled:
            raise ValueError(
                "PYTHON_REFERENCE_BACKEND_REQUIRES_EXPLICIT_DEBUG: "
                "set PLANNING_COLLISION_REFERENCE=1"
            )
        return PYTHON_REFERENCE_COLLISION_BACKEND
    raise ValueError("UNSUPPORTED_COLLISION_BACKEND: {}".format(backend))


def collision_backend_cli_type(value: str) -> str:
    """Argparse type that emits the production backend rejection contract."""

    raw = str(value).strip().lower()
    if raw == "python":
        raise argparse.ArgumentTypeError(
            "UNSUPPORTED_COLLISION_BACKEND: python is test/debug-only"
        )
    try:
        resolve_collision_backend(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))
    return raw


def _find_collision_library() -> Optional[Path]:
    """Compatibility seam; discovery is implemented by native.loader."""

    enabled = parse_bool(os.environ.get("PLANNING_COLLISION_CHECK", "1"))
    if not enabled:
        return None
    return resolve_collision_library()


VoxelMapConfig = VoxelCacheConfig

def unity_to_ros(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    out = np.empty_like(pts, dtype=np.float32)
    out[:, 0] = pts[:, 2]
    out[:, 1] = -pts[:, 0]
    out[:, 2] = pts[:, 1]
    return out

class VoxelCollisionChecker:
    def __init__(
        self,
        occupied_keys: np.ndarray,
        origin_ijk: np.ndarray,
        grid_shape: np.ndarray,
        voxel_size: float,
        inflate_radius: float,
        _native_map_handle=None,
        _native_geometry_context=None,
    ):
        self._pid = os.getpid()
        self.native_geometry_context = _native_geometry_context
        self._closed = False
        if self.native_geometry_context is not None:
            self.native_geometry_context._assert_open()
        self.occupied_keys = np.asarray(occupied_keys, dtype=np.int64)
        self.origin_ijk = np.asarray(origin_ijk, dtype=np.int64).reshape(3)
        self.grid_shape = np.asarray(grid_shape, dtype=np.int64).reshape(3)
        self.voxel_size = float(voxel_size)
        self.inflate_radius = float(inflate_radius)

        if self.occupied_keys.ndim != 1:
            raise ValueError("occupied_keys must be 1D")
        if self.occupied_keys.size == 0:
            raise ValueError("empty occupancy map")
        if np.any(self.grid_shape <= 0):
            raise ValueError("invalid grid_shape {}".format(self.grid_shape))

        self.collision_radius = self.inflate_radius + 0.5 * math.sqrt(3.0) * self.voxel_size
        self._neighbor_offsets = self._make_neighbor_offsets(self.collision_radius)
        self._fast = None
        self._fast_error = None
        self._backend_name = resolve_collision_backend()
        self._python_reference_enabled = (
            self._backend_name == PYTHON_REFERENCE_COLLISION_BACKEND
        )
        if self._python_reference_enabled:
            return
        try:
            self._fast = _FastCollisionBackend(
                self.occupied_keys,
                self.origin_ijk,
                self.grid_shape,
                self.voxel_size,
                self.inflate_radius,
                native_map_handle=_native_map_handle,
                native_geometry_context=self.native_geometry_context,
                library_path=_find_collision_library(),
            )
        except Exception as error:
            self._fast_error = repr(error)
            raise RuntimeError(
                "C++ collision backend required but initialization failed: {}".format(
                    self._fast_error
                )
            ) from error

    @property
    def fast_enabled(self) -> bool:
        return self._fast is not None

    @property
    def native_map_id(self) -> Optional[int]:
        return None if self._fast is None else self._fast.native_map_id

    @property
    def native_checker_id(self) -> Optional[int]:
        if self._fast is None:
            return None
        return getattr(self._fast, "native_checker_id", None)

    def _assert_open(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError(
                "NATIVE_CONTEXT_FORK_REUSE_FORBIDDEN: collision checker"
            )
        if self._closed:
            raise RuntimeError("collision checker is closed")
        if self.native_geometry_context is not None:
            try:
                self.native_geometry_context._assert_open()
            except Exception as error:
                raise RuntimeError(str(error)) from error

    def close(self) -> None:
        if self._closed:
            return
        if self._fast is not None:
            self._fast.close()
        self._closed = True

    @property
    def collision_backend(self) -> str:
        # Keep the existing public value stable; the canonical resolved ID is
        # available through ``resolve_collision_backend`` and provenance
        # metadata without exposing implementation details to callers.
        return "cpp" if self._backend_name == FORMAL_COLLISION_BACKEND else "python"

    @property
    def collision_backend_contract_id(self) -> str:
        return (
            COLLISION_BACKEND_CONTRACT_ID
            if self._backend_name == FORMAL_COLLISION_BACKEND
            else COLLISION_REFERENCE_CONTRACT_ID
        )

    @property
    def collision_source_id(self) -> str:
        return COLLISION_SOURCE_ID

    @property
    def collision_source_files(self) -> Tuple[str, ...]:
        return COLLISION_SOURCE_FILES

    def _fail_closed_native_operation(self, operation: str, error: Exception) -> None:
        self._fast_error = repr(error)
        raise RuntimeError(
            "C++ collision backend failed during {}; Python reference "
            "fallback is disabled: {}".format(operation, self._fast_error)
        ) from error

    def _require_reference_or_fail_closed(self, operation: str) -> None:
        if not self._python_reference_enabled:
            detail = self._fast_error or "native backend is unavailable"
            raise RuntimeError(
                "C++ collision backend unavailable during {}; Python "
                "reference fallback is disabled: {}".format(operation, detail)
            )

    @classmethod
    def from_config(cls, cfg: VoxelMapConfig, rebuild_cache: bool = False):
        map_path = resolve_package_path(cfg.map_bin)
        cache_path = resolve_package_path(cfg.cache_npz)

        if cache_path.exists() and not rebuild_cache:
            return cls.load_cache(
                cache_path,
                voxel_size=cfg.voxel_size,
                inflate_radius=cfg.inflate_radius,
            )

        return cls.build_from_bin(cfg, map_path, cache_path)

    @classmethod
    def load_cache(
        cls,
        cache_path: Path,
        inflate_radius: float,
        voxel_size: Optional[float] = None,
    ):
        with np.load(str(cache_path), allow_pickle=False) as data:
            cached_voxel_size = float(data["voxel_size"])
            effective_voxel_size = cached_voxel_size
            if voxel_size is not None:
                effective_voxel_size = float(voxel_size)
                if not math.isclose(cached_voxel_size, effective_voxel_size, rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError(
                        "voxel cache size mismatch: cache={} requested={}; rebuild the cache".format(
                            cached_voxel_size, effective_voxel_size
                        )
                    )
            occupied_keys = data["occupied_keys"].copy()
            origin_ijk = data["origin_ijk"].copy()
            grid_shape = data["grid_shape"].copy()
        checker = cls(
            occupied_keys=occupied_keys,
            origin_ijk=origin_ijk,
            grid_shape=grid_shape,
            # Use the requested decimal value so floating-point serialization
            # cannot shift floor(point / voxel_size) at voxel boundaries.
            voxel_size=effective_voxel_size,
            inflate_radius=float(inflate_radius),
        )
        print("VOXEL_CACHE_LOAD=PASS")
        print("  cache:", cache_path)
        print("  occupied_voxels:", checker.occupied_keys.size)
        print("  voxel_size:", checker.voxel_size)
        if voxel_size is not None and cached_voxel_size != effective_voxel_size:
            print("  cache_voxel_size:", cached_voxel_size)
        print("  inflate_radius:", checker.inflate_radius)
        print("  collision_radius:", checker.collision_radius)
        print("  grid_shape:", checker.grid_shape.tolist())
        print("  origin_ijk:", checker.origin_ijk.tolist())
        print("  backend:", checker.collision_backend)
        print("  backend_contract_id:", checker.collision_backend_contract_id)
        if not checker.fast_enabled and checker._fast_error:
            print("  backend_reason:", checker._fast_error)
        return checker

    @classmethod
    def build_from_bin(cls, cfg: VoxelMapConfig, map_path: Path, cache_path: Path):
        build_result = build_voxel_cache(
            VoxelCacheConfig(
                map_bin=Path(map_path),
                file_frame=str(cfg.file_frame),
                voxel_size=float(cfg.voxel_size),
                inflate_radius=float(cfg.inflate_radius),
                cache_npz=Path(cache_path),
                data_offset_bytes=int(cfg.data_offset_bytes),
                chunk_points=int(cfg.chunk_points),
                max_points=int(cfg.max_points),
                z_min=float(cfg.z_min),
                z_max=float(cfg.z_max),
            )
        )
        with np.load(str(build_result.cache_path), allow_pickle=False) as data:
            all_keys = data["occupied_keys"].copy()
            min_ijk = data["origin_ijk"].copy()
            grid_shape = data["grid_shape"].copy()
        checker = cls(
            occupied_keys=all_keys,
            origin_ijk=min_ijk,
            grid_shape=grid_shape,
            voxel_size=float(cfg.voxel_size),
            inflate_radius=float(cfg.inflate_radius),
        )
        print("  backend:", checker.collision_backend)
        print("  backend_contract_id:", checker.collision_backend_contract_id)
        if not checker.fast_enabled and checker._fast_error:
            print("  backend_reason:", checker._fast_error)
        return checker

    @staticmethod
    def _resolve_data_offset(file_size: int, requested_offset: int) -> int:
        """Resolve byte offset before float32 xyz payload.

        Supported:
          requested_offset >= 0: use exactly that offset.
          requested_offset == -1: auto-detect 0 or 4 byte prefix.

        Reason:
          Some generated forest_point_cloud.bin files start with a 4-byte count
          or version field, making file_size % 12 == 4. The xyz payload still
          consists of float32 triples after that prefix.
        """
        if requested_offset >= 0:
            return int(requested_offset)

        if file_size % 12 == 0:
            return 0
        if file_size >= 4 and (file_size - 4) % 12 == 0:
            return 4

        # General fallback for rare small metadata prefixes.
        for off in range(0, 65, 4):
            if file_size >= off and (file_size - off) % 12 == 0:
                return off

        raise ValueError(
            "cannot auto-detect xyz payload offset: file_size={} file_size%12={}".format(
                file_size, file_size % 12
            )
        )

    @staticmethod
    def _convert_frame(points: np.ndarray, file_frame: str) -> np.ndarray:
        if file_frame == "unity":
            return unity_to_ros(points)
        if file_frame == "ros":
            return np.asarray(points, dtype=np.float32)
        raise ValueError("file_frame must be unity or ros, got {}".format(file_frame))

    @staticmethod
    def _pack_keys_static(ijk: np.ndarray, origin_ijk: np.ndarray, grid_shape: np.ndarray) -> np.ndarray:
        rel = ijk.astype(np.int64) - origin_ijk.astype(np.int64)
        sx = np.int64(grid_shape[1] * grid_shape[2])
        sy = np.int64(grid_shape[2])
        return rel[:, 0] * sx + rel[:, 1] * sy + rel[:, 2]

    def _pack_keys(self, ijk: np.ndarray) -> np.ndarray:
        return self._pack_keys_static(ijk, self.origin_ijk, self.grid_shape)

    def _voxel_centers(self, ijk: np.ndarray) -> np.ndarray:
        return (ijk.astype(np.float32) + 0.5) * self.voxel_size

    def _make_neighbor_offsets(self, radius: float) -> np.ndarray:
        r_vox = int(math.ceil(float(radius) / self.voxel_size))
        offsets = []
        for ix in range(-r_vox, r_vox + 1):
            for iy in range(-r_vox, r_vox + 1):
                for iz in range(-r_vox, r_vox + 1):
                    minimum_possible_distance = self.voxel_size * math.sqrt(
                        max(0.0, abs(ix) - 0.5) ** 2
                        + max(0.0, abs(iy) - 0.5) ** 2
                        + max(0.0, abs(iz) - 0.5) ** 2
                    )
                    if minimum_possible_distance > float(radius):
                        continue
                    offsets.append((ix, iy, iz))
        return np.asarray(offsets, dtype=np.int64)

    def check_path(self, path_map: np.ndarray, check_step: int = 1) -> Dict:
        self._assert_open()
        path = _prepare_native_array(
            path_map,
            np.dtype(np.float32),
            (-1, 3),
            "path_map",
            self.native_geometry_context,
        )

        if self._fast is not None:
            try:
                return self._fast.check_path(path, check_step=check_step)
            except Exception as error:
                self._fail_closed_native_operation("check_path", error)

        self._require_reference_or_fail_closed("check_path")

        step = max(1, int(check_step))
        path = path[::step]

        global_min = float("inf")
        first_collision_index = -1
        first_collision_point = None
        collision = False

        for i, p in enumerate(path):
            q_ijk = np.floor(p / self.voxel_size).astype(np.int64).reshape(1, 3)
            cand_ijk = q_ijk + self._neighbor_offsets
            cand_keys = self._pack_keys(cand_ijk)

            idx = np.searchsorted(self.occupied_keys, cand_keys)
            inside = idx < self.occupied_keys.size
            if not np.any(inside):
                continue

            cand_keys_inside = cand_keys[inside]
            idx_inside = idx[inside]
            hit_mask = self.occupied_keys[idx_inside] == cand_keys_inside
            if not np.any(hit_mask):
                continue

            hit_ijk = cand_ijk[inside][hit_mask]
            centers = self._voxel_centers(hit_ijk)
            dists = np.linalg.norm(centers - p.reshape(1, 3), axis=1)
            local_min = float(np.min(dists))
            if local_min < global_min:
                global_min = local_min

            if local_min <= self.collision_radius:
                collision = True
                first_collision_index = int(i * step)
                first_collision_point = p.copy()
                break

        if not math.isfinite(global_min):
            global_min = float("inf")

        return {
            "collision": bool(collision),
            "valid": bool(not collision),
            "min_distance_voxel_center_m": float(global_min),
            "collision_radius_m": float(self.collision_radius),
            "first_collision_index": int(first_collision_index),
            "first_collision_point": None if first_collision_point is None else first_collision_point.astype(np.float32),
        }

    def check_action(
        self,
        mpl,
        action_id: int,
        position: Sequence[float],
        yaw: float,
        check_step: int = 1,
        return_path_map: bool = True,
    ) -> Dict:
        self._assert_open()
        pos = _prepare_native_array(
            position,
            np.dtype(np.float32),
            (3,),
            "position",
            self.native_geometry_context,
        )

        if self._fast is not None:
            try:
                result = self._fast.check_action(mpl, int(action_id), pos, float(yaw), check_step=check_step)
                result["action_id"] = int(action_id)
                if return_path_map:
                    path_body = mpl.reference_path(int(action_id))
                    path_map = pos.reshape(1, 3) + path_body.dot(rotation_z(float(yaw)).T)
                    result["path_map"] = path_map
                    result["endpoint_map"] = path_map[-1]
                return result
            except Exception as error:
                self._fail_closed_native_operation("check_action", error)

        self._require_reference_or_fail_closed("check_action")

        path_body = mpl.reference_path(int(action_id))
        path_map = pos.reshape(1, 3) + path_body.dot(rotation_z(float(yaw)).T)
        result = self.check_path(path_map, check_step=check_step)
        result["action_id"] = int(action_id)
        if return_path_map:
            result["path_map"] = path_map
        result["endpoint_map"] = path_map[-1]
        return result

    def check_actions_array(
        self,
        motion_primitives,
        action_ids: Sequence[int],
        position: Sequence[float],
        yaw: float,
        check_step: int = 1,
    ) -> Dict[str, np.ndarray]:
        self._assert_open()
        ids = _prepare_native_array(
            action_ids,
            np.dtype(np.int32),
            (-1,),
            "action_ids",
            self.native_geometry_context,
        )
        if ids.size == 0:
            return {
                "action_ids": ids,
                "valid": np.empty((0,), dtype=np.bool_),
                "minimum_distances": np.empty((0,), dtype=np.float64),
                "first_collision_indices": np.empty((0,), dtype=np.int32),
                "endpoints": np.empty((0, 3), dtype=np.float32),
            }
        position_array = _prepare_native_array(
            position,
            np.dtype(np.float32),
            (3,),
            "position",
            self.native_geometry_context,
        )
        if self._fast is not None:
            try:
                arrays = self._fast.check_actions_array(
                    motion_primitives, ids, position_array, float(yaw), check_step
                )
                return {
                    "action_ids": arrays["action_ids"],
                    "valid": ~arrays["collisions"],
                    "minimum_distances": arrays["minimum_distances"],
                    "first_collision_indices": arrays["first_collision_indices"],
                    "endpoints": arrays["endpoints"],
                }
            except Exception as error:
                self._fail_closed_native_operation("check_actions_array", error)

        self._require_reference_or_fail_closed("check_actions_array")
        rows = [
            self.check_action(
                motion_primitives,
                int(action_id),
                position_array,
                yaw,
                check_step=check_step,
                return_path_map=False,
            )
            for action_id in ids
        ]
        return {
            "action_ids": ids,
            "valid": np.asarray([bool(row["valid"]) for row in rows], dtype=np.bool_),
            "minimum_distances": np.asarray(
                [float(row["min_distance_voxel_center_m"]) for row in rows],
                dtype=np.float64,
            ),
            "first_collision_indices": np.asarray(
                [int(row["first_collision_index"]) for row in rows], dtype=np.int32
            ),
            "endpoints": np.asarray([row["endpoint_map"] for row in rows], dtype=np.float32),
        }

    def check_pose_actions_array(
        self,
        motion_primitives,
        positions: np.ndarray,
        yaws: np.ndarray,
        action_ids: Sequence[int],
        check_step: int = 1,
    ) -> Dict[str, np.ndarray]:
        self._assert_open()
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
            raise ValueError("positions and yaws must have matching pose counts")
        ids = _prepare_native_array(
            action_ids,
            np.dtype(np.int32),
            (-1,),
            "action_ids",
            self.native_geometry_context,
        )
        if self._fast is not None:
            try:
                arrays = self._fast.check_pose_actions_array(
                    motion_primitives,
                    positions_array,
                    yaws_array,
                    ids,
                    check_step,
                )
                return {
                    "action_ids": arrays["action_ids"],
                    "valid": ~arrays["collisions"],
                    "minimum_distances": arrays["minimum_distances"],
                    "first_collision_indices": arrays["first_collision_indices"],
                    "endpoints": arrays["endpoints"],
                }
            except Exception as error:
                self._fail_closed_native_operation("check_pose_actions_array", error)

        self._require_reference_or_fail_closed("check_pose_actions_array")
        rows = [
            self.check_actions_array(
                motion_primitives, ids, position, yaw, check_step=check_step
            )
            for position, yaw in zip(positions_array, yaws_array)
        ]
        return {
            "action_ids": ids,
            "valid": np.stack([row["valid"] for row in rows]),
            "minimum_distances": np.stack(
                [row["minimum_distances"] for row in rows]
            ),
            "first_collision_indices": np.stack(
                [row["first_collision_indices"] for row in rows]
            ),
            "endpoints": np.stack([row["endpoints"] for row in rows]),
        }


    def check_all_actions_array(
        self,
        motion_primitives,
        position: Sequence[float],
        yaw: float,
        check_step: int = 1,
    ) -> Dict[str, np.ndarray]:
        return self.check_actions_array(
            motion_primitives,
            np.arange(motion_primitives.num_actions, dtype=np.int32),
            position,
            yaw,
            check_step,
        )


    def check_actions_batch(
        self,
        motion_primitives,
        action_ids: Sequence[int],
        position: Sequence[float],
        yaw: float,
        check_step: int = 1,
        return_path_map: bool = True,
    ) -> List[Dict]:
        arrays = self.check_actions_array(
            motion_primitives, action_ids, position, yaw, check_step
        )
        position_array = np.asarray(position, dtype=np.float32).reshape(3)
        rotation = rotation_z(float(yaw)).T
        output = []
        for index, action_id in enumerate(arrays["action_ids"]):
            row = {
                "action_id": int(action_id),
                "collision": not bool(arrays["valid"][index]),
                "valid": bool(arrays["valid"][index]),
                "min_distance_voxel_center_m": float(
                    arrays["minimum_distances"][index]
                ),
                "collision_radius_m": float(self.collision_radius),
                "first_collision_index": int(
                    arrays["first_collision_indices"][index]
                ),
                "first_collision_point": None,
                "endpoint_map": arrays["endpoints"][index].copy(),
            }
            if return_path_map:
                path_map = position_array.reshape(1, 3) + motion_primitives.pos_ref[
                    int(action_id)
                ].dot(rotation)
                row["path_map"] = path_map
            output.append(row)
        return output

    def check_all_actions(
        self,
        motion_primitives,
        position: Sequence[float],
        yaw: float,
        check_step: int = 1,
        return_path_map: bool = True,
    ) -> List[Dict]:
        return self.check_actions_batch(
            motion_primitives,
            range(motion_primitives.num_actions),
            position,
            yaw,
            check_step=check_step,
            return_path_map=return_path_map,
        )


def _parse_action_ids(text: str, num_actions: int) -> List[int]:
    if not text or text.lower() == "all":
        return list(range(num_actions))

    action_ids = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = [int(value) for value in part.split("-", 1)]
            if end < start:
                start, end = end, start
            action_ids.extend(range(start, end + 1))
        elif part:
            action_ids.append(int(part))
    return sorted(set(action_id for action_id in action_ids if 0 <= action_id < num_actions))


def _write_collision_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Run the MPL collision-check CLI previously exposed as a test script."""
    from planning.primitives.library import MotionPrimitiveLibrary

    parser = argparse.ArgumentParser(description="Check MPL actions against the forest voxel map.")
    parser.add_argument("--map-bin", default="data/map_data/forest_point_cloud.bin")
    parser.add_argument("--file-frame", default="unity", choices=["unity", "ros"])
    parser.add_argument("--cache", default="data/map_data/forest_voxels_10cm.npz")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--data-offset-bytes", type=int, default=-1)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--inflate-radius", type=float, default=0.35)
    parser.add_argument("--check-step", type=int, default=1)
    parser.add_argument("--chunk-points", type=int, default=2_000_000)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--z-min", type=float, default=-1e9)
    parser.add_argument("--z-max", type=float, default=1e9)
    parser.add_argument("--pose", type=float, nargs=4, default=[0.0, 0.0, 2.0, 0.0], metavar=("X", "Y", "Z", "YAW_DEG"))
    parser.add_argument("--actions", default="all")
    parser.add_argument("--csv", default="data/map_data/motion_primitive_collision_origin_10cm.csv")
    args = parser.parse_args()

    cfg = VoxelMapConfig(
        map_bin=args.map_bin,
        file_frame=args.file_frame,
        voxel_size=args.voxel_size,
        inflate_radius=args.inflate_radius,
        cache_npz=args.cache,
        data_offset_bytes=args.data_offset_bytes,
        chunk_points=args.chunk_points,
        max_points=args.max_points,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    mpl = MotionPrimitiveLibrary()
    checker = VoxelCollisionChecker.from_config(cfg, rebuild_cache=args.rebuild_cache)
    position = np.asarray(args.pose[:3], dtype=np.float32)
    yaw = math.radians(float(args.pose[3]))
    action_ids = _parse_action_ids(args.actions, mpl.num_actions)

    print("MOTION_PRIMITIVE_COLLISION_CHECK_START")
    print("  position:", position.tolist())
    print("  yaw_deg:", args.pose[3])
    print("  actions:", len(action_ids))
    print("  inflate_radius:", args.inflate_radius)
    print("  collision_radius:", checker.collision_radius)
    print("  check_step:", args.check_step)

    rows = []
    for action_id in action_ids:
        result = checker.check_action(mpl, action_id, position, yaw, check_step=args.check_step)
        meta = mpl.action_metadata(action_id)
        endpoint = result["endpoint_map"]
        collision_point = result["first_collision_point"]
        row = {
            "action_id": int(action_id),
            "horizontal_index": int(meta["horizontal_index"]),
            "vertical_index": int(meta["vertical_index"]),
            "lateral_endpoint_m": float(meta["lateral_endpoint_m"]),
            "vertical_endpoint_m": float(meta["vertical_endpoint_m"]),
            "heading_deg": float(meta["terminal_heading_deg"]),
            "valid": bool(result["valid"]),
            "collision": bool(result["collision"]),
            "min_distance_voxel_center_m": float(result["min_distance_voxel_center_m"]),
            "collision_radius_m": float(result["collision_radius_m"]),
            "first_collision_index": int(result["first_collision_index"]),
            "endpoint_x": float(endpoint[0]),
            "endpoint_y": float(endpoint[1]),
            "endpoint_z": float(endpoint[2]),
            "collision_x": float("nan") if collision_point is None else float(collision_point[0]),
            "collision_y": float("nan") if collision_point is None else float(collision_point[1]),
            "collision_z": float("nan") if collision_point is None else float(collision_point[2]),
        }
        rows.append(row)

        if action_id in [0, 6, 52, 98, 104] or len(action_ids) <= 20:
            status = "VALID" if row["valid"] else "INVALID"
            distance = row["min_distance_voxel_center_m"]
            distance_text = "inf" if not math.isfinite(distance) else "{:.3f}".format(distance)
            print(
                "  id={:03d} {:7s} h={:02d} v={} y={:+.2f} z={:+.2f} min_d={} first_idx={}".format(
                    action_id, status, row["horizontal_index"], row["vertical_index"], row["lateral_endpoint_m"],
                    row["vertical_endpoint_m"], distance_text, row["first_collision_index"],
                )
            )

    csv_path = resolve_package_path(args.csv)
    _write_collision_csv(csv_path, rows)
    valid_count = sum(1 for row in rows if row["valid"])
    finite_distances = np.asarray(
        [row["min_distance_voxel_center_m"] for row in rows if math.isfinite(row["min_distance_voxel_center_m"])],
        dtype=np.float64,
    )
    print("\nMOTION_PRIMITIVE_COLLISION_CHECK_SUMMARY")
    print("  tested:", len(rows))
    print("  valid_count:", valid_count)
    print("  invalid_count:", len(rows) - valid_count)
    print("  valid_rate: {:.1f}%".format(100.0 * valid_count / max(1, len(rows))))
    if finite_distances.size:
        print("  min_distance_voxel_center_m: min={:.3f} mean={:.3f} max={:.3f}".format(
            float(finite_distances.min()), float(finite_distances.mean()), float(finite_distances.max())
        ))
    else:
        print("  min_distance_voxel_center_m: all inf")
    print("  csv:", csv_path)
    print("RESULT=PASS" if valid_count > 0 else "RESULT=FAIL")
    return 0 if valid_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
