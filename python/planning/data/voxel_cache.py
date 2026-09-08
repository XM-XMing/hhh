"""Canonical, deterministic voxel-cache construction for Planning data.

The cache builder owns only the offline point-cloud-to-occupancy conversion.
Runtime collision and route consumers load the resulting immutable NPZ through
``NativeGeometryContext``.  The conversion below intentionally preserves the
historical two-pass memmap algorithm used by ``safety.collision_checker``;
moving the owner does not change voxel coordinates, key packing, ordering, or
the NPZ business fields.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from planning.common import file_sha256, write_json_atomic, write_npz_atomic


@dataclass(frozen=True)
class VoxelCacheConfig:
    """Inputs for the formal forest voxel-cache build."""

    map_bin: Path = Path("data/map_data/forest_point_cloud.bin")
    file_frame: str = "unity"
    voxel_size: float = 0.10
    inflate_radius: float = 0.35
    cache_npz: Path = Path("data/map_data/forest_voxels_10cm.npz")
    data_offset_bytes: int = -1
    chunk_points: int = 2_000_000
    max_points: int = 0
    z_min: float = -1e9
    z_max: float = 1e9


@dataclass(frozen=True)
class VoxelCacheBuildResult:
    cache_path: Path
    metadata: Dict[str, object]


def _resolve_data_offset(file_size: int, requested_offset: int) -> int:
    """Resolve the byte offset before float32 XYZ triples."""

    if requested_offset >= 0:
        return int(requested_offset)
    if file_size % 12 == 0:
        return 0
    if file_size >= 4 and (file_size - 4) % 12 == 0:
        return 4
    for offset in range(0, 65, 4):
        if file_size >= offset and (file_size - offset) % 12 == 0:
            return offset
    raise ValueError(
        "cannot auto-detect xyz payload offset: file_size={} file_size%12={}".format(
            file_size, file_size % 12
        )
    )


def _convert_frame(points: np.ndarray, file_frame: str) -> np.ndarray:
    if file_frame == "unity":
        pts = np.asarray(points, dtype=np.float32)
        out = np.empty_like(pts, dtype=np.float32)
        out[:, 0] = pts[:, 2]
        out[:, 1] = -pts[:, 0]
        out[:, 2] = pts[:, 1]
        return out
    if file_frame == "ros":
        return np.asarray(points, dtype=np.float32)
    raise ValueError("file_frame must be unity or ros, got {}".format(file_frame))


def build_voxel_cache(
    config: VoxelCacheConfig,
    *,
    write_provenance: bool = True,
) -> VoxelCacheBuildResult:
    """Build one NPZ occupancy cache from the supplied point-cloud binary.

    This is the former ``VoxelCollisionChecker.build_from_bin`` algorithm
    moved behind a data-layer owner.  It deliberately does not instantiate a
    runtime checker or load a native library.
    """

    map_path = Path(config.map_bin).expanduser().resolve()
    cache_path = Path(config.cache_npz).expanduser().resolve()
    if not map_path.exists():
        raise FileNotFoundError("map bin not found: {}".format(map_path))
    if float(config.voxel_size) <= 0.0 or not math.isfinite(float(config.voxel_size)):
        raise ValueError("voxel_size must be finite and positive")
    if int(config.chunk_points) <= 0:
        raise ValueError("chunk_points must be positive")
    if float(config.z_min) > float(config.z_max):
        raise ValueError("z_min must be <= z_max")

    file_size = map_path.stat().st_size
    data_offset = _resolve_data_offset(file_size, int(config.data_offset_bytes))
    data_bytes = file_size - data_offset
    if data_bytes <= 0 or data_bytes % 12 != 0:
        raise ValueError(
            "map bin payload size invalid: file_size={} data_offset={} payload={} payload%12={}".format(
                file_size, data_offset, data_bytes, data_bytes % 12
            )
        )

    total_points_in_file = data_bytes // 12
    total_points = total_points_in_file
    if int(config.max_points) > 0:
        total_points = min(total_points, int(config.max_points))

    header_value = None
    if data_offset >= 4:
        with map_path.open("rb") as handle:
            header_value = int(np.frombuffer(handle.read(4), dtype=np.uint32)[0])

    print("VOXEL_CACHE_BUILD_START")
    print("  map:", map_path)
    print("  file_size:", file_size)
    print("  data_offset_bytes:", data_offset)
    if header_value is not None:
        print("  first_u32_header:", header_value)
    print("  file_frame:", config.file_frame)
    print("  points_in_file:", total_points_in_file)
    print("  points_used:", total_points)
    print("  voxel_size:", config.voxel_size)
    print("  z_filter:", config.z_min, config.z_max)

    mm = np.memmap(
        str(map_path),
        dtype=np.float32,
        mode="r",
        offset=data_offset,
        shape=(total_points_in_file, 3),
    )
    n = int(total_points)
    chunk = int(config.chunk_points)

    min_ijk: Optional[np.ndarray] = None
    max_ijk: Optional[np.ndarray] = None
    kept_points = 0
    started = time.time()

    # First pass: find the exact global integer-coordinate bounds.
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        points = np.asarray(mm[start:end], dtype=np.float32)
        points = _convert_frame(points, config.file_frame)
        points = points[
            (points[:, 2] >= float(config.z_min))
            & (points[:, 2] <= float(config.z_max))
        ]
        if points.size == 0:
            continue
        ijk = np.floor(points / float(config.voxel_size)).astype(np.int64)
        kept_points += int(ijk.shape[0])
        current_min = ijk.min(axis=0)
        current_max = ijk.max(axis=0)
        min_ijk = current_min if min_ijk is None else np.minimum(min_ijk, current_min)
        max_ijk = current_max if max_ijk is None else np.maximum(max_ijk, current_max)
        print(
            "  pass1 {:>10}/{:<10} kept={:<10} elapsed={:.1f}s".format(
                end, n, kept_points, time.time() - started
            ),
            end="\r",
        )

    print("")
    if min_ijk is None or max_ijk is None or kept_points == 0:
        raise RuntimeError("no map points kept after filtering")

    min_ijk = np.asarray(min_ijk, dtype=np.int64)
    max_ijk = np.asarray(max_ijk, dtype=np.int64)
    grid_shape = max_ijk - min_ijk + 1
    if np.any(grid_shape <= 0):
        raise RuntimeError("invalid grid shape {}".format(grid_shape))
    if int(grid_shape[0]) * int(grid_shape[1]) * int(grid_shape[2]) >= 9_000_000_000_000_000_000:
        raise RuntimeError("grid too large for int64 key packing: {}".format(grid_shape))

    print("  kept_points:", kept_points)
    print("  origin_ijk:", min_ijk.tolist())
    print("  max_ijk:", max_ijk.tolist())
    print("  grid_shape:", grid_shape.tolist())

    # Second pass: preserve the historical sorted unique packed-key output.
    unique_chunks = []
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        points = np.asarray(mm[start:end], dtype=np.float32)
        points = _convert_frame(points, config.file_frame)
        points = points[
            (points[:, 2] >= float(config.z_min))
            & (points[:, 2] <= float(config.z_max))
        ]
        if points.size == 0:
            continue
        ijk = np.floor(points / float(config.voxel_size)).astype(np.int64)
        keys = _pack_keys(ijk, min_ijk, grid_shape)
        unique_chunks.append(np.unique(keys))
        print(
            "  pass2 {:>10}/{:<10} chunks={} elapsed={:.1f}s".format(
                end, n, len(unique_chunks), time.time() - started
            ),
            end="\r",
        )

    print("")
    all_keys = np.unique(np.concatenate(unique_chunks).astype(np.int64))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the historical business payload and field order unchanged.  Source
    # identity is recorded in the adjacent sidecar rather than changing the
    # runtime NPZ layout consumed by the native loaders.
    write_npz_atomic(
        cache_path,
        {
            "occupied_keys": all_keys,
            "origin_ijk": min_ijk,
            "grid_shape": grid_shape,
            "voxel_size": np.asarray(float(config.voxel_size), dtype=np.float64),
            "data_offset_bytes": np.asarray(int(data_offset), dtype=np.int64),
            "file_frame": np.asarray(str(config.file_frame)),
            "map_bin": np.asarray(str(map_path)),
            "kept_points": np.asarray(kept_points, dtype=np.int64),
        },
        compress=True,
    )

    cache_sha256 = file_sha256(cache_path)
    metadata: Dict[str, object] = {
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256,
        "point_cloud_path": str(map_path),
        "point_cloud_sha256": file_sha256(map_path),
        "point_cloud_size_bytes": int(file_size),
        "data_offset_bytes": int(data_offset),
        "file_frame": str(config.file_frame),
        "voxel_size": float(config.voxel_size),
        "inflate_radius": float(config.inflate_radius),
        "occupied_count": int(all_keys.size),
        "occupied_key_dtype": "int64",
        "origin_ijk": [int(value) for value in min_ijk],
        "grid_shape": [int(value) for value in grid_shape],
        "kept_points": int(kept_points),
        "chunk_points": int(config.chunk_points),
        "max_points": int(config.max_points),
        "z_min": float(config.z_min),
        "z_max": float(config.z_max),
        "builder": "planning.data.voxel_cache",
        "algorithm_contract": "historical_collision_checker_two_pass_voxelization_v1",
    }
    if write_provenance:
        write_json_atomic(
            cache_path.with_suffix(cache_path.suffix + ".meta.json"),
            metadata,
            trailing_newline=True,
        )

    print("VOXEL_CACHE_BUILD=PASS")
    print("  cache:", cache_path)
    print("  occupied_voxels:", all_keys.size)
    print(
        "  compression_ratio points/voxels: {:.2f}".format(
            kept_points / max(1, all_keys.size)
        )
    )
    print("  elapsed_s:", round(time.time() - started, 2))
    return VoxelCacheBuildResult(cache_path=cache_path, metadata=metadata)


def _pack_keys(ijk: np.ndarray, origin_ijk: np.ndarray, grid_shape: np.ndarray) -> np.ndarray:
    """Pack local voxel coordinates using the existing collision contract."""

    local = np.asarray(ijk, dtype=np.int64) - np.asarray(origin_ijk, dtype=np.int64)
    shape = np.asarray(grid_shape, dtype=np.int64)
    return (local[:, 0] * shape[1] + local[:, 1]) * shape[2] + local[:, 2]


__all__ = [
    "VoxelCacheBuildResult",
    "VoxelCacheConfig",
    "build_voxel_cache",
]
