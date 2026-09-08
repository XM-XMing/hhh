#!/usr/bin/env python3
"""Build the canonical Planning voxel cache from a point-cloud binary."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.common.paths import resolve_package_path
from planning.data.voxel_cache import VoxelCacheConfig, build_voxel_cache


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one deterministic voxel occupancy cache from XYZ points."
    )
    parser.add_argument("--map-bin", default="data/map_data/forest_point_cloud.bin")
    parser.add_argument("--file-frame", choices=("unity", "ros"), default="unity")
    parser.add_argument("--cache", dest="cache_npz", default="data/map_data/forest_voxels_10cm.npz")
    parser.add_argument("--data-offset-bytes", type=int, default=-1)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--inflate-radius", type=float, default=0.35)
    parser.add_argument("--chunk-points", type=int, default=2_000_000)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--z-min", type=float, default=-1e9)
    parser.add_argument("--z-max", type=float, default=1e9)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    result = build_voxel_cache(
        VoxelCacheConfig(
            map_bin=resolve_package_path(args.map_bin),
            cache_npz=resolve_package_path(args.cache_npz),
            file_frame=args.file_frame,
            voxel_size=args.voxel_size,
            inflate_radius=args.inflate_radius,
            data_offset_bytes=args.data_offset_bytes,
            chunk_points=args.chunk_points,
            max_points=args.max_points,
            z_min=args.z_min,
            z_max=args.z_max,
        )
    )
    print("VOXEL_CACHE_SHA256={}".format(result.metadata["cache_sha256"]))
    print("POINT_CLOUD_SHA256={}".format(result.metadata["point_cloud_sha256"]))
    print("OCCUPIED_COUNT={}".format(result.metadata["occupied_count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
