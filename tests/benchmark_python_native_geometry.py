#!/usr/bin/env python3
"""Measure P direct adapters against the shared Python native geometry context.

The direct mode is the pre-C5 ownership seam: CollisionChecker and
GlobalRoutePlanner each construct their own native map.  The shared mode uses
NativeGeometryContext and retains one immutable native VoxelMap.  Both modes
use the same public Python APIs and the same read-only fixtures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List


DEFAULT_ROOT = Path("/home/xm/XM/src")
DEFAULT_CACHE = Path("/tmp/xmflight_collision_c2/forest_voxels_10cm.npz")
DEFAULT_CANDIDATES = Path(
    "/home/xm/XM/xm_ws/src/planning/data_back_20260826/teach/flight_20260717/mission_candidates.csv"
)
DEFAULT_COLLISION = Path(
    "/tmp/xm-cxx-c2-2-p-devel/planning/lib/libplanning_collision_checker.so"
)
DEFAULT_VOXEL = Path(
    "/tmp/xm-cxx-c2-2-p-devel/planning/lib/libplanning_voxel_map.so"
)
DEFAULT_ROUTE = Path(
    "/tmp/xm-cxx-c2-2-p-devel/planning/lib/libplanning_global_route.so"
)
DEFAULT_ARTIFACT = Path("/tmp/xmflight_python_binding_c5_context_benchmark.json")


RUNNER = r'''
from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np


root = Path(sys.argv[1]).resolve()
cache_path = Path(sys.argv[2]).resolve()
candidates_path = Path(sys.argv[3]).resolve()
variant = str(sys.argv[4])
size = int(sys.argv[5])
iterations = int(sys.argv[6])
collision_library = str(Path(sys.argv[7]).resolve())
voxel_library = str(Path(sys.argv[8]).resolve())
route_library = str(Path(sys.argv[9]).resolve())
sys.path.insert(0, str(root / "python"))

native_calls = {}
counted_symbols = {
    "planning_voxel_map_create",
    "planning_collision_create",
    "planning_collision_create_from_voxel_map",
    "planning_global_route_create",
    "planning_global_route_create_from_voxel_map",
}


class _FunctionProxy:
    def __init__(self, function, symbol):
        object.__setattr__(self, "_function", function)
        object.__setattr__(self, "_symbol", symbol)

    def __call__(self, *args, **kwargs):
        native_calls[self._symbol] = native_calls.get(self._symbol, 0) + 1
        return self._function(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._function, name)

    def __setattr__(self, name, value):
        setattr(self._function, name, value)


class _LibraryProxy:
    def __init__(self, library):
        object.__setattr__(self, "_library", library)

    def __getattr__(self, name):
        value = getattr(self._library, name)
        if name in counted_symbols:
            return _FunctionProxy(value, name)
        return value


real_cdll = ctypes.CDLL
ctypes.CDLL = lambda *args, **kwargs: _LibraryProxy(real_cdll(*args, **kwargs))

load_calls = 0
real_np_load = np.load


def counted_np_load(*args, **kwargs):
    global load_calls
    load_calls += 1
    return real_np_load(*args, **kwargs)


np.load = counted_np_load
os.environ["PLANNING_COLLISION_LIBRARY"] = collision_library
os.environ["PLANNING_VOXEL_MAP_LIBRARY"] = voxel_library
os.environ["PLANNING_GLOBAL_ROUTE_LIBRARY"] = route_library
os.environ["PLANNING_COLLISION_BACKEND"] = "cpp_cpu"
os.environ["PLANNING_GLOBAL_ROUTE_BACKEND"] = "cpp_native"

from planning.mission.global_route import (
    GlobalRouteConfig,
    GlobalRoutePlanner2D,
    GlobalRouteUnavailableError,
)
from planning.safety.collision_checker import VoxelCollisionChecker

route_config = GlobalRouteConfig(
    resolution_m=0.25,
    flight_z_min_m=1.0,
    flight_z_max_m=3.0,
    lookahead_m=3.0,
    tracking_margin_m=0.0,
    nearest_free_radius_cells=6,
)

with candidates_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))[:size]
if len(rows) != size:
    raise RuntimeError("candidate fixture has {} rows, expected {}".format(len(rows), size))
pairs = [
    (
        [float(row["start_x"]), float(row["start_y"]), float(row["start_z"])],
        [float(row["goal_x"]), float(row["goal_y"]), float(row["goal_z"])],
    )
    for row in rows
]
pair_sha = hashlib.sha256(
    json.dumps(pairs, separators=(",", ":"), sort_keys=False).encode("utf-8")
).hexdigest()

rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
cache_started = time.perf_counter()
if variant == "direct":
    checker = VoxelCollisionChecker.load_cache(
        cache_path, inflate_radius=0.35, voxel_size=0.10
    )
else:
    from planning.native.geometry import NativeGeometryContext

    context = NativeGeometryContext.from_voxel_cache(
        cache_path, voxel_size=0.10, route_config=route_config
    )
    checker = context.collision_checker(0.35)
cache_setup_s = time.perf_counter() - cache_started

route_setup_started = time.perf_counter()
if variant == "direct":
    planner = GlobalRoutePlanner2D(checker, route_config)
else:
    planner = context.global_route_planner(checker, route_config)
route_setup_s = time.perf_counter() - route_setup_started

motion_primitives = SimpleNamespace(
    num_actions=105,
    pos_ref=np.zeros((105, 25, 3), dtype=np.float32),
)
action_ids = np.arange(105, dtype=np.int32)
positions = np.zeros((32, 3), dtype=np.float32)
positions[:, 2] = 2.0
yaws = np.zeros((32,), dtype=np.float64)
checker.check_pose_actions_array(
    motion_primitives, positions, yaws, action_ids, check_step=1
)

collision_started = time.perf_counter()
collision_digest = hashlib.sha256()
for _ in range(iterations):
    result = checker.check_pose_actions_array(
        motion_primitives, positions, yaws, action_ids, check_step=1
    )
    for name in ("valid", "minimum_distances", "first_collision_indices", "endpoints"):
        collision_digest.update(np.ascontiguousarray(result[name]).view(np.uint8))
collision_wall_s = time.perf_counter() - collision_started
collision_operations = int(iterations * positions.shape[0] * action_ids.size)

route_started = time.perf_counter()
route_digest = hashlib.sha256()
found = 0
for start, goal in pairs:
    try:
        route = planner.plan(start, goal)
        found += 1
        route_digest.update(np.ascontiguousarray(route).view(np.uint8))
    except GlobalRouteUnavailableError:
        route_digest.update(b"NO_ROUTE")
route_wall_s = time.perf_counter() - route_started

stats = context.stats if variant != "direct" else {}
peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
map_create_count = int(native_calls.get("planning_voxel_map_create", 0))
legacy_map_create_count = int(
    native_calls.get("planning_collision_create", 0)
    + native_calls.get("planning_global_route_create", 0)
)
native_map_count = map_create_count + legacy_map_create_count
map_ids = {
    "checker": getattr(checker, "native_map_id", None),
    "route": getattr(planner, "native_map_id", None),
}
if variant != "direct":
    map_ids["context"] = int(context.native_map_id)

if variant != "direct":
    planner.close()
    checker.close()
    context.close()
else:
    planner.close()
    checker.close()

print(json.dumps({
    "variant": variant,
    "cache_path": str(cache_path),
    "candidate_count": size,
    "pair_fixture_sha256": pair_sha,
    "np_load_calls": int(load_calls),
    "native_calls": native_calls,
    "native_map_construction_count": native_map_count,
    "occupied_buffer_allocation_count": native_map_count,
    "map_ids": map_ids,
    "collision_route_share_map": bool(
        variant != "direct"
        and map_ids.get("checker") is not None
        and map_ids.get("checker") == map_ids.get("route") == map_ids.get("context")
    ),
    "cache_setup_s": cache_setup_s,
    "route_setup_s": route_setup_s,
    "geometry_init_s": cache_setup_s + route_setup_s,
    "collision_wall_s": collision_wall_s,
    "collision_operations": collision_operations,
    "collision_poses_x_actions_per_sec": collision_operations / max(collision_wall_s, 1.0e-12),
    "route_wall_s": route_wall_s,
    "routes_found": found,
    "routes_per_sec": size / max(route_wall_s, 1.0e-12),
    "collision_digest": collision_digest.hexdigest(),
    "route_digest": route_digest.hexdigest(),
    "rss_before_kb": rss_before,
    "peak_rss_kb": peak_rss,
    "peak_rss_delta_kb": max(0, peak_rss - rss_before),
    "context_stats": stats,
}, sort_keys=True))
'''


def _run_once(
    root: Path,
    cache: Path,
    candidates: Path,
    variant: str,
    size: int,
    iterations: int,
    collision: Path,
    voxel: Path,
    route: Path,
) -> Dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(root / "python")
    environment["LD_LIBRARY_PATH"] = ":".join(
        [
            str(collision.parent),
            str(voxel.parent),
            str(route.parent),
            environment.get("LD_LIBRARY_PATH", ""),
        ]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            RUNNER,
            str(root),
            str(cache),
            str(candidates),
            variant,
            str(size),
            str(iterations),
            str(collision),
            str(voxel),
            str(route),
        ],
        cwd=str(root),
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "{} context benchmark failed ({}):\n{}\n{}".format(
                variant, result.returncode, result.stdout, result.stderr
            )
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _aggregate(values: List[Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as np

    return {
        "runs": values,
        "best_collision_poses_x_actions_per_sec": max(
            item["collision_poses_x_actions_per_sec"] for item in values
        ),
        "mean_collision_poses_x_actions_per_sec": float(
            np.mean([item["collision_poses_x_actions_per_sec"] for item in values])
        ),
        "best_routes_per_sec": max(item["routes_per_sec"] for item in values),
        "mean_routes_per_sec": float(
            np.mean([item["routes_per_sec"] for item in values])
        ),
        "mean_geometry_init_s": float(
            np.mean([item["geometry_init_s"] for item in values])
        ),
        "max_peak_rss_kb": max(item["peak_rss_kb"] for item in values),
        "native_map_construction_counts": sorted(
            set(item["native_map_construction_count"] for item in values)
        ),
        "occupied_buffer_allocation_counts": sorted(
            set(item["occupied_buffer_allocation_count"] for item in values)
        ),
        "np_load_call_counts": sorted(set(item["np_load_calls"] for item in values)),
        "collision_route_share_map_all": all(
            item["collision_route_share_map"] for item in values
        ),
        "pair_fixture_sha256": sorted(
            set(item["pair_fixture_sha256"] for item in values)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--collision", type=Path, default=DEFAULT_COLLISION)
    parser.add_argument("--voxel", type=Path, default=DEFAULT_VOXEL)
    parser.add_argument("--route", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--collision-iterations", type=int, default=200)
    args = parser.parse_args()
    if int(args.runs) < 3:
        raise ValueError("at least three benchmark runs are required")
    if int(args.size) <= 0 or int(args.collision_iterations) <= 0:
        raise ValueError("size and collision iterations must be positive")
    for path in (args.cache, args.candidates, args.collision, args.voxel, args.route):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    root = args.root.resolve()
    results: Dict[str, Any] = {
        "runs": int(args.runs),
        "size": int(args.size),
        "collision_iterations": int(args.collision_iterations),
        "direct": _aggregate(
            [
                _run_once(
                    root,
                    args.cache.resolve(),
                    args.candidates.resolve(),
                    "direct",
                    int(args.size),
                    int(args.collision_iterations),
                    args.collision.resolve(),
                    args.voxel.resolve(),
                    args.route.resolve(),
                )
                for _ in range(int(args.runs))
            ]
        ),
        "shared": _aggregate(
            [
                _run_once(
                    root,
                    args.cache.resolve(),
                    args.candidates.resolve(),
                    "shared",
                    int(args.size),
                    int(args.collision_iterations),
                    args.collision.resolve(),
                    args.voxel.resolve(),
                    args.route.resolve(),
                )
                for _ in range(int(args.runs))
            ]
        ),
    }
    direct = results["direct"]
    shared = results["shared"]
    results["shared_collision_delta_percent"] = 100.0 * (
        shared["best_collision_poses_x_actions_per_sec"]
        / max(direct["best_collision_poses_x_actions_per_sec"], 1.0e-12)
        - 1.0
    )
    results["shared_route_delta_percent"] = 100.0 * (
        shared["best_routes_per_sec"]
        / max(direct["best_routes_per_sec"], 1.0e-12)
        - 1.0
    )
    args.artifact.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.artifact.resolve().write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PYTHON_CONTEXT_BENCHMARK_RUNS={}".format(args.runs))
    print(
        "DIRECT_BEST_COLLISION={:.6f} SHARED_BEST_COLLISION={:.6f}".format(
            direct["best_collision_poses_x_actions_per_sec"],
            shared["best_collision_poses_x_actions_per_sec"],
        )
    )
    print(
        "DIRECT_BEST_ROUTE={:.6f} SHARED_BEST_ROUTE={:.6f}".format(
            direct["best_routes_per_sec"], shared["best_routes_per_sec"]
        )
    )
    print("ARTIFACT={}".format(args.artifact.resolve()))
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
