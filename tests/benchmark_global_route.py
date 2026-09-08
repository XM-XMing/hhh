#!/usr/bin/env python3
"""Apples-to-apples O Python vs P C++ global-route benchmark.

Each repeat is a fresh process with one cache load and one planner. The
benchmark is read-only with respect to the project; its output belongs under
/tmp.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List


O_ROOT = Path("/home/xm/XM/xm_ws/src/planning").resolve()
P_ROOT = Path("/home/xm/XM/src").resolve()
DEFAULT_CACHE = Path("/tmp/xmflight_collision_c2/forest_voxels_10cm.npz")
DEFAULT_CANDIDATES = Path(
    "/home/xm/XM/xm_ws/src/planning/data_back_20260826/teach/flight_20260717/mission_candidates.csv"
)
DEFAULT_ARTIFACT = Path("/tmp/xmflight_global_route_c3/global_route_benchmark.json")
DEFAULT_P_ROUTE = Path(
    "/tmp/xm-cxx-c2-2-p-devel/planning/lib/libplanning_global_route.so"
)


RUNNER = r'''
from __future__ import annotations

import csv
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


tree = sys.argv[1]
root = Path(sys.argv[2]).resolve()
cache_path = Path(sys.argv[3]).resolve()
candidates_path = Path(sys.argv[4]).resolve()
size = int(sys.argv[5])
sys.path.insert(0, str(root / "python"))

if tree == "O":
    from planning.global_route import GlobalRouteConfig, GlobalRoutePlanner2D, GlobalRouteUnavailableError
else:
    from planning.mission.global_route import GlobalRouteConfig, GlobalRoutePlanner2D, GlobalRouteUnavailableError


with candidates_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))[:size]
if len(rows) != size:
    raise RuntimeError("benchmark candidate fixture has {} rows, expected {}".format(len(rows), size))
pairs = [
    (
        [float(row["start_x"]), float(row["start_y"]), float(row["start_z"])],
        [float(row["goal_x"]), float(row["goal_y"]), float(row["goal_z"])],
    )
    for row in rows
]
pair_fixture_sha256 = hashlib.sha256(
    json.dumps(pairs, separators=(",", ":"), sort_keys=False).encode("utf-8")
).hexdigest()

rss_before_init_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
cache_started = time.perf_counter()
with np.load(str(cache_path), allow_pickle=False) as data:
    checker = SimpleNamespace(
        occupied_keys=data["occupied_keys"].copy(),
        origin_ijk=data["origin_ijk"].copy(),
        grid_shape=data["grid_shape"].copy(),
        voxel_size=float(data["voxel_size"]),
        collision_radius=0.35 + 0.5 * np.sqrt(3.0) * 0.10,
    )
cache_read_time_s = time.perf_counter() - cache_started

planner_started = time.perf_counter()
planner = GlobalRoutePlanner2D(
    checker,
    GlobalRouteConfig(
        resolution_m=0.25,
        flight_z_min_m=1.0,
        flight_z_max_m=3.0,
        lookahead_m=3.0,
        tracking_margin_m=0.0,
        nearest_free_radius_cells=6,
    ),
)
planner_init_time_s = time.perf_counter() - planner_started

latencies = []
found = 0
batch_started = time.perf_counter()
for start, goal in pairs:
    call_started = time.perf_counter()
    try:
        planner.plan(start, goal)
        found += 1
    except GlobalRouteUnavailableError:
        pass
    latencies.append(time.perf_counter() - call_started)
batch_wall_time_s = time.perf_counter() - batch_started
peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
n_cells = int(np.prod(np.asarray(planner.shape, dtype=np.int64)))
if tree == "P":
    # blocked + float costs + int parents + two uint32 generation arrays.
    # Queue and last-route capacities are dynamic and excluded here.
    persistent_workspace_bytes = n_cells * (1 + 4 + 4 + 4 + 4)
    per_plan_workspace_bytes = 0
else:
    persistent_workspace_bytes = 0
    per_plan_workspace_bytes = n_cells * (4 + 4 + 1)


def percentile(value):
    return float(np.percentile(np.asarray(latencies, dtype=np.float64), value))


print(json.dumps({
    "tree": tree,
    "size": size,
    "pair_fixture_sha256": pair_fixture_sha256,
    "found_routes": found,
    "route_count": size,
    "batch_wall_time_s": batch_wall_time_s,
    "routes_per_sec": size / max(batch_wall_time_s, 1.0e-12),
    "found_routes_per_sec": found / max(batch_wall_time_s, 1.0e-12),
    "latency_mean_ms": 1000.0 * float(np.mean(latencies)),
    "latency_p50_ms": 1000.0 * percentile(50),
    "latency_p95_ms": 1000.0 * percentile(95),
    "cache_read_time_s": cache_read_time_s,
    "planner_init_time_s": planner_init_time_s,
    "init_time_s": cache_read_time_s + planner_init_time_s,
    "rss_before_init_kb": rss_before_init_kb,
    "peak_rss_kb": peak_rss_kb,
    "peak_rss_delta_kb": max(0, peak_rss_kb - rss_before_init_kb),
    "voxel_map_load_count": 1,
    "route_planner_count": 1,
    "grid_shape": [int(value) for value in planner.shape],
    "planner_workspace_persistent_bytes": persistent_workspace_bytes,
    "planner_workspace_per_plan_bytes": per_plan_workspace_bytes,
    "backend": getattr(planner, "backend_name", "python-reference"),
}, sort_keys=True))
'''


def _run_once(
    tree: str,
    root: Path,
    cache: Path,
    candidates: Path,
    size: int,
    route_library: Path,
) -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "python")
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(route_library.parent), env.get("LD_LIBRARY_PATH", "")]
    )
    if tree == "P":
        env["PLANNING_GLOBAL_ROUTE_BACKEND"] = "cpp"
        env["PLANNING_GLOBAL_ROUTE_LIBRARY"] = str(route_library)
    else:
        env.pop("PLANNING_GLOBAL_ROUTE_BACKEND", None)
        env.pop("PLANNING_GLOBAL_ROUTE_LIBRARY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            RUNNER,
            tree,
            str(root),
            str(cache),
            str(candidates),
            str(size),
        ],
        cwd=str(root),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "{} benchmark failed ({}):\n{}\n{}".format(
                tree, result.returncode, result.stdout, result.stderr
            )
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as np

    return {
        "runs": runs,
        "best_routes_per_sec": max(run["routes_per_sec"] for run in runs),
        "median_routes_per_sec": float(np.median([run["routes_per_sec"] for run in runs])),
        "best_found_routes_per_sec": max(run["found_routes_per_sec"] for run in runs),
        "mean_latency_ms": float(np.mean([run["latency_mean_ms"] for run in runs])),
        "p50_latency_ms": float(np.mean([run["latency_p50_ms"] for run in runs])),
        "p95_latency_ms": float(np.mean([run["latency_p95_ms"] for run in runs])),
        "init_time_s_mean": float(np.mean([run["init_time_s"] for run in runs])),
        "peak_rss_kb_max": max(run["peak_rss_kb"] for run in runs),
        "peak_rss_delta_kb_max": max(run["peak_rss_delta_kb"] for run in runs),
        "voxel_map_load_count_each_run": sorted(set(run["voxel_map_load_count"] for run in runs)),
        "pair_fixture_sha256": sorted(set(run["pair_fixture_sha256"] for run in runs)),
        "found_routes_each_run": [run["found_routes"] for run in runs],
        "route_count": runs[0]["route_count"],
    }


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--p-route", type=Path, default=DEFAULT_P_ROUTE)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sizes", default="100,1000")
    args = parser.parse_args()
    if int(args.runs) < 3:
        raise ValueError("at least three benchmark runs are required")
    sizes = [int(value) for value in str(args.sizes).split(",") if value.strip()]
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("--sizes must contain positive integers")
    all_results: Dict[str, Any] = {
        "runs": int(args.runs), "sizes": sizes, "O": {}, "P": {}
    }
    for size in sizes:
        for tree, root in (("O", O_ROOT), ("P", P_ROOT)):
            run_results = [
                _run_once(
                    tree,
                    root,
                    args.cache.resolve(),
                    args.candidates.resolve(),
                    size,
                    args.p_route.resolve(),
                )
                for _ in range(int(args.runs))
            ]
            all_results[tree][str(size)] = _aggregate(run_results)
    for size in sizes:
        o = all_results["O"][str(size)]
        p = all_results["P"][str(size)]
        p["speedup_over_o_best"] = p["best_routes_per_sec"] / max(o["best_routes_per_sec"], 1.0e-12)
        p["throughput_delta_percent_over_o_best"] = 100.0 * (p["speedup_over_o_best"] - 1.0)
    args.artifact.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.artifact.resolve().write_text(
        json.dumps(all_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("BENCHMARK_RUNS={}".format(args.runs))
    for size in sizes:
        o = all_results["O"][str(size)]
        p = all_results["P"][str(size)]
        print(
            "SIZE={} O_BEST_ROUTES_PER_SEC={:.6f} P_BEST_ROUTES_PER_SEC={:.6f} SPEEDUP={:.6f}".format(
                size, o["best_routes_per_sec"], p["best_routes_per_sec"], p["speedup_over_o_best"]
            )
        )
    print("ARTIFACT={}".format(args.artifact.resolve()))
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
