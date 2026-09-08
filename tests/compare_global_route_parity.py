#!/usr/bin/env python3
"""Compare the O Python route contract with the P native route contract.

The worker processes deliberately import the two trees through separate
``PYTHONPATH`` values.  Expected behavior is always observed from O at
runtime; the comparator does not reimplement A* to manufacture an oracle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


O_ROOT = Path("/home/xm/XM/xm_ws/src/planning")
P_ROOT = Path("/home/xm/XM/src")
DEFAULT_ARTIFACT_DIR = Path("/tmp/xmflight_global_route_c3")
DEFAULT_PRODUCTION_CACHE = Path(
    "/tmp/xmflight_collision_c2/forest_voxels_10cm.npz"
)
DEFAULT_PRODUCTION_CANDIDATES = Path(
    "/home/xm/XM/xm_ws/src/planning/data_back_20260826/teach/flight_20260717/mission_candidates.csv"
)


def _key(x: int, y: int, z: int, shape: Sequence[int]) -> int:
    return int(x) * int(shape[1]) * int(shape[2]) + int(y) * int(shape[2]) + int(z)


def _synthetic_map(
    name: str,
    shape: Sequence[int] = (24, 24, 1),
    occupied_xy: Sequence[Tuple[int, int]] = (),
    collision_radius: float = 0.0,
    nearest_free_radius_cells: int = 0,
) -> Dict[str, Any]:
    shape_list = [int(value) for value in shape]
    occupied = sorted(_key(x, y, 0, shape_list) for x, y in occupied_xy)
    return {
        "name": name,
        "source": "synthetic",
        "occupied_keys": occupied,
        "origin_ijk": [0, 0, 1],
        "grid_shape": shape_list,
        "voxel_size": 1.0,
        "collision_radius": float(collision_radius),
        "config": {
            "resolution_m": 1.0,
            "flight_z_min_m": 1.0,
            "flight_z_max_m": 3.0,
            "lookahead_m": 3.0,
            "tracking_margin_m": 0.0,
            "nearest_free_radius_cells": int(nearest_free_radius_cells),
        },
    }


def _add_case(
    cases: List[Dict[str, Any]],
    case_id: str,
    category: str,
    map_id: str,
    start: Sequence[float],
    goal: Sequence[float],
) -> None:
    cases.append(
        {
            "id": case_id,
            "category": category,
            "map_id": map_id,
            "start": [float(value) for value in start],
            "goal": [float(value) for value in goal],
        }
    )


def build_synthetic_fixture() -> Dict[str, Any]:
    maps: Dict[str, Dict[str, Any]] = {}
    maps["empty"] = _synthetic_map("empty", shape=(32, 32, 1))
    maps["single_obstacle"] = _synthetic_map(
        "single_obstacle", shape=(24, 24, 1), occupied_xy=((12, 12),)
    )
    maps["narrow_corridor"] = _synthetic_map(
        "narrow_corridor",
        shape=(28, 20, 1),
        occupied_xy=[
            (x, y)
            for x in range(28)
            for y in range(20)
            if y not in (9, 10)
        ],
    )
    diagonal_occupied = []
    for x in range(20):
        for y in range(20):
            if abs(y - x) > 1:
                diagonal_occupied.append((x, y))
    maps["diagonal_corridor"] = _synthetic_map(
        "diagonal_corridor", shape=(20, 20, 1), occupied_xy=diagonal_occupied
    )
    maps["corner_cutting"] = _synthetic_map(
        "corner_cutting",
        shape=(4, 4, 1),
        occupied_xy=((1, 0), (0, 1)),
    )
    maps["blocked_start"] = _synthetic_map(
        "blocked_start",
        shape=(12, 8, 1),
        occupied_xy=((0, 0),),
        nearest_free_radius_cells=2,
    )
    maps["blocked_goal"] = _synthetic_map(
        "blocked_goal",
        shape=(12, 8, 1),
        occupied_xy=((5, 0),),
        nearest_free_radius_cells=2,
    )
    maps["no_path"] = _synthetic_map(
        "no_path",
        shape=(12, 12, 1),
        occupied_xy=[(4, y) for y in range(12)],
    )
    maps["tie"] = _synthetic_map("tie", shape=(8, 8, 1))
    maps["boundary"] = _synthetic_map("boundary", shape=(8, 8, 1), nearest_free_radius_cells=1)
    maps["long"] = _synthetic_map("long", shape=(72, 72, 1))

    dense_occupied = []
    rng = np.random.default_rng(20260827)
    for x in range(24):
        for y in range(24):
            if rng.random() < 0.28 and (x, y) not in ((1, 1), (22, 22)):
                dense_occupied.append((x, y))
    maps["dense"] = _synthetic_map(
        "dense", shape=(24, 24, 1), occupied_xy=dense_occupied, nearest_free_radius_cells=1
    )

    cases: List[Dict[str, Any]] = []
    for index in range(15):
        _add_case(
            cases,
            "empty-{:03d}".format(index),
            "empty",
            "empty",
            (0.1 + index * 0.13, 0.2 + index * 0.07, 1.5),
            (20.1 - index * 0.11, 18.2 - index * 0.09, 2.5),
        )
    for index in range(10):
        _add_case(
            cases,
            "single-obstacle-{:03d}".format(index),
            "single_obstacle",
            "single_obstacle",
            (1.1, 1.1 + index * 0.31, 1.5),
            (21.1, 21.1 - index * 0.27, 2.0),
        )
    for index in range(10):
        _add_case(
            cases,
            "narrow-corridor-{:03d}".format(index),
            "narrow_corridor",
            "narrow_corridor",
            (0.1 + index * 0.2, 9.1, 1.0),
            (27.1 - index * 0.15, 10.1, 3.0),
        )
    for index in range(10):
        _add_case(
            cases,
            "diagonal-corridor-{:03d}".format(index),
            "diagonal_corridor",
            "diagonal_corridor",
            (0.1, 0.1 + index * 0.05, 1.5),
            (19.1, 19.1 - index * 0.05, 1.5),
        )
    _add_case(cases, "corner-cutting-000", "corner_cutting", "corner_cutting", (0.1, 0.1, 1.5), (1.1, 1.1, 1.5))
    for index in range(4):
        _add_case(cases, "blocked-start-{:03d}".format(index), "blocked_start", "blocked_start", (0.1, 0.1, 1.5), (5.1 + index, 0.1, 1.5))
        _add_case(cases, "blocked-goal-{:03d}".format(index), "blocked_goal", "blocked_goal", (0.1, 0.1 + index, 1.5), (5.1, 0.1, 1.5))
    for index in range(8):
        _add_case(cases, "no-path-{:03d}".format(index), "no_path", "no_path", (1.1, 1.1 + index, 1.5), (8.1, 10.1 - index, 1.5))
    for index in range(10):
        _add_case(cases, "tie-{:03d}".format(index), "tie", "tie", (0.1, 0.1, 1.5), (2.1 + (index % 2), 1.1, 1.5))
    boundary_pairs = [
        ((0.0, 0.0, 1.0), (8.0, 8.0, 3.0)),
        ((-0.000001, 0.0, 1.0), (7.999999, 7.999999, 3.0)),
        ((0.0, 7.999999, 1.0), (7.999999, 0.0, 3.0)),
    ]
    for index in range(10):
        start, goal = boundary_pairs[index % len(boundary_pairs)]
        _add_case(cases, "boundary-{:03d}".format(index), "boundary", "boundary", start, goal)
    for index in range(5):
        _add_case(cases, "long-{:03d}".format(index), "long_route", "long", (0.1, 0.1 + index, 1.0), (65.1, 65.1 - index, 3.0))
    for index in range(20):
        _add_case(cases, "dense-{:03d}".format(index), "dense_obstacle", "dense", (1.1 + index % 4, 1.1, 1.5), (22.1, 18.1 - index % 5, 2.5))

    # Fill the required 100+ deterministic cases with additional varied
    # empty/sparse-map pairs; all are still observed through O in the worker.
    for index in range(20):
        _add_case(cases, "sparse-extra-{:03d}".format(index), "sparse_extra", "single_obstacle", (2.1, 2.1 + index * 0.2, 1.5), (18.1, 4.1 + index * 0.3, 2.5))

    workspace_cases = [
        {"id": "tie-000", "map_id": "tie"},
        {"id": "corner-cutting-000", "map_id": "corner_cutting"},
        {"id": "blocked-start-000", "map_id": "blocked_start"},
        {"id": "no-path-000", "map_id": "no_path"},
        {"id": "empty-000", "map_id": "empty"},
        {"id": "empty-001", "map_id": "empty"},
    ]
    return {"kind": "synthetic", "maps": maps, "cases": cases, "workspace_cases": workspace_cases}


def load_production_cases(
    cache_path: Path,
    candidates_path: Path,
    limit: int = 1000,
) -> Dict[str, Any]:
    if not cache_path.is_file():
        raise FileNotFoundError("production cache unavailable: {}".format(cache_path))
    if not candidates_path.is_file():
        raise FileNotFoundError("production candidate fixture unavailable: {}".format(candidates_path))
    with candidates_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[: int(limit)]
    if len(rows) < int(limit):
        raise ValueError("production candidate fixture has only {} rows".format(len(rows)))
    # Keep a bounded slice of the fixed candidate artifact, then add
    # deterministic pairs sampled from the actual O route grid.  This gives
    # the production-like route gate explicit short/medium/long, blocked,
    # near-obstacle, and boundary coverage without generating formal data.
    module = _load_o_module()
    with np.load(str(cache_path), allow_pickle=False) as data:
        checker = SimpleNamespace(
            occupied_keys=data["occupied_keys"].copy(),
            origin_ijk=data["origin_ijk"].copy(),
            grid_shape=data["grid_shape"].copy(),
            voxel_size=float(data["voxel_size"]),
            collision_radius=0.35 + 0.5 * math.sqrt(3.0) * 0.10,
        )
    grid_planner = module.GlobalRoutePlanner2D(checker, module.GlobalRouteConfig())
    blocked = np.asarray(grid_planner.blocked, dtype=np.bool_)
    free_cells = np.argwhere(~blocked)
    blocked_cells = np.argwhere(blocked)
    adjacent_to_blocked = np.zeros_like(blocked, dtype=np.bool_)
    adjacent_to_blocked[1:, :] |= blocked[:-1, :]
    adjacent_to_blocked[:-1, :] |= blocked[1:, :]
    adjacent_to_blocked[:, 1:] |= blocked[:, :-1]
    adjacent_to_blocked[:, :-1] |= blocked[:, 1:]
    near_obstacle_cells = np.argwhere(~blocked & adjacent_to_blocked)
    nx, ny = blocked.shape
    boundary_cells = free_cells[
        (free_cells[:, 0] < 3)
        | (free_cells[:, 1] < 3)
        | (free_cells[:, 0] >= nx - 3)
        | (free_cells[:, 1] >= ny - 3)
    ]
    if min(len(free_cells), len(blocked_cells), len(near_obstacle_cells), len(boundary_cells)) == 0:
        raise ValueError("production route grid lacks required fixture categories")

    def world_point(cell: Sequence[int], z: float) -> List[float]:
        xy = np.asarray(grid_planner.cell_to_world(cell), dtype=np.float32)
        return [float(xy[0]), float(xy[1]), float(z)]

    def offset_pair(
        cells: np.ndarray,
        index: int,
        dx: int,
        dy: int,
        start_z: float = 2.0,
        goal_z: float = 2.0,
    ) -> Tuple[List[float], List[float]]:
        start_cell = cells[(int(index) * 7919) % len(cells)]
        goal_cell = np.asarray(
            [
                min(nx - 1, max(0, int(start_cell[0]) + int(dx))),
                min(ny - 1, max(0, int(start_cell[1]) + int(dy))),
            ],
            dtype=np.int64,
        )
        if np.array_equal(start_cell, goal_cell):
            goal_cell[0] = min(nx - 1, int(start_cell[0]) + 1)
        return world_point(start_cell, start_z), world_point(goal_cell, goal_z)

    cases = []
    fixed_count = min(len(rows), max(0, int(limit) - 900))
    for index, row in enumerate(rows[:fixed_count]):
        cases.append(
            {
                "id": "production-candidate-{:04d}".format(index),
                "category": "production_candidate",
                "map_id": "production",
                "start": [float(row["start_x"]), float(row["start_y"]), float(row["start_z"])],
                "goal": [float(row["goal_x"]), float(row["goal_y"]), float(row["goal_z"])],
                "mission_id": row.get("mission_id", ""),
            }
        )

    category_specs = (
        ("short", free_cells, 180, 8, 0),
        ("medium", free_cells, 180, 80, 24),
        ("long", free_cells, 180, 240, 96),
        ("blocked", blocked_cells, 120, 24, 16),
        ("near_obstacle", near_obstacle_cells, 120, 48, -32),
        ("boundary", boundary_cells, 120, -(nx // 2), ny // 3),
    )
    category_index = 0
    for category, cells, count, dx, dy in category_specs:
        for local_index in range(count):
            start, goal = offset_pair(
                cells,
                category_index,
                dx,
                dy,
                start_z=1.0 + 0.5 * (local_index % 5),
                goal_z=3.0 - 0.5 * (local_index % 5),
            )
            cases.append(
                {
                    "id": "production-{}-{:04d}".format(category, local_index),
                    "category": category,
                    "map_id": "production",
                    "start": start,
                    "goal": goal,
                }
            )
            category_index += 1
    if len(cases) != int(limit):
        raise AssertionError("production fixture case count is {} not {}".format(len(cases), limit))
    return {
        "kind": "production_like",
        "maps": {
            "production": {
                "name": "production",
                "source": "npz",
                "path": str(cache_path),
                "collision_radius": 0.35 + 0.5 * math.sqrt(3.0) * 0.10,
                "config": {
                    "resolution_m": 0.25,
                    "flight_z_min_m": 1.0,
                    "flight_z_max_m": 3.0,
                    "lookahead_m": 3.0,
                    "tracking_margin_m": 0.0,
                    "nearest_free_radius_cells": 6,
                },
            }
        },
        "cases": cases,
        "workspace_cases": [
            {"id": "production-candidate-0000", "map_id": "production"},
            {"id": "production-candidate-0001", "map_id": "production"},
            {"id": "production-candidate-0002", "map_id": "production"},
        ],
    }


def _load_o_module():
    path = O_ROOT / "python/planning/mission/global_route.py"
    spec = importlib.util.spec_from_file_location("xm_c3_o_global_route", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load O route module: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _production_map_from_o(cache_path: Path) -> Dict[str, Any]:
    module = _load_o_module()
    with np.load(str(cache_path), allow_pickle=False) as data:
        checker = SimpleNamespace(
            occupied_keys=data["occupied_keys"].copy(),
            origin_ijk=data["origin_ijk"].copy(),
            grid_shape=data["grid_shape"].copy(),
            voxel_size=float(data["voxel_size"]),
            collision_radius=0.35 + 0.5 * math.sqrt(3.0) * 0.10,
        )
    planner = module.GlobalRoutePlanner2D(checker, module.GlobalRouteConfig())
    return {
        "origin_xy": planner.origin_xy.tolist(),
        "shape": planner.shape.tolist(),
        "blocked": planner.blocked,
        "checker": checker,
    }


def _result_for_route(planner: Any, route: np.ndarray) -> Dict[str, Any]:
    route_array = np.ascontiguousarray(route, dtype=np.float32)
    cells = [list(planner.world_to_cell(point)) for point in route_array[1:-1]]
    cost = 0.0
    for before, after in zip(cells, cells[1:]):
        cost += math.hypot(float(after[0] - before[0]), float(after[1] - before[1]))
    length = float(np.sum(np.linalg.norm(np.diff(route_array[:, :2], axis=0), axis=1)))
    payload = route_array.tobytes()
    return {
        "status": "found",
        "point_count": int(route_array.shape[0]),
        "grid_sequence": cells,
        "route_shape": list(route_array.shape),
        "route_hex": payload.hex(),
        "route_sha256": hashlib.sha256(payload).hexdigest(),
        "route_cost": float(cost),
        "route_length": length,
    }


def _run_one(planner: Any, unavailable_type: Any, start: Sequence[float], goal: Sequence[float]) -> Dict[str, Any]:
    start_cell = list(planner.world_to_cell(start))
    goal_cell = list(planner.world_to_cell(goal))
    conversion = {
        "start_cell": start_cell,
        "goal_cell": goal_cell,
        "start_cell_world_hex": np.asarray(
            planner.cell_to_world(start_cell), dtype=np.float64
        ).tobytes().hex(),
        "goal_cell_world_hex": np.asarray(
            planner.cell_to_world(goal_cell), dtype=np.float64
        ).tobytes().hex(),
    }
    try:
        route = planner.plan(start, goal)
    except unavailable_type as error:
        return {"status": "no_route", "error": str(error), "conversion": conversion}
    except Exception as error:  # pragma: no cover - surfaced as comparator failure
        return {
            "status": "error",
            "error_type": type(error).__name__,
            "error": repr(error),
            "conversion": conversion,
        }
    result = _result_for_route(planner, route)
    result["conversion"] = conversion
    return result


def worker_main(tree: str, fixture_path: Path, output_path: Path) -> int:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if tree == "O":
        from planning.mission.global_route import GlobalRoutePlanner2D, GlobalRouteUnavailableError
    elif tree == "P":
        from planning.mission.global_route import GlobalRoutePlanner2D, GlobalRouteUnavailableError
    else:
        raise ValueError("tree must be O or P")

    maps = fixture["maps"]
    checker_cache: Dict[str, Any] = {}
    planner_cache: Dict[str, Any] = {}

    def make_checker(map_id: str) -> Any:
        if map_id in checker_cache:
            return checker_cache[map_id]
        spec = maps[map_id]
        if spec["source"] == "npz":
            with np.load(spec["path"], allow_pickle=False) as data:
                checker = SimpleNamespace(
                    occupied_keys=data["occupied_keys"].copy(),
                    origin_ijk=data["origin_ijk"].copy(),
                    grid_shape=data["grid_shape"].copy(),
                    voxel_size=float(data["voxel_size"]),
                    collision_radius=float(spec["collision_radius"]),
                )
        else:
            checker = SimpleNamespace(
                occupied_keys=np.asarray(spec["occupied_keys"], dtype=np.int64),
                origin_ijk=np.asarray(spec["origin_ijk"], dtype=np.int64),
                grid_shape=np.asarray(spec["grid_shape"], dtype=np.int64),
                voxel_size=float(spec["voxel_size"]),
                collision_radius=float(spec["collision_radius"]),
            )
        checker_cache[map_id] = checker
        return checker

    def make_planner(map_id: str) -> Any:
        if map_id in planner_cache:
            return planner_cache[map_id]
        spec = maps[map_id]
        planner = GlobalRoutePlanner2D(
            make_checker(map_id),
            type_from_config(spec["config"]),
        )
        planner_cache[map_id] = planner
        return planner

    def type_from_config(config: Dict[str, Any]) -> Any:
        from planning.mission.global_route import GlobalRouteConfig
        return GlobalRouteConfig(**config)

    results: Dict[str, Any] = {"tree": tree, "cases": {}, "workspace": [], "map_metadata": {}}
    for case in fixture["cases"]:
        result = _run_one(
            make_planner(case["map_id"]),
            GlobalRouteUnavailableError,
            case["start"],
            case["goal"],
        )
        result["category"] = case["category"]
        results["cases"][case["id"]] = result

    workspace_results = []
    for item in fixture.get("workspace_cases", []):
        case = next(case for case in fixture["cases"] if case["id"] == item["id"])
        reuse = _run_one(
            make_planner(item["map_id"]),
            GlobalRouteUnavailableError,
            case["start"],
            case["goal"],
        )
        fresh = _run_one(
            GlobalRoutePlanner2D(make_checker(item["map_id"]), type_from_config(maps[item["map_id"]]["config"])),
            GlobalRouteUnavailableError,
            case["start"],
            case["goal"],
        )
        workspace_results.append({"id": item["id"], "reuse": reuse, "fresh": fresh})
    results["workspace"] = workspace_results
    for map_id, planner in planner_cache.items():
        blocked = np.ascontiguousarray(planner.blocked, dtype=np.bool_)
        results["map_metadata"][map_id] = {
            "shape": [int(value) for value in planner.shape],
            "origin_xy_hex": np.asarray(planner.origin_xy, dtype=np.float64).tobytes().hex(),
            "resolution_hex": np.asarray([float(planner.resolution_m)], dtype=np.float64).tobytes().hex(),
            "blocked_sha256": hashlib.sha256(blocked.tobytes()).hexdigest(),
            "blocked_count": int(np.count_nonzero(blocked)),
        }
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def _float_bits(value: Optional[float]) -> Optional[bytes]:
    if value is None:
        return None
    return np.asarray([float(value)], dtype=np.float64).tobytes()


def compare_case(o: Dict[str, Any], p: Dict[str, Any]) -> List[str]:
    differences = []
    for field in ("status", "point_count", "grid_sequence", "route_shape", "route_hex", "conversion"):
        if o.get(field) != p.get(field):
            differences.append(field)
    for field in ("route_cost", "route_length"):
        if _float_bits(o.get(field)) != _float_bits(p.get(field)):
            differences.append(field + "_bitwise")
    return differences


def compare_outputs(o: Dict[str, Any], p: Dict[str, Any], fixture: Dict[str, Any]) -> Dict[str, Any]:
    mismatches = []
    category_counts: Dict[str, int] = {}
    for case in fixture["cases"]:
        case_id = case["id"]
        category_counts[case["category"]] = category_counts.get(case["category"], 0) + 1
        differences = compare_case(o["cases"][case_id], p["cases"][case_id])
        if differences:
            mismatches.append({"id": case_id, "category": case["category"], "fields": differences})

    workspace_leaks = []
    for tree_output in (o, p):
        for item in tree_output["workspace"]:
            if compare_case(item["reuse"], item["fresh"]):
                workspace_leaks.append(tree_output["tree"] + ":" + item["id"])
    metadata_mismatches = []
    for map_id in fixture["maps"]:
        if o["map_metadata"].get(map_id) != p["map_metadata"].get(map_id):
            metadata_mismatches.append(map_id)
    route_parity = not mismatches
    return {
        "cases": len(fixture["cases"]),
        "category_counts": category_counts,
        "mismatches": mismatches,
        "metadata_mismatches": metadata_mismatches,
        "metadata_parity": not metadata_mismatches,
        "route_found_parity": route_parity,
        "route_point_count_parity": route_parity,
        "route_grid_sequence_parity": route_parity,
        "route_world_sequence_parity": route_parity,
        "route_cost_parity": route_parity,
        "route_length_parity": route_parity,
        "workspace_state_leak": workspace_leaks,
        "workspace_state_leak_zero": not workspace_leaks,
    }


def run_tree(tree: str, fixture_path: Path, output_path: Path, python_root: Path, native_library: Optional[Path]) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(python_root)
    if tree == "P":
        if native_library is None:
            raise ValueError("P native route library is required")
        environment["PLANNING_GLOBAL_ROUTE_BACKEND"] = "cpp"
        environment["PLANNING_GLOBAL_ROUTE_LIBRARY"] = str(native_library)
    else:
        environment.pop("PLANNING_GLOBAL_ROUTE_BACKEND", None)
        environment.pop("PLANNING_GLOBAL_ROUTE_LIBRARY", None)
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", "--tree", tree, "--fixture", str(fixture_path), "--output", str(output_path)],
        check=True,
        env=environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tree", choices=("O", "P"))
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--production-cache", type=Path, default=DEFAULT_PRODUCTION_CACHE)
    parser.add_argument("--production-candidates", type=Path, default=DEFAULT_PRODUCTION_CANDIDATES)
    parser.add_argument("--skip-production", action="store_true")
    parser.add_argument("--p-route", type=Path, default=Path("/tmp/xm-cxx-c2-2-p-devel/planning/lib/libplanning_global_route.so"))
    args = parser.parse_args()

    if args.worker:
        if args.tree is None or args.fixture is None or args.output is None:
            raise ValueError("worker requires --tree, --fixture and --output")
        return worker_main(args.tree, args.fixture, args.output)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    synthetic = build_synthetic_fixture()
    synthetic_path = args.artifact_dir / "c3_synthetic_fixture.json"
    synthetic_path.write_text(json.dumps(synthetic, indent=2, sort_keys=True), encoding="utf-8")
    o_synthetic_path = args.artifact_dir / "c3_synthetic_o.json"
    p_synthetic_path = args.artifact_dir / "c3_synthetic_p.json"
    run_tree("O", synthetic_path, o_synthetic_path, O_ROOT / "python", None)
    run_tree("P", synthetic_path, p_synthetic_path, P_ROOT / "python", args.p_route)
    o_synthetic = json.loads(o_synthetic_path.read_text(encoding="utf-8"))
    p_synthetic = json.loads(p_synthetic_path.read_text(encoding="utf-8"))
    synthetic_comparison = compare_outputs(o_synthetic, p_synthetic, synthetic)

    production_comparison = None
    if not args.skip_production:
        production = load_production_cases(args.production_cache, args.production_candidates)
        production_path = args.artifact_dir / "c3_production_fixture.json"
        production_path.write_text(json.dumps(production, indent=2, sort_keys=True), encoding="utf-8")
        o_production_path = args.artifact_dir / "c3_production_o.json"
        p_production_path = args.artifact_dir / "c3_production_p.json"
        run_tree("O", production_path, o_production_path, O_ROOT / "python", None)
        run_tree("P", production_path, p_production_path, P_ROOT / "python", args.p_route)
        o_production = json.loads(o_production_path.read_text(encoding="utf-8"))
        p_production = json.loads(p_production_path.read_text(encoding="utf-8"))
        production_comparison = compare_outputs(o_production, p_production, production)

    summary = {
        "synthetic": synthetic_comparison,
        "production_like": production_comparison,
        "status": "PASS"
        if synthetic_comparison["metadata_parity"]
        and synthetic_comparison["route_world_sequence_parity"]
        and synthetic_comparison["workspace_state_leak_zero"]
        and (production_comparison is None or (production_comparison["metadata_parity"] and production_comparison["route_world_sequence_parity"] and production_comparison["workspace_state_leak_zero"]))
        else "FAIL",
    }
    summary_path = args.artifact_dir / "c3_global_route_parity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("SYNTHETIC_CASES={}".format(synthetic_comparison["cases"]))
    print("SYNTHETIC_ROUTE_PARITY={}".format("PASS" if synthetic_comparison["route_world_sequence_parity"] else "FAIL"))
    print("SYNTHETIC_METADATA_PARITY={}".format("PASS" if synthetic_comparison["metadata_parity"] else "FAIL"))
    print("SYNTHETIC_WORKSPACE_STATE_LEAK={}".format(len(synthetic_comparison["workspace_state_leak"])))
    if production_comparison is not None:
        print("PRODUCTION_CASES={}".format(production_comparison["cases"]))
        print("PRODUCTION_ROUTE_PARITY={}".format("PASS" if production_comparison["route_world_sequence_parity"] else "FAIL"))
        print("PRODUCTION_METADATA_PARITY={}".format("PASS" if production_comparison["metadata_parity"] else "FAIL"))
        print("PRODUCTION_WORKSPACE_STATE_LEAK={}".format(len(production_comparison["workspace_state_leak"])))
    print("SUMMARY_ARTIFACT={}".format(summary_path))
    print("RESULT={}".format(summary["status"]))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
