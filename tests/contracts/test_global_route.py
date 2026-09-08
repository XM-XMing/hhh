#!/usr/bin/env python3
"""Synthetic regression test for privileged global A* routing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


from planning.mission.global_route import (
    DEFAULT_GLOBAL_ROUTE_LOOKAHEAD_M,
    DEFAULT_GLOBAL_ROUTE_RESOLUTION_M,
    DEFAULT_GLOBAL_ROUTE_TRACKING_MARGIN_M,
    GlobalRoutePlanner2D,
)


def main() -> int:
    shape = np.asarray([20, 20, 5], dtype=np.int64)
    occupied = []
    for y in range(20):
        if y in (8, 9, 10, 11):
            continue
        for z in (1, 2, 3):
            occupied.append(10 * shape[1] * shape[2] + y * shape[2] + z)
    checker = SimpleNamespace(
        occupied_keys=np.asarray(sorted(occupied), dtype=np.int64),
        origin_ijk=np.asarray([0, 0, 0], dtype=np.int64),
        grid_shape=shape,
        voxel_size=0.5,
        collision_radius=0.20,
    )
    planner = GlobalRoutePlanner2D(checker)
    route = planner.plan([1.0, 2.0, 1.0], [8.0, 2.0, 2.0])
    cells = [planner.world_to_cell(point) for point in route]
    checks = {
        "route_exists": route.shape[0] > 2,
        "route_avoids_wall": all(not planner.blocked[cell] for cell in cells[1:-1]),
        "route_changes_altitude": abs(float(route[-1, 2] - route[0, 2]) - 1.0) < 1.0e-6,
        "guidance_is_forward": planner.guidance_goal(route, route[0])[0] > route[0, 0],
        "production_resolution_is_025m": abs(DEFAULT_GLOBAL_ROUTE_RESOLUTION_M - 0.25) < 1.0e-9,
        "production_lookahead_is_3m": abs(DEFAULT_GLOBAL_ROUTE_LOOKAHEAD_M - 3.0) < 1.0e-9,
        "production_tracking_margin_is_zero": abs(DEFAULT_GLOBAL_ROUTE_TRACKING_MARGIN_M) < 1.0e-9,
    }
    print("GLOBAL_ROUTE_TEST")
    for name, passed in checks.items():
        print("  {}: {}".format(name, passed))
    passed = all(checks.values())
    print("RESULT={}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
