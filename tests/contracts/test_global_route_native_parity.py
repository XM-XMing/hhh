#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from planning.mission.global_route import (
    GlobalRouteConfig,
    GlobalRouteUnavailableError,
    _NativeGlobalRouteBackend,
    _PythonReferenceGlobalRoutePlanner2D,
)


def main() -> int:
    rng = np.random.default_rng(20260717)
    shape = np.asarray([36, 32, 8], dtype=np.int64)
    occupied = []
    for x in range(int(shape[0])):
        for y in range(int(shape[1])):
            if rng.random() < 0.07:
                for z in (2, 3, 4, 5):
                    occupied.append(
                        x * int(shape[1]) * int(shape[2])
                        + y * int(shape[2])
                        + z
                    )
    checker = SimpleNamespace(
        occupied_keys=np.asarray(sorted(occupied), dtype=np.int64),
        origin_ijk=np.asarray([-18, -16, 0], dtype=np.int64),
        grid_shape=shape,
        voxel_size=0.25,
        collision_radius=0.35,
    )
    config = GlobalRouteConfig()
    reference = _PythonReferenceGlobalRoutePlanner2D(checker, config)
    native = _NativeGlobalRouteBackend(checker, config)
    assert np.array_equal(reference.blocked, native.blocked)

    checked = 0
    for _ in range(40):
        start = np.asarray(
            [
                rng.uniform(-3.5, 3.5),
                rng.uniform(-3.0, 3.0),
                rng.uniform(1.0, 3.0),
            ],
            dtype=np.float32,
        )
        goal = np.asarray(
            [
                rng.uniform(-3.5, 3.5),
                rng.uniform(-3.0, 3.0),
                rng.uniform(1.0, 3.0),
            ],
            dtype=np.float32,
        )
        try:
            expected = reference.plan(start, goal)
        except GlobalRouteUnavailableError:
            try:
                native.plan(start, goal)
            except GlobalRouteUnavailableError:
                checked += 1
                continue
            raise AssertionError("native route availability differs from reference")
        actual = native.plan(start, goal)
        assert np.array_equal(actual, expected)
        checked += 1

    print("GLOBAL_ROUTE_NATIVE_PARITY")
    print("  cases:", checked)
    print("  blocked_grid_exact: True")
    print("  route_exact: True")
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
