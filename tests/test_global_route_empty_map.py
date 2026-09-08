"""Public route-contract regression for an empty occupancy map."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from planning.mission.global_route import (
    FORMAL_GLOBAL_ROUTE_BACKEND,
    GlobalRoutePlanner2D,
)


pytestmark = pytest.mark.unit


def test_native_route_accepts_empty_occupancy_map():
    checker = SimpleNamespace(
        occupied_keys=np.empty((0,), dtype=np.int64),
        origin_ijk=np.asarray([0, 0, 1], dtype=np.int64),
        grid_shape=np.asarray([16, 16, 1], dtype=np.int64),
        voxel_size=1.0,
        collision_radius=0.0,
    )

    planner = GlobalRoutePlanner2D(checker)
    route = planner.plan(
        np.asarray([0.1, 0.1, 1.5], dtype=np.float32),
        np.asarray([10.1, 10.1, 2.5], dtype=np.float32),
    )

    assert planner.backend_name == FORMAL_GLOBAL_ROUTE_BACKEND
    assert route.dtype == np.float32
    assert route.shape[0] > 2
