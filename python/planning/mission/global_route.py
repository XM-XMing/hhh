"""Privileged coarse global routing for the online teacher.

The learned policy never receives this route.  It is built from the global
voxel map solely to keep teacher rollouts from entering local dead ends.
"""

from __future__ import annotations
import ctypes  # compatibility namespace for legacy tests; loading lives in planning.native
import heapq
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from planning.native.route_backend import _NativeGlobalRouteBackend
from planning.common.config import parse_bool
from planning.native.loader import resolve_global_route_library

GLOBAL_ROUTE_CONTRACT_ID = "coarse_xy_astar_025m_lookahead3m_flight_band"
FORMAL_GLOBAL_ROUTE_BACKEND = "cpp_native"
PYTHON_REFERENCE_GLOBAL_ROUTE_BACKEND = "python_reference"
GLOBAL_ROUTE_SOURCE_ID = "planning_global_route_cpp17"
GLOBAL_ROUTE_SOURCE_FILES = (
    "include/planning/geometry/voxel_map.hpp",
    "src/geometry/voxel_map.cpp",
    "include/planning/navigation/global_route_planner.hpp",
    "src/navigation/global_route_planner.cpp",
    "python/planning/mission/global_route.py",
)
GLOBAL_ROUTE_BACKEND_ENV = "PLANNING_GLOBAL_ROUTE_BACKEND"
GLOBAL_ROUTE_REFERENCE_ENV = "PLANNING_GLOBAL_ROUTE_REFERENCE"
DEFAULT_GLOBAL_ROUTE_RESOLUTION_M = 0.25
DEFAULT_GLOBAL_ROUTE_LOOKAHEAD_M = 3.0
DEFAULT_GLOBAL_ROUTE_TRACKING_MARGIN_M = 0.0
DEFAULT_MAX_ROUTE_STRETCH = 2.0

_FORMAL_GLOBAL_ROUTE_ALIASES = frozenset(
    {"auto", "cpp", "native", "cpp_native"}
)


class GlobalRouteUnavailableError(RuntimeError):
    """A mission has no route under the configured static-map contract."""


def _find_global_route_library() -> Optional[Path]:
    """Compatibility seam; discovery is implemented by ``native.loader``."""

    return resolve_global_route_library()


@dataclass(frozen=True)
class GlobalRouteConfig:
    resolution_m: float = DEFAULT_GLOBAL_ROUTE_RESOLUTION_M
    flight_z_min_m: float = 1.0
    flight_z_max_m: float = 3.0
    lookahead_m: float = DEFAULT_GLOBAL_ROUTE_LOOKAHEAD_M
    tracking_margin_m: float = DEFAULT_GLOBAL_ROUTE_TRACKING_MARGIN_M
    nearest_free_radius_cells: int = 6


def resolve_global_route_backend(requested: Optional[str] = None) -> str:
    """Resolve the one production route backend without implicit fallback."""

    raw = (
        os.environ.get(GLOBAL_ROUTE_BACKEND_ENV, "cpp")
        if requested is None
        else requested
    )
    backend = str(raw).strip().lower()
    if not backend:
        backend = "cpp"
    if backend in _FORMAL_GLOBAL_ROUTE_ALIASES:
        return FORMAL_GLOBAL_ROUTE_BACKEND
    if backend == "python":
        reference_enabled = parse_bool(
            os.environ.get(GLOBAL_ROUTE_REFERENCE_ENV, "false")
        )
        if not reference_enabled:
            raise ValueError(
                "PYTHON_REFERENCE_GLOBAL_ROUTE_REQUIRES_EXPLICIT_DEBUG: "
                "set {}=1".format(GLOBAL_ROUTE_REFERENCE_ENV)
            )
        return PYTHON_REFERENCE_GLOBAL_ROUTE_BACKEND
    raise ValueError("UNSUPPORTED_GLOBAL_ROUTE_BACKEND: {}".format(backend))


def global_route_identity(config: Optional[GlobalRouteConfig] = None) -> Dict[str, object]:
    """Return the persisted production route source/config manifest."""

    resolved = config or GlobalRouteConfig()
    return {
        "global_route_backend": FORMAL_GLOBAL_ROUTE_BACKEND,
        "global_route_contract_id": GLOBAL_ROUTE_CONTRACT_ID,
        "global_route_source_id": GLOBAL_ROUTE_SOURCE_ID,
        "global_route_source_files": list(GLOBAL_ROUTE_SOURCE_FILES),
        "global_route_config": {
            "resolution_m": float(resolved.resolution_m),
            "flight_z_min_m": float(resolved.flight_z_min_m),
            "flight_z_max_m": float(resolved.flight_z_max_m),
            "lookahead_m": float(resolved.lookahead_m),
            "tracking_margin_m": float(resolved.tracking_margin_m),
            "nearest_free_radius_cells": int(resolved.nearest_free_radius_cells),
        },
    }

class _PythonReferenceGlobalRoutePlanner2D:
    """A* on a collision-inflated XY projection of the flight band."""

    def __init__(self, checker, config: GlobalRouteConfig = GlobalRouteConfig()):
        self.config = config
        self.resolution_m = float(config.resolution_m)
        if self.resolution_m <= 0.0:
            raise ValueError("global route resolution must be positive")
        lower = np.asarray(checker.origin_ijk, dtype=np.float64) * float(checker.voxel_size)
        upper = (
            np.asarray(checker.origin_ijk, dtype=np.float64)
            + np.asarray(checker.grid_shape, dtype=np.float64)
        ) * float(checker.voxel_size)
        self.origin_xy = np.floor(lower[:2] / self.resolution_m) * self.resolution_m
        self.shape = np.ceil((upper[:2] - self.origin_xy) / self.resolution_m).astype(np.int32)
        if np.any(self.shape <= 0):
            raise ValueError("invalid global route grid shape")

        keys = np.asarray(checker.occupied_keys, dtype=np.int64)
        grid_shape = np.asarray(checker.grid_shape, dtype=np.int64)
        rel_z = keys % grid_shape[2]
        world_z = (rel_z + int(checker.origin_ijk[2]) + 0.5) * float(checker.voxel_size)
        keep = (
            (world_z >= float(config.flight_z_min_m) - float(checker.collision_radius))
            & (world_z <= float(config.flight_z_max_m) + float(checker.collision_radius))
        )
        selected = keys[keep]
        stride_x = int(grid_shape[1] * grid_shape[2])
        rel_x = selected // stride_x
        rel_y = (selected % stride_x) // grid_shape[2]
        world_x = (rel_x + int(checker.origin_ijk[0]) + 0.5) * float(checker.voxel_size)
        world_y = (rel_y + int(checker.origin_ijk[1]) + 0.5) * float(checker.voxel_size)
        coarse_x = np.floor((world_x - self.origin_xy[0]) / self.resolution_m).astype(np.int32)
        coarse_y = np.floor((world_y - self.origin_xy[1]) / self.resolution_m).astype(np.int32)
        inside = (
            (coarse_x >= 0) & (coarse_x < self.shape[0])
            & (coarse_y >= 0) & (coarse_y < self.shape[1])
        )
        occupied = np.zeros(tuple(int(value) for value in self.shape), dtype=np.bool_)
        occupied[coarse_x[inside], coarse_y[inside]] = True

        # Unity endpoint tracking error is a measured execution uncertainty,
        # not part of the static collision body radius. Reserve it only in
        # global guidance; primitive-level collision checks remain exact.
        inflation_m = float(checker.collision_radius) + float(config.tracking_margin_m)
        radius_cells = int(math.ceil(inflation_m / self.resolution_m))
        self.blocked = self._dilate(occupied, radius_cells)

    @staticmethod
    def _dilate(occupied: np.ndarray, radius_cells: int) -> np.ndarray:
        if int(radius_cells) <= 0:
            return occupied.copy()
        output = occupied.copy()
        nx, ny = occupied.shape
        for dx in range(-int(radius_cells), int(radius_cells) + 1):
            for dy in range(-int(radius_cells), int(radius_cells) + 1):
                if dx * dx + dy * dy > int(radius_cells) * int(radius_cells):
                    continue
                src_x0, src_x1 = max(0, -dx), min(nx, nx - dx)
                src_y0, src_y1 = max(0, -dy), min(ny, ny - dy)
                dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
                dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
                output[dst_x0:dst_x1, dst_y0:dst_y1] |= occupied[src_x0:src_x1, src_y0:src_y1]
        return output

    def world_to_cell(self, xy: Sequence[float]) -> tuple[int, int]:
        cell = np.floor((np.asarray(xy, dtype=np.float64)[:2] - self.origin_xy) / self.resolution_m).astype(np.int32)
        return int(cell[0]), int(cell[1])

    def cell_to_world(self, cell: Sequence[int]) -> np.ndarray:
        return self.origin_xy + (np.asarray(cell, dtype=np.float32)[:2] + 0.5) * self.resolution_m

    def _nearest_free(self, cell: tuple[int, int]) -> tuple[int, int]:
        x0, y0 = cell
        best = None
        best_distance = float("inf")
        radius = int(self.config.nearest_free_radius_cells)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = x0 + dx, y0 + dy
                if x < 0 or y < 0 or x >= self.blocked.shape[0] or y >= self.blocked.shape[1]:
                    continue
                if self.blocked[x, y]:
                    continue
                distance = dx * dx + dy * dy
                if distance < best_distance:
                    best, best_distance = (x, y), distance
        if best is None:
            raise GlobalRouteUnavailableError(
                "no free global-route cell near {}".format(cell)
            )
        return best

    def plan(self, start: Sequence[float], goal: Sequence[float]) -> np.ndarray:
        start_cell = self._nearest_free(self.world_to_cell(start))
        goal_cell = self._nearest_free(self.world_to_cell(goal))
        nx, ny = self.blocked.shape
        start_id = start_cell[0] * ny + start_cell[1]
        goal_id = goal_cell[0] * ny + goal_cell[1]
        costs = np.full((nx * ny,), np.inf, dtype=np.float32)
        parents = np.full((nx * ny,), -1, dtype=np.int32)
        closed = np.zeros((nx * ny,), dtype=np.bool_)
        costs[start_id] = 0.0
        queue = [(0.0, start_id)]
        neighbors = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                     (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
                     (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)))
        while queue:
            _, current_id = heapq.heappop(queue)
            if closed[current_id]:
                continue
            closed[current_id] = True
            if current_id == goal_id:
                break
            x, y = divmod(current_id, ny)
            for dx, dy, step_cost in neighbors:
                xx, yy = x + dx, y + dy
                if xx < 0 or yy < 0 or xx >= nx or yy >= ny or self.blocked[xx, yy]:
                    continue
                if dx and dy and (self.blocked[x + dx, y] or self.blocked[x, y + dy]):
                    continue
                neighbor_id = xx * ny + yy
                candidate = float(costs[current_id]) + float(step_cost)
                if candidate >= float(costs[neighbor_id]):
                    continue
                costs[neighbor_id] = candidate
                parents[neighbor_id] = current_id
                heuristic = math.hypot(goal_cell[0] - xx, goal_cell[1] - yy)
                heapq.heappush(queue, (candidate + heuristic, neighbor_id))
        if not closed[goal_id]:
            raise GlobalRouteUnavailableError(
                "no global route from {} to {}".format(start_cell, goal_cell)
            )
        cells = []
        current = goal_id
        while current >= 0:
            cells.append(divmod(current, ny))
            if current == start_id:
                break
            current = int(parents[current])
        cells.reverse()
        xy = np.asarray([self.cell_to_world(cell) for cell in cells], dtype=np.float32)
        route = np.empty((xy.shape[0] + 2, 3), dtype=np.float32)
        route[0] = np.asarray(start, dtype=np.float32)[:3]
        route[1:-1, :2] = xy
        route[1:-1, 2] = np.linspace(float(start[2]), float(goal[2]), xy.shape[0], dtype=np.float32)
        route[-1] = np.asarray(goal, dtype=np.float32)[:3]
        return route

    def guidance_goal(self, route: np.ndarray, position: Sequence[float]) -> np.ndarray:
        points = np.asarray(route, dtype=np.float32)
        position_array = np.asarray(position, dtype=np.float32)[:3]
        nearest = int(np.argmin(np.linalg.norm(points[:, :2] - position_array[:2], axis=1)))
        travelled = 0.0
        target = nearest
        while target + 1 < points.shape[0] and travelled < float(self.config.lookahead_m):
            travelled += float(np.linalg.norm(points[target + 1, :2] - points[target, :2]))
            target += 1
        return points[target].copy()


class GlobalRoutePlanner2D:
    """Stable Python API backed by the production C++17 A* planner."""

    def __init__(
        self,
        checker,
        config: GlobalRouteConfig = GlobalRouteConfig(),
        _native_map_handle=None,
        _native_geometry_context=None,
    ):
        self._pid = os.getpid()
        self.config = config
        self.native_geometry_context = (
            _native_geometry_context
            if _native_geometry_context is not None
            else getattr(checker, "native_geometry_context", None)
        )
        if _native_map_handle is None and self.native_geometry_context is not None:
            _native_map_handle = self.native_geometry_context._map.handle
        self._closed = False
        backend = resolve_global_route_backend()
        if backend == PYTHON_REFERENCE_GLOBAL_ROUTE_BACKEND:
            self._backend = _PythonReferenceGlobalRoutePlanner2D(checker, config)
            self.backend_name = PYTHON_REFERENCE_GLOBAL_ROUTE_BACKEND
        else:
            self._backend = _NativeGlobalRouteBackend(
                checker,
                config,
                native_map_handle=_native_map_handle,
                native_geometry_context=self.native_geometry_context,
                library_path=_find_global_route_library(),
            )
            self.backend_name = FORMAL_GLOBAL_ROUTE_BACKEND
        self.resolution_m = float(self._backend.resolution_m)
        self.origin_xy = np.asarray(self._backend.origin_xy, dtype=np.float64)
        self.shape = np.asarray(self._backend.shape, dtype=np.int32)
        self.blocked = np.asarray(self._backend.blocked, dtype=np.bool_)

    @property
    def native_map_id(self) -> Optional[int]:
        return getattr(self._backend, "native_map_id", None)

    @property
    def native_planner_id(self) -> Optional[int]:
        return getattr(self._backend, "native_planner_id", None)

    def _assert_open(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError(
                "NATIVE_CONTEXT_FORK_REUSE_FORBIDDEN: global route planner"
            )
        if self._closed:
            raise RuntimeError("global route planner is closed")
        if self.native_geometry_context is not None:
            try:
                self.native_geometry_context._assert_open()
            except Exception as error:
                raise RuntimeError(str(error)) from error

    def world_to_cell(self, xy: Sequence[float]) -> tuple[int, int]:
        self._assert_open()
        cell = np.floor(
            (np.asarray(xy, dtype=np.float64)[:2] - self.origin_xy) / self.resolution_m
        ).astype(np.int32)
        return int(cell[0]), int(cell[1])

    def cell_to_world(self, cell: Sequence[int]) -> np.ndarray:
        self._assert_open()
        return self.origin_xy + (
            np.asarray(cell, dtype=np.float32)[:2] + 0.5
        ) * self.resolution_m

    def plan(self, start: Sequence[float], goal: Sequence[float]) -> np.ndarray:
        self._assert_open()
        return self._backend.plan(start, goal)

    def guidance_goal(self, route: np.ndarray, position: Sequence[float]) -> np.ndarray:
        self._assert_open()
        points = np.asarray(route, dtype=np.float32)
        position_array = np.asarray(position, dtype=np.float32)[:3]
        nearest = int(np.argmin(np.linalg.norm(points[:, :2] - position_array[:2], axis=1)))
        travelled = 0.0
        target = nearest
        while target + 1 < points.shape[0] and travelled < float(self.config.lookahead_m):
            travelled += float(np.linalg.norm(points[target + 1, :2] - points[target, :2]))
            target += 1
        return points[target].copy()

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self._backend, "close", None)
        if close is not None:
            close()
        self._closed = True

    def __enter__(self):
        self._assert_open()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
