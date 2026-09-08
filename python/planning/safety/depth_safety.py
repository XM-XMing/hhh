"""Local depth-only collision screening for motion primitives.

This module deliberately has no map, point-cloud, ROS, or Unity dependency.
It projects each body-frame MPL centerline into the current forward depth image
and rejects actions whose visible swept centerline is blocked.  It is a local
safety screen, not a substitute for a globally complete planner: regions
outside the camera frustum remain unknown rather than being treated as free.
"""

from __future__ import annotations
import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from weakref import WeakKeyDictionary
import numpy as np

from planning.native.loader import NativeLibraryError, load_native_library, require_symbol

_PROJECTION_CACHE = WeakKeyDictionary()
_NATIVE_BACKEND = None
_NATIVE_BACKEND_ERROR = None

FORMAL_DEPTH_SAFETY_BACKEND = "cpp_native"
PYTHON_REFERENCE_DEPTH_SAFETY_BACKEND = "python_reference"
DEPTH_SAFETY_BACKEND_ENV = "PLANNING_DEPTH_SAFETY_BACKEND"
DEPTH_SAFETY_REFERENCE_ENV = "PLANNING_DEPTH_SAFETY_REFERENCE"
_FORMAL_BACKEND_ALIASES = frozenset({"auto", "cpp", "native", "cpp_native"})
_REFERENCE_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def resolve_depth_safety_backend(requested: Optional[str] = None) -> str:
    """Resolve the formal backend without permitting an implicit fallback."""
    value = (
        os.environ.get(DEPTH_SAFETY_BACKEND_ENV, "")
        if requested is None
        else requested
    )
    value = str(value).strip().lower() or "auto"
    if value in _FORMAL_BACKEND_ALIASES:
        return FORMAL_DEPTH_SAFETY_BACKEND
    if value == "python":
        reference = os.environ.get(DEPTH_SAFETY_REFERENCE_ENV, "").strip().lower()
        if reference in _REFERENCE_TRUE_VALUES:
            return PYTHON_REFERENCE_DEPTH_SAFETY_BACKEND
        raise ValueError(
            "PYTHON_REFERENCE_DEPTH_SAFETY_REQUIRES_EXPLICIT_DEBUG: set "
            "PLANNING_DEPTH_SAFETY_REFERENCE=1 for test/reference use"
        )
    raise ValueError("UNSUPPORTED_DEPTH_SAFETY_BACKEND: {}".format(value))

class _NativeDepthSafety:
    CONTRACT_ID = "cpp_depth_patch"

    def __init__(self):
        try:
            self.library_path, self.lib = load_native_library("depth_safety")
            function = require_symbol(
                self.lib, "planning_depth_safety_mask", "DEPTH_SAFETY"
            )
        except NativeLibraryError as error:
            raise RuntimeError(str(error)) from error
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int64),
        ]
        function.restype = ctypes.c_int
        self.function = function

    @staticmethod
    def _pointer(array, scalar):
        return array.ctypes.data_as(ctypes.POINTER(scalar))

    def evaluate(self, image, projection, config):
        image = np.ascontiguousarray(image, dtype=np.float32)
        pixel_x = np.ascontiguousarray(projection["pixel_x"], dtype=np.int32)
        pixel_y = np.ascontiguousarray(projection["pixel_y"], dtype=np.int32)
        point_depth = np.ascontiguousarray(projection["point_depth_m"], dtype=np.float32)
        radius = np.ascontiguousarray(projection["patch_radius_px"], dtype=np.int32)
        raw_radius = np.ascontiguousarray(
            projection["unclipped_patch_radius_px"], dtype=np.int32
        )
        valid = np.ascontiguousarray(projection["valid"], dtype=np.uint8)
        action_count, sample_count = valid.shape
        mask = np.empty(action_count, dtype=np.uint8)
        checked = np.empty(action_count, dtype=np.int32)
        valid_patches = np.empty(action_count, dtype=np.int32)
        capped = np.empty(action_count, dtype=np.int32)
        max_radius = np.empty(action_count, dtype=np.int32)
        invalid_sum = np.empty(action_count, dtype=np.float64)
        min_clearance = np.empty(action_count, dtype=np.float32)
        counters = np.empty(3, dtype=np.int64)
        result = self.function(
            self._pointer(image, ctypes.c_float), image.shape[0], image.shape[1],
            self._pointer(pixel_x, ctypes.c_int32), self._pointer(pixel_y, ctypes.c_int32),
            self._pointer(point_depth, ctypes.c_float), self._pointer(radius, ctypes.c_int32),
            self._pointer(raw_radius, ctypes.c_int32), self._pointer(valid, ctypes.c_uint8),
            action_count, sample_count, float(config.valid_depth_min_m),
            float(config.valid_depth_max_m),
            float(config.collision_radius_m) + float(config.depth_slack_m),
            self._pointer(mask, ctypes.c_uint8), self._pointer(checked, ctypes.c_int32),
            self._pointer(valid_patches, ctypes.c_int32), self._pointer(capped, ctypes.c_int32),
            self._pointer(max_radius, ctypes.c_int32), self._pointer(invalid_sum, ctypes.c_double),
            self._pointer(min_clearance, ctypes.c_float), self._pointer(counters, ctypes.c_int64),
        )
        if result != 0:
            raise RuntimeError(
                "DEPTH_SAFETY_NATIVE_EXECUTION_FAILED: {}".format(result)
            )
        return mask.astype(np.bool_), checked, valid_patches, capped, max_radius, invalid_sum, min_clearance, counters

def _get_native_backend():
    global _NATIVE_BACKEND, _NATIVE_BACKEND_ERROR
    requested = resolve_depth_safety_backend()
    if requested == PYTHON_REFERENCE_DEPTH_SAFETY_BACKEND:
        return None
    if _NATIVE_BACKEND is None and _NATIVE_BACKEND_ERROR is None:
        try:
            _NATIVE_BACKEND = _NativeDepthSafety()
        except Exception as error:
            _NATIVE_BACKEND_ERROR = repr(error)
    if _NATIVE_BACKEND is None:
        raise RuntimeError(
            "DEPTH_SAFETY_NATIVE_UNAVAILABLE: {}".format(_NATIVE_BACKEND_ERROR)
        )
    return _NATIVE_BACKEND

@dataclass(frozen=True)
class DepthSafetyConfig:
    """Geometry and robustness settings for local depth screening."""

    horizontal_fov_deg: float = 87.0
    vertical_fov_deg: float = 58.0
    min_forward_m: float = 0.45
    path_sample_stride: int = 4
    collision_radius_m: float = 0.40
    depth_slack_m: float = 0.08
    patch_radius_px: int = 2
    max_patch_radius_px: int = 14
    valid_depth_min_m: float = 0.30
    valid_depth_max_m: float = 5.95

def camera_intrinsics_from_fov(width: int, height: int, horizontal_fov_deg: float, vertical_fov_deg: float) -> Dict[str, float]:
    """Return pinhole intrinsics for a depth image with the configured FOV."""
    if int(width) <= 1 or int(height) <= 1:
        raise ValueError("depth image must be at least 2x2")
    h_half = np.deg2rad(float(horizontal_fov_deg)) * 0.5
    v_half = np.deg2rad(float(vertical_fov_deg)) * 0.5
    return {
        "fx": float((float(width) - 1.0) / (2.0 * np.tan(h_half))),
        "fy": float((float(height) - 1.0) / (2.0 * np.tan(v_half))),
        "cx": float((float(width) - 1.0) * 0.5),
        "cy": float((float(height) - 1.0) * 0.5),
    }

def _sample_indices(length: int, stride: int) -> np.ndarray:
    indices = np.arange(0, int(length), max(1, int(stride)), dtype=np.int32)
    if indices.size == 0 or int(indices[-1]) != int(length) - 1:
        indices = np.append(indices, int(length) - 1)
    return indices

def _body_to_optical(points_body: np.ndarray) -> np.ndarray:
    """Convert base_link (forward,left,up) points to optical (right,down,forward)."""
    points = np.asarray(points_body, dtype=np.float32)
    return np.stack((-points[:, 1], -points[:, 2], points[:, 0]), axis=1)


def primitive_depth_projection_table(
    mpl: Any,
    height: int,
    width: int,
    config: DepthSafetyConfig = DepthSafetyConfig(),
    intrinsics: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Precompute the depth pixels and swept-radius samples for every action.

    This is the same body-to-camera geometry used by
    :func:`local_depth_action_mask`, exposed as a fixed table so a critic can
    consume continuous path clearance without introducing map information.
    """
    if intrinsics is None:
        intrinsics = camera_intrinsics_from_fov(
            int(width), int(height),
            config.horizontal_fov_deg, config.vertical_fov_deg,
        )
    fx, fy, cx, cy = (
        float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")
    )
    rows = []
    for action_id in range(int(mpl.num_actions)):
        path = np.asarray(mpl.reference_path(action_id), dtype=np.float32)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError("MPL reference path must be [T,3], got {}".format(path.shape))
        path = path[_sample_indices(path.shape[0], config.path_sample_stride)]
        path = path[path[:, 0] >= float(config.min_forward_m)]
        optical = _body_to_optical(path)
        forward = optical[:, 2]
        u = np.rint(fx * optical[:, 0] / forward + cx).astype(np.int32)
        v = np.rint(fy * optical[:, 1] / forward + cy).astype(np.int32)
        visible = (
            (forward > 1.0e-4) & (u >= 0) & (u < int(width))
            & (v >= 0) & (v < int(height))
        )
        forward, u, v = forward[visible], u[visible], v[visible]
        unclipped_radius = np.ceil(
            max(fx, fy) * float(config.collision_radius_m)
            / np.maximum(forward, 0.25)
        ).astype(np.int32)
        radius = np.minimum(
            int(config.max_patch_radius_px),
            np.maximum(
                int(config.patch_radius_px),
                unclipped_radius,
            ),
        ).astype(np.int32)
        rows.append((u, v, forward.astype(np.float32), radius, unclipped_radius))
    max_samples = max((len(row[0]) for row in rows), default=0)
    shape = (int(mpl.num_actions), int(max_samples))
    pixel_x = np.zeros(shape, dtype=np.int32)
    pixel_y = np.zeros(shape, dtype=np.int32)
    point_depth = np.zeros(shape, dtype=np.float32)
    patch_radius = np.zeros(shape, dtype=np.int32)
    unclipped_patch_radius = np.zeros(shape, dtype=np.int32)
    valid = np.zeros(shape, dtype=np.bool_)
    for action_id, (u, v, forward, radius, unclipped_radius) in enumerate(rows):
        count = len(u)
        pixel_x[action_id, :count] = u
        pixel_y[action_id, :count] = v
        point_depth[action_id, :count] = forward
        patch_radius[action_id, :count] = radius
        unclipped_patch_radius[action_id, :count] = unclipped_radius
        valid[action_id, :count] = True
    return {
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "point_depth_m": point_depth,
        "patch_radius_px": patch_radius,
        "unclipped_patch_radius_px": unclipped_patch_radius,
        "valid": valid,
    }

def cached_primitive_depth_projection_table(
    mpl: Any,
    height: int,
    width: int,
    config: DepthSafetyConfig = DepthSafetyConfig(),
    intrinsics: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Cache immutable MPL projection geometry per library/configuration."""

    if intrinsics is None:
        intrinsics = camera_intrinsics_from_fov(
            int(width), int(height),
            config.horizontal_fov_deg, config.vertical_fov_deg,
        )
    intrinsics_key = tuple(float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))
    key = (int(height), int(width), config, intrinsics_key)
    tables = _PROJECTION_CACHE.setdefault(mpl, {})
    if key not in tables:
        tables[key] = primitive_depth_projection_table(
            mpl, int(height), int(width), config=config, intrinsics=intrinsics,
        )
    return tables[key]

def local_depth_action_mask(
    mpl: Any,
    depth_m: np.ndarray,
    config: DepthSafetyConfig = DepthSafetyConfig(),
    intrinsics: Optional[Dict[str, float]] = None,
    projection: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return one collision-free bit per MPL action using only the current depth frame.

    A candidate sample is blocked when the closest valid depth in a small image
    patch lies in front of its planned center point plus the configured vehicle
    safety radius.  Invalid or out-of-range pixels do not create obstacles.
    """
    image = np.asarray(depth_m, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("depth_m must have shape [H,W], got {}".format(image.shape))
    height, width = image.shape
    if intrinsics is None:
        intrinsics = camera_intrinsics_from_fov(
            width, height, config.horizontal_fov_deg, config.vertical_fov_deg
        )
    fx, fy = (float(intrinsics[key]) for key in ("fx", "fy"))
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("depth intrinsics must have positive fx/fy")
    valid_depth = np.isfinite(image) & (image >= float(config.valid_depth_min_m)) & (image < float(config.valid_depth_max_m))
    if projection is None:
        projection = cached_primitive_depth_projection_table(
            mpl, height, width, config=config, intrinsics=intrinsics,
        )

    native = _get_native_backend()
    if native is not None:
        (
            mask,
            action_checked,
            action_valid_patches,
            action_capped_patches,
            action_max_unclipped_radius,
            action_invalid_fraction_sum,
            action_min_ray_clearance,
            counters,
        ) = native.evaluate(image, projection, config)
        action_mean_invalid_fraction = np.divide(
            action_invalid_fraction_sum,
            action_checked,
            out=np.full_like(action_invalid_fraction_sum, np.nan),
            where=action_checked > 0,
        ).astype(np.float32)
        return mask, {
            "depth_safety_backend_contract_id": native.CONTRACT_ID,
            "depth_checked_sample_count": int(counters[0]),
            "depth_visible_sample_count": int(counters[1]),
            "depth_blocked_sample_count": int(counters[2]),
            "depth_valid_count": int(mask.sum()),
            "depth_blocked_count": int((~mask).sum()),
            "depth_frame_valid_fraction": float(np.mean(valid_depth)),
            "depth_action_checked_sample_count": action_checked,
            "depth_action_valid_patch_sample_count": action_valid_patches,
            "depth_action_capped_patch_sample_count": action_capped_patches,
            "depth_action_max_unclipped_patch_radius_px": action_max_unclipped_radius,
            "depth_action_mean_invalid_patch_fraction": action_mean_invalid_fraction,
            "depth_action_min_ray_clearance_m": action_min_ray_clearance,
        }

    mask = np.ones((int(mpl.num_actions),), dtype=np.bool_)
    action_checked = np.zeros((int(mpl.num_actions),), dtype=np.int32)
    action_valid_patches = np.zeros((int(mpl.num_actions),), dtype=np.int32)
    action_capped_patches = np.zeros((int(mpl.num_actions),), dtype=np.int32)
    action_max_unclipped_radius = np.zeros((int(mpl.num_actions),), dtype=np.int32)
    action_invalid_fraction_sum = np.zeros((int(mpl.num_actions),), dtype=np.float64)
    action_min_ray_clearance = np.full((int(mpl.num_actions),), np.nan, dtype=np.float32)
    checked_samples = 0
    visible_samples = 0
    blocked_samples = 0
    for action_id in range(int(mpl.num_actions)):
        valid_samples = np.flatnonzero(projection["valid"][action_id])
        for sample_id in valid_samples:
            point_depth = float(projection["point_depth_m"][action_id, sample_id])
            pixel_x = int(projection["pixel_x"][action_id, sample_id])
            pixel_y = int(projection["pixel_y"][action_id, sample_id])
            checked_samples += 1
            action_checked[action_id] += 1
            unclipped_radius = int(
                projection["unclipped_patch_radius_px"][action_id, sample_id]
            )
            action_max_unclipped_radius[action_id] = max(
                int(action_max_unclipped_radius[action_id]), unclipped_radius
            )
            radius = int(projection["patch_radius_px"][action_id, sample_id])
            if radius < unclipped_radius:
                action_capped_patches[action_id] += 1
            x0, x1 = max(0, int(pixel_x) - radius), min(width, int(pixel_x) + radius + 1)
            y0, y1 = max(0, int(pixel_y) - radius), min(height, int(pixel_y) + radius + 1)
            patch_valid = valid_depth[y0:y1, x0:x1]
            action_invalid_fraction_sum[action_id] += 1.0 - float(np.mean(patch_valid))
            values = image[y0:y1, x0:x1][patch_valid]
            if values.size == 0:
                continue
            visible_samples += 1
            action_valid_patches[action_id] += 1
            ray_clearance = float(np.min(values)) - float(point_depth)
            if not np.isfinite(action_min_ray_clearance[action_id]):
                action_min_ray_clearance[action_id] = ray_clearance
            else:
                action_min_ray_clearance[action_id] = min(
                    float(action_min_ray_clearance[action_id]), ray_clearance
                )
            if ray_clearance <= float(config.collision_radius_m) + float(config.depth_slack_m):
                mask[action_id] = False
                blocked_samples += 1
                break

    action_mean_invalid_fraction = np.divide(
        action_invalid_fraction_sum,
        action_checked,
        out=np.full_like(action_invalid_fraction_sum, np.nan),
        where=action_checked > 0,
    ).astype(np.float32)
    return mask, {
        "depth_safety_backend_contract_id": "numpy_depth_patch",
        "depth_checked_sample_count": int(checked_samples),
        "depth_visible_sample_count": int(visible_samples),
        "depth_blocked_sample_count": int(blocked_samples),
        "depth_valid_count": int(mask.sum()),
        "depth_blocked_count": int((~mask).sum()),
        "depth_frame_valid_fraction": float(np.mean(valid_depth)),
        "depth_action_checked_sample_count": action_checked,
        "depth_action_valid_patch_sample_count": action_valid_patches,
        "depth_action_capped_patch_sample_count": action_capped_patches,
        "depth_action_max_unclipped_patch_radius_px": action_max_unclipped_radius,
        "depth_action_mean_invalid_patch_fraction": action_mean_invalid_fraction,
        "depth_action_min_ray_clearance_m": action_min_ray_clearance,
    }
