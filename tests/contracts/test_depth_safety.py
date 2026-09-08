#!/usr/bin/env python3
"""Unit checks for local depth-only primitive collision screening."""

from __future__ import annotations




import numpy as np


from planning.safety.depth_safety import (
    DepthSafetyConfig, local_depth_action_mask, primitive_depth_projection_table,
)
from planning.primitives.library import MotionPrimitiveLibrary


def main() -> int:
    mpl = MotionPrimitiveLibrary()
    config = DepthSafetyConfig(collision_radius_m=0.35, path_sample_stride=4)
    clear = np.full((90, 160), 6.0, dtype=np.float32)
    clear_mask, clear_info = local_depth_action_mask(mpl, clear, config)
    if not bool(np.all(clear_mask)):
        raise AssertionError("empty depth image must not reject actions: {}".format(clear_info))

    blocked = clear.copy()
    blocked[38:53, 72:88] = 0.80
    blocked_mask, blocked_info = local_depth_action_mask(mpl, blocked, config)
    center_action = mpl.nearest_action_by_endpoint(0.0, 0.0)
    projection = primitive_depth_projection_table(mpl, 90, 160, config)
    if not bool(np.all(projection["valid"][center_action])):
        raise AssertionError("center primitive projection unexpectedly has padding")
    if not bool(np.any(
        (projection["pixel_x"][center_action] >= 72)
        & (projection["pixel_x"][center_action] < 88)
        & (projection["pixel_y"][center_action] >= 38)
        & (projection["pixel_y"][center_action] < 53)
    )):
        raise AssertionError("center primitive projection misses the forward obstacle patch")
    if bool(blocked_mask[center_action]) or int(blocked_info["depth_blocked_count"]) <= 0:
        raise AssertionError("forward obstacle was not screened: {}".format(blocked_info))

    max_unclipped = np.asarray(clear_info["depth_action_max_unclipped_patch_radius_px"])
    capped = np.asarray(clear_info["depth_action_capped_patch_sample_count"])
    if int(max_unclipped[center_action]) <= int(config.max_patch_radius_px):
        raise AssertionError("near-field projected radius was expected to exceed the configured cap")
    if int(capped[center_action]) <= 0:
        raise AssertionError("radius-cap diagnostics did not record capped samples")

    unknown = np.full((90, 160), 6.0, dtype=np.float32)
    unknown_mask, unknown_info = local_depth_action_mask(mpl, unknown, config)
    invalid_fraction = np.asarray(unknown_info["depth_action_mean_invalid_patch_fraction"])
    if not bool(np.all(unknown_mask)) or float(invalid_fraction[center_action]) < 0.999:
        raise AssertionError("current unknown-depth behavior or its diagnostics changed unexpectedly")

    print("DEPTH_SAFETY_TEST")
    print("  clear_valid_count:", int(clear_info["depth_valid_count"]))
    print("  blocked_valid_count:", int(blocked_info["depth_valid_count"]))
    print("  center_action:", int(center_action))
    print("  center_max_unclipped_patch_radius_px:", int(max_unclipped[center_action]))
    print("  center_capped_patch_samples:", int(capped[center_action]))
    print("  unknown_frame_valid_fraction:", float(unknown_info["depth_frame_valid_fraction"]))
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
