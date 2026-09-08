#!/usr/bin/env python3
"""Test the MPL altitude mask against generated terminal z displacements."""

from __future__ import annotations

import numpy as np

from planning.primitives.library import MotionPrimitiveLibrary


def main() -> int:
    mpl = MotionPrimitiveLibrary()
    cases = [(1.1, "near_low"), (2.0, "middle"), (2.9, "near_high")]
    z_min, z_max, margin = 1.0, 3.0, 0.02
    all_match = True

    print("ACTION_MASK_TEST")
    for z, name in cases:
        mask = mpl.valid_action_mask(z, z_min=z_min, z_max=z_max, margin=margin)
        expected = ((z + mpl.z_end >= z_min + margin) & (z + mpl.z_end <= z_max - margin))
        matches = bool(np.array_equal(mask, expected))
        all_match &= matches
        invalid_ids = np.flatnonzero(~mask).tolist()
        print(
            "  {} z={:.2f}: valid={} invalid={} matches_geometry={} invalid_ids={}".format(
                name,
                z,
                int(np.count_nonzero(mask)),
                int(mask.size - np.count_nonzero(mask)),
                matches,
                invalid_ids[:30] if len(invalid_ids) > 30 else invalid_ids,
            )
        )

    mask_low = mpl.valid_action_mask(1.1, z_min=z_min, z_max=z_max, margin=margin)
    mask_mid = mpl.valid_action_mask(2.0, z_min=z_min, z_max=z_max, margin=margin)
    mask_high = mpl.valid_action_mask(2.9, z_min=z_min, z_max=z_max, margin=margin)
    descend_ids = np.flatnonzero(mpl.z_end < 0.0)
    ascend_ids = np.flatnonzero(mpl.z_end > 0.0)
    low_blocks_some_descents = bool(np.any(~mask_low[descend_ids]))
    middle_all_valid = bool(np.all(mask_mid))
    high_blocks_some_ascents = bool(np.any(~mask_high[ascend_ids]))

    print("  low_blocks_some_descents:", low_blocks_some_descents)
    print("  middle_all_valid:", middle_all_valid)
    print("  high_blocks_some_ascents:", high_blocks_some_ascents)
    ok = all_match and low_blocks_some_descents and middle_all_valid and high_blocks_some_ascents
    print("RESULT={}".format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
