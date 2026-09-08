#!/usr/bin/env python3
"""Regression test for descriptive expert action-distribution metrics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


from planning.diagnostics.action_distribution import summarize_action_histogram


def main() -> int:
    primitives = SimpleNamespace(
        num_actions=4,
        center_action_id=1,
        y_end=np.asarray([-0.2, 0.0, 0.2, 0.0]),
        z_end=np.asarray([0.0, 0.0, 0.0, 0.2]),
        terminal_heading_rad=np.asarray([-0.1, 0.0, 0.1, 0.0]),
        horizontal_index=np.asarray([0, 1, 2, 1]),
        vertical_index=np.asarray([1, 1, 1, 2]),
    )
    summary = summarize_action_histogram([2, 4, 2, 2], primitives)
    checks = {
        "total": summary["total_actions"] == 10,
        "coverage": summary["unique_actions"] == 4,
        "center_fraction": abs(summary["center_action_fraction"] - 0.4) < 1.0e-12,
        "lateral_fraction": abs(summary["lateral_action_fraction"] - 0.4) < 1.0e-12,
        "vertical_fraction": abs(summary["vertical_action_fraction"] - 0.2) < 1.0e-12,
        "bin_coverage": summary["used_horizontal_bins"] == 3
        and summary["used_vertical_bins"] == 2,
    }
    print("ACTION_DISTRIBUTION_AUDIT_TEST")
    for name, passed in checks.items():
        print("  {}: {}".format(name, passed))
    passed = all(checks.values())
    print("RESULT={}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
