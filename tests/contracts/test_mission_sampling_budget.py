#!/usr/bin/env python3
"""Regression checks for scale-safe mission sampling attempt budgets."""

from __future__ import annotations



from planning.mission.sampling import resolve_sampling_max_attempts


def main() -> int:
    assert resolve_sampling_max_attempts(200) == 200000
    assert resolve_sampling_max_attempts(184000) == 736000
    assert resolve_sampling_max_attempts(184000, 500000) == 500000

    rejected_impossible_budget = False
    try:
        resolve_sampling_max_attempts(100, 99)
    except ValueError:
        rejected_impossible_budget = True
    assert rejected_impossible_budget

    print("MISSION_SAMPLING_BUDGET_TEST")
    print("  legacy_small_run_floor: True")
    print("  budget_scales_with_candidate_count: True")
    print("  explicit_override_preserved: True")
    print("  impossible_budget_rejected: True")
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
