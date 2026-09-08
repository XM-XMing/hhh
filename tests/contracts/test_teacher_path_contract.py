#!/usr/bin/env python3
"""Regression test for expert plan/actual path-length thresholds."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


from planning.contracts.teacher_path import (
    TEACHER_ACTUAL_PATH_MAX_M,
    TEACHER_PATH_LENGTH_CONTRACT_ID,
    TEACHER_PLAN_PATH_MAX_M,
    actual_path_eligible,
    minimum_remaining_path_to_goal_region_m,
    observed_polyline_length_m,
    plan_path_eligible,
    primitive_reference_path_lengths_m,
    validate_plan_filtered_mission_rows,
)


def main() -> int:
    polyline = observed_polyline_length_m(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]]
    )
    fake_mpl = SimpleNamespace(
        pos_ref=np.asarray(
            [
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            ],
            dtype=np.float32,
        )
    )
    reference_lengths = primitive_reference_path_lengths_m(fake_mpl)
    valid_mission = {
        "teacher_path_length_contract_id": TEACHER_PATH_LENGTH_CONTRACT_ID,
        "teacher_plan_path_length_m": TEACHER_PLAN_PATH_MAX_M,
    }
    mission_contract_rejects_unfiltered = False
    try:
        validate_plan_filtered_mission_rows([{}])
    except ValueError:
        mission_contract_rejects_unfiltered = True
    validate_plan_filtered_mission_rows([valid_mission])
    remaining_outside_both = minimum_remaining_path_to_goal_region_m(
        [0.0, 0.0, 0.0], [4.2, 0.0, 1.2], 1.2, 0.2
    )
    remaining_inside_goal = minimum_remaining_path_to_goal_region_m(
        [3.5, 0.0, 1.1], [4.2, 0.0, 1.2], 1.2, 0.2
    )
    checks = {
        "contract_is_semantic": (
            TEACHER_PATH_LENGTH_CONTRACT_ID
            == "teacher_plan44_actual46_observed_polyline"
        ),
        "polyline_not_endpoint_chord": abs(polyline - 7.0) < 1.0e-9,
        "primitive_arc_lengths": np.allclose(reference_lengths, [2.0, 2.0]),
        "goal_region_distance_lower_bound": abs(
            remaining_outside_both - np.hypot(3.0, 1.0)
        ) < 1.0e-9,
        "inside_goal_region_distance_zero": remaining_inside_goal == 0.0,
        "plan_44m_inclusive": plan_path_eligible(TEACHER_PLAN_PATH_MAX_M),
        "plan_over_44m_rejected": not plan_path_eligible(
            TEACHER_PLAN_PATH_MAX_M + 0.01
        ),
        "actual_44_to_46m_accepted": actual_path_eligible(45.0),
        "actual_over_46m_rejected": not actual_path_eligible(
            TEACHER_ACTUAL_PATH_MAX_M + 0.01
        ),
        "unfiltered_mission_rejected": mission_contract_rejects_unfiltered,
    }
    print("TEACHER_PATH_CONTRACT_TEST")
    for name, passed in checks.items():
        print("  {}: {}".format(name, passed))
    passed = all(checks.values())
    print("RESULT={}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
