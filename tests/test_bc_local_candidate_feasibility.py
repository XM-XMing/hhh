from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from planning.diagnostics.bc_local_candidate_feasibility import (
    FAMILIES,
    _coverage_payload,
    _neighbor_orders,
    _random_control,
    build_candidate_set,
)


def _definitions():
    return {
        0: {"descriptor": [-1.0, -1.0, -1.0], "neighbors_8": [1, 2]},
        1: {"descriptor": [0.0, -1.0, -1.0], "neighbors_8": [0, 2]},
        2: {"descriptor": [1.0, -1.0, -1.0], "neighbors_8": [0, 1, 3]},
        3: {"descriptor": [1.0, 0.0, -1.0], "neighbors_8": [2]},
    }


def test_candidate_is_mask_valid_bc_first_and_nested():
    definitions = _definitions()
    orders = _neighbor_orders(definitions)
    valid = [0, 1, 2, 3]
    k0 = build_candidate_set(family="K0_BC_ONLY", bc_action=0, valid_actions=valid, neighbor_order=orders)
    k2 = build_candidate_set(family="K2_LOCAL", bc_action=0, valid_actions=valid, neighbor_order=orders)
    k4 = build_candidate_set(family="K4_LOCAL", bc_action=0, valid_actions=valid, neighbor_order=orders)
    k8 = build_candidate_set(family="K8_LOCAL", bc_action=0, valid_actions=valid, neighbor_order=orders)
    assert k0 == [0]
    assert k2[0] == k4[0] == k8[0] == 0
    assert set(k2).issubset(k4)
    assert set(k4).issubset(k8)
    assert len(k2) <= 3 and len(k4) <= 5 and len(k8) <= 9
    assert set(k2).issubset(valid)


def test_candidate_order_is_deterministic_and_full_valid_is_mask_only():
    definitions = _definitions()
    first = _neighbor_orders(definitions)
    second = _neighbor_orders(definitions)
    assert first == second
    assert build_candidate_set(family="FULL_VALID", bc_action=0, valid_actions=[0, 2], neighbor_order=first) == [0, 2]


def test_coverage_and_bootstrap_are_reproducible_and_tie_safe():
    rows = [
        {"mission_id": "m0", "oracle_covered": True},
        {"mission_id": "m0", "oracle_covered": False},
        {"mission_id": "m1", "oracle_covered": True},
    ]
    first = _coverage_payload(rows, "oracle_covered", seed=7, repeats=100)
    second = _coverage_payload(rows, "oracle_covered", seed=7, repeats=100)
    assert first == second
    assert first["state_micro_coverage"] == 2.0 / 3.0
    assert first["mission_macro_coverage"] == 0.75


def test_random_control_uses_exact_local_sizes_without_outcome_inputs():
    records = []
    for index in range(3):
        family_rows = {}
        for family, count in (("K2_LOCAL", 2), ("K4_LOCAL", 2), ("K8_LOCAL", 2), ("K16_LOCAL", 2)):
            family_rows[family] = {"oracle_covered": index == 0}
        records.append({
            "state_id": "s{}".format(index),
            "mission_id": "m{}".format(index % 2),
            "bc_action": 0,
            "valid_actions": [0, 1, 2, 3],
            "best_actions": [2],
            "candidates": {"K2_LOCAL": [0, 1, 2], "K4_LOCAL": [0, 1, 2], "K8_LOCAL": [0, 1, 2], "K16_LOCAL": [0, 1, 2]},
            "families": family_rows,
        })
    result = _random_control(records, repeats=10, seed=3, bootstrap_repeats=20)
    assert result["families"]["K2_LOCAL"]["candidate_size_match"] is True
    assert result["families"]["K2_LOCAL"]["local_minus_random_mission_bootstrap"]["repeats"] == 20
