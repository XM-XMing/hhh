from __future__ import annotations

import numpy as np

from planning.diagnostics.decision_state_completeness import (
    build_feature_inventory,
    build_missing_variable_inventory,
)


def test_feature_inventory_matches_policy_vector_contract():
    inventory = build_feature_inventory()
    assert inventory["continuous_vector_dim"] == 22
    assert inventory["policy_vector_dim"] == 127
    assert [row["index"] for row in inventory["state_fields"]] == list(range(22))
    assert inventory["additional_state_fields"][0]["shape"] == [90, 160]


def test_missing_variable_inventory_keeps_unknown_distinct_from_missing():
    variables = {row["name"]: row for row in build_missing_variable_inventory()["variables"]}
    assert variables["path_deviation"]["status"] == "MISSING"
    assert variables["depth_temporal_change"]["status"] == "MISSING"
    assert variables["obstacle_motion"]["status"] == "UNKNOWN"
    assert variables["goal_relative_state"]["status"] == "AVAILABLE"


def test_inventory_does_not_claim_future_or_privileged_fields():
    inventory = build_missing_variable_inventory()
    assert "future reward" in inventory["forbidden_additions"]
    assert "privileged route/global map" in inventory["forbidden_additions"]
