from __future__ import annotations

import numpy as np

from planning.diagnostics.state_sufficiency import (
    MISSION_SPLIT_SCHEMA_ID,
    _state_feature,
    binary_metrics,
    build_mission_split,
    mission_cluster_bootstrap,
)


def test_mission_split_keeps_all_rows_of_a_mission_together():
    missions = {
        "m0": {"state_ids": ["s0", "s1"], "transition_count": 4, "pair_count": 3},
        "m1": {"state_ids": ["s2"], "transition_count": 2, "pair_count": 1},
        "m2": {"state_ids": ["s3", "s4"], "transition_count": 5, "pair_count": 4},
    }
    split = build_mission_split(missions, seed=17, fold_count=3)
    assert split["schema_id"] == MISSION_SPLIT_SCHEMA_ID
    assert set(split["mission_to_fold"]) == set(missions)
    for mission_id, item in missions.items():
        fold = split["mission_to_fold"][mission_id]
        assert all(split["state_to_fold"][state_id] == fold for state_id in item["state_ids"])


def test_binary_metrics_and_cluster_bootstrap_are_deterministic():
    metrics = binary_metrics(
        np.asarray([0, 1, 1, 0]),
        np.asarray([0.1, 0.9, 0.6, 0.4]),
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["f1"] == 1.0

    values = {"m0": [1.0, 0.0], "m1": [0.5, 0.5], "m2": [0.0, 1.0]}
    first = mission_cluster_bootstrap(values, repeats=100, seed=23)
    second = mission_cluster_bootstrap(values, repeats=100, seed=23)
    assert first == second
    assert first["repeats"] == 100
    assert 0.0 <= first["ci95"][0] <= first["estimate"] <= first["ci95"][1] <= 1.0


def test_state_feature_accepts_no_previous_action_sentinel():
    state = {
        "vector": np.zeros(22, dtype=np.float32),
        "mask": np.ones(105, dtype=np.bool_),
        "depth": np.ones((90, 160), dtype=np.float32),
        "previous_action": np.asarray([-1], dtype=np.int64),
    }
    feature = _state_feature(state)
    assert feature.shape == (304,)
    assert np.all(feature[-105:] == 0.0)
