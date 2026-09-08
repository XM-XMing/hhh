from __future__ import annotations

import numpy as np

from planning.diagnostics.temporal_observation_ranking import (
    TEMPORAL_VARIANTS,
    build_history_index,
    build_pair_feature_matrix,
    temporal_variant_dimensions,
)


def _raw_state(value: float, previous_action: int) -> dict[str, np.ndarray]:
    return {
        "vector": np.full(22, value, dtype=np.float32),
        "depth": np.full(72, value, dtype=np.float32),
        "mask": np.ones(105, dtype=np.float32),
        "previous_action": np.eye(105, dtype=np.float32)[previous_action],
    }


def test_temporal_variant_dimensions_are_explicit_and_ordered():
    assert TEMPORAL_VARIANTS == ("S0", "S1", "S2", "S3")
    dimensions = temporal_variant_dimensions()
    assert dimensions["S0"] == {"state_feature_dim": 304, "pair_input_dim": 514}
    assert dimensions["S1"] == {"state_feature_dim": 901, "pair_input_dim": 1111}
    assert dimensions["S2"] == {"state_feature_dim": 1299, "pair_input_dim": 1509}
    assert dimensions["S3"] == {"state_feature_dim": 1824, "pair_input_dim": 2034}


def test_history_index_is_past_only_and_zero_pads_episode_boundaries():
    rows = [
        {"state_id": "s0", "mission_id": "m", "source_episode_id": "e", "step_id": 0},
        {"state_id": "s1", "mission_id": "m", "source_episode_id": "e", "step_id": 1},
        {"state_id": "s2", "mission_id": "m", "source_episode_id": "e", "step_id": 2},
    ]
    history, coverage = build_history_index(rows, history_length=5)
    assert history["s2"] == ("s1", "s0", None, None, None)
    assert history["s0"] == (None, None, None, None, None)
    assert coverage["full_history_count"] == 0
    assert coverage["available_history_count"] == 2


def test_pair_matrix_has_fixed_shapes_and_excludes_future_state():
    rows = [
        {"state_id": "s0", "mission_id": "m", "source_episode_id": "e", "step_id": 0},
        {"state_id": "s1", "mission_id": "m", "source_episode_id": "e", "step_id": 1},
        {"state_id": "s2", "mission_id": "m", "source_episode_id": "e", "step_id": 2},
        {"state_id": "s3", "mission_id": "m", "source_episode_id": "e", "step_id": 3},
    ]
    history, _ = build_history_index(rows, history_length=5)
    states = {row["state_id"]: _raw_state(float(row["step_id"]), row["step_id"] + 1) for row in rows}
    pairs = [{"state_id": "s2", "action1": 4, "action2": 7, "label": 1, "mission_id": "m"}]
    matrix = build_pair_feature_matrix(pairs, states, history, "S3")
    assert matrix.shape == (1, 2034)
    # S3 contains s1 and s0, but no future s3.  The first current-state value
    # is 2, followed by the immediate-past base-state value 1.
    assert matrix[0, 0] == 2.0
    assert matrix[0, 304] == 1.0
    assert matrix[0, 304 + 199] == 0.0
