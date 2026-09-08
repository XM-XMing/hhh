from __future__ import annotations

import numpy as np

from planning.diagnostics.observation_augmentation_ranking import (
    VARIANTS,
    build_augmentation_features,
    build_pair_feature_matrix,
    build_previous_state_index,
    variant_dimensions,
)


def _state(value: float, previous_action: int) -> dict[str, np.ndarray]:
    previous = np.zeros(105, dtype=np.float32)
    if previous_action >= 0:
        previous[previous_action] = 1.0
    vector = np.arange(22, dtype=np.float32) + value
    vector[3:6] = np.asarray([value, value + 1.0, value + 2.0], dtype=np.float32)
    return {
        "vector": vector,
        "depth": np.full(72, value, dtype=np.float32),
        "mask": np.ones(105, dtype=np.float32),
        "previous_action": previous,
    }


def test_variants_have_fixed_dimensions_and_order():
    assert VARIANTS == ("S0", "S1", "S2", "S3", "S4")
    assert variant_dimensions() == {
        "S0": {"state_feature_dim": 304, "pair_input_dim": 514},
        "S1": {"state_feature_dim": 307, "pair_input_dim": 517},
        "S2": {"state_feature_dim": 376, "pair_input_dim": 586},
        "S3": {"state_feature_dim": 379, "pair_input_dim": 589},
        "S4": {"state_feature_dim": 380, "pair_input_dim": 590},
    }


def test_previous_join_is_strictly_past_and_budget_uses_current_step():
    rows = [
        {"state_id": "s0", "mission_id": "m", "source_episode_id": "e", "step_id": 0},
        {"state_id": "s1", "mission_id": "m", "source_episode_id": "e", "step_id": 1},
        {"state_id": "s2", "mission_id": "m", "source_episode_id": "e", "step_id": 2},
    ]
    previous, coverage = build_previous_state_index(rows, max_steps=45)
    assert previous == {"s0": None, "s1": "s0", "s2": "s1"}
    assert coverage["previous_state_available_count"] == 2
    states = {"s0": _state(0.0, -1), "s1": _state(1.0, 2), "s2": _state(2.0, 3)}
    features, _ = build_augmentation_features(states, rows, max_steps=45)
    np.testing.assert_allclose(features["s1"]["S1"][-3:], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(features["s1"]["S2"][-72:], np.ones(72, dtype=np.float32))
    np.testing.assert_allclose(features["s2"]["S4"][-1:], [(45.0 - 2.0) / 45.0])
    np.testing.assert_allclose(features["s0"]["S1"][-3:], np.zeros(3, dtype=np.float32))


def test_pair_matrix_keeps_same_pair_order_for_all_variants():
    rows = [
        {"state_id": "s1", "mission_id": "m", "step_id": 1, "action1": 4, "action2": 7, "higher_return_action": 4},
    ]
    manifest = [{"state_id": "s1", "mission_id": "m", "source_episode_id": "e", "step_id": 1}]
    states = {"s1": _state(1.0, 2)}
    features, _ = build_augmentation_features(states, manifest, max_steps=45)
    matrices = []
    for variant in VARIANTS:
        matrix, labels, state_ids, mission_ids = build_pair_feature_matrix(rows, features, variant)
        assert matrix.shape == (1, variant_dimensions()[variant]["pair_input_dim"])
        assert labels.tolist() == [1]
        assert state_ids.tolist() == ["s1"]
        assert mission_ids.tolist() == ["m"]
        matrices.append(matrix)
    np.testing.assert_allclose(matrices[0][0, :304], matrices[1][0, :304])
