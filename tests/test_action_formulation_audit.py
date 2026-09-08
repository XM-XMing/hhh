from __future__ import annotations

import numpy as np

from planning.diagnostics.action_formulation_audit import (
    build_action_definitions,
    build_pair_action_features,
    classify_action_pair,
)


def _action(action_id: int, horizontal_index: int, vertical_index: int) -> dict:
    return {
        "id": action_id,
        "horizontal_index": horizontal_index,
        "vertical_index": vertical_index,
        "lateral_endpoint_m": float(horizontal_index) * 0.1,
        "vertical_endpoint_m": float(vertical_index) * 0.02,
        "terminal_heading_deg": float(horizontal_index) * 2.0,
        "duration_s": 0.5,
        "command_frames": 25,
    }


def test_action_definitions_keep_semantic_parameters_and_grid_neighbors():
    ranges = {
        "lateral_endpoint_m": {"min": 0.0, "max": 0.2},
        "vertical_endpoint_m": {"min": 0.0, "max": 0.04},
        "terminal_heading_deg": {"min": 0.0, "max": 4.0},
    }
    definitions = build_action_definitions(
        [_action(0, 0, 0), _action(1, 0, 1), _action(2, 1, 0), _action(3, 1, 1)],
        parameter_ranges=ranges,
        horizontal_count=2,
        vertical_count=2,
    )
    assert definitions[0]["parameter_dim"] == 3
    assert np.allclose(definitions[0]["descriptor"], [-1.0, -1.0, -1.0])
    assert definitions[0]["neighbors_4"] == [1, 2]
    assert definitions[1]["neighbors_4"] == [0, 3]


def test_pair_features_support_one_hot_and_semantic_descriptor_modes():
    state = np.zeros((2, 304), dtype=np.float32)
    actions = {
        0: np.asarray([-1.0, -1.0, -1.0], dtype=np.float32),
        1: np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    }
    one_hot = build_pair_action_features(state, [0, 1], [1, 0], representation="one_hot")
    descriptor = build_pair_action_features(
        state, [0, 1], [1, 0], representation="primitive_descriptor", descriptors=actions
    )
    assert one_hot.shape == (2, 514)
    assert descriptor.shape == (2, 310)
    assert np.allclose(descriptor[0, 304:307], actions[0])
    assert np.allclose(descriptor[0, 307:310], actions[1])


def test_action_pair_classification_is_explicit():
    first = _action(0, 2, 3)
    second = _action(1, 2, 3)
    third = _action(2, 3, 3)
    assert classify_action_pair(first, second)["same_grid_cell"] is False
    assert classify_action_pair(first, second)["same_horizontal_family"] is True
    assert classify_action_pair(first, third)["same_vertical_family"] is True
