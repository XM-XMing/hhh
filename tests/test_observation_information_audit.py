from __future__ import annotations

import numpy as np

from planning.diagnostics.observation_information_audit import (
    build_feature_names,
    component_feature_dimensions,
    within_state_return_statistics,
)


def test_feature_names_are_unique_and_component_dimensions_are_explicit():
    names = build_feature_names()
    assert len(names) == len(set(names))
    dimensions = component_feature_dimensions()
    assert dimensions["vector"] == 22
    assert dimensions["depth"] == 72
    assert dimensions["vector_plus_depth"] == 94


def test_within_state_return_statistics_uses_state_clusters():
    rows = [
        {"state_id": "s0", "episode_return": 1.0},
        {"state_id": "s0", "episode_return": 3.0},
        {"state_id": "s1", "episode_return": 2.0},
        {"state_id": "s1", "episode_return": 2.0},
    ]
    result = within_state_return_statistics(rows)
    assert result["state_count"] == 2
    assert result["multi_action_state_count"] == 2
    assert result["variance_mean"] == 0.5
    assert result["variance_median"] == 0.5
    assert np.isclose(result["variance_p90"], 0.9)
