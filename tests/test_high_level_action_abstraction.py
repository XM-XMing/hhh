from __future__ import annotations

import numpy as np

from planning.diagnostics.high_level_action_abstraction import (
    _candidate_mappings,
    _geometry_kmeans,
    _mapping_payload,
)


def _raw_action(action_id: int, lateral: str, vertical: str) -> dict:
    return {
        "id": action_id,
        "lateral_mode": lateral,
        "vertical_mode": vertical,
        "lateral_endpoint_m": {"left": -0.2, "straight": 0.0, "right": 0.2}[lateral],
        "vertical_endpoint_m": {"down": -0.08, "level": 0.0, "up": 0.08}[vertical],
        "terminal_heading_deg": {"left": -10.0, "straight": 0.0, "right": 10.0}[lateral],
        "duration_s": 0.5,
        "command_frames": 25,
    }


def test_geometry_kmeans_is_deterministic_and_return_free():
    points = np.asarray([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.9, 1.0, 0.8]])
    first = _geometry_kmeans(points, cluster_count=2)
    second = _geometry_kmeans(points, cluster_count=2)
    assert np.array_equal(first[0], second[0])
    assert np.allclose(first[1], second[1])


def test_mapping_candidates_do_not_depend_on_outcome_data():
    actions = [_raw_action(index, "left" if index % 3 == 0 else "straight", "up" if index % 2 else "level") for index in range(105)]
    definitions = {
        index: {"descriptor": [float(action["lateral_endpoint_m"]), float(action["vertical_endpoint_m"]), float(action["terminal_heading_deg"]) ]}
        for index, action in enumerate(actions)
    }
    library = {"actions": actions, "definitions": definitions}
    first = _candidate_mappings(library)
    second = _candidate_mappings(library)
    assert first == second
    assert first["primitive_family"]["high_level_count"] == 1
    assert first["duration_bucket"]["high_level_count"] == 1
    assert first["motion_direction"]["high_level_count"] >= 2


def test_mapping_payload_contains_every_primitive_once():
    actions = [_raw_action(index, "straight", "level") for index in range(105)]
    definitions = {index: {"descriptor": [0.0, 0.0, 0.0]} for index in range(105)}
    library = {
        "actions": actions,
        "definitions": definitions,
        "json_path": "/tmp/motion.json",
        "npz_path": "/tmp/motion.npz",
        "json_sha256": "json",
        "npz_sha256": "npz",
        "meta": {"contract_sha256": "contract"},
    }
    payload = _mapping_payload(library, _candidate_mappings(library))
    assert len(payload["actions"]) == 105
    assert sorted(row["action_id"] for row in payload["actions"]) == list(range(105))

