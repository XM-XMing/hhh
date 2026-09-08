from __future__ import annotations

import numpy as np

from planning.diagnostics.memory_state_sufficiency import (
    _masked_order,
    build_history_tensors,
)


def test_history_is_prefix_only_and_zero_pads_missing_past():
    labels = [
        {"state_id": "s1", "mission_id": "m", "oracle_action": 0},
        {"state_id": "s2", "mission_id": "m", "oracle_action": 0},
    ]
    features = {"s1": np.ones(304, dtype=np.float32), "s2": np.full(304, 2.0, dtype=np.float32)}
    manifest = {
        "s1": {"mission_id": "m", "source_episode_id": "e", "state_index": 4, "episode_state_ids": {4: "s1"}},
        "s2": {"mission_id": "m", "source_episode_id": "e", "state_index": 5, "episode_state_ids": {4: "s1", 5: "s2"}},
    }
    history, valid, provenance = build_history_tensors(labels, features, manifest, sequence_length=3)
    assert valid.tolist() == [[False, False, True], [False, True, True]]
    assert np.all(history[0, 0] == 0.0)
    assert np.all(history[0, 2] == 1.0)
    assert np.all(history[1, 1] == 1.0)
    assert provenance["future_information_used"] is False


def test_masked_order_keeps_invalid_actions_out_of_top1():
    order = _masked_order(np.asarray([4.0, 3.0, 2.0], dtype=np.float32), np.asarray([False, True, False]))
    assert int(order[0]) == 1
