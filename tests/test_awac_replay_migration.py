"""Replay public contract for the AWAC-only migration."""

from __future__ import annotations

import numpy as np
import pytest

from planning.awac.contract import AWAC_REPLAY_CONTRACT_ID
from planning.awac.interaction import BehaviorSource
from planning.awac.replay import AWACReplayBuffer, _FIELD_SPECS
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT


def _create(tmp_path):
    return AWACReplayBuffer.create(
        tmp_path / "replay",
        capacity=4,
        depth_shape=(1, 4, 4),
        vector_dim=3,
        action_dim=2,
        bc_checkpoint_sha256="a" * 64,
        task_contract_id="task",
        task_contract_sha256="b" * 64,
        mpl_contract_sha256="c" * 64,
        run_identity="run",
        run_contract_sha256="d" * 64,
    )


def _transition():
    return {
        "depth": np.zeros((1, 4, 4), dtype=np.float32),
        "vector": np.zeros(3, dtype=np.float32),
        "action_mask": np.array([1, 1], dtype=np.bool_),
        "action": 1,
        "reward": 1.0,
        "next_depth": np.ones((1, 4, 4), dtype=np.float32),
        "next_vector": np.ones(3, dtype=np.float32),
        "next_action_mask": np.array([1, 0], dtype=np.bool_),
        "done": False,
        "behavior_source": int(BehaviorSource.ACCEPTED_COLLECTION_POLICY),
    }


@pytest.mark.unit
def test_awac_replay_has_only_core_transition_fields(tmp_path):
    replay = _create(tmp_path)
    assert set(_FIELD_SPECS) == {
        "depth", "vector", "action_mask", "action", "reward",
        "next_depth", "next_vector", "next_action_mask", "done",
        "behavior_source",
    }
    replay.add(**_transition())
    replay.flush()
    reopened = AWACReplayBuffer.open(tmp_path / "replay")
    assert reopened.size == 1
    assert reopened.metadata["contract_id"] == AWAC_REPLAY_CONTRACT_ID
    assert reopened.metadata["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT


@pytest.mark.unit
def test_awac_replay_rejects_old_candidate_fields(tmp_path):
    replay = _create(tmp_path)
    transition = _transition()
    transition["candidate_action"] = 0
    with pytest.raises(ValueError, match="unsupported fields"):
        replay.add(**transition)
