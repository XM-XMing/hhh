"""AWAC runtime/replay contracts at the training construction boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from planning.awac.replay import AWACReplayBuffer
from planning.runtime.reliable_training import (
    RELIABLE_V4_OBSERVATION_SEMANTICS,
    build_reliable_v4_runtime_config,
)


pytestmark = pytest.mark.unit


def _replay(path: Path) -> AWACReplayBuffer:
    return AWACReplayBuffer.create(
        path,
        capacity=8,
        depth_shape=(1, 2, 2),
        vector_dim=3,
        action_dim=5,
        training_config_sha256="a" * 64,
        observation_semantics=RELIABLE_V4_OBSERVATION_SEMANTICS,
        observation_contract=RELIABLE_V4_OBSERVATION_SEMANTICS,
        observation_source=RELIABLE_V4_OBSERVATION_SEMANTICS,
        bc_checkpoint_sha256="b" * 64,
        task_contract_id="xm_test_task",
        task_contract_sha256="t" * 64,
        mpl_contract_sha256="m" * 64,
        run_identity="awac-test-run",
        run_contract_sha256="r" * 64,
        reliable_v4_transition_count=0,
    )


def test_reliable_v4_runtime_config_is_explicit_and_serializable():
    config = build_reliable_v4_runtime_config(
        runtime_instance_id="worker-00-runtime-test",
        command_endpoint="tcp://127.0.0.1:10560",
        result_endpoint="tcp://127.0.0.1:10562",
        snapshot_endpoint="tcp://127.0.0.1:10564",
        timeout_s=10.0,
    )

    assert config["observation_semantics"] == RELIABLE_V4_OBSERVATION_SEMANTICS
    assert config["runtime_instance_id"] == "worker-00-runtime-test"
    assert config["command_endpoint"].endswith(":10560")
    assert config["result_endpoint"].endswith(":10562")
    assert config["snapshot_endpoint"].endswith(":10564")
    assert config["timeout_s"] == 10.0


def test_fresh_reliable_v4_awac_replay_records_zero_legacy_rows(tmp_path: Path):
    replay = _replay(tmp_path / "replay")

    assert replay.metadata["observation_semantics"] == (
        RELIABLE_V4_OBSERVATION_SEMANTICS
    )
    assert replay.metadata["legacy_replay_transition_count"] == 0
    assert replay.metadata["reliable_v4_transition_count"] == 0


def test_awac_replay_rejects_legacy_transition_fields(tmp_path: Path):
    replay = _replay(tmp_path / "replay")
    mask = np.ones((5,), dtype=np.bool_)
    transition = {
        "depth": np.zeros((1, 2, 2), dtype=np.float32),
        "vector": np.zeros((3,), dtype=np.float32),
        "action_mask": mask,
        "action": 0,
        "reward": 0.0,
        "next_depth": np.zeros((1, 2, 2), dtype=np.float32),
        "next_vector": np.zeros((3,), dtype=np.float32),
        "next_action_mask": mask,
        "done": True,
        "behavior_source": 0,
        "candidate_action": 1,
    }

    with pytest.raises(ValueError, match="unsupported fields"):
        replay.add(**transition)
