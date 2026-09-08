"""CPU-only transactional batch tests for the persistent AWAC replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from planning.awac.interaction import BehaviorSource
from planning.awac.replay import AWACReplayBuffer


pytestmark = pytest.mark.unit


def _create_replay(path: Path, capacity: int = 4) -> AWACReplayBuffer:
    return AWACReplayBuffer.create(
        path,
        capacity=capacity,
        depth_shape=(1, 2, 2),
        vector_dim=3,
        action_dim=5,
        training_config_sha256="a" * 64,
        bc_checkpoint_sha256="b" * 64,
        task_contract_id="task",
        task_contract_sha256="c" * 64,
        mpl_contract_sha256="d" * 64,
        run_identity="awac-test",
        run_contract_sha256="e" * 64,
    )


def _transition(index: int, *, action: int = 1):
    mask = np.asarray([True, True, True, True, True], dtype=np.bool_)
    return {
        "depth": np.full((1, 2, 2), index / 10.0, dtype=np.float32),
        "vector": np.asarray([index, index + 1, index + 2], dtype=np.float32),
        "action_mask": mask,
        "action": int(action),
        "reward": float(index),
        "next_depth": np.full(
            (1, 2, 2), (index + 1) / 10.0, dtype=np.float32
        ),
        "next_vector": np.asarray(
            [index + 1, index + 2, index + 3], dtype=np.float32
        ),
        "next_action_mask": mask.copy(),
        "done": False,
        "behavior_source": int(BehaviorSource.ACCEPTED_COLLECTION_POLICY),
    }


def _snapshot(replay: AWACReplayBuffer):
    return {
        "arrays": {
            name: np.array(array, copy=True)
            for name, array in replay.arrays.items()
        },
        "size": replay.size,
        "position": replay.position,
        "total_added": replay.total_added,
    }


def _assert_snapshot(replay: AWACReplayBuffer, expected) -> None:
    assert replay.size == expected["size"]
    assert replay.position == expected["position"]
    assert replay.total_added == expected["total_added"]
    for name, value in expected["arrays"].items():
        assert np.array_equal(np.asarray(replay.arrays[name]), value), name


class _FailFirstWrite:
    """Array proxy that raises one BaseException, then permits rollback."""

    def __init__(self, array):
        self.array = array
        self.failed = False

    def __getattr__(self, name):
        return getattr(self.array, name)

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        if not self.failed:
            self.failed = True
            raise KeyboardInterrupt("injected Replay field write interruption")
        self.array[key] = value


def test_add_batch_commits_fully_encoded_rows(tmp_path: Path):
    replay = _create_replay(tmp_path / "replay")

    replay.add_batch([_transition(1), _transition(2)])

    assert (replay.size, replay.position, replay.total_added) == (2, 2, 2)
    assert np.asarray(replay.arrays["reward"][:2]).tolist() == [1.0, 2.0]
    assert np.asarray(replay.arrays["action"][:2]).tolist() == [1, 1]
    expected_depth = int(round(0.1 * 255.0))
    assert bool(np.all(replay.arrays["depth"][0] == expected_depth))


def test_add_batch_prevalidation_failure_writes_nothing(tmp_path: Path):
    replay = _create_replay(tmp_path / "replay")
    replay.add_batch([_transition(1)])
    before = _snapshot(replay)
    invalid = _transition(3, action=4)
    invalid["action_mask"][4] = False

    with pytest.raises(ValueError, match="invalid under the stored mask"):
        replay.add_batch([_transition(2), invalid])

    _assert_snapshot(replay, before)


def test_add_batch_mid_write_base_exception_restores_rows_and_counters(
    tmp_path: Path,
):
    replay = _create_replay(tmp_path / "replay")
    replay.add_batch([_transition(1), _transition(2)])
    before = _snapshot(replay)
    original_reward_array = replay.arrays["reward"]
    failing_reward_array = _FailFirstWrite(original_reward_array)
    replay.arrays["reward"] = failing_reward_array

    with pytest.raises(KeyboardInterrupt, match="injected Replay"):
        replay.add_batch([_transition(7), _transition(8)])

    assert failing_reward_array.failed
    replay.arrays["reward"] = original_reward_array
    _assert_snapshot(replay, before)


def test_add_batch_ring_wrap_commits_in_order_and_reopens(tmp_path: Path):
    replay_path = tmp_path / "replay"
    replay = _create_replay(replay_path, capacity=4)
    replay.add_batch([_transition(0), _transition(1), _transition(2)])

    replay.add_batch([_transition(3), _transition(4), _transition(5)])

    assert (replay.size, replay.position, replay.total_added) == (4, 2, 6)
    assert np.asarray(replay.arrays["reward"]).tolist() == [4.0, 5.0, 2.0, 3.0]
    replay.flush()

    reopened = AWACReplayBuffer.open(replay_path)
    assert (reopened.size, reopened.position, reopened.total_added) == (4, 2, 6)
    assert np.asarray(reopened.arrays["reward"]).tolist() == [4.0, 5.0, 2.0, 3.0]
