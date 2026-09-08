import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from planning.data.bc_mmap import BC_MMAP_DATASET_CONTRACT_ID, MappedSoftRolloutDataset
from planning.contracts.feature import CONTINUOUS_DIM, NUM_ACTIONS, POLICY_VECTOR_DIM
from planning.contracts.task import task_contract_fields

pytestmark = pytest.mark.unit

class IdentityNormalizer:
    def transform_continuous(self, values):
        return np.asarray(values, dtype=np.float32)

def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_mmap_history_and_previous_action_contract(tmp_path):
    index = tmp_path / "rollout_index.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "dataset_npz"])
        writer.writeheader()
        writer.writerow({"episode_id": 7, "dataset_npz": "episode_7.npz"})

    cache = tmp_path / "cache"
    cache.mkdir()
    count = 3
    shapes = {
        "depths": ((count, 2, 2), np.float16),
        "continuous": ((count, CONTINUOUS_DIM), np.float32),
        "prev_actions": ((count,), np.int16),
        "height_masks": ((count, NUM_ACTIONS), np.bool_),
        "local_depth_masks": ((count, NUM_ACTIONS), np.bool_),
        "behavior_actions": ((count,), np.int16),
        "teacher_actions": ((count,), np.int16),
        "soft_targets": ((count, NUM_ACTIONS), np.float32),
        "history_starts": ((count,), np.int64),
        "episode_ids": ((count,), np.int64),
    }
    for name, (shape, dtype) in shapes.items():
        values = np.zeros(shape, dtype=dtype)
        if name == "depths":
            values[:] = np.arange(count, dtype=np.float16)[:, None, None]
        elif name == "continuous":
            values[:] = np.arange(count, dtype=np.float32)[:, None]
        elif name == "prev_actions":
            values[:] = [-1, 4, 8]
        elif name.endswith("masks"):
            values[:] = True
        elif name == "history_starts":
            values[:] = 0
        np.save(str(cache / (name + ".npy")), values)

    manifest = {
        "contract_id": BC_MMAP_DATASET_CONTRACT_ID,
        "source_index_sha256": _hash(index),
        "source_labels_sha256": "unused",
        "source_depth_masks_sha256": "",
        "max_steps": 45,
        **task_contract_fields(),
        "episodes": [
            {"episode_id": 7, "dataset_npz": "episode_7.npz", "offset": 0, "length": 3}
        ],
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = MappedSoftRolloutDataset(
        cache, [{"episode_id": "7", "dataset_npz": "episode_7.npz"}], index
    )
    dataset.apply_normalizer(IdentityNormalizer())
    global_index, history, vector = dataset.sample(1, depth_history_frames=3)

    assert global_index == 1
    np.testing.assert_array_equal(history, [0, 0, 1])
    assert vector.shape == (POLICY_VECTOR_DIM,)
    np.testing.assert_array_equal(vector[:CONTINUOUS_DIM], np.ones(CONTINUOUS_DIM))
    assert vector[CONTINUOUS_DIM + 4] == 1.0
    assert float(vector[CONTINUOUS_DIM:].sum()) == 1.0


def test_initial_previous_action_has_zero_onehot(tmp_path):
    # Reuse the complete contract test while checking the special -1 encoding.
    test_mmap_history_and_previous_action_contract(tmp_path)
    cache = tmp_path / "cache"
    index = tmp_path / "rollout_index.csv"
    dataset = MappedSoftRolloutDataset(
        cache, [{"episode_id": "7", "dataset_npz": "episode_7.npz"}], index
    )
    dataset.apply_normalizer(IdentityNormalizer())
    _, _, vector = dataset.sample(0, depth_history_frames=1)
    assert float(vector[CONTINUOUS_DIM:].sum()) == 0.0
