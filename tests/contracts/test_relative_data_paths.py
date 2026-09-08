#!/usr/bin/env python3
"""Verify that movable rollout indexes and structured archives use relative paths."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from planning.data.rollout import LABEL_CONTRACT_ID, TeacherLabelStore, resolve_dataset_path
from planning.safety.depth_mask import DEPTH_ACTION_MASK_CONTRACT_ID, DepthActionMaskStore
from planning.contracts.feature import FEATURE_CONTRACT_ID, NUM_ACTIONS
from planning.mission.spec import TASK_CONTRACT_ID, task_contract_fields, task_contract_sha256


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="planning_relative_paths_") as directory:
        root = Path(directory)
        episode = root / "episodes" / "episode_000001.npz"
        episode.parent.mkdir()
        episode.touch()

        index = root / "rollout_index.csv"
        with index.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["episode_id", "dataset_npz"])
            writer.writeheader()
            writer.writerow({"episode_id": 1, "dataset_npz": "episodes/episode_000001.npz"})
        assert resolve_dataset_path(
            {"dataset_npz": "episodes/episode_000001.npz"}, index
        ) == episode.resolve()

        label_metadata = {
            "label_contract_id": LABEL_CONTRACT_ID,
            "feature_contract_id": FEATURE_CONTRACT_ID,
            **task_contract_fields(),
            "mpl_contract_sha256": "0" * 64,
            "actual_state_relabeling": True,
        }
        labels = root / "teacher_labels.npz"
        action_masks = np.ones((1, NUM_ACTIONS), dtype=np.bool_)
        soft_targets = np.zeros((1, NUM_ACTIONS), dtype=np.float32)
        soft_targets[0, 0] = 1.0
        np.savez(
            labels,
            soft_targets=soft_targets,
            global_action_masks=action_masks,
            teacher_argmax=np.asarray([0], dtype=np.int64),
            behavior_actions=np.asarray([0], dtype=np.int64),
            valid_counts=np.asarray([NUM_ACTIONS], dtype=np.int16),
            teacher_entropies=np.asarray([0.0], dtype=np.float32),
            episode_npz_paths=np.asarray(["episodes/episode_000001.npz"]),
            episode_offsets=np.asarray([0], dtype=np.int64),
            episode_lengths=np.asarray([1], dtype=np.int64),
            metadata_json=np.asarray(json.dumps(label_metadata, sort_keys=True)),
        )
        label_store = TeacherLabelStore(labels)
        assert str(episode.resolve()) in label_store.path_to_slice

        masks = root / "depth_action_masks.npz"
        mask_metadata = {
            "depth_action_mask_contract_id": DEPTH_ACTION_MASK_CONTRACT_ID,
            "observation_contract": "reliable_exact_endpoint_snapshot",
            "observation_source": "reliable_exact_endpoint_snapshot",
            **task_contract_fields(),
        }
        np.savez(
            masks,
            local_action_masks=action_masks,
            episode_npz_paths=np.asarray(["episodes/episode_000001.npz"]),
            episode_offsets=np.asarray([0], dtype=np.int64),
            episode_lengths=np.asarray([1], dtype=np.int64),
            metadata_json=np.asarray(json.dumps(mask_metadata, sort_keys=True)),
        )
        mask_store = DepthActionMaskStore(masks)
        assert str(episode.resolve()) in mask_store.path_to_slice

    print("RELATIVE_DATA_PATHS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
