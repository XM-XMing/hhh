"""Focused M0.1 tests for the BC mmap observation provenance seam."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from planning.contracts.feature import NUM_ACTIONS, STATE_DIM, GOAL_DIM
from planning.data.bc_mmap import build_bc_mmap_dataset
from planning.data.rollout import (
    LABEL_CONTRACT_ID,
    ROLLOUT_CONTRACT_ID,
    TeacherLabelStore,
    make_rollout_metadata,
    save_rollout_episode,
)
from planning.mission.spec import TASK_CONTRACT_ID, task_contract_fields, task_contract_sha256
from planning.safety.depth_mask import DepthActionMaskStore
from planning.contracts.teacher_path import (
    TEACHER_ACTUAL_PATH_MAX_M,
    TEACHER_PATH_LENGTH_CONTRACT_ID,
)


EXACT_OBSERVATION_CONTRACT = "reliable_exact_endpoint_snapshot"
LEGACY_OBSERVATION_CONTRACT = "legacy_async_telemetry"
MPL_CONTRACT_SHA256 = "a" * 64
_MISSING = object()


def _metadata_with_contract(base, value):
    metadata = dict(base)
    if value is not _MISSING:
        metadata["observation_contract"] = value
        metadata["observation_source"] = value
    return metadata


def _make_fixture(
    root: Path,
    *,
    rollout_contracts=(EXACT_OBSERVATION_CONTRACT,),
    label_contract=EXACT_OBSERVATION_CONTRACT,
    mask_contract=EXACT_OBSERVATION_CONTRACT,
):
    root.mkdir(parents=True, exist_ok=True)
    episode_dir = root / "episodes"
    episode_dir.mkdir()
    transition_count = 2
    action_masks = np.ones((transition_count, NUM_ACTIONS), dtype=np.bool_)
    behavior_actions = np.asarray([1, 2], dtype=np.int64)
    rows = []
    label_paths = []
    mask_paths = []
    all_soft_targets = []
    all_global_masks = []
    all_teacher_actions = []
    all_behavior_actions = []
    all_valid_counts = []
    all_entropies = []
    all_masks = []
    offsets = []
    offset = 0

    for episode_id, contract in enumerate(rollout_contracts):
        episode_path = episode_dir / "episode_{}.npz".format(episode_id)
        depths = np.asarray(
            [
                [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]],
                [[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]],
            ],
            dtype=np.float32,
        )
        states = np.zeros((transition_count, STATE_DIM), dtype=np.float32)
        goals = np.zeros((transition_count, GOAL_DIM), dtype=np.float32)
        metadata = make_rollout_metadata(
            mpl_contract_sha256=MPL_CONTRACT_SHA256,
            teacher_path_length_contract_id=TEACHER_PATH_LENGTH_CONTRACT_ID,
            teacher_actual_path_max_m=TEACHER_ACTUAL_PATH_MAX_M,
            actual_path_length_m=0.0,
            reliable_execution=True,
            telemetry_observation=False,
            state_depth_skew_ns=0,
            endpoint_identity_available=True,
            asynchronous_prefetch=False,
            asynchronous_prefetch_status="obsolete_for_reliable_exact",
        )
        metadata = _metadata_with_contract(metadata, contract)
        save_rollout_episode(
            episode_path,
            {
                "depths": depths,
                "states": states,
                "goals": goals,
                "height_action_masks": action_masks,
                "execution_action_masks": action_masks,
                "behavior_actions": behavior_actions,
                "prev_actions": np.asarray([-1, 1], dtype=np.int64),
                "poses_before": np.zeros((transition_count, 4), dtype=np.float32),
                "velocities_before": np.zeros((transition_count, 3), dtype=np.float32),
                "state_ids": np.asarray([episode_id * 10, episode_id * 10 + 1], dtype=np.int64),
                "sim_time_ns": np.asarray([0, 1], dtype=np.int64),
                "state_stamp_ns": np.asarray([0, 1], dtype=np.int64),
                "depth_stamp_ns": np.asarray([0, 1], dtype=np.int64),
                "sensor_skew_ns": np.asarray([0, 0], dtype=np.int64),
                "primitive_actual_path_lengths_m": np.zeros((transition_count,), dtype=np.float32),
                "start": np.asarray([0.0, 0.0, 1.5], dtype=np.float32),
                "goal": np.asarray([0.0, 0.0, 1.5], dtype=np.float32),
            },
            metadata,
        )
        relative_episode_path = "episodes/episode_{}.npz".format(episode_id)
        rows.append(
            {
                "episode_id": str(episode_id),
                "execute_ok": "true",
                "dataset_npz": relative_episode_path,
            }
        )
        label_paths.append(relative_episode_path)
        mask_paths.append(relative_episode_path)
        offsets.append(offset)
        offset += transition_count
        all_soft_targets.append(np.eye(NUM_ACTIONS, dtype=np.float32)[behavior_actions])
        all_global_masks.append(action_masks)
        all_teacher_actions.append(behavior_actions)
        all_behavior_actions.append(behavior_actions)
        all_valid_counts.append(np.full((transition_count,), NUM_ACTIONS, dtype=np.int16))
        all_entropies.append(np.zeros((transition_count,), dtype=np.float32))
        all_masks.append(action_masks)

    index_path = root / "index.csv"
    index_path.write_text(
        "episode_id,execute_ok,dataset_npz\n"
        + "".join(
            "{episode_id},{execute_ok},{dataset_npz}\n".format(**row)
            for row in rows
        ),
        encoding="utf-8",
    )

    label_metadata = {
        "label_contract_id": LABEL_CONTRACT_ID,
        "feature_contract_id": "depth_goal_state_prev_action",
        **task_contract_fields(),
        "mpl_contract_sha256": MPL_CONTRACT_SHA256,
        "actual_state_relabeling": True,
        "reliable_execution": True,
        "telemetry_observation": False,
        "state_depth_skew_ns": 0,
        "endpoint_identity_available": True,
        "asynchronous_prefetch": False,
        "asynchronous_prefetch_status": "obsolete_for_reliable_exact",
    }
    label_metadata = _metadata_with_contract(label_metadata, label_contract)
    labels_path = root / "labels.npz"
    np.savez(
        labels_path,
        soft_targets=np.concatenate(all_soft_targets, axis=0),
        global_action_masks=np.concatenate(all_global_masks, axis=0),
        teacher_argmax=np.concatenate(all_teacher_actions, axis=0),
        behavior_actions=np.concatenate(all_behavior_actions, axis=0),
        valid_counts=np.concatenate(all_valid_counts, axis=0),
        teacher_entropies=np.concatenate(all_entropies, axis=0),
        episode_npz_paths=np.asarray(label_paths),
        episode_offsets=np.asarray(offsets, dtype=np.int64),
        episode_lengths=np.full((len(rows),), transition_count, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(label_metadata, sort_keys=True)),
    )

    mask_metadata = _metadata_with_contract(
        {
            "depth_action_mask_contract_id": "local_depth_action_masks",
            **task_contract_fields(),
            "reliable_execution": True,
            "telemetry_observation": False,
            "state_depth_skew_ns": 0,
            "endpoint_identity_available": True,
            "asynchronous_prefetch": False,
            "asynchronous_prefetch_status": "obsolete_for_reliable_exact",
        },
        mask_contract,
    )
    masks_path = root / "depth_masks.npz"
    np.savez(
        masks_path,
        local_action_masks=np.concatenate(all_masks, axis=0),
        episode_npz_paths=np.asarray(mask_paths),
        episode_offsets=np.asarray(offsets, dtype=np.int64),
        episode_lengths=np.full((len(rows),), transition_count, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(mask_metadata, sort_keys=True)),
    )
    return index_path, labels_path, masks_path, rows


def _build(fixture, out_dir: Path):
    index_path, labels_path, masks_path, rows = fixture
    return build_bc_mmap_dataset(
        rows,
        index_path,
        TeacherLabelStore(labels_path),
        out_dir,
        DepthActionMaskStore(masks_path),
    )


@pytest.mark.unit
@pytest.mark.parametrize("source", ("rollout", "label", "mask"))
def test_missing_observation_contract_is_rejected(tmp_path: Path, source: str):
    kwargs = {}
    kwargs["rollout_contracts"] = (_MISSING,) if source == "rollout" else (EXACT_OBSERVATION_CONTRACT,)
    kwargs["label_contract"] = _MISSING if source == "label" else EXACT_OBSERVATION_CONTRACT
    kwargs["mask_contract"] = _MISSING if source == "mask" else EXACT_OBSERVATION_CONTRACT
    fixture = _make_fixture(tmp_path / source, **kwargs)
    with pytest.raises(ValueError, match="observation contract"):
        _build(fixture, tmp_path / source / "out")


@pytest.mark.unit
def test_unknown_observation_contract_is_rejected(tmp_path: Path):
    fixture = _make_fixture(tmp_path, rollout_contracts=("future_observation_v9",))
    with pytest.raises(ValueError, match="observation contract"):
        _build(fixture, tmp_path / "out")


@pytest.mark.unit
def test_rollout_label_observation_contract_mismatch_is_rejected(tmp_path: Path):
    fixture = _make_fixture(tmp_path, label_contract=LEGACY_OBSERVATION_CONTRACT)
    with pytest.raises(ValueError, match="observation contract"):
        _build(fixture, tmp_path / "out")


@pytest.mark.unit
def test_rollout_mask_observation_contract_mismatch_is_rejected(tmp_path: Path):
    fixture = _make_fixture(tmp_path, mask_contract=LEGACY_OBSERVATION_CONTRACT)
    with pytest.raises(ValueError, match="observation contract"):
        _build(fixture, tmp_path / "out")


@pytest.mark.unit
def test_legacy_and_reliable_rollouts_cannot_be_mixed(tmp_path: Path):
    fixture = _make_fixture(
        tmp_path,
        rollout_contracts=(EXACT_OBSERVATION_CONTRACT, LEGACY_OBSERVATION_CONTRACT),
    )
    with pytest.raises(ValueError, match="observation contract"):
        _build(fixture, tmp_path / "out")


@pytest.mark.unit
def test_reliable_exact_fixture_passes_and_preserves_mmap_layout(tmp_path: Path):
    fixture = _make_fixture(tmp_path)
    manifest = _build(fixture, tmp_path / "out")

    assert manifest["observation_contract"] == EXACT_OBSERVATION_CONTRACT
    assert manifest["observation_source"] == EXACT_OBSERVATION_CONTRACT
    assert manifest["reliable_rows"] == manifest["transition_count"]
    assert manifest["legacy_rows"] == 0
    assert manifest["episode_count"] == 1
    assert manifest["transition_count"] == 2
    assert manifest["episodes"][0]["episode_id"] == 0

    expected_arrays = {
        "depths": ((2, 2, 3), np.float16),
        "continuous": ((2, 22), np.float32),
        "prev_actions": ((2,), np.int16),
        "height_masks": ((2, NUM_ACTIONS), np.bool_),
        "local_depth_masks": ((2, NUM_ACTIONS), np.bool_),
        "behavior_actions": ((2,), np.int16),
        "teacher_actions": ((2,), np.int16),
        "soft_targets": ((2, NUM_ACTIONS), np.float32),
        "history_starts": ((2,), np.int64),
        "episode_ids": ((2,), np.int64),
    }
    for name, (shape, dtype) in expected_arrays.items():
        array = np.load(tmp_path / "out" / (name + ".npy"), mmap_mode="r")
        assert array.shape == shape
        assert array.dtype == dtype
