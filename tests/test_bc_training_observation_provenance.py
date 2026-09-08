"""M0.2 RED/GREEN tests for the BC training provenance seam."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from planning.bc.trainer import (
    build_argument_parser,
    save_checkpoint,
    split_rows_from_checkpoint,
    validate_bc_mmap_provenance,
)
from planning.bc.model import VectorNormalizer, build_model, require_torch
from planning.contracts.feature import CONTINUOUS_DIM
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.task import task_contract_fields


def _manifest(**overrides):
    manifest = {
        "contract_id": "bc_mmap_dataset",
        "transition_count": 8,
        "episode_count": 4,
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "reliable_rows": 8,
        "legacy_rows": 0,
        "source_index_sha256": "a" * 64,
        "source_labels_sha256": "b" * 64,
        "source_depth_masks_sha256": "c" * 64,
        "mpl_contract_sha256": "d" * 64,
        **task_contract_fields(),
    }
    manifest.update(overrides)
    return manifest


@pytest.mark.unit
def test_bc_trainer_rejects_manifest_missing_observation_contract():
    manifest = _manifest()
    manifest.pop("observation_contract")
    with pytest.raises(ValueError, match="observation contract"):
        validate_bc_mmap_provenance(manifest)


@pytest.mark.unit
def test_bc_trainer_rejects_cli_observation_contract_mismatch():
    with pytest.raises(ValueError, match="observation contract"):
        validate_bc_mmap_provenance(
            _manifest(), expected_observation_contract="legacy_async_telemetry"
        )


@pytest.mark.unit
def test_bc_trainer_rejects_reliable_manifest_with_legacy_rows():
    manifest = _manifest(legacy_rows=1, reliable_rows=7)
    with pytest.raises(ValueError, match="legacy"):
        validate_bc_mmap_provenance(manifest)


@pytest.mark.unit
def test_bc_trainer_rejects_reliable_count_mismatch():
    manifest = _manifest(reliable_rows=7)
    with pytest.raises(ValueError, match="reliable_rows"):
        validate_bc_mmap_provenance(manifest)


@pytest.mark.unit
def test_valid_reliable_mmap_provenance_is_resolved_without_data_changes():
    manifest = _manifest()
    resolved = validate_bc_mmap_provenance(manifest)
    assert resolved["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    assert resolved["observation_source"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    assert resolved["reliable_rows"] == 8
    assert resolved["legacy_rows"] == 0
    assert resolved["transition_count"] == 8
    assert resolved["source_index_sha256"] == manifest["source_index_sha256"]


@pytest.mark.unit
def test_bc_cli_has_one_formal_observation_contract_default():
    args = build_argument_parser().parse_args(
        [
            "--index",
            "index.csv",
            "--labels",
            "labels.npz",
            "--out-dir",
            "out",
            "--deployment-safety-mask",
            "depth",
            "--deployment-execution-mode",
            "continuous",
        ]
    )
    assert args.observation_contract == EXACT_ENDPOINT_OBSERVATION_CONTRACT


@pytest.mark.unit
def test_checkpoint_split_accepts_known_nested_episode_subset():
    rows = [
        {"episode_id": str(index), "mission_id": "mission-{}".format(index)}
        for index in range(4)
    ]
    checkpoint = {
        "train_mission_ids": ["mission-0", "mission-1"],
        "val_mission_ids": ["mission-2", "mission-3"],
    }
    train, val = split_rows_from_checkpoint(
        rows[:3], checkpoint, allow_subset=True
    )
    assert [row["episode_id"] for row in train] == ["0", "1"]
    assert [row["episode_id"] for row in val] == ["2"]


@pytest.mark.unit
def test_checkpoint_split_rejects_unknown_nested_episode_subset():
    rows = [
        {"episode_id": "0", "mission_id": "mission-0"},
        {"episode_id": "9", "mission_id": "mission-9"},
    ]
    checkpoint = {
        "train_mission_ids": ["mission-0"],
        "val_mission_ids": ["mission-1"],
    }
    with pytest.raises(ValueError, match="absent from checkpoint split"):
        split_rows_from_checkpoint(rows, checkpoint, allow_subset=True)


@pytest.mark.unit
def test_checkpoint_split_preserves_full_index_missing_validation_failure():
    rows = [{"episode_id": "0", "mission_id": "mission-0"}]
    checkpoint = {
        "train_mission_ids": ["mission-0"],
        "val_mission_ids": ["mission-1"],
    }
    with pytest.raises(ValueError, match="missing checkpoint validation"):
        split_rows_from_checkpoint(rows, checkpoint, allow_subset=False)


@pytest.mark.unit
def test_invalid_provenance_does_not_mutate_manifest():
    manifest = _manifest()
    original = deepcopy(manifest)
    with pytest.raises(ValueError):
        validate_bc_mmap_provenance(
            manifest, expected_observation_contract="legacy_async_telemetry"
        )
    assert manifest == original


@pytest.mark.unit
def test_checkpoint_persists_training_provenance_without_changing_model_state(tmp_path):
    torch, nn, _, _, _ = require_torch()
    model = build_model(nn, depth_channels=1)
    normalizer = VectorNormalizer(
        mean=np.zeros(CONTINUOUS_DIM, dtype=np.float32),
        std=np.ones(CONTINUOUS_DIM, dtype=np.float32),
    )
    args = SimpleNamespace(
        index="index.csv",
        labels="labels.npz",
        depth_action_masks="depth_masks.npz",
        dataset_cache="cache",
        out_dir="out",
        tensorboard_log_dir="",
        val_episodes_from_checkpoint="",
        observation_contract=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        deployment_safety_mask="depth",
        deployment_execution_mode="continuous",
    )
    provenance = {
        **validate_bc_mmap_provenance(_manifest()),
        "dataset_manifest_sha256": "e" * 64,
        "normalizer_sha256": "f" * 64,
        "resolved_training_config": {"observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT},
        "resolved_training_config_sha256": "0" * 64,
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        1,
        args,
        normalizer,
        [{"episode_id": "1", "mission_id": "train"}],
        [{"episode_id": "2", "mission_id": "val"}],
        {"epoch": 1, "train_loss": 0.0},
        "d" * 64,
        provenance,
    )
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    for field in (
        "observation_contract",
        "observation_source",
        "reliable_rows",
        "legacy_rows",
        "dataset_manifest_sha256",
        "dataset_manifest_contract_id",
        "source_index_sha256",
        "source_labels_sha256",
        "source_depth_masks_sha256",
        "normalizer_sha256",
        "resolved_training_config",
        "resolved_training_config_sha256",
    ):
        assert field in checkpoint
    assert checkpoint["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    assert checkpoint["legacy_rows"] == 0
    assert checkpoint["model_state_dict"]
