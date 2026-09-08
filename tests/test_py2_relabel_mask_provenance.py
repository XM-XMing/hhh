"""PY2 RED tests for relabel/depth-mask producer provenance.

The fixture is deliberately small and synthetic.  It exercises the public
provenance boundary without starting Unity, ROS, native collision code, or a
formal data collection run.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from planning.contracts.feature import NUM_ACTIONS, STATE_DIM, GOAL_DIM
from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    LEGACY_ASYNC_OBSERVATION_CONTRACT,
    exact_endpoint_metadata,
)
from planning.contracts.pipeline_provenance import (
    artifact_provenance_metadata,
    validate_artifact_provenance,
    validate_cross_artifact_consistency,
    validate_rollout_provenance,
)
from planning.contracts.task import task_contract_fields
from planning.data.rollout import make_rollout_metadata, save_rollout_episode


def _write_fixture(root: Path) -> tuple[Path, Path, object]:
    worker = root / "workers" / "worker_00"
    episode_dir = worker / "episodes"
    episode_dir.mkdir(parents=True)
    episode_path = episode_dir / "episode_000000.npz"
    transition_count = 2
    runtime_id = "runtime-py2-00"
    run_id = "py2-fixture"
    mission_sha = "a" * 64
    config_sha = "b" * 64
    mpl_sha = "c" * 64

    metadata = make_rollout_metadata(
        collection_run_id=run_id,
        runtime_instance_id=runtime_id,
        episode_id=0,
        mission_id="mission-0",
        mpl_contract_sha256=mpl_sha,
        teacher_path_length_contract_id="teacher_plan44_actual46_observed_polyline",
        teacher_actual_path_max_m=46.0,
        actual_path_length_m=0.0,
        reliable_rows=transition_count,
        legacy_rows=0,
        telemetry_lookup_count=0,
        snapshot_missing_count=0,
        state_depth_skew_max_ns=0,
        frame_contract_failures=0,
        endpoint_identity_chain_valid=True,
        mission_index_sha256=mission_sha,
        resolved_config_sha256=config_sha,
        **exact_endpoint_metadata(),
    )
    arrays = {
        "depths": np.zeros((transition_count, 2, 3), dtype=np.float32),
        "states": np.zeros((transition_count, STATE_DIM), dtype=np.float32),
        "goals": np.zeros((transition_count, GOAL_DIM), dtype=np.float32),
        "height_action_masks": np.ones((transition_count, NUM_ACTIONS), dtype=np.bool_),
        "execution_action_masks": np.ones((transition_count, NUM_ACTIONS), dtype=np.bool_),
        "behavior_actions": np.asarray([1, 2], dtype=np.int64),
        "prev_actions": np.asarray([-1, 1], dtype=np.int64),
        "poses_before": np.zeros((transition_count, 4), dtype=np.float32),
        "velocities_before": np.zeros((transition_count, 3), dtype=np.float32),
        "state_ids": np.asarray([0, 1], dtype=np.int64),
        "sim_time_ns": np.asarray([0, 1], dtype=np.int64),
        "state_stamp_ns": np.asarray([0, 1], dtype=np.int64),
        "depth_stamp_ns": np.asarray([0, 1], dtype=np.int64),
        "sensor_skew_ns": np.asarray([0, 0], dtype=np.int64),
        "primitive_actual_path_lengths_m": np.zeros((transition_count,), dtype=np.float32),
        "start": np.asarray([0.0, 0.0, 1.5], dtype=np.float32),
        "goal": np.asarray([1.0, 0.0, 1.5], dtype=np.float32),
    }
    save_rollout_episode(episode_path, arrays, metadata)

    row = {
        "episode_id": "0",
        "mission_id": "mission-0",
        "collection_run_id": run_id,
        "runtime_instance_id": runtime_id,
        **exact_endpoint_metadata(),
        "reliable_rows": str(transition_count),
        "legacy_rows": "0",
        "telemetry_lookup_count": "0",
        "snapshot_missing_count": "0",
        "state_depth_skew_max_ns": "0",
        "frame_contract_failures": "0",
        "endpoint_identity_chain_valid": "True",
        "transition_ids": json.dumps([runtime_id + ":0", runtime_id + ":1"]),
        "task_contract_id": metadata["task_contract_id"],
        **task_contract_fields(),
        "task_contract_sha256": metadata["task_contract_sha256"],
        "mission_index_sha256": mission_sha,
        "resolved_config_sha256": config_sha,
        "execute_ok": "True",
        "dataset_npz": "workers/worker_00/episodes/episode_000000.npz",
        "start_x": "0.0",
        "start_y": "0.0",
        "start_z": "1.5",
        "goal_x": "1.0",
        "goal_y": "0.0",
        "goal_z": "1.5",
    }
    index_path = root / "rollout_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    shared = {
        **exact_endpoint_metadata(),
        **task_contract_fields(),
        "reliable_rows": transition_count,
        "legacy_rows": 0,
        "telemetry_lookup_count": 0,
        "snapshot_missing_count": 0,
        "state_depth_skew_max_ns": 0,
        "frame_contract_failures": 0,
        "endpoint_identity_chain_valid": True,
    }
    worker_summary = {
        **shared,
        "worker_id": 0,
        "workers": 1,
        "collection_run_id": run_id,
        "runtime_instance_id": runtime_id,
        "resolved_config_sha256": config_sha,
        "mission_index_sha256": mission_sha,
        "mpl_contract_sha256": mpl_sha,
        "collection_config_contract_id": "teacher_rollout_collection_resolved_config",
        "accepted_total": 1,
        "attempted_total": 1,
        "rollout_index": "rollout_index.csv",
    }
    (worker / "collection_summary.json").write_text(
        json.dumps(worker_summary, sort_keys=True), encoding="utf-8"
    )
    (worker / "resolved_collection_config.json").write_text(
        json.dumps(
            {
                "resolved_config_sha256": config_sha,
                **task_contract_fields(),
                "shared_collection_config": task_contract_fields(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    root_summary = {
        **shared,
        "workers": 1,
        "collection_run_id": run_id,
        "accepted_total": 1,
        "attempted_total": 1,
        "reliable_rows": transition_count,
        "mission_index_sha256": mission_sha,
        "mpl_contract_sha256": mpl_sha,
        "resolved_config_sha256": config_sha,
        "rollout_index": "rollout_index.csv",
        "collection_report": "collection_report.csv",
        "quality_pass": True,
        "worker_resolved_configs": [
            "workers/worker_00/resolved_collection_config.json"
        ],
    }
    (root / "collection_summary.json").write_text(
        json.dumps(root_summary, sort_keys=True), encoding="utf-8"
    )
    return index_path, episode_path, metadata


def _artifact(provenance, path: Path, *, kind: str = "teacher_labels"):
    return artifact_provenance_metadata(
        provenance,
        artifact_path=path,
        artifact_kind=kind,
        row_count=2,
        episode_ids=[0],
        transition_offsets=[0],
        transition_lengths=[2],
    )


@pytest.mark.unit
def test_missing_rollout_observation_contract_is_rejected(tmp_path: Path):
    index_path, episode_path, _ = _write_fixture(tmp_path)
    with np.load(episode_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata_json"}
        metadata = json.loads(data["metadata_json"].item())
    metadata.pop("observation_contract")
    save_rollout_episode(episode_path, arrays, metadata)
    with pytest.raises(ValueError, match="observation contract"):
        validate_rollout_provenance(index_path)


@pytest.mark.unit
def test_legacy_rollout_is_rejected(tmp_path: Path):
    index_path, episode_path, _ = _write_fixture(tmp_path)
    with np.load(episode_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata_json"}
        metadata = json.loads(data["metadata_json"].item())
    metadata["observation_contract"] = LEGACY_ASYNC_OBSERVATION_CONTRACT
    metadata["observation_source"] = LEGACY_ASYNC_OBSERVATION_CONTRACT
    save_rollout_episode(episode_path, arrays, metadata)
    with pytest.raises(ValueError, match="formal|legacy|observation contract"):
        validate_rollout_provenance(index_path)


@pytest.mark.unit
def test_label_rollout_contract_mismatch_is_rejected(tmp_path: Path):
    index_path, _, _ = _write_fixture(tmp_path)
    provenance = validate_rollout_provenance(index_path)
    labels = _artifact(provenance, tmp_path / "labels.npz")
    labels["observation_contract"] = LEGACY_ASYNC_OBSERVATION_CONTRACT
    labels["observation_source"] = LEGACY_ASYNC_OBSERVATION_CONTRACT
    with pytest.raises(ValueError, match="observation contract"):
        validate_artifact_provenance(labels, artifact_kind="teacher_labels")


@pytest.mark.unit
def test_mask_rollout_contract_mismatch_is_rejected(tmp_path: Path):
    index_path, _, _ = _write_fixture(tmp_path)
    provenance = validate_rollout_provenance(index_path)
    labels = _artifact(provenance, tmp_path / "labels.npz")
    masks = _artifact(provenance, tmp_path / "masks.npz", kind="depth_masks")
    masks["observation_contract"] = LEGACY_ASYNC_OBSERVATION_CONTRACT
    masks["observation_source"] = LEGACY_ASYNC_OBSERVATION_CONTRACT
    with pytest.raises(ValueError, match="observation contract"):
        validate_cross_artifact_consistency(provenance, labels, masks)


@pytest.mark.unit
def test_row_count_and_episode_order_mismatches_are_rejected(tmp_path: Path):
    index_path, _, _ = _write_fixture(tmp_path)
    provenance = validate_rollout_provenance(index_path)
    labels = _artifact(provenance, tmp_path / "labels.npz")
    masks = _artifact(provenance, tmp_path / "masks.npz", kind="depth_masks")
    labels["row_count"] = 3
    with pytest.raises(ValueError, match="row|transition"):
        validate_cross_artifact_consistency(provenance, labels, masks)
    labels = _artifact(provenance, tmp_path / "labels.npz")
    labels["episode_order"] = [1]
    with pytest.raises(ValueError, match="episode"):
        validate_cross_artifact_consistency(provenance, labels, masks)


@pytest.mark.unit
def test_valid_reliable_chain_and_metadata_round_trip(tmp_path: Path):
    index_path, _, _ = _write_fixture(tmp_path)
    provenance = validate_rollout_provenance(index_path)
    labels = _artifact(provenance, tmp_path / "labels.npz")
    masks = _artifact(provenance, tmp_path / "masks.npz", kind="depth_masks")
    validate_artifact_provenance(labels, artifact_kind="teacher_labels")
    validate_artifact_provenance(masks, artifact_kind="depth_masks")
    validate_cross_artifact_consistency(provenance, labels, masks)
    round_trip = json.loads(json.dumps(labels, sort_keys=True))
    assert round_trip["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    assert round_trip["reliable_rows"] == 2
    assert round_trip["legacy_rows"] == 0
    assert round_trip["episode_order"] == [0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("observation_contract", "unknown-contract"),
        ("observation_source", LEGACY_ASYNC_OBSERVATION_CONTRACT),
        ("legacy_rows", 1),
        ("telemetry_lookup_count", 1),
        ("snapshot_missing_count", 1),
        ("state_depth_skew_max_ns", 1),
        ("frame_contract_failures", 1),
        ("endpoint_identity_chain_valid", False),
    ),
)
def test_non_exact_rollout_provenance_is_rejected(tmp_path: Path, field, value):
    index_path, _, _ = _write_fixture(tmp_path)
    summary_path = tmp_path / "collection_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_rollout_provenance(index_path)


@pytest.mark.unit
def test_legacy_and_reliable_episode_rows_are_rejected_as_mixed(tmp_path: Path):
    index_path, episode_path, _ = _write_fixture(tmp_path)
    with np.load(episode_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata_json"}
        metadata = json.loads(data["metadata_json"].item())
    metadata["observation_contract"] = LEGACY_ASYNC_OBSERVATION_CONTRACT
    metadata["observation_source"] = LEGACY_ASYNC_OBSERVATION_CONTRACT
    metadata["reliable_execution"] = False
    metadata["telemetry_observation"] = True
    metadata["endpoint_identity_available"] = False
    metadata["asynchronous_prefetch"] = True
    save_rollout_episode(episode_path, arrays, metadata)
    with pytest.raises(ValueError):
        validate_rollout_provenance(index_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "script_name",
    (
        "label_teacher_rollouts.py",
        "generate_depth_action_masks.py",
        "audit_teacher_dataset.py",
        "build_bc_mmap_dataset.py",
    ),
)
def test_py2_cli_help(script_name):
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "python")
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / script_name), "--help"],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "usage:" in result.stdout
