"""Focused PY0 tests for exact shard merge and worker isolation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from planning.contracts.collection import (
    build_resolved_collection_config,
)
from planning.contracts.feature import GOAL_DIM, NUM_ACTIONS, STATE_DIM
from planning.contracts.observation import exact_endpoint_metadata
from planning.contracts.task import task_contract_fields
from planning.data.rollout import make_rollout_metadata, save_rollout_episode
from planning.data.rollout_merge import merge_worker_rollouts
from planning.data.rollout_shard import RolloutShardIdentity, worker_shard_paths
from planning.runtime.ports import build_worker_runtime_specs
from planning.runtime.collection_worker import RuntimeWorker
from planning.teacher.collection_run import CollectionResumeLedger


def _args(worker_id: int, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        index="missions.csv",
        out_dir=str(out_dir),
        worker_id=worker_id,
        num_workers=2,
        max_episodes=0,
        target_accepted=0,
        max_steps=45,
        stream_horizon=1,
        max_endpoint_error_m=0.6,
        max_sensor_skew_ms=80.0,
        max_stream_drift_m=0.35,
        max_actual_path_length_m=46.0,
        collision_cache="forest.npz",
        stop_file="stop",
        disable_async_prefetch=False,
        reliable_exact_runtime_instance_id="runtime-{}".format(worker_id),
        reliable_exact_command_endpoint="tcp://127.0.0.1:{}".format(
            10559 + worker_id * 20
        ),
        reliable_exact_result_endpoint="tcp://127.0.0.1:{}".format(
            10556 + worker_id * 20
        ),
        reliable_exact_snapshot_endpoint="tcp://127.0.0.1:{}".format(
            10558 + worker_id * 20
        ),
        reliable_exact_timeout=10.0,
        voxel_size=0.1,
        inflate_radius=0.35,
    )


def _write_exact_shards(root: Path, monkeypatch) -> tuple[Path, Path]:
    monkeypatch.setenv("PLANNING_COLLECTION_RUN_ID", "py0-run")
    monkeypatch.setenv("PLANNING_COLLECTION_WORKERS_REQUESTED", "2")
    monkeypatch.setenv("PLANNING_COLLECTION_GLOBAL_MAX_EPISODES", "0")
    monkeypatch.setenv("PLANNING_COLLECTION_GLOBAL_TARGET_ACCEPTED", "0")
    monkeypatch.setenv("PLANNING_COLLISION_THREADS", "1")
    monkeypatch.setenv("PLANNING_COLLECTION_CODE_SHA256", "e" * 64)
    package_root = Path(__file__).resolve().parents[1]
    workers = root / "workers"
    workers.mkdir(parents=True)
    rows_by_worker = {}
    configs = {}
    artifact_identity = {
        "unity_player_path": "/tmp/py0/XMflight.x86_64",
        "unity_player_sha256": "f" * 64,
        "runtime_assembly_path": "/tmp/py0/Assembly-CSharp.dll",
        "runtime_assembly_sha256": "a" * 64,
        "assembly_csharp_sha256": "a" * 64,
        "bridge_path": "/tmp/py0/unity_bridge_node",
        "bridge_sha256": "b" * 64,
    }
    for worker_id in range(2):
        worker_dir = workers / "worker_{:02d}".format(worker_id)
        worker_dir.mkdir()
        args = _args(worker_id, worker_dir)
        config = build_resolved_collection_config(
            args,
            mission_index_sha256="a" * 64,
            collision_cache_sha256="b" * 64,
            code_version_sha256="e" * 64,
            teacher_config={"beam_depth": 3},
            mpl_contract_sha256="d" * 64,
            async_prefetch_enabled=False,
            observation_contract="reliable_exact_endpoint_snapshot",
            observation_source="reliable_exact_endpoint_snapshot",
            reliable_execution=True,
            runtime_artifact_identity=artifact_identity,
            package_root=package_root,
        )
        runtime_id = "runtime-{}".format(worker_id)
        row = {
            "episode_id": worker_id,
            "mission_id": "mission-{}".format(worker_id),
            "collection_run_id": "py0-run",
            "runtime_instance_id": runtime_id,
            **task_contract_fields(),
            "unity_player_sha256": artifact_identity["unity_player_sha256"],
            "runtime_assembly_sha256": artifact_identity["runtime_assembly_sha256"],
            "bridge_sha256": artifact_identity["bridge_sha256"],
            **exact_endpoint_metadata(),
            "reliable_rows": 1,
            "legacy_rows": 0,
            "telemetry_lookup_count": 0,
            "snapshot_missing_count": 0,
            "state_depth_skew_max_ns": 0,
            "frame_contract_failures": 0,
            "endpoint_identity_chain_valid": True,
            "transition_ids": json.dumps(["{}:1".format(runtime_id)]),
            "global_collision_mask_enabled": True,
            "async_prefetch_enabled": False,
            "execute_ok": True,
            "success": True,
            "collision": False,
            "dead_end": False,
            "hard_altitude": False,
            "stop_reason": "success",
            "actual_path_length_m": 1.0,
            "path_length_exceeded": False,
            "dataset_npz": "",
            "error": "",
        }
        episode_path = worker_dir / "episodes" / "episode_{:06d}.npz".format(worker_id)
        metadata = make_rollout_metadata(
            max_steps=45,
            mpl_contract_sha256="d" * 64,
            teacher_path_length_contract_id=(
                "teacher_plan44_actual46_observed_polyline"
            ),
            teacher_actual_path_max_m=46.0,
            actual_path_length_m=1.0,
            episode_id=worker_id,
            collection_run_id="py0-run",
            runtime_instance_id=runtime_id,
            transition_ids=["{}:1".format(runtime_id)],
            observation_contract="reliable_exact_endpoint_snapshot",
            observation_source="reliable_exact_endpoint_snapshot",
            reliable_execution=True,
            telemetry_observation=False,
            state_depth_skew_ns=0,
            endpoint_identity_available=True,
            asynchronous_prefetch=False,
            asynchronous_prefetch_status="obsolete_for_reliable_exact",
            reliable_rows=1,
            legacy_rows=0,
            telemetry_lookup_count=0,
            snapshot_missing_count=0,
            frame_contract_failures=0,
            endpoint_identity_chain_valid=True,
        )
        save_rollout_episode(
            episode_path,
            {
                "depths": np.zeros((1, 1, 1), dtype=np.float32),
                "states": np.zeros((1, STATE_DIM), dtype=np.float32),
                "goals": np.zeros((1, GOAL_DIM), dtype=np.float32),
                "height_action_masks": np.ones(
                    (1, NUM_ACTIONS), dtype=np.bool_
                ),
                "execution_action_masks": np.ones(
                    (1, NUM_ACTIONS), dtype=np.bool_
                ),
                "behavior_actions": np.zeros((1,), dtype=np.int64),
                "prev_actions": np.full((1,), -1, dtype=np.int64),
                "poses_before": np.zeros((1, 4), dtype=np.float32),
                "velocities_before": np.zeros((1, 3), dtype=np.float32),
                "state_ids": np.zeros((1,), dtype=np.int64),
                "sim_time_ns": np.zeros((1,), dtype=np.int64),
                "state_stamp_ns": np.zeros((1,), dtype=np.int64),
                "depth_stamp_ns": np.zeros((1,), dtype=np.int64),
                "sensor_skew_ns": np.zeros((1,), dtype=np.int64),
                "primitive_actual_path_lengths_m": np.ones(
                    (1,), dtype=np.float32
                ),
                "start": np.zeros((3,), dtype=np.float32),
                "goal": np.ones((3,), dtype=np.float32),
            },
            metadata,
        )
        row["dataset_npz"] = "episodes/episode_{:06d}.npz".format(worker_id)
        summary = {
            "workers": 2,
            "worker_id": worker_id,
            "collection_run_id": "py0-run",
            "runtime_instance_id": runtime_id,
            **task_contract_fields(),
            "unity_player_sha256": artifact_identity["unity_player_sha256"],
            "runtime_assembly_sha256": artifact_identity["runtime_assembly_sha256"],
            "bridge_sha256": artifact_identity["bridge_sha256"],
            "attempted_total": 1,
            "accepted_total": 1,
            "offline_relabel_required": True,
            "teacher_planning_contract_id": "global_route_bounded_beam",
            "teacher_config": {"beam_depth": 3},
            "mpl_duration_s": 0.5,
            "mpl_forward_distance_m": 1.5,
            "mpl_contract_sha256": "d" * 64,
            "teacher_path_length_contract_id": "teacher_plan44_actual46_observed_polyline",
            "max_actual_path_length_m": 46.0,
            "max_steps": 45,
            "stream_horizon": 1,
            "max_sensor_skew_ms": 80.0,
            "max_endpoint_error_m": 0.6,
            "max_stream_drift_m": 0.35,
            "collection_config_contract_id": config["collection_config_contract_id"],
            "resolved_config_sha256": config["resolved_config_sha256"],
            "mission_index_sha256": "a" * 64,
            "collision_cache_sha256": "b" * 64,
            "code_version_sha256": "e" * 64,
            "shared_collection_config": config["shared_collection_config"],
            "resolved_cli_args": config["resolved_cli_args"],
            "worker_runtime": config["worker_runtime"],
            "runtime_artifact_identity": artifact_identity,
            "merge_compatibility_sha256": config["merge_compatibility_sha256"],
            "protocol_version": 4,
            **exact_endpoint_metadata(),
            "reliable_rows": 1,
            "legacy_rows": 0,
            "telemetry_lookup_count": 0,
            "snapshot_missing_count": 0,
            "state_depth_skew_max_ns": 0,
            "exact_primitive_frame_count_failures": 0,
            "frame_contract_failures": 0,
            "endpoint_identity_chain_valid": True,
        }
        fields = list(row.keys())
        with (worker_dir / "collection_report.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        (worker_dir / "collection_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        (worker_dir / "resolved_collection_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        rows_by_worker[worker_id] = row
        configs[worker_id] = config
    return workers, root / "merged"


def _append_failed_attempt(
    workers: Path,
    *,
    episode_id: int = 2,
    worker_id: int = 1,
    orphan_npz: bool = False,
) -> Path:
    """Add a non-accepted attempt without making it an accepted fixture."""

    worker_dir = workers / "worker_{:02d}".format(worker_id)
    report_path = worker_dir / "collection_report.csv"
    with report_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys())
    row = dict(rows[0])
    runtime_id = "runtime-{}".format(worker_id)
    row.update(
        {
            "episode_id": episode_id,
            "mission_id": "mission-{}".format(episode_id),
            "runtime_instance_id": runtime_id,
            "transition_ids": json.dumps(["{}:failed".format(runtime_id)]),
            "execute_ok": False,
            "success": False,
            "stop_reason": "collector_error",
            "actual_path_length_m": "",
            "error": "PHYSICS_COMPLETE_OBSERVATION_FAILED",
        }
    )
    orphan_path = worker_dir / "episodes" / "episode_{:06d}.npz".format(episode_id)
    if orphan_npz:
        orphan_path.write_bytes(b"not-a-rollout")
        row["dataset_npz"] = "episodes/{}".format(orphan_path.name)
    else:
        row["dataset_npz"] = ""
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows + [row])
    return orphan_path


def test_nonaccepted_collector_errors_do_not_block_clean_index(
    tmp_path: Path, monkeypatch
):
    workers, output = _write_exact_shards(tmp_path, monkeypatch)
    _append_failed_attempt(workers)

    assert merge_worker_rollouts(
        workers_dir=workers,
        out_dir=output,
        expected_workers=2,
        target_accepted=1,
    ) == 0
    summary = json.loads(
        (output / "collection_summary.json").read_text(encoding="utf-8")
    )
    with (output / "rollout_index.csv").open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    assert len(index_rows) == 1
    assert summary["accepted_total"] == 2
    assert summary["accepted_dataset_count"] == 1
    assert summary["raw_collector_error_total"] == 1
    assert summary["runtime_quality_pass"] is False
    assert summary["accepted_dataset_quality_pass"] is True


def test_failed_attempt_orphan_artifacts_are_removed(
    tmp_path: Path, monkeypatch
):
    workers, output = _write_exact_shards(tmp_path, monkeypatch)
    orphan_path = _append_failed_attempt(workers, orphan_npz=True)
    unreferenced = workers / "worker_00" / "episodes" / "orphan.npz"
    unreferenced.write_bytes(b"orphan")
    partial = workers / "worker_00" / "rollout_index.partial.csv"
    partial.write_text("partial", encoding="utf-8")

    assert merge_worker_rollouts(
        workers_dir=workers,
        out_dir=output,
        expected_workers=2,
        target_accepted=2,
    ) == 0
    assert not orphan_path.exists()
    assert not unreferenced.exists()
    assert not partial.exists()
    assert (workers / "worker_00" / "episodes" / "episode_000000.npz").exists()
    summary = json.loads(
        (output / "collection_summary.json").read_text(encoding="utf-8")
    )
    assert summary["failed_attempt_npz_removed"] == 1
    assert summary["orphan_artifact_removed_count"] >= 1


def test_invalid_accepted_npz_fails_closed(tmp_path: Path, monkeypatch):
    workers, output = _write_exact_shards(tmp_path, monkeypatch)
    invalid = workers / "worker_00" / "episodes" / "episode_000000.npz"
    invalid.write_bytes(b"invalid-npz")

    assert merge_worker_rollouts(
        workers_dir=workers,
        out_dir=output,
        expected_workers=2,
        target_accepted=2,
    ) == 1
    assert not (output / "rollout_index.csv").exists()
    summary = json.loads(
        (output / "collection_summary.json").read_text(encoding="utf-8")
    )
    assert summary["accepted_dataset_quality_pass"] is False
    assert summary["accepted_dataset_invalid_npz_count"] == 1


def test_missing_accepted_npz_fails_closed(tmp_path: Path, monkeypatch):
    workers, output = _write_exact_shards(tmp_path, monkeypatch)
    missing = workers / "worker_00" / "episodes" / "episode_000000.npz"
    missing.unlink()

    assert merge_worker_rollouts(
        workers_dir=workers,
        out_dir=output,
        expected_workers=2,
        target_accepted=2,
    ) == 1
    assert not (output / "rollout_index.csv").exists()
    summary = json.loads(
        (output / "collection_summary.json").read_text(encoding="utf-8")
    )
    assert summary["accepted_dataset_quality_pass"] is False
    assert summary["accepted_dataset_missing_npz_count"] == 1


def test_insufficient_valid_accepted_fails_closed(tmp_path: Path, monkeypatch):
    workers, output = _write_exact_shards(tmp_path, monkeypatch)

    assert merge_worker_rollouts(
        workers_dir=workers,
        out_dir=output,
        expected_workers=2,
        target_accepted=3,
    ) == 1
    assert not (output / "rollout_index.csv").exists()
    summary = json.loads(
        (output / "collection_summary.json").read_text(encoding="utf-8")
    )
    assert summary["accepted_dataset_quality_pass"] is False
    assert summary["accepted_dataset_count"] == 2


def test_observation_pollution_fails_closed(tmp_path: Path, monkeypatch):
    workers, output = _write_exact_shards(tmp_path, monkeypatch)
    report_path = workers / "worker_00" / "collection_report.csv"
    with report_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys())
    rows[0]["observation_contract"] = "legacy_async_telemetry"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RuntimeError, match="observation contract"):
        merge_worker_rollouts(
            workers_dir=workers, out_dir=output, expected_workers=2
        )


@pytest.mark.parametrize(
    "field",
    (
        "legacy_rows",
        "telemetry_lookup_count",
        "snapshot_missing_count",
        "state_depth_skew_max_ns",
        "frame_contract_failures",
        "endpoint_identity_chain_valid",
    ),
)
def test_runtime_observation_pollution_fails_closed(
    tmp_path: Path, monkeypatch, field: str
):
    workers, output = _write_exact_shards(tmp_path, monkeypatch)
    report_path = workers / "worker_00" / "collection_report.csv"
    with report_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys())
    rows[0][field] = "false" if field == "endpoint_identity_chain_valid" else "1"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RuntimeError):
        merge_worker_rollouts(
            workers_dir=workers, out_dir=output, expected_workers=2
        )


def test_exact_merge_and_resume_are_idempotent(tmp_path: Path, monkeypatch):
    workers, output = _write_exact_shards(tmp_path, monkeypatch)
    assert merge_worker_rollouts(
        workers_dir=workers, out_dir=output, expected_workers=2
    ) == 0
    summary = json.loads(
        (output / "collection_summary.json").read_text(encoding="utf-8")
    )
    assert summary["observation_contract"] == "reliable_exact_endpoint_snapshot"
    assert summary["reliable_rows"] == 2
    assert summary["legacy_rows"] == 0
    assert summary["telemetry_lookup_count"] == 0
    assert summary["snapshot_missing_count"] == 0
    assert summary["state_depth_skew_max_ns"] == 0

    rows = []
    with (workers / "worker_00" / "collection_report.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows.extend(csv.DictReader(handle))
    ledger = CollectionResumeLedger.from_rows(
        rows, runtime_instance_id="runtime-0", run_id="py0-run"
    )
    assert ledger.attempted_total == 1
    assert ledger.accepted_total == 1
    with pytest.raises(ValueError, match="duplicate"):
        ledger.add_row(rows[0])


def test_exact_merge_accepts_worker_summary_without_aggregate_frame_counter(
    tmp_path: Path, monkeypatch
):
    workers, output = _write_exact_shards(tmp_path, monkeypatch)
    for worker_id in (0, 1):
        summary_path = workers / "worker_{:02d}".format(worker_id) / "collection_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.pop("frame_contract_failures")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert merge_worker_rollouts(
        workers_dir=workers, out_dir=output, expected_workers=2
    ) == 0
    merged = json.loads(
        (output / "collection_summary.json").read_text(encoding="utf-8")
    )
    assert merged["frame_contract_failures"] == 0


def test_exact_merge_rejects_blank_terminal_boolean_before_aggregation(
    tmp_path: Path, monkeypatch
):
    workers, output = _write_exact_shards(tmp_path, monkeypatch)
    report_path = workers / "worker_00" / "collection_report.csv"
    with report_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys())
    rows[0]["collision"] = ""
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RuntimeError, match="collision"):
        merge_worker_rollouts(
            workers_dir=workers, out_dir=output, expected_workers=2
        )


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("episode", "duplicate episode_id"),
        ("mission", "duplicate mission_id"),
        ("transition", "duplicate transition_id"),
    ],
)
def test_exact_merge_rejects_cross_worker_collisions(
    tmp_path: Path, monkeypatch, mutation: str, pattern: str
):
    workers, output = _write_exact_shards(tmp_path, monkeypatch)
    report_path = workers / "worker_01" / "collection_report.csv"
    with report_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys())
    if mutation == "episode":
        rows[0]["episode_id"] = "0"
    elif mutation == "mission":
        rows[0]["mission_id"] = "mission-0"
    else:
        rows[0]["transition_ids"] = json.dumps(["runtime-0:1"])
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match=pattern):
        merge_worker_rollouts(
            workers_dir=workers, out_dir=output, expected_workers=2
        )


def test_twelve_worker_runtime_identity_and_ports_are_isolated(tmp_path: Path):
    specs = build_worker_runtime_specs(
        worker_count=12,
        runtime_instance_template="py0-{training_run_id}-{runtime_launch_nonce}-w{worker_id:02d}",
        ros_root=tmp_path / "ros",
        training_run_id="run",
        runtime_launch_nonce="nonce",
    )
    assert len({spec.runtime_instance_id for spec in specs}) == 12
    ports = [port for spec in specs for port in spec.all_ports]
    assert len(ports) == len(set(ports))


def test_rollout_shard_owner_is_deterministic_and_identity_bound(tmp_path: Path):
    identity = RolloutShardIdentity(
        worker_id=3, runtime_instance_id="runtime-3", collection_run_id="run"
    )
    paths = worker_shard_paths(tmp_path / "workers", identity.worker_id)
    assert paths["root"].name == "worker_03"
    assert paths["report"].parent == paths["root"]
    assert paths["episodes"].parent == paths["root"]
    with pytest.raises(ValueError, match="runtime_instance_id"):
        RolloutShardIdentity(
            worker_id=3, runtime_instance_id="", collection_run_id="run"
        )


def test_runtime_worker_failure_cleanup_terminates_all_children(tmp_path: Path):
    class _FakeProcess:
        def __init__(self):
            self.terminate_calls = 0
            self.wait_calls = 0
            self.kill_calls = 0

        def terminate(self):
            self.terminate_calls += 1

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 0

        def kill(self):
            self.kill_calls += 1

    spec = build_worker_runtime_specs(
        worker_count=1,
        runtime_instance_template="cleanup-{worker_id}",
        ros_root=tmp_path / "ros",
    )[0]
    worker = RuntimeWorker(
        spec=spec,
        out_dir=tmp_path,
        unity_bin=Path("/does/not/exist"),
        window_width=320,
        window_height=240,
        startup_timeout_s=1.0,
    )
    processes = [_FakeProcess() for _ in range(4)]
    worker.collector, worker.bridge, worker.unity, worker.roscore = processes
    worker.stop()
    assert all(process.terminate_calls == 1 for process in processes)
    assert all(process.wait_calls >= 1 for process in processes)
    assert all(process.kill_calls == 0 for process in processes)
