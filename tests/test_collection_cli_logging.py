"""Focused public tests for the formal collection CLI/logging seam."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from planning.common.logging import (
    CollectionLogSession,
    bridge_log_path,
    collection_log_path,
    prepare_log_directories,
    worker_log_path,
    unity_log_path,
    write_resume_manifest,
)
from planning.runtime.collection_worker import RuntimeWorker
from planning.runtime.ports import build_worker_runtime_specs
from planning.runtime.process import ManagedProcess
from planning.teacher import parallel_collection
from planning.teacher.parallel_collection import build_argument_parser
from planning.teacher.collection_config import ParallelCollectionConfig


def _runtime_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    unity = tmp_path / "XMflight.x86_64"
    bridge = tmp_path / "unity_bridge_node"
    cache = tmp_path / "forest_voxels.npz"
    for path in (unity, bridge):
        path.write_text("binary\n", encoding="utf-8")
        path.chmod(0o755)
    cache.write_bytes(b"cache")
    return unity, bridge, cache


def _config(tmp_path: Path, **overrides) -> ParallelCollectionConfig:
    mission = tmp_path / "missions.csv"
    mission.write_text("episode_id\n0\n", encoding="utf-8")
    unity, bridge, cache = _runtime_files(tmp_path)
    values = {
        "unity_bin": unity,
        "bridge_binary": bridge,
        "collision_cache": cache,
        "collection_run_id": "test-run",
        "training_run_id": "test-run",
        "runtime_launch_nonce": "test-run",
    }
    values.update(overrides)
    return ParallelCollectionConfig.from_environment(
        mission_index=mission,
        out_dir=tmp_path / "rollouts",
        workers=2,
        environment={},
        default_run_id="default-run",
        overrides=values,
    )


def test_collection_help_exposes_typed_runtime_and_teacher_flags():
    parser = build_argument_parser()
    text = parser.format_help()
    assert parallel_collection.DEFAULT_COLLECTION_WORKERS == 20
    args = parser.parse_args(
        ["--mission-index", "missions.csv", "--out-dir", "rollouts"]
    )
    _, _, resolved_workers, _ = parallel_collection._resolved_cli_values(parser, args)
    assert resolved_workers == 20
    for flag in (
        "--target-accepted",
        "--max-steps",
        "--unity-bin",
        "--bridge",
        "--collision-cache",
        "--collision-backend",
        "--depth-safety-backend",
        "--collision-threads",
        "--reliable-exact-timeout",
        "--beam-depth",
        "--beam-width",
        "--beam-branching",
        "--beam-discount",
        "--global-route-resolution-m",
        "--global-route-lookahead-m",
        "--global-route-tracking-margin-m",
        "--disk-warning-free-gb",
        "--disk-stop-free-gb",
        "--disk-check-interval-sec",
        "--disk-safety-margin-gb",
        "--disk-estimated-episode-bytes",
    ):
        assert flag in text


def test_cli_values_override_environment_and_do_not_require_extra_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mission = tmp_path / "missions.csv"
    mission.write_text("episode_id\n0\n", encoding="utf-8")
    unity, bridge, cache = _runtime_files(tmp_path)
    monkeypatch.setenv("TARGET_ACCEPTED", "999")
    captured = []
    monkeypatch.setattr(parallel_collection, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(
        parallel_collection,
        "run_parallel_collection",
        lambda config: captured.append(config) or 0,
    )

    result = parallel_collection.main(
        [
            "--mission-index",
            str(mission),
            "--out-dir",
            str(tmp_path / "rollouts"),
            "--workers",
            "12",
            "--target-accepted",
            "60",
            "--max-episodes",
            "77",
            "--unity-bin",
            str(unity),
            "--bridge",
            str(bridge),
            "--collision-cache",
            str(cache),
            "--collision-backend",
            "cpp_cpu",
            "--depth-safety-backend",
            "cpp_native",
            "--collision-threads",
            "2",
            "--beam-depth",
            "3",
            "--beam-width",
            "8",
            "--beam-branching",
            "4",
            "--beam-discount",
            "0.95",
            "--global-route-resolution-m",
            "0.25",
            "--global-route-lookahead-m",
            "3.0",
            "--global-route-tracking-margin-m",
            "0.0",
            "--run-id",
            "cli-run",
            "--training-run-id",
            "train-run",
            "--runtime-launch-nonce",
            "nonce-run",
        ]
    )

    assert result == 0
    assert len(captured) == 1
    config = captured[0]
    assert config.target_accepted == 60
    assert config.max_episodes == 77
    assert config.workers_requested == 12
    assert config.collision_threads == 2
    assert config.collection_run_id == "cli-run"
    assert config.training_run_id == "train-run"
    assert config.runtime_launch_nonce == "nonce-run"
    assert config.beam_depth == 3
    assert config.beam_width == 8
    assert config.beam_branching == 4
    assert config.beam_discount == 0.95
    assert config.collector_extra_args == ()


def test_explicit_cli_config_matches_legacy_environment_values(tmp_path: Path):
    mission = tmp_path / "missions.csv"
    mission.write_text("episode_id\n0\n", encoding="utf-8")
    unity, bridge, cache = _runtime_files(tmp_path)
    environment = {
        "TARGET_ACCEPTED": "60",
        "MAX_EPISODES": "77",
        "MAX_STEPS": "45",
        "COLLISION_THREADS": "2",
        "VOXEL_SIZE": "0.1",
        "INFLATE_RADIUS": "0.35",
        "EXPECTED_PLANAR_DISTANCE": "40.0",
        "RELIABLE_EXACT_TIMEOUT": "10.0",
        "COLLECTION_RUN_ID": "cli-run",
        "TRAINING_RUN_ID": "train-run",
        "RUNTIME_LAUNCH_NONCE": "nonce-run",
        "UNITY_BIN": str(unity),
        "BRIDGE_BIN": str(bridge),
        "COLLISION_CACHE": str(cache),
        "PLANNING_COLLISION_BACKEND": "cpp_cpu",
        "PLANNING_DEPTH_SAFETY_BACKEND": "cpp_native",
        "BEAM_DEPTH": "3",
        "BEAM_WIDTH": "8",
        "BEAM_BRANCHING": "4",
        "BEAM_DISCOUNT": "0.95",
        "GLOBAL_ROUTE_RESOLUTION_M": "0.25",
        "GLOBAL_ROUTE_LOOKAHEAD_M": "3.0",
        "GLOBAL_ROUTE_TRACKING_MARGIN_M": "0.0",
    }
    old = ParallelCollectionConfig.from_environment(
        mission_index=mission,
        out_dir=tmp_path / "old",
        workers=12,
        environment=environment,
        default_run_id="default-run",
    )
    new = ParallelCollectionConfig.from_environment(
        mission_index=mission,
        out_dir=tmp_path / "new",
        workers=12,
        environment={},
        default_run_id="default-run",
        overrides={
            "target_accepted": 60,
            "max_episodes": 77,
            "max_steps": 45,
            "collision_threads": 2,
            "voxel_size": 0.1,
            "inflate_radius": 0.35,
            "expected_planar_distance": 40.0,
            "reliable_timeout_s": 10.0,
            "collection_run_id": "cli-run",
            "training_run_id": "train-run",
            "runtime_launch_nonce": "nonce-run",
            "unity_bin": unity,
            "bridge_binary": bridge,
            "collision_cache": cache,
            "collision_backend": "cpp_cpu",
            "depth_safety_backend": "cpp_native",
            "beam_depth": 3,
            "beam_width": 8,
            "beam_branching": 4,
            "beam_discount": 0.95,
            "global_route_resolution_m": 0.25,
            "global_route_lookahead_m": 3.0,
            "global_route_tracking_margin_m": 0.0,
        },
    )
    for field in (
        "target_accepted",
        "max_episodes",
        "max_steps",
        "collision_threads",
        "voxel_size",
        "inflate_radius",
        "expected_planar_distance",
        "reliable_timeout_s",
        "collection_run_id",
        "training_run_id",
        "runtime_launch_nonce",
        "collision_backend",
        "depth_safety_backend",
        "beam_depth",
        "beam_width",
        "beam_branching",
        "beam_discount",
        "global_route_resolution_m",
        "global_route_lookahead_m",
        "global_route_tracking_margin_m",
    ):
        assert getattr(old, field) == getattr(new, field)


def test_collection_log_is_append_only_and_marks_resume(tmp_path: Path):
    path = collection_log_path(tmp_path / "rollouts")
    first = CollectionLogSession(path, run_id="run", command="collect")
    first.event("first event", mission_index=1)
    first.close("PASS")
    before = path.read_text(encoding="utf-8")

    second = CollectionLogSession(path, run_id="run", command="collect")
    second.event("resume event")
    second.close("PASS")
    after = path.read_text(encoding="utf-8")

    assert before in after
    assert "SESSION START" in after
    assert "RESUME SESSION" in after
    assert after.count("SESSION END") == 2


def test_log_paths_are_unique_and_created(tmp_path: Path):
    root = tmp_path / "rollouts"
    prepare_log_directories(root, worker_count=2)
    assert worker_log_path(root, 0) != worker_log_path(root, 1)
    assert unity_log_path(root, 0) != unity_log_path(root, 1)
    assert bridge_log_path(root, 0) != bridge_log_path(root, 1)
    assert (root / "logs" / "workers").is_dir()
    assert (root / "logs" / "unity").is_dir()
    assert (root / "logs" / "bridge").is_dir()


def test_resume_manifest_rejects_identity_change_without_overwriting_history(
    tmp_path: Path,
):
    path = tmp_path / "command.json"
    write_resume_manifest(
        path,
        {"run_id": "run", "target_accepted": 2, "argv": ["first"]},
        immutable_keys=("run_id", "target_accepted"),
        schema="collection_command_v1",
    )
    write_resume_manifest(
        path,
        {"run_id": "run", "target_accepted": 2, "argv": ["resume"]},
        immutable_keys=("run_id", "target_accepted"),
        schema="collection_command_v1",
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert len(document["sessions"]) == 2
    assert document["sessions"][0]["argv"] == ["first"]
    with pytest.raises(ValueError, match="RESUME_IDENTITY_MISMATCH"):
        write_resume_manifest(
            path,
            {"run_id": "different", "target_accepted": 2, "argv": ["bad"]},
            immutable_keys=("run_id", "target_accepted"),
            schema="collection_command_v1",
        )


def test_worker_runtime_log_names_are_independent(tmp_path: Path):
    spec = build_worker_runtime_specs(
        worker_count=1,
        runtime_instance_template="log-{worker_id}",
        ros_root=tmp_path / "ros",
    )[0]
    worker = RuntimeWorker(
        spec=spec,
        out_dir=tmp_path,
        unity_bin=tmp_path / "unity",
        window_width=320,
        window_height=240,
        startup_timeout_s=1.0,
    )
    assert worker.worker_log_path == worker_log_path(tmp_path, 0)
    assert worker.unity_log_path == unity_log_path(tmp_path, 0)
    assert worker.bridge_log_path == bridge_log_path(tmp_path, 0)


def test_collector_argv_contains_typed_teacher_flags_only(tmp_path: Path):
    config = _config(tmp_path)
    spec = build_worker_runtime_specs(
        worker_count=1,
        runtime_instance_template="args-{worker_id}",
        ros_root=tmp_path / "ros",
    )[0]
    argv = parallel_collection._collector_args(
        config=config,
        spec=spec,
        worker_dir=config.out_dir / "workers" / "worker_00",
        global_stop_file=config.out_dir / ".collection_stop",
        collision_cache=config.collision_cache,
    )
    for flag, value in (
        ("--beam-depth", "3"),
        ("--beam-width", "8"),
        ("--beam-branching", "4"),
        ("--beam-discount", "0.95"),
        ("--global-route-resolution-m", "0.25"),
        ("--global-route-lookahead-m", "3.0"),
        ("--global-route-tracking-margin-m", "0.0"),
    ):
        assert value == argv[argv.index(flag) + 1]
    assert "COLLECTOR_EXTRA_ARGS" not in " ".join(argv)


def test_runtime_starts_unique_unity_and_bridge_log_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = []

    class FakeProcess:
        _next_pid = 100

        def __init__(self):
            self.pid = FakeProcess._next_pid
            FakeProcess._next_pid += 1

    def fake_start(**kwargs):
        calls.append(kwargs)
        return FakeProcess()

    monkeypatch.setattr(ManagedProcess, "start", staticmethod(fake_start))
    specs = build_worker_runtime_specs(
        worker_count=2,
        runtime_instance_template="runtime-{worker_id}",
        ros_root=tmp_path / "ros",
    )
    for spec in specs:
        RuntimeWorker(
            spec=spec,
            out_dir=tmp_path,
            unity_bin=tmp_path / "unity",
            bridge_binary=tmp_path / "bridge",
            window_width=320,
            window_height=240,
            startup_timeout_s=1.0,
        ).start_runtime()
    unity_logs = [
        call["argv"][call["argv"].index("-logFile") + 1]
        for call in calls
        if call["name"].startswith("unity-")
    ]
    bridge_logs = [str(call["log_path"]) for call in calls if call["name"].startswith("bridge-")]
    assert len(set(unity_logs)) == 2
    assert len(set(bridge_logs)) == 2
    assert all("/logs/unity/worker_" in value for value in unity_logs)
    assert all("/logs/bridge/worker_" in value for value in bridge_logs)


def test_invalid_runtime_identity_fails_closed_and_leaves_main_log(tmp_path: Path):
    config = _config(
        tmp_path,
        unity_bin=tmp_path / "missing-unity",
    )
    with pytest.raises(FileNotFoundError):
        parallel_collection.run_parallel_collection(config)
    assert collection_log_path(config.out_dir).is_file()
    assert "ERROR" in collection_log_path(config.out_dir).read_text(encoding="utf-8")


def test_collection_log_write_failure_does_not_escape_cleanup(capsys, tmp_path: Path):
    session = CollectionLogSession(
        collection_log_path(tmp_path / "rollouts"),
        run_id="run",
        command="collect",
    )

    class FailingHandle:
        def write(self, value):
            raise OSError("no space left on device")

        def flush(self):
            raise OSError("no space left on device")

        def close(self):
            return None

    session._handle.close()
    session._handle = FailingHandle()
    session.event("event after ENOSPC")
    session.close("FAIL")
    assert "collection log write failed" in capsys.readouterr().err


def test_disk_capacity_config_is_resolved_and_validated(tmp_path: Path):
    config = _config(
        tmp_path,
        disk_warning_free_gb=4.0,
        disk_stop_free_gb=2.0,
        disk_check_interval_s=5.0,
        disk_safety_margin_gb=1.0,
        disk_estimated_episode_bytes=1234,
    )
    assert config.disk_warning_free_gb == 4.0
    assert config.disk_stop_free_gb == 2.0
    assert config.disk_check_interval_s == 5.0
    assert config.disk_safety_margin_gb == 1.0
    assert config.disk_estimated_episode_bytes == 1234
