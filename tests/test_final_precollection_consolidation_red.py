"""RED characterization tests for the final pre-collection consolidation.

These tests intentionally describe the new canonical seams before their
implementation exists.  They are kept small so the first run proves that the
old embedded cache builder, route-store facade, and list-based preparation
path are not being mistaken for the final contracts.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_voxel_cache_has_one_data_layer_owner_and_cli(tmp_path: Path):
    from planning.data.voxel_cache import VoxelCacheConfig, build_voxel_cache

    point_cloud = tmp_path / "forest.bin"
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.11, 0.0, 0.0], [0.0, 0.21, 0.0]],
        dtype=np.float32,
    )
    point_cloud.write_bytes(np.asarray([len(points)], dtype="<u4").tobytes() + points.tobytes())
    cache = tmp_path / "forest.npz"
    result = build_voxel_cache(
        VoxelCacheConfig(
            map_bin=point_cloud,
            cache_npz=cache,
            file_frame="ros",
            voxel_size=0.10,
            data_offset_bytes=-1,
            chunk_points=2,
        )
    )
    assert result.cache_path == cache.resolve()
    assert result.metadata["voxel_size"] == 0.10
    assert result.metadata["occupied_count"] == 3
    assert result.metadata["point_cloud_sha256"]
    assert result.metadata["cache_sha256"]

    help_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_voxel_cache.py"), "--help"],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--map-bin" in help_result.stdout


def test_mission_route_store_data_module_is_the_implementation_owner():
    from planning.data.mission_routes import MissionRouteStore, MissionRouteStoreWriter

    assert MissionRouteStore.__module__ == "planning.data.mission_routes"
    assert MissionRouteStoreWriter.__module__ == "planning.data.mission_routes"
    assert not (ROOT / "python/planning/mission/route_store.py").exists()


def test_streaming_preparation_routes_once_and_audits_immediately(tmp_path: Path):
    from planning.mission.preparation import MissionPreparationConfig, MissionPreparationRun

    candidates = [
        {"episode_id": 0, "mission_id": "m0", "start_x": 0, "goal_x": 1},
        {"episode_id": 1, "mission_id": "m1", "start_x": 1, "goal_x": 2},
        {"episode_id": 2, "mission_id": "m2", "start_x": 2, "goal_x": 3},
    ]
    route_calls = []
    audit_calls = []
    appended = []

    def route(candidate):
        route_calls.append(candidate["episode_id"])
        return np.asarray(
            [[candidate["start_x"], 0.0, 1.5], [candidate["goal_x"], 0.0, 1.5]],
            dtype=np.float32,
        )

    def audit(candidate, route_value):
        audit_calls.append(candidate["episode_id"])
        return {"result": "success" if candidate["episode_id"] != 1 else "dead_end"}

    class Writer:
        def append(self, route_value, *, goal=None):
            appended.append(np.asarray(route_value).copy())
            return len(appended) - 1

        def set_candidate_index_sha256(self, value):
            self.candidate_sha = value

        def commit(self):
            return self

        def close(self):
            return None

        metadata = {
            "contract_id": "test-route-store",
            "schema_version": 1,
            "points_sha256": "0" * 64,
            "offsets_sha256": "1" * 64,
        }
        meta_path = tmp_path / "routes.meta.json"

    candidate_csv = tmp_path / "candidates.csv"
    missions_csv = tmp_path / "missions.csv"
    run = MissionPreparationRun(
        candidates=iter(candidates),
        route_planner=route,
        mission_auditor=audit,
        route_store_writer=Writer(),
        candidate_output=candidate_csv,
        mission_output=missions_csv,
        config=MissionPreparationConfig(
            required_passing=2,
            max_candidates=3,
            max_sampling_attempts=3,
            max_inflight_results=2,
        ),
    )
    result = run.execute()

    assert route_calls == [0, 1, 2]
    assert audit_calls == [0, 1, 2]
    assert len(appended) == 3
    assert result.passing_count == 2
    with missions_csv.open(newline="", encoding="utf-8") as handle:
        assert [row["mission_id"] for row in csv.DictReader(handle)] == ["m0", "m2"]


def test_preparation_audit_journal_accepts_rejection_before_success(tmp_path: Path):
    from planning.mission.preparation import MissionPreparationConfig, MissionPreparationRun

    candidates = [
        {"episode_id": 0, "mission_id": "m0"},
        {"episode_id": 1, "mission_id": "m1"},
    ]

    def route(candidate):
        if candidate["episode_id"] == 0:
            return "no_route", 0.0, 0.0, None
        return np.asarray([[0.0, 0.0, 1.5], [1.0, 0.0, 1.5]], dtype=np.float32)

    def audit(_candidate, _route):
        return {"result": "success"}

    class Writer:
        def __init__(self):
            self.metadata = {}
            self.routes = []

        def append(self, route_value, *, goal=None):
            self.routes.append(np.asarray(route_value).copy())
            return len(self.routes) - 1

        def set_candidate_index_sha256(self, value):
            self.candidate_sha = value

        def commit(self):
            return self

        def close(self):
            return None

    run = MissionPreparationRun(
        candidates=iter(candidates),
        route_planner=route,
        mission_auditor=audit,
        route_store_writer=Writer(),
        candidate_output=tmp_path / "candidates.csv",
        mission_output=tmp_path / "missions.csv",
        config=MissionPreparationConfig(
            required_passing=1,
            max_candidates=2,
            max_sampling_attempts=2,
        ),
    )

    result = run.execute()
    assert result.candidate_count == 2
    assert result.passing_count == 1


def test_preparation_publishes_collector_path_contract(tmp_path: Path):
    from planning.contracts.teacher_path import (
        TEACHER_PATH_LENGTH_CONTRACT_ID,
        TEACHER_PLAN_PATH_MAX_M,
        validate_plan_filtered_mission_rows,
    )
    from planning.mission.preparation import MissionPreparationConfig, MissionPreparationRun

    candidates = [{"episode_id": 0, "mission_id": "m0"}]

    def route(_candidate):
        return np.asarray(
            [[0.0, 0.0, 1.5], [1.0, 0.0, 1.5]], dtype=np.float32
        )

    def audit(_candidate, _route):
        return {
            "result": "success",
            "executed_path_length_m": 42.0,
            "path_stretch": 1.05,
            "path_length_contract_id": TEACHER_PATH_LENGTH_CONTRACT_ID,
            "plan_path_max_m": TEACHER_PLAN_PATH_MAX_M,
        }

    class Writer:
        metadata = {}

        def append(self, _route_value, *, goal=None):
            return 0

        def set_candidate_index_sha256(self, _value):
            return None

        def commit(self):
            return self

        def close(self):
            return None

    run = MissionPreparationRun(
        candidates=iter(candidates),
        route_planner=route,
        mission_auditor=audit,
        route_store_writer=Writer(),
        candidate_output=tmp_path / "candidates.csv",
        mission_output=tmp_path / "missions.csv",
        config=MissionPreparationConfig(
            required_passing=1,
            max_candidates=1,
            max_sampling_attempts=1,
        ),
    )
    run.execute()

    with (tmp_path / "missions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_plan_filtered_mission_rows(rows)
    assert rows[0]["teacher_path_length_contract_id"] == TEACHER_PATH_LENGTH_CONTRACT_ID
    assert float(rows[0]["teacher_plan_path_length_m"]) == 42.0


def test_merge_rebases_worker_route_store_reference_to_merged_index(tmp_path: Path):
    from planning.data.rollout_merge import _normalise_artifact_reference

    merged_dir = tmp_path / "collection"
    report_dir = merged_dir / "workers" / "worker_00"
    source_store = tmp_path / "preparation" / "mission_routes.meta.json"
    report_dir.mkdir(parents=True)
    source_store.parent.mkdir(parents=True)
    source_store.write_text("{}", encoding="utf-8")

    worker_reference = os.path.relpath(str(source_store), str(report_dir))
    merged_reference = _normalise_artifact_reference(
        worker_reference,
        source_dir=report_dir,
        output_dir=merged_dir,
    )

    assert (merged_dir / merged_reference).resolve() == source_store.resolve()


def test_preparation_resume_skips_committed_candidates_without_replanning(tmp_path: Path):
    from planning.mission.preparation import MissionPreparationConfig, MissionPreparationRun

    candidates = [
        {"episode_id": 0, "mission_id": "m0"},
        {"episode_id": 1, "mission_id": "m1"},
        {"episode_id": 2, "mission_id": "m2"},
    ]
    routes = []
    fail_once = {"enabled": True}

    class Writer:
        def __init__(self):
            self.metadata = {}

        @property
        def mission_count(self):
            return len(routes)

        def append(self, route_value, *, goal=None):
            routes.append(np.asarray(route_value).copy())
            return len(routes) - 1

        def set_candidate_index_sha256(self, value):
            self.candidate_sha = value

        def commit(self):
            return self

        def close(self):
            return None

    def route(candidate):
        if candidate["episode_id"] == 2 and fail_once["enabled"]:
            fail_once["enabled"] = False
            raise RuntimeError("controlled interruption")
        return np.asarray([[0.0, 0.0, 1.5], [1.0, 0.0, 1.5]], dtype=np.float32)

    def audit(candidate, route_value):
        return {"result": "success"}

    config = MissionPreparationConfig(
        required_passing=3,
        max_candidates=3,
        max_sampling_attempts=3,
        checkpoint_interval=1,
    )
    kwargs = dict(
        candidates=iter(candidates),
        route_planner=route,
        mission_auditor=audit,
        candidate_output=tmp_path / "candidates.csv",
        mission_output=tmp_path / "missions.csv",
        config=config,
    )
    try:
        MissionPreparationRun(route_store_writer=Writer(), **kwargs).execute()
    except RuntimeError as error:
        assert str(error) == "controlled interruption"
    else:
        raise AssertionError("controlled interruption did not fire")

    resumed = MissionPreparationRun(
        route_store_writer=Writer(),
        candidates=iter(candidates),
        config=MissionPreparationConfig(
            required_passing=3,
            max_candidates=3,
            max_sampling_attempts=3,
            checkpoint_interval=1,
            resume=True,
        ),
        **{
            key: value
            for key, value in kwargs.items()
            if key not in {"config", "candidates"}
        },
    ).execute()
    assert resumed.resumed_candidate_count == 2
    assert resumed.replanned_committed_route_count == 0
    assert len(routes) == 3


def test_formal_seed_owner_is_stable_and_uses_2026():
    from planning.common.random import DEFAULT_SEED, derive_seed

    assert DEFAULT_SEED == 2026
    assert derive_seed(2026, "mission", 0) == derive_seed(2026, "mission", 0)
    assert derive_seed(2026, "mission", 0) != derive_seed(2026, "mission", 1)


def test_cmake_installs_formal_preparation_and_not_diagnostics():
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "scripts/build_voxel_cache.py" in cmake
    assert "scripts/prepare_teacher_missions.py" in cmake
    assert "scripts/check_base_system.py" not in cmake
    assert "scripts/visualize_rollout.py" not in cmake


def test_preparation_journals_and_result_window_are_bounded(tmp_path: Path):
    from planning.data.csv_journal import CsvJournal
    from planning.mission.preparation import MissionPreparationConfig

    journal = CsvJournal(
        tmp_path / "rows.journal.jsonl",
        fieldnames=("candidate_number", "status"),
        key_field="candidate_number",
        retain_rows=False,
    )
    for index in range(1000):
        journal.append({"candidate_number": index, "status": "ok"})
    assert journal.row_count == 1000
    assert journal._rows == []
    assert journal.compact(tmp_path / "rows.csv") == 1000
    journal.close()

    config = MissionPreparationConfig(workers=12, max_inflight_results=24)
    assert config.max_inflight_results <= config.workers * 2
    source = (ROOT / "python/planning/mission/preparation.py").read_text(encoding="utf-8")
    for forbidden in (
        "all_specs =",
        "all_routes =",
        "list(as_completed",
        "np.concatenate(all_routes",
        "np.vstack(all_routes",
    ):
        assert forbidden not in source


def test_sampling_attempt_budget_is_global_and_scale_safe():
    from planning.mission.sampling import resolve_sampling_max_attempts

    assert resolve_sampling_max_attempts(2_000_000, 0) == 8_000_000
    with __import__("pytest").raises(ValueError):
        resolve_sampling_max_attempts(2_000_000, 800_000)
