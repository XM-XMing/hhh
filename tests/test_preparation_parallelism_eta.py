"""Regression tests for bounded preparation parallelism and ETA output."""

from __future__ import annotations

import csv
import json
import time
from itertools import islice
from pathlib import Path

import numpy as np


def _parallel_fixture_worker(task):
    """Small picklable worker used to exercise the parent orchestration seam."""

    attempt, candidate = task
    row = dict(candidate)
    row["sample_attempts"] = int(attempt)
    route = np.asarray(
        [[float(row["start_x"]), 0.0, 1.5], [float(row["goal_x"]), 0.0, 1.5]],
        dtype=np.float32,
    )
    return {
        "safe": True,
        "candidate": row,
        "route_value": ("ok", 1.0, 1.0, route),
        "audit": {"result": "success"},
    }


def _failing_parallel_fixture_worker(task):
    attempt, _candidate = task
    if int(attempt) == 2:
        raise RuntimeError("intentional uncommitted worker failure")
    return _parallel_fixture_worker(task)


def _candidate_rows(count: int):
    for index in range(int(count)):
        yield {
            "episode_id": -1,
            "mission_id": "m{}".format(index),
            "start_x": float(index),
            "start_y": 0.0,
            "start_z": 1.5,
            "start_yaw_deg": 0.0,
            "goal_x": float(index + 1),
            "goal_y": 0.0,
            "goal_z": 1.5,
            "sample_attempts": index + 1,
        }


def test_parallel_preparation_does_not_call_parent_route_or_teacher_audit(tmp_path: Path):
    from planning.mission.preparation import (
        MissionPreparationConfig,
        MissionPreparationRun,
    )

    def parent_route_must_not_run(_candidate):
        raise AssertionError("parent route planner was called")

    def parent_audit_must_not_run(_candidate, _route):
        raise AssertionError("parent Teacher audit was called")

    class Writer:
        metadata = {}

        def __init__(self):
            self.routes = []

        def append(self, route, *, goal=None):
            self.routes.append(np.asarray(route).copy())
            return len(self.routes) - 1

        def set_candidate_index_sha256(self, _value):
            return None

        def commit(self):
            return self

        def close(self):
            return None

    result = MissionPreparationRun(
        candidates=_candidate_rows(3),
        route_planner=parent_route_must_not_run,
        mission_auditor=parent_audit_must_not_run,
        route_store_writer=Writer(),
        candidate_output=tmp_path / "candidates.csv",
        mission_output=tmp_path / "missions.csv",
        config=MissionPreparationConfig(
            required_passing=2,
            max_candidates=3,
            max_sampling_attempts=3,
            workers=2,
            max_inflight_results=2,
        ),
        parallel_route_payload={},
        parallel_worker=_parallel_fixture_worker,
    ).execute()

    assert result.passing_count == 2
    with (tmp_path / "missions.csv").open(newline="", encoding="utf-8") as handle:
        assert [row["mission_id"] for row in csv.DictReader(handle)] == ["m0", "m1"]


def test_raw_candidate_sequence_is_independent_of_worker_count():
    from planning.mission.sampling import iter_raw_mission_candidates

    def sequence(_workers):
        rows = iter_raw_mission_candidates(
            seed=2026,
            max_attempts=20,
            max_accepted=20,
        )
        return [
            (
                int(row["sample_attempts"]),
                tuple(float(value) for value in row["start"]),
                tuple(float(value) for value in row["goal"]),
            )
            for row in islice(rows, 8)
        ]

    assert sequence(1) == sequence(3) == sequence(12)


def test_raw_candidate_stream_can_omit_fixed_smoke_mission_for_holdout():
    from planning.mission.sampling import iter_raw_mission_candidates

    first = next(
        iter_raw_mission_candidates(
            seed=3026,
            max_attempts=2,
            max_accepted=2,
            include_canonical_first=False,
        )
    )

    assert first["sample_attempts"] == 1
    assert first["start"] != [0.0, 0.0, 2.0, 0.0]
    assert first["goal"] != [40.0, 0.0, 2.0]


def test_preparation_progress_persists_eta_and_human_line(tmp_path: Path, capsys):
    from planning.mission.preparation import (
        MissionPreparationConfig,
        MissionPreparationRun,
    )

    candidate = {
        "episode_id": 0,
        "mission_id": "m0",
        "sample_attempts": 1,
        "start_x": 0.0,
        "start_y": 0.0,
        "start_z": 1.5,
        "goal_x": 1.0,
        "goal_y": 0.0,
        "goal_z": 1.5,
    }

    class Writer:
        metadata = {}

        def append(self, _route, *, goal=None):
            return 0

        def set_candidate_index_sha256(self, _value):
            return None

        def commit(self):
            return self

        def close(self):
            return None

    run = MissionPreparationRun(
        candidates=iter([candidate]),
        route_planner=lambda _candidate: np.asarray(
            [[0.0, 0.0, 1.5], [1.0, 0.0, 1.5]], dtype=np.float32
        ),
        mission_auditor=lambda _candidate, _route: {"result": "success"},
        route_store_writer=Writer(),
        candidate_output=tmp_path / "candidates.csv",
        mission_output=tmp_path / "missions.csv",
        config=MissionPreparationConfig(
            required_passing=1,
            max_candidates=1,
            max_sampling_attempts=1,
            checkpoint_interval=1,
        ),
    )
    run.execute()

    progress = json.loads(
        (tmp_path / "missions.preparation.progress.json").read_text(encoding="utf-8")
    )
    assert "progress" in progress
    assert "eta_h=" in progress["progress"]
    assert "throughput_per_s" in progress
    assert "eta_h" in progress
    assert "PROGRESS" in capsys.readouterr().out


def test_parallel_scheduler_exposes_bounded_inflight_window():
    from planning.mission.preparation import MissionPreparationConfig

    config = MissionPreparationConfig(workers=12, max_inflight_results=24)
    assert config.max_inflight_results <= config.workers * 2


def test_promoted_preparation_defaults_use_twenty_workers_and_full_inflight_window(
    monkeypatch,
):
    from planning.mission import preparation

    args = preparation.build_argument_parser().parse_args([])
    assert args.workers == 20
    assert args.max_inflight_results == 40
    assert preparation.DEFAULT_PREPARATION_WORKERS == 20

    config = preparation.MissionPreparationConfig()
    assert config.workers == 20
    assert config.max_inflight_results == 40

    captured = {}

    class Executor:
        def __init__(self, **kwargs):
            captured["max_workers"] = kwargs["max_workers"]

        def shutdown(self, wait=True):
            captured["wait"] = wait

    monkeypatch.setattr(preparation, "ProcessPoolExecutor", Executor)
    scheduler = preparation._BoundedPreparationRouteResults(
        candidates=iter(()),
        start_candidate_number=0,
        start_sampling_attempts=0,
        max_candidates=1,
        max_sampling_attempts=1,
        workers=20,
        max_inflight_results=40,
        route_payload={},
        max_accepted=1,
        worker=lambda _task: {},
    )
    scheduler.close()

    assert captured["max_workers"] == 20


def test_preparation_cli_exposes_typed_parallelism_and_progress_options():
    from planning.mission.preparation import build_argument_parser

    args = build_argument_parser().parse_args(
        ["--workers", "12", "--max-inflight-results", "24", "--collision-threads", "1", "--progress-interval-sec", "10"]
    )
    assert args.workers == 12
    assert args.max_inflight_results == 24
    assert args.collision_threads == 1
    assert args.progress_interval_sec == 10.0


def test_parallel_scheduler_preserves_result_order_across_worker_counts():
    from planning.mission.preparation import _BoundedPreparationRouteResults

    def collect(workers):
        scheduler = _BoundedPreparationRouteResults(
            candidates=_candidate_rows(6),
            start_candidate_number=0,
            start_sampling_attempts=0,
            max_candidates=6,
            max_sampling_attempts=6,
            workers=workers,
            max_inflight_results=min(2, workers * 2),
            route_payload={},
            max_accepted=6,
            worker=_parallel_fixture_worker,
        )
        try:
            return [
                (attempt, result["candidate"]["mission_id"])
                for attempt, _candidate, result in scheduler
            ]
        finally:
            scheduler.close()

    assert collect(2) == collect(3)


def test_parallel_preparation_resume_replays_only_uncommitted_attempt(tmp_path: Path):
    from planning.mission.preparation import (
        MissionPreparationConfig,
        MissionPreparationRun,
    )

    routes = []

    class Writer:
        metadata = {}

        @property
        def mission_count(self):
            return len(routes)

        def append(self, route, *, goal=None):
            routes.append(np.asarray(route).copy())
            return len(routes) - 1

        def set_candidate_index_sha256(self, _value):
            return None

        def commit(self):
            return self

        def close(self):
            return None

    common = dict(
        candidates=_candidate_rows(3),
        route_planner=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("parent route planner was called")
        ),
        mission_auditor=lambda _candidate, _route: (_ for _ in ()).throw(
            AssertionError("parent Teacher audit was called")
        ),
        route_store_writer=Writer(),
        candidate_output=tmp_path / "candidates.csv",
        mission_output=tmp_path / "missions.csv",
        parallel_route_payload={},
    )
    first = MissionPreparationRun(
        **common,
        parallel_worker=_failing_parallel_fixture_worker,
        config=MissionPreparationConfig(
            required_passing=3,
            max_candidates=3,
            max_sampling_attempts=3,
            workers=2,
            max_inflight_results=2,
            checkpoint_interval=1,
        ),
    )
    try:
        first.execute()
    except RuntimeError as error:
        assert "intentional uncommitted worker failure" in str(error)
    else:
        raise AssertionError("worker failure did not interrupt the run")

    resumed = MissionPreparationRun(
        **{
            **common,
            "candidates": _candidate_rows(3),
            "route_store_writer": Writer(),
            "parallel_worker": _parallel_fixture_worker,
            "config": MissionPreparationConfig(
                required_passing=3,
                max_candidates=3,
                max_sampling_attempts=3,
                workers=2,
                max_inflight_results=2,
                checkpoint_interval=1,
                resume=True,
            ),
        }
    )
    result = resumed.execute()
    assert result.resumed_candidate_count == 1
    assert result.replanned_committed_route_count == 0
    assert len(routes) == 3


def test_preparation_progress_reports_budget_risk_without_changing_artifacts(
    tmp_path: Path,
):
    from planning.mission.preparation import (
        MissionPreparationConfig,
        MissionPreparationRun,
    )

    class Writer:
        metadata = {}

    run = MissionPreparationRun(
        candidates=iter(()),
        route_planner=lambda _candidate: None,
        mission_auditor=lambda _candidate, _route: {"result": "success"},
        route_store_writer=Writer(),
        candidate_output=tmp_path / "candidates.csv",
        mission_output=tmp_path / "missions.csv",
        config=MissionPreparationConfig(
            required_passing=10,
            max_candidates=10,
            max_sampling_attempts=10,
            workers=1,
        ),
    )
    run._resume_sampling_attempts = 0
    run._resume_passing_count = 0
    run._write_progress(
        status="checkpoint",
        candidate_count=2,
        routed_count=2,
        passing_count=1,
        sampling_attempts=2,
        started=time.monotonic() - 10.0,
    )

    progress = json.loads(
        (tmp_path / "missions.preparation.progress.json").read_text(
            encoding="utf-8"
        )
    )
    assert progress["TARGET_AT_RISK"] == "YES"
    assert progress["ATTEMPT_BUDGET_AT_RISK"] == "YES"
    assert progress["estimated_candidates_required"] == 20
    assert progress["remaining_candidate_budget"] == 8
