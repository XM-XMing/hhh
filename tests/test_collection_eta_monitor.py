"""Focused tests for aggregate collection ETA reporting."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_collection_monitor_aggregates_worker_eta(tmp_path: Path, capsys):
    root = tmp_path / "rollouts"
    _write_json(
        root / "collection_summary.json",
        {"workers": 2, "attempted_total": 0, "accepted_total": 0},
    )
    for worker_id in (0, 1):
        _write_json(
            root / "workers" / "worker_{}".format(worker_id) / "collection_progress.json",
            {
                "status": "running",
                "attempted_total": 10,
                "accepted_total": 5,
                "reliable_rows": 5,
                "elapsed_s": 10.0,
                "accepted_throughput_per_s": 0.5,
                "target_accepted": 100,
            },
        )

    from planning.teacher.collection_monitor import (
        collection_progress_snapshot,
        print_collection_progress,
    )

    snapshot = collection_progress_snapshot(root)
    assert snapshot["attempted"] == 20
    assert snapshot["accepted"] == 10
    assert snapshot["accepted_throughput_per_s"] == 1.0
    assert math.isclose(snapshot["eta_h"], 90.0 / 3600.0)
    print_collection_progress(snapshot)
    assert "eta_h" in capsys.readouterr().out


def test_collection_monitor_marks_reached_target_ready(tmp_path: Path):
    root = tmp_path / "rollouts"
    _write_json(
        root / "collection_summary.json",
        {"workers": 1, "attempted_total": 10, "accepted_total": 10},
    )
    _write_json(
        root / "workers" / "worker_0" / "collection_progress.json",
        {
            "status": "completed",
            "attempted_total": 10,
            "accepted_total": 10,
            "reliable_rows": 10,
            "elapsed_s": 1.0,
            "accepted_throughput_per_s": 1.0,
            "target_accepted": 10,
            "eta_status": "CALIBRATING",
        },
    )

    from planning.teacher.collection_monitor import collection_progress_snapshot

    snapshot = collection_progress_snapshot(root)
    assert snapshot["eta_h"] == 0.0
    assert snapshot["eta_status"] == "READY"


def test_collection_monitor_marks_exhausted_pool_unreachable(tmp_path: Path):
    root = tmp_path / "rollouts"
    _write_json(
        root / "collection_summary.json",
        {"workers": 2, "attempted_total": 20, "accepted_total": 10},
    )
    for worker_id in (0, 1):
        _write_json(
            root / "workers" / "worker_{}".format(worker_id) / "collection_progress.json",
            {
                "status": "completed",
                "complete": True,
                "attempted_total": 10,
                "accepted_total": 5,
                "reliable_rows": 5,
                "elapsed_s": 10.0,
                "accepted_throughput_per_s": 1000000.0,
                "target_accepted": 100,
            },
        )

    from planning.teacher.collection_monitor import collection_progress_snapshot

    snapshot = collection_progress_snapshot(root)
    assert snapshot["active_workers"] == 0
    assert snapshot["mission_pool_exhausted"] is True
    assert snapshot["eta_status"] == "TARGET_UNREACHABLE"
    assert snapshot["eta_h"] is None
