"""Contract tests for the diagnostic multi-action replay seam."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from planning.diagnostics.multi_action_replay import (
    ACTION_SOURCES,
    audit_multi_action_replay,
    load_multi_action_replay,
    plan_action_sources,
    write_multi_action_replay,
)


pytestmark = pytest.mark.unit


def _transition(index=0, *, source="BC", action=1):
    mask = np.asarray([1, 1, 0, 1], dtype=np.bool_)
    return {
        "state_id": "state-{}".format(index),
        "mission_id": "mission-{}".format(index),
        "episode_id": "episode-{}".format(index),
        "step_id": index,
        "route_id": "route-{}".format(index),
        "action_source": source,
        "action": action,
        "mask": mask,
        "reward": 1.25,
        "state_vector": np.full((3,), index, dtype=np.float32),
        "state_depth": np.full((2, 2), index, dtype=np.float32),
        "next_state_id": "next-state-{}".format(index),
        "next_state_vector": np.full((3,), index + 1, dtype=np.float32),
        "next_state_depth": np.full((2, 2), index + 1, dtype=np.float32),
        "next_mask": mask,
        "done": False,
        "transition_id": "transition-{}".format(index),
    }


def _metadata():
    return {
        "observation_contract": "reliable_exact_endpoint_snapshot",
        "task_contract_sha256": "t" * 64,
        "mpl_contract_sha256": "m" * 64,
        "source_run": "diagnostic-fixture",
    }


def test_round_trip_preserves_identity_and_arrays(tmp_path):
    path = tmp_path / "multi_action_replay"
    write_multi_action_replay(path, [_transition(0)], _metadata())

    loaded = load_multi_action_replay(path)

    assert loaded["metadata"]["schema_id"] == "multi_action_replay_v1"
    assert loaded["state_id"].tolist() == ["state-0"]
    assert loaded["action_source"].tolist() == ["BC"]
    np.testing.assert_array_equal(loaded["mask"], np.asarray([[1, 1, 0, 1]], dtype=np.uint8))


def test_schema_requires_all_action_sources_to_be_known(tmp_path):
    row = _transition(0, source="UNKNOWN")

    with pytest.raises(ValueError, match="action_source"):
        write_multi_action_replay(tmp_path / "replay", [row], _metadata())


def test_schema_rejects_action_outside_mask(tmp_path):
    row = _transition(0, action=2)

    with pytest.raises(ValueError, match="mask"):
        write_multi_action_replay(tmp_path / "replay", [row], _metadata())


def test_audit_reports_counts_and_source_ratios(tmp_path):
    rows = [
        _transition(0, source="BC", action=1),
        _transition(1, source="TEACHER", action=3),
    ]
    path = tmp_path / "replay"
    write_multi_action_replay(path, rows, _metadata())

    report = audit_multi_action_replay(path, min_states=2, min_transitions=2)

    assert report["status"] == "PASS"
    assert report["state_count"] == 2
    assert report["transition_count"] == 2
    assert report["mean_actions_per_state"] == 1.0
    assert report["source_counts"] == {"BC": 1, "TEACHER": 1, "NEIGHBOR": 0, "RANDOM": 0}
    assert report["unique_action_ratio"] == 1.0
    assert report["production_replay"] is False
    assert report["training_consumed"] is False


def test_audit_fails_below_required_scale(tmp_path):
    path = tmp_path / "replay"
    write_multi_action_replay(path, [_transition(0)], _metadata())

    report = audit_multi_action_replay(path, min_states=2, min_transitions=2500)

    assert report["status"] == "FAIL"
    assert "state_count_below_minimum" in report["failures"]
    assert "transition_count_below_minimum" in report["failures"]


def test_action_planner_is_masked_deterministic_and_diverse():
    mask = np.asarray([True, True, False, True, True, True], dtype=np.bool_)

    first = plan_action_sources(
        mask,
        bc_action=1,
        teacher_action=3,
        neighbor_actions=[4, 1, 2],
        random_count=3,
        seed=55,
    )
    second = plan_action_sources(
        mask,
        bc_action=1,
        teacher_action=3,
        neighbor_actions=[4, 1, 2],
        random_count=3,
        seed=55,
    )

    assert first == second
    assert first[0] == ("BC", 1)
    assert ("TEACHER", 3) in first
    assert all(mask[action] for _, action in first)
    assert len({action for _, action in first}) == len(first)
    assert {source for source, _ in first}.issuperset({"BC", "TEACHER", "NEIGHBOR", "RANDOM"})
    assert set(ACTION_SOURCES) == {"BC", "TEACHER", "NEIGHBOR", "RANDOM"}


def test_audit_cli_writes_machine_readable_report(tmp_path):
    replay = tmp_path / "replay"
    write_multi_action_replay(replay, [_transition(0)], _metadata())
    script = Path(__file__).parents[1] / "scripts" / "audit_multi_action_replay.py"
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--replay-dir",
            str(replay),
            "--min-states",
            "1",
            "--min-transitions",
            "1",
            "--output",
            str(output),
        ],
        cwd=str(Path(__file__).parents[1]),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MULTI_ACTION_REPLAY_AUDIT=PASS" in result.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"


def test_collector_help_is_available_without_runtime_startup():
    script = Path(__file__).parents[1] / "scripts" / "collect_multi_action_replay.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(Path(__file__).parents[1]),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--max-states" in result.stdout
    assert "--min-transitions" in result.stdout
