"""Real Player regression for the default-off execution transport audit.

The test is intentionally opt-in: it starts isolated ROS/Unity processes and
uses the separately built diagnostic Player.  It never points at the formal
Player and never changes the execution contract.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


if os.environ.get("PLANNING_TEST_UNITY") != "1":
    pytest.skip("set PLANNING_TEST_UNITY=1 to enable", allow_module_level=True)


PLANNING_DIR = Path(__file__).resolve().parents[1]
RUNNER = PLANNING_DIR / "scripts" / "evaluate_policy_unity_managed.sh"
CHECKPOINT = PLANNING_DIR / "data/bc/flight_20260717_6w/model/checkpoint_best.pt"
MISSIONS = PLANNING_DIR / "data/smoke/bc_flight_20260717_6w/holdout_seed55/fixed_dev_100.csv"
DIAGNOSTIC_PLAYER = Path(os.environ.get("PLANNING_DIAGNOSTIC_UNITY_BIN", ""))


if not DIAGNOSTIC_PLAYER.is_file() or not CHECKPOINT.is_file() or not MISSIONS.is_file():
    pytest.skip("diagnostic Unity Player or episode-81660 fixture is unavailable", allow_module_level=True)


def _run(tmp_path: Path, *, audit: bool, failing_unity_audit_path: bool = False):
    out_dir = tmp_path / ("audit_on" if audit else "audit_off")
    env = dict(os.environ)
    env.update({
        "UNITY_BIN": str(DIAGNOSTIC_PLAYER),
        "EVAL_REPRO_EPISODE_ID": "81660",
        "EVAL_EXECUTION_TRANSPORT_AUDIT": "1" if audit else "0",
        "PYTHON_EXECUTABLE": sys.executable,
    })
    if failing_unity_audit_path:
        # /dev/full is a file, so Unity's audit Directory.CreateDirectory
        # fails inside its audit-only try/catch before any file can be written.
        env["EVAL_UNITY_EXECUTION_TRANSPORT_AUDIT_PATH"] = "/dev/full/audit.json"
        out_dir = tmp_path / "audit_write_failure"
    result = subprocess.run(
        ["bash", str(RUNNER), str(CHECKPOINT), str(MISSIONS), str(out_dir), "1"],
        cwd=PLANNING_DIR,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    return result, out_dir


def _assert_success_episode(out_dir: Path) -> None:
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["success_count"] == 1
    assert summary["collision_count"] == 0
    with (out_dir / "steps" / "episode_081660.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 29


def test_diagnostic_player_execution_transport_audit_contract(tmp_path):
    off, off_dir = _run(tmp_path, audit=False)
    assert off.returncode == 0, off.stdout + off.stderr
    _assert_success_episode(off_dir)
    assert not list(off_dir.rglob("*execution_transport_audit*.json"))

    on, on_dir = _run(tmp_path, audit=True)
    assert on.returncode == 0, on.stdout + on.stderr
    _assert_success_episode(on_dir)
    unity = json.loads((on_dir / "runtime_logs" / "unity_execution_transport_audit.json").read_text())
    bridge = json.loads((on_dir / "runtime_logs" / "bridge_execution_transport_audit.json").read_text())
    planning = json.loads((on_dir / "execution_transport_audit_episode_081660.json").read_text())
    assert unity["audit_schema_version"] == bridge["audit_schema_version"] == planning["audit_schema_version"] == 2
    assert unity["audit_run_id"] == bridge["audit_run_id"] == planning["audit_run_id"]
    assert unity["runtime_identity"] == bridge["unity_runtime_identity"] == planning["unity_runtime_identity"]
    assert not unity["capture_overflow"] and not bridge["capture_overflow"] and not planning["capture_overflow"]
    assert len(unity["records"]) == len(bridge["records"]) == 29 * 25
    assert all(record["frame_applied"] for record in unity["records"])
    assert all(isinstance(record["try_send_return"], bool) for record in unity["records"])
    unity_by_execution = {}
    for record in unity["records"]:
        unity_by_execution.setdefault(record["execution_id"], []).append(record)
    assert len(unity_by_execution) == 29
    assert all(
        [record["frame_index"] for record in records] == list(range(25))
        for records in unity_by_execution.values()
    )
    for execution in planning["executions"]:
        assert [record["frame_index"] for record in execution["planning_received"]] == list(range(25))

    write_failure, failed_out_dir = _run(
        tmp_path, audit=True, failing_unity_audit_path=True
    )
    assert write_failure.returncode != 0
    incomplete_dir = Path(str(failed_out_dir) + ".incomplete")
    _assert_success_episode(incomplete_dir)
    assert (incomplete_dir / "execution_transport_audit_episode_081660.json").is_file()
    assert (incomplete_dir / "runtime_logs" / "bridge_execution_transport_audit.json").is_file()
    assert not (incomplete_dir / "runtime_logs" / "unity_execution_transport_audit.json").exists()
    assert "audit write failed" in (incomplete_dir / "runtime_logs" / "unity.log").read_text(encoding="utf-8")
