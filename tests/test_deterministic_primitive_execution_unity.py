"""RED Unity tests for deterministic execution of episode 451 primitive zero."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data/bc/flight_20260717_6w/model/checkpoint_best.pt"
MISSION_INDEX = (
    ROOT / "data/smoke/bc_flight_20260717_6w/holdout_seed55/fixed_dev_100.csv"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "188e41301b67cd8c53ffdd45d55684c5f7626b4a4208eca599e322870e8c31f7"
)
EPISODE_ID = 451
ACTION_ID = 52
COMMAND_FRAMES = 25


def _load_primitive(trace_path: Path):
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    return episode, episode["primitives"][0]


@pytest.fixture(scope="module")
def repeated_execution(tmp_path_factory):
    if not CHECKPOINT.is_file() or not MISSION_INDEX.is_file():
        pytest.skip("episode 451 deterministic execution test requires local BC/dev assets")
    output = tmp_path_factory.mktemp("episode451_primitive_execution")
    environment = dict(os.environ)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_command_tick_audit.sh"),
            str(CHECKPOINT),
            str(MISSION_INDEX),
            str(output),
            str(EPISODE_ID),
            "5",
        ],
        cwd=str(ROOT),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    report_path = output / "command_tick_audit_report.json"
    assert report_path.is_file(), result.stdout + "\n" + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    traces = sorted(output.glob("repeat_*/first_divergence_trace.json"))
    assert len(traces) == 5
    episodes_and_primitives = [_load_primitive(path) for path in traces]
    return report, episodes_and_primitives, result


@pytest.mark.unity
def test_primitive_execution_reports_unique_execution_ids_and_applied_frames(
    repeated_execution,
):
    _, episodes_and_primitives, _ = repeated_execution
    executions = [primitive.get("primitive_execution") for _, primitive in episodes_and_primitives]

    assert all(execution is not None for execution in executions), (
        "UnityForestEnv does not expose the execution_id/applied-frame contract"
    )
    execution_ids = [execution["execution_id"] for execution in executions]
    assert len(set(execution_ids)) == len(execution_ids)
    for execution in executions:
        applied_frames = execution["applied_frames"]
        assert [frame["frame_index"] for frame in applied_frames] == list(
            range(COMMAND_FRAMES)
        )


@pytest.mark.unity
def test_primitive_execution_applies_exactly_25_physics_ticks(repeated_execution):
    report, _, _ = repeated_execution

    assert report["effective_command_tick_counts"] == [COMMAND_FRAMES] * 5


@pytest.mark.unity
def test_endpoint_is_post_integration_state_for_frame_24(repeated_execution):
    _, episodes_and_primitives, _ = repeated_execution
    for _, primitive in episodes_and_primitives:
        execution = primitive.get("primitive_execution")
        assert execution is not None, "missing primitive_execution state acknowledgement"
        first_applied = int(execution["applied_frames"][0]["applied_state_id"])
        endpoint_state = int(execution["endpoint_state_id"])
        assert endpoint_state == first_applied + COMMAND_FRAMES - 1
        assert int(execution["applied_frames"][-1]["applied_state_id"]) == endpoint_state


@pytest.mark.unity
def test_repeated_episode451_primitive_zero_has_identical_endpoint(repeated_execution):
    _, episodes_and_primitives, _ = repeated_execution
    positions = [
        np.asarray(primitive["after"]["position"], dtype=np.float64)
        for _, primitive in episodes_and_primitives
    ]
    maximum_delta = max(
        float(np.linalg.norm(left - right))
        for left in positions
        for right in positions
    )

    assert maximum_delta < 1.0e-4


@pytest.mark.unity
def test_episode451_primitive_after_zero_does_not_diverge(repeated_execution):
    _, episodes_and_primitives, _ = repeated_execution
    for episode, primitive in episodes_and_primitives:
        assert episode["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
        assert int(episode["episode_id"]) == EPISODE_ID
        assert int(primitive["selected_action"]) == ACTION_ID

    after_fingerprints = {
        primitive["after"]["observation_fingerprint"]
        for _, primitive in episodes_and_primitives
    }
    assert len(after_fingerprints) == 1
