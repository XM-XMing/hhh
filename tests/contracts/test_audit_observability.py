#!/usr/bin/env python3
"""Contract checks for partial audit publication and structured run logs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

from planning.mission.preparation import build_passing_mission_rows
from planning.diagnostics.observability import REQUIRED_LOG_FIELDS, StructuredRunLogger


def test_passing_snapshot_contract() -> None:
    candidates = [
        {"mission_id": "mission-a", "global_route_stretch": "1.1"},
        {"mission_id": "mission-b", "global_route_stretch": "1.2"},
    ]
    audits = [
        {
            "candidate_number": "0",
            "result": "dead_end",
            "executed_path_length_m": "5.0",
            "path_stretch": "1.0",
        },
        {
            "candidate_number": "1",
            "result": "success",
            "executed_path_length_m": "42.0",
            "path_stretch": "1.05",
        },
    ]
    passing = build_passing_mission_rows(audits, candidates, required_passing=1)
    assert len(passing) == 1
    assert passing[0]["mission_id"] == "mission-b"
    assert passing[0]["episode_id"] == 0
    assert passing[0]["teacher_plan_path_length_m"] == 42.0


def test_structured_logging_contract() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "events.jsonl"
        logger = StructuredRunLogger(
            path, run_id="contract-run", component="contract-test"
        )
        logger.event(
            "progress",
            mission_id="mission-a",
            episode=7,
            step=11,
            duration_ms=12.5,
        )
        logger.close()

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(REQUIRED_LOG_FIELDS).issubset(payload)
        assert payload["run_id"] == "contract-run"
        assert payload["mission_id"] == "mission-a"
        assert payload["episode"] == 7
        assert payload["step"] == 11
        assert payload["component"] == "contract-test"
        assert payload["status"] == "progress"
        assert payload["error_type"] == ""


def test_gpu_sampling_can_be_disabled() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cpu-events.jsonl"
        with mock.patch(
            "planning.diagnostics.observability.subprocess.run",
            side_effect=AssertionError("nvidia-smi must not run"),
        ):
            logger = StructuredRunLogger(
                path,
                run_id="cpu-contract-run",
                component="cpu-contract-test",
                sample_gpu=False,
            )
            payload = logger.event("progress")
            logger.close()
        assert payload["gpu_memory_mb"] is None


def main() -> None:
    test_passing_snapshot_contract()
    test_structured_logging_contract()
    test_gpu_sampling_can_be_disabled()
    print("AUDIT_OBSERVABILITY_CONTRACT=PASS")


if __name__ == "__main__":
    main()
