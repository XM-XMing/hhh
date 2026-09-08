"""Public CLI seams for the final Pre-Collection command normalization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "validate_pre_collection_runtime.py",
    "validate_teacher_missions.py",
    "monitor_teacher_collection.py",
    "validate_teacher_collection.py",
    "validate_bc_dataset.py",
    "collect_rollouts_parallel.py",
)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_path = [str(ROOT / "python")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    return environment


def _run(script_name: str, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name), *arguments],
        cwd=str(ROOT),
        env=_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


@pytest.mark.unit
@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_normalized_cli_help(script_name: str):
    result = _run(script_name, "--help")
    assert result.returncode == 0, result.stdout
    assert "usage:" in result.stdout


@pytest.mark.unit
def test_runtime_validator_accepts_current_release_fixture():
    manifest = json.loads(
        (ROOT / "docs" / "pre_collection_runtime_final_v3.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = manifest["runtime_identity"]
    maps = manifest["map_identity"]
    contracts = manifest["contracts"]
    result = _run(
        "validate_pre_collection_runtime.py",
        "--unity-bin",
        runtime["unity_player_path"],
        "--unity-data",
        runtime["unity_data_path"],
        "--bridge",
        runtime["bridge_path"],
        "--point-cloud",
        maps["point_cloud_path"],
        "--voxel-cache",
        maps["voxel_cache_path"],
        "--mpl-npz",
        str(ROOT / "data/motion_primitives/motion_primitives_105.npz"),
        "--mpl-json",
        str(ROOT / "data/motion_primitives/motion_primitives_105.json"),
        "--max-steps",
        "45",
        "--expected-observation-contract",
        contracts["observation_contract"],
    )
    assert result.returncode == 0, result.stdout
    assert "RUNTIME_VALIDATION=PASS" in result.stdout


@pytest.mark.unit
def test_current_release_manifest_separates_historical_bridge_identity():
    manifest_path = ROOT / "docs" / "pre_collection_runtime_final_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = manifest["runtime_identity"]
    current_sha = hashlib.sha256(Path(runtime["bridge_path"]).read_bytes()).hexdigest()

    assert runtime["bridge_sha256"] == current_sha

    historical = manifest["historical_runtime_identity"]
    assert historical["classification"] == "HISTORICAL_RUNTIME_FIXTURE"
    assert historical["bridge_sha256"] == (
        "462e9f7bc25d5bef63ce17b863daa256726e17de870bda04a1bcb72e79e80f2b"
    )
    assert historical["bridge_sha256"] != runtime["bridge_sha256"]


@pytest.mark.unit
def test_monitor_once_accepts_bounded_in_progress_fixture(tmp_path: Path):
    result = _run(
        "monitor_teacher_collection.py",
        "--rollout-dir",
        str(tmp_path),
        "--interval-sec",
        "0",
        "--once",
    )
    assert result.returncode == 0, result.stdout
    assert "COLLECTION_MONITOR" in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize(
    ("script_name", "arguments"),
    (
        (
            "validate_pre_collection_runtime.py",
            ("--unity-bin", "missing", "--unity-data", "missing"),
        ),
        (
            "validate_teacher_missions.py",
            (
                "--candidate-index",
                "missing",
                "--missions",
                "missing",
                "--route-store-prefix",
                "missing",
                "--expected-passing",
                "1",
                "--max-steps",
                "45",
                "--expected-seed",
                "2026",
            ),
        ),
        (
            "monitor_teacher_collection.py",
            ("--rollout-dir", "missing", "--once"),
        ),
        (
            "validate_teacher_collection.py",
            (
                "--rollout-dir",
                "missing",
                "--missions",
                "missing",
                "--route-store-prefix",
                "missing",
                "--min-accepted",
                "1",
                "--expected-workers",
                "1",
                "--max-steps",
                "45",
                "--expected-observation-contract",
                "reliable_exact_endpoint_snapshot",
            ),
        ),
        (
            "validate_bc_dataset.py",
            (
                "--index",
                "missing",
                "--labels",
                "missing",
                "--depth-action-masks",
                "missing",
                "--dataset-audit",
                "missing",
                "--bc-mmap",
                "missing",
                "--expected-observation-contract",
                "reliable_exact_endpoint_snapshot",
            ),
        ),
    ),
)
def test_normalized_cli_rejects_invalid_fixture(script_name: str, arguments: tuple[str, ...]):
    result = _run(script_name, *arguments)
    assert result.returncode != 0, result.stdout


@pytest.mark.unit
def test_parallel_collector_accepts_flag_style_without_starting_collection(tmp_path: Path):
    result = _run(
        "collect_rollouts_parallel.py",
        "--mission-index",
        str(tmp_path / "missing.csv"),
        "--out-dir",
        str(tmp_path / "out"),
        "--workers",
        "1",
    )
    assert result.returncode != 0, result.stdout
    assert "mission index" in result.stdout.lower()
