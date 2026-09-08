"""Expose every existing contract script to pytest with dependency markers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "tests" / "contracts"
PUBLIC_SCRIPT_DIR = ROOT / "scripts"
INTERNAL_PYTHON_DIR = ROOT / "python"

UNIT_SCRIPTS = (
    "test_action_distribution_audit.py",
    "test_audit_observability.py",
    "test_contract_naming.py",
    "test_depth_safety.py",
    "test_global_route.py",
    "test_merge_teacher_rollouts.py",
    "test_mission_sampling_budget.py",
    "test_motion_primitives.py",
    "test_policy_input_contract.py",
    "test_relative_data_paths.py",
    "test_policy_runtime_contract.py",
    "test_task_contract.py",
    "test_teacher_beam.py",
    "test_teacher_collection_contract.py",
    "test_teacher_path_contract.py",
)
ROS_SCRIPTS = (
    "test_collision_batch.py",
    "test_global_route_native_parity.py",
    "test_motion_primitive_rollouts.py",
)
UNITY_SCRIPTS = (
    "test_action_mask.py",
    "test_unity_env_task_contract.py",
    "test_unity_motion_primitives.py",
    "test_unity_motion_sequence.py",
    "test_unity_terminal_precedence.py",
)

SCRIPT_ARGS = {}

def _run_script(name: str) -> None:
    environment = dict(os.environ)
    python_path = [str(ROOT / "python")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    arguments = list(SCRIPT_ARGS.get(name, ()))
    subprocess.run(
        [sys.executable, str(CONTRACTS / name), *arguments],
        cwd=str(ROOT), env=environment, check=True, timeout=180,
    )


def test_only_public_ros_wrappers_are_executable_python_files():
    """Keep internal modules out of rosrun's recursive executable search."""

    internal_executables = [
        path.relative_to(ROOT).as_posix()
        for path in INTERNAL_PYTHON_DIR.rglob("*.py")
        if os.access(path, os.X_OK)
    ]
    assert internal_executables == []

    public_wrappers = sorted(PUBLIC_SCRIPT_DIR.glob("*.py"))
    assert public_wrappers
    non_executable_wrappers = [
        path.relative_to(ROOT).as_posix()
        for path in public_wrappers
        if not os.access(path, os.X_OK)
    ]
    assert non_executable_wrappers == []


@pytest.mark.unit
def test_awac_and_evaluation_entrypoints_are_python_only():
    entrypoints = (
        PUBLIC_SCRIPT_DIR / "train_awac.py",
        PUBLIC_SCRIPT_DIR / "evaluate_policy_unity.py",
        PUBLIC_SCRIPT_DIR / "audit_awac_replay.py",
        PUBLIC_SCRIPT_DIR / "select_awac_dev_checkpoint.py",
    )
    for path in entrypoints:
        assert os.access(path, os.X_OK), path
        subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=str(ROOT),
            check=True,
            timeout=60,
        )

    assert not (PUBLIC_SCRIPT_DIR / "run_awac_guarded.sh").exists()
    assert not (PUBLIC_SCRIPT_DIR / "run_awac_reliable_v4.sh").exists()


@pytest.mark.unit
@pytest.mark.parametrize("script", UNIT_SCRIPTS)
def test_unit_contract_script(script):
    _run_script(script)

@pytest.mark.ros
@pytest.mark.parametrize("script", ROS_SCRIPTS)
def test_ros_contract_script(script):
    _run_script(script)

@pytest.mark.unity
@pytest.mark.parametrize("script", UNITY_SCRIPTS)
def test_unity_contract_script(script):
    _run_script(script)
