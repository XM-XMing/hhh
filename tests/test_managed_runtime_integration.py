"""RED contract tests for wiring the shared managed runtime into callers."""

from __future__ import annotations

from pathlib import Path

import pytest

from planning.runtime.ports import build_worker_runtime_specs
from planning.runtime.worker import write_worker_runtime_spec_file


pytestmark = pytest.mark.unit


def _worker_file(tmp_path: Path, count: int = 2) -> Path:
    specs = build_worker_runtime_specs(
        worker_count=count,
        master_port_base=11621,
        command_port_base=10553,
        depth_port_base=12554,
        port_stride=20,
        runtime_instance_template="calibration-{worker_id:02d}",
        ros_root=tmp_path / "ros",
        training_run_id="integration-test",
        runtime_launch_nonce="launch-test",
    )
    path = tmp_path / "worker_runtime_specs.json"
    write_worker_runtime_spec_file(path, specs)
    return path


def test_trainer_builds_shared_runtime_from_canonical_worker_specs(
    tmp_path: Path, monkeypatch
):
    import planning.runtime.managed_runtime as managed_runtime
    from planning.awac import trainer

    worker_file = _worker_file(tmp_path)
    seen = {}

    class FakeManagedRuntimePool:
        def __init__(self, specs, **kwargs):
            seen["specs"] = tuple(specs)
            seen["kwargs"] = dict(kwargs)

    monkeypatch.setattr(managed_runtime, "ManagedRuntimePool", FakeManagedRuntimePool)
    runtime = trainer.build_managed_runtime_pool(
        worker_file,
        worker_count=2,
        output_dir=tmp_path / "run",
        task_contract_sha256="t" * 64,
        mpl_contract_sha256="m" * 64,
        max_steps=45,
    )

    assert isinstance(runtime, FakeManagedRuntimePool)
    assert [spec.worker_id for spec in seen["specs"]] == [0, 1]
    assert [spec.runtime_instance_id for spec in seen["specs"]] == [
        "calibration-00",
        "calibration-01",
    ]
    assert seen["kwargs"]["log_dir"] == (tmp_path / "run" / "runtime").resolve()
    assert seen["kwargs"]["task_contract_sha256"] == "t" * 64
    assert seen["kwargs"]["mpl_contract_sha256"] == "m" * 64


def test_trainer_starts_managed_runtime_before_parallel_pool(tmp_path: Path, monkeypatch):
    from planning.awac import trainer

    events = []

    class FakeManagedRuntime:
        def start(self):
            events.append("managed_start")
            return {"runtime": "identity"}

        def close(self):
            events.append("managed_close")

    def fake_build_managed(*args, **kwargs):
        return FakeManagedRuntime()

    def fake_build_pool(*args, **kwargs):
        events.append("parallel_pool_build")
        return object()

    monkeypatch.setattr(trainer, "build_managed_runtime_pool", fake_build_managed)
    monkeypatch.setattr(trainer, "build_runtime_pool", fake_build_pool)

    managed, pool, identity = trainer.start_managed_calibration_runtime(
        tmp_path / "workers.json",
        worker_count=2,
        output_dir=tmp_path / "run",
        task_contract_sha256="t" * 64,
        mpl_contract_sha256="m" * 64,
        max_steps=45,
    )

    assert isinstance(managed, FakeManagedRuntime)
    assert pool is not None
    assert identity == {"runtime": "identity"}
    assert events == ["managed_start", "parallel_pool_build"]


def test_managed_evaluator_delegates_process_lifecycle_to_shared_owner():
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_policy_unity_managed.sh"
    source = script.read_text(encoding="utf-8")

    assert "scripts/run_managed_runtime.py" in source
    assert "setsid env ROS_HOME=\"$ROS_HOME_DIR\" roscore" not in source
    assert "setsid \"$UNITY_BIN\"" not in source
    assert "setsid env ROS_MASTER_URI=\"$MASTER_URI\"" not in source


def test_managed_runner_is_a_thin_shared_owner_entrypoint():
    runner = Path(__file__).resolve().parents[1] / "scripts" / "run_managed_runtime.py"
    source = runner.read_text(encoding="utf-8")
    assert "ManagedRuntimePool" in source
    assert "load_worker_runtime_spec_file" in source
    assert "run_command" in source
