"""Contract tests for the shared managed ROS/Unity/Bridge lifecycle owner."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.runtime.ports import build_worker_runtime_specs


pytestmark = pytest.mark.unit


class _FakeProcess:
    _next_pid = 41000

    def __init__(self, argv, *, env, cwd, stdout, stderr, start_new_session, events):
        self.argv = tuple(str(value) for value in argv)
        self.env = dict(env)
        self.cwd = str(cwd)
        self.stdout = stdout
        self.stderr = stderr
        self.start_new_session = bool(start_new_session)
        self.events = events
        self.pid = _FakeProcess._next_pid
        _FakeProcess._next_pid += 1
        self.returncode = None
        self.events.append(("start", self))

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.events.append(("wait", self, timeout))
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.events.append(("terminate", self))
        self.returncode = -15

    def kill(self):
        self.events.append(("kill", self))
        self.returncode = -9


def _artifacts(tmp_path: Path):
    unity = tmp_path / "XMflight.x86_64"
    unity.write_bytes(b"unity-player")
    unity.chmod(0o755)
    assembly = tmp_path / "XMflight_Data" / "Managed" / "Assembly-CSharp.dll"
    assembly.parent.mkdir(parents=True)
    assembly.write_bytes(b"unity-assembly")
    bridge = tmp_path / "unity_bridge_node"
    bridge.write_bytes(b"bridge-binary")
    bridge.chmod(0o755)
    return unity, assembly, bridge


def _specs(tmp_path: Path, count=2):
    return build_worker_runtime_specs(
        worker_count=count,
        master_port_base=11621,
        command_port_base=10553,
        depth_port_base=12554,
        port_stride=20,
        runtime_instance_template="calibration-{worker_id:02d}",
        ros_root=tmp_path / "ros",
        training_run_id="calibration-test",
        runtime_launch_nonce="test-launch",
    )


def _factory(events):
    def launch(argv, **kwargs):
        return _FakeProcess(argv, events=events, **kwargs)

    return launch


def _ready(events, *, fail_at=None):
    def check(stage, spec, process_handles, env, timeout_s):
        events.append(("ready", stage, int(spec.worker_id), dict(env)))
        if fail_at == (stage, int(spec.worker_id)):
            raise RuntimeError("fake {} failure for worker {}".format(stage, spec.worker_id))
        return {"stage": stage, "worker_id": int(spec.worker_id)}

    return check


def test_managed_pool_starts_each_worker_in_order_and_isolates_environment(tmp_path):
    from planning.runtime.managed_runtime import ManagedRuntimePool

    unity, assembly, bridge = _artifacts(tmp_path)
    events = []
    pool = ManagedRuntimePool(
        _specs(tmp_path, count=2),
        unity_bin=unity,
        bridge_bin=bridge,
        workspace=tmp_path,
        log_dir=tmp_path / "logs",
        task_contract_sha256="task-sha",
        mpl_contract_sha256="mpl-sha",
        process_factory=_factory(events),
        readiness_checker=_ready(events),
        port_available=lambda port: True,
    )

    identity = pool.start()

    starts = [event[1] for event in events if event[0] == "start"]
    assert len(starts) == 6
    for worker_id in (0, 1):
        worker_starts = [
            process
            for process in starts
            if process.env["PLANNING_WORKER_ID"] == str(worker_id)
        ]
        assert [process.argv[0] for process in worker_starts] == [
            "roscore",
            str(unity),
            "roslaunch",
        ]
        assert {process.env["ROS_MASTER_URI"] for process in worker_starts} == {
            "http://127.0.0.1:{}".format(11621 + worker_id)
        }
        assert {process.env["ROS_HOME"] for process in worker_starts} == {
            str(tmp_path / "ros" / "worker_{:02d}".format(worker_id))
        }
        assert all(process.start_new_session for process in worker_starts)

    ready_stages = [
        (event[1], event[2]) for event in events if event[0] == "ready"
    ]
    assert ready_stages == [("ros_master", 0), ("bridge", 0), ("ros_master", 1), ("bridge", 1)]
    assert identity["workers"][0]["worker_id"] == 0
    assert identity["workers"][1]["runtime_instance_id"] == "calibration-01"
    assert identity["contracts"]["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    assert identity["artifacts"]["unity_player_sha256"] == hashlib.sha256(
        unity.read_bytes()
    ).hexdigest()
    for worker in identity["workers"]:
        assert worker["ros_master_uri"] == "http://127.0.0.1:{}".format(
            11621 + int(worker["worker_id"])
        )
        assert worker["unity_executable_identity"] == {
            "path": str(unity.resolve()),
            "sha256": identity["artifacts"]["unity_player_sha256"],
        }
        assert worker["bridge_executable_identity"] == {
            "path": str(bridge.resolve()),
            "sha256": identity["artifacts"]["bridge_sha256"],
        }
        assert worker["task_contract_sha256"] == "task-sha"
        assert worker["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
        assert worker["mpl_contract_sha256"] == "mpl-sha"

    pool.close()
    assert all(process.returncode is not None for process in starts)


def test_managed_pool_rejects_occupied_port_before_launch(tmp_path):
    from planning.runtime.managed_runtime import ManagedRuntimePool, ManagedRuntimeError

    unity, _assembly, bridge = _artifacts(tmp_path)
    events = []
    pool = ManagedRuntimePool(
        _specs(tmp_path, count=2),
        unity_bin=unity,
        bridge_bin=bridge,
        workspace=tmp_path,
        log_dir=tmp_path / "logs",
        task_contract_sha256="task-sha",
        mpl_contract_sha256="mpl-sha",
        process_factory=_factory(events),
        readiness_checker=_ready(events),
        port_available=lambda port: int(port) != 11622,
    )

    with pytest.raises(ManagedRuntimeError, match="port"):
        pool.start()
    assert not [event for event in events if event[0] == "start"]
    pool.close()


def test_managed_pool_cleans_all_owned_processes_after_partial_startup(tmp_path):
    from planning.runtime.managed_runtime import ManagedRuntimePool

    unity, _assembly, bridge = _artifacts(tmp_path)
    events = []
    pool = ManagedRuntimePool(
        _specs(tmp_path, count=2),
        unity_bin=unity,
        bridge_bin=bridge,
        workspace=tmp_path,
        log_dir=tmp_path / "logs",
        task_contract_sha256="task-sha",
        mpl_contract_sha256="mpl-sha",
        process_factory=_factory(events),
        readiness_checker=_ready(events, fail_at=("bridge", 1)),
        port_available=lambda port: True,
    )

    with pytest.raises(RuntimeError, match="fake bridge failure"):
        pool.start()

    starts = [event[1] for event in events if event[0] == "start"]
    assert len(starts) == 6
    assert all(process.returncode is not None for process in starts)
    assert len([event for event in events if event[0] == "terminate"]) == 6
    pool.close()


def test_managed_pool_cleanup_is_idempotent_and_never_uses_global_kill(tmp_path):
    from planning.runtime.managed_runtime import ManagedRuntimePool

    unity, _assembly, bridge = _artifacts(tmp_path)
    events = []
    pool = ManagedRuntimePool(
        _specs(tmp_path, count=1),
        unity_bin=unity,
        bridge_bin=bridge,
        workspace=tmp_path,
        log_dir=tmp_path / "logs",
        task_contract_sha256="task-sha",
        mpl_contract_sha256="mpl-sha",
        process_factory=_factory(events),
        readiness_checker=_ready(events),
        port_available=lambda port: True,
    )
    pool.start()
    pool.close()
    first_cleanup_count = len([event for event in events if event[0] == "terminate"])
    pool.close()
    assert first_cleanup_count == 3
    assert len([event for event in events if event[0] == "terminate"]) == first_cleanup_count


def test_managed_pool_runs_caller_in_worker_environment(tmp_path, monkeypatch):
    import planning.runtime.managed_runtime as managed_runtime
    from planning.runtime.managed_runtime import ManagedRuntimePool

    unity, _assembly, bridge = _artifacts(tmp_path)
    events = []
    pool = ManagedRuntimePool(
        _specs(tmp_path, count=2),
        unity_bin=unity,
        bridge_bin=bridge,
        workspace=tmp_path,
        log_dir=tmp_path / "logs",
        task_contract_sha256="task-sha",
        mpl_contract_sha256="mpl-sha",
        process_factory=_factory(events),
        readiness_checker=_ready(events),
        port_available=lambda port: True,
    )
    command_processes = []

    def command_factory(argv, **kwargs):
        process = _FakeProcess(
            argv,
            env=kwargs["env"],
            cwd=kwargs["cwd"],
            stdout=None,
            stderr=None,
            start_new_session=kwargs["start_new_session"],
            events=events,
        )
        command_processes.append(process)
        return process

    monkeypatch.setattr(managed_runtime.subprocess, "Popen", command_factory)
    pool.start()
    assert pool.run_command(("fake-calibration",), worker_id=1) == 0
    assert len(command_processes) == 1
    command = command_processes[0]
    assert command.env["ROS_MASTER_URI"] == "http://127.0.0.1:11622"
    assert command.env["ROS_HOME"] == str(tmp_path / "ros" / "worker_01")
    assert command.start_new_session is True
    pool.close()


def test_managed_pool_keyboard_interrupt_terminates_owned_caller(tmp_path, monkeypatch):
    import planning.runtime.managed_runtime as managed_runtime
    from planning.runtime.managed_runtime import ManagedRuntimePool

    unity, _assembly, bridge = _artifacts(tmp_path)
    events = []
    pool = ManagedRuntimePool(
        _specs(tmp_path, count=1),
        unity_bin=unity,
        bridge_bin=bridge,
        workspace=tmp_path,
        log_dir=tmp_path / "logs",
        task_contract_sha256="task-sha",
        mpl_contract_sha256="mpl-sha",
        process_factory=_factory(events),
        readiness_checker=_ready(events),
        port_available=lambda port: True,
    )

    class InterruptingProcess(_FakeProcess):
        def wait(self, timeout=None):
            if self.returncode is None:
                raise KeyboardInterrupt
            return super().wait(timeout=timeout)

    caller = []

    def command_factory(argv, **kwargs):
        process = InterruptingProcess(
            argv,
            env=kwargs["env"],
            cwd=kwargs["cwd"],
            stdout=None,
            stderr=None,
            start_new_session=kwargs["start_new_session"],
            events=events,
        )
        caller.append(process)
        return process

    monkeypatch.setattr(managed_runtime.subprocess, "Popen", command_factory)
    pool.start()
    with pytest.raises(KeyboardInterrupt):
        pool.run_command(("fake-calibration",))
    assert caller[0].returncode == -15
    pool.close()
