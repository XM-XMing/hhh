"""CPU-only request/reply tests for the spawn-isolated parallel env pool."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import planning.runtime.parallel_env as parallel_env
from planning.runtime.parallel_env import (
    EnvWorkerSpec,
    ParallelEnvError,
    ParallelEnvPool,
    ParallelEnvTimeout,
    ParallelEnvWorkerError,
)


pytestmark = pytest.mark.unit


class _FakeEnvironment:
    def __init__(self, worker_id, env_kwargs):
        self.worker_id = int(worker_id)
        self.marker = str(env_kwargs.get("marker", ""))
        self.episode = 0
        self.step_count = 0
        self.stopped = False
        self.stop_log_path = str(env_kwargs.get("stop_log_path", ""))

    def wait_until_ready(self, timeout_s):
        return {
            "worker_id": self.worker_id,
            "timeout_s": float(timeout_s),
            "ros_master_uri": os.environ["ROS_MASTER_URI"],
            "ros_home": os.environ["ROS_HOME"],
            "worker_environment_id": os.environ["PLANNING_WORKER_ID"],
            "marker": self.marker,
        }

    def reset(self, start=None, goal=None, settle=None):
        self.episode += 1
        self.step_count = 0
        self.stopped = False
        return {
            "worker_id": self.worker_id,
            "episode": self.episode,
            "start": start,
            "goal": goal,
            "settle": settle,
        }

    def get_action_mask(self, observation, return_info=False):
        mask = [True, self.worker_id == 0, self.step_count < 2]
        info = {
            "worker_id": self.worker_id,
            "observation_step": int(observation.get("step", 0)),
            "combined_valid_count": int(sum(mask)),
            "environment_stopped": bool(self.stopped),
        }
        return (mask, info) if return_info else mask

    def step_primitive(
        self,
        action_id,
        obs_before=None,
        precomputed_mask=None,
        precomputed_mask_info=None,
    ):
        action_id = int(action_id)
        if action_id < 0:
            raise ValueError("negative fake action")
        if action_id == 999:
            time.sleep(0.30)
        if obs_before is None or precomputed_mask is None or precomputed_mask_info is None:
            raise AssertionError("fake step did not receive the cached observation and mask")
        expected_mask, expected_info = self.get_action_mask(obs_before, return_info=True)
        if expected_mask != precomputed_mask:
            raise AssertionError("cached action mask does not match observation")
        if expected_info != precomputed_mask_info:
            raise AssertionError("cached mask info does not match observation")
        self.step_count += 1
        return (
            {"worker_id": self.worker_id, "step": self.step_count},
            float(action_id),
            bool(action_id == 998),
            {"action_id": action_id, "used_precomputed_observation": True},
        )

    def stop(self):
        self.stopped = True
        if self.stop_log_path:
            Path(self.stop_log_path).write_text("stopped\n", encoding="utf-8")
        return {"worker_id": self.worker_id, "stopped": True}

    def close(self):
        self.stopped = True
        return {"worker_id": self.worker_id, "closed": True}


def _fake_environment_factory(worker_id, env_kwargs):
    return _FakeEnvironment(worker_id, env_kwargs)


def _specs(tmp_path, count=2):
    return [
        EnvWorkerSpec(
            worker_id=worker_id,
            ros_master_uri="http://127.0.0.1:{}".format(11621 + worker_id),
            ros_home=str(tmp_path / "ros_home_{:02d}".format(worker_id)),
            env_kwargs={"marker": "worker-{}".format(worker_id)},
            extra_environment={"FAKE_POOL_VALUE": "value-{}".format(worker_id)},
        )
        for worker_id in range(count)
    ]


def test_module_has_no_top_level_ros_or_unity_import():
    assert "rospy" not in parallel_env.__dict__
    assert "UnityForestEnv" not in parallel_env.__dict__


def test_worker_session_requires_reset_after_init_terminal_stop_and_error():
    environment = _FakeEnvironment(0, {})
    session = parallel_env._WorkerEnvironmentSession(environment)

    def assert_step_rejected():
        with pytest.raises(RuntimeError, match="step requires a successful reset"):
            session.serve("step", {"action_id": 1})
        assert session.needs_reset is True

    assert_step_rejected()

    session.serve("reset", {"kwargs": {}})
    assert session.needs_reset is False
    session.serve("step", {"action_id": 1})
    assert session.needs_reset is False

    terminal = session.serve("step", {"action_id": 998})
    assert terminal["done"] is True
    assert terminal["action_mask_info"]["environment_stopped"] is True
    assert_step_rejected()

    session.serve("reset", {"kwargs": {}})
    assert session.needs_reset is False
    session.serve("stop", {})
    assert_step_rejected()

    session.serve("reset", {"kwargs": {}})
    assert session.needs_reset is False
    with pytest.raises(ValueError, match="negative fake action"):
        session.serve("step", {"action_id": -1})
    assert_step_rejected()

    session.serve("reset", {"kwargs": {}})
    assert session.needs_reset is False


def test_spawn_pool_ready_reset_step_stop_close_and_environment_isolation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ROS_MASTER_URI", "http://parent.invalid:1")
    monkeypatch.setenv("ROS_HOME", str(tmp_path / "parent_ros_home"))

    pool = ParallelEnvPool(
        _specs(tmp_path),
        env_factory=_fake_environment_factory,
        startup_timeout_s=5.0,
        request_timeout_s=2.0,
    )
    assert pool.worker_ids == (0, 1)
    assert pool.startup_metadata[0]["pid"] != pool.startup_metadata[1]["pid"]
    assert pool.startup_metadata[0]["ros_master_uri"].endswith(":11621")
    assert pool.startup_metadata[1]["ros_master_uri"].endswith(":11622")

    ready = pool.ready(worker_ready_timeout_s=1.25)
    for worker_id in pool.worker_ids:
        assert ready[worker_id]["worker_id"] == worker_id
        assert ready[worker_id]["worker_environment_id"] == str(worker_id)
        assert ready[worker_id]["marker"] == "worker-{}".format(worker_id)
        assert ready[worker_id]["ros_home"].endswith(
            "ros_home_{:02d}".format(worker_id)
        )

    observations = pool.reset(
        {
            0: {"start": [0.0, 0.0, 2.0], "goal": [10.0, 0.0, 2.0]},
            1: {"start": [1.0, 0.0, 2.0], "goal": [11.0, 0.0, 2.0]},
        }
    )
    assert set(observations[0]) == {
        "observation",
        "action_mask",
        "action_mask_info",
    }
    assert observations[0]["observation"]["episode"] == 1
    assert observations[1]["observation"]["start"] == [1.0, 0.0, 2.0]
    assert observations[0]["action_mask"] == [True, True, True]
    assert observations[1]["action_mask"] == [True, False, True]
    assert observations[1]["action_mask_info"]["worker_id"] == 1

    transitions = pool.step({0: 3, 1: 7})
    assert set(transitions[0]) == {
        "observation",
        "action_mask",
        "action_mask_info",
        "reward",
        "done",
        "info",
        "worker_step_completed_monotonic_s",
    }
    assert transitions[0]["observation"]["worker_id"] == 0
    assert transitions[0]["observation"]["step"] == 1
    assert transitions[0]["action_mask"] == [True, True, True]
    assert transitions[0]["reward"] == 3.0
    assert transitions[1]["info"]["action_id"] == 7
    assert transitions[1]["info"]["used_precomputed_observation"] is True
    assert transitions[0]["worker_step_completed_monotonic_s"] > 0.0

    stopped = pool.stop()
    assert stopped == {
        0: {"worker_id": 0, "stopped": True},
        1: {"worker_id": 1, "stopped": True},
    }

    pool.close()
    pool.close()
    assert pool.is_closed
    assert not any(process.is_alive() for process in pool._processes.values())
    assert os.environ["ROS_MASTER_URI"] == "http://parent.invalid:1"
    assert os.environ["ROS_HOME"] == str(tmp_path / "parent_ros_home")


@pytest.mark.parametrize(
    "specs,error",
    (
        (
            [
                EnvWorkerSpec(1, "http://127.0.0.1:1", "/tmp/ros-home-1"),
            ],
            "contiguous worker_id",
        ),
        (
            [
                EnvWorkerSpec(0, "http://127.0.0.1:1", "/tmp/ros-home-0"),
                EnvWorkerSpec(1, "http://127.0.0.1:1", "/tmp/ros-home-1"),
            ],
            "unique ros_master_uri",
        ),
        (
            [
                EnvWorkerSpec(
                    0,
                    "http://127.0.0.1:1",
                    "/tmp/ros-home-0",
                    extra_environment={"ROS_MASTER_URI": "override"},
                ),
            ],
            "cannot override ROS_MASTER_URI",
        ),
    ),
)
def test_worker_specs_enforce_identity_and_ros_isolation(specs, error):
    with pytest.raises(ValueError, match=error):
        ParallelEnvPool(specs, env_factory=_fake_environment_factory)


def test_worker_exception_preserves_worker_identity_and_breaks_pool(tmp_path):
    stop_log = tmp_path / "worker_error_stop.log"
    spec = _specs(tmp_path, count=1)[0]
    spec = EnvWorkerSpec(
        worker_id=spec.worker_id,
        ros_master_uri=spec.ros_master_uri,
        ros_home=spec.ros_home,
        env_kwargs={**spec.env_kwargs, "stop_log_path": str(stop_log)},
        extra_environment=spec.extra_environment,
    )
    pool = ParallelEnvPool(
        [spec],
        env_factory=_fake_environment_factory,
        startup_timeout_s=5.0,
        request_timeout_s=1.0,
    )
    try:
        pool.reset({0: {}})
        with pytest.raises(ParallelEnvWorkerError) as caught:
            pool.step({0: -1})
        assert caught.value.worker_id == 0
        assert caught.value.command == "step"
        assert caught.value.error_type == "ValueError"
        assert "negative fake action" in caught.value.error_message
        assert stop_log.read_text(encoding="utf-8") == "stopped\n"

        with pytest.raises(ParallelEnvError, match="unusable"):
            pool.stop()
    finally:
        pool.close(timeout_s=0.25)


def test_request_timeout_names_pending_worker_and_forces_shutdown(tmp_path):
    pool = ParallelEnvPool(
        _specs(tmp_path, count=1),
        env_factory=_fake_environment_factory,
        startup_timeout_s=5.0,
        request_timeout_s=1.0,
    )
    try:
        pool.reset({0: {}})
        with pytest.raises(ParallelEnvTimeout, match=r"pending workers=\[0\]"):
            pool.step({0: 999}, timeout_s=0.05)
    finally:
        pool.close(timeout_s=0.10)
    assert not pool._processes[0].is_alive()


def test_expired_deadline_drains_reply_that_is_already_ready(
    tmp_path,
    monkeypatch,
):
    pool = ParallelEnvPool(
        _specs(tmp_path, count=1),
        env_factory=_fake_environment_factory,
        startup_timeout_s=5.0,
        request_timeout_s=1.0,
    )
    try:
        pool.reset({0: {}})
        handle = pool.begin_step({0: 4}, timeout_s=0.01)
        assert pool._connections[0].poll(1.0)
        time.sleep(max(0.0, handle.deadline_monotonic - time.monotonic()) + 0.01)

        wait_timeouts = []
        original_wait = parallel_env.wait

        def recording_wait(connections, timeout=None):
            wait_timeouts.append(timeout)
            return original_wait(connections, timeout=timeout)

        monkeypatch.setattr(parallel_env, "wait", recording_wait)
        transition = pool.finish_step(handle)[0]

        assert transition["reward"] == 4.0
        assert wait_timeouts == [0.0]
        assert pool._broken is False
    finally:
        pool.close(timeout_s=0.25)


def test_begin_finish_step_allows_learner_overlap_and_blocks_second_request(tmp_path):
    pool = ParallelEnvPool(
        _specs(tmp_path, count=1),
        env_factory=_fake_environment_factory,
        startup_timeout_s=5.0,
        request_timeout_s=1.0,
    )
    try:
        pool.reset({0: {"start": [0.0, 0.0, 2.0], "goal": [5.0, 0.0, 2.0]}})
        handle = pool.begin_step({0: 999})

        # The central learner may compute here, but no second Pipe request may
        # overtake the outstanding Unity step.
        with pytest.raises(ParallelEnvError, match="still in flight"):
            pool.begin_step({0: 1})
        with pytest.raises(ParallelEnvError, match="still in flight"):
            pool.stop()

        transitions = pool.finish_step(handle)
        assert transitions[0]["reward"] == 999.0
        assert transitions[0]["action_mask_info"]["observation_step"] == 1

        # A handle is single-use and normal requests resume only after finish.
        with pytest.raises(ParallelEnvError, match="does not match"):
            pool.finish_step(handle)
        assert pool.stop()[0]["stopped"] is True
    finally:
        pool.close(timeout_s=0.25)


def test_terminal_continuous_worker_is_stopped_before_reply(tmp_path):
    pool = ParallelEnvPool(
        _specs(tmp_path, count=1),
        env_factory=_fake_environment_factory,
        startup_timeout_s=5.0,
        request_timeout_s=1.0,
    )
    try:
        pool.reset({0: {}})
        transition = pool.step({0: 998})[0]
        assert transition["done"] is True
        assert transition["action_mask_info"]["environment_stopped"] is True
    finally:
        pool.close(timeout_s=0.25)


def test_reply_validation_rejects_wrong_worker_id():
    reply = {
        "protocol_id": parallel_env.PARALLEL_ENV_PROTOCOL_ID,
        "protocol_version": parallel_env.PARALLEL_ENV_PROTOCOL_VERSION,
        "worker_id": 7,
        "request_id": 3,
        "command": "step",
        "status": "ok",
        "result": None,
    }
    with pytest.raises(ParallelEnvError, match="worker_id 7"):
        ParallelEnvPool._validate_reply(
            reply,
            expected_worker_id=0,
            expected_request_id=3,
            expected_command="step",
        )
