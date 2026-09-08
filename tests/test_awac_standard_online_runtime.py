"""Public seams for the Standard AWAC online runtime.

These tests use an injected environment pool and a small replay double.  They
must never start ROS, ZMQ, Unity, or Bridge processes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def _observation(step: int, *, episode_id: str = "episode-0"):
    return {
        "depth": np.full((90, 160), 0.5, dtype=np.float32),
        "state": {
            "position": [float(step), 0.0, 2.0],
            "velocity": [0.0, 0.0, 0.0],
            "acceleration": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "z_to_min": 1.0,
            "z_to_max": 1.0,
        },
        "goal": {
            "relative": [40.0 - float(step), 0.0, 0.0],
            "direction_xy": [1.0, 0.0],
            "direction_body_xy": [1.0, 0.0],
            "distance_xy": 40.0 - float(step),
            "distance_xy_norm40": max(0.0, (40.0 - float(step)) / 40.0),
            "dz": 0.0,
        },
        "episode_id": episode_id,
        "reset_id": "reset-0",
        "observation_contract": "reliable_exact_endpoint_snapshot",
        "observation_source": "reliable_exact_endpoint_snapshot",
        "observation_semantics": "reliable_exact_endpoint_snapshot",
        "reliable_execution": True,
        "telemetry_observation": False,
        "endpoint_identity_available": True,
        "asynchronous_prefetch": False,
        "asynchronous_prefetch_status": "obsolete_for_reliable_exact",
        "legacy_observation": False,
        "telemetry_lookup_count": 0,
        "snapshot_missing_count": 0,
        "state_depth_skew_ns": 0,
        "frame_contract_failures": 0,
    }


def _transition(index: int, *, behavior_source: int):
    mask = np.ones((5,), dtype=np.bool_)
    return {
        "depth": np.full((1, 2, 2), index / 10.0, dtype=np.float32),
        "vector": np.asarray([index, index + 1, index + 2], dtype=np.float32),
        "action_mask": mask,
        "action": 1,
        "reward": float(index),
        "next_depth": np.full((1, 2, 2), (index + 1) / 10.0, dtype=np.float32),
        "next_vector": np.asarray([index + 1, index + 2, index + 3], dtype=np.float32),
        "next_action_mask": mask.copy(),
        "done": True,
        "behavior_source": int(behavior_source),
    }


def test_first_actor_update_budget_uses_existing_schedule_counters():
    from planning.awac.optimization import (
        minimum_online_transitions_for_first_actor_update,
    )

    assert minimum_online_transitions_for_first_actor_update(
        starting_replay_total_added=5142,
        starting_critic_update_count=71,
        starting_learner_update_step=71,
        learning_starts=5000,
        actor_learning_starts=8000,
        critic_burnin_updates=2000,
        actor_update_interval=4,
        updates_per_step=0.50,
    ) == 3865


def test_replay_clone_preserves_prefix_but_is_write_independent(tmp_path: Path):
    from planning.awac.interaction import BehaviorSource
    from planning.awac.replay import AWACReplayBuffer

    source = AWACReplayBuffer.create(
        tmp_path / "source",
        capacity=8,
        depth_shape=(1, 2, 2),
        vector_dim=3,
        action_dim=5,
        training_config_sha256="a" * 64,
        bc_checkpoint_sha256="b" * 64,
        task_contract_id="task",
        task_contract_sha256="c" * 64,
        mpl_contract_sha256="d" * 64,
        run_identity="calibration-source",
        run_contract_sha256="e" * 64,
        behavior_source_phase="critic_calibration",
    )
    source.add_batch(
        [
            _transition(1, behavior_source=int(BehaviorSource.BC_CALIBRATION)),
            _transition(2, behavior_source=int(BehaviorSource.BC_CALIBRATION)),
        ]
    )
    source.flush()
    source_metadata = (tmp_path / "source" / "metadata.json").read_bytes()
    source_rows = {
        name: np.array(source.arrays[name][: source.size], copy=True)
        for name in source.arrays
    }

    clone = AWACReplayBuffer.clone_from(
        source,
        tmp_path / "standard" / "replay",
        run_identity="standard-run",
        run_contract_sha256="f" * 64,
        training_config_sha256="g" * 64,
        behavior_source_phase="awac_training",
    )

    assert clone.size == 2
    assert clone.metadata["behavior_source_phase"] == "awac_training"
    assert clone.metadata["source_replay_run_identity"] == "calibration-source"
    for name, expected in source_rows.items():
        np.testing.assert_array_equal(clone.arrays[name][:2], expected)

    clone.add(
        **_transition(3, behavior_source=int(BehaviorSource.AWAC_ONLINE))
    )
    clone.flush()
    assert clone.size == 3
    assert source.size == 2
    for name, expected in source_rows.items():
        np.testing.assert_array_equal(source.arrays[name][:2], expected)
    assert (tmp_path / "source" / "metadata.json").read_bytes() == source_metadata


class _Replay:
    def __init__(self):
        self.rows = []
        self.total_added = 0

    @property
    def size(self):
        return len(self.rows)

    def add_batch(self, rows):
        self.rows.extend(dict(row) for row in rows)
        self.total_added += len(rows)

    def sample(self, batch_size, *, rng, torch, device):
        return {"batch_size": int(batch_size)}

    def flush(self):
        return None


class _Actor:
    def __init__(self):
        self.weight = np.asarray([1.0], dtype=np.float32)

    def state_dict(self):
        return {"weight": self.weight.copy()}


class _Learner:
    def __init__(self):
        self.actor = _Actor()
        self.update_step = 0
        self.critic_update_count = 0
        self.actor_update_count = 0
        self.actor_optimizer_step_count = 0
        self.device = "cpu"

    def state_dict(self):
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": {"weight": np.asarray([1.0], dtype=np.float32)},
            "critic2_state_dict": {"weight": np.asarray([1.0], dtype=np.float32)},
            "update_step": int(self.update_step),
            "actor_update_count": int(self.actor_update_count),
            "actor_optimizer_step_count": int(self.actor_optimizer_step_count),
            "critic_update_count": int(self.critic_update_count),
        }

    def update(self, batch, *, update_actor=True, actor_trust_batch=None):
        del batch, actor_trust_batch
        self.critic_update_count += 1
        self.update_step += 1
        if update_actor:
            self.actor_update_count += 1
            self.actor_optimizer_step_count += 1
            self.actor.weight[0] += 1.0
        return {
            "critic_td_loss": 1.0,
            "q_mean": 1.0,
            "q_std": 0.0,
            "awac_advantage_mean": 0.0,
            "awac_weight_mean": 1.0,
            "bc_kl": 0.0,
        }


class _Pool:
    worker_ids = (0,)

    def __init__(self, *, terminal_after=2):
        self.step_count = 0
        self.closed = False
        self.terminal_after = int(terminal_after)

    def ready(self, **kwargs):
        del kwargs
        return {0: {"ready": True}}

    def reset(self, payload, **kwargs):
        del payload, kwargs
        self.step_count = 0
        return {
            0: {
                "observation": _observation(0),
                "action_mask": np.ones(105, dtype=np.bool_),
                "action_mask_info": {},
            }
        }

    def step(self, actions, **kwargs):
        del actions, kwargs
        self.step_count += 1
        done = self.step_count >= self.terminal_after
        return {
            0: {
                "observation": _observation(self.step_count),
                "action_mask": np.ones(105, dtype=np.bool_),
                "action_mask_info": {},
                "reward": 30.0 if done else -0.02,
                "done": done,
                "info": {
                    "done_reason": "success" if done else "",
                    "reliable_v4": True,
                    "reliable_v4_runtime_instance_id": "runtime-0",
                    "success": bool(done),
                    "observation_contract": "reliable_exact_endpoint_snapshot",
                    "observation_source": "reliable_exact_endpoint_snapshot",
                    "observation_semantics": "reliable_exact_endpoint_snapshot",
                    "reliable_execution": True,
                    "telemetry_observation": False,
                    "endpoint_identity_available": True,
                    "asynchronous_prefetch": False,
                    "asynchronous_prefetch_status": "obsolete_for_reliable_exact",
                    "telemetry_lookup_count": 0,
                    "snapshot_missing_count": 0,
                    "state_depth_skew_ns": 0,
                    "frame_contract_failures": 0,
                },
            }
        }

    def stop(self, **kwargs):
        del kwargs
        return {0: {"stopped": True}}

    def close(self, **kwargs):
        del kwargs
        self.closed = True


class _TwoWorkerPool:
    worker_ids = (0, 1)

    def __init__(self):
        self.step_count = {0: 0, 1: 0}
        self.closed = False

    def ready(self, **kwargs):
        del kwargs
        return {worker_id: {"ready": True} for worker_id in self.worker_ids}

    def reset(self, payloads, **kwargs):
        del kwargs
        responses = {}
        for worker_id in payloads:
            self.step_count[worker_id] = 0
            responses[worker_id] = {
                "observation": _observation(
                    0, episode_id="episode-{}".format(worker_id)
                ),
                "action_mask": np.ones(105, dtype=np.bool_),
                "action_mask_info": {},
            }
        return responses

    def step(self, actions, **kwargs):
        del kwargs
        responses = {}
        for worker_id in actions:
            self.step_count[worker_id] += 1
            responses[worker_id] = {
                "observation": _observation(
                    self.step_count[worker_id],
                    episode_id="episode-{}".format(worker_id),
                ),
                "action_mask": np.ones(105, dtype=np.bool_),
                "action_mask_info": {},
                "reward": -0.02,
                "done": False,
                "info": {
                    "done_reason": "",
                    "reliable_v4": True,
                    "reliable_v4_runtime_instance_id": "runtime-{}".format(worker_id),
                    "success": False,
                    "observation_contract": "reliable_exact_endpoint_snapshot",
                    "observation_source": "reliable_exact_endpoint_snapshot",
                    "observation_semantics": "reliable_exact_endpoint_snapshot",
                    "reliable_execution": True,
                    "telemetry_observation": False,
                    "endpoint_identity_available": True,
                    "asynchronous_prefetch": False,
                    "asynchronous_prefetch_status": "obsolete_for_reliable_exact",
                    "telemetry_lookup_count": 0,
                    "snapshot_missing_count": 0,
                    "state_depth_skew_ns": 0,
                    "frame_contract_failures": 0,
                },
            }
        return responses

    def stop(self, **kwargs):
        del kwargs
        return {worker_id: {"stopped": True} for worker_id in self.worker_ids}

    def close(self, **kwargs):
        del kwargs
        self.closed = True


def test_standard_runner_commits_awac_online_rows_and_updates_actor(tmp_path: Path):
    from planning.awac.calibration_runtime import CalibrationMission
    from planning.awac.interaction import BehaviorSource
    from planning.awac.online_runtime import StandardAWACOnlineRunner

    replay = _Replay()
    learner = _Learner()
    runner = StandardAWACOnlineRunner(
        missions=(
            CalibrationMission(
                episode_id="episode-0",
                mission_id="mission-0",
                start=(0.0, 0.0, 2.0),
                goal=(40.0, 0.0, 2.0),
            ),
        ),
        pool=_Pool(),
        replay=replay,
        learner=learner,
        action_selector=lambda **kwargs: 7,
        batch_size=1,
        learning_starts=1,
        actor_learning_starts=1,
        critic_burnin_updates=0,
        actor_update_interval=1,
        updates_per_step=1.0,
        online_env_steps=2,
        max_episodes=1,
        output_dir=tmp_path,
    )

    result = runner.run()

    assert result["status"] == "ONLINE_ENV_STEP_BUDGET_REACHED"
    assert result["online_env_steps"] == 2
    assert len(replay.rows) == 2
    assert all(
        row["behavior_source"] == int(BehaviorSource.AWAC_ONLINE)
        for row in replay.rows
    )
    assert learner.actor_update_count > 0
    assert result["actor_state_before"] != result["actor_state_after"]
    assert runner.pool.closed is True


def test_tensorboard_write_failure_is_diagnostic_only(tmp_path: Path):
    from planning.awac.online_runtime import StandardAWACOnlineRunner

    class FailingLogger:
        def log_runner(self, runner, *, force=False):
            del runner, force
            raise RuntimeError("synthetic TensorBoard write failure")

    replay = _Replay()
    runner = StandardAWACOnlineRunner(
        missions=(_mission(0),),
        pool=_Pool(terminal_after=2),
        replay=replay,
        learner=_Learner(),
        action_selector=lambda **kwargs: 7,
        batch_size=1,
        learning_starts=1,
        actor_learning_starts=1,
        critic_burnin_updates=0,
        actor_update_interval=1,
        updates_per_step=1.0,
        online_env_steps=2,
        max_episodes=1,
        output_dir=tmp_path,
        tensorboard_logger=FailingLogger(),
    )

    result = runner.run()

    assert result["status"] == "ONLINE_ENV_STEP_BUDGET_REACHED"
    assert len(replay.rows) == 2
    assert runner.tensorboard_logger is None
    assert "synthetic TensorBoard write failure" in runner.metrics["tensorboard_error"]


def test_tensorboard_snapshot_exposes_ess_independently_from_weight_clipping(
    tmp_path: Path,
):
    runner = _runner_for_schedule(
        tmp_path=tmp_path,
        missions=(_mission(0),),
        budget=1,
    )
    runner._online_metric_rows.append(
        {
            "awac_raw_weight_ess_fraction": 1.0,
            "awac_weight_ess_fraction": 0.25,
            "awac_weight_clip_fraction": 0.0,
        }
    )

    snapshot = runner.tensorboard_snapshot()["awac"]

    assert snapshot["raw_weight_ess_fraction"] == pytest.approx(1.0)
    assert snapshot["normalized_clipped_weight_ess_fraction"] == pytest.approx(0.25)
    assert snapshot["weight_clip_fraction"] == pytest.approx(0.0)
    assert snapshot["ess_batch_size"] == pytest.approx(1.0)
    assert snapshot["ess_aggregation_window_updates"] == pytest.approx(1.0)


def test_standard_online_counter_continues_from_97_to_100(tmp_path: Path):
    from planning.awac.online_runtime import StandardAWACOnlineRunner

    runner = StandardAWACOnlineRunner(
        missions=(_mission(0),),
        pool=_Pool(terminal_after=3),
        replay=_Replay(),
        learner=_Learner(),
        action_selector=lambda **kwargs: 7,
        batch_size=1,
        learning_starts=100,
        actor_learning_starts=100,
        critic_burnin_updates=0,
        actor_update_interval=1,
        updates_per_step=1.0,
        online_env_steps=100,
        max_episodes=1,
        output_dir=tmp_path,
    )
    runner._environment_step_count = 97
    runner._online_transitions_committed = 97

    result = runner.run()

    assert result["online_env_steps"] == 100
    assert result["online_transitions_committed"] == 100


def test_standard_checkpoint_snapshot_exposes_committed_counter(tmp_path: Path):
    runner = _runner_for_schedule(
        tmp_path=tmp_path,
        missions=(_mission(0),),
        budget=100,
    )
    runner.replay_identity = {"run_identity": "standard-run"}
    runner._runtime_ids = {0: "runtime-0"}
    runner._environment_step_count = 100
    runner._online_transitions_committed = 97

    snapshot = runner.checkpoint_snapshot({"status": "RUNNING"})

    assert snapshot["online_env_steps"] == 100
    assert snapshot["online_transitions_committed"] == 97


def test_standard_online_resume_restores_committed_counter_exactly(tmp_path: Path):
    runner = _runner_for_schedule(
        tmp_path=tmp_path / "source",
        missions=(_mission(0),),
        budget=100,
    )
    runner.replay_identity = {"run_identity": "standard-run"}
    runner._runtime_ids = {0: "runtime-0"}
    runner._environment_step_count = 100
    runner._online_transitions_committed = 97
    snapshot = runner.checkpoint_snapshot({"status": "RUNNING"})

    restored = _runner_for_schedule(
        tmp_path=tmp_path / "restored",
        missions=(_mission(0),),
        budget=100,
    )
    restored.replay_identity = {"run_identity": "standard-run"}
    restored._runtime_ids = {0: "runtime-0"}
    restored.restore_progress(
        snapshot["progress"],
        exact_resume_state=snapshot["standard_online_resume_state"],
    )

    assert restored.online_env_steps == 100
    assert restored._online_transitions_committed == 97


def _mission(index: int):
    from planning.awac.calibration_runtime import CalibrationMission

    return CalibrationMission(
        episode_id="episode-{}".format(index),
        mission_id="mission-{}".format(index),
        start=(0.0, 0.0, 2.0),
        goal=(40.0, 0.0, 2.0),
    )


def _runner_for_schedule(*, tmp_path: Path, missions, budget: int):
    from planning.awac.online_runtime import StandardAWACOnlineRunner

    return StandardAWACOnlineRunner(
        missions=tuple(missions),
        pool=_Pool(),
        replay=_Replay(),
        learner=_Learner(),
        action_selector=lambda **kwargs: 7,
        batch_size=1,
        learning_starts=3,
        actor_learning_starts=3,
        critic_burnin_updates=0,
        actor_update_interval=1,
        updates_per_step=1.0,
        online_env_steps=budget,
        max_episodes=len(tuple(missions)),
        output_dir=tmp_path,
    )


def test_actor_is_not_updated_before_warmup_but_updates_after_threshold(tmp_path: Path):
    before = _runner_for_schedule(
        tmp_path=tmp_path / "before", missions=(_mission(0),), budget=2
    )
    before_result = before.run()
    assert before_result["online_env_steps"] == 2
    assert before_result["critic_update_count"] == 0
    assert before_result["actor_update_count"] == 0

    after = _runner_for_schedule(
        tmp_path=tmp_path / "after", missions=(_mission(0), _mission(1)), budget=4
    )
    after_result = after.run()
    assert after_result["online_env_steps"] == 4
    assert after_result["critic_update_count"] > 0
    assert after_result["actor_update_count"] > 0
    assert after_result["actor_changed"] is True


@pytest.mark.parametrize("budget", (1, 2, 5))
def test_single_worker_budget_consumes_exactly_requested_steps(tmp_path: Path, budget):
    runner = _runner_for_schedule(
        tmp_path=tmp_path,
        missions=(_mission(0), _mission(1), _mission(2)),
        budget=budget,
    )

    result = runner.run()

    assert result["online_env_steps"] == budget
    assert result["online_env_steps"] <= result["online_env_steps_budget"]


def test_parallel_budget_shortfall_is_bounded_by_inflight_worker_wave(tmp_path: Path):
    from planning.awac.online_runtime import StandardAWACOnlineRunner

    runner = StandardAWACOnlineRunner(
        missions=(_mission(0), _mission(1)),
        pool=_TwoWorkerPool(),
        replay=_Replay(),
        learner=_Learner(),
        action_selector=lambda **kwargs: 7,
        batch_size=1,
        learning_starts=100,
        actor_learning_starts=100,
        critic_burnin_updates=0,
        actor_update_interval=1,
        updates_per_step=1.0,
        online_env_steps=5,
        max_episodes=2,
        output_dir=tmp_path,
    )

    result = runner.run()

    assert result["online_env_steps"] == 4
    assert result["online_env_steps_budget"] - result["online_env_steps"] == 1
    assert result["online_transitions_committed"] == 0


def test_mixed_replay_keeps_calibration_and_online_behavior_sources(tmp_path: Path):
    from planning.awac.interaction import BehaviorSource
    from planning.awac.replay import AWACReplayBuffer

    source = AWACReplayBuffer.create(
        tmp_path / "source",
        capacity=8,
        depth_shape=(1, 2, 2),
        vector_dim=3,
        action_dim=5,
        training_config_sha256="a" * 64,
        bc_checkpoint_sha256="b" * 64,
        task_contract_id="task",
        task_contract_sha256="c" * 64,
        mpl_contract_sha256="d" * 64,
        run_identity="calibration-source",
        run_contract_sha256="e" * 64,
        behavior_source_phase="critic_calibration",
    )
    source.add_batch(
        [
            _transition(1, behavior_source=int(BehaviorSource.BC_CALIBRATION)),
            _transition(2, behavior_source=int(BehaviorSource.BC_CALIBRATION)),
        ]
    )
    source.flush()
    clone = AWACReplayBuffer.clone_from(
        source,
        tmp_path / "standard" / "replay",
        run_identity="standard-run",
        run_contract_sha256="f" * 64,
        training_config_sha256="g" * 64,
        behavior_source_phase="awac_training",
    )
    clone.add(
        **_transition(3, behavior_source=int(BehaviorSource.AWAC_ONLINE))
    )
    clone.flush()
    assert clone.arrays["behavior_source"][:3].tolist() == [
        int(BehaviorSource.BC_CALIBRATION),
        int(BehaviorSource.BC_CALIBRATION),
        int(BehaviorSource.AWAC_ONLINE),
    ]


def test_persistent_replay_identity_is_refreshed_at_checkpoint_boundary(tmp_path: Path):
    from planning.awac.checkpoint import calibration_replay_identity
    from planning.awac.interaction import BehaviorSource
    from planning.awac.replay import AWACReplayBuffer
    from planning.awac.trainer import _standard_replay_identity
    from planning.awac.online_runtime import StandardAWACOnlineRunner

    replay = AWACReplayBuffer.create(
        tmp_path / "source",
        capacity=8,
        depth_shape=(1, 2, 2),
        vector_dim=3,
        action_dim=5,
        training_config_sha256="a" * 64,
        bc_checkpoint_sha256="b" * 64,
        task_contract_id="task",
        task_contract_sha256="c" * 64,
        mpl_contract_sha256="d" * 64,
        run_identity="standard-run",
        run_contract_sha256="e" * 64,
        behavior_source_phase="awac_training",
    )
    runner = StandardAWACOnlineRunner.__new__(StandardAWACOnlineRunner)
    runner.replay = replay
    runner.replay_identity = calibration_replay_identity(replay)
    assert _standard_replay_identity(replay)["behavior_source_phase"] == "awac_training"
    replay.add(
        **_transition(1, behavior_source=int(BehaviorSource.AWAC_ONLINE))
    )
    refreshed = runner._current_replay_identity()
    assert refreshed["replay_size"] == 1
    assert refreshed["replay_total_added"] == 1
    assert refreshed["replay_metadata_sha256"] == calibration_replay_identity(
        replay
    )["replay_metadata_sha256"]


def test_standard_exact_resume_state_inventory_and_rng_contract():
    from planning.awac.checkpoint import (
        build_standard_online_exact_resume_state,
        validate_standard_online_exact_resume_state,
    )

    progress = {
        "schema_id": "awac_standard_online_runtime_v1",
        "mission_source_identity": {"sha256": "a" * 64},
        "ordered_mission_ids": ["mission-0"],
        "ordered_episode_ids": ["episode-0"],
        "next_mission_index": 1,
        "completed_mission_ids": ["mission-0"],
        "completed_episode_ids": ["episode-0"],
        "environment_step_count": 2,
        "online_env_steps": 2,
        "online_env_steps_budget": 4,
        "online_transitions_committed": 2,
        "completed_episode_count": 1,
        "critic_update_count": 1,
        "actor_update_count": 1,
        "actor_optimizer_step_count": 1,
        "update_step": 1,
        "phase1_state": {"actor_update_enabled": True},
    }
    state = build_standard_online_exact_resume_state(
        learner_state={
            "actor_state_dict": {"weight": [1.0]},
            "critic1_state_dict": {"weight": [2.0]},
            "critic2_state_dict": {"weight": [3.0]},
            "update_step": 1,
            "actor_update_count": 1,
            "actor_awac_update_count": 1,
            "actor_recovery_update_count": 0,
            "actor_trust_region_rejection_count": 0,
            "actor_optimizer_step_count": 1,
            "critic_update_count": 1,
        },
        producer_rng=np.random.RandomState(4027),
        torch=None,
        mission_source_identity=progress["mission_source_identity"],
        mission_progress=progress,
        runtime_identity={0: "runtime-0"},
        phase1_state=progress["phase1_state"],
        replay_identity={"run_identity": "standard-run", "replay_size": 2},
    )
    validated = validate_standard_online_exact_resume_state(state)
    assert validated["required_state_count"] == 14
    assert validated["environment_and_online_counters"][
        "online_transitions_committed"
    ] == 2


def test_standard_exact_resume_fingerprint_supports_integer_optimizer_keys():
    from planning.awac.checkpoint import build_standard_online_exact_resume_state

    progress = {
        "schema_id": "awac_standard_online_runtime_v1",
        "mission_source_identity": {"sha256": "a" * 64},
        "ordered_mission_ids": ["mission-0"],
        "ordered_episode_ids": ["episode-0"],
        "next_mission_index": 1,
        "completed_mission_ids": ["mission-0"],
        "completed_episode_ids": ["episode-0"],
        "environment_step_count": 2,
        "online_env_steps": 2,
        "online_env_steps_budget": 4,
        "online_transitions_committed": 2,
        "completed_episode_count": 1,
        "critic_update_count": 1,
        "actor_update_count": 1,
        "actor_optimizer_step_count": 1,
        "update_step": 1,
    }
    state = build_standard_online_exact_resume_state(
        learner_state={
            "actor_state_dict": {"weight": [1.0]},
            "critic1_state_dict": {"weight": [2.0]},
            "critic2_state_dict": {"weight": [3.0]},
            "actor_optimizer_state_dict": {
                "state": {0: {"step": 1}},
                "param_groups": [{"params": [0]}],
            },
            "update_step": 1,
            "actor_update_count": 1,
            "actor_awac_update_count": 1,
            "actor_recovery_update_count": 0,
            "actor_trust_region_rejection_count": 0,
            "actor_optimizer_step_count": 1,
            "critic_update_count": 1,
        },
        producer_rng=np.random.RandomState(4027),
        torch=None,
        mission_source_identity=progress["mission_source_identity"],
        mission_progress=progress,
        runtime_identity={0: "runtime-0"},
        phase1_state={"actor_update_enabled": True},
        replay_identity={"run_identity": "standard-run", "replay_size": 2},
    )

    assert len(state["learner_state_sha256"]) == 64


def test_resume_fingerprint_distinguishes_integer_and_string_mapping_keys():
    from planning.awac.checkpoint import _resume_state_sha256

    assert _resume_state_sha256({0: "same"}) != _resume_state_sha256(
        {"0": "same"}
    )
    assert _resume_state_sha256({True: "same"}) != _resume_state_sha256(
        {1: "same"}
    )
    assert _resume_state_sha256({0: "int", "0": "str"}) != _resume_state_sha256(
        {0: "str", "0": "int"}
    )


def test_resume_fingerprint_nested_mapping_order_is_deterministic():
    from planning.awac.checkpoint import _resume_state_sha256

    left = {
        "outer": {
            0: {"bytes": b"zero", "value": np.int64(3)},
            1: [None, ("tuple", np.float32(0.5))],
        }
    }
    right = {
        "outer": {
            1: [None, ("tuple", np.float32(0.5))],
            0: {"value": np.int64(3), "bytes": b"zero"},
        }
    }

    assert _resume_state_sha256(left) == _resume_state_sha256(right)


def test_resume_fingerprint_supports_numpy_and_torch_values():
    torch = pytest.importorskip("torch")
    from planning.awac.checkpoint import _resume_state_sha256

    value = {
        "numpy_int": np.int64(7),
        "numpy_float": np.float32(0.25),
        "array": np.asarray([[1, 2]], dtype=np.int16),
        "tensor": torch.tensor([3.0], dtype=torch.float32),
        "bytes": b"payload",
        "tuple": (None, "value"),
    }

    assert len(_resume_state_sha256(value)) == 64


def test_resume_fingerprint_rejects_unsupported_mapping_keys():
    from planning.awac.checkpoint import _resume_state_sha256

    class UnsupportedKey:
        pass

    with pytest.raises(TypeError, match="mapping key"):
        _resume_state_sha256({UnsupportedKey(): "value"})


def test_standard_online_start_manifest_persists_actor_identity(tmp_path: Path):
    from planning.awac.trainer import _write_standard_online_start_manifest

    path = _write_standard_online_start_manifest(
        output_dir=tmp_path,
        resume_checkpoint_path=tmp_path / "checkpoint.pt",
        resume_checkpoint_sha256="b" * 64,
        source_checkpoint_sha256="c" * 64,
        task_contract_sha256_value="d" * 64,
        actor_state_before_sha256="a" * 64,
        starting_replay_size=5142,
        starting_replay_total_added=5142,
        online_env_steps_budget=3965,
    )

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["actor_state_before_sha256"] == "a" * 64
    assert payload["starting_replay_size"] == 5142
    assert payload["online_env_steps_budget"] == 3965


def test_standard_online_failure_payload_keeps_finalization_evidence():
    from planning.awac.trainer import _standard_online_failure_payload

    runner = SimpleNamespace(
        _actor_before="a" * 64,
        _actor_fingerprint=lambda: "b" * 64,
        progress={"online_env_steps": 3964, "online_transitions_committed": 3941},
        metrics={"replay_final_flush": "PASS", "runtime_close": "PASS"},
        replay=SimpleNamespace(
            directory=Path("replay"),
            size=9083,
            total_added=9083,
            metadata={
                "position": 9083,
                "total_added": 9083,
                "behavior_source_phase": "awac_training",
                "replay_contract_sha256": "c" * 64,
            },
        ),
    )

    payload = _standard_online_failure_payload(
        runner=runner,
        error=KeyError("0"),
        error_traceback="traceback",
        source_checkpoint_sha256="d" * 64,
        mission_source_identity={"sha256": "e" * 64},
        source_replay_identity={"replay_size": 5142},
        runtime_identity={"worker_count": 2},
        checkpoint_save_reason="run_complete",
        checkpoint_final_commit="NOT_ATTEMPTED",
        checkpoint_paths=[],
    )

    assert payload["checkpoint_save_reason"] == "run_complete"
    assert payload["checkpoint_final_commit"] == "NOT_ATTEMPTED"
    assert payload["replay_committed_state"]["size"] == 9083
    assert payload["runtime_metrics"]["replay_final_flush"] == "PASS"
    assert payload["runtime_metrics"]["runtime_close"] == "PASS"
    assert payload["actor_state_before_sha256"] == "a" * 64
    assert payload["actor_state_after_sha256"] == "b" * 64
    assert payload["actor_changed"] is True


def test_standard_online_checkpoint_roundtrip_preserves_resume_fingerprint(
    tmp_path: Path,
):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from planning.awac.checkpoint import (
        _learner_state_from_checkpoint,
        _resume_state_sha256,
        build_standard_online_exact_resume_state,
        load_awac_checkpoint,
        save_awac_checkpoint,
        validate_standard_awac_checkpoint_payload,
    )
    from planning.awac.learner import (
        AWACOptimizationConfig,
        DiscreteAWACLearner,
    )
    from planning.awac.trainer import (
        _build_standard_checkpoint_payload,
        _resolved_training_config,
        build_parser,
    )
    from planning.bc.model import build_model
    from planning.common.hashing import file_sha256

    torch.manual_seed(4027)
    bc_model = build_model(nn, depth_channels=1)
    bc_checkpoint = {"model_state_dict": bc_model.state_dict()}
    bc_path = tmp_path / "bc_checkpoint.pt"
    torch.save(bc_checkpoint, str(bc_path))

    learner = DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_state_dict=bc_checkpoint["model_state_dict"],
        depth_channels=1,
        config=AWACOptimizationConfig(
            actor_depth_lr=1.0e-6,
            critic_depth_lr=1.0e-5,
        ),
    )
    # A zero-gradient optimizer step creates the same real PyTorch optimizer
    # state shape as a Standard online update, including integer parameter IDs,
    # without exercising any AWAC math in this checkpoint-only test.
    for parameter in learner.actor.parameters():
        parameter.grad = torch.zeros_like(parameter)
    learner.actor_optimizer.step()
    for parameter in learner.critic1.parameters():
        parameter.grad = torch.zeros_like(parameter)
    for parameter in learner.critic2.parameters():
        parameter.grad = torch.zeros_like(parameter)
    learner.critic_optimizer.step()
    learner.update_step = 1
    learner.actor_update_count = 1
    learner.actor_awac_update_count = 1
    learner.critic_update_count = 1
    learner.actor_optimizer_step_count = 1

    parser = build_parser()
    args = parser.parse_args(
        [
            "--bc-checkpoint",
            str(bc_path),
            "--phase",
            "awac_training",
            "--train-index",
            str(tmp_path / "missions.csv"),
            "--out-dir",
            str(tmp_path / "run"),
            "--replay-dir",
            str(tmp_path / "replay"),
            "--worker-spec-file",
            str(tmp_path / "workers.json"),
            "--reliable-v4",
            "--online-env-steps",
            "100",
        ]
    )
    mpl_sha = "m" * 64
    resolved = _resolved_training_config(
        args,
        source_bc_sha256=file_sha256(bc_path),
        mpl_sha256=mpl_sha,
    )
    mission_source_identity = {
        "sha256": "n" * 64,
        "row_count": 1,
        "ordered_ids_sha256": "o" * 64,
    }
    progress = {
        "schema_id": "awac_standard_online_runtime_v1",
        "mission_source_identity": dict(mission_source_identity),
        "ordered_mission_ids": ["mission-0"],
        "ordered_episode_ids": ["episode-0"],
        "next_mission_index": 1,
        "completed_mission_ids": ["mission-0"],
        "completed_episode_ids": ["episode-0"],
        "environment_step_count": 100,
        "online_env_steps": 100,
        "online_env_steps_budget": 100,
        "online_transitions_committed": 97,
        "completed_episode_count": 1,
        "critic_update_count": learner.critic_update_count,
        "actor_update_count": learner.actor_update_count,
        "actor_optimizer_step_count": learner.actor_optimizer_step_count,
        "update_step": learner.update_step,
    }
    phase1_state = {"actor_update_enabled": True, "phase": "standard_awac"}
    replay_identity = {
        "run_identity": "standard-run",
        "behavior_source_phase": "awac_training",
        "replay_size": 99,
        "replay_total_added": 99,
        "replay_metadata_sha256": "p" * 64,
    }
    source_replay_identity = {
        "run_identity": "calibration-source",
        "behavior_source_phase": "critic_calibration",
        "replay_size": 2,
        "replay_total_added": 2,
        "replay_metadata_sha256": "q" * 64,
    }
    exact_resume_state = build_standard_online_exact_resume_state(
        learner_state=learner.state_dict(),
        producer_rng=np.random.RandomState(4027),
        torch=torch,
        mission_source_identity=mission_source_identity,
        mission_progress=progress,
        runtime_identity={"0": "runtime-0"},
        phase1_state=phase1_state,
        replay_identity=replay_identity,
    )
    snapshot = {
        "progress": progress,
        "standard_online_resume_state": exact_resume_state,
        "phase1_state": phase1_state,
        "runtime_identity": {"0": "runtime-0"},
        "environment_step_count": 100,
        "online_env_steps": 100,
        "online_env_steps_budget": 100,
        "online_transitions_committed": 97,
        "starting_replay_size": 2,
        "starting_replay_total_added": 2,
        "completed_episode_count": 1,
        "runtime_metrics": {},
        "learner_metrics": {},
    }
    payload = _build_standard_checkpoint_payload(
        learner=learner,
        args=args,
        bc_checkpoint=bc_checkpoint,
        bc_checkpoint_path=bc_path,
        mpl_sha=mpl_sha,
        resolved=resolved,
        replay_identity=replay_identity,
        source_replay_identity=source_replay_identity,
        source_calibration_checkpoint_sha256="r" * 64,
        mission_source_identity=mission_source_identity,
        worker_identity={"0": "worker-0"},
        managed_runtime_identity={"runtime_instance_id": "runtime-0"},
        snapshot=snapshot,
    )

    missing_payload = dict(payload)
    missing_payload.pop("online_transitions_committed")
    with pytest.raises(ValueError, match="online_transitions_committed"):
        validate_standard_awac_checkpoint_payload(missing_payload)

    missing_counter_snapshot = dict(snapshot)
    missing_counter_snapshot.pop("online_transitions_committed")
    with pytest.raises(ValueError, match="online_transitions_committed"):
        _build_standard_checkpoint_payload(
            learner=learner,
            args=args,
            bc_checkpoint=bc_checkpoint,
            bc_checkpoint_path=bc_path,
            mpl_sha=mpl_sha,
            resolved=resolved,
            replay_identity=replay_identity,
            source_replay_identity=source_replay_identity,
            source_calibration_checkpoint_sha256="r" * 64,
            mission_source_identity=mission_source_identity,
            worker_identity={"0": "worker-0"},
            managed_runtime_identity={"runtime_instance_id": "runtime-0"},
            snapshot=missing_counter_snapshot,
        )

    validate_standard_awac_checkpoint_payload(payload)
    fingerprint_before = _resume_state_sha256(
        _learner_state_from_checkpoint(payload)
    )
    checkpoint_path = tmp_path / "checkpoint_last.pt"
    save_awac_checkpoint(checkpoint_path, payload, torch=torch)
    loaded = load_awac_checkpoint(checkpoint_path, torch=torch)
    validate_standard_awac_checkpoint_payload(loaded)
    fingerprint_after = _resume_state_sha256(
        _learner_state_from_checkpoint(loaded)
    )

    assert fingerprint_before == fingerprint_after
    assert loaded["standard_online_resume_state"]["learner_state_sha256"] == (
        fingerprint_after
    )
    assert isinstance(
        loaded["actor_optimizer_state_dict"]["state"], dict
    )
    assert all(isinstance(key, int) for key in loaded["actor_optimizer_state_dict"]["state"])
    assert loaded["online_transition_count"] == 97


def test_standard_checkpoint_rejects_replay_committed_counter_mismatch():
    from planning.awac.checkpoint import validate_standard_online_replay_counter

    replay = SimpleNamespace(
        arrays={"behavior_source": np.ones((96,), dtype=np.uint8)},
        size=96,
        capacity=128,
        total_added=96,
    )
    with pytest.raises(ValueError, match="AWAC_ONLINE"):
        validate_standard_online_replay_counter(
            {
                "online_transitions_committed": 97,
                "online_transition_count": 97,
                "starting_replay_total_added": 0,
            },
            replay,
        )


def test_standard_online_summary_contract_and_validator_cli(tmp_path: Path):
    import importlib.util

    from planning.awac.online_runtime import build_standard_online_summary

    summary = build_standard_online_summary(
        result={
            "online_transitions_committed": 1,
            "online_env_steps": 2,
            "online_env_steps_budget": 4,
            "completed_episode_count": 1,
            "actor_update_count": 1,
            "critic_update_count": 2,
            "actor_state_before": "a" * 64,
            "actor_state_after": "b" * 64,
            "metrics": {
                "learner": {"critic_td_loss_mean": 1.0},
                "nan_inf_count": 0,
                "nonfinite_metric_count": 0,
            },
            "stop_reason": "ONLINE_ENV_STEP_BUDGET_REACHED",
        },
        phase="awac_training",
        start_checkpoint_identity={"path": "checkpoint.pt", "sha256": "c" * 64},
        starting_replay_size=2,
        ending_replay_size=3,
        behavior_source_counts={"BC_CALIBRATION": 2, "AWAC_ONLINE": 1},
        runtime_identity={"worker_count": 1},
        checkpoint_paths=["checkpoint_last.pt"],
    )
    path = tmp_path / "summary.json"
    path.write_text(__import__("json").dumps(summary), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / (
        "validate_standard_awac_online_summary.py"
    )
    spec = importlib.util.spec_from_file_location("standard_summary_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--summary", str(path)]) == 0

    invalid = dict(summary)
    invalid["online_env_steps"] = 5
    with pytest.raises(ValueError, match="budget"):
        from planning.awac.online_runtime import validate_standard_online_summary

        validate_standard_online_summary(invalid)
