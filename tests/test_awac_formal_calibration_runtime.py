"""Red tests for the formal online AWAC calibration runtime seam.

These tests intentionally use a fake ParallelEnvPool.  They specify the
parent-side producer contract without opening ROS, ZMQ, Unity, or Bridge
processes.
"""

from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import pytest
import json

from planning.awac.calibration import CriticCalibrationConfig, summarize_calibration_window
from planning.awac.interaction import BehaviorSource
from planning.awac.calibration_runtime import (
    CalibrationMission,
    CalibrationReplayProducer,
    CalibrationRuntimeError,
    load_calibration_missions,
)
from planning.bc.model import require_torch
from planning.contracts.observation import exact_endpoint_metadata


pytestmark = pytest.mark.unit


def _observation(step: int, *, episode_id: str = "episode-0"):
    position = [float(step), 0.0, 2.0]
    return {
        "depth": np.full((90, 160), 0.5, dtype=np.float32),
        "state": {
            "position": position,
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
        **exact_endpoint_metadata(),
    }


def _mission(index: int) -> CalibrationMission:
    return CalibrationMission(
        episode_id="episode-{}".format(index),
        mission_id="mission-{}".format(index),
        start=(0.0, 0.0, 2.0),
        goal=(40.0, 0.0, 2.0),
        source_row={"episode_id": "episode-{}".format(index)},
    )


class _Replay:
    def __init__(self):
        self.rows = []

    @property
    def size(self):
        return len(self.rows)

    def add(self, **row):
        self.rows.append(dict(row))

    def sample(self, batch_size, *, rng, torch, device):
        return {"batch_size": int(batch_size)}

    def flush(self):
        return None


class _Learner:
    def __init__(self):
        self.critic_update_count = 0
        self.actor_update_count = 0
        self.actor_optimizer_step_count = 0
        self.device = "cpu"

    def freeze_actor_for_calibration(self):
        return None

    def calibration_update(self, batch):
        self.critic_update_count += 1
        return {"critic_update_count": float(self.critic_update_count)}


class _Pool:
    def __init__(
        self,
        *,
        terminal_after=2,
        fail=False,
        terminal_reason="success",
        zero_next_mask=False,
        zero_initial_mask=False,
    ):
        self.worker_ids = (0,)
        self.terminal_after = int(terminal_after)
        self.fail = bool(fail)
        self.terminal_reason = str(terminal_reason)
        self.zero_next_mask = bool(zero_next_mask)
        self.zero_initial_mask = bool(zero_initial_mask)
        self.step_count = 0
        self.reset_calls = []
        self.closed = False

    def ready(self, **kwargs):
        return {0: {"ready": True}}

    def reset(self, payload):
        self.reset_calls.append(dict(payload))
        initial_mask = np.ones(105, dtype=np.bool_)
        if self.zero_initial_mask:
            initial_mask[:] = False
        return {
            0: {
                "observation": _observation(0),
                "action_mask": initial_mask,
                "action_mask_info": {},
            }
        }

    def step(self, actions):
        if self.fail:
            raise RuntimeError("fake worker failure")
        self.step_count += 1
        done = self.step_count >= self.terminal_after
        next_mask = np.ones(105, dtype=np.bool_)
        if self.zero_next_mask:
            next_mask[:] = False
        terminal_flags = {
            "success": {"success": True},
            "collision": {"collided": True},
            "dead_end": {"dead_end": True},
        }
        return {
            0: {
                "observation": _observation(self.step_count),
                "action_mask": next_mask,
                "action_mask_info": {},
                "reward": 30.0 if done and self.terminal_reason == "success" else (-100.0 if done else -0.02),
                "done": done,
                "info": {
                    "done_reason": self.terminal_reason if done else "",
                    "reliable_v4": True,
                    "reliable_v4_runtime_instance_id": "runtime-0",
                    **(terminal_flags.get(self.terminal_reason, {}) if done else {}),
                    **exact_endpoint_metadata(),
                },
            }
        }

    def close(self):
        self.closed = True


class _TwoWorkerPool:
    """Small deterministic two-worker source for mission-cursor resume tests."""

    worker_ids = (0, 1)

    def __init__(self):
        self.closed = False
        self._next_episode = 0
        self._episodes = {}

    def ready(self, **kwargs):
        return {worker_id: {"ready": True} for worker_id in self.worker_ids}

    def reset(self, payloads, **kwargs):
        result = {}
        for worker_id in sorted(payloads):
            episode_id = "episode-{}".format(self._next_episode)
            self._next_episode += 1
            self._episodes[int(worker_id)] = episode_id
            observation = _observation(0, episode_id=episode_id)
            observation["reset_id"] = "reset-{}".format(episode_id)
            result[int(worker_id)] = {
                "observation": observation,
                "action_mask": np.ones(105, dtype=np.bool_),
                "action_mask_info": {},
            }
        return result

    def step(self, actions, **kwargs):
        result = {}
        for worker_id in sorted(actions):
            episode_id = self._episodes[int(worker_id)]
            observation = _observation(1, episode_id=episode_id)
            observation["reset_id"] = "reset-{}".format(episode_id)
            result[int(worker_id)] = {
                "observation": observation,
                "action_mask": np.ones(105, dtype=np.bool_),
                "action_mask_info": {},
                "reward": 30.0,
                "done": True,
                "info": {
                    "done_reason": "success",
                    "success": True,
                    "reliable_v4": True,
                    "reliable_v4_runtime_instance_id": "runtime-{}".format(
                        worker_id
                    ),
                    **exact_endpoint_metadata(),
                },
            }
        return result

    def close(self):
        self.closed = True


def _producer(tmp_path, *, gate_states=None, **kwargs):
    replay = _Replay()
    learner = _Learner()
    states = iter(gate_states or ["PENDING"])
    gate_calls = []

    def gate(**values):
        gate_calls.append(values)
        try:
            state = next(states)
        except StopIteration:
            state = "PENDING"
        return {"state": state}

    producer = CalibrationReplayProducer(
        missions=[_mission(0), _mission(1), _mission(2)],
        pool=_Pool(**kwargs.pop("pool_kwargs", {})),
        replay=replay,
        learner=learner,
        train_episode_ids={"episode-0", "episode-1", "episode-2"},
        holdout_episode_ids=set(),
        action_selector=kwargs.pop("action_selector", lambda **_: 7),
        gate_evaluator=gate,
        calibration_window_interval_steps=1,
        min_replay_transitions=1,
        min_completed_episodes=1,
        min_critic_updates=1,
        min_holdout_episodes=1,
        min_stability_windows=1,
        batch_size=1,
        learning_starts=1,
        updates_per_step=1.0,
        max_transitions=kwargs.pop("max_transitions", 20),
        max_episodes=kwargs.pop("max_episodes", 20),
        output_dir=tmp_path,
        **kwargs
    )
    return producer, replay, learner, gate_calls


def _formal_gate_window():
    return summarize_calibration_window(
        critic1_td_loss=1.0,
        critic2_td_loss=1.1,
        holdout_td_loss=1.05,
        q1=[4.0, 2.0, 3.0, 1.0],
        q2=[3.8, 1.8, 3.1, 0.9],
        rewards=[1.0, -1.0, 1.0, -1.0],
        returns=[2.0, -2.0, 2.0, -2.0],
        terminal_reasons=["success", "collision", "success", "timeout"],
    )


def _default_gate_producer(tmp_path):
    replay = _Replay()
    replay.rows = [{} for _ in range(95)]
    learner = _Learner()
    window = _formal_gate_window()
    producer = CalibrationReplayProducer(
        missions=[_mission(index) for index in range(5)],
        pool=_Pool(),
        replay=replay,
        learner=learner,
        train_episode_ids={"episode-0", "episode-1", "episode-2"},
        holdout_episode_ids={"episode-3", "episode-4"},
        action_selector=lambda **_: 1,
        holdout_evaluator=lambda _: dict(window),
        calibration_config=CriticCalibrationConfig(),
        calibration_window_interval_steps=1,
        batch_size=1,
        learning_starts=1,
        updates_per_step=1.0,
        max_transitions=1000,
        max_episodes=1000,
        output_dir=tmp_path,
    )
    producer._environment_step_count = 128
    producer._completed_episode_count = 5
    return producer


def test_missing_training_index_fails_before_runtime(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_calibration_missions(tmp_path / "missing.csv", max_steps=45)


def test_final_test_training_index_is_rejected(tmp_path: Path):
    index = tmp_path / "final_test.csv"
    index.write_text(
        "episode_id,mission_id,start_x,start_y,start_z,goal_x,goal_y,goal_z,"
        "task_contract_id,task_contract_schema_version,task_contract_sha256,max_primitive_steps\n"
        "e0,m0,0,0,2,40,0,2,xm_3d_flight_z1_3,2,bad,45\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"role":"FINAL_TEST_ONLY"}', encoding="utf-8")
    with pytest.raises(ValueError, match="FINAL_TEST_ONLY"):
        load_calibration_missions(index, max_steps=45, final_test_manifest=manifest)


def test_runtime_transition_is_appended_with_raw_reward_and_bc_source(tmp_path):
    producer, replay, learner, _ = _producer(tmp_path, pool_kwargs={"terminal_after": 2})
    result = producer.run()
    assert result["status"] == "PENDING"
    assert len(replay.rows) >= 2
    assert replay.rows[-1]["done"] is True
    assert replay.rows[-1]["reward"] == 30.0
    assert replay.rows[0]["behavior_source"] == int(BehaviorSource.BC_CALIBRATION)
    assert learner.critic_update_count >= 1


def test_runtime_failure_does_not_append_or_fabricate_terminal(tmp_path):
    producer, replay, _, _ = _producer(
        tmp_path, pool_kwargs={"fail": True}, max_transitions=2
    )
    with pytest.raises(CalibrationRuntimeError):
        producer.run()
    assert replay.rows == []
    assert producer.metrics["runtime_failed_attempt_count"] == 1
    assert producer.metrics["replay_final_flush"] == "PASS"
    assert producer.metrics["runtime_close"] == "PASS"
    assert producer.pool.closed is True


def test_dead_end_terminal_with_zero_next_mask_commits_replay_transition(tmp_path):
    producer, replay, _, _ = _producer(
        tmp_path,
        max_episodes=1,
        pool_kwargs={
            "terminal_after": 1,
            "terminal_reason": "dead_end",
            "zero_next_mask": True,
        },
    )

    result = producer.run()

    assert result["status"] == "BLOCKED_PENDING"
    assert len(replay.rows) == 1
    assert replay.rows[0]["done"] is True
    assert replay.rows[0]["next_action_mask"].shape == (105,)
    assert int(replay.rows[0]["next_action_mask"].sum()) == 0


@pytest.mark.parametrize(
    "terminal_reason", ("collision", "success", "timeout", "hard_altitude")
)
def test_terminal_zero_next_mask_commits_canonical_terminal_outcomes(
    tmp_path,
    terminal_reason,
):
    producer, replay, _, _ = _producer(
        tmp_path,
        max_episodes=1,
        pool_kwargs={
            "terminal_after": 1,
            "terminal_reason": terminal_reason,
            "zero_next_mask": True,
        },
    )

    producer.run()

    assert len(replay.rows) == 1
    assert replay.rows[0]["done"] is True
    assert int(replay.rows[0]["next_action_mask"].sum()) == 0


def test_nonterminal_zero_next_mask_remains_fail_closed(tmp_path):
    producer, replay, _, _ = _producer(
        tmp_path,
        pool_kwargs={
            "terminal_after": 2,
            "zero_next_mask": True,
        },
    )

    with pytest.raises(CalibrationRuntimeError, match="action mask has no valid action"):
        producer.run()

    assert replay.rows == []


def test_current_zero_mask_fails_before_bc_action_selection(tmp_path):
    action_selection_calls = []

    def select_action(**_):
        action_selection_calls.append(True)
        return 7

    producer, replay, _, _ = _producer(
        tmp_path,
        pool_kwargs={"zero_initial_mask": True},
        action_selector=select_action,
    )

    with pytest.raises(CalibrationRuntimeError, match="action mask has no valid action"):
        producer.run()

    assert action_selection_calls == []
    assert replay.rows == []


def test_continuous_gate_pending_pending_pass_stops_and_closes(tmp_path):
    producer, replay, learner, gate_calls = _producer(
        tmp_path, gate_states=["PENDING", "PENDING", "PASS"]
    )
    result = producer.run()
    assert result["status"] == "PASS"
    assert result["stop_reason"] == "CALIBRATION_GATE_PASS"
    assert len(gate_calls) == 3
    assert producer.pool.closed is True
    assert learner.actor_update_count == 0
    assert learner.actor_optimizer_step_count == 0
    assert len(replay.rows) == 3


def test_pending_cap_stops_without_pass(tmp_path):
    producer, _, _, _ = _producer(
        tmp_path, gate_states=["PENDING"], max_transitions=1
    )
    result = producer.run()
    assert result["status"] == "BLOCKED_PENDING"
    assert result["pass_checkpoint_allowed"] is False
    assert producer.pool.closed is True


def test_diverged_gate_stops_without_pass(tmp_path):
    producer, replay, _, gate_calls = _producer(
        tmp_path, gate_states=["FAIL_DIVERGED"], max_transitions=20
    )
    result = producer.run()
    assert result["status"] == "FAIL_DIVERGED"
    assert result["stop_reason"] == "CALIBRATION_DIVERGED"
    assert len(gate_calls) == 1
    assert producer.pool.closed is True
    assert result["pass_checkpoint_allowed"] is False
    assert replay.size == 0
    assert producer.pool.step_count == 1


def test_holdout_episode_never_enters_replay(tmp_path):
    replay = _Replay()
    learner = _Learner()
    producer = CalibrationReplayProducer(
        missions=[_mission(0), _mission(1)],
        pool=_Pool(terminal_after=1),
        replay=replay,
        learner=learner,
        train_episode_ids={"episode-0"},
        holdout_episode_ids={"episode-1"},
        action_selector=lambda **_: 1,
        gate_evaluator=lambda **_: {"state": "PENDING"},
        calibration_window_interval_steps=1,
        min_replay_transitions=1,
        min_completed_episodes=1,
        min_critic_updates=1,
        min_holdout_episodes=1,
        min_stability_windows=1,
        batch_size=1,
        learning_starts=1,
        updates_per_step=1.0,
        max_transitions=2,
        max_episodes=2,
        output_dir=tmp_path,
    )
    result = producer.run()
    assert result["holdout_transition_count"] == 1
    assert len(replay.rows) == 1
    record = producer._holdout_records[0]
    assert record["episode_id"] == "episode-1"
    assert record["mission_id"] == "mission-1"
    assert record["episode_transition_index"] == 0
    assert record["holdout_record_index"] == 0
    assert record["terminal_reason"] == "success"


def test_default_gate_excludes_holdout_unavailable_placeholders_and_persists_reason(tmp_path):
    producer = _default_gate_producer(tmp_path)

    unavailable = producer._observe_gate(force=True)
    assert unavailable["state"] == "PENDING"
    assert unavailable["reason"] == "holdout_not_ready"
    assert unavailable["gate_state"] == "PENDING"
    assert unavailable["gate_reason"] == "holdout_not_ready"
    assert unavailable["maturity_ready"] is False
    assert unavailable["replay_transitions"] == 95
    assert unavailable["completed_episodes"] == 5
    assert unavailable["critic_updates"] == 0
    assert unavailable["holdout_episodes"] == 0
    assert unavailable["stability_windows"] == 0
    assert unavailable["hard_divergence_reason"] == ""

    producer._holdout_records = [
        {
            "episode_id": "episode-3",
            "mission_id": "mission-3",
            "episode_transition_index": index,
            "holdout_record_index": index,
            "depth": np.zeros((1, 1, 1), dtype=np.float32),
            "vector": np.zeros(1, dtype=np.float32),
            "action_mask": np.ones(1, dtype=np.bool_),
            "action": 0,
            "reward": 1.0,
            "next_depth": np.ones((1, 1, 1), dtype=np.float32),
            "next_vector": np.ones(1, dtype=np.float32),
            "next_action_mask": np.ones(1, dtype=np.bool_),
            "done": True,
            "behavior_source": int(BehaviorSource.BC_CALIBRATION),
            "terminal_reason": "success",
        }
        for index in range(2)
    ]
    producer._holdout_completed_ids = {"episode-3", "episode-4"}
    finite = producer._observe_gate(force=True)

    assert finite["state"] == "PENDING"
    assert finite["reason"] == "insufficient_data"
    assert finite["gate_state"] == "PENDING"
    assert finite["gate_reason"] == "insufficient_data"
    assert finite["maturity_ready"] is False
    assert finite["replay_transitions"] == 95
    assert finite["completed_episodes"] == 5
    assert finite["critic_updates"] == 0
    assert finite["holdout_episodes"] == 2
    assert finite["stability_windows"] == 1
    assert finite["hard_divergence_reason"] == ""
    assert producer.checkpoint_snapshot(finite)["gate_history"][-1] == finite


def test_resume_progress_does_not_replay_completed_missions(tmp_path):
    producer, replay, _, _ = _producer(
        tmp_path, gate_states=["PENDING"], max_transitions=2, max_episodes=1
    )
    result = producer.run()
    assert result["next_mission_index"] == 1
    resumed, _, _, _ = _producer(
        tmp_path, gate_states=["PENDING"], max_transitions=2, max_episodes=1
    )
    resumed.restore_progress(result["progress"])
    assert resumed.progress["next_mission_index"] == 1


def test_resume_with_same_replay_continues_after_committed_boundary(tmp_path):
    replay = _Replay()
    learner = _Learner()
    first = CalibrationReplayProducer(
        missions=[_mission(0), _mission(1), _mission(2)],
        pool=_Pool(terminal_after=1),
        replay=replay,
        learner=learner,
        train_episode_ids={"episode-0", "episode-1", "episode-2"},
        holdout_episode_ids=set(),
        action_selector=lambda **_: 1,
        gate_evaluator=lambda **_: {"state": "PENDING"},
        calibration_window_interval_steps=1,
        min_replay_transitions=1,
        min_completed_episodes=1,
        min_critic_updates=1,
        min_holdout_episodes=1,
        min_stability_windows=1,
        batch_size=1,
        learning_starts=1,
        updates_per_step=1.0,
        max_transitions=20,
        max_episodes=1,
        mission_source_identity={"sha256": "s" * 64},
        output_dir=tmp_path,
    )
    first_result = first.run()
    assert first_result["next_mission_index"] == 1
    assert replay.size == 1

    resumed = CalibrationReplayProducer(
        missions=[_mission(0), _mission(1), _mission(2)],
        pool=_Pool(terminal_after=1),
        replay=replay,
        learner=learner,
        train_episode_ids={"episode-0", "episode-1", "episode-2"},
        holdout_episode_ids=set(),
        action_selector=lambda **_: 1,
        gate_evaluator=lambda **_: {"state": "PENDING"},
        calibration_window_interval_steps=1,
        min_replay_transitions=1,
        min_completed_episodes=1,
        min_critic_updates=1,
        min_holdout_episodes=1,
        min_stability_windows=1,
        batch_size=1,
        learning_starts=1,
        updates_per_step=1.0,
        max_transitions=20,
        max_episodes=20,
        mission_source_identity={"sha256": "s" * 64},
        output_dir=tmp_path,
    )
    resumed.restore_progress(first_result["progress"])
    resumed_result = resumed.run()
    assert resumed_result["next_mission_index"] == 3
    assert replay.size == 3


def test_two_worker_mission_cursor_restores_the_first_uncommitted_mission(tmp_path):
    """A checkpoint cursor preserves ordered ownership across two workers."""

    replay = _Replay()
    learner = _Learner()
    first = CalibrationReplayProducer(
        missions=[_mission(0), _mission(1), _mission(2)],
        pool=_TwoWorkerPool(),
        replay=replay,
        learner=learner,
        train_episode_ids={"episode-0", "episode-1", "episode-2"},
        holdout_episode_ids=set(),
        action_selector=lambda **_: 1,
        gate_evaluator=lambda **_: {"state": "PENDING"},
        calibration_window_interval_steps=1,
        min_replay_transitions=1,
        min_completed_episodes=1,
        min_critic_updates=1,
        min_holdout_episodes=1,
        min_stability_windows=1,
        batch_size=1,
        learning_starts=1,
        updates_per_step=1.0,
        max_transitions=20,
        max_episodes=2,
        mission_source_identity={"sha256": "s" * 64},
        expected_runtime_ids={0: "runtime-0", 1: "runtime-1"},
        output_dir=tmp_path,
    )
    first_result = first.run()
    assert first_result["completed_episode_count"] == 2
    assert first_result["next_mission_index"] == 2
    assert first_result["progress"]["completed_mission_ids"] == [
        "mission-0",
        "mission-1",
    ]

    resumed = CalibrationReplayProducer(
        missions=[_mission(0), _mission(1), _mission(2)],
        pool=_TwoWorkerPool(),
        replay=replay,
        learner=learner,
        train_episode_ids={"episode-0", "episode-1", "episode-2"},
        holdout_episode_ids=set(),
        action_selector=lambda **_: 1,
        gate_evaluator=lambda **_: {"state": "PENDING"},
        calibration_window_interval_steps=1,
        min_replay_transitions=1,
        min_completed_episodes=1,
        min_critic_updates=1,
        min_holdout_episodes=1,
        min_stability_windows=1,
        batch_size=1,
        learning_starts=1,
        updates_per_step=1.0,
        max_transitions=20,
        max_episodes=3,
        mission_source_identity={"sha256": "s" * 64},
        expected_runtime_ids={0: "runtime-0", 1: "runtime-1"},
        output_dir=tmp_path,
    )
    resumed.restore_progress(first_result["progress"])
    resumed_result = resumed.run()
    assert resumed_result["next_mission_index"] == 3
    assert resumed_result["progress"]["completed_mission_ids"] == [
        "mission-0",
        "mission-1",
        "mission-2",
    ]
    assert replay.size == 3


def test_policy_vectors_are_normalized_before_replay_commit(tmp_path):
    class _Normalizer:
        def transform_continuous(self, values):
            return (np.asarray(values, dtype=np.float32) - 1.0) / 2.0

    producer, replay, _, _ = _producer(tmp_path, pool_kwargs={"terminal_after": 1})
    producer.normalizer = _Normalizer()
    producer.run()
    raw = np.asarray(
        [
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            40.0,
            0.0,
            40.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            40.0,
            1.0,
            0.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(replay.rows[0]["vector"][:22], (raw - 1.0) / 2.0)


def test_multi_worker_runtime_failure_does_not_commit_other_worker_episode(tmp_path):
    class _TwoWorkerPool(_Pool):
        def __init__(self):
            super().__init__(terminal_after=1)
            self.worker_ids = (0, 1)

        def ready(self, **kwargs):
            return {0: {"ready": True}, 1: {"ready": True}}

        def reset(self, payload):
            return {
                worker_id: {
                    "observation": _observation(0, episode_id="episode-{}".format(worker_id)),
                    "action_mask": np.ones(105, dtype=np.bool_),
                    "action_mask_info": {},
                }
                for worker_id in payload
            }

        def step(self, actions):
            return {
                0: {
                    "observation": _observation(1),
                    "action_mask": np.ones(105, dtype=np.bool_),
                    "action_mask_info": {},
                    "reward": 30.0,
                    "done": True,
                    "info": {
                        "done_reason": "success",
                        "reliable_v4": True,
                        "reliable_v4_runtime_instance_id": "runtime-0",
                        **exact_endpoint_metadata(),
                    },
                },
                1: {
                    "observation": {"invalid": True},
                    "action_mask": np.ones(105, dtype=np.bool_),
                    "action_mask_info": {},
                    "reward": 30.0,
                    "done": True,
                    "info": {
                        "done_reason": "success",
                        "reliable_v4": True,
                        "reliable_v4_runtime_instance_id": "runtime-1",
                        **exact_endpoint_metadata(),
                    },
                },
            }

    replay = _Replay()
    producer = CalibrationReplayProducer(
        missions=[_mission(0), _mission(1)],
        pool=_TwoWorkerPool(),
        replay=replay,
        learner=_Learner(),
        train_episode_ids={"episode-0", "episode-1"},
        action_selector=lambda **_: 1,
        gate_evaluator=lambda **_: {"state": "PENDING"},
        calibration_window_interval_steps=1,
        min_replay_transitions=1,
        min_completed_episodes=1,
        min_critic_updates=1,
        min_holdout_episodes=1,
        min_stability_windows=1,
        batch_size=1,
        learning_starts=1,
        updates_per_step=1.0,
        max_transitions=10,
        max_episodes=10,
        output_dir=tmp_path,
    )
    with pytest.raises(CalibrationRuntimeError):
        producer.run()
    assert replay.rows == []


def test_bc_calibration_action_uses_seeded_masked_categorical_temperature_one(
    monkeypatch,
):
    """Calibration must not reuse the formal deterministic evaluator action."""

    torch, _, _, _, _ = require_torch()
    import planning.evaluation.policy_evaluator as evaluator

    depth = torch.zeros((1, 1, 90, 160), dtype=torch.float32)
    vector = torch.zeros((1, 127), dtype=torch.float32)
    seen = {}

    def fake_observation_tensors(*args, **kwargs):
        seen["depth_history"] = kwargs["depth_history"]
        return depth, vector

    class FixedLogits(torch.nn.Module):
        def forward(self, unused_depth, unused_vector):
            logits = torch.full((1, 105), -100.0, dtype=torch.float32)
            logits[0, 4] = 0.0
            logits[0, 7] = float(np.log(2.0))
            return logits

    monkeypatch.setattr(evaluator, "observation_tensors", fake_observation_tensors)
    monkeypatch.setattr(
        evaluator,
        "choose_action",
        lambda *args, **kwargs: pytest.fail(
            "calibration must not call the formal argmax evaluator"
        ),
    )
    from planning.awac.calibration_runtime import bc_calibration_action

    mask = np.zeros(105, dtype=np.bool_)
    mask[[4, 7]] = True
    action_rng = np.random.RandomState(7201)
    actions = [
        bc_calibration_action(
            model=FixedLogits(),
            observation={},
            previous_action=-1,
            action_mask=mask,
            normalizer=object(),
            torch=torch,
            device=torch.device("cpu"),
            depth_history=[np.zeros((90, 160), dtype=np.float32)],
            rng=action_rng,
        )
        for _ in range(12)
    ]

    expected_rng = np.random.RandomState(7201)
    expected = [
        int(expected_rng.choice(np.asarray([4, 7]), p=np.asarray([1.0 / 3.0, 2.0 / 3.0])))
        for _ in range(12)
    ]
    assert actions == expected
    assert set(actions) == {4, 7}
    assert len(seen["depth_history"]) == 1


def test_new_calibration_policy_context_matches_behavior_target_and_return():
    """A new calibration cannot claim target-value certification cross-policy."""

    from planning.awac.calibration import assess_value_policy_alignment
    from planning.awac.calibration_runtime import calibration_behavior_policy_identity
    from planning.awac.trainer import _calibration_value_policy_context

    policy = calibration_behavior_policy_identity(
        checkpoint_sha256="a" * 64,
        rng_seed=7203,
    )
    context = _calibration_value_policy_context(
        bc_checkpoint_sha256="a" * 64,
        behavior_policy=policy,
    )

    alignment = assess_value_policy_alignment(**context)
    assert alignment["status"] == "MATCH"
    assert context["behavior_policy"] == policy
    assert context["target_policy"] == policy
    assert context["return_policy"] == policy


def test_calibration_episode_worker_rng_is_reproducible_and_persisted(tmp_path):
    """Parallel calibration must not reuse one action stream across workers."""

    from planning.awac.calibration_runtime import (
        calibration_behavior_policy_identity,
        calibration_episode_rng,
    )

    first_rng, first_identity = calibration_episode_rng(
        base_seed=7204,
        worker_id=0,
        episode_id="episode-0",
        mission_id="mission-0",
    )
    repeated_rng, repeated_identity = calibration_episode_rng(
        base_seed=7204,
        worker_id=0,
        episode_id="episode-0",
        mission_id="mission-0",
    )
    _, other_worker_identity = calibration_episode_rng(
        base_seed=7204,
        worker_id=1,
        episode_id="episode-0",
        mission_id="mission-0",
    )
    assert first_identity == repeated_identity
    assert first_identity["derived_seed"] != other_worker_identity["derived_seed"]
    assert first_identity["rng_scope"] == "per_episode_per_worker"
    assert first_rng.randint(0, 2**31 - 1) == repeated_rng.randint(0, 2**31 - 1)

    policy = calibration_behavior_policy_identity(
        checkpoint_sha256="a" * 64,
        rng_seed=7204,
    )
    producer = CalibrationReplayProducer(
        missions=[_mission(0), _mission(1)],
        pool=_TwoWorkerPool(),
        replay=_Replay(),
        learner=_Learner(),
        train_episode_ids={"episode-0", "episode-1"},
        holdout_episode_ids=set(),
        action_selector=lambda **_: 1,
        gate_evaluator=lambda **_: {"state": "PENDING"},
        calibration_window_interval_steps=1,
        batch_size=1,
        learning_starts=1,
        updates_per_step=1.0,
        max_transitions=10,
        max_episodes=2,
        behavior_policy=policy,
        output_dir=tmp_path,
    )
    producer._assign_missions()
    assignments = producer.progress["episode_rng_assignments"]
    assert set(assignments) == {"mission-0", "mission-1"}
    assert assignments["mission-0"]["worker_id"] == 0
    assert assignments["mission-1"]["worker_id"] == 1


def test_build_runtime_pool_hands_requested_worker_count_to_pool(tmp_path):
    worker_file = tmp_path / "workers.json"
    worker_file.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "worker_id": index,
                        "ros_master_uri": "http://127.0.0.1:{}".format(11621 + index),
                        "ros_home": str(tmp_path / "ros-{}".format(index)),
                        "runtime_instance_id": "runtime-{}".format(index),
                    }
                    for index in range(3)
                ]
            }
        ),
        encoding="utf-8",
    )
    seen = {}

    def fake_pool(specs, **kwargs):
        seen["count"] = len(specs)
        seen["workers"] = tuple(spec.worker_id for spec in specs)
        return object()

    from planning.awac.trainer import build_runtime_pool

    build_runtime_pool(worker_file, worker_count=2, pool_factory=fake_pool)
    assert seen == {"count": 2, "workers": (0, 1)}


def test_checkpoint_snapshot_restores_raw_holdout_records_and_rng_sequences(
    tmp_path,
):
    """A producer resume must continue every real RNG and gate input exactly."""

    torch, _, _, _, _ = require_torch()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    try:
        random.seed(7101)
        np.random.seed(7102)
        torch.manual_seed(7103)
        producer = _default_gate_producer(tmp_path)
        producer.torch = torch
        producer.rng = np.random.RandomState(7104)
        mask = np.ones(105, dtype=np.bool_)
        producer._holdout_records = [
            {
                    "episode_id": "episode-3",
                    "mission_id": "mission-3",
                    "episode_transition_index": 0,
                    "holdout_record_index": 0,
                "depth": np.zeros((1, 90, 160), dtype=np.float32),
                "vector": np.zeros(127, dtype=np.float32),
                "action_mask": mask,
                "action": 1,
                "reward": 30.0,
                "next_depth": np.ones((1, 90, 160), dtype=np.float32),
                "next_vector": np.ones(127, dtype=np.float32),
                "next_action_mask": mask,
                "done": True,
                "behavior_source": int(BehaviorSource.BC_CALIBRATION),
                "terminal_reason": "success",
            }
        ]
        producer._holdout_completed_ids = {"episode-3"}
        producer._gate_history = [{"state": "PENDING", "window": {"finite": True}}]
        producer._completed_mission_ids = {
            mission.mission_id for mission in producer.missions
        }
        producer._next_unassigned_index = len(producer.missions)

        snapshot = producer.checkpoint_snapshot({"state": "PENDING"})
        resume_state = snapshot["exact_resume_state"]
        assert resume_state["required_state_count"] == 23
        assert resume_state["raw_holdout_records"]

        expected = {
            "python": [random.random() for _ in range(4)],
            "numpy": np.random.random_sample(4),
            "torch": torch.rand(4),
            "producer": producer.rng.randint(0, 100000, size=4),
        }

        random.seed(1)
        np.random.seed(2)
        torch.manual_seed(3)
        resumed = _default_gate_producer(tmp_path)
        resumed.torch = torch
        resumed.restore_progress(
            snapshot["progress"],
            exact_resume_state=resume_state,
        )

        assert resumed.checkpoint_snapshot()["exact_resume_state"][
            "raw_holdout_records_sha256"
        ] == resume_state["raw_holdout_records_sha256"]
        assert resumed.progress["gate_history"] == snapshot["progress"]["gate_history"]
        assert resumed.controller.certification == producer.controller.certification
        assert [random.random() for _ in range(4)] == expected["python"]
        np.testing.assert_array_equal(
            np.random.random_sample(4), expected["numpy"]
        )
        assert torch.equal(torch.rand(4), expected["torch"])
        np.testing.assert_array_equal(
            resumed.rng.randint(0, 100000, size=4), expected["producer"]
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
