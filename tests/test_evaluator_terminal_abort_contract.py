"""RED CLI contract for continuing formal evaluation after a collision abort."""

from __future__ import annotations

import csv
import json
import sys

import numpy as np
import pytest

from planning.contracts.feature import (
    FEATURE_CONTRACT_ID,
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
)
from planning.mission.spec import TASK_CONTRACT_ID
from planning.contracts.task import task_contract_fields
from planning.runtime.unity_env import PrimitiveExecutionAbortedError


def _observation(*, collided: bool = False):
    return {
        "depth": np.zeros((2, 2), dtype=np.float32),
        "depth_m": np.ones((2, 2), dtype=np.float32),
        "depth_intrinsics": {},
        "state": {
            "position": np.asarray([53.9201507568, -10.4713783264, 1.3939377069], dtype=np.float32),
            "velocity": np.zeros(3, dtype=np.float32),
            "acceleration": np.zeros(3, dtype=np.float32),
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "yaw": 0.0,
            "z": 1.3939377069,
            "flags": 1 if collided else 0,
            "state_id": 356,
            "sim_time_ns": 7_099_999_841,
        },
        "goal": {
            "position": np.asarray([77.0, -12.0, 1.8], dtype=np.float32),
            "relative": np.asarray([23.0, -1.0, 0.4], dtype=np.float32),
            "distance_xy": 23.0,
            "dz": 0.4,
        },
        "safety": {
            "collided": collided,
            "altitude_violation": False,
            "min_clearance": 0.0 if collided else 2.0,
            "front_clearances": [0.0, 0.0, 0.0] if collided else [2.0, 2.0, 2.0],
        },
        "sensor_time": {"state_stamp_ns": 7_099_999_841, "depth_stamp_ns": 7_099_999_841, "skew_ns": 0},
        "sensor_seq": {"state": 353, "depth": 356},
        "action_mask": np.ones(105, dtype=np.bool_),
    }


def _terminal_abort_error():
    frames = [
        {
            "execution_id": 778899,
            "frame_index": index,
            "command_id": 9000 + index,
            "applied_state_id": 342 + index,
            "sim_time_ns": 6_819_999_847 + index * 20_000_000,
            "execution_status": 1,
            "collision": False,
            "velocity": [3.0, 0.0, 0.0],
        }
        for index in range(14)
    ]
    frames.append({
        "execution_id": 778899,
        "frame_index": -1,
        "command_id": -1,
        "applied_state_id": 356,
        "sim_time_ns": 7_099_999_841,
        "execution_status": 3,
        "collision": True,
        "altitude_violation": False,
        "position": [53.9201507568, -10.4713783264, 1.3939377069],
        "velocity": [0.0, 0.0, 0.0],
    })
    return PrimitiveExecutionAbortedError(
        execution_id=778899, frame_count=25, frames=frames
    )


class _FakeMpl:
    contract_sha256 = "mpl-contract"
    duration_s = 0.5
    forward_distance_m = 1.5


class _FakeEnv:
    instances = []

    def __init__(self, *args, **kwargs):
        self.config = kwargs["config"]
        self.step_calls = 0
        self.terminal_abort_calls = 0
        self.reset_calls = 0
        self.stop_calls = 0
        self._steps_this_episode = 0
        type(self).instances.append(self)

    def wait_until_ready(self, timeout):
        return None

    def reset(self, *, start, goal):
        self.reset_calls += 1
        self._steps_this_episode = 0
        return _observation()

    def get_action_mask(self, obs, return_info=False):
        mask = np.ones(105, dtype=np.bool_)
        info = {
            "combined_valid_count": int(mask.sum()),
            "depth_valid_count": int(mask.sum()),
            "depth_blocked_count": 0,
            "global_valid_count": -1,
            "height_mask": mask,
            "depth_mask": mask,
            "global_mask": None,
            "dead_end": False,
            "depth_action_min_ray_clearance_m": np.full(105, 2.0, dtype=np.float32),
        }
        return (mask, info) if return_info else mask

    def step_primitive(self, action_id, **kwargs):
        self.step_calls += 1
        self._steps_this_episode += 1
        if self.reset_calls == 1 and self._steps_this_episode == 12:
            raise _terminal_abort_error()
        if self.reset_calls == 1:
            return _observation(), 0.0, False, {
                "success": False,
                "collided": False,
                "dead_end": False,
                "timeout": False,
                "far": False,
                "hard_altitude_violation": False,
                "done_reason": "",
                "primitive_endpoint_error_body_m": 0.0,
                "next_action_mask_info": {"combined_valid_count": 105, "depth_valid_count": 105, "global_valid_count": -1},
                "primitive_execution": {"execution_id": self.step_calls, "requested_frame_count": 25, "applied_frames": [], "endpoint_state_id": self.step_calls},
            }
        info = {
            "success": True,
            "collided": False,
            "dead_end": False,
            "timeout": False,
            "far": False,
            "hard_altitude_violation": False,
            "done_reason": "success",
            "primitive_endpoint_error_body_m": 0.0,
            "next_action_mask_info": {"combined_valid_count": 105, "depth_valid_count": 105, "global_valid_count": -1},
            "primitive_execution": {
                "execution_id": 9,
                "requested_frame_count": 25,
                "applied_frames": [],
                "endpoint_state_id": 9,
            },
        }
        return _observation(), 1.0, True, info

    def materialize_terminal_abort(self, error, **kwargs):
        self.terminal_abort_calls += 1
        result = error.execution_result
        assert result is not None
        assert result["kind"] == "TERMINAL_ABORT"
        return _observation(collided=True), -100.0, True, {
            "success": False,
            "collided": True,
            "dead_end": False,
            "timeout": False,
            "far": False,
            "hard_altitude_violation": False,
            "done_reason": "collision",
            "next_action_mask_info": {
                "combined_valid_count": 105,
                "depth_valid_count": 105,
                "global_valid_count": -1,
            },
            "primitive_completed": False,
            "terminal_abort": True,
            "terminal_reason": "collision",
            "effective_integration_ticks": 14,
            "applied_frame_count": 14,
            "primitive_execution_result": result,
        }

    def stop(self):
        self.stop_calls += 1


class _FakeModel:
    def to(self, device):
        return self

    def load_state_dict(self, state):
        return None

    def eval(self):
        return self


class _FakeTorch:
    def __init__(self, checkpoint):
        self._checkpoint = checkpoint

    def device(self, value):
        return value

    def load(self, *args, **kwargs):
        return self._checkpoint


def test_cli_records_collision_terminal_abort_and_continues_to_next_mission(
    tmp_path, monkeypatch
):
    """One abort ends only its episode; it never retries or halts the holdout."""
    import planning.evaluation.policy_evaluator as evaluator

    _FakeEnv.instances = []
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    index_path = tmp_path / "index.csv"
    index_path.write_text("episode_id\n12116\n12117\n", encoding="utf-8")
    output = tmp_path / "evaluation"
    checkpoint = {
        "feature_contract_id": FEATURE_CONTRACT_ID,
        **task_contract_fields(),
        "task_contract_id": TASK_CONTRACT_ID,
        "vec_dim": POLICY_VECTOR_DIM,
        "num_actions": NUM_ACTIONS,
        "depth_history_frames": 1,
        "initial_prev_action": INITIAL_PREV_ACTION,
        "model_type": "bc_soft",
        "model_state_dict": {},
        "mpl_contract_sha256": "mpl-contract",
        "observation_contract": "reliable_exact_endpoint_snapshot",
        "observation_source": "reliable_exact_endpoint_snapshot",
    }
    rows = [
        {"episode_id": "12116", "mission_id": "abort-mission"},
        {"episode_id": "12117", "mission_id": "next-mission"},
    ]
    monkeypatch.setattr(evaluator, "require_torch", lambda: (_FakeTorch(checkpoint), object(), None, None, None))
    monkeypatch.setattr(evaluator, "VectorNormalizer", type("N", (), {"from_checkpoint": staticmethod(lambda _: object())}))
    monkeypatch.setattr(
        evaluator,
        "resolve_policy_checkpoint_normalizer",
        lambda *args, **kwargs: (object(), checkpoint_path.resolve()),
    )
    monkeypatch.setattr(evaluator, "build_model", lambda *args, **kwargs: _FakeModel())
    monkeypatch.setattr(evaluator, "validate_checkpoint_policy_input_contract", lambda _: None)
    monkeypatch.setattr(evaluator, "validate_policy_checkpoint_algorithm", lambda *args: None)
    monkeypatch.setattr(evaluator, "resolve_evaluation_runtime", lambda *args, **kwargs: ("depth", "continuous", False))
    monkeypatch.setattr(evaluator, "resolve_evaluation_depth_mask_numeric_config", lambda *args, **kwargs: ({
        "depth_mask_collision_radius_m": 0.4, "depth_mask_slack_m": 0.08,
        "depth_mask_sample_stride": 4, "depth_mask_max_patch_radius_px": 14,
    }, False))
    monkeypatch.setattr(evaluator, "checkpoint_depth_mask_numeric_config", lambda _: {})
    monkeypatch.setattr(evaluator, "file_sha256", lambda _: "sha")
    monkeypatch.setattr(evaluator, "read_csv", lambda _: rows)
    monkeypatch.setattr(evaluator, "validate_mission_rows", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator, "row_start_goal", lambda row: ([0.0, 0.0, 1.8], [40.0, 0.0, 1.8]))
    monkeypatch.setattr(evaluator, "MotionPrimitiveLibrary", _FakeMpl)
    monkeypatch.setattr(evaluator, "UnityForestEnv", _FakeEnv)
    monkeypatch.setattr(evaluator, "choose_action", lambda *args, **kwargs: (
        98, 1.0, [98, 99, 97, 96, 95], {
            "actor_logits_fingerprint": "logits", "actor_top_k": [
                {"action": 98, "logit": 3.0}, {"action": 99, "logit": 2.0},
            ], "actor_logits": [0.0] * 105,
        }
    ))
    monkeypatch.setattr(evaluator, "terminal_reward_input_fingerprint", lambda *args, **kwargs: "terminal")
    monkeypatch.setattr(sys, "argv", [
        "evaluate_policy_unity.py", "--checkpoint", str(checkpoint_path), "--index", str(index_path),
        "--out-dir", str(output), "--max-episodes", "2", "--audit-only",
        "--first-divergence-trace", "--audit-allow-runtime-override",
        "--device", "cpu",
    ])

    assert evaluator.main() == 0

    with (output / "rollout_index.csv").open(encoding="utf-8", newline="") as stream:
        rollout = list(csv.DictReader(stream))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    trace = json.loads((output / "first_divergence_trace.json").read_text(encoding="utf-8"))
    assert [row["stop_reason"] for row in rollout] == ["collision", "success"]
    assert rollout[0]["collision"] == "True"
    assert rollout[0]["success"] == "False"
    assert summary["episodes"] == 2
    assert summary["collision_count"] == 1
    assert rollout[0]["steps"] == "12"
    assert float(rollout[0]["final_x"]) == pytest.approx(53.9201507568)
    assert float(rollout[0]["final_y"]) == pytest.approx(-10.4713783264)
    assert float(rollout[0]["final_z"]) == pytest.approx(1.3939377069)
    assert _FakeEnv.instances[0].step_calls == 13
    assert _FakeEnv.instances[0].terminal_abort_calls == 1
    abort_trace = trace["episodes"][0]["primitives"][-1]
    assert abort_trace["selected_action"] == 98
    receipt = abort_trace["primitive_execution_result"]["partial_receipt"]
    assert [frame["frame_index"] for frame in receipt["received_frames"]] == list(range(14)) + [-1]
