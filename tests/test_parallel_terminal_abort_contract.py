"""RED contract for legal terminal aborts returned by parallel environment workers."""

from __future__ import annotations

import numpy as np
import pytest

from planning.contracts.primitive_execution import (
    PRIMITIVE_EXECUTION_RESULT_COMPLETED,
    PRIMITIVE_EXECUTION_RESULT_PROTOCOL_ERROR,
    PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT,
    classify_primitive_execution_result,
)
from planning.runtime.parallel_env import (
    EnvWorkerSpec,
    ParallelEnvPool,
    ParallelEnvWorkerError,
)
from planning.runtime.unity_env import PrimitiveExecutionAbortedError


pytestmark = pytest.mark.unit


FRAME_COUNT = 25
ACTION_ID = 98
EXECUTION_ID = 918273645
COMMAND_IDS = list(range(70_000, 70_000 + FRAME_COUNT))
FIRST_APPLIED_STATE_ID = 1_200
FAILED_STATE_ID = FIRST_APPLIED_STATE_ID + 10


def _initial_observation():
    return {
        "depth": np.zeros((2, 2), dtype=np.float32),
        "state": {"state_id": FIRST_APPLIED_STATE_ID - 1, "marker": "before"},
        "safety": {"collided": False},
    }


def _failed_collision_observation():
    return {
        "depth": np.full((2, 2), 7.0, dtype=np.float32),
        "state": {"state_id": FAILED_STATE_ID, "marker": "failed_collision"},
        "safety": {"collided": True},
    }


def _partial_abort_frames():
    frames = [
        {
            "execution_id": EXECUTION_ID,
            "frame_index": frame_index,
            "command_id": COMMAND_IDS[frame_index],
            "applied_state_id": FIRST_APPLIED_STATE_ID + frame_index,
            "sim_time_ns": 3_000_000_000 + frame_index * 20_000_000,
            "execution_status": 1,
            "collision": False,
        }
        for frame_index in range(10)
    ]
    frames.append(
        {
            "execution_id": EXECUTION_ID,
            "frame_index": -1,
            "command_id": -1,
            "applied_state_id": FAILED_STATE_ID,
            "sim_time_ns": 3_200_000_000,
            "execution_status": 3,
            "collision": True,
            "velocity": [0.0, 0.0, 0.0],
        }
    )
    return frames


def _execution_error(fixture_kind: str):
    frames = _partial_abort_frames()
    timed_out = False
    receipt_extra = {}
    if fixture_kind == "timeout_without_terminal_state":
        frames = frames[:-1]
        timed_out = True
    elif fixture_kind == "failed_without_terminal_evidence":
        frames[-1]["collision"] = False
    elif fixture_kind == "corrupt_prefix_with_collision":
        del frames[4]
    elif fixture_kind == "reordered_prefix_with_collision":
        frames[4], frames[5] = frames[5], frames[4]
    elif fixture_kind == "duplicate_prefix_frame_with_collision":
        frames[5] = dict(frames[4])
    elif fixture_kind == "wrong_execution_id_with_collision":
        frames[4]["execution_id"] = EXECUTION_ID + 1
    elif fixture_kind == "wrong_command_id_with_collision":
        frames[4]["command_id"] = COMMAND_IDS[4] + 1_000
    elif fixture_kind == "discontinuous_state_id_with_collision":
        frames[6]["applied_state_id"] += 2
    elif fixture_kind == "impossible_frame_index_with_collision":
        frames[5]["frame_index"] = 99
    elif fixture_kind == "fake_endpoint_with_collision":
        receipt_extra["endpoint_state_id"] = FIRST_APPLIED_STATE_ID + 24
    elif fixture_kind == "bad_failed_state_relation_with_collision":
        frames[-1]["applied_state_id"] += 9
    elif fixture_kind != "valid_collision_abort":
        raise ValueError("unknown terminal-abort fixture kind: {}".format(fixture_kind))

    result = classify_primitive_execution_result(
        {
            "execution_id": EXECUTION_ID,
            "requested_frame_count": FRAME_COUNT,
            "received_frames": frames,
            **receipt_extra,
        },
        expected_frame_count=FRAME_COUNT,
        expected_command_ids=COMMAND_IDS,
        timed_out=timed_out,
    )
    expected_kind = (
        PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT
        if fixture_kind == "valid_collision_abort"
        else PRIMITIVE_EXECUTION_RESULT_PROTOCOL_ERROR
    )
    assert result["kind"] == expected_kind
    error = PrimitiveExecutionAbortedError(
        execution_id=EXECUTION_ID,
        frame_count=FRAME_COUNT,
        frames=frames,
        execution_result=result,
    )
    error.previous_depth_seq = 41
    return error


def _completed_execution_result():
    frames = [
        {
            "execution_id": EXECUTION_ID,
            "frame_index": frame_index,
            "command_id": COMMAND_IDS[frame_index],
            "applied_state_id": FIRST_APPLIED_STATE_ID + frame_index,
            "sim_time_ns": 3_000_000_000 + frame_index * 20_000_000,
            "execution_status": 2 if frame_index == FRAME_COUNT - 1 else 1,
            "collision": False,
        }
        for frame_index in range(FRAME_COUNT)
    ]
    result = classify_primitive_execution_result(
        {
            "execution_id": EXECUTION_ID,
            "requested_frame_count": FRAME_COUNT,
            "applied_frames": frames,
            "endpoint_state_id": FIRST_APPLIED_STATE_ID + FRAME_COUNT - 1,
        },
        expected_frame_count=FRAME_COUNT,
        expected_command_ids=COMMAND_IDS,
    )
    assert result["kind"] == PRIMITIVE_EXECUTION_RESULT_COMPLETED
    return result


class _TerminalAbortEnvironment:
    """External Unity boundary double with a classifier-valid abort fixture."""

    def __init__(self, worker_id, env_kwargs):
        self.worker_id = int(worker_id)
        self.step_primitive_call_count = 0
        self.materialize_call_count = 0
        self.stopped = False
        self.fixture_kind = str(
            env_kwargs.get("fixture_kind", "valid_collision_abort")
        )

    def reset(self, **kwargs):
        self.stopped = False
        return _initial_observation()

    def get_action_mask(self, observation, return_info=False):
        mask = np.asarray([True] * 105, dtype=np.bool_)
        info = {
            "combined_valid_count": int(mask.sum()),
            "environment_stopped": bool(self.stopped),
        }
        return (mask, info) if return_info else mask

    def step_primitive(self, action_id, **kwargs):
        self.step_primitive_call_count += 1
        assert int(action_id) == ACTION_ID
        if self.step_primitive_call_count != 1:
            raise AssertionError("worker retried the same primitive")
        if self.fixture_kind == "completed_execution":
            return _failed_collision_observation(), 1.0, False, {
                "success": False,
                "collided": False,
                "done_reason": "",
                "primitive_completed": True,
                "terminal_abort": False,
                "effective_integration_ticks": FRAME_COUNT,
                "primitive_execution_result": _completed_execution_result(),
                "step_primitive_call_count": self.step_primitive_call_count,
                "materialize_call_count": self.materialize_call_count,
            }
        raise _execution_error(self.fixture_kind)

    def materialize_terminal_abort(self, error, *, action_id, obs_before, **kwargs):
        self.materialize_call_count += 1
        assert error.execution_result["kind"] == PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT
        assert int(action_id) == ACTION_ID
        assert obs_before["state"]["marker"] == "before"
        return _failed_collision_observation(), -100.0, True, {
            "success": False,
            "collided": True,
            "dead_end": False,
            "timeout": False,
            "far": False,
            "hard_altitude_violation": False,
            "done_reason": "collision",
            "primitive_completed": False,
            "terminal_abort": True,
            "terminal_reason": "collision",
            "effective_integration_ticks": 10,
            "applied_frame_count": 10,
            "primitive_execution_result": error.execution_result,
            "selected_action": int(action_id),
            "step_primitive_call_count": self.step_primitive_call_count,
            "materialize_call_count": self.materialize_call_count,
            "terminal_state_id": FAILED_STATE_ID,
        }

    def stop(self):
        self.stopped = True
        return {"stopped": True}

    def close(self):
        self.stopped = True
        return {"closed": True}


def _terminal_abort_factory(worker_id, env_kwargs):
    return _TerminalAbortEnvironment(worker_id, env_kwargs)


def _pool(tmp_path, *, fixture_kind="valid_collision_abort"):
    return ParallelEnvPool(
        [
            EnvWorkerSpec(
                worker_id=0,
                ros_master_uri="http://127.0.0.1:11999",
                ros_home=str(tmp_path / "ros_home_00"),
                env_kwargs={"fixture_kind": fixture_kind},
            )
        ],
        env_factory=_terminal_abort_factory,
        startup_timeout_s=5.0,
        request_timeout_s=2.0,
    )


def test_parallel_worker_matches_formal_collision_terminal_abort_semantics(
    tmp_path,
):
    """A legal abort has the same failed-state semantics as formal evaluation."""

    pool = _pool(tmp_path)
    try:
        pool.reset({0: {}})

        transition = pool.step({0: ACTION_ID})[0]

        assert transition["done"] is True
        assert transition["reward"] == pytest.approx(-100.0)
        assert transition["observation"]["state"]["marker"] == "failed_collision"
        assert transition["observation"]["state"]["state_id"] == FAILED_STATE_ID
        assert transition["observation"]["safety"]["collided"] is True
        assert transition["info"]["collided"] is True
        assert transition["info"]["done_reason"] == "collision"
        assert transition["info"]["primitive_completed"] is False
        assert transition["info"]["terminal_abort"] is True
        assert transition["info"]["terminal_reason"] == "collision"
        assert transition["info"]["effective_integration_ticks"] == 10
        assert transition["info"]["selected_action"] == ACTION_ID
        assert transition["info"]["terminal_state_id"] == FAILED_STATE_ID
        assert transition["info"]["step_primitive_call_count"] == 1
        assert transition["info"]["materialize_call_count"] == 1
        assert transition["info"]["primitive_execution_result"]["kind"] == (
            PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT
        )
        assert transition["action_mask_info"]["environment_stopped"] is True
    finally:
        pool.close(timeout_s=0.25)


@pytest.mark.parametrize(
    "fixture_kind",
    (
        "timeout_without_terminal_state",
        "failed_without_terminal_evidence",
        "corrupt_prefix_with_collision",
        "reordered_prefix_with_collision",
        "duplicate_prefix_frame_with_collision",
        "wrong_execution_id_with_collision",
        "wrong_command_id_with_collision",
        "discontinuous_state_id_with_collision",
        "impossible_frame_index_with_collision",
        "fake_endpoint_with_collision",
        "bad_failed_state_relation_with_collision",
    ),
)
def test_parallel_worker_keeps_nonterminal_or_corrupt_partial_receipts_hard_errors(
    tmp_path,
    fixture_kind,
):
    """Only classifier-valid terminal aborts may cross the worker seam as steps."""

    pool = _pool(tmp_path, fixture_kind=fixture_kind)
    try:
        pool.reset({0: {}})

        with pytest.raises(ParallelEnvWorkerError) as caught:
            pool.step({0: ACTION_ID})

        assert caught.value.error_type == "PrimitiveExecutionAbortedError"
        assert "attempted materialization" not in caught.value.remote_traceback
    finally:
        pool.close(timeout_s=0.25)


def test_parallel_worker_keeps_complete_25_frame_execution_out_of_terminal_abort_path(
    tmp_path,
):
    """A completed receipt remains a nonterminal normal transition."""

    pool = _pool(tmp_path, fixture_kind="completed_execution")
    try:
        pool.reset({0: {}})

        transition = pool.step({0: ACTION_ID})[0]

        assert transition["done"] is False
        assert transition["info"]["primitive_completed"] is True
        assert transition["info"]["terminal_abort"] is False
        assert transition["info"]["effective_integration_ticks"] == FRAME_COUNT
        assert transition["info"]["materialize_call_count"] == 0
        assert transition["info"]["primitive_execution_result"]["kind"] == (
            PRIMITIVE_EXECUTION_RESULT_COMPLETED
        )
    finally:
        pool.close(timeout_s=0.25)
