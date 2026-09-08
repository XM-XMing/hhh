"""RED specification for completed, terminal-abort, and protocol-error results."""

from __future__ import annotations

import copy

import pytest

from planning.contracts.primitive_execution import (
    PrimitiveExecutionContractError,
    validate_primitive_execution,
)


FRAME_COUNT = 25
EXECUTION_ID = 778899
COMMAND_IDS = list(range(9000, 9000 + FRAME_COUNT))
FIRST_STATE_ID = 342


def _applied_frame(index: int):
    return {
        "execution_id": EXECUTION_ID,
        "frame_index": index,
        "command_id": COMMAND_IDS[index],
        "applied_state_id": FIRST_STATE_ID + index,
        "sim_time_ns": 1_000_000_000 + index * 20_000_000,
        "execution_status": 2 if index == FRAME_COUNT - 1 else 1,
        "collision": False,
        "velocity": [3.0, 0.0, 0.0],
    }


def _completed_receipt():
    return {
        "execution_id": EXECUTION_ID,
        "requested_frame_count": FRAME_COUNT,
        "applied_frames": [_applied_frame(index) for index in range(FRAME_COUNT)],
        "endpoint_state_id": FIRST_STATE_ID + FRAME_COUNT - 1,
    }


def _collision_abort_receipt(*, collision: bool = True):
    frames = [_applied_frame(index) for index in range(14)]
    frames.append({
        "execution_id": EXECUTION_ID,
        "frame_index": -1,
        "command_id": -1,
        "applied_state_id": FIRST_STATE_ID + 14,
        "sim_time_ns": 1_000_000_000 + 14 * 20_000_000,
        "execution_status": 3,
        "collision": collision,
        "altitude_violation": False,
        "position": [53.9201507568, -10.4713783264, 1.3939377069],
        "velocity": [0.0, 0.0, 0.0],
    })
    return {
        "execution_id": EXECUTION_ID,
        "requested_frame_count": FRAME_COUNT,
        "received_frames": frames,
    }


def _classify(receipt, *, timed_out: bool = False):
    # This is the production public seam specified by this RED suite.  Using
    # getattr keeps RED as an assertion failure until the API is implemented,
    # rather than a test-collection import error.
    import planning.contracts.primitive_execution as contract

    classifier = getattr(contract, "classify_primitive_execution_result", None)
    assert callable(classifier), (
        "primitive_execution_contract must expose "
        "classify_primitive_execution_result()"
    )
    return classifier(
        receipt,
        expected_frame_count=FRAME_COUNT,
        expected_command_ids=COMMAND_IDS,
        timed_out=timed_out,
    )


@pytest.mark.unit
def test_complete_25_frame_execution_is_completed_with_k_plus_24_endpoint():
    result = _classify(_completed_receipt())

    assert result["kind"] == "COMPLETED"
    assert result["receipt"]["effective_integration_ticks"] == FRAME_COUNT
    assert result["receipt"]["first_applied_state_id"] == FIRST_STATE_ID
    assert result["receipt"]["endpoint_state_id"] == FIRST_STATE_ID + 24
    assert result["terminal_state"] is None


@pytest.mark.unit
def test_collision_failed_acknowledgement_is_terminal_abort_with_real_failure_state():
    abort = _collision_abort_receipt(collision=True)
    result = _classify(abort)

    assert result["kind"] == "TERMINAL_ABORT"
    assert result["terminal_reason"] == "collision"
    assert result["retry_allowed"] is False
    assert result["receipt"] is None
    assert result["terminal_state"]["applied_state_id"] == FIRST_STATE_ID + 14
    assert result["terminal_state"]["execution_status"] == 3
    assert result["terminal_state"]["collision"] is True
    assert [frame["frame_index"] for frame in result["partial_receipt"]["received_frames"]] == list(range(14)) + [-1]
    assert "endpoint_state_id" not in result["partial_receipt"]
    assert result["partial_receipt"]["received_frames"][-1]["velocity"] == [0.0, 0.0, 0.0]


@pytest.mark.unit
def test_xmstate_collided_field_is_a_collision_terminal_abort():
    abort = _collision_abort_receipt(collision=True)
    failed_state = abort["received_frames"][-1]
    failed_state["collided"] = failed_state.pop("collision")

    result = _classify(abort)

    assert result["kind"] == "TERMINAL_ABORT"
    assert result["terminal_state"]["collided"] is True


@pytest.mark.unit
def test_failed_acknowledgement_without_terminal_state_is_hard_protocol_error():
    result = _classify(_collision_abort_receipt(collision=False))

    assert result["kind"] == "PROTOCOL_ERROR"
    assert result["hard_failure"] is True
    assert result["terminal_state"] is None


@pytest.mark.unit
def test_timeout_after_partial_execution_is_hard_protocol_error():
    partial = _collision_abort_receipt(collision=True)
    partial["received_frames"] = partial["received_frames"][:-1]
    result = _classify(partial, timed_out=True)

    assert result["kind"] == "PROTOCOL_ERROR"
    assert result["hard_failure"] is True


@pytest.mark.unit
def test_collision_abort_rejects_forged_frames_after_failed_acknowledgement():
    forged = _collision_abort_receipt(collision=True)
    forged["received_frames"].extend(
        _applied_frame(index) for index in range(14, FRAME_COUNT)
    )
    result = _classify(forged)

    assert result["kind"] == "PROTOCOL_ERROR"
    assert result["hard_failure"] is True


@pytest.mark.unit
def test_collision_abort_explicitly_disallows_retry_of_same_primitive():
    result = _classify(_collision_abort_receipt(collision=True))

    assert result["kind"] == "TERMINAL_ABORT"
    assert result["retry_allowed"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "defect",
    ("missing_applied_frame", "reordered_applied_frames", "wrong_execution_id", "wrong_command_id", "inconsistent_failure_state_id"),
)
def test_collision_flag_does_not_mask_partial_receipt_protocol_corruption(defect):
    corrupted = _collision_abort_receipt(collision=True)
    frames = corrupted["received_frames"]
    if defect == "missing_applied_frame":
        del frames[7]
    elif defect == "reordered_applied_frames":
        frames[5], frames[6] = frames[6], frames[5]
    elif defect == "wrong_execution_id":
        frames[4]["execution_id"] = EXECUTION_ID + 1
    elif defect == "wrong_command_id":
        frames[4]["command_id"] = COMMAND_IDS[4] + 10_000
    else:
        frames[-1]["applied_state_id"] = FIRST_STATE_ID + 99

    result = _classify(corrupted)

    assert result["kind"] == "PROTOCOL_ERROR"
    assert result["hard_failure"] is True


@pytest.mark.unit
def test_terminal_abort_does_not_weaken_exact_n_completed_receipt_validator():
    partial_as_completed = _collision_abort_receipt(collision=True)
    partial_as_completed["applied_frames"] = partial_as_completed.pop("received_frames")[:-1]
    partial_as_completed["endpoint_state_id"] = FIRST_STATE_ID + 13

    with pytest.raises(PrimitiveExecutionContractError):
        validate_primitive_execution(
            partial_as_completed,
            expected_frame_count=FRAME_COUNT,
            expected_command_ids=COMMAND_IDS,
        )
