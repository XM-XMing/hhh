"""Validation for physics-clock-bound primitive execution receipts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


PRIMITIVE_EXECUTION_CONTRACT_ID = "physics_clock_primitive_execution"
PRIMITIVE_EXECUTION_RESULT_COMPLETED = "COMPLETED"
PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT = "TERMINAL_ABORT"
PRIMITIVE_EXECUTION_RESULT_PROTOCOL_ERROR = "PROTOCOL_ERROR"


class PrimitiveExecutionContractError(RuntimeError):
    """Raised when Unity's execution receipt violates the primitive contract."""


def _protocol_error(reason: str) -> Dict[str, Any]:
    """Return a non-recoverable result without treating partial work as complete."""
    return {
        "kind": PRIMITIVE_EXECUTION_RESULT_PROTOCOL_ERROR,
        "receipt": None,
        "partial_receipt": None,
        "terminal_state": None,
        "terminal_reason": None,
        "retry_allowed": False,
        "hard_failure": True,
        "error": str(reason),
    }


def _positive_int(value: Any, *, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise PrimitiveExecutionContractError("{} is required".format(name)) from error
    if result < 0:
        raise PrimitiveExecutionContractError("{} is required".format(name))
    return result


def _status(frame: Mapping[str, Any]) -> int:
    try:
        return int(frame.get("execution_status", -1))
    except (TypeError, ValueError):
        return -1


def classify_primitive_execution_result(
    receipt: Mapping[str, Any], *, expected_frame_count: int,
    expected_command_ids: Optional[Sequence[int]] = None,
    timed_out: bool = False,
) -> Dict[str, Any]:
    """Classify a complete receipt or a structurally valid collision abort.

    ``validate_primitive_execution`` remains the only completed-execution
    validator.  This classifier deliberately never turns a partial receipt
    into an endpoint: an early failed acknowledgement is accepted only when
    its complete acknowledgement sequence proves a collision terminal abort.
    All other failed or incomplete sequences remain hard protocol errors.
    """
    expected = int(expected_frame_count)
    if expected <= 0:
        return _protocol_error("expected_frame_count must be positive")
    if not isinstance(receipt, Mapping):
        return _protocol_error("receipt must be a mapping")

    execution_id = receipt.get("execution_id")
    if execution_id is None or execution_id == "":
        return _protocol_error("execution_id is required")
    try:
        requested = int(receipt.get("requested_frame_count", -1))
    except (TypeError, ValueError):
        return _protocol_error("requested_frame_count is required")
    if requested != expected:
        return _protocol_error(
            "frame_count mismatch: requested={} expected={}".format(requested, expected)
        )
    if bool(timed_out):
        return _protocol_error("primitive execution acknowledgement timeout")

    if "applied_frames" in receipt:
        if "received_frames" in receipt:
            return _protocol_error("receipt cannot contain completed and partial frames")
        try:
            validated = validate_primitive_execution(
                receipt,
                expected_frame_count=expected,
                expected_command_ids=expected_command_ids,
            )
        except PrimitiveExecutionContractError as error:
            return _protocol_error(str(error))
        statuses = [_status(frame) for frame in validated["applied_frames"]]
        if statuses[:-1] != [1] * (expected - 1) or statuses[-1:] != [2]:
            return _protocol_error(
                "completed execution status sequence must be applying...complete"
            )
        return {
            "kind": PRIMITIVE_EXECUTION_RESULT_COMPLETED,
            "receipt": validated,
            "partial_receipt": None,
            "terminal_state": None,
            "terminal_reason": None,
            "retry_allowed": False,
            "hard_failure": False,
        }

    if "endpoint_state_id" in receipt:
        return _protocol_error("partial receipt must not claim an endpoint_state_id")
    raw_frames = receipt.get("received_frames")
    if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)):
        return _protocol_error("partial receipt requires received_frames")
    frames = [dict(frame) for frame in raw_frames if isinstance(frame, Mapping)]
    if len(frames) != len(raw_frames):
        return _protocol_error("received_frames entries must be mappings")
    if not frames:
        return _protocol_error("partial receipt has no failed terminal state")

    failed_indices = [
        index
        for index, frame in enumerate(frames)
        if _status(frame) == 3
    ]
    if failed_indices != [len(frames) - 1]:
        return _protocol_error("partial receipt must end with exactly one failed state")
    if any(_status(frame) == 2 for frame in frames):
        return _protocol_error("partial receipt cannot contain a completed state")

    failed_state = dict(frames[-1])
    if failed_state.get("execution_id") != execution_id:
        return _protocol_error("failed state execution_id mismatch")
    try:
        failed_frame_index = int(failed_state.get("frame_index", -99))
    except (TypeError, ValueError):
        return _protocol_error("failed state frame_index must be -1")
    # ``frame_index`` uses -1 as an explicit failed-state sentinel, so it
    # deliberately sits outside the applied-frame index domain.
    if failed_frame_index != -1:
        return _protocol_error("failed state frame_index must be -1")
    if int(failed_state.get("command_id", -99)) != -1:
        return _protocol_error("failed state command_id must be -1")
    # The public audit field follows XMState's ``collided`` spelling.  The
    # concise ``collision`` alias is accepted for protocol/unit callers.
    if not bool(failed_state.get("collided", failed_state.get("collision", False))):
        return _protocol_error("failed state is not a collision terminal state")

    applied_frames = frames[:-1]
    if len(applied_frames) >= expected:
        return _protocol_error("failed state cannot follow a full execution")
    try:
        indices = [_positive_int(frame.get("frame_index"), name="frame_index") for frame in applied_frames]
        if indices != list(range(len(applied_frames))):
            raise PrimitiveExecutionContractError("partial frame indices must be ordered and contiguous")
        if any(frame.get("execution_id") != execution_id for frame in applied_frames):
            raise PrimitiveExecutionContractError("partial frame execution_id mismatch")
        command_ids = [
            _positive_int(frame.get("command_id"), name="command_id")
            for frame in applied_frames
        ]
        if len(set(command_ids)) != len(command_ids):
            raise PrimitiveExecutionContractError("partial command_id values must be unique")
        if expected_command_ids is not None:
            submitted = [int(value) for value in expected_command_ids]
            if command_ids != submitted[:len(command_ids)]:
                raise PrimitiveExecutionContractError(
                    "partial acknowledged command_id sequence does not match submitted prefix"
                )
        state_ids = [
            _positive_int(frame.get("applied_state_id"), name="applied_state_id")
            for frame in applied_frames
        ]
        if state_ids and state_ids != list(
            range(state_ids[0], state_ids[0] + len(state_ids))
        ):
            raise PrimitiveExecutionContractError("partial applied state ids must be ordered and contiguous")
        failed_state_id = _positive_int(
            failed_state.get("applied_state_id"), name="failed applied_state_id"
        )
        if state_ids and failed_state_id != state_ids[-1] + 1:
            raise PrimitiveExecutionContractError(
                "failed state must immediately follow the last applied state"
            )
    except PrimitiveExecutionContractError as error:
        return _protocol_error(str(error))

    partial_receipt = {
        "execution_id": execution_id,
        "requested_frame_count": expected,
        "received_frames": frames,
    }
    return {
        "kind": PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT,
        "receipt": None,
        "partial_receipt": partial_receipt,
        "terminal_state": failed_state,
        "terminal_reason": "collision",
        "retry_allowed": False,
        "hard_failure": False,
    }


def validate_primitive_execution(
    receipt: Mapping[str, Any], *, expected_frame_count: int,
    expected_command_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Validate and return a normalized deterministic execution receipt."""
    expected = int(expected_frame_count)
    if expected <= 0:
        raise PrimitiveExecutionContractError("expected_frame_count must be positive")
    execution_id = receipt.get("execution_id")
    if execution_id is None or execution_id == "":
        raise PrimitiveExecutionContractError("execution_id is required")
    requested = int(receipt.get("requested_frame_count", -1))
    if requested != expected:
        raise PrimitiveExecutionContractError(
            "frame_count mismatch: requested={} expected={}".format(requested, expected)
        )
    frames = [dict(frame) for frame in receipt.get("applied_frames", [])]
    if len(frames) != expected:
        raise PrimitiveExecutionContractError(
            "applied frame count mismatch: actual={} expected={}".format(
                len(frames), expected
            )
        )
    frame_indices = [int(frame.get("frame_index", -1)) for frame in frames]
    if frame_indices != list(range(expected)):
        raise PrimitiveExecutionContractError(
            "applied frame indices must be ordered and contiguous: {}".format(
                frame_indices
            )
        )
    frame_execution_ids = [frame.get("execution_id") for frame in frames]
    if any(value != execution_id for value in frame_execution_ids):
        raise PrimitiveExecutionContractError("applied frame execution_id mismatch")
    command_ids = [int(frame.get("command_id", -1)) for frame in frames]
    if any(value < 0 for value in command_ids) or len(set(command_ids)) != expected:
        raise PrimitiveExecutionContractError("command_id values must be present and unique")
    if expected_command_ids is not None:
        submitted_command_ids = [int(value) for value in expected_command_ids]
        if command_ids != submitted_command_ids:
            raise PrimitiveExecutionContractError(
                "acknowledged command_id sequence does not match submitted commands"
            )
    state_ids = [int(frame.get("applied_state_id", -1)) for frame in frames]
    if any(value < 0 for value in state_ids):
        raise PrimitiveExecutionContractError("applied_state_id is required for every frame")
    first_state_id = state_ids[0]
    expected_state_ids = list(range(first_state_id, first_state_id + expected))
    if state_ids != expected_state_ids:
        raise PrimitiveExecutionContractError(
            "applied state ids must be ordered and contiguous: {}".format(state_ids)
        )
    endpoint_state_id = int(receipt.get("endpoint_state_id", -1))
    required_endpoint = first_state_id + expected - 1
    if endpoint_state_id != required_endpoint or endpoint_state_id != state_ids[-1]:
        raise PrimitiveExecutionContractError(
            "endpoint state mismatch: actual={} expected={}".format(
                endpoint_state_id, required_endpoint
            )
        )
    normalized = dict(receipt)
    normalized["contract_id"] = PRIMITIVE_EXECUTION_CONTRACT_ID
    normalized["execution_id"] = execution_id
    normalized["requested_frame_count"] = expected
    normalized["applied_frames"] = frames
    normalized["first_applied_state_id"] = first_state_id
    normalized["endpoint_state_id"] = endpoint_state_id
    normalized["effective_integration_ticks"] = expected
    return normalized
