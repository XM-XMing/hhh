"""Public seam tests for the schema-v4 primitive execution model.

These tests intentionally exercise protocol mappings and public identity APIs;
they do not inspect implementation details or transport behavior.
"""

from collections import OrderedDict

import pytest

from planning.protocol.primitive_execution_schema_v4 import (
    PrimitiveExecutionProtocolError,
    ResultIdentityRegistry,
    SchemaMismatchError,
    canonical_command_sequence_hash,
    canonical_result_payload_hash,
    validate_execution_result,
)


EXECUTION_ID = 123456789
RUNTIME_INSTANCE_ID = "worker-0-runtime-a"
FIRST_STATE_ID = 1000
ENDPOINT_STATE_ID = FIRST_STATE_ID + 24
COMMAND_FRAMES = tuple(
    {
        "frame_index": index,
        "command_id": 5000 + index,
        "action": [1.0, 0.0, float(index)],
    }
    for index in range(25)
)
COMMAND_SEQUENCE_HASH = canonical_command_sequence_hash(COMMAND_FRAMES)


def _complete_result(**overrides):
    payload = {
        "schema_version": 4,
        "message_type": "PrimitiveExecutionResult",
        "runtime_instance_id": RUNTIME_INSTANCE_ID,
        "execution_id": EXECUTION_ID,
        "status": "COMPLETE",
        "requested_frame_count": 25,
        "applied_frame_count": 25,
        "first_applied_state_id": FIRST_STATE_ID,
        "endpoint_state_id": ENDPOINT_STATE_ID,
        "last_applied_frame_index": 24,
        "reason_code": "NONE",
        "command_sequence_hash": COMMAND_SEQUENCE_HASH,
        "endpoint_sim_time_ns": 20_000_000,
        "endpoint_observation_ref": {
            "schema_version": 4,
            "runtime_instance_id": RUNTIME_INSTANCE_ID,
            "episode_id": "episode-1024",
            "reset_id": "reset-1024",
            "state_id": ENDPOINT_STATE_ID,
            "depth_id": "depth-1024",
            "sim_time_ns": 20_000_000,
        },
        "result_generation": 0,
    }
    payload.update(overrides)
    return payload


def _rejected_result(**overrides):
    payload = _complete_result(
        status="REJECTED",
        applied_frame_count=0,
        first_applied_state_id=None,
        endpoint_state_id=None,
        last_applied_frame_index=-1,
        reason_code="MALFORMED_COMMAND",
        endpoint_sim_time_ns=None,
        endpoint_observation_ref=None,
    )
    payload.update(overrides)
    return payload


def _physics_complete_observation_failed_result(**overrides):
    payload = _complete_result(
        status="FAILED",
        applied_frame_count=25,
        endpoint_state_id=None,
        last_applied_frame_index=24,
        reason_code="PHYSICS_COMPLETE_OBSERVATION_FAILED",
        endpoint_sim_time_ns=None,
        endpoint_observation_ref=None,
    )
    payload.update(overrides)
    return payload


def _collision_failed_result(**overrides):
    payload = _complete_result(
        status="FAILED",
        applied_frame_count=3,
        first_applied_state_id=FIRST_STATE_ID,
        endpoint_state_id=FIRST_STATE_ID + 3,
        last_applied_frame_index=2,
        reason_code="COLLISION",
        endpoint_sim_time_ns=80_000_000,
        endpoint_observation_ref={
            "schema_version": 4,
            "runtime_instance_id": RUNTIME_INSTANCE_ID,
            "episode_id": "episode-1024",
            "reset_id": "reset-1024",
            "state_id": FIRST_STATE_ID + 3,
            "depth_id": "depth-terminal-1024",
            "sim_time_ns": 80_000_000,
        },
    )
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_complete_result_with_exact_25_frame_invariant_is_valid():
    result = validate_execution_result(
        _complete_result(),
        expected_execution_id=EXECUTION_ID,
        expected_command_sequence_hash=COMMAND_SEQUENCE_HASH,
    )

    assert result.status == "COMPLETE"
    assert result.requested_frame_count == 25
    assert result.applied_frame_count == 25
    assert result.last_applied_frame_index == 24
    assert result.endpoint_state_id == result.first_applied_state_id + 24


@pytest.mark.unit
def test_complete_result_that_claims_25_but_applied_24_is_rejected():
    with pytest.raises(PrimitiveExecutionProtocolError, match="applied_frame_count"):
        validate_execution_result(
            _complete_result(applied_frame_count=24),
            expected_execution_id=EXECUTION_ID,
            expected_command_sequence_hash=COMMAND_SEQUENCE_HASH,
        )


@pytest.mark.unit
def test_physics_complete_observation_failure_preserves_25_applied_frames():
    result = validate_execution_result(
        _physics_complete_observation_failed_result(),
        expected_execution_id=EXECUTION_ID,
        expected_command_sequence_hash=COMMAND_SEQUENCE_HASH,
    )

    assert result.status == "FAILED"
    assert result.applied_frame_count == 25
    assert result.last_applied_frame_index == 24
    assert result.endpoint_observation_ref is None


@pytest.mark.unit
def test_failed_collision_with_exact_terminal_observation_is_valid():
    result = validate_execution_result(
        _collision_failed_result(),
        expected_execution_id=EXECUTION_ID,
        expected_command_sequence_hash=COMMAND_SEQUENCE_HASH,
    )

    assert result.status == "FAILED"
    assert result.reason_code == "COLLISION"
    assert result.endpoint_observation_ref["state_id"] == FIRST_STATE_ID + 3


@pytest.mark.unit
def test_failed_collision_without_exact_terminal_observation_is_rejected():
    with pytest.raises(PrimitiveExecutionProtocolError, match="terminal endpoint_state_id"):
        validate_execution_result(
            _collision_failed_result(endpoint_observation_ref=None, endpoint_state_id=None,
                                     endpoint_sim_time_ns=None),
            expected_execution_id=EXECUTION_ID,
            expected_command_sequence_hash=COMMAND_SEQUENCE_HASH,
        )


@pytest.mark.unit
def test_rejected_result_has_zero_applied_frames_and_minus_one_last_index():
    result = validate_execution_result(
        _rejected_result(),
        expected_execution_id=EXECUTION_ID,
        expected_command_sequence_hash=COMMAND_SEQUENCE_HASH,
    )

    assert result.status == "REJECTED"
    assert result.applied_frame_count == 0
    assert result.last_applied_frame_index == -1


@pytest.mark.unit
def test_result_with_wrong_execution_id_is_rejected():
    with pytest.raises(PrimitiveExecutionProtocolError, match="execution_id"):
        validate_execution_result(
            _complete_result(execution_id=EXECUTION_ID + 1),
            expected_execution_id=EXECUTION_ID,
            expected_command_sequence_hash=COMMAND_SEQUENCE_HASH,
        )


@pytest.mark.unit
def test_result_with_command_sequence_hash_mismatch_is_rejected():
    with pytest.raises(PrimitiveExecutionProtocolError, match="command_sequence_hash"):
        validate_execution_result(
            _complete_result(command_sequence_hash="0" * 64),
            expected_execution_id=EXECUTION_ID,
            expected_command_sequence_hash=COMMAND_SEQUENCE_HASH,
        )


@pytest.mark.unit
def test_schema_version_other_than_four_fails_fast():
    with pytest.raises(SchemaMismatchError, match="schema_version"):
        validate_execution_result(
            _complete_result(schema_version=3),
            expected_execution_id=EXECUTION_ID,
            expected_command_sequence_hash=COMMAND_SEQUENCE_HASH,
        )


@pytest.mark.unit
def test_semantically_identical_result_payload_has_stable_hash():
    first = validate_execution_result(_complete_result())
    reordered = _complete_result(
        endpoint_observation_ref=OrderedDict(
            [
                ("depth_id", "depth-1024"),
                ("episode_id", "episode-1024"),
                ("reset_id", "reset-1024"),
                ("runtime_instance_id", RUNTIME_INSTANCE_ID),
                ("schema_version", 4),
                ("sim_time_ns", 20_000_000),
                ("state_id", ENDPOINT_STATE_ID),
            ]
        )
    )
    second = validate_execution_result(reordered)

    assert canonical_result_payload_hash(first) == canonical_result_payload_hash(second)


@pytest.mark.unit
def test_same_execution_with_different_result_hash_is_protocol_error():
    registry = ResultIdentityRegistry()
    first = validate_execution_result(_complete_result())
    changed = validate_execution_result(
        _complete_result(
            endpoint_sim_time_ns=20_000_001,
            endpoint_observation_ref={
                "schema_version": 4,
                "runtime_instance_id": RUNTIME_INSTANCE_ID,
                "episode_id": "episode-1024",
                "reset_id": "reset-1024",
                "state_id": ENDPOINT_STATE_ID,
                "depth_id": "depth-1024-retransmit-conflict",
                "sim_time_ns": 20_000_001,
            },
        )
    )

    assert registry.observe(first) == "FIRST"
    assert registry.observe(first) == "DUPLICATE"
    with pytest.raises(PrimitiveExecutionProtocolError, match="result_payload_hash"):
        registry.observe(changed)


@pytest.mark.unit
def test_command_sequence_hash_is_stable_for_mapping_key_order():
    equivalent_frames = tuple(
        OrderedDict(
            [
                ("action", frame["action"]),
                ("command_id", frame["command_id"]),
                ("frame_index", frame["frame_index"]),
            ]
        )
        for frame in COMMAND_FRAMES
    )

    assert canonical_command_sequence_hash(COMMAND_FRAMES) == canonical_command_sequence_hash(
        equivalent_frames
    )
