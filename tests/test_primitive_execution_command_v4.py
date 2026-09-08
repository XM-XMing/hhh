from planning.protocol.primitive_execution_command_v4 import (
    FRAME_COUNT,
    PrimitiveExecutionCommand,
    PrimitiveExecutionCommandSchemaError,
    build_command_wire,
)


def _frames():
    return [
        {
            "frame_index": index,
            "command_id": 1000 + index,
            "action": [0.1, 0.2, 0.3, 0.4],
        }
        for index in range(FRAME_COUNT)
    ]


def test_command_wire_is_immutable_and_deterministic():
    first, first_wire = build_command_wire(
        runtime_instance_id="worker-00", execution_id=44000000000001, frames=_frames()
    )
    for _ in range(100):
        second, second_wire = build_command_wire(
            runtime_instance_id="worker-00", execution_id=44000000000001, frames=_frames()
        )
        assert second == first
        assert second_wire == first_wire
    assert first_wire == first.canonical_bytes()


def test_duplicate_identity_can_be_reparsed_without_reencoding():
    command, wire = build_command_wire(
        runtime_instance_id="worker-00", execution_id=44000000000001, frames=_frames()
    )
    reparsed = PrimitiveExecutionCommand.from_mapping(command.canonical_payload())
    assert reparsed == command
    assert reparsed.canonical_bytes() == wire


def test_frame_order_and_hash_mismatch_fail_closed():
    frames = _frames()
    frames[10], frames[11] = frames[11], frames[10]
    try:
        PrimitiveExecutionCommand.from_mapping(
            {
                "schema_version": 4,
                "message_type": "PrimitiveExecutionCommand",
                "runtime_instance_id": "worker-00",
                "execution_id": 44000000000001,
                "command_sequence_hash": "00" * 32,
                "frames": frames,
            }
        )
    except PrimitiveExecutionCommandSchemaError:
        return
    raise AssertionError("invalid command must be rejected")
