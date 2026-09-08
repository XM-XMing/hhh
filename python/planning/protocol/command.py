"""Canonical command protocol owner and schema-v4 compatibility facade."""

from planning.protocol.constants import PROTOCOL_VERSION
from planning.protocol.msgpack import messagepack_bytes

from planning.protocol.primitive_execution_command_v4 import (
    FRAME_COUNT,
    MESSAGE_TYPE,
    PrimitiveExecutionCommand,
    PrimitiveExecutionCommandSchemaError,
    SCHEMA_VERSION,
    build_command_wire,
)


def build_command_ready_wire(runtime_instance_id: str) -> bytes:
    """Encode the existing command READY envelope byte-for-byte."""

    return messagepack_bytes(
        {
            "message_type": "PrimitiveExecutionCommandReady",
            "runtime_instance_id": runtime_instance_id,
            "schema_version": PROTOCOL_VERSION,
        }
    )

__all__ = [
    "FRAME_COUNT",
    "MESSAGE_TYPE",
    "PrimitiveExecutionCommand",
    "PrimitiveExecutionCommandSchemaError",
    "SCHEMA_VERSION",
    "build_command_wire",
    "build_command_ready_wire",
]
