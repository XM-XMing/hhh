"""Pure v4 primitive-command identity and immutable wire model.

This module is deliberately transport-neutral.  Its public boundary is the
command mapping -> canonical bytes/hash conversion used by the Python DEALER,
bridge broker, and Unity admission seam.  Retransmission must reuse the
returned bytes rather than reconstructing a semantically equivalent mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence, Tuple

from planning.protocol.primitive_execution_schema_v4 import (
    PrimitiveExecutionProtocolError,
    canonical_v4_command_sequence_bytes,
)
from planning.protocol.msgpack import CanonicalMessagePackWriter
from planning.protocol.validation import (
    require_digest,
    require_field,
    require_non_empty_text,
    require_uint64,
)
from planning.protocol.constants import (
    PRIMITIVE_FRAME_COUNT,
    SCHEMA_VERSION as PROTOCOL_SCHEMA_VERSION,
)


SCHEMA_VERSION = PROTOCOL_SCHEMA_VERSION
MESSAGE_TYPE = "PrimitiveExecutionCommand"
RECEIPT_MESSAGE_TYPE = "PrimitiveExecutionCommandReceiptAck"
FRAME_COUNT = PRIMITIVE_FRAME_COUNT


class PrimitiveExecutionCommandSchemaError(PrimitiveExecutionProtocolError):
    """Raised when a v4 primitive command violates its wire contract."""


@dataclass(frozen=True)
class PrimitiveExecutionCommand:
    schema_version: int
    message_type: str
    runtime_instance_id: str
    execution_id: int
    command_sequence_hash: str
    frames: Tuple[Mapping[str, Any], ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PrimitiveExecutionCommand":
        if not isinstance(payload, Mapping):
            raise PrimitiveExecutionCommandSchemaError("command must be a mapping")
        schema_version = payload.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise PrimitiveExecutionCommandSchemaError(
                "schema_version mismatch: received={} expected={}".format(
                    schema_version, SCHEMA_VERSION
                )
            )
        message_type = payload.get("message_type")
        if message_type != MESSAGE_TYPE:
            raise PrimitiveExecutionCommandSchemaError("message_type mismatch")
        frames = payload.get("frames")
        if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
            raise PrimitiveExecutionCommandSchemaError("frames must be a sequence")
        if len(frames) != FRAME_COUNT:
            raise PrimitiveExecutionCommandSchemaError("frame count mismatch")
        normalized = []
        for index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise PrimitiveExecutionCommandSchemaError("frame must be a mapping")
            if frame.get("frame_index") != index:
                raise PrimitiveExecutionCommandSchemaError(
                    "frame indices must be contiguous"
                )
            action = frame.get("action")
            if isinstance(action, (str, bytes)) or not isinstance(action, Sequence):
                raise PrimitiveExecutionCommandSchemaError("frame action must be a sequence")
            for value in action:
                if not math.isfinite(float(value)):
                    raise PrimitiveExecutionCommandSchemaError(
                        "frame action rejects NaN and infinity"
                    )
            normalized.append(
                {
                    "frame_index": int(frame["frame_index"]),
                    "command_id": require_uint64(
                        require_field(frame, "command_id"),
                        name="command_id",
                        error_type=PrimitiveExecutionCommandSchemaError,
                    ),
                    "action": list(action),
                }
            )
        result = cls(
            schema_version=int(schema_version),
            message_type=message_type,
            runtime_instance_id=require_non_empty_text(
                require_field(payload, "runtime_instance_id"),
                name="runtime_instance_id",
                error_type=PrimitiveExecutionCommandSchemaError,
            ),
            execution_id=require_uint64(
                require_field(payload, "execution_id"),
                name="execution_id",
                error_type=PrimitiveExecutionCommandSchemaError,
            ),
            command_sequence_hash=require_non_empty_text(
                require_field(payload, "command_sequence_hash"),
                name="command_sequence_hash",
                error_type=PrimitiveExecutionCommandSchemaError,
            ),
            frames=tuple(normalized),
        )
        actual = result.command_sequence_hash_value()
        if actual != result.command_sequence_hash:
            raise PrimitiveExecutionCommandSchemaError(
                "command_sequence_hash mismatch"
            )
        return result

    def command_sequence_hash_value(self) -> str:
        return hashlib.sha256(
            canonical_v4_command_sequence_bytes(self.frames)
        ).hexdigest()

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "command_sequence_hash": self.command_sequence_hash,
            "execution_id": self.execution_id,
            "frames": list(self.frames),
            "message_type": self.message_type,
            "runtime_instance_id": self.runtime_instance_id,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        writer = CanonicalMessagePackWriter(
            error_type=PrimitiveExecutionCommandSchemaError
        )
        writer.map_header(6)
        writer.string("command_sequence_hash")
        writer.binary32(
            bytes.fromhex(
                require_digest(
                    self.command_sequence_hash,
                    name="command_sequence_hash",
                    error_type=PrimitiveExecutionCommandSchemaError,
                )
            )
        )
        writer.string("execution_id")
        writer.uint64(self.execution_id)
        writer.string("frames")
        sequence = canonical_v4_command_sequence_bytes(self.frames)
        writer.data.extend(sequence)
        writer.string("message_type")
        writer.string(self.message_type)
        writer.string("runtime_instance_id")
        writer.string(self.runtime_instance_id)
        writer.string("schema_version")
        writer.uint32(self.schema_version)
        return writer.bytes()


def build_command_wire(
    *, runtime_instance_id: str, execution_id: int, frames: Sequence[Mapping[str, Any]]
) -> Tuple[PrimitiveExecutionCommand, bytes]:
    sequence_hash = hashlib.sha256(
        canonical_v4_command_sequence_bytes(frames)
    ).hexdigest()
    command = PrimitiveExecutionCommand.from_mapping(
        {
            "schema_version": SCHEMA_VERSION,
            "message_type": MESSAGE_TYPE,
            "runtime_instance_id": runtime_instance_id,
            "execution_id": execution_id,
            "command_sequence_hash": sequence_hash,
            "frames": list(frames),
        }
    )
    return command, command.canonical_bytes()


__all__ = [
    "FRAME_COUNT",
    "MESSAGE_TYPE",
    "PrimitiveExecutionCommand",
    "PrimitiveExecutionCommandSchemaError",
    "SCHEMA_VERSION",
    "build_command_wire",
]
