"""Pure reset-message contract helpers.

Reset requests/completions use the existing v4 mapping and field names.  The
runtime transport owns sockets; this module owns validation and canonical
construction so reset retries cannot accidentally invent a second schema.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from planning.protocol.constants import PROTOCOL_VERSION
from planning.protocol.msgpack import messagepack_bytes


RESET_REQUEST_MESSAGE_TYPE = "PrimitiveResetRequest"
RESET_COMPLETE_MESSAGE_TYPE = "PrimitiveResetComplete"
RESET_RECEIVED_ACK_MESSAGE_TYPE = "PrimitiveResetReceivedAck"


def _text(mapping: Mapping[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be a non-empty string".format(name))
    return value


def _xyz(values: Sequence[float], name: str):
    if isinstance(values, (str, bytes)) or len(values) < 3:
        raise ValueError("{} requires xyz".format(name))
    return [float(values[0]), float(values[1]), float(values[2])]


def build_reset_request(
    *,
    runtime_instance_id: str,
    episode_id: str,
    reset_id: str,
    start: Sequence[float],
    goal: Sequence[float],
) -> Dict[str, Any]:
    return {
        "schema_version": PROTOCOL_VERSION,
        "message_type": RESET_REQUEST_MESSAGE_TYPE,
        "runtime_instance_id": _text({"runtime_instance_id": runtime_instance_id}, "runtime_instance_id"),
        "episode_id": _text({"episode_id": episode_id}, "episode_id"),
        "reset_id": _text({"reset_id": reset_id}, "reset_id"),
        "start": _xyz(start, "start"),
        "goal": _xyz(goal, "goal"),
    }


def validate_reset_completion(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("reset completion must be a mapping")
    if payload.get("schema_version") != PROTOCOL_VERSION:
        raise ValueError("reset completion schema_version mismatch")
    if payload.get("message_type") != RESET_COMPLETE_MESSAGE_TYPE:
        raise ValueError("reset completion message_type mismatch")
    return {
        "schema_version": PROTOCOL_VERSION,
        "message_type": RESET_COMPLETE_MESSAGE_TYPE,
        "runtime_instance_id": _text(payload, "runtime_instance_id"),
        "episode_id": _text(payload, "episode_id"),
        "reset_id": _text(payload, "reset_id"),
    }


def build_reset_request_wire(**kwargs: Any) -> bytes:
    return messagepack_bytes(build_reset_request(**kwargs))


def build_reset_received_ack_wire(
    *, runtime_instance_id: str, episode_id: str, reset_id: str
) -> bytes:
    return messagepack_bytes(
        {
            "schema_version": PROTOCOL_VERSION,
            "message_type": RESET_RECEIVED_ACK_MESSAGE_TYPE,
            "runtime_instance_id": runtime_instance_id,
            "episode_id": episode_id,
            "reset_id": reset_id,
        }
    )


__all__ = [
    "RESET_COMPLETE_MESSAGE_TYPE",
    "RESET_RECEIVED_ACK_MESSAGE_TYPE",
    "RESET_REQUEST_MESSAGE_TYPE",
    "build_reset_request",
    "build_reset_request_wire",
    "build_reset_received_ack_wire",
    "validate_reset_completion",
]
