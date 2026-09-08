"""Canonical result protocol owner and schema-v4 compatibility facade."""

from typing import Any, Mapping

from planning.protocol.constants import PROTOCOL_VERSION
from planning.protocol.msgpack import messagepack_bytes

from planning.protocol.primitive_execution_schema_v4 import (
    PrimitiveExecutionProtocolError,
    PrimitiveExecutionResult,
    ResultIdentityRegistry,
    SchemaMismatchError,
    canonical_result_payload_hash,
    validate_execution_result,
)


RECEIPT_ACK_MESSAGE_TYPE = "PrimitiveExecutionResultReceiptAck"
RECEIPT_STATUS_RECEIVED = "RECEIVED"


def build_result_ready_wire() -> bytes:
    return messagepack_bytes(
        {
            "schema_version": PROTOCOL_VERSION,
            "message_type": "PrimitiveExecutionResultReady",
        }
    )


def build_result_receipt_ack_wire(result: Mapping[str, Any]) -> bytes:
    validated = validate_execution_result(result)
    return messagepack_bytes(
        {
            "command_sequence_hash": bytes.fromhex(validated.command_sequence_hash),
            "execution_id": validated.execution_id,
            "message_type": RECEIPT_ACK_MESSAGE_TYPE,
            "receipt_status": RECEIPT_STATUS_RECEIVED,
            "result_payload_hash": bytes.fromhex(validated.result_payload_hash),
            "runtime_instance_id": validated.runtime_instance_id,
            "schema_version": PROTOCOL_VERSION,
        }
    )


def build_result_commit_wire(result: Mapping[str, Any]) -> bytes:
    return messagepack_bytes(
        {
            "schema_version": PROTOCOL_VERSION,
            "message_type": "PrimitiveExecutionResultCommit",
            "commit_status": "COMMITTED",
            "runtime_instance_id": result["runtime_instance_id"],
            "execution_id": result["execution_id"],
            "command_sequence_hash": bytes.fromhex(result["command_sequence_hash"]),
            "result_payload_hash": bytes.fromhex(result["result_payload_hash"]),
        }
    )

__all__ = [
    "PrimitiveExecutionProtocolError",
    "PrimitiveExecutionResult",
    "ResultIdentityRegistry",
    "SchemaMismatchError",
    "canonical_result_payload_hash",
    "build_result_ready_wire",
    "build_result_receipt_ack_wire",
    "build_result_commit_wire",
    "RECEIPT_ACK_MESSAGE_TYPE",
    "RECEIPT_STATUS_RECEIVED",
    "validate_execution_result",
]
