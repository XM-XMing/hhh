"""Python application receipt-ACK wire contract."""

import json
from pathlib import Path

import msgpack
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _complete_result():
    fixture = json.loads(
        (ROOT / "tests/fixtures/primitive_execution_v4/complete.json").read_text()
    )
    result = dict(fixture["result"])
    result["result_payload_hash"] = fixture["result_payload_hash"]
    return result


@pytest.mark.unit
def test_validated_result_builds_application_receipt_ack():
    from planning.protocol.primitive_execution_result_receipt import (
        build_result_receipt_ack_wire,
    )

    result = _complete_result()
    ack = msgpack.unpackb(build_result_receipt_ack_wire(result), raw=False)

    assert ack == {
        "schema_version": 4,
        "message_type": "PrimitiveExecutionResultReceiptAck",
        "runtime_instance_id": result["runtime_instance_id"],
        "execution_id": result["execution_id"],
        "receipt_status": "RECEIVED",
        "result_payload_hash": bytes.fromhex(result["result_payload_hash"]),
        "command_sequence_hash": bytes.fromhex(result["command_sequence_hash"]),
    }


@pytest.mark.unit
def test_invalid_result_hash_cannot_build_receipt_ack():
    from planning.protocol.primitive_execution_result_receipt import (
        build_result_receipt_ack_wire,
    )

    result = _complete_result()
    result["result_payload_hash"] = "00" * 32

    with pytest.raises(ValueError, match="result_payload_hash"):
        build_result_receipt_ack_wire(result)
