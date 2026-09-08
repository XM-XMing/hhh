"""Language-neutral schema-v4 canonical encoding contract tests."""

from collections import OrderedDict
import json
from pathlib import Path

import pytest

from planning.protocol.primitive_execution_schema_v4 import (
    PrimitiveExecutionProtocolError,
    ResultIdentityRegistry,
    canonical_v4_command_sequence_bytes,
    canonical_v4_command_sequence_hash,
    canonical_v4_result_payload_bytes,
    canonical_result_payload_hash,
    validate_execution_result,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests/fixtures/primitive_execution_v4"
VECTOR_NAMES = ("complete", "rejected", "cancelled", "failed")


def _fixture(name):
    return json.loads(
        (FIXTURE_DIR / "{}.json".format(name)).read_text(encoding="utf-8")
    )


@pytest.mark.unit
def test_language_neutral_golden_vectors_match_canonical_bytes_and_hashes():
    command_fixture = _fixture("command_sequence")
    for name in VECTOR_NAMES:
        fixture = _fixture(name)
        result = validate_execution_result(fixture["result"])
        command_bytes = canonical_v4_command_sequence_bytes(command_fixture["command_frames"])
        result_bytes = canonical_v4_result_payload_bytes(result)
        expected_result_hex = (
            FIXTURE_DIR / fixture["result_msgpack_hex_file"]
        ).read_text(encoding="utf-8").strip()

        assert command_bytes.hex() == command_fixture["command_msgpack_hex"]
        assert result_bytes.hex() == expected_result_hex
        assert canonical_v4_command_sequence_hash(command_fixture["command_frames"]) == fixture[
            "command_sequence_hash"
        ]
        assert canonical_result_payload_hash(result) == fixture["result_payload_hash"]
        assert result.result_generation == 0


@pytest.mark.unit
def test_golden_vector_bytes_and_hashes_are_stable_across_100_repetitions():
    command_fixture = _fixture("command_sequence")
    fixture = _fixture("complete")
    result = validate_execution_result(fixture["result"])
    expected_command = bytes.fromhex(command_fixture["command_msgpack_hex"])
    expected_result = bytes.fromhex(
        (FIXTURE_DIR / fixture["result_msgpack_hex_file"])
        .read_text(encoding="utf-8")
        .strip()
    )

    for _ in range(100):
        assert canonical_v4_command_sequence_bytes(command_fixture["command_frames"]) == expected_command
        assert canonical_v4_result_payload_bytes(result) == expected_result
        assert canonical_v4_command_sequence_hash(command_fixture["command_frames"]) == fixture[
            "command_sequence_hash"
        ]
        assert canonical_result_payload_hash(result) == fixture["result_payload_hash"]


@pytest.mark.unit
def test_field_insertion_order_does_not_change_canonical_result_bytes_or_hash():
    fixture = _fixture("complete")
    original = fixture["result"]
    reordered = OrderedDict()
    for key in reversed(list(original.keys())):
        value = original[key]
        if key == "endpoint_observation_ref":
            value = OrderedDict(
                (nested_key, value[nested_key])
                for nested_key in reversed(list(value.keys()))
            )
        reordered[key] = value

    first = validate_execution_result(original)
    second = validate_execution_result(reordered)
    assert canonical_v4_result_payload_bytes(first) == canonical_v4_result_payload_bytes(second)
    assert canonical_result_payload_hash(first) == canonical_result_payload_hash(second)


@pytest.mark.unit
def test_complete_requires_full_exact_observation_ref():
    partial = dict(_fixture("complete")["result"])
    partial["endpoint_observation_ref"] = dict(partial["endpoint_observation_ref"])
    partial["endpoint_observation_ref"].pop("episode_id")

    with pytest.raises(PrimitiveExecutionProtocolError, match="endpoint_observation_ref"):
        validate_execution_result(partial)


@pytest.mark.unit
def test_complete_observation_ref_is_part_of_result_identity():
    fixture = _fixture("complete")
    first = validate_execution_result(fixture["result"])
    changed = dict(fixture["result"])
    changed["endpoint_observation_ref"] = dict(changed["endpoint_observation_ref"])
    changed["endpoint_observation_ref"]["reset_id"] = "reset-v4-0002"
    conflicting = validate_execution_result(changed)
    registry = ResultIdentityRegistry()

    assert canonical_result_payload_hash(first) != canonical_result_payload_hash(conflicting)
    assert registry.observe(first) == "FIRST"
    with pytest.raises(PrimitiveExecutionProtocolError, match="result_payload_hash"):
        registry.observe(conflicting)


@pytest.mark.unit
def test_result_payload_hash_is_excluded_and_utf8_is_encoded_as_utf8_text():
    fixture = _fixture("complete")
    original = validate_execution_result(fixture["result"])
    with_hash = dict(fixture["result"])
    with_hash["result_payload_hash"] = fixture["result_payload_hash"]
    with_hash["runtime_instance_id"] = "worker-00-运行时-test"
    with_hash["endpoint_observation_ref"] = dict(with_hash["endpoint_observation_ref"])
    with_hash["endpoint_observation_ref"]["runtime_instance_id"] = with_hash[
        "runtime_instance_id"
    ]
    without_hash = dict(with_hash)
    without_hash.pop("result_payload_hash")

    utf8_result = validate_execution_result(without_hash)
    with_hash_result = validate_execution_result(
        dict(with_hash, result_payload_hash=canonical_result_payload_hash(utf8_result))
    )
    assert canonical_v4_result_payload_bytes(utf8_result) == canonical_v4_result_payload_bytes(
        with_hash_result
    )
    assert canonical_result_payload_hash(utf8_result) == canonical_result_payload_hash(
        with_hash_result
    )
    assert "运行时".encode("utf-8") in canonical_v4_result_payload_bytes(utf8_result)
    assert canonical_result_payload_hash(original) != canonical_result_payload_hash(utf8_result)


@pytest.mark.unit
def test_command_sequence_order_value_count_and_non_finite_values_are_not_silent():
    command_fixture = _fixture("command_sequence")
    fixture = _fixture("complete")
    frames = command_fixture["command_frames"]
    expected_hash = fixture["command_sequence_hash"]

    swapped = list(frames)
    swapped[10], swapped[11] = swapped[11], swapped[10]
    changed_vx = [dict(frame) for frame in frames]
    changed_vx[10]["action"] = list(changed_vx[10]["action"])
    changed_vx[10]["action"][0] += 1.0
    fewer = frames[:-1]

    assert canonical_v4_command_sequence_hash(frames) == expected_hash
    assert canonical_v4_command_sequence_hash(swapped) != expected_hash
    assert canonical_v4_command_sequence_hash(changed_vx) != expected_hash
    assert canonical_v4_command_sequence_hash(fewer) != expected_hash
    with pytest.raises(PrimitiveExecutionProtocolError):
        canonical_v4_command_sequence_bytes(
            [dict(frames[0], action=[float("nan"), 0.0, 0.0])]
        )
    with pytest.raises(PrimitiveExecutionProtocolError):
        canonical_v4_command_sequence_bytes(
            [dict(frames[0], action=[float("inf"), 0.0, 0.0])]
        )
