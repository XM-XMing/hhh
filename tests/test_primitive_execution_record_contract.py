"""RED unit specification for validating Unity primitive execution receipts."""

import copy

import pytest


def _valid_execution_receipt():
    return {
        "execution_id": "episode451-primitive0",
        "requested_frame_count": 25,
        "applied_frames": [
            {
                "execution_id": "episode451-primitive0",
                "frame_index": frame_index,
                "command_id": 1000 + frame_index,
                "applied_state_id": 500 + frame_index,
            }
            for frame_index in range(25)
        ],
        "endpoint_state_id": 524,
    }


@pytest.mark.unit
def test_exact_25_tick_execution_receipt_satisfies_endpoint_contract():
    from planning.contracts.primitive_execution import validate_primitive_execution

    validate_primitive_execution(_valid_execution_receipt(), expected_frame_count=25)


@pytest.mark.unit
@pytest.mark.parametrize("actual_frames", [24, 26])
def test_execution_receipt_rejects_n_minus_or_plus_one_tick(actual_frames):
    from planning.contracts.primitive_execution import (
        PrimitiveExecutionContractError,
        validate_primitive_execution,
    )

    receipt = _valid_execution_receipt()
    receipt["applied_frames"] = receipt["applied_frames"][:actual_frames]
    if actual_frames == 26:
        receipt["applied_frames"].append(
            {"frame_index": 25, "command_id": 1025, "applied_state_id": 525}
        )
    with pytest.raises(PrimitiveExecutionContractError):
        validate_primitive_execution(receipt, expected_frame_count=25)


@pytest.mark.unit
@pytest.mark.parametrize("endpoint_state_id", [523, 525])
def test_execution_receipt_rejects_endpoint_before_or_after_k_plus_24(
    endpoint_state_id,
):
    from planning.contracts.primitive_execution import (
        PrimitiveExecutionContractError,
        validate_primitive_execution,
    )

    receipt = copy.deepcopy(_valid_execution_receipt())
    receipt["endpoint_state_id"] = endpoint_state_id
    with pytest.raises(PrimitiveExecutionContractError):
        validate_primitive_execution(receipt, expected_frame_count=25)


@pytest.mark.unit
@pytest.mark.parametrize(
    "defect", ["duplicate", "missing", "reordered", "wrong_execution", "absent_execution"]
)
def test_execution_receipt_rejects_non_contiguous_or_wrong_execution_frames(defect):
    from planning.contracts.primitive_execution import (
        PrimitiveExecutionContractError,
        validate_primitive_execution,
    )

    receipt = copy.deepcopy(_valid_execution_receipt())
    if defect == "duplicate":
        receipt["applied_frames"][10]["frame_index"] = 9
    elif defect == "missing":
        del receipt["applied_frames"][10]
    elif defect == "reordered":
        receipt["applied_frames"][10], receipt["applied_frames"][11] = (
            receipt["applied_frames"][11], receipt["applied_frames"][10]
        )
    elif defect == "wrong_execution":
        receipt["applied_frames"][10]["execution_id"] = "another-execution"
    else:
        del receipt["applied_frames"][10]["execution_id"]
    with pytest.raises(PrimitiveExecutionContractError):
        validate_primitive_execution(receipt, expected_frame_count=25)


@pytest.mark.unit
def test_execution_receipt_rejects_wrong_acknowledged_command_identity():
    from planning.contracts.primitive_execution import (
        PrimitiveExecutionContractError,
        validate_primitive_execution,
    )

    receipt = _valid_execution_receipt()
    submitted_command_ids = [1000 + index for index in range(25)]
    receipt["applied_frames"][10]["command_id"] = 9999
    with pytest.raises(PrimitiveExecutionContractError):
        validate_primitive_execution(
            receipt,
            expected_frame_count=25,
            expected_command_ids=submitted_command_ids,
        )
