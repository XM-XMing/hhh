#!/usr/bin/env python3
"""Regression test for deployable observations and training-only route labels."""

from __future__ import annotations




from planning.bc.model import build_model, require_torch
from planning.contracts.feature import (
    FEATURE_CONTRACT_ID,
    FORBIDDEN_PRIVILEGED_RUNTIME_FIELDS,
    POLICY_OBSERVATION_FIELDS,
    POLICY_VECTOR_DIM,
    policy_input_contract_sha256,
    validate_checkpoint_policy_input_contract,
)


def main() -> int:
    expected_fields = ("depth", "state", "final_goal", "previous_action")
    assert POLICY_OBSERVATION_FIELDS == expected_fields
    assert FEATURE_CONTRACT_ID == "depth_goal_state_prev_action"
    assert POLICY_VECTOR_DIM == 127

    validate_checkpoint_policy_input_contract({
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "privileged_runtime_inputs": [],
    })
    privileged_rejected = False
    try:
        validate_checkpoint_policy_input_contract({"privileged_runtime_inputs": ["global_route"]})
    except ValueError:
        privileged_rejected = True
    assert privileged_rejected
    hash_rejected = False
    try:
        validate_checkpoint_policy_input_contract({"policy_input_contract_sha256": "0" * 64})
    except ValueError:
        hash_rejected = True
    assert hash_rejected

    # The deployable model accepts only depth plus the fixed 127-D vector.
    torch, nn, _, _, _ = require_torch()
    model = build_model(nn)
    depth = torch.zeros((2, 1, 90, 160), dtype=torch.float32)
    vector = torch.zeros((2, POLICY_VECTOR_DIM), dtype=torch.float32)
    assert model(depth, vector).shape == (2, 105)
    assert not any(
        forbidden in name
        for forbidden in FORBIDDEN_PRIVILEGED_RUNTIME_FIELDS
        for name in model.state_dict().keys()
    )

    print("POLICY_INPUT_CONTRACT_TEST")
    print("  fields:", POLICY_OBSERVATION_FIELDS)
    print("  vector_dim:", POLICY_VECTOR_DIM)
    print("  policy_input_contract_sha256:", policy_input_contract_sha256())
    print("  privileged_runtime_inputs: []")
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
