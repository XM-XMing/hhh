"""Canonical reliable-v4 baseline contract used before P3 collection.

This is deliberately a data-contract validator.  It does not launch Unity,
read telemetry, or inspect learner state.  A baseline can open the P3 replay
gate only when both the evaluation summary and the frozen runtime manifest
identify the same reliable-v4 contract.
"""

from __future__ import annotations

from typing import Mapping

from planning.contracts.policy_runtime import (
    POLICY_RUNTIME_CONTRACT_ID,
    POLICY_UNITY_EVALUATION_CONTRACT_ID,
    POLICY_UNITY_OUTCOME_CONTRACT_ID,
)
from planning.contracts.task import TASK_CONTRACT_ID


BASELINE_CONTRACT_ID = "reliable_v4_fixed_dev_baseline_v1"
EXPECTED_MISSION_INDEX_SHA256 = (
    "135f21f73eeae2b034eda2ddcec0dd0dcaa915e8bf18426ebf9a04508f050316"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "188e41301b67cd8c53ffdd45d55684c5f7626b4a4208eca599e322870e8c31f7"
)
EXPECTED_TASK_CONTRACT_SHA256 = (
    "5862af0f354c408d74cf4904dc7ecf609944553443ae1128097136be979fd30a"
)
EXPECTED_PLAYER_SHA256 = (
    "61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365"
)
EXPECTED_ASSEMBLY_CSHARP_SHA256 = (
    "ca470709131f27586832749d02687b8697de16c9e08920c1485ba674c3d05045"
)
EXPECTED_BRIDGE_SHA256 = (
    "1e614e13577c1d6a3c1d46318eddd31668fd235a89e3a7456ddf6da11ddcd6e7"
)
EXPECTED_SCHEMA_SPEC_SHA256 = (
    "948056976bd90b89504420712d3b23cc7cadda2090cd8aab5b223aa5b296c40a"
)
EXPECTED_HORIZON = 45
EXPECTED_SAFETY_MASK = "depth"
EXPECTED_EXECUTION_MODE = "continuous"
EXPECTED_POLICY_ACTION_MODE = "deterministic_argmax"
EXPECTED_POLICY_TEMPERATURE = 0.0
EXPECTED_DEPTH_MASK_CONFIG = {
    "collision_radius_m": 0.40,
    "slack_m": 0.08,
    "sample_stride": 4,
    "max_patch_radius_px": 14,
}
REQUIRED_CAPABILITIES = (
    "reliable_command_v4",
    "primitive_result_v4",
    "snapshot_v4",
    "reset_v4",
)
REFERENCE_OUTCOMES = {
    "success_count": 48,
    "collision_count": 17,
    "dead_end_count": 30,
    "timeout_count": 5,
    "far_count": 0,
    "altitude_violation_count": 0,
}


def _same(errors, field: str, actual, expected) -> None:
    if actual != expected:
        errors.append("{}={} expected={}".format(field, actual, expected))


def validate_reliable_v4_baseline(
    summary: Mapping[str, object],
    runtime_manifest: Mapping[str, object],
) -> dict:
    """Validate one exact fixed-dev reliable-v4 baseline.

    The validator intentionally rejects the historical 76/2/19/3 result and
    any other mission set, even if the remaining metadata looks compatible.
    """

    errors = []
    summary_fields = {
        "evaluation_contract_id": POLICY_UNITY_EVALUATION_CONTRACT_ID,
        "policy_unity_outcome_contract_id": POLICY_UNITY_OUTCOME_CONTRACT_ID,
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "task_contract_id": TASK_CONTRACT_ID,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "mission_index_sha256": EXPECTED_MISSION_INDEX_SHA256,
        "task_contract_sha256": EXPECTED_TASK_CONTRACT_SHA256,
        "max_steps": EXPECTED_HORIZON,
        "policy_action_mode": EXPECTED_POLICY_ACTION_MODE,
        "safety_mask": EXPECTED_SAFETY_MASK,
        "execution_mode": EXPECTED_EXECUTION_MODE,
    }
    for field, expected in summary_fields.items():
        _same(errors, field, summary.get(field), expected)
    _same(errors, "policy_temperature", summary.get("policy_temperature"), EXPECTED_POLICY_TEMPERATURE)
    if summary.get("reliable_v4") is not True:
        errors.append("reliable_v4 must be true")
    _same(errors, "episodes", summary.get("episodes"), 100)
    _same(errors, "terminal_outcome_total", summary.get("terminal_outcome_total"), 100)
    if summary.get("outcome_partition_valid") is not True:
        errors.append("outcome_partition_valid must be true")
    _same(errors, "unclassified_count", summary.get("unclassified_count"), 0)
    _same(errors, "ambiguous_outcome_count", summary.get("ambiguous_outcome_count"), 0)
    if summary.get("depth_mask_config") != EXPECTED_DEPTH_MASK_CONFIG:
        errors.append("depth_mask_config mismatch")
    for field, expected in REFERENCE_OUTCOMES.items():
        _same(errors, field, summary.get(field), expected)

    manifest_fields = {
        "schema": "p3_runtime_identity_manifest_v1",
        "player_sha256": EXPECTED_PLAYER_SHA256,
        "assembly_csharp_sha256": EXPECTED_ASSEMBLY_CSHARP_SHA256,
        "bridge_sha256": EXPECTED_BRIDGE_SHA256,
        "schema_spec_sha256": EXPECTED_SCHEMA_SPEC_SHA256,
        "bc_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    for field, expected in manifest_fields.items():
        _same(errors, field, runtime_manifest.get(field), expected)
    if runtime_manifest.get("identity_validated") is not True:
        errors.append("identity_validated must be true")
    capabilities = set(runtime_manifest.get("capabilities") or ())
    missing = [capability for capability in REQUIRED_CAPABILITIES if capability not in capabilities]
    if missing:
        errors.append("missing runtime capabilities: {}".format(",".join(missing)))

    if errors:
        raise ValueError("P3 baseline contract mismatch: {}".format("; ".join(errors)))
    return {
        "contract_id": BASELINE_CONTRACT_ID,
        "mission_index_sha256": EXPECTED_MISSION_INDEX_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "task_contract_sha256": EXPECTED_TASK_CONTRACT_SHA256,
        "runtime_identity": {
            "player_sha256": EXPECTED_PLAYER_SHA256,
            "assembly_csharp_sha256": EXPECTED_ASSEMBLY_CSHARP_SHA256,
            "bridge_sha256": EXPECTED_BRIDGE_SHA256,
            "schema_spec_sha256": EXPECTED_SCHEMA_SPEC_SHA256,
        },
        "horizon": EXPECTED_HORIZON,
        "reference_outcomes": dict(REFERENCE_OUTCOMES),
    }
