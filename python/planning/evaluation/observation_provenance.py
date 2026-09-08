"""Fail-closed observation provenance checks for policy evaluation."""

from __future__ import annotations

from typing import Any, Mapping

from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    observation_provenance,
)


def _nonnegative_count(value: Any, *, name: str) -> int:
    count = int(value)
    if count < 0:
        raise ValueError("{} must be non-negative".format(name))
    return count


def resolve_evaluation_observation_provenance(
    checkpoint: Mapping[str, Any],
    *,
    expected_observation_contract: str = EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    runtime_observation_contract: str,
    runtime_observation_source: str,
    reliable_execution_enabled: bool,
    telemetry_observation_enabled: bool,
    telemetry_fallback_enabled: bool,
    state_depth_exact_endpoint_binding: bool,
    snapshot_missing_count: int = 0,
    telemetry_lookup_count: int = 0,
    allow_override: bool = False,
) -> dict[str, Any]:
    """Validate checkpoint/runtime observation provenance before evaluation.

    The exact endpoint snapshot contract is the only formal evaluation mode.
    ``allow_override`` is intentionally explicit and is reserved for labeled
    diagnostic runs; every relaxed condition is retained in the returned
    provenance record.
    """

    expected = observation_provenance(
        {
            "observation_contract": expected_observation_contract,
            "observation_source": expected_observation_contract,
        },
        path="expected observation",
    )
    checkpoint_value = observation_provenance(checkpoint, path="checkpoint")
    runtime_value = observation_provenance(
        {
            "observation_contract": runtime_observation_contract,
            "observation_source": runtime_observation_source,
        },
        path="runtime",
    )

    snapshot_missing = _nonnegative_count(
        snapshot_missing_count, name="snapshot_missing_count"
    )
    telemetry_lookup = _nonnegative_count(
        telemetry_lookup_count, name="telemetry_lookup_count"
    )
    violations: list[str] = []
    if expected["observation_contract"] != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        violations.append(
            "formal evaluation requires {}".format(
                EXACT_ENDPOINT_OBSERVATION_CONTRACT
            )
        )
    if checkpoint_value["observation_contract"] != expected["observation_contract"]:
        violations.append(
            "checkpoint observation contract mismatch: received={} expected={}".format(
                checkpoint_value["observation_contract"],
                expected["observation_contract"],
            )
        )
    if checkpoint_value["observation_contract"] != runtime_value["observation_contract"]:
        violations.append(
            "checkpoint/runtime observation contract mismatch: checkpoint={} runtime={}".format(
                checkpoint_value["observation_contract"],
                runtime_value["observation_contract"],
            )
        )
    if checkpoint_value["observation_source"] != runtime_value["observation_source"]:
        violations.append(
            "checkpoint/runtime observation source mismatch: checkpoint={} runtime={}".format(
                checkpoint_value["observation_source"],
                runtime_value["observation_source"],
            )
        )

    exact_runtime = runtime_value["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    if exact_runtime and not bool(reliable_execution_enabled):
        violations.append("reliable exact endpoint execution is disabled")
    if exact_runtime and bool(telemetry_observation_enabled):
        violations.append("reliable exact endpoint runtime consulted telemetry")
    if exact_runtime and bool(telemetry_fallback_enabled):
        violations.append("reliable exact endpoint runtime enabled telemetry fallback")
    if exact_runtime and not bool(state_depth_exact_endpoint_binding):
        violations.append("reliable exact endpoint state/depth binding is disabled")
    if exact_runtime and snapshot_missing != 0:
        violations.append(
            "reliable exact endpoint snapshot missing count is {}".format(
                snapshot_missing
            )
        )
    if exact_runtime and telemetry_lookup != 0:
        violations.append(
            "reliable exact endpoint telemetry lookup count is {}".format(
                telemetry_lookup
            )
        )

    if violations and not bool(allow_override):
        raise ValueError("reliable exact endpoint observation validation failed: {}".format("; ".join(violations)))

    return {
        "checkpoint_observation_contract": checkpoint_value["observation_contract"],
        "checkpoint_observation_source": checkpoint_value["observation_source"],
        "expected_observation_contract": expected["observation_contract"],
        "runtime_observation_contract": runtime_value["observation_contract"],
        "runtime_observation_source": runtime_value["observation_source"],
        "reliable_execution": bool(reliable_execution_enabled),
        "telemetry_observation": bool(telemetry_observation_enabled),
        "telemetry_fallback_enabled": bool(telemetry_fallback_enabled),
        "state_depth_exact_endpoint_binding": bool(state_depth_exact_endpoint_binding),
        "snapshot_missing_count": snapshot_missing,
        "telemetry_lookup_count": telemetry_lookup,
        "runtime_contract_override": bool(violations),
    }


def build_evaluation_provenance_summary(
    provenance: Mapping[str, Any], *, runtime_contract_override: bool | None = None
) -> dict[str, Any]:
    """Project validated provenance into the stable evaluation summary schema."""

    override = (
        bool(provenance.get("runtime_contract_override", False))
        if runtime_contract_override is None
        else bool(runtime_contract_override)
    )
    runtime_contract = str(provenance["runtime_observation_contract"])
    runtime_source = str(provenance["runtime_observation_source"])
    return {
        "observation_contract": runtime_contract,
        "observation_source": runtime_source,
        "checkpoint_observation_contract": str(
            provenance["checkpoint_observation_contract"]
        ),
        "checkpoint_observation_source": str(
            provenance["checkpoint_observation_source"]
        ),
        "expected_observation_contract": str(
            provenance["expected_observation_contract"]
        ),
        "runtime_observation_contract": runtime_contract,
        "runtime_observation_source": runtime_source,
        "runtime_contract_override": override,
        "reliable_execution": bool(provenance["reliable_execution"]),
        "telemetry_observation": bool(provenance["telemetry_observation"]),
        "telemetry_fallback_enabled": bool(
            provenance["telemetry_fallback_enabled"]
        ),
        "state_depth_exact_endpoint_binding": bool(
            provenance["state_depth_exact_endpoint_binding"]
        ),
        "snapshot_missing_count": int(provenance["snapshot_missing_count"]),
        "telemetry_lookup_count": int(provenance["telemetry_lookup_count"]),
    }


__all__ = [
    "build_evaluation_provenance_summary",
    "resolve_evaluation_observation_provenance",
]
