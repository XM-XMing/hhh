"""Observation provenance contract used by formal BC mmap inputs."""

from __future__ import annotations

from typing import Any, Dict, Mapping


EXACT_ENDPOINT_OBSERVATION_CONTRACT = "reliable_exact_endpoint_snapshot"
LEGACY_ASYNC_OBSERVATION_CONTRACT = "legacy_async_telemetry"
_SUPPORTED_CONTRACTS = frozenset(
    (EXACT_ENDPOINT_OBSERVATION_CONTRACT, LEGACY_ASYNC_OBSERVATION_CONTRACT)
)


def _serialized_exact_value(key: str, value: Any) -> Any:
    """Normalize JSON/CSV scalar encodings without weakening the contract."""

    if key in {
        "reliable_execution",
        "telemetry_observation",
        "endpoint_identity_available",
        "asynchronous_prefetch",
    }:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
        return value
    if key == "state_depth_skew_ns":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return value


def exact_endpoint_metadata() -> Dict[str, Any]:
    """Return the immutable provenance fields for a new formal row.

    The feature payload is intentionally absent here.  This helper only owns
    how the state/depth pair was acquired; callers remain responsible for the
    existing state, depth, action, and mask layouts.
    """

    return {
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "reliable_execution": True,
        "telemetry_observation": False,
        "state_depth_skew_ns": 0,
        "endpoint_identity_available": True,
        "asynchronous_prefetch": False,
        "asynchronous_prefetch_status": "obsolete_for_reliable_exact",
    }


def observation_contract(metadata: Mapping[str, Any], *, path: str = "<memory>") -> str:
    """Return an explicitly declared supported contract, failing closed."""

    value = str(metadata.get("observation_contract", "")).strip()
    if not value:
        raise ValueError("{} missing observation contract".format(path))
    if value not in _SUPPORTED_CONTRACTS:
        raise ValueError("{} unknown observation contract: {}".format(path, value))
    source = str(metadata.get("observation_source", "")).strip()
    if source and source != value:
        raise ValueError(
            "{} observation contract/source mismatch: {} != {}".format(
                path, value, source
            )
        )
    return value


def observation_provenance(
    metadata: Mapping[str, Any], *, path: str = "<memory>"
) -> dict[str, str]:
    """Return an explicit contract/source pair, failing closed on missing source."""

    contract = observation_contract(metadata, path=path)
    source = str(metadata.get("observation_source", "")).strip()
    if not source:
        raise ValueError("{} missing observation source".format(path))
    if source != contract:
        raise ValueError(
            "{} observation contract/source mismatch: {} != {}".format(
                path, contract, source
            )
        )
    return {
        "observation_contract": contract,
        "observation_source": source,
    }


def validate_reliable_exact_metadata(
    metadata: Mapping[str, Any],
    *,
    path: str = "<memory>",
    require_counters: bool = False,
) -> Dict[str, Any]:
    """Validate metadata for the formal reliable-exact collection path.

    This is deliberately fail-closed.  A missing field, an unknown contract,
    legacy async provenance, or a non-zero endpoint skew cannot be promoted to
    a formal rollout artifact.  ``require_counters`` is used for worker/run
    manifests, where the aggregate provenance counters are part of the public
    contract; individual observations do not carry those counters.
    """

    if not isinstance(metadata, Mapping):
        raise ValueError("{} observation metadata must be a mapping".format(path))
    provenance = observation_provenance(metadata, path=path)
    if provenance["observation_contract"] != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError(
            "{} formal collection requires {}, received {}".format(
                path,
                EXACT_ENDPOINT_OBSERVATION_CONTRACT,
                provenance["observation_contract"],
            )
        )
    expected = exact_endpoint_metadata()
    for key, expected_value in expected.items():
        actual_value = _serialized_exact_value(key, metadata.get(key))
        if actual_value != expected_value:
            raise ValueError(
                "{} exact observation metadata {}={!r} expected {!r}".format(
                    path, key, metadata.get(key), expected_value
                )
            )
    if require_counters:
        for key in ("reliable_rows", "legacy_rows"):
            if key not in metadata:
                raise ValueError("{} missing exact provenance counter {}".format(path, key))
            try:
                value = int(metadata[key])
            except (TypeError, ValueError):
                raise ValueError("{} invalid exact provenance counter {}".format(path, key))
            if value < 0:
                raise ValueError("{} negative exact provenance counter {}".format(path, key))
    return dict(metadata)


__all__ = [
    "EXACT_ENDPOINT_OBSERVATION_CONTRACT",
    "LEGACY_ASYNC_OBSERVATION_CONTRACT",
    "exact_endpoint_metadata",
    "observation_contract",
    "observation_provenance",
    "validate_reliable_exact_metadata",
]
