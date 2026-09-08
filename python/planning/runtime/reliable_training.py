"""Small, serializable training-time seam for reliable-v4 Unity workers.

The parent process passes only this configuration through the spawned-worker
boundary.  ZMQ sockets and the endpoint provider are created inside the child,
after its ROS environment has been installed.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    exact_endpoint_metadata,
)

# Protocol version 4 remains part of the transport/runtime name.  The
# observation semantic is the shared formal contract used by BC, replay, and
# evaluation.
RELIABLE_V4_OBSERVATION_SEMANTICS = EXACT_ENDPOINT_OBSERVATION_CONTRACT


def build_reliable_v4_runtime_config(
    *,
    runtime_instance_id: str,
    command_endpoint: str,
    result_endpoint: str,
    snapshot_endpoint: str,
    timeout_s: float,
) -> Dict[str, Any]:
    """Return the immutable, spawn-safe runtime configuration."""

    values = {
        "runtime_instance_id": str(runtime_instance_id),
        "command_endpoint": str(command_endpoint),
        "result_endpoint": str(result_endpoint),
        "snapshot_endpoint": str(snapshot_endpoint),
        "timeout_s": float(timeout_s),
        "observation_semantics": RELIABLE_V4_OBSERVATION_SEMANTICS,
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    }
    for name in (
        "runtime_instance_id",
        "command_endpoint",
        "result_endpoint",
        "snapshot_endpoint",
    ):
        if not values[name]:
            raise ValueError("reliable-v4 {} is required".format(name))
    if values["timeout_s"] <= 0.0:
        raise ValueError("reliable-v4 timeout_s must be positive")
    return values


def build_reliable_v4_backend(config: Mapping[str, Any]):
    """Construct sockets/provider/backend inside a worker process only."""

    if not isinstance(config, Mapping):
        raise TypeError("reliable-v4 runtime config must be a mapping")
    if config.get("observation_semantics") != RELIABLE_V4_OBSERVATION_SEMANTICS:
        raise ValueError("reliable-v4 observation semantics mismatch")
    exact_metadata = exact_endpoint_metadata()
    for key, expected in (
        ("observation_contract", exact_metadata["observation_contract"]),
        ("observation_source", exact_metadata["observation_source"]),
    ):
        if config.get(key) != expected:
            raise ValueError("reliable-v4 {} mismatch".format(key))

    import zmq

    from planning.runtime.reliable_endpoint_snapshot_provider import (
        BridgeSnapshotEndpointProvider,
        ZmqBridgeSnapshotRetriever,
    )
    from planning.runtime.reliable_unity_env_backend import (
        ZmqReliableV4EnvironmentBackend,
    )

    context = zmq.Context()
    timeout_s = float(config["timeout_s"])
    retriever = ZmqBridgeSnapshotRetriever(
        context,
        str(config["snapshot_endpoint"]),
        timeout_ms=max(1, int(timeout_s * 1000.0)),
    )
    provider = BridgeSnapshotEndpointProvider(retriever)
    return ZmqReliableV4EnvironmentBackend(
        context,
        runtime_instance_id=str(config["runtime_instance_id"]),
        command_endpoint=str(config["command_endpoint"]),
        result_endpoint=str(config["result_endpoint"]),
        snapshot_provider=provider,
        timeout_s=timeout_s,
    )


__all__ = [
    "RELIABLE_V4_OBSERVATION_SEMANTICS",
    "build_reliable_v4_backend",
    "build_reliable_v4_runtime_config",
]
