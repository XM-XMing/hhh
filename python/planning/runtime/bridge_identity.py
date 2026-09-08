"""Canonical identity for the production split Bridge implementation.

This module contains only runtime/source provenance.  It does not select a
port, alter a protocol payload, or implement a learner.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

from planning.protocol.constants import SCHEMA_VERSION

BRIDGE_BINARY_NAME = "unity_bridge_node"
BRIDGE_IMPLEMENTATION = "split"
BRIDGE_OWNER = "BridgeNode"
BRIDGE_PROTOCOL_SCHEMA_VERSION = SCHEMA_VERSION
BRIDGE_SOURCE_MANIFEST_VERSION = "bridge_source_manifest_v1"

# Keep this list aligned with PLANNING_BRIDGE_SOURCES in CMakeLists.txt and
# with the public headers included by that target.  Compatibility forwarding
# headers are intentionally not owners of the production protocol.
BRIDGE_SOURCE_FILES: Tuple[str, ...] = (
    "CMakeLists.txt",
    "package.xml",
    "include/planning/bridge/bridge_util.hpp",
    "include/planning/bridge/command_gateway.hpp",
    "include/planning/bridge/observation_retrieval_broker.hpp",
    "include/planning/bridge/primitive_execution_command_broker.hpp",
    "include/planning/bridge/primitive_execution_result_broker.hpp",
    "include/planning/bridge/reset_gateway.hpp",
    "include/planning/bridge/result_gateway.hpp",
    "include/planning/bridge/snapshot_gateway.hpp",
    "include/planning/bridge/telemetry_bridge.hpp",
    "include/planning/bridge/unity_bridge_node.hpp",
    "include/planning/protocol/endpoint_observation_snapshot_wire.hpp",
    "include/planning/protocol/primitive_execution_command_wire.hpp",
    "include/planning/protocol/primitive_execution_result_wire.hpp",
    "include/planning/protocol/primitive_reset_wire.hpp",
    "include/planning/protocol/telemetry_wire.hpp",
    "include/planning/protocol/xm_protocol.hpp",
    "include/planning/transport/bridge_transport.hpp",
    "include/planning/transport/command_transport.hpp",
    "include/planning/transport/reliable_zmq_adapters.hpp",
    "include/planning/transport/result_transport.hpp",
    "include/planning/transport/snapshot_transport.hpp",
    "include/planning/transport/telemetry_transport.hpp",
    "include/planning/transport/zmq_option_contract.hpp",
    "include/planning/transport/zmq_socket.hpp",
    "launch/sim_realtime.launch",
    "launch/unity_bridge.launch",
    "src/bridge/bridge_node.cpp",
    "src/bridge/bridge_util.cpp",
    "src/bridge/command_gateway.cpp",
    "src/bridge/reset_gateway.cpp",
    "src/bridge/result_gateway.cpp",
    "src/bridge/snapshot_gateway.cpp",
    "src/bridge/telemetry_bridge.cpp",
    "src/protocol/endpoint_observation_snapshot_wire.cpp",
    "src/protocol/telemetry_wire.cpp",
    "src/transport/bridge_transport.cpp",
    "src/transport/command_transport.cpp",
    "src/transport/result_transport.cpp",
    "src/transport/snapshot_transport.cpp",
    "src/transport/telemetry_transport.cpp",
    "src/transport/zmq_option_contract.cpp",
    "src/transport/zmq_socket.cpp",
    "src/unity_bridge_main.cpp",
)


def planning_bridge_binary(package_root: Optional[Path] = None) -> Path:
    """Return the package's canonical catkin/devel Bridge path.

    A caller may still override this path explicitly through its existing
    ``PLANNING_BRIDGE_BINARY``/``BRIDGE_BIN`` contract.  This default never
    points at a package-local ``devel`` directory: catkin places the devel
    space beside the workspace ``src`` directory.
    """

    root = (
        Path(package_root).expanduser().resolve()
        if package_root is not None
        else Path(__file__).resolve().parents[3]
    )
    if root.parent.name == "src":
        workspace = root.parent.parent
    else:
        workspace = next(
            (
                parent
                for parent in root.parents
                if (parent / "src" / "planning" / "package.xml").is_file()
            ),
            root.parent.parent,
        )
    return workspace / "devel" / "lib" / "planning" / BRIDGE_BINARY_NAME


def bridge_source_manifest_sha256(package_root: Optional[Path] = None) -> str:
    """Hash the exact source list that defines the formal split Bridge."""

    root = (
        Path(package_root).expanduser().resolve()
        if package_root is not None
        else Path(__file__).resolve().parents[3]
    )
    digest = hashlib.sha256()
    for relative in sorted(BRIDGE_SOURCE_FILES):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                "Bridge source manifest file missing: {}".format(path)
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def bridge_identity_manifest(package_root: Optional[Path] = None) -> Dict[str, object]:
    """Return deterministic identity metadata for runtime manifests."""

    return {
        "owner": BRIDGE_OWNER,
        "implementation": BRIDGE_IMPLEMENTATION,
        "binary_name": BRIDGE_BINARY_NAME,
        "protocol_schema_version": BRIDGE_PROTOCOL_SCHEMA_VERSION,
        "source_manifest_version": BRIDGE_SOURCE_MANIFEST_VERSION,
        "source_files": list(BRIDGE_SOURCE_FILES),
        "source_manifest_sha256": bridge_source_manifest_sha256(package_root),
    }


__all__ = [
    "BRIDGE_BINARY_NAME",
    "BRIDGE_IMPLEMENTATION",
    "BRIDGE_OWNER",
    "BRIDGE_PROTOCOL_SCHEMA_VERSION",
    "BRIDGE_SOURCE_FILES",
    "BRIDGE_SOURCE_MANIFEST_VERSION",
    "bridge_identity_manifest",
    "bridge_source_manifest_sha256",
    "planning_bridge_binary",
]
