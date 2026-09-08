"""Explicit Unity final-owner paths used by standalone contract tests.

The frozen Unity checkout is intentionally not imported as a Python package.
These helpers make every test name its final owner, required declaration, and
legacy duplicate that must remain absent.
"""

from __future__ import annotations

import re
from pathlib import Path


UNITY_ROOT = Path("/home/xm/XM/XMflight")
UNITY_DATA = Path("/home/xm/XM/xm_ws/src/unity/XMflight_Data")
UNITY_MANAGED = UNITY_DATA / "Managed"
UNITY_MESSAGEPACK = UNITY_MANAGED / "MessagePack.dll"
RUNTIME_ROOT = UNITY_ROOT / "Assets" / "Scripts" / "Runtime"


def require_owner(relative_path: str, declaration: str, *legacy_paths: str) -> Path:
    """Validate one final owner and return its source path."""

    owner = UNITY_ROOT / relative_path
    assert owner.is_file(), "final Unity owner is missing: {}".format(relative_path)
    source = owner.read_text(encoding="utf-8")
    pattern = r"\b(?:class|struct|enum|interface)\s+{}\b".format(
        re.escape(declaration)
    )
    assert re.search(pattern, source), (
        "final Unity owner {} does not declare {}".format(relative_path, declaration)
    )
    same_name = sorted(UNITY_ROOT.rglob(owner.name))
    assert same_name == [owner], (
        "Unity owner {} is not unique: {}".format(relative_path, same_name)
    )
    for legacy_path in legacy_paths:
        assert not (UNITY_ROOT / legacy_path).exists(), (
            "legacy Unity owner still exists: {}".format(legacy_path)
        )
    return owner


def require_protocol_core() -> list[Path]:
    """Return the split protocol owners used by pure C# contracts."""

    return [
        require_owner(
            "Assets/Scripts/Runtime/Protocol/ProtocolCanonical.cs",
            "XMProtocolV4",
            "Assets/Scripts/XMProtocol.cs",
        ),
        require_owner(
            "Assets/Scripts/Runtime/Protocol/ProtocolModels.cs",
            "EndpointObservationSnapshotV4",
            "Assets/Scripts/XMProtocol.cs",
        ),
    ]


def require_protocol_constants() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Protocol/ProtocolConstants.cs",
        "XMProtocol",
        "Assets/Scripts/XMProtocol.cs",
    )


def require_protocol_identity_registry() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Protocol/ObservationSnapshotIdentityRegistry.cs",
        "ObservationSnapshotV4IdentityRegistry",
        "Assets/Scripts/ObservationSnapshotIdentityRegistry.cs",
    )


def require_observation_cache() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Observation/EndpointObservationCache.cs",
        "EndpointObservationCache",
        "Assets/Scripts/EndpointObservationCache.cs",
    )


def require_capture_ownership() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Observation/EndpointObservationCaptureOwnership.cs",
        "EndpointObservationCaptureOwnership",
        "Assets/Scripts/EndpointObservationCaptureOwnership.cs",
    )


def require_snapshot_service() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Observation/EndpointObservationSnapshotService.cs",
        "EndpointObservationSnapshotService",
        "Assets/Scripts/EndpointObservationSnapshotService.cs",
    )


def require_snapshot_wire_codec() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Protocol/EndpointObservationSnapshotWireCodec.cs",
        "EndpointObservationSnapshotWireCodec",
        "Assets/Scripts/EndpointObservationSnapshotWireCodec.cs",
    )


def require_command_admission() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Primitive/PrimitiveExecutionCommandAdmission.cs",
        "PrimitiveExecutionCommandAdmission",
        "Assets/Scripts/PrimitiveExecutionCommandAdmission.cs",
    )


def require_command_wire_codec() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Protocol/PrimitiveExecutionCommandWireCodec.cs",
        "PrimitiveExecutionCommandWireCodec",
        "Assets/Scripts/PrimitiveExecutionCommandWireCodec.cs",
    )


def require_result_lifecycle() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Primitive/PrimitiveExecutionResultLifecycle.cs",
        "PrimitiveExecutionResultLifecycle",
        "Assets/Scripts/PrimitiveExecutionResultLifecycle.cs",
    )


def require_result_transport_adapter() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Primitive/PrimitiveExecutionResultTransportAdapter.cs",
        "PrimitiveExecutionResultTransportAdapter",
        "Assets/Scripts/PrimitiveExecutionResultTransportAdapter.cs",
    )


def require_reliable_result_receiver() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Primitive/ReliableResultReceiver.cs",
        "ReliableResultReceiver",
        "Assets/Scripts/ReliableResultReceiver.cs",
    )


def require_runtime_integration() -> Path:
    return require_owner(
        "Assets/Scripts/Runtime/Primitive/PrimitiveExecutionRuntimeIntegration.cs",
        "PrimitiveExecutionRuntimeIntegration",
        "Assets/Scripts/PrimitiveExecutionRuntimeIntegration.cs",
    )


def require_execution_controller() -> list[Path]:
    return [
        require_owner(
            "Assets/Scripts/Runtime/Primitive/PrimitiveExecutionController.cs",
            "PrimitiveExecutionController",
            "Assets/Scripts/PrimitiveExecutionController.cs",
        ),
        require_owner(
            "Assets/Scripts/Runtime/Primitive/PrimitiveExecutionSession.cs",
            "PrimitiveExecutionSession",
            "Assets/Scripts/PrimitiveExecutionSession.cs",
        ),
    ]
