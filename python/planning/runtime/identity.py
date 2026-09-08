"""Fail-fast identity and capability checks for the reliable-v4 runtime.

This module deliberately has no ROS, ZMQ, Unity, or learner dependency.  It
is a startup gate: all inputs are inspected before a reset or a training
worker is started.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping, Optional, Tuple

from planning.common import file_sha256, read_json, write_json_atomic
from planning.runtime.bridge_identity import bridge_identity_manifest


RUNTIME_IDENTITY_ERROR = "STARTUP_RUNTIME_IDENTITY_MISMATCH"
RUNTIME_CAPABILITY_ERROR = "STARTUP_RUNTIME_CAPABILITY_MISSING"

DEFAULT_REQUIRED_CAPABILITIES = (
    "reliable_command_v4",
    "primitive_result_v4",
    "snapshot_v4",
    "reset_v4",
)

# The short capability names are the manifest contract.  The aliases are
# stable managed-code type/method markers used by the currently frozen player;
# older players contain none of the corresponding marker sets.
_CAPABILITY_MARKERS = {
    "reliable_command_v4": (
        "reliable_command_v4",
        "ZmqPrimitiveExecutionCommandTransport",
        "PrimitiveExecutionCommandWireCodec",
        "PrimitiveExecutionCommandAdmission",
    ),
    "primitive_result_v4": (
        "primitive_result_v4",
        "PrimitiveExecutionResultWireCodec",
        "ZmqPrimitiveExecutionResultTransport",
        "PrimitiveExecutionResultLifecycle",
    ),
    "snapshot_v4": (
        "snapshot_v4",
        "EndpointObservationSnapshotV4",
        "EndpointObservationSnapshotWireCodec",
        "ZmqEndpointObservationSnapshotTransport",
        "EndpointObservationCache",
    ),
    "reset_v4": (
        "reset_v4",
        "PrimitiveResetV4WireCodec",
        "PrimitiveResetV4Request",
        "PrimitiveResetV4Complete",
        "SendResetComplete",
    ),
}


class RuntimeIdentityMismatchError(RuntimeError):
    """The selected runtime does not match the expected frozen identity."""


@dataclass(frozen=True)
class RuntimeIdentityExpectation:
    """Expected hashes and capabilities for one frozen runtime contract."""

    player_sha256: str
    assembly_csharp_sha256: str
    bridge_sha256: str
    schema_spec_sha256: str
    bc_checkpoint_sha256: str
    required_capabilities: Tuple[str, ...] = DEFAULT_REQUIRED_CAPABILITIES


def runtime_artifact_identity(
    *,
    player: Path,
    assembly: Optional[Path] = None,
    bridge: Optional[Path] = None,
) -> Dict[str, object]:
    """Resolve and hash the immutable runtime artifacts used by collection.

    Collection does not require a BC checkpoint, so it uses this smaller
    identity seam instead of the learner-oriented full runtime manifest.
    Missing artifacts fail closed before a formal reset is attempted.
    """

    player_path = Path(player).expanduser().resolve()
    assembly_value = str(os.environ.get("PLANNING_UNITY_ASSEMBLY", "")).strip()
    if assembly is not None:
        assembly_path = Path(assembly).expanduser().resolve()
    elif assembly_value:
        assembly_path = Path(assembly_value).expanduser().resolve()
    else:
        assembly_path = (
            player_path.parent
            / (player_path.stem + "_Data")
            / "Managed"
            / "Assembly-CSharp.dll"
        )

    bridge_value = str(os.environ.get("PLANNING_BRIDGE_BINARY", "")).strip()
    if bridge is not None:
        bridge_path = Path(bridge).expanduser().resolve()
    elif bridge_value:
        bridge_path = Path(bridge_value).expanduser().resolve()
    else:
        from planning.runtime.bridge_identity import planning_bridge_binary

        canonical = planning_bridge_binary()
        discovered = shutil.which("unity_bridge_node")
        bridge_path = canonical if canonical.is_file() else Path(discovered or "")
        bridge_path = bridge_path.expanduser().resolve()

    paths = {
        "unity_player": player_path,
        "runtime_assembly": assembly_path,
        "bridge": bridge_path,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeIdentityMismatchError(
            "{}: missing runtime artifacts {}".format(
                RUNTIME_IDENTITY_ERROR,
                ",".join(
                    "{}={}".format(name, paths[name]) for name in missing
                ),
            )
        )
    assembly_sha256 = file_sha256(assembly_path)
    return {
        "unity_player_path": str(player_path),
        "unity_player_sha256": file_sha256(player_path),
        "runtime_assembly_path": str(assembly_path),
        "runtime_assembly_sha256": assembly_sha256,
        "assembly_csharp_sha256": assembly_sha256,
        "bridge_path": str(bridge_path),
        "bridge_sha256": file_sha256(bridge_path),
    }


def _actual_hashes(paths: Mapping[str, Path]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for key, path in paths.items():
        if not path.is_file():
            raise RuntimeIdentityMismatchError(
                "{}: missing {} file: {}".format(RUNTIME_IDENTITY_ERROR, key, path)
            )
        values[key] = file_sha256(path)
    return values


def _assembly_capabilities(path: Path) -> Tuple[str, ...]:
    """Return v4 capabilities explicitly present in managed assembly bytes.

    Unity's managed assembly is a binary artifact.  The v4 implementation
    exports stable type/feature marker strings, so checking those bytes is a
    cheap preflight capability gate and avoids discovering an old player only
    after a reset timeout.
    """

    payload = path.read_bytes()
    return tuple(
        capability
        for capability in DEFAULT_REQUIRED_CAPABILITIES
        if any(marker.encode("ascii") in payload for marker in _CAPABILITY_MARKERS[capability])
    )


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    write_json_atomic(path, manifest, trailing_newline=True)


def build_and_validate_runtime_manifest(
    *,
    player: Path,
    assembly: Path,
    bridge: Path,
    schema: Path,
    bc: Path,
    expected: RuntimeIdentityExpectation,
    manifest_path: Path,
) -> Dict[str, object]:
    """Validate all frozen artifacts and persist a JSON runtime manifest.

    The function returns only after both identity and capability checks pass.
    A mismatch is reported with a stable startup error prefix so callers can
    distinguish it from a reset or transport timeout.
    """

    paths = {
        "player_sha256": Path(player),
        "assembly_csharp_sha256": Path(assembly),
        "bridge_sha256": Path(bridge),
        "schema_spec_sha256": Path(schema),
        "bc_checkpoint_sha256": Path(bc),
    }
    actual = _actual_hashes(paths)
    expected_values = {
        "player_sha256": expected.player_sha256.lower(),
        "assembly_csharp_sha256": expected.assembly_csharp_sha256.lower(),
        "bridge_sha256": expected.bridge_sha256.lower(),
        "schema_spec_sha256": expected.schema_spec_sha256.lower(),
        "bc_checkpoint_sha256": expected.bc_checkpoint_sha256.lower(),
    }
    mismatches = {
        key: {"expected": expected_values[key], "actual": actual[key]}
        for key in expected_values
        if actual[key] != expected_values[key]
    }
    if mismatches:
        raise RuntimeIdentityMismatchError(
            "{}: {}".format(
                RUNTIME_IDENTITY_ERROR,
                json.dumps(mismatches, sort_keys=True, separators=(",", ":")),
            )
        )

    capabilities = _assembly_capabilities(paths["assembly_csharp_sha256"])
    required = tuple(expected.required_capabilities)
    missing = tuple(capability for capability in required if capability not in capabilities)
    if missing:
        raise RuntimeIdentityMismatchError(
            "{}: missing={}".format(RUNTIME_CAPABILITY_ERROR, ",".join(missing))
        )

    manifest: Dict[str, object] = {
        "schema": "p3_runtime_identity_manifest_v1",
        "player_path": str(paths["player_sha256"].resolve()),
        "assembly_csharp_path": str(paths["assembly_csharp_sha256"].resolve()),
        "bridge_path": str(paths["bridge_sha256"].resolve()),
        "schema_spec_path": str(paths["schema_spec_sha256"].resolve()),
        "bc_checkpoint_path": str(paths["bc_checkpoint_sha256"].resolve()),
        "player_sha256": actual["player_sha256"],
        "assembly_csharp_sha256": actual["assembly_csharp_sha256"],
        "bridge_sha256": actual["bridge_sha256"],
        "bridge_identity": bridge_identity_manifest(),
        "schema_spec_sha256": actual["schema_spec_sha256"],
        "bc_checkpoint_sha256": actual["bc_checkpoint_sha256"],
        "capabilities": list(capabilities),
        "required_capabilities": list(required),
        "identity_validated": True,
    }
    _write_manifest(Path(manifest_path), manifest)
    return manifest


def validate_pre_collection_runtime(
    *,
    player: Path,
    unity_data: Path,
    bridge: Path,
    point_cloud: Path,
    voxel_cache: Path,
    mpl_npz: Path,
    mpl_json: Path,
    max_steps: int,
    expected_observation_contract: str,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate the immutable runtime inputs used by formal collection.

    The release JSON is the current runtime identity manifest for the
    source-only, devel build.  This function compares its recorded identities
    with the selected files using the existing hash, MPL, task, and native
    geometry owners; it does not introduce a second hashing or contract
    implementation.
    """

    from planning.common.paths import planning_package_root
    from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
    from planning.contracts.task import task_contract_sha256
    from planning.native.geometry import NativeGeometryContext
    from planning.primitives.library import MotionPrimitiveLibrary

    player_path = Path(player).expanduser().resolve()
    unity_data_path = Path(unity_data).expanduser().resolve()
    bridge_path = Path(bridge).expanduser().resolve()
    point_cloud_path = Path(point_cloud).expanduser().resolve()
    cache_path = Path(voxel_cache).expanduser().resolve()
    mpl_npz_path = Path(mpl_npz).expanduser().resolve()
    mpl_json_path = Path(mpl_json).expanduser().resolve()
    if not unity_data_path.is_dir():
        raise RuntimeIdentityMismatchError(
            "{}: missing Unity data directory {}".format(
                RUNTIME_IDENTITY_ERROR, unity_data_path
            )
        )
    assembly_path = unity_data_path / "Managed" / "Assembly-CSharp.dll"
    manifest_file = Path(manifest_path).expanduser().resolve() if manifest_path else (
        planning_package_root() / "docs" / "pre_collection_runtime_final_v3.json"
    ).resolve()
    manifest = read_json(manifest_file)
    if not isinstance(manifest, Mapping):
        raise ValueError("runtime identity manifest must be a mapping")
    runtime = manifest.get("runtime_identity")
    maps = manifest.get("map_identity")
    contracts = manifest.get("contracts")
    if not isinstance(runtime, Mapping) or not isinstance(maps, Mapping) or not isinstance(contracts, Mapping):
        raise ValueError("runtime identity manifest is missing identity sections")

    expected_observation = str(expected_observation_contract).strip()
    if expected_observation != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError(
            "formal collection requires observation contract {}".format(
                EXACT_ENDPOINT_OBSERVATION_CONTRACT
            )
        )
    if str(contracts.get("observation_contract", "")).strip() != expected_observation:
        raise ValueError("runtime manifest observation contract mismatch")
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    if int(contracts.get("task_contract_schema_version", -1)) != 2:
        raise ValueError("runtime manifest task contract schema is not V2")
    expected_task_sha = str(contracts.get("task_contract_sha256", "")).strip().lower()
    if task_contract_sha256(int(max_steps)) != expected_task_sha:
        raise ValueError("runtime manifest task contract SHA mismatch")

    identity = runtime_artifact_identity(
        player=player_path, assembly=assembly_path, bridge=bridge_path
    )
    actual_hashes = {
        "unity_player_sha256": str(identity["unity_player_sha256"]),
        "assembly_sha256": str(identity["assembly_csharp_sha256"]),
        "bridge_sha256": str(identity["bridge_sha256"]),
        "point_cloud_sha256": file_sha256(point_cloud_path),
        "voxel_cache_sha256": file_sha256(cache_path),
        "mpl_npz_sha256": file_sha256(mpl_npz_path),
        "mpl_json_sha256": file_sha256(mpl_json_path),
    }
    expected_hashes = {
        key: str(section.get(key, "")).strip().lower()
        for key, section in (
            ("unity_player_sha256", runtime),
            ("assembly_sha256", runtime),
            ("bridge_sha256", runtime),
            ("point_cloud_sha256", maps),
            ("voxel_cache_sha256", maps),
            ("mpl_npz_sha256", contracts),
            ("mpl_json_sha256", contracts),
        )
    }
    mismatches = {
        key: {"expected": expected_hashes[key], "actual": actual_hashes[key]}
        for key in expected_hashes
        if expected_hashes[key] != actual_hashes[key]
    }
    if mismatches:
        raise RuntimeIdentityMismatchError(
            "{}: {}".format(
                RUNTIME_IDENTITY_ERROR,
                json.dumps(mismatches, sort_keys=True, separators=(",", ":")),
            )
        )

    mpl = MotionPrimitiveLibrary(
        npz_path=str(mpl_npz_path), metadata_json=str(mpl_json_path)
    )
    if mpl.num_actions != 105 or mpl.frames != 25 or mpl.reference_path(0).shape[0] != 26:
        raise ValueError(
            "MPL contract dimensions are {},{},{}; expected 105,25,26".format(
                mpl.num_actions, mpl.frames, mpl.reference_path(0).shape[0]
            )
        )
    if str(mpl.contract_sha256).lower() != str(contracts.get("mpl_contract_sha256", "")).lower():
        raise ValueError("MPL contract SHA mismatch")

    geometry = NativeGeometryContext.from_voxel_cache(
        cache_path,
        voxel_size=float(maps.get("voxel_size_m", 0.0)),
    )
    try:
        cache_metadata = {
            "voxel_size_m": float(geometry.voxel_size),
            "origin_ijk": [int(value) for value in geometry.origin_ijk],
            "grid_shape": [int(value) for value in geometry.grid_shape],
            "occupied_voxel_count": int(geometry.occupied_keys.shape[0]),
        }
    finally:
        geometry.close()

    return {
        "runtime_manifest": str(manifest_file),
        "runtime_manifest_sha256": file_sha256(manifest_file),
        "unity_player_sha256": actual_hashes["unity_player_sha256"],
        "assembly_sha256": actual_hashes["assembly_sha256"],
        "bridge_sha256": actual_hashes["bridge_sha256"],
        "point_cloud_sha256": actual_hashes["point_cloud_sha256"],
        "voxel_cache_sha256": actual_hashes["voxel_cache_sha256"],
        "mpl_npz_sha256": actual_hashes["mpl_npz_sha256"],
        "mpl_json_sha256": actual_hashes["mpl_json_sha256"],
        "mpl_contract_sha256": str(mpl.contract_sha256),
        "task_contract_sha256": expected_task_sha,
        "max_steps": int(max_steps),
        "observation_contract": expected_observation,
        "cache_metadata": cache_metadata,
    }
