"""Resolved configuration contract for formal teacher rollout collection."""

from __future__ import annotations

import hashlib
import os
import copy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Mapping

from planning.common.paths import planning_package_root
from planning.common import (
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
)
from planning.protocol.constants import PROTOCOL_VERSION
from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    LEGACY_ASYNC_OBSERVATION_CONTRACT,
    exact_endpoint_metadata,
)
from planning.mission.global_route import GlobalRouteConfig, global_route_identity
from planning.contracts.task import DEFAULT_MAX_PRIMITIVE_STEPS, task_contract_fields


COLLECTION_CONFIG_CONTRACT_ID = "teacher_rollout_collection_resolved_config"
HASH_ENVIRONMENT_FIELDS = {
    "mission_index_sha256": "PLANNING_MISSION_INDEX_SHA256",
    "collision_cache_sha256": "PLANNING_COLLISION_CACHE_SHA256",
    "code_version_sha256": "PLANNING_COLLECTION_CODE_SHA256",
}
WORKER_SPECIFIC_CLI_FIELDS = frozenset(
    {
        "out_dir",
        "worker_id",
        "stop_file",
        "runtime_instance_id",
        "reliable_v4_runtime_instance_id",
        "reliable_exact_runtime_instance_id",
        "reliable_v4_command_endpoint",
        "reliable_v4_result_endpoint",
        "reliable_v4_snapshot_endpoint",
        "reliable_exact_command_endpoint",
        "reliable_exact_result_endpoint",
        "reliable_exact_snapshot_endpoint",
    }
)
PATH_CLI_FIELDS = frozenset(
    {"collision_cache", "index", "out_dir", "stop_file"}
)
SOURCE_SUFFIXES = frozenset(
    {".cpp", ".cu", ".hpp", ".launch", ".py", ".sh", ".xml", ".yaml"}
)
SOURCE_TOP_LEVEL_FILES = frozenset(
    {"CMakeLists.txt", "package.xml", "setup.py"}
)
SOURCE_DIRECTORIES = (
    "config",
    "include",
    "launch",
    "python/planning",
    "scripts",
    "src",
)

def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a collection contract identically for logs and hashes."""

    return canonical_json_bytes(payload, ensure_ascii=False).decode("utf-8")


canonical_sha256 = partial(canonical_json_sha256, ensure_ascii=False)


def _relative_path(path_like: Any, package_root: Path) -> str:
    if not str(path_like).strip():
        return ""
    path = Path(str(path_like)).expanduser()
    resolved = path.resolve() if path.is_absolute() else (package_root / path).resolve()
    return os.path.relpath(str(resolved), str(package_root))


def resolved_cli_arguments(args, package_root: Path = None) -> Dict[str, Any]:
    """Return every final argparse value with persisted paths made relative."""

    root = Path(package_root or planning_package_root()).resolve()
    resolved: Dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if key in PATH_CLI_FIELDS:
            resolved[key] = _relative_path(value, root)
        elif isinstance(value, tuple):
            resolved[key] = list(value)
        else:
            resolved[key] = value
    return resolved


def source_tree_sha256(package_root: Path = None) -> str:
    """Hash collection-relevant source content, including relative filenames."""

    root = Path(package_root or planning_package_root()).resolve()
    files = []
    for filename in SOURCE_TOP_LEVEL_FILES:
        path = root / filename
        if path.is_file():
            files.append(path)
    for directory in SOURCE_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            continue
        files.extend(
            path
            for path in base.rglob("*")
            if path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and "__pycache__" not in path.parts
        )
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_input_sha256(path: Path, environment_name: str) -> str:
    """Use a launcher-computed hash or compute it for a direct collector run."""

    supplied = str(os.environ.get(environment_name, "")).strip().lower()
    if supplied:
        if len(supplied) != 64 or any(
            character not in "0123456789abcdef" for character in supplied
        ):
            raise ValueError(
                "{} must be a lowercase SHA256 digest".format(environment_name)
            )
        return supplied
    return file_sha256(Path(path))


def resolve_source_tree_sha256() -> str:
    """Use the launcher's code digest, with direct CLI execution as fallback."""

    environment_name = HASH_ENVIRONMENT_FIELDS["code_version_sha256"]
    supplied = str(os.environ.get(environment_name, "")).strip().lower()
    if supplied:
        if len(supplied) != 64 or any(
            character not in "0123456789abcdef" for character in supplied
        ):
            raise ValueError(
                "{} must be a lowercase SHA256 digest".format(environment_name)
            )
        return supplied
    return source_tree_sha256()


def merge_compatibility_sha256(shared_collection_config: Mapping[str, Any]) -> str:
    """Hash only worker-independent collection semantics.

    Runtime identities and endpoints identify a worker instance, but do not
    change the data/algorithm contract used to merge its shard.  Keep this
    projection explicit so a worker-specific endpoint cannot silently split a
    valid aggregate run while semantic settings still fail closed.
    """

    projected = copy.deepcopy(dict(shared_collection_config))
    for key in (
        "worker_id",
        "runtime_instance_id",
        "reliable_v4_runtime_instance_id",
        "reliable_exact_runtime_instance_id",
        "reliable_v4_command_endpoint",
        "reliable_v4_result_endpoint",
        "reliable_v4_snapshot_endpoint",
        "reliable_exact_command_endpoint",
        "reliable_exact_result_endpoint",
        "reliable_exact_snapshot_endpoint",
    ):
        projected.pop(key, None)
    cli = projected.get("resolved_cli_args")
    if isinstance(cli, Mapping):
        projected["resolved_cli_args"] = {
            key: value
            for key, value in cli.items()
            if key not in WORKER_SPECIFIC_CLI_FIELDS
        }
    return canonical_sha256(projected)


def build_resolved_collection_config(
    args,
    *,
    mission_index_sha256: str,
    collision_cache_sha256: str,
    code_version_sha256: str,
    teacher_config: Mapping[str, Any],
    mpl_contract_sha256: str,
    async_prefetch_enabled: bool,
    observation_contract: str = None,
    observation_source: str = None,
    reliable_execution: bool = None,
    runtime_artifact_identity: Mapping[str, Any] = None,
    route_store_identity: Mapping[str, Any] = None,
    package_root: Path = None,
) -> Dict[str, Any]:
    """Build full worker config plus the worker-independent merge contract."""

    root = Path(package_root or planning_package_root()).resolve()
    # Existing direct callers may still describe the diagnostic legacy path.
    # The formal collector passes the exact contract explicitly; no formal
    # artifact is allowed to rely on this compatibility inference.
    if observation_contract is None:
        observation_contract = (
            LEGACY_ASYNC_OBSERVATION_CONTRACT
            if bool(async_prefetch_enabled)
            else EXACT_ENDPOINT_OBSERVATION_CONTRACT
        )
    observation_contract = str(observation_contract).strip()
    if observation_contract not in (
        EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        LEGACY_ASYNC_OBSERVATION_CONTRACT,
    ):
        raise ValueError("unknown observation contract: {}".format(observation_contract))
    observation_source = (
        observation_contract
        if observation_source is None
        else str(observation_source).strip()
    )
    if observation_source != observation_contract:
        raise ValueError(
            "observation contract/source mismatch: {} != {}".format(
                observation_contract, observation_source
            )
        )
    if observation_contract == EXACT_ENDPOINT_OBSERVATION_CONTRACT and bool(
        async_prefetch_enabled
    ):
        raise ValueError("reliable-exact collection cannot enable async prefetch")
    if reliable_execution is None:
        reliable_execution = observation_contract == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    all_cli = resolved_cli_arguments(args, root)
    shared_cli = {
        key: value
        for key, value in all_cli.items()
        if key not in WORKER_SPECIFIC_CLI_FIELDS
    }
    launcher_context = {
        "run_id": str(
            os.environ.get("PLANNING_COLLECTION_RUN_ID", "")
        ).strip(),
        "workers_requested": int(
            os.environ.get("PLANNING_COLLECTION_WORKERS_REQUESTED", args.num_workers)
        ),
        "global_max_episodes": int(
            os.environ.get("PLANNING_COLLECTION_GLOBAL_MAX_EPISODES", 0)
        ),
        "global_target_accepted": int(
            os.environ.get("PLANNING_COLLECTION_GLOBAL_TARGET_ACCEPTED", 0)
        ),
        "collision_threads_per_worker": int(
            os.environ.get("PLANNING_COLLISION_THREADS", 1)
        ),
        "unity_window_width": int(
            os.environ.get("PLANNING_UNITY_WINDOW_WIDTH", 0)
        ),
        "unity_window_height": int(
            os.environ.get("PLANNING_UNITY_WINDOW_HEIGHT", 0)
        ),
    }
    route_defaults = GlobalRouteConfig()
    route_identity = global_route_identity(
        GlobalRouteConfig(
            resolution_m=float(
                teacher_config.get(
                    "global_route_resolution_m", route_defaults.resolution_m
                )
            ),
            flight_z_min_m=float(
                teacher_config.get("z_min", route_defaults.flight_z_min_m)
            ),
            flight_z_max_m=float(
                teacher_config.get("z_max", route_defaults.flight_z_max_m)
            ),
            lookahead_m=float(
                teacher_config.get(
                    "global_route_lookahead_m", route_defaults.lookahead_m
                )
            ),
            tracking_margin_m=float(
                teacher_config.get(
                    "global_route_tracking_margin_m",
                    route_defaults.tracking_margin_m,
                )
            ),
        )
    )
    route_identity["global_route_map_identity"] = {
        "cache_sha256": str(collision_cache_sha256),
        "voxel_size_m": float(getattr(args, "voxel_size", 0.0)),
        "inflate_radius_m": float(getattr(args, "inflate_radius", 0.0)),
    }
    shared = {
        "collection_config_contract_id": COLLECTION_CONFIG_CONTRACT_ID,
        "resolved_cli_args": shared_cli,
        "launcher_context": launcher_context,
        "mission_index_sha256": str(mission_index_sha256),
        "collision_cache_sha256": str(collision_cache_sha256),
        "code_version_sha256": str(code_version_sha256),
        "teacher_config": dict(teacher_config),
        "mpl_contract_sha256": str(mpl_contract_sha256),
        "asynchronous_prefetch": bool(async_prefetch_enabled),
        "observation_contract": observation_contract,
        "observation_source": observation_source,
        "reliable_execution": bool(reliable_execution),
        "telemetry_observation": observation_contract
        != EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "protocol_version": PROTOCOL_VERSION
        if observation_contract == EXACT_ENDPOINT_OBSERVATION_CONTRACT
        else 3,
        **task_contract_fields(
            int(getattr(args, "max_steps", DEFAULT_MAX_PRIMITIVE_STEPS))
        ),
    }
    if observation_contract == EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        shared.update(exact_endpoint_metadata())
        if runtime_artifact_identity is not None:
            shared["runtime_artifact_identity"] = dict(runtime_artifact_identity)
    shared.update(route_identity)
    if route_store_identity is not None:
        shared["mission_route_store"] = dict(route_store_identity)
    return {
        "collection_config_contract_id": COLLECTION_CONFIG_CONTRACT_ID,
        "resolved_config_sha256": canonical_sha256(shared),
        "merge_compatibility_sha256": merge_compatibility_sha256(shared),
        "shared_collection_config": shared,
        "resolved_cli_args": all_cli,
        **task_contract_fields(
            int(getattr(args, "max_steps", DEFAULT_MAX_PRIMITIVE_STEPS))
        ),
        "worker_runtime": {
            key: all_cli[key]
            for key in sorted(WORKER_SPECIFIC_CLI_FIELDS)
            if key in all_cli
        },
    }
