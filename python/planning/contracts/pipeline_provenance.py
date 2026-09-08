"""Fail-closed provenance for the PY2 offline producer chain.

This module owns only artifact identity and cross-stage validation.  It does
not own Teacher scoring, depth-safety geometry, array dtypes, or row data.  A
producer can therefore add provenance without changing the numerical
algorithm or the existing payload layout.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from planning.common import canonical_json_sha256, file_sha256
from planning.common.config import parse_bool
from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    validate_reliable_exact_metadata,
)
from planning.contracts.task import (
    DEFAULT_MAX_PRIMITIVE_STEPS,
    task_contract_fields,
    validate_task_contract,
)


PROVENANCE_SCHEMA_VERSION = "planning_py2_producer_provenance_v1"
TEACHER_CONTRACT_ID = "global_route_bounded_beam"
DEPTH_MASK_CONTRACT_ID = "local_depth_action_masks"


# These fields are persisted in the formal collection CSV.  They are not
# optional diagnostics: a blank value means that the producer did not know the
# terminal state and must therefore fail closed before merge.
FORMAL_COLLECTION_BOOL_FIELDS = (
    "reliable_execution",
    "telemetry_observation",
    "endpoint_identity_available",
    "asynchronous_prefetch",
    "endpoint_identity_chain_valid",
    "global_collision_mask_enabled",
    "path_length_exceeded",
    "success",
    "collision",
    "dead_end",
    "hard_altitude",
    "execute_ok",
    "async_prefetch_enabled",
)

FORMAL_COLLECTION_SUMMARY_BOOL_FIELDS = (
    "reliable_execution",
    "telemetry_observation",
    "endpoint_identity_available",
    "asynchronous_prefetch",
    "endpoint_identity_chain_valid",
)


TEACHER_CONTRACT_SHA256 = canonical_json_sha256(
    {"contract_id": TEACHER_CONTRACT_ID, "schema": PROVENANCE_SCHEMA_VERSION}
)
DEPTH_MASK_CONTRACT_SHA256 = canonical_json_sha256(
    {"contract_id": DEPTH_MASK_CONTRACT_ID, "schema": PROVENANCE_SCHEMA_VERSION}
)


def _int_field(mapping: Mapping[str, Any], field: str, *, path: str, default: Any = None) -> int:
    value = mapping.get(field, default)
    if value is None or str(value).strip() == "":
        raise ValueError("{} missing {}".format(path, field))
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError("{} invalid {}={!r}".format(path, field, value)) from error
    if parsed < 0:
        raise ValueError("{} negative {}={}".format(path, field, parsed))
    return parsed


def _required_sha256(mapping: Mapping[str, Any], field: str, *, path: str) -> str:
    value = str(mapping.get(field, "")).strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("{} invalid {}".format(path, field))
    return value


def _canonical_required_bool(
    mapping: Mapping[str, Any], field: str, *, path: str
) -> bool:
    """Parse only explicit formal boolean values; never infer a blank."""

    if field not in mapping:
        raise ValueError("{} missing required boolean field {}".format(path, field))
    value = mapping[field]
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token == "true":
        return True
    if token == "false":
        return False
    raise ValueError(
        "{} invalid required boolean field {}={!r}; expected true or false".format(
            path, field, value
        )
    )


def _validate_formal_boolean_fields(
    mapping: Mapping[str, Any], *, path: str, fields: Sequence[str]
) -> Dict[str, bool]:
    return {
        field: _canonical_required_bool(mapping, field, path=path)
        for field in fields
    }


def _zero_field(mapping: Mapping[str, Any], field: str, *, path: str, required: bool = True) -> int:
    if field not in mapping or str(mapping.get(field, "")).strip() == "":
        if required:
            raise ValueError("{} missing {}".format(path, field))
        return 0
    value = _int_field(mapping, field, path=path)
    if value != 0:
        raise ValueError("{} {} must be zero, got {}".format(path, field, value))
    return value


def _validate_exact_summary(
    metadata: Mapping[str, Any],
    *,
    path: str,
    require_frame_contract_failures: bool = True,
) -> Dict[str, Any]:
    try:
        validated = validate_reliable_exact_metadata(
            metadata, path=path, require_counters=True
        )
    except ValueError:
        raise
    expected_max_steps = validated.get(
        "max_steps",
        validated.get("max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS),
    )
    validate_task_contract(
        validated,
        expected_max_primitive_steps=expected_max_steps,
        path="{} task contract".format(path),
    )
    _zero_field(validated, "legacy_rows", path=path)
    _zero_field(validated, "telemetry_lookup_count", path=path)
    _zero_field(validated, "snapshot_missing_count", path=path)
    _zero_field(validated, "state_depth_skew_max_ns", path=path)
    # The merged/root collection summary owns the complete aggregate counter.
    # Older reliable worker summaries may omit this one field; omission there
    # is not evidence of a failure, while any present non-zero value remains a
    # hard rejection.
    _zero_field(
        validated,
        "frame_contract_failures",
        path=path,
        required=require_frame_contract_failures,
    )
    _zero_field(validated, "exact_primitive_frame_count_failures", path=path, required=False)
    if not parse_bool(validated.get("endpoint_identity_chain_valid", False)):
        raise ValueError("{} endpoint identity chain is invalid".format(path))
    # ``quality_pass`` is retained for historical readers, but accepted-only
    # finalization deliberately separates runtime quality from dataset
    # quality.  A collection may have failed attempts and still publish a
    # clean accepted-only index; strict runtime consumers must inspect the
    # explicit runtime_quality_pass field instead.
    return dict(validated)


def _validate_exact_row(
    row: Mapping[str, Any], *, path: str, require_execute_ok: bool = True
) -> Dict[str, Any]:
    try:
        validated = validate_reliable_exact_metadata(row, path=path)
    except ValueError:
        raise
    expected_max_steps = validated.get(
        "max_steps",
        validated.get("max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS),
    )
    validate_task_contract(
        validated,
        expected_max_primitive_steps=expected_max_steps,
        path="{} task contract".format(path),
    )
    _int_field(validated, "reliable_rows", path=path)
    _zero_field(validated, "legacy_rows", path=path)
    _zero_field(validated, "telemetry_lookup_count", path=path)
    _zero_field(validated, "snapshot_missing_count", path=path)
    _zero_field(validated, "state_depth_skew_max_ns", path=path)
    _zero_field(validated, "frame_contract_failures", path=path)
    if not parse_bool(validated.get("endpoint_identity_chain_valid", False)):
        raise ValueError("{} endpoint identity chain is invalid".format(path))
    if require_execute_ok and not parse_bool(validated.get("execute_ok", True)):
        raise ValueError("{} is not an accepted rollout row".format(path))
    return dict(validated)


def _collection_transition_ids(row: Mapping[str, Any], *, path: str) -> List[str]:
    """Decode the persisted transition identity list from a collection row."""

    raw = row.get("transition_ids", "")
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        text = str(raw).strip()
        if not text:
            return []
        try:
            values = json.loads(text)
        except (TypeError, ValueError) as error:
            raise ValueError("{} transition_ids is not valid JSON".format(path)) from error
    if not isinstance(values, list):
        raise ValueError("{} transition_ids must be a list".format(path))
    result = [str(value).strip() for value in values]
    if any(not value for value in result):
        raise ValueError("{} transition_ids contains an empty identity".format(path))
    if len(result) != len(set(result)):
        raise ValueError("{} contains duplicate transition identities".format(path))
    return result


def validate_collection_summary(
    summary: Mapping[str, Any], *, path: str = "<collection summary>"
) -> Dict[str, Any]:
    """Validate one formal worker collection summary.

    ``frame_contract_failures`` is an aggregate field owned by the merged
    root summary.  Reliable worker summaries produced by older collectors may
    omit it, so omission is accepted here while a present non-zero value is
    still rejected.  Root-summary validation continues to use
    ``_validate_exact_summary`` directly with its required default.
    """

    validated = _validate_exact_summary(
        summary, path=path, require_frame_contract_failures=False
    )
    summary_bools = _validate_formal_boolean_fields(
        validated, path=path, fields=FORMAL_COLLECTION_SUMMARY_BOOL_FIELDS
    )
    if validated.get("observation_contract") != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("{} is not the exact observation contract".format(path))
    if validated.get("observation_source") != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("{} has a non-exact observation source".format(path))
    if summary_bools["reliable_execution"] is not True:
        raise ValueError("{} reliable_execution must be true".format(path))
    if summary_bools["telemetry_observation"] is not False:
        raise ValueError("{} telemetry_observation must be false".format(path))
    runtime_identity = validated.get("runtime_artifact_identity")
    if not isinstance(runtime_identity, Mapping):
        raise ValueError("{} runtime_artifact_identity is missing".format(path))
    for field in (
        "unity_player_sha256",
        "runtime_assembly_sha256",
        "bridge_sha256",
    ):
        summary_digest = _required_sha256(validated, field, path=path)
        identity_digest = _required_sha256(runtime_identity, field, path=path)
        if summary_digest != identity_digest:
            raise ValueError("{} {} disagrees with runtime identity".format(path, field))
    return validated


def validate_collection_row(
    row: Mapping[str, Any],
    *,
    worker_summary: Optional[Mapping[str, Any]] = None,
    path: str = "<collection row>",
) -> List[str]:
    """Validate one formal collection report row and return transition IDs."""

    # Reuse the canonical row metadata checks but allow rejected attempts to
    # remain in the report; only accepted rows must carry transitions.
    validated = _validate_exact_row(
        row, path=path, require_execute_ok=False
    )
    row_bools = _validate_formal_boolean_fields(
        validated, path=path, fields=FORMAL_COLLECTION_BOOL_FIELDS
    )
    if worker_summary is not None:
        for field in (
            "runtime_instance_id",
            "collection_run_id",
            "unity_player_sha256",
            "runtime_assembly_sha256",
            "bridge_sha256",
        ):
            if str(validated.get(field, "")).strip() != str(
                worker_summary.get(field, "")
            ).strip():
                raise ValueError("{} {} disagrees with worker summary".format(path, field))
    if not str(validated.get("mission_id", "")).strip():
        raise ValueError("{} mission_id is missing".format(path))
    reliable_rows = _int_field(validated, "reliable_rows", path=path)
    transition_ids = _collection_transition_ids(validated, path=path)
    if reliable_rows != len(transition_ids):
        raise ValueError(
            "{} reliable_rows={} but transition_ids={}".format(
                path, reliable_rows, len(transition_ids)
            )
        )
    if row_bools["execute_ok"] and not transition_ids:
        raise ValueError("{} accepted row has no transition identity".format(path))
    return transition_ids


def _metadata_from_npz(path: Path) -> Dict[str, Any]:
    try:
        with np.load(str(path), allow_pickle=False) as data:
            if "metadata_json" not in data:
                raise ValueError("{} missing metadata_json".format(path))
            value = data["metadata_json"]
            if isinstance(value, np.ndarray):
                value = value.item() if value.ndim == 0 else value.tolist()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            metadata = json.loads(str(value))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        if isinstance(error, ValueError) and str(error).endswith("missing metadata_json"):
            raise
        raise ValueError("{} invalid rollout metadata: {}".format(path, error)) from error
    if not isinstance(metadata, dict):
        raise ValueError("{} rollout metadata is not a mapping".format(path))
    return metadata


def _episode_transition_count(path: Path) -> int:
    with np.load(str(path), allow_pickle=False) as data:
        if "behavior_actions" not in data:
            raise ValueError("{} missing behavior_actions".format(path))
        behavior = np.asarray(data["behavior_actions"])
        if behavior.ndim != 1 or behavior.shape[0] <= 0:
            raise ValueError("{} behavior_actions has invalid shape".format(path))
        return int(behavior.shape[0])


def _episode_transition_ids(path: Path) -> List[str]:
    metadata = _metadata_from_npz(path)
    raw = metadata.get("transition_ids", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("{} transition_ids must be a list".format(path))
    values = [str(value).strip() for value in raw]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError("{} transition_ids are invalid or duplicated".format(path))
    return values


def validate_episode_provenance(
    metadata: Mapping[str, Any],
    *,
    path: str = "<episode>",
    expected_episode_id: Optional[int] = None,
    expected_transition_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate the exact provenance carried by one rollout episode."""

    validated = validate_reliable_exact_metadata(
        metadata, path=path, require_counters=True
    )
    expected_max_steps = validated.get(
        "max_steps",
        validated.get("max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS),
    )
    validate_task_contract(
        validated,
        expected_max_primitive_steps=expected_max_steps,
        path="{} task contract".format(path),
    )
    _zero_field(validated, "legacy_rows", path=path)
    _zero_field(validated, "telemetry_lookup_count", path=path, required=False)
    _zero_field(validated, "snapshot_missing_count", path=path, required=False)
    _zero_field(validated, "state_depth_skew_max_ns", path=path, required=False)
    _zero_field(validated, "frame_contract_failures", path=path, required=False)
    if "endpoint_identity_chain_valid" in validated and not parse_bool(
        validated["endpoint_identity_chain_valid"]
    ):
        raise ValueError("{} endpoint identity chain is invalid".format(path))
    if expected_episode_id is not None and int(validated.get("episode_id", -1)) != int(
        expected_episode_id
    ):
        raise ValueError("{} episode identity mismatch".format(path))
    if expected_transition_count is not None and _int_field(
        validated, "reliable_rows", path=path
    ) != int(expected_transition_count):
        raise ValueError("{} reliable row count mismatch".format(path))
    return dict(validated)


def _find_manifest(root: Path) -> Path:
    for name in ("rollout_manifest.json", "collection_summary.json", "manifest.json"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise ValueError("{} missing rollout manifest/collection_summary.json".format(root))


def _worker_id_from_dir(path: Path) -> int:
    match = re.fullmatch(r"worker_(\d+)", path.name)
    if match is None:
        raise ValueError("invalid worker directory {}".format(path))
    return int(match.group(1))


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("{} invalid JSON: {}".format(path, error)) from error
    if not isinstance(payload, dict):
        raise ValueError("{} JSON root is not a mapping".format(path))
    return payload


@dataclass(frozen=True)
class RolloutProvenance:
    """Validated source identity for one accepted rollout-index selection."""

    index_path: Path
    manifest_path: Path
    root_metadata: Dict[str, Any]
    rows: Tuple[Dict[str, Any], ...]
    episode_paths: Tuple[Path, ...]
    episode_metadata: Tuple[Dict[str, Any], ...]
    episode_ids: Tuple[int, ...]
    transition_offsets: Tuple[int, ...]
    transition_lengths: Tuple[int, ...]
    index_sha256: str
    manifest_sha256: str
    root_reliable_rows: int
    accepted_reliable_rows: int
    worker_ids: Tuple[int, ...]
    runtime_instance_ids: Tuple[str, ...]

    @property
    def root_path(self) -> Path:
        return self.index_path.parent

    @property
    def collection_run_id(self) -> str:
        return str(self.root_metadata.get("collection_run_id", ""))

    @property
    def mission_index_sha256(self) -> str:
        return str(self.root_metadata.get("mission_index_sha256", ""))

    @property
    def mpl_contract_sha256(self) -> str:
        return str(self.root_metadata.get("mpl_contract_sha256", ""))

    @property
    def resolved_config_sha256(self) -> str:
        return str(self.root_metadata.get("resolved_config_sha256", ""))

    def subset(self, episode_ids: Iterable[int]) -> "RolloutProvenance":
        wanted = [int(value) for value in episode_ids]
        by_id = {
            episode_id: index for index, episode_id in enumerate(self.episode_ids)
        }
        if len(wanted) != len(set(wanted)):
            raise ValueError("duplicate episode selection")
        missing = [value for value in wanted if value not in by_id]
        if missing:
            raise ValueError("selected episodes absent from rollout provenance: {}".format(missing))
        selected_rows = tuple(self.rows[by_id[value]] for value in wanted)
        selected_paths = tuple(self.episode_paths[by_id[value]] for value in wanted)
        selected_metadata = tuple(self.episode_metadata[by_id[value]] for value in wanted)
        selected_lengths = tuple(self.transition_lengths[by_id[value]] for value in wanted)
        offsets = []
        total = 0
        for length in selected_lengths:
            offsets.append(total)
            total += int(length)
        return RolloutProvenance(
            index_path=self.index_path,
            manifest_path=self.manifest_path,
            root_metadata=self.root_metadata,
            rows=selected_rows,
            episode_paths=selected_paths,
            episode_metadata=selected_metadata,
            episode_ids=tuple(wanted),
            transition_offsets=tuple(offsets),
            transition_lengths=selected_lengths,
            index_sha256=self.index_sha256,
            manifest_sha256=self.manifest_sha256,
            root_reliable_rows=self.root_reliable_rows,
            accepted_reliable_rows=total,
            worker_ids=tuple(
                _worker_id_from_dir(path.parent.parent)
                for path in selected_paths
            ),
            runtime_instance_ids=tuple(
                str(metadata.get("runtime_instance_id", ""))
                for metadata in selected_metadata
            ),
        )


def validate_rollout_provenance(
    index_path: Path,
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> RolloutProvenance:
    """Validate root, worker, index-row, and episode provenance.

    The merged ``rollout_index.csv`` is the accepted selection, while the
    root summary's ``reliable_rows`` intentionally counts every attempt.  The
    distinction is retained in the returned object for producer manifests.
    """

    index_path = Path(index_path).expanduser().resolve()
    if not index_path.is_file():
        raise ValueError("missing rollout index: {}".format(index_path))
    root = index_path.parent
    manifest_path = _find_manifest(root)
    root_metadata = _validate_exact_summary(
        _load_json(manifest_path), path=str(manifest_path)
    )
    resolved_max_steps = root_metadata.get(
        "max_steps", root_metadata.get("max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS)
    )
    declared_index = str(root_metadata.get("rollout_index", "")).strip()
    if declared_index and Path(declared_index).name != index_path.name:
        raise ValueError("{} rollout_index does not identify {}".format(manifest_path, index_path.name))

    with index_path.open("r", newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("{} is empty".format(index_path))
    if rows is not None:
        selected_paths = {str(item.get("dataset_npz", "")).strip() for item in rows}
        source_rows = [
            item for item in source_rows
            if str(item.get("dataset_npz", "")).strip() in selected_paths
        ]

    worker_dirs = sorted((root / "workers").glob("worker_*"))
    worker_dirs = [path for path in worker_dirs if path.is_dir()]
    if not worker_dirs:
        raise ValueError("{} missing worker provenance directories".format(root))
    declared_workers = root_metadata.get("workers")
    if declared_workers is not None and int(declared_workers) != len(worker_dirs):
        raise ValueError("{} worker count does not match worker directories".format(manifest_path))

    worker_summaries: Dict[int, Dict[str, Any]] = {}
    for worker_dir in worker_dirs:
        worker_id = _worker_id_from_dir(worker_dir)
        summary_path = worker_dir / "collection_summary.json"
        config_path = worker_dir / "resolved_collection_config.json"
        if not summary_path.is_file():
            raise ValueError("missing worker summary: {}".format(summary_path))
        if not config_path.is_file():
            raise ValueError("missing worker resolved config: {}".format(config_path))
        summary = _validate_exact_summary(
            _load_json(summary_path),
            path=str(summary_path),
            require_frame_contract_failures=False,
        )
        if int(summary.get("worker_id", -1)) != worker_id:
            raise ValueError("{} worker_id does not match directory".format(summary_path))
        config = _load_json(config_path)
        validate_task_contract(
            config,
            expected_max_primitive_steps=resolved_max_steps,
            path="{} task contract".format(config_path),
        )
        shared_config = config.get("shared_collection_config")
        if not isinstance(shared_config, Mapping):
            raise ValueError("{} missing shared_collection_config".format(config_path))
        validate_task_contract(
            shared_config,
            expected_max_primitive_steps=resolved_max_steps,
            path="{} shared task contract".format(config_path),
        )
        if str(config.get("resolved_config_sha256", "")) != str(
            summary.get("resolved_config_sha256", "")
        ):
            raise ValueError("{} resolved config hash mismatch".format(config_path))
        for field in (
            "collection_run_id",
            "mission_index_sha256",
            "mpl_contract_sha256",
            "task_contract_id",
            "task_contract_schema_version",
            "task_contract_sha256",
            "max_primitive_steps",
        ):
            if field in root_metadata and str(summary.get(field, "")) != str(root_metadata.get(field, "")):
                raise ValueError("{} {} disagrees with root manifest".format(summary_path, field))
        worker_summaries[worker_id] = summary

    seen_ids = set()
    seen_paths = set()
    normalized_rows: List[Dict[str, Any]] = []
    episode_paths: List[Path] = []
    episode_metadata: List[Dict[str, Any]] = []
    lengths: List[int] = []
    worker_ids: List[int] = []
    runtime_ids: List[str] = []
    for row_number, source_row in enumerate(source_rows, start=2):
        path_label = "{}:{}".format(index_path, row_number)
        row = _validate_exact_row(source_row, path=path_label)
        for field in (
            "task_contract_id",
            "task_contract_schema_version",
            "task_contract_sha256",
            "max_primitive_steps",
        ):
            if str(row.get(field, "")) != str(root_metadata.get(field, "")):
                raise ValueError("{} {} disagrees with root task contract".format(path_label, field))
        try:
            episode_id = int(float(row.get("episode_id", "")))
        except (TypeError, ValueError) as error:
            raise ValueError("{} invalid episode_id".format(path_label)) from error
        if episode_id in seen_ids:
            raise ValueError("{} duplicate episode_id {}".format(path_label, episode_id))
        seen_ids.add(episode_id)
        raw_path = str(row.get("dataset_npz", "")).strip()
        if not raw_path:
            raise ValueError("{} missing dataset_npz".format(path_label))
        episode_path = Path(raw_path).expanduser()
        if not episode_path.is_absolute():
            episode_path = (root / episode_path).resolve()
        else:
            episode_path = episode_path.resolve()
        if not episode_path.is_file():
            raise ValueError("{} missing episode artifact {}".format(path_label, episode_path))
        try:
            relative = episode_path.relative_to(root)
        except ValueError as error:
            raise ValueError("{} episode is outside rollout root".format(path_label)) from error
        if len(relative.parts) < 3 or relative.parts[0] != "workers":
            raise ValueError("{} episode is not under workers/<worker>/episodes".format(path_label))
        worker_dir = root / relative.parts[0] / relative.parts[1]
        worker_id = _worker_id_from_dir(worker_dir)
        if worker_id not in worker_summaries:
            raise ValueError("{} worker summary is missing".format(path_label))
        if str(episode_path) in seen_paths:
            raise ValueError("{} duplicate dataset_npz {}".format(path_label, episode_path))
        seen_paths.add(str(episode_path))
        worker_summary = worker_summaries[worker_id]
        for field in ("collection_run_id", "runtime_instance_id"):
            if str(row.get(field, "")).strip() != str(worker_summary.get(field, "")).strip():
                raise ValueError("{} {} disagrees with worker summary".format(path_label, field))
        metadata = _metadata_from_npz(episode_path)
        try:
            validate_reliable_exact_metadata(
                metadata, path=str(episode_path), require_counters=True
            )
        except ValueError:
            raise
        validate_task_contract(
            metadata,
            expected_max_primitive_steps=resolved_max_steps,
            path="{} task contract".format(episode_path),
        )
        _zero_field(metadata, "legacy_rows", path=str(episode_path))
        _zero_field(metadata, "telemetry_lookup_count", path=str(episode_path), required=False)
        _zero_field(metadata, "snapshot_missing_count", path=str(episode_path), required=False)
        _zero_field(metadata, "state_depth_skew_max_ns", path=str(episode_path), required=False)
        _zero_field(metadata, "frame_contract_failures", path=str(episode_path), required=False)
        if "endpoint_identity_chain_valid" in metadata and not parse_bool(metadata["endpoint_identity_chain_valid"]):
            raise ValueError("{} endpoint identity chain is invalid".format(episode_path))
        if int(metadata.get("episode_id", -1)) != episode_id:
            raise ValueError("{} episode_id disagrees with index".format(episode_path))
        for field in ("collection_run_id", "runtime_instance_id"):
            if str(metadata.get(field, "")).strip() != str(row.get(field, "")).strip():
                raise ValueError("{} {} disagrees with index".format(episode_path, field))
        for field in (
            "task_contract_id",
            "task_contract_schema_version",
            "task_contract_sha256",
            "max_primitive_steps",
        ):
            if str(metadata.get(field, "")) != str(root_metadata.get(field, "")):
                raise ValueError("{} {} disagrees with root task contract".format(episode_path, field))
        length = _episode_transition_count(episode_path)
        if _int_field(metadata, "reliable_rows", path=str(episode_path)) != length:
            raise ValueError("{} reliable_rows does not match episode transitions".format(episode_path))
        if _int_field(row, "reliable_rows", path=path_label) != length:
            raise ValueError("{} reliable_rows does not match episode transitions".format(path_label))
        transition_ids = _episode_transition_ids(episode_path)
        if transition_ids and len(transition_ids) != length:
            raise ValueError("{} transition_ids length mismatch".format(episode_path))
        row_transition_ids = str(row.get("transition_ids", "")).strip()
        if row_transition_ids:
            try:
                decoded = json.loads(row_transition_ids)
            except (TypeError, ValueError) as error:
                raise ValueError("{} transition_ids is invalid JSON".format(path_label)) from error
            if not isinstance(decoded, list) or len(decoded) != length:
                raise ValueError("{} transition_ids length mismatch".format(path_label))
        normalized_rows.append(dict(row))
        episode_paths.append(episode_path)
        episode_metadata.append(metadata)
        lengths.append(length)
        worker_ids.append(worker_id)
        runtime_ids.append(str(metadata.get("runtime_instance_id", "")))

    order = sorted(range(len(normalized_rows)), key=lambda index: int(float(normalized_rows[index]["episode_id"])))
    if order != list(range(len(order))):
        raise ValueError("{} episode order is not canonical".format(index_path))
    offsets: List[int] = []
    total = 0
    for length in lengths:
        offsets.append(total)
        total += int(length)
    return RolloutProvenance(
        index_path=index_path,
        manifest_path=manifest_path,
        root_metadata=root_metadata,
        rows=tuple(normalized_rows),
        episode_paths=tuple(episode_paths),
        episode_metadata=tuple(episode_metadata),
        episode_ids=tuple(int(float(row["episode_id"])) for row in normalized_rows),
        transition_offsets=tuple(offsets),
        transition_lengths=tuple(lengths),
        index_sha256=file_sha256(index_path),
        manifest_sha256=file_sha256(manifest_path),
        root_reliable_rows=_int_field(root_metadata, "reliable_rows", path=str(manifest_path)),
        accepted_reliable_rows=total,
        worker_ids=tuple(worker_ids),
        runtime_instance_ids=tuple(runtime_ids),
    )


def _relative_to(path: Path, base: Path) -> str:
    return os.path.relpath(str(Path(path).resolve()), str(Path(base).resolve()))


def artifact_provenance_metadata(
    provenance: RolloutProvenance,
    *,
    artifact_path: Path,
    artifact_kind: str,
    row_count: int,
    episode_ids: Sequence[int],
    transition_offsets: Sequence[int],
    transition_lengths: Sequence[int],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the common metadata block for a label or depth-mask artifact."""

    episode_ids = [int(value) for value in episode_ids]
    offsets = [int(value) for value in transition_offsets]
    lengths = [int(value) for value in transition_lengths]
    if len(episode_ids) != len(offsets) or len(offsets) != len(lengths):
        raise ValueError("artifact episode mapping length mismatch")
    if episode_ids != list(provenance.episode_ids):
        raise ValueError("artifact episode order differs from rollout provenance")
    if offsets != list(provenance.transition_offsets):
        raise ValueError("artifact transition offsets differ from rollout provenance")
    if lengths != list(provenance.transition_lengths):
        raise ValueError("artifact transition lengths differ from rollout provenance")
    if int(row_count) != sum(lengths):
        raise ValueError("artifact row count differs from episode lengths")
    artifact_path = Path(artifact_path).expanduser().resolve()
    root = provenance.root_metadata
    max_primitive_steps = root.get(
        "max_primitive_steps", root.get("max_steps", DEFAULT_MAX_PRIMITIVE_STEPS)
    )
    metadata: Dict[str, Any] = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_kind": str(artifact_kind),
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "reliable_execution": True,
        "telemetry_observation": False,
        "state_depth_skew_ns": 0,
        "state_depth_skew_max_ns": 0,
        "endpoint_identity_available": True,
        "endpoint_identity_chain_valid": True,
        "asynchronous_prefetch": False,
        "asynchronous_prefetch_status": "obsolete_for_reliable_exact",
        "reliable_rows": int(row_count),
        "legacy_rows": 0,
        "telemetry_lookup_count": 0,
        "snapshot_missing_count": 0,
        "frame_contract_failures": 0,
        "input_rollout_index": _relative_to(provenance.index_path, artifact_path.parent),
        "input_rollout_index_sha256": provenance.index_sha256,
        "input_rollout_manifest": _relative_to(provenance.manifest_path, artifact_path.parent),
        "input_rollout_manifest_sha256": provenance.manifest_sha256,
        # Explicit aliases make the manifest/summary terminology unambiguous.
        "rollout_index_sha256": provenance.index_sha256,
        "rollout_manifest_sha256": provenance.manifest_sha256,
        "rollout_root": str(provenance.root_path),
        "collection_run_id": provenance.collection_run_id,
        "input_rollout_reliable_rows": int(provenance.root_reliable_rows),
        "input_accepted_reliable_rows": int(provenance.accepted_reliable_rows),
        "input_rollout_legacy_rows": int(root.get("legacy_rows", 0)),
        "input_rollout_telemetry_lookup_count": int(root.get("telemetry_lookup_count", 0)),
        "input_rollout_snapshot_missing_count": int(root.get("snapshot_missing_count", 0)),
        "input_rollout_state_depth_skew_max_ns": int(root.get("state_depth_skew_max_ns", 0)),
        "input_rollout_frame_contract_failures": int(root.get("frame_contract_failures", 0)),
        "episode_count": len(episode_ids),
        "episodes": len(episode_ids),
        "row_count": int(row_count),
        "transition_count": int(row_count),
        "transitions": int(row_count),
        "action_count": 105,
        "episode_ids": episode_ids,
        "episode_order": episode_ids,
        "transition_offsets": offsets,
        "transition_lengths": lengths,
        "worker_ids": [int(value) for value in provenance.worker_ids],
        "runtime_instance_ids": list(provenance.runtime_instance_ids),
        "mission_index_sha256": str(root.get("mission_index_sha256", "")),
        "mission_sha256": str(root.get("mission_index_sha256", "")),
        "resolved_config_sha256": str(root.get("resolved_config_sha256", "")),
        "mpl_contract_sha256": str(root.get("mpl_contract_sha256", "")),
        "teacher_contract_id": str(root.get("teacher_planning_contract_id", TEACHER_CONTRACT_ID)),
        "teacher_contract_sha256": TEACHER_CONTRACT_SHA256,
        "teacher_planning_contract_id": str(root.get("teacher_planning_contract_id", TEACHER_CONTRACT_ID)),
        "teacher_planning_contract_sha256": TEACHER_CONTRACT_SHA256,
        **task_contract_fields(max_primitive_steps),
        "max_steps": int(max_primitive_steps),
    }
    if str(artifact_kind) == "depth_masks":
        metadata.update(
            {
                "depth_mask_contract_id": DEPTH_MASK_CONTRACT_ID,
                "depth_mask_contract_sha256": DEPTH_MASK_CONTRACT_SHA256,
                "collision_radius_m": 0.40,
                "depth_slack_m": 0.08,
                "path_sample_stride": 4,
                "max_patch_radius_px": 14,
            }
        )
    if extra:
        metadata.update(dict(extra))
    return metadata


def validate_artifact_provenance(
    metadata: Mapping[str, Any],
    *,
    artifact_kind: str,
    expected_row_count: Optional[int] = None,
    expected_episode_ids: Optional[Sequence[int]] = None,
    expected_offsets: Optional[Sequence[int]] = None,
    expected_lengths: Optional[Sequence[int]] = None,
    require_route_store: bool = False,
) -> Dict[str, Any]:
    """Validate the strict metadata emitted by PY2 producers."""

    path = "{} artifact".format(artifact_kind)
    if str(metadata.get("provenance_schema_version", "")) != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("{} missing or unknown provenance schema".format(path))
    if str(metadata.get("artifact_kind", "")) != str(artifact_kind):
        raise ValueError("{} artifact kind mismatch".format(path))
    if str(metadata.get("observation_contract", "")) != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("{} observation contract mismatch".format(path))
    expected_max_steps = metadata.get(
        "max_steps", metadata.get("max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS)
    )
    validate_task_contract(
        metadata,
        expected_max_primitive_steps=expected_max_steps,
        path="{} task contract".format(path),
    )
    _validate_exact_summary(metadata, path=path)
    for field in (
        "input_rollout_index_sha256",
        "input_rollout_manifest_sha256",
        "mission_index_sha256",
        "mission_sha256",
        "resolved_config_sha256",
        "mpl_contract_sha256",
        "teacher_contract_sha256",
        "teacher_planning_contract_sha256",
    ):
        _required_sha256(metadata, field, path=path)
    if str(metadata.get("mission_index_sha256")) != str(metadata.get("mission_sha256")):
        raise ValueError("{} mission identity mismatch".format(path))
    if str(metadata.get("teacher_contract_id", "")) != str(metadata.get("teacher_planning_contract_id", "")):
        raise ValueError("{} Teacher contract identity mismatch".format(path))
    if str(metadata.get("teacher_contract_sha256", "")) != TEACHER_CONTRACT_SHA256:
        raise ValueError("{} Teacher contract hash mismatch".format(path))
    if str(metadata.get("teacher_planning_contract_sha256", "")) != TEACHER_CONTRACT_SHA256:
        raise ValueError("{} Teacher planning contract hash mismatch".format(path))
    if int(metadata.get("action_count", -1)) != 105:
        raise ValueError("{} action count mismatch".format(path))
    row_count = _int_field(metadata, "row_count", path=path)
    if row_count != _int_field(metadata, "transition_count", path=path):
        raise ValueError("{} row/transition count mismatch".format(path))
    if row_count != _int_field(metadata, "transitions", path=path):
        raise ValueError("{} row/transitions count mismatch".format(path))
    if expected_row_count is not None and row_count != int(expected_row_count):
        raise ValueError("{} row count {} != {}".format(path, row_count, expected_row_count))
    episode_ids = [int(value) for value in metadata.get("episode_order", [])]
    if episode_ids != [int(value) for value in metadata.get("episode_ids", [])]:
        raise ValueError("{} episode order/identity mismatch".format(path))
    offsets = [int(value) for value in metadata.get("transition_offsets", [])]
    lengths = [int(value) for value in metadata.get("transition_lengths", [])]
    if len(episode_ids) != int(metadata.get("episode_count", -1)):
        raise ValueError("{} episode count mismatch".format(path))
    if len(episode_ids) != len(offsets) or len(offsets) != len(lengths):
        raise ValueError("{} transition mapping mismatch".format(path))
    if sum(lengths) != row_count:
        raise ValueError("{} transition lengths do not sum to row count".format(path))
    if expected_episode_ids is not None and episode_ids != [int(value) for value in expected_episode_ids]:
        raise ValueError("{} episode identity mismatch".format(path))
    if expected_offsets is not None and offsets != [int(value) for value in expected_offsets]:
        raise ValueError("{} transition offset mismatch".format(path))
    if expected_lengths is not None and lengths != [int(value) for value in expected_lengths]:
        raise ValueError("{} transition length mismatch".format(path))
    if artifact_kind == "teacher_labels":
        if not str(metadata.get("teacher_contract_id", "")).strip():
            raise ValueError("{} missing Teacher contract identity".format(path))
    elif artifact_kind == "depth_masks":
        if str(metadata.get("depth_mask_contract_id", "")) != DEPTH_MASK_CONTRACT_ID:
            raise ValueError("{} depth-mask contract mismatch".format(path))
        if str(metadata.get("depth_mask_contract_sha256", "")) != DEPTH_MASK_CONTRACT_SHA256:
            raise ValueError("{} depth-mask contract hash mismatch".format(path))
        for field in (
            "collision_radius_m",
            "depth_slack_m",
            "path_sample_stride",
            "max_patch_radius_px",
        ):
            if field not in metadata:
                raise ValueError("{} missing {}".format(path, field))
    route_fields = (
        "mission_route_store_contract_id",
        "mission_route_store_schema_version",
        "mission_route_store",
        "mission_route_store_candidate_index_sha256",
        "mission_route_store_points_sha256",
        "mission_route_store_offsets_sha256",
    )
    if require_route_store or any(field in metadata for field in route_fields):
        missing_route = [field for field in route_fields if not str(metadata.get(field, "")).strip()]
        if missing_route:
            raise ValueError("{} missing route-store provenance: {}".format(path, ",".join(missing_route)))
        if int(metadata["mission_route_store_schema_version"]) != 1:
            raise ValueError("{} route-store schema mismatch".format(path))
        for field in (
            "mission_route_store_candidate_index_sha256",
            "mission_route_store_points_sha256",
            "mission_route_store_offsets_sha256",
        ):
            _required_sha256(metadata, field, path=path)
    return dict(metadata)


def validate_cross_artifact_consistency(
    provenance: RolloutProvenance,
    labels_metadata: Mapping[str, Any],
    masks_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Require rollout, labels, and masks to describe one exact row domain."""

    expected = {
        "row_count": provenance.accepted_reliable_rows,
        "episode_ids": provenance.episode_ids,
        "offsets": provenance.transition_offsets,
        "lengths": provenance.transition_lengths,
    }
    labels = validate_artifact_provenance(
        labels_metadata,
        artifact_kind="teacher_labels",
        expected_row_count=expected["row_count"],
        expected_episode_ids=expected["episode_ids"],
        expected_offsets=expected["offsets"],
        expected_lengths=expected["lengths"],
    )
    masks = validate_artifact_provenance(
        masks_metadata,
        artifact_kind="depth_masks",
        expected_row_count=expected["row_count"],
        expected_episode_ids=expected["episode_ids"],
        expected_offsets=expected["offsets"],
        expected_lengths=expected["lengths"],
    )
    for field in (
        "observation_contract",
        "observation_source",
        "input_rollout_index_sha256",
        "input_rollout_manifest_sha256",
        "mission_sha256",
        "resolved_config_sha256",
        "mpl_contract_sha256",
        "collection_run_id",
        "task_contract_id",
        "task_contract_schema_version",
        "task_contract_sha256",
        "max_primitive_steps",
    ):
        if str(labels.get(field, "")) != str(masks.get(field, "")):
            raise ValueError("labels/depth masks {} mismatch".format(field))
    if labels["input_rollout_index_sha256"] != provenance.index_sha256:
        raise ValueError("labels input rollout index hash mismatch")
    if labels["input_rollout_manifest_sha256"] != provenance.manifest_sha256:
        raise ValueError("labels input rollout manifest hash mismatch")
    if masks["input_rollout_index_sha256"] != provenance.index_sha256:
        raise ValueError("depth masks input rollout index hash mismatch")
    if masks["input_rollout_manifest_sha256"] != provenance.manifest_sha256:
        raise ValueError("depth masks input rollout manifest hash mismatch")
    for field, expected_value in (
        ("mission_sha256", provenance.mission_index_sha256),
        ("resolved_config_sha256", provenance.resolved_config_sha256),
        ("mpl_contract_sha256", provenance.mpl_contract_sha256),
        ("collection_run_id", provenance.collection_run_id),
        ("task_contract_id", provenance.root_metadata.get("task_contract_id", "")),
        (
            "task_contract_schema_version",
            provenance.root_metadata.get("task_contract_schema_version", ""),
        ),
        ("task_contract_sha256", provenance.root_metadata.get("task_contract_sha256", "")),
        ("max_primitive_steps", provenance.root_metadata.get("max_primitive_steps", "")),
    ):
        if str(labels.get(field, "")) != str(expected_value):
            raise ValueError("labels {} disagrees with rollout provenance".format(field))
        if str(masks.get(field, "")) != str(expected_value):
            raise ValueError("depth masks {} disagrees with rollout provenance".format(field))
    expected_teacher_id = str(
        provenance.root_metadata.get("teacher_planning_contract_id", TEACHER_CONTRACT_ID)
    )
    if str(labels.get("teacher_contract_id", "")) != expected_teacher_id:
        raise ValueError("labels Teacher contract identity disagrees with rollout provenance")
    route_fields = (
        "mission_route_store_contract_id",
        "mission_route_store_schema_version",
        "mission_route_store",
        "mission_route_store_candidate_index_sha256",
        "mission_route_store_points_sha256",
        "mission_route_store_offsets_sha256",
    )
    label_route = any(field in labels for field in route_fields)
    mask_route = any(field in masks for field in route_fields)
    if label_route != mask_route:
        raise ValueError("labels/depth masks route-store provenance presence mismatch")
    if label_route:
        for field in route_fields:
            if str(labels.get(field, "")) != str(masks.get(field, "")):
                raise ValueError("labels/depth masks {} mismatch".format(field))
    result = {
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "row_count": int(expected["row_count"]),
        "episode_count": len(expected["episode_ids"]),
        "episode_ids": list(expected["episode_ids"]),
        "transition_offsets": list(expected["offsets"]),
        "transition_lengths": list(expected["lengths"]),
        "action_count": 105,
        "input_rollout_index_sha256": provenance.index_sha256,
        "input_rollout_manifest_sha256": provenance.manifest_sha256,
        "mission_sha256": provenance.mission_index_sha256,
        "resolved_config_sha256": provenance.resolved_config_sha256,
        "mpl_contract_sha256": provenance.mpl_contract_sha256,
        "collection_run_id": provenance.collection_run_id,
        "task_contract_id": provenance.root_metadata["task_contract_id"],
        "task_contract_schema_version": int(
            provenance.root_metadata["task_contract_schema_version"]
        ),
        "task_contract_sha256": provenance.root_metadata["task_contract_sha256"],
        "max_primitive_steps": int(provenance.root_metadata["max_primitive_steps"]),
        "max_steps": int(
            provenance.root_metadata.get(
                "max_steps", provenance.root_metadata["max_primitive_steps"]
            )
        ),
    }
    if label_route:
        result.update({field: labels[field] for field in route_fields})
    return result


__all__ = [
    "FORMAL_COLLECTION_BOOL_FIELDS",
    "FORMAL_COLLECTION_SUMMARY_BOOL_FIELDS",
    "DEPTH_MASK_CONTRACT_ID",
    "DEPTH_MASK_CONTRACT_SHA256",
    "PROVENANCE_SCHEMA_VERSION",
    "RolloutProvenance",
    "TEACHER_CONTRACT_ID",
    "TEACHER_CONTRACT_SHA256",
    "artifact_provenance_metadata",
    "validate_artifact_provenance",
    "validate_cross_artifact_consistency",
    "validate_episode_provenance",
    "validate_rollout_provenance",
]
