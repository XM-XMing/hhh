"""Disk-backed behavior-cloning dataset with an auditable source contract.

The cache keeps the large depth tensor in ``.npy`` files opened through
``numpy.memmap``.  Training workers therefore share the kernel page cache
instead of each materializing and concatenating every rollout in RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import numpy as np

from planning.data.rollout import (
    TeacherLabelStore,
    load_rollout_episode,
    resolve_dataset_path,
)
from planning.safety.depth_mask import DepthActionMaskStore
from planning.contracts.feature import CONTINUOUS_DIM, NUM_ACTIONS, POLICY_VECTOR_DIM
from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    observation_contract,
)
from planning.contracts.pipeline_provenance import (
    validate_artifact_provenance,
    validate_cross_artifact_consistency,
    validate_rollout_provenance,
)
from planning.contracts.task import (
    DEFAULT_MAX_PRIMITIVE_STEPS,
    task_contract_fields,
    validate_task_contract,
)
from planning.common import file_sha256, read_csv, read_json, write_json_atomic

BC_MMAP_DATASET_CONTRACT_ID = "bc_mmap_dataset"


def read_episode_ids_file(path: Path) -> List[int]:
    """Read a deterministic episode selection from text or a small JSON file."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("missing episode selection: {}".format(path))
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            values = payload.get("episode_ids")
            if values is None:
                values = payload.get("all_episode_ids")
        else:
            values = payload
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("episode selection JSON must contain episode_ids")
    else:
        values = path.read_text(encoding="utf-8").replace(",", " ").split()
    try:
        result = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("episode selection contains a non-integer id") from error
    if not result:
        raise ValueError("episode selection is empty: {}".format(path))
    if any(value < 0 for value in result):
        raise ValueError("episode selection contains a negative id")
    if len(result) != len(set(result)):
        raise ValueError("episode selection contains duplicate ids")
    return result

def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(str(Path(path).resolve()), str(Path(base).resolve()))

def _open_array(directory: Path, name: str, dtype, shape):
    return np.lib.format.open_memmap(
        str(directory / (name + ".npy")), mode="w+", dtype=dtype, shape=shape
    )

def build_bc_mmap_dataset(
    rows: List[Dict],
    index_path: Path,
    labels: TeacherLabelStore,
    out_dir: Path,
    depth_masks: Optional[DepthActionMaskStore] = None,
    overwrite: bool = False,
    expected_observation_contract: str = EXACT_ENDPOINT_OBSERVATION_CONTRACT,
) -> Dict:
    """Build one immutable mmap cache from validated rollout episodes."""

    index_path = Path(index_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    building = out_dir.with_name(out_dir.name + ".building")
    if out_dir.exists() and not overwrite:
        raise FileExistsError("dataset cache already exists: {}".format(out_dir))
    if building.exists():
        shutil.rmtree(str(building))
    building.mkdir(parents=True)

    entries = []
    total = 0
    source_contracts = set()
    source_task_contracts = set()
    source_task_metadata = None
    for row in rows:
        rollout_path = resolve_dataset_path(row, index_path)
        key = str(rollout_path)
        if key not in labels.path_to_slice:
            raise KeyError("rollout missing from labels: {}".format(rollout_path))
        _, length = labels.path_to_slice[key]
        if int(length) <= 0:
            raise ValueError("rollout has no labeled transitions: {}".format(rollout_path))
        entries.append(
            {
                "episode_id": int(float(row.get("episode_id", -1))),
                "dataset_npz": _relative(rollout_path, index_path.parent),
                "offset": int(total),
                "length": int(length),
            }
        )
        total += int(length)
    if total <= 0:
        raise RuntimeError("cannot build an empty BC dataset cache")

    # Label/mask provenance already carries the deterministic episode
    # lengths.  Load the first episode once to establish the depth shape, then
    # validate and block-write each remaining episode exactly once.  The old
    # implementation opened every episode in a metadata pre-pass and again in
    # the block-write pass without changing any validation result.
    first_path = (index_path.parent / entries[0]["dataset_npz"]).resolve()
    first = load_rollout_episode(first_path, validate=True)
    depth_shape = tuple(np.asarray(first["depths"]).shape[1:])
    if len(depth_shape) != 2:
        raise ValueError("depth frames must be [H,W], got {}".format(depth_shape))

    # The first episode supplies the shape and the source contract before
    # artifact-level provenance checks.  It is retained in memory for the
    # block-write loop below, so it is not opened a second time.
    first_metadata = first["metadata"]
    first_contract = observation_contract(first_metadata, path=str(first_path))
    first_max_steps = first_metadata.get(
        "max_steps",
        first_metadata.get("max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS),
    )
    validate_task_contract(
        first_metadata,
        expected_max_primitive_steps=first_max_steps,
        path="{} task contract".format(first_path),
    )
    source_task_metadata = first_metadata
    source_contracts.add(first_contract)
    source_task_contracts.add(
        (
            str(first_metadata.get("task_contract_id", "")),
            int(first_metadata.get("task_contract_schema_version", -1)),
            str(first_metadata.get("task_contract_sha256", "")),
            int(first_metadata.get("max_primitive_steps", -1)),
        )
    )

    if len(source_contracts) != 1:
        raise ValueError(
            "BC dataset mixes observation contracts: {}".format(
                sorted(source_contracts)
            )
        )
    if len(source_task_contracts) != 1 or source_task_metadata is None:
        raise ValueError("BC dataset mixes task contracts")
    source_task_fields = task_contract_fields(
        int(source_task_metadata["max_primitive_steps"])
    )
    source_observation_contract = next(iter(source_contracts))
    expected_contract = str(expected_observation_contract).strip()
    if expected_contract != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError(
            "BC mmap requires observation contract {}".format(
                EXACT_ENDPOINT_OBSERVATION_CONTRACT
            )
        )
    if source_observation_contract != expected_contract:
        raise ValueError(
            "BC dataset requires observation contract {}: got {}".format(
                expected_contract, source_observation_contract
            )
        )
    label_observation_contract = observation_contract(
        labels.metadata, path=str(labels.path)
    )
    if label_observation_contract != source_observation_contract:
        raise ValueError(
            "label observation contract {} != rollout contract {}".format(
                label_observation_contract, source_observation_contract
            )
        )
    if depth_masks is None:
        raise ValueError("depth mask observation contract is missing")
    mask_observation_contract = observation_contract(
        depth_masks.metadata, path=str(depth_masks.path)
    )
    if mask_observation_contract != source_observation_contract:
        raise ValueError(
            "depth-mask observation contract {} != rollout contract {}".format(
                mask_observation_contract, source_observation_contract
            )
        )
    for artifact_name, artifact_metadata in (
        ("labels", labels.metadata),
        ("depth masks", depth_masks.metadata),
    ):
        validate_task_contract(
            artifact_metadata,
            expected_max_primitive_steps=int(source_task_metadata["max_primitive_steps"]),
            path="{} task contract".format(artifact_name),
        )
        for field in (
            "task_contract_id",
            "task_contract_schema_version",
            "task_contract_sha256",
            "max_primitive_steps",
        ):
            if str(artifact_metadata.get(field, "")) != str(
                source_task_metadata.get(field, "")
            ):
                raise ValueError("{} {} mismatch with rollout".format(artifact_name, field))

    labels_have_producer_provenance = (
        "provenance_schema_version" in labels.metadata
    )
    masks_have_producer_provenance = (
        "provenance_schema_version" in depth_masks.metadata
    )
    if labels_have_producer_provenance != masks_have_producer_provenance:
        raise ValueError("labels and depth masks have inconsistent producer provenance")
    producer_provenance = {}
    source_episode_count = len(entries)
    source_transition_count = int(total)
    selected_episode_ids = [int(entry["episode_id"]) for entry in entries]
    if labels_have_producer_provenance:
        label_provenance = validate_artifact_provenance(
            labels.metadata, artifact_kind="teacher_labels"
        )
        mask_provenance = validate_artifact_provenance(
            depth_masks.metadata, artifact_kind="depth_masks"
        )
        source_episode_ids = [int(value) for value in labels.metadata.get("episode_order", [])]
        if source_episode_ids != [int(value) for value in depth_masks.metadata.get("episode_order", [])]:
            raise ValueError("labels/depth masks source episode order mismatch")
        if not set(selected_episode_ids).issubset(set(source_episode_ids)):
            raise ValueError("selected episodes are absent from producer provenance")
        source_positions = {value: index for index, value in enumerate(source_episode_ids)}
        selected_positions = [source_positions[value] for value in selected_episode_ids]
        if selected_positions != sorted(selected_positions):
            raise ValueError("selected episode order is not canonical")
        source_episode_count = len(source_episode_ids)
        source_transition_count = int(labels.metadata["row_count"])
        producer_provenance = label_provenance
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
            if str(labels.metadata.get(field, "")) != str(depth_masks.metadata.get(field, "")):
                raise ValueError("labels/depth masks {} mismatch".format(field))
        if producer_provenance["input_rollout_index_sha256"] != file_sha256(index_path):
            raise ValueError("producer provenance input rollout index hash mismatch")

    arrays = {
        "depths": _open_array(building, "depths", np.float16, (total,) + depth_shape),
        "continuous": _open_array(building, "continuous", np.float32, (total, CONTINUOUS_DIM)),
        "prev_actions": _open_array(building, "prev_actions", np.int16, (total,)),
        "height_masks": _open_array(building, "height_masks", np.bool_, (total, NUM_ACTIONS)),
        "local_depth_masks": _open_array(building, "local_depth_masks", np.bool_, (total, NUM_ACTIONS)),
        "behavior_actions": _open_array(building, "behavior_actions", np.int16, (total,)),
        "teacher_actions": _open_array(building, "teacher_actions", np.int16, (total,)),
        "soft_targets": _open_array(building, "soft_targets", np.float32, (total, NUM_ACTIONS)),
        "history_starts": _open_array(building, "history_starts", np.int64, (total,)),
        "episode_ids": _open_array(building, "episode_ids", np.int64, (total,)),
    }

    for entry_index, entry in enumerate(entries):
        rollout_path = (index_path.parent / entry["dataset_npz"]).resolve()
        # ``first`` was validated above and is reused here; every other
        # episode has exactly one full validated load in this build.
        episode = first if entry_index == 0 else load_rollout_episode(
            rollout_path, validate=True
        )
        if entry_index != 0:
            source_metadata = episode["metadata"]
            source_contract = observation_contract(
                source_metadata, path=str(rollout_path)
            )
            source_max_steps = source_metadata.get(
                "max_steps",
                source_metadata.get(
                    "max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS
                ),
            )
            validate_task_contract(
                source_metadata,
                expected_max_primitive_steps=source_max_steps,
                path="{} task contract".format(rollout_path),
            )
            source_contracts.add(source_contract)
            source_task_contracts.add(
                (
                    str(source_metadata.get("task_contract_id", "")),
                    int(source_metadata.get("task_contract_schema_version", -1)),
                    str(source_metadata.get("task_contract_sha256", "")),
                    int(source_metadata.get("max_primitive_steps", -1)),
                )
            )
        behavior = np.asarray(episode["behavior_actions"], dtype=np.int64)
        label = labels.get_episode(rollout_path, expected_behavior_actions=behavior)
        length = int(entry["length"])
        if int(behavior.shape[0]) != length:
            raise ValueError("rollout/label length changed: {}".format(rollout_path))
        offset = int(entry["offset"])
        sl = slice(offset, offset + length)
        arrays["depths"][sl] = np.asarray(episode["depths"], dtype=np.float16)
        arrays["continuous"][sl] = np.concatenate(
            [
                np.asarray(episode["states"], dtype=np.float32),
                np.asarray(episode["goals"], dtype=np.float32),
            ],
            axis=1,
        )
        arrays["prev_actions"][sl] = np.asarray(episode["prev_actions"], dtype=np.int16)
        arrays["height_masks"][sl] = np.asarray(episode["height_action_masks"], dtype=np.bool_)
        arrays["local_depth_masks"][sl] = depth_masks.get_episode(rollout_path, length)
        arrays["behavior_actions"][sl] = behavior.astype(np.int16)
        arrays["teacher_actions"][sl] = np.asarray(label["teacher_argmax"], dtype=np.int16)
        arrays["soft_targets"][sl] = np.asarray(label["soft_targets"], dtype=np.float32)
        arrays["history_starts"][sl] = offset
        arrays["episode_ids"][sl] = int(entry["episode_id"])

    if len(source_contracts) != 1:
        raise ValueError(
            "BC dataset mixes observation contracts: {}".format(
                sorted(source_contracts)
            )
        )
    if len(source_task_contracts) != 1:
        raise ValueError("BC dataset mixes task contracts")

    for array in arrays.values():
        array.flush()
    manifest = {
        "contract_id": BC_MMAP_DATASET_CONTRACT_ID,
        "transition_count": int(total),
        "episode_count": len(entries),
        "depth_shape": list(depth_shape),
        "num_actions": NUM_ACTIONS,
        "continuous_dim": CONTINUOUS_DIM,
        "source_index": _relative(index_path, out_dir.parent),
        "source_index_sha256": file_sha256(index_path),
        "source_labels": _relative(labels.path, out_dir.parent),
        "source_labels_sha256": file_sha256(labels.path),
        "teacher_label_contract_id": str(labels.metadata.get("label_contract_id", "")),
        "mpl_contract_sha256": str(labels.metadata.get("mpl_contract_sha256", "")),
        "source_depth_masks": (
            _relative(depth_masks.path, out_dir.parent) if depth_masks is not None else ""
        ),
        "source_depth_masks_sha256": (
            file_sha256(depth_masks.path) if depth_masks is not None else ""
        ),
        "observation_contract": source_observation_contract,
        "observation_source": source_observation_contract,
        **source_task_fields,
        "max_steps": int(source_task_metadata["max_primitive_steps"]),
        "reliable_rows": int(
            total
            if source_observation_contract == EXACT_ENDPOINT_OBSERVATION_CONTRACT
            else 0
        ),
        "legacy_rows": int(
            total
            if source_observation_contract != EXACT_ENDPOINT_OBSERVATION_CONTRACT
            else 0
        ),
        "source_episode_count": int(source_episode_count),
        "source_transition_count": int(source_transition_count),
        "selected_episode_count": len(entries),
        "selected_transition_count": int(total),
        "selection_kind": (
            "canonical_episode_id_subset"
            if len(entries) != int(source_episode_count)
            else "full_source"
        ),
        "episodes": entries,
    }
    if producer_provenance:
        for key in (
            "provenance_schema_version",
            "artifact_kind",
            "input_rollout_index",
            "input_rollout_index_sha256",
            "input_rollout_manifest",
            "input_rollout_manifest_sha256",
            "rollout_index_sha256",
            "rollout_manifest_sha256",
            "collection_run_id",
            "mission_index_sha256",
            "mission_sha256",
            "resolved_config_sha256",
            "mpl_contract_sha256",
            "teacher_contract_id",
            "teacher_contract_sha256",
            "teacher_planning_contract_id",
            "teacher_planning_contract_sha256",
            "input_rollout_reliable_rows",
            "input_accepted_reliable_rows",
            "input_rollout_legacy_rows",
            "input_rollout_telemetry_lookup_count",
            "input_rollout_snapshot_missing_count",
            "input_rollout_state_depth_skew_max_ns",
            "input_rollout_frame_contract_failures",
            "endpoint_identity_chain_valid",
            "task_contract_id",
            "task_contract_schema_version",
            "task_contract_sha256",
            "max_primitive_steps",
            "task_contract_mode",
            "max_steps",
        ):
            if key in producer_provenance:
                manifest[key] = producer_provenance[key]
        for key in (
            "depth_mask_contract_id",
            "depth_mask_contract_sha256",
            "collision_radius_m",
            "depth_slack_m",
            "path_sample_stride",
            "max_patch_radius_px",
        ):
            if key in depth_masks.metadata:
                manifest[key] = depth_masks.metadata[key]
    write_json_atomic(building / "manifest.json", manifest)
    if out_dir.exists():
        shutil.rmtree(str(out_dir))
    os.replace(str(building), str(out_dir))
    return manifest

class MappedSoftRolloutDataset:
    """A row-selected view over an immutable mmap cache."""

    ARRAY_NAMES = (
        "depths", "continuous", "prev_actions", "height_masks",
        "local_depth_masks", "behavior_actions", "teacher_actions",
        "soft_targets", "history_starts", "episode_ids",
    )

    def __init__(
        self,
        cache_dir: Path,
        rows: List[Dict],
        index_path: Path,
        labels_path: Optional[Path] = None,
        depth_masks_path: Optional[Path] = None,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve()
        manifest_path = self.cache_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("contract_id") != BC_MMAP_DATASET_CONTRACT_ID:
            raise ValueError("BC mmap dataset contract mismatch")
        validate_task_contract(
            self.manifest,
            expected_max_primitive_steps=self.manifest.get(
                "max_steps", self.manifest.get("max_primitive_steps")
            ),
            path="{} task contract".format(manifest_path),
        )
        if self.manifest.get("source_index_sha256") != file_sha256(self.index_path):
            raise ValueError("BC mmap cache index hash mismatch; rebuild the cache")
        if labels_path is not None and self.manifest.get("source_labels_sha256") != file_sha256(labels_path):
            raise ValueError("BC mmap cache label hash mismatch; rebuild the cache")
        expected_masks_hash = str(self.manifest.get("source_depth_masks_sha256", ""))
        if depth_masks_path is not None:
            if expected_masks_hash != file_sha256(depth_masks_path):
                raise ValueError("BC mmap cache depth-mask hash mismatch; rebuild the cache")
        for name in self.ARRAY_NAMES:
            setattr(self, name, np.load(str(self.cache_dir / (name + ".npy")), mmap_mode="r"))

        by_path = {str(item["dataset_npz"]): item for item in self.manifest["episodes"]}
        selected = []
        for row in rows:
            path = resolve_dataset_path(row, self.index_path)
            key = _relative(path, self.index_path.parent)
            if key not in by_path:
                raise KeyError("rollout is absent from BC mmap cache: {}".format(key))
            entry = by_path[key]
            selected.append(
                np.arange(
                    int(entry["offset"]),
                    int(entry["offset"]) + int(entry["length"]),
                    dtype=np.int64,
                )
            )
        if not selected:
            raise RuntimeError("empty mapped dataset selection")
        self.indices = np.concatenate(selected)
        self.normalizer = None

    def apply_normalizer(self, normalizer) -> None:
        self.normalizer = normalizer

    def normalizer_values(self) -> np.ndarray:
        return np.asarray(self.continuous[self.indices], dtype=np.float32)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def teacher_hist(self) -> np.ndarray:
        values = np.asarray(self.teacher_actions[self.indices], dtype=np.int64)
        valid = values[(values >= 0) & (values < NUM_ACTIONS)]
        return np.bincount(valid, minlength=NUM_ACTIONS)

    def behavior_hist(self) -> np.ndarray:
        return np.bincount(
            np.asarray(self.behavior_actions[self.indices], dtype=np.int64),
            minlength=NUM_ACTIONS,
        )

    def sample(self, local_index: int, depth_history_frames: int):
        if self.normalizer is None:
            raise RuntimeError("normalizer has not been applied")
        global_index = int(self.indices[int(local_index)])
        history_start = int(self.history_starts[global_index])
        history = np.maximum(
            global_index - np.arange(int(depth_history_frames) - 1, -1, -1),
            history_start,
        )
        continuous = np.asarray(self.continuous[global_index], dtype=np.float32)
        normalized = self.normalizer.transform_continuous(continuous)
        vector = np.zeros(POLICY_VECTOR_DIM, dtype=np.float32)
        vector[:CONTINUOUS_DIM] = normalized
        previous = int(self.prev_actions[global_index])
        if 0 <= previous < NUM_ACTIONS:
            vector[CONTINUOUS_DIM + previous] = 1.0
        return global_index, history, vector


def validate_bc_dataset_artifacts(
    *,
    index_path: Path,
    labels_path: Path,
    depth_masks_path: Path,
    dataset_audit_path: Path,
    bc_mmap_dir: Path,
    expected_observation_contract: str = EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    mode: str = "full",
    sample_count: int = 10000,
) -> Dict[str, Any]:
    """Validate the published rollout -> labels/masks -> BC mmap chain.

    Each contract check remains delegated to its existing owner.  This
    function only joins the validated identities and checks the publication
    accounting at the final artifact boundary.
    """

    if str(mode) not in {"fast", "full"}:
        raise ValueError("BC dataset validation mode must be fast or full")
    if int(sample_count) < 0:
        raise ValueError("BC dataset validation sample_count must be non-negative")
    index_path = Path(index_path).expanduser().resolve()
    labels_path = Path(labels_path).expanduser().resolve()
    depth_masks_path = Path(depth_masks_path).expanduser().resolve()
    dataset_audit_path = Path(dataset_audit_path).expanduser().resolve()
    bc_mmap_dir = Path(bc_mmap_dir).expanduser().resolve()
    for path, label in (
        (index_path, "rollout index"),
        (labels_path, "teacher labels"),
        (depth_masks_path, "depth masks"),
        (dataset_audit_path, "dataset audit"),
        (bc_mmap_dir / "manifest.json", "BC mmap manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError("missing {}: {}".format(label, path))

    if str(mode) == "fast":
        return _validate_bc_dataset_fast(
            index_path=index_path,
            labels_path=labels_path,
            depth_masks_path=depth_masks_path,
            dataset_audit_path=dataset_audit_path,
            bc_mmap_dir=bc_mmap_dir,
            expected_observation_contract=expected_observation_contract,
            sample_count=int(sample_count),
        )

    provenance = validate_rollout_provenance(index_path)
    labels = TeacherLabelStore(labels_path)
    masks = DepthActionMaskStore(depth_masks_path)
    cross = validate_cross_artifact_consistency(
        provenance, labels.metadata, masks.metadata
    )
    audit = read_json(dataset_audit_path)
    if not isinstance(audit, Mapping):
        raise ValueError("dataset audit must be a mapping")
    if str(audit.get("observation_contract", "")) != str(expected_observation_contract):
        raise ValueError("dataset audit observation contract mismatch")
    if str(audit.get("observation_source", "")) != str(expected_observation_contract):
        raise ValueError("dataset audit observation source mismatch")
    if audit.get("provenance_checked") is not True:
        raise ValueError("dataset audit did not validate producer provenance")
    if int(audit.get("episodes", -1)) != len(provenance.rows):
        raise ValueError("dataset audit episode count mismatch")
    if int(audit.get("transitions", -1)) != int(provenance.accepted_reliable_rows):
        raise ValueError("dataset audit transition count mismatch")
    if int(audit.get("reliable_rows", -1)) != int(provenance.accepted_reliable_rows):
        raise ValueError("dataset audit reliable row count mismatch")
    if int(audit.get("legacy_rows", -1)) != 0:
        raise ValueError("dataset audit contains legacy rows")
    manifest = read_json(bc_mmap_dir / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise ValueError("BC mmap manifest must be a mapping")
    from planning.bc.trainer import validate_bc_mmap_provenance

    bc_provenance = validate_bc_mmap_provenance(
        manifest,
        expected_observation_contract=expected_observation_contract,
        path=str(bc_mmap_dir / "manifest.json"),
    )
    if str(manifest.get("source_index_sha256", "")) != file_sha256(index_path):
        raise ValueError("BC mmap source index hash mismatch")
    if str(manifest.get("source_labels_sha256", "")) != file_sha256(labels_path):
        raise ValueError("BC mmap source labels hash mismatch")
    if str(manifest.get("source_depth_masks_sha256", "")) != file_sha256(depth_masks_path):
        raise ValueError("BC mmap source depth masks hash mismatch")
    if int(bc_provenance["transition_count"]) != int(provenance.accepted_reliable_rows):
        raise ValueError("BC mmap transition count mismatch")
    if int(manifest.get("episode_count", -1)) != len(provenance.rows):
        raise ValueError("BC mmap episode count mismatch")
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or len(entries) != len(provenance.rows):
        raise ValueError("BC mmap episode table mismatch")
    offset = 0
    for entry, episode_id, length in zip(
        entries, provenance.episode_ids, provenance.transition_lengths
    ):
        if not isinstance(entry, Mapping):
            raise ValueError("BC mmap episode entry is not a mapping")
        if int(entry.get("episode_id", -1)) != int(episode_id):
            raise ValueError("BC mmap episode order mismatch")
        if int(entry.get("offset", -1)) != offset:
            raise ValueError("BC mmap transition offset mismatch")
        if int(entry.get("length", -1)) != int(length):
            raise ValueError("BC mmap transition length mismatch")
        offset += int(length)

    return {
        "row_count": int(provenance.accepted_reliable_rows),
        "episode_count": len(provenance.rows),
        "observation_contract": str(bc_provenance["observation_contract"]),
        "rollout_index_sha256": provenance.index_sha256,
        "rollout_manifest_sha256": provenance.manifest_sha256,
        "labels_sha256": file_sha256(labels_path),
        "depth_masks_sha256": file_sha256(depth_masks_path),
        "bc_mmap_manifest_sha256": file_sha256(bc_mmap_dir / "manifest.json"),
        "dataset_audit": str(dataset_audit_path),
        "cross_artifact": cross,
        "mode": "full",
    }


def _npz_metadata_only(path: Path) -> Dict[str, Any]:
    """Read one artifact metadata field without materializing its arrays."""

    with np.load(str(path), allow_pickle=False) as data:
        if "metadata_json" not in data:
            raise ValueError("{} missing metadata_json".format(path))
        value = data["metadata_json"]
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    metadata = json.loads(str(value))
    if not isinstance(metadata, Mapping):
        raise ValueError("{} metadata_json is not a mapping".format(path))
    return dict(metadata)


def _validate_bc_manifest_episode_table(
    manifest: Mapping[str, Any], provenance, *, path: str
) -> None:
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or len(entries) != len(provenance.rows):
        raise ValueError("{} episode table mismatch".format(path))
    offset = 0
    for entry, episode_id, length in zip(
        entries, provenance.episode_ids, provenance.transition_lengths
    ):
        if not isinstance(entry, Mapping):
            raise ValueError("{} episode entry is not a mapping".format(path))
        if int(entry.get("episode_id", -1)) != int(episode_id):
            raise ValueError("{} episode order mismatch".format(path))
        if int(entry.get("offset", -1)) != int(offset):
            raise ValueError("{} transition offset mismatch".format(path))
        if int(entry.get("length", -1)) != int(length):
            raise ValueError("{} transition length mismatch".format(path))
        offset += int(length)


def _validate_bc_mmap_array_headers(
    bc_mmap_dir: Path,
    manifest: Mapping[str, Any],
    *,
    path: str,
) -> Dict[str, Any]:
    """Validate mmap headers and physical sizes without scanning full arrays."""

    total = int(manifest["transition_count"])
    depth_shape = tuple(int(value) for value in manifest["depth_shape"])
    actions = int(manifest["num_actions"])
    continuous_dim = int(manifest["continuous_dim"])
    specs = {
        "depths": ((total,) + depth_shape, np.dtype(np.float16)),
        "continuous": ((total, continuous_dim), np.dtype(np.float32)),
        "prev_actions": ((total,), np.dtype(np.int16)),
        "height_masks": ((total, actions), np.dtype(np.bool_)),
        "local_depth_masks": ((total, actions), np.dtype(np.bool_)),
        "behavior_actions": ((total,), np.dtype(np.int16)),
        "teacher_actions": ((total,), np.dtype(np.int16)),
        "soft_targets": ((total, actions), np.dtype(np.float32)),
        "history_starts": ((total,), np.dtype(np.int64)),
        "episode_ids": ((total,), np.dtype(np.int64)),
    }
    arrays = {}
    for name, (expected_shape, expected_dtype) in specs.items():
        array_path = bc_mmap_dir / (name + ".npy")
        if not array_path.is_file():
            raise FileNotFoundError("{} missing mmap array {}".format(path, array_path))
        array = np.load(str(array_path), mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != tuple(expected_shape):
            raise ValueError(
                "{} {} shape {} != {}".format(path, name, array.shape, expected_shape)
            )
        if np.dtype(array.dtype) != expected_dtype:
            raise ValueError(
                "{} {} dtype {} != {}".format(path, name, array.dtype, expected_dtype)
            )
        if int(array_path.stat().st_size) < int(array.nbytes):
            raise ValueError("{} {} file is shorter than its array payload".format(path, name))
        arrays[name] = array
    return arrays


def _sample_bc_mmap_values(
    arrays: Mapping[str, Any], *, total: int, sample_count: int
) -> None:
    if int(sample_count) <= 0 or int(total) <= 0:
        return
    count = min(int(total), int(sample_count))
    indices = np.linspace(0, int(total) - 1, num=count, dtype=np.int64)
    for name, array in arrays.items():
        for start in range(0, len(indices), 512):
            values = np.asarray(array[indices[start : start + 512]])
            if values.dtype.kind == "f" and not np.isfinite(values).all():
                raise ValueError("BC mmap sampled {} values are non-finite".format(name))


def _validate_bc_dataset_fast(
    *,
    index_path: Path,
    labels_path: Path,
    depth_masks_path: Path,
    dataset_audit_path: Path,
    bc_mmap_dir: Path,
    expected_observation_contract: str,
    sample_count: int,
) -> Dict[str, Any]:
    """Run the bounded pre-training validation path."""

    from planning.bc.trainer import validate_bc_mmap_provenance

    provenance = validate_rollout_provenance(index_path)
    labels_metadata = _npz_metadata_only(labels_path)
    masks_metadata = _npz_metadata_only(depth_masks_path)
    cross = validate_cross_artifact_consistency(
        provenance, labels_metadata, masks_metadata
    )
    if str(cross["observation_contract"]) != str(expected_observation_contract):
        raise ValueError("fast BC validation observation contract mismatch")
    audit = read_json(dataset_audit_path)
    if not isinstance(audit, Mapping):
        raise ValueError("dataset audit must be a mapping")
    if audit.get("provenance_checked") is not True:
        raise ValueError("dataset audit did not validate producer provenance")
    if int(audit.get("episodes", -1)) != len(provenance.rows):
        raise ValueError("dataset audit episode count mismatch")
    if int(audit.get("transitions", -1)) != int(provenance.accepted_reliable_rows):
        raise ValueError("dataset audit transition count mismatch")

    manifest = read_json(bc_mmap_dir / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise ValueError("BC mmap manifest must be a mapping")
    bc_provenance = validate_bc_mmap_provenance(
        manifest,
        expected_observation_contract=expected_observation_contract,
        path=str(bc_mmap_dir / "manifest.json"),
    )
    if str(manifest.get("source_index_sha256", "")) != file_sha256(index_path):
        raise ValueError("BC mmap source index hash mismatch")
    if str(manifest.get("source_labels_sha256", "")) != file_sha256(labels_path):
        raise ValueError("BC mmap source labels hash mismatch")
    if str(manifest.get("source_depth_masks_sha256", "")) != file_sha256(depth_masks_path):
        raise ValueError("BC mmap source depth masks hash mismatch")
    if int(manifest.get("episode_count", -1)) != len(provenance.rows):
        raise ValueError("BC mmap episode count mismatch")
    _validate_bc_manifest_episode_table(
        manifest, provenance, path=str(bc_mmap_dir / "manifest.json")
    )
    arrays = _validate_bc_mmap_array_headers(
        bc_mmap_dir, manifest, path=str(bc_mmap_dir / "manifest.json")
    )
    _sample_bc_mmap_values(
        arrays,
        total=int(bc_provenance["transition_count"]),
        sample_count=int(sample_count),
    )
    return {
        "row_count": int(provenance.accepted_reliable_rows),
        "episode_count": len(provenance.rows),
        "observation_contract": str(bc_provenance["observation_contract"]),
        "rollout_index_sha256": provenance.index_sha256,
        "rollout_manifest_sha256": provenance.manifest_sha256,
        "labels_sha256": file_sha256(labels_path),
        "depth_masks_sha256": file_sha256(depth_masks_path),
        "bc_mmap_manifest_sha256": file_sha256(bc_mmap_dir / "manifest.json"),
        "dataset_audit": str(dataset_audit_path),
        "cross_artifact": cross,
        "mode": "fast",
        "sample_count": min(int(sample_count), int(provenance.accepted_reliable_rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--depth-action-masks", required=True)
    parser.add_argument(
        "--observation-contract",
        default=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        help="require the formal observation provenance contract",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="limit the validated rollout selection for a bounded fixture smoke",
    )
    parser.add_argument(
        "--episode-ids-file",
        default="",
        help="select canonical episode ids from a newline/comma text or JSON file",
    )
    args = parser.parse_args()
    if int(args.max_episodes) < 0:
        raise ValueError("--max-episodes must be non-negative")

    index_path = Path(args.index).expanduser().resolve()
    labels = TeacherLabelStore(Path(args.labels))
    masks = DepthActionMaskStore(Path(args.depth_action_masks))
    rollout_provenance = validate_rollout_provenance(index_path)
    rows = [dict(row) for row in rollout_provenance.rows]
    if str(args.episode_ids_file).strip():
        wanted = set(read_episode_ids_file(Path(args.episode_ids_file)))
        available = {int(float(row["episode_id"])) for row in rows}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError("selected episodes absent from rollout index: {}".format(missing[:10]))
        rows = [row for row in rows if int(float(row["episode_id"])) in wanted]
        rollout_provenance = rollout_provenance.subset(
            int(float(row["episode_id"])) for row in rows
        )
    if int(args.max_episodes) > 0:
        rows = rows[: int(args.max_episodes)]
        rollout_provenance = rollout_provenance.subset(
            int(float(row["episode_id"])) for row in rows
        )
    if not rows:
        raise RuntimeError("no accepted rollout rows")
    manifest = build_bc_mmap_dataset(
        rows,
        index_path,
        labels,
        Path(args.out_dir),
        masks,
        overwrite=bool(args.overwrite),
        expected_observation_contract=str(args.observation_contract),
    )
    print("BC_MMAP_DATASET_SUMMARY")
    for key in ("contract_id", "episode_count", "transition_count", "depth_shape"):
        print("  {}: {}".format(key, manifest[key]))
    print("  out_dir:", Path(args.out_dir).expanduser().resolve())
    print("RESULT=PASS")
    return 0
