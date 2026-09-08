"""Dataset I/O helpers for rollout episodes and offline teacher labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import numpy as np

from planning.contracts.feature import (
    FEATURE_CONTRACT_ID,
    GOAL_DIM,
    NUM_ACTIONS,
    STATE_DIM,
    validate_feature_arrays,
)
from planning.mission.spec import TASK_CONTRACT_ID
from planning.contracts.task import (
    DEFAULT_MAX_PRIMITIVE_STEPS,
    task_contract_fields,
    validate_task_contract,
)
from planning.contracts.teacher_path import (
    TEACHER_ACTUAL_PATH_MAX_M,
    TEACHER_PATH_LENGTH_CONTRACT_ID,
    actual_path_eligible,
)
from planning.contracts.pipeline_provenance import validate_artifact_provenance
from planning.common.io import read_csv, write_csv_atomic, write_npz_atomic

ROLLOUT_CONTRACT_ID = "teacher_rollout_plan44_actual46"
LABEL_CONTRACT_ID = "teacher_soft_labels_plan44_actual46"

ROLLOUT_REQUIRED_KEYS = (
    "depths",
    "states",
    "goals",
    "height_action_masks",
    "execution_action_masks",
    "behavior_actions",
    "prev_actions",
    "poses_before",
    "velocities_before",
    "state_ids",
    "sim_time_ns",
    "state_stamp_ns",
    "depth_stamp_ns",
    "sensor_skew_ns",
    "primitive_actual_path_lengths_m",
    "start",
    "goal",
    "metadata_json",
)

def row_episode_id(row: Dict) -> int:
    """Parse an episode identifier consistently across index utilities."""
    return int(float(row["episode_id"]))

def resolve_dataset_path(row: Dict, index_path: Path) -> Path:
    raw = str(row.get("dataset_npz", "")).strip()
    if not raw:
        raise ValueError("missing dataset_npz")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(index_path).expanduser().resolve().parent / path).resolve()

def resolve_artifact_path(value, artifact_path: Path) -> Path:
    """Resolve a path stored relative to the structured artifact containing it."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(artifact_path).expanduser().resolve().parent / path).resolve()

def _metadata_from_array(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not value:
        return {}
    return json.loads(str(value))

def save_rollout_episode(path: Path, arrays: Dict[str, np.ndarray], metadata: Dict, compress: bool = False) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    write_npz_atomic(path, payload, compress=compress)

def load_rollout_episode(path: Path, validate: bool = True) -> Dict[str, Union[np.ndarray, Dict]]:
    path = Path(path).expanduser().resolve()
    with np.load(str(path), allow_pickle=False) as data:
        missing = [key for key in ROLLOUT_REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError("{} missing keys {}".format(path, missing))
        episode = {key: data[key] for key in data.files if key != "metadata_json"}
        metadata = _metadata_from_array(data["metadata_json"])
    episode["metadata"] = metadata
    if validate:
        validate_rollout_episode(episode, path)
    return episode


def load_rollout_episode_fields(
    path: Path,
    fields: Sequence[str],
) -> Dict[str, Union[np.ndarray, Dict]]:
    """Load only the arrays needed by a bounded offline operation."""

    path = Path(path).expanduser().resolve()
    requested = tuple(dict.fromkeys(str(field) for field in fields))
    if "metadata_json" in requested or "metadata" in requested:
        raise ValueError("metadata is loaded automatically")
    with np.load(str(path), allow_pickle=False) as data:
        missing = [
            key for key in requested + ("metadata_json",)
            if key not in data
        ]
        if missing:
            raise KeyError("{} missing keys {}".format(path, missing))
        episode = {key: data[key] for key in requested}
        metadata = _metadata_from_array(data["metadata_json"])
    episode["metadata"] = metadata
    return episode


def validate_rollout_metadata(
    metadata: Dict,
    path: Union[Path, str] = "<memory>",
) -> None:
    metadata = dict(metadata)
    if str(metadata.get("rollout_contract_id", "")) != ROLLOUT_CONTRACT_ID:
        raise ValueError("{} rollout contract mismatch: {}".format(path, metadata.get("rollout_contract_id")))
    if str(metadata.get("feature_contract_id", "")) != FEATURE_CONTRACT_ID:
        raise ValueError("{} feature contract mismatch: {}".format(path, metadata.get("feature_contract_id")))
    expected_max_steps = metadata.get(
        "max_steps", metadata.get("max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS)
    )
    validate_task_contract(
        metadata,
        expected_max_primitive_steps=expected_max_steps,
        path="{} task contract".format(path),
    )

    mpl_hash = str(metadata.get("mpl_contract_sha256", ""))
    if len(mpl_hash) != 64:
        raise ValueError("{} missing or invalid motion-primitive contract hash".format(path))
    if (
        str(metadata.get("teacher_path_length_contract_id", ""))
        != TEACHER_PATH_LENGTH_CONTRACT_ID
    ):
        raise ValueError("{} teacher path-length contract mismatch".format(path))
    if (
        abs(
            float(metadata.get("teacher_actual_path_max_m", float("nan")))
            - TEACHER_ACTUAL_PATH_MAX_M
        )
        > 1.0e-9
    ):
        raise ValueError("{} actual path limit mismatch".format(path))


def validate_rollout_episode(episode: Dict, path: Union[Path, str] = "<memory>") -> None:
    metadata = dict(episode.get("metadata", {}))
    validate_rollout_metadata(metadata, path)
    behavior_actions = np.asarray(episode["behavior_actions"], dtype=np.int64)
    transition_count = int(behavior_actions.shape[0])
    if behavior_actions.ndim != 1:
        raise ValueError("{} behavior_actions must be [T]".format(path))
    if transition_count <= 0:
        raise ValueError("{} contains zero transitions".format(path))
    if np.any((behavior_actions < 0) | (behavior_actions >= NUM_ACTIONS)):
        raise ValueError("{} behavior action out of range".format(path))

    depths = np.asarray(episode["depths"])
    states = np.asarray(episode["states"])
    goals = np.asarray(episode["goals"])
    validate_feature_arrays(states, goals)
    if depths.ndim != 3 or depths.shape[0] != transition_count:
        raise ValueError("{} depths shape bad: {}".format(path, depths.shape))
    if states.shape[0] != transition_count or goals.shape[0] != transition_count:
        raise ValueError("{} feature transition mismatch".format(path))
    if not np.isfinite(depths).all():
        raise ValueError("{} non-finite depth".format(path))

    expected_shapes = {
        "height_action_masks": (transition_count, NUM_ACTIONS),
        "execution_action_masks": (transition_count, NUM_ACTIONS),
        "prev_actions": (transition_count,),
        "poses_before": (transition_count, 4),
        "velocities_before": (transition_count, 3),
        "state_ids": (transition_count,),
        "sim_time_ns": (transition_count,),
        "state_stamp_ns": (transition_count,),
        "depth_stamp_ns": (transition_count,),
        "sensor_skew_ns": (transition_count,),
        "primitive_actual_path_lengths_m": (transition_count,),
    }
    for key, expected in expected_shapes.items():
        actual = np.asarray(episode[key]).shape
        if actual != expected:
            raise ValueError("{} {} shape {} != {}".format(path, key, actual, expected))

    prev_actions = np.asarray(episode["prev_actions"], dtype=np.int64)
    if np.any((prev_actions < -1) | (prev_actions >= NUM_ACTIONS)):
        raise ValueError("{} prev action out of range".format(path))
    sensor_skew_ns = np.asarray(episode["sensor_skew_ns"], dtype=np.int64)
    if np.any(sensor_skew_ns < 0):
        raise ValueError("{} contains negative sensor skew".format(path))
    primitive_path_lengths = np.asarray(
        episode["primitive_actual_path_lengths_m"], dtype=np.float64
    )
    if not np.isfinite(primitive_path_lengths).all() or np.any(primitive_path_lengths < 0.0):
        raise ValueError("{} contains invalid primitive actual path lengths".format(path))
    actual_path_length = float(primitive_path_lengths.sum())
    if not actual_path_eligible(actual_path_length):
        raise ValueError("{} exceeds actual path-length contract".format(path))
    if (
        abs(float(metadata.get("actual_path_length_m", float("nan"))) - actual_path_length)
        > 1.0e-3
    ):
        raise ValueError("{} actual path length metadata mismatch".format(path))
    execution_masks = np.asarray(episode["execution_action_masks"], dtype=np.bool_)
    if not np.all(execution_masks[np.arange(transition_count), behavior_actions]):
        raise ValueError("{} contains behavior action outside execution mask".format(path))
    if np.asarray(episode["start"]).shape != (3,) or np.asarray(episode["goal"]).shape != (3,):
        raise ValueError("{} start/goal shape bad".format(path))

def make_rollout_metadata(**extra) -> Dict:
    max_primitive_steps = extra.get(
        "max_primitive_steps", extra.get("max_steps", DEFAULT_MAX_PRIMITIVE_STEPS)
    )
    metadata = {
        "rollout_contract_id": ROLLOUT_CONTRACT_ID,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        **task_contract_fields(max_primitive_steps),
        "num_actions": NUM_ACTIONS,
        "state_dim": STATE_DIM,
        "goal_dim": GOAL_DIM,
    }
    metadata.update(extra)
    return metadata

class TeacherLabelStore:
    """Memory-backed exact labels mapped by rollout episode path."""

    REQUIRED_KEYS = (
        "soft_targets",
        "global_action_masks",
        "teacher_argmax",
        "behavior_actions",
        "valid_counts",
        "teacher_entropies",
        "episode_npz_paths",
        "episode_offsets",
        "episode_lengths",
        "metadata_json",
    )

    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        with np.load(str(self.path), allow_pickle=False) as data:
            missing = [key for key in self.REQUIRED_KEYS if key not in data]
            if missing:
                raise KeyError("{} missing label keys {}".format(self.path, missing))
            self.soft_targets = data["soft_targets"].astype(np.float32)
            self.global_action_masks = data["global_action_masks"].astype(np.bool_)
            self.teacher_argmax = data["teacher_argmax"].astype(np.int64)
            self.behavior_actions = data["behavior_actions"].astype(np.int64)
            self.valid_counts = data["valid_counts"].astype(np.int16)
            self.teacher_entropies = data["teacher_entropies"].astype(np.float32)
            self.episode_npz_paths = [
                str(resolve_artifact_path(value, self.path))
                for value in data["episode_npz_paths"].tolist()
            ]
            self.episode_offsets = data["episode_offsets"].astype(np.int64)
            self.episode_lengths = data["episode_lengths"].astype(np.int64)
            self.metadata = _metadata_from_array(data["metadata_json"])
            self.teacher_scores = data["teacher_scores"].astype(np.float32) if "teacher_scores" in data else None

        if str(self.metadata.get("label_contract_id", "")) != LABEL_CONTRACT_ID:
            raise ValueError("label contract mismatch in {}".format(self.path))
        if str(self.metadata.get("feature_contract_id", "")) != FEATURE_CONTRACT_ID:
            raise ValueError("feature contract mismatch in {}".format(self.path))
        expected_max_steps = self.metadata.get(
            "max_steps",
            self.metadata.get("max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS),
        )
        validate_task_contract(
            self.metadata,
            expected_max_primitive_steps=expected_max_steps,
            path="{} task contract".format(self.path),
        )
        mpl_hash = str(self.metadata.get("mpl_contract_sha256", ""))
        if len(mpl_hash) != 64:
            raise ValueError("missing or invalid motion-primitive contract hash in {}".format(self.path))
        if self.soft_targets.ndim != 2 or self.soft_targets.shape[1] != NUM_ACTIONS:
            raise ValueError("soft_targets shape bad: {}".format(self.soft_targets.shape))
        transition_count = self.soft_targets.shape[0]
        if "provenance_schema_version" in self.metadata:
            validate_artifact_provenance(
                self.metadata,
                artifact_kind="teacher_labels",
                expected_row_count=int(transition_count),
            )
        if self.global_action_masks.shape != self.soft_targets.shape:
            raise ValueError("global_action_masks shape mismatch")
        if not bool(self.metadata.get("actual_state_relabeling", False)):
            raise ValueError("labels were not generated from actual rollout states")
        if not np.isfinite(self.soft_targets).all() or np.any(self.soft_targets < 0.0):
            raise ValueError("soft_targets contain invalid probabilities")
        teacher_in_range = (self.teacher_argmax >= 0) & (self.teacher_argmax < NUM_ACTIONS)
        safe_teacher = np.clip(self.teacher_argmax, 0, NUM_ACTIONS - 1)
        if np.any(teacher_in_range & ~self.global_action_masks[np.arange(transition_count), safe_teacher]):
            raise ValueError("teacher_argmax outside global action mask")
        probability_sums = self.soft_targets.sum(axis=1, dtype=np.float64)
        if np.any(np.abs(probability_sums[teacher_in_range] - 1.0) > 5.0e-3):
            raise ValueError("soft target probability sums are invalid")
        if np.any(np.abs(probability_sums[~teacher_in_range]) > 5.0e-3):
            raise ValueError("dead-end rows must have zero soft target mass")
        for name, value in (
            ("teacher_argmax", self.teacher_argmax),
            ("behavior_actions", self.behavior_actions),
            ("valid_counts", self.valid_counts),
            ("teacher_entropies", self.teacher_entropies),
        ):
            if value.shape != (transition_count,):
                raise ValueError("{} shape mismatch".format(name))
        if len(self.episode_npz_paths) != len(self.episode_offsets) or len(self.episode_offsets) != len(self.episode_lengths):
            raise ValueError("episode mapping shape mismatch")
        self.path_to_slice = {
            path_value: (int(offset), int(length))
            for path_value, offset, length in zip(
                self.episode_npz_paths, self.episode_offsets, self.episode_lengths
            )
        }

    def get_episode(self, rollout_path: Path, expected_behavior_actions: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        key = str(Path(rollout_path).expanduser().resolve())
        if key not in self.path_to_slice:
            raise KeyError("rollout path not found in labels: {}".format(key))
        offset, length = self.path_to_slice[key]
        sl = slice(offset, offset + length)
        if expected_behavior_actions is not None:
            expected = np.asarray(expected_behavior_actions, dtype=np.int64)
            if expected.shape != (length,) or not np.array_equal(self.behavior_actions[sl], expected):
                raise ValueError("behavior action mismatch for {}".format(key))
        output = {
            "soft_targets": self.soft_targets[sl],
            "global_action_masks": self.global_action_masks[sl],
            "teacher_argmax": self.teacher_argmax[sl],
            "behavior_actions": self.behavior_actions[sl],
            "valid_counts": self.valid_counts[sl],
        }
        if self.teacher_scores is not None:
            output["teacher_scores"] = self.teacher_scores[sl]
        return output
