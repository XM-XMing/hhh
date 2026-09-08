"""Storage and reconstruction helpers for local depth-only action masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import numpy as np

from planning.contracts.feature import NUM_ACTIONS
from planning.data.rollout import resolve_artifact_path
from planning.contracts.pipeline_provenance import validate_artifact_provenance
from planning.contracts.task import DEFAULT_MAX_PRIMITIVE_STEPS, validate_task_contract

DEPTH_ACTION_MASK_CONTRACT_ID = "local_depth_action_masks"

def normalized_depth_to_metric(depth: np.ndarray, depth_min_m: float, depth_max_m: float) -> np.ndarray:
    """Recover clipped metric depth from rollout observations stored in [0,1]."""
    values = np.asarray(depth, dtype=np.float32)
    return float(depth_min_m) + np.clip(values, 0.0, 1.0) * (float(depth_max_m) - float(depth_min_m))

def _metadata(value) -> Dict:
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value)) if value else {}

class DepthActionMaskStore:
    """Read local depth masks and map every rollout path to its transition range."""

    REQUIRED_KEYS = (
        "local_action_masks",
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
                raise KeyError("{} missing keys {}".format(self.path, missing))
            self.local_action_masks = np.asarray(data["local_action_masks"], dtype=np.bool_)
            self.paths = [
                str(resolve_artifact_path(value, self.path))
                for value in data["episode_npz_paths"].tolist()
            ]
            self.offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
            self.lengths = np.asarray(data["episode_lengths"], dtype=np.int64)
            self.metadata = _metadata(data["metadata_json"])
        if str(self.metadata.get("depth_action_mask_contract_id", "")) != DEPTH_ACTION_MASK_CONTRACT_ID:
            raise ValueError("depth action mask contract mismatch in {}".format(self.path))
        expected_max_steps = self.metadata.get(
            "max_steps",
            self.metadata.get("max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS),
        )
        validate_task_contract(
            self.metadata,
            expected_max_primitive_steps=expected_max_steps,
            path="{} task contract".format(self.path),
        )
        if self.local_action_masks.ndim != 2 or self.local_action_masks.shape[1] != NUM_ACTIONS:
            raise ValueError("local action mask shape mismatch: {}".format(self.local_action_masks.shape))
        if "provenance_schema_version" in self.metadata:
            validate_artifact_provenance(
                self.metadata,
                artifact_kind="depth_masks",
                expected_row_count=int(self.local_action_masks.shape[0]),
            )
        if not (len(self.paths) == len(self.offsets) == len(self.lengths)):
            raise ValueError("depth action mask episode mapping length mismatch")
        self.path_to_slice = {}
        for path, offset, length in zip(self.paths, self.offsets, self.lengths):
            if int(offset) < 0 or int(length) <= 0 or int(offset) + int(length) > self.local_action_masks.shape[0]:
                raise ValueError("invalid depth action mask range for {}".format(path))
            if path in self.path_to_slice:
                raise ValueError("duplicate depth action mask path {}".format(path))
            self.path_to_slice[path] = (int(offset), int(length))

    def get_episode(self, rollout_path: Path, expected_length: int) -> np.ndarray:
        key = str(Path(rollout_path).expanduser().resolve())
        if key not in self.path_to_slice:
            raise KeyError("no depth action masks for {}".format(key))
        offset, length = self.path_to_slice[key]
        if int(length) != int(expected_length):
            raise ValueError("depth action mask length {} != {} for {}".format(length, expected_length, key))
        return self.local_action_masks[offset:offset + length]
