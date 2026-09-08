"""Diagnostic multi-action replay contract and integrity audit.

This module deliberately owns a separate artifact from
``planning.awac.replay.AWACReplayBuffer``.  It records several legal actions
for the same frozen observation so that action-level ranking can be audited
without changing the production AWAC model, replay, or learner.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MULTI_ACTION_REPLAY_SCHEMA_ID = "multi_action_replay_v1"
ACTION_SOURCES = ("BC", "TEACHER", "NEIGHBOR", "RANDOM")
_ACTION_SOURCE_SET = frozenset(ACTION_SOURCES)
_REQUIRED_FIELDS = (
    "state_id",
    "mission_id",
    "episode_id",
    "step_id",
    "route_id",
    "action_source",
    "action",
    "mask",
    "reward",
    "state_vector",
    "state_depth",
    "next_state_id",
    "next_state_vector",
    "next_state_depth",
    "next_mask",
    "done",
    "transition_id",
)
_STRING_FIELDS = {
    "state_id",
    "mission_id",
    "episode_id",
    "route_id",
    "action_source",
    "next_state_id",
    "transition_id",
}
_ARRAY_FIELDS = _REQUIRED_FIELDS
_FIELD_DTYPES = {
    "step_id": np.int64,
    "action": np.int16,
    "mask": np.uint8,
    "reward": np.float32,
    "state_vector": np.float32,
    "state_depth": np.float32,
    "next_state_vector": np.float32,
    "next_state_depth": np.float32,
    "next_mask": np.uint8,
    "done": np.uint8,
}


def _as_nonempty_text(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("{} must be a non-empty string".format(field))
    return text


def _as_mask(value: Any, field: str) -> np.ndarray:
    mask = np.asarray(value, dtype=np.bool_)
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError("{} must be a non-empty one-dimensional mask".format(field))
    return mask


def _as_numeric_array(value: Any, field: str, *, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError("{} must have {} dimensions".format(field, ndim))
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("{} must contain finite, non-empty values".format(field))
    return np.ascontiguousarray(array, dtype=np.float32)


def _normalise_transition(row: Mapping[str, Any], index: int) -> Dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("transition {} must be a mapping".format(index))
    missing = [field for field in _REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(
            "transition {} missing fields: {}".format(index, ", ".join(missing))
        )

    result = {
        "state_id": _as_nonempty_text(row["state_id"], "state_id"),
        "mission_id": _as_nonempty_text(row["mission_id"], "mission_id"),
        "episode_id": _as_nonempty_text(row["episode_id"], "episode_id"),
        "route_id": _as_nonempty_text(row["route_id"], "route_id"),
        "action_source": _as_nonempty_text(row["action_source"], "action_source").upper(),
        "next_state_id": _as_nonempty_text(row["next_state_id"], "next_state_id"),
        "transition_id": _as_nonempty_text(row["transition_id"], "transition_id"),
    }
    if result["action_source"] not in _ACTION_SOURCE_SET:
        raise ValueError("unknown action_source: {}".format(result["action_source"]))

    try:
        step_id = int(row["step_id"])
        action = int(row["action"])
    except (TypeError, ValueError) as exc:
        raise ValueError("step_id and action must be integers") from exc
    if step_id < 0:
        raise ValueError("step_id must be non-negative")
    if action < 0:
        raise ValueError("action must be non-negative")
    result["step_id"] = step_id
    result["action"] = action
    result["mask"] = _as_mask(row["mask"], "mask")
    if action >= result["mask"].size or not bool(result["mask"][action]):
        raise ValueError("action is outside mask")
    result["next_mask"] = _as_mask(row["next_mask"], "next_mask")
    if result["next_mask"].size != result["mask"].size:
        raise ValueError("next_mask action dimension mismatch")

    try:
        reward = float(row["reward"])
    except (TypeError, ValueError) as exc:
        raise ValueError("reward must be numeric") from exc
    if not np.isfinite(reward):
        raise ValueError("reward must be finite")
    result["reward"] = reward
    result["state_vector"] = _as_numeric_array(row["state_vector"], "state_vector", ndim=1)
    result["next_state_vector"] = _as_numeric_array(
        row["next_state_vector"], "next_state_vector", ndim=1
    )
    if result["next_state_vector"].shape != result["state_vector"].shape:
        raise ValueError("next_state_vector shape mismatch")
    result["state_depth"] = _as_numeric_array(row["state_depth"], "state_depth")
    result["next_state_depth"] = _as_numeric_array(
        row["next_state_depth"], "next_state_depth"
    )
    if result["next_state_depth"].shape != result["state_depth"].shape:
        raise ValueError("next_state_depth shape mismatch")
    if isinstance(row["done"], (str, bytes)):
        raise ValueError("done must be boolean-like")
    result["done"] = bool(row["done"])
    return result


def _normalise_metadata(metadata: Mapping[str, Any], *, transition_count: int) -> Dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    result = dict(metadata)
    if str(result.get("observation_contract", "")) != "reliable_exact_endpoint_snapshot":
        raise ValueError("multi-action replay requires reliable exact observations")
    result["schema_id"] = MULTI_ACTION_REPLAY_SCHEMA_ID
    result.setdefault("observation_source", "reliable_exact_endpoint_snapshot")
    result["transition_count"] = int(transition_count)
    result.setdefault("production_replay", False)
    result.setdefault("training_consumed", False)
    result["action_sources"] = list(ACTION_SOURCES)
    return result


def _check_unique(rows: Sequence[Mapping[str, Any]]) -> None:
    transition_ids = [str(row["transition_id"]) for row in rows]
    if len(set(transition_ids)) != len(transition_ids):
        raise ValueError("duplicate transition_id")
    state_action = [(str(row["state_id"]), int(row["action"])) for row in rows]
    if len(set(state_action)) != len(state_action):
        raise ValueError("duplicate state/action pair")


def _fixed_width(values: Sequence[str], minimum: int = 1) -> np.ndarray:
    width = max(minimum, max(len(value) for value in values) if values else minimum)
    return np.asarray(values, dtype="U{}".format(width))


def write_multi_action_replay(
    directory: Path,
    transitions: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Write one immutable diagnostic artifact and return its manifest."""

    root = Path(directory).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("multi-action replay directory is not empty: {}".format(root))
    rows = [_normalise_transition(row, index) for index, row in enumerate(transitions)]
    if not rows:
        raise ValueError("multi-action replay cannot be empty")
    _check_unique(rows)
    action_dim = int(rows[0]["mask"].size)
    vector_shape = rows[0]["state_vector"].shape
    depth_shape = rows[0]["state_depth"].shape
    for row in rows:
        if row["mask"].size != action_dim:
            raise ValueError("mask action dimension mismatch")
        if row["state_vector"].shape != vector_shape:
            raise ValueError("state_vector shape mismatch across transitions")
        if row["state_depth"].shape != depth_shape:
            raise ValueError("state_depth shape mismatch across transitions")

    manifest = _normalise_metadata(metadata, transition_count=len(rows))
    manifest.update(
        {
            "state_count": len({row["state_id"] for row in rows}),
            "action_count": action_dim,
            "vector_dim": int(vector_shape[0]),
            "depth_shape": list(depth_shape),
            "unique_state_action_pairs": len(rows),
        }
    )
    root.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for field in _STRING_FIELDS:
        arrays[field] = _fixed_width([str(row[field]) for row in rows])
    arrays["step_id"] = np.asarray([row["step_id"] for row in rows], dtype=np.int64)
    arrays["action"] = np.asarray([row["action"] for row in rows], dtype=np.int16)
    arrays["mask"] = np.asarray([row["mask"] for row in rows], dtype=np.uint8)
    arrays["reward"] = np.asarray([row["reward"] for row in rows], dtype=np.float32)
    arrays["state_vector"] = np.stack([row["state_vector"] for row in rows]).astype(np.float32)
    arrays["state_depth"] = np.stack([row["state_depth"] for row in rows]).astype(np.float32)
    arrays["next_state_vector"] = np.stack(
        [row["next_state_vector"] for row in rows]
    ).astype(np.float32)
    arrays["next_state_depth"] = np.stack(
        [row["next_state_depth"] for row in rows]
    ).astype(np.float32)
    arrays["next_mask"] = np.asarray([row["next_mask"] for row in rows], dtype=np.uint8)
    arrays["done"] = np.asarray([row["done"] for row in rows], dtype=np.uint8)
    tmp_path = root / "transitions.npz.tmp"
    final_path = root / "transitions.npz"
    try:
        with open(tmp_path, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(str(tmp_path), str(final_path))
        (root / "metadata.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return manifest


def _load_arrays(root: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    metadata_path = root / "metadata.json"
    transitions_path = root / "transitions.npz"
    if not metadata_path.is_file() or not transitions_path.is_file():
        raise FileNotFoundError("multi-action replay requires metadata.json and transitions.npz")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("schema_id", "")) != MULTI_ACTION_REPLAY_SCHEMA_ID:
        raise ValueError("multi-action replay schema mismatch")
    with np.load(str(transitions_path), allow_pickle=False) as loaded:
        missing = sorted(set(_ARRAY_FIELDS).difference(loaded.files))
        if missing:
            raise ValueError("multi-action replay missing fields: {}".format(", ".join(missing)))
        arrays = {field: np.asarray(loaded[field]) for field in _ARRAY_FIELDS}
    return arrays, metadata


def _validate_arrays(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    count = int(arrays["action"].shape[0])
    if count <= 0:
        raise ValueError("multi-action replay cannot be empty")
    for field, array in arrays.items():
        if array.shape[0] != count:
            raise ValueError("{} row count mismatch".format(field))
    action_dim = int(arrays["mask"].shape[1])
    if arrays["mask"].ndim != 2 or arrays["next_mask"].shape != arrays["mask"].shape:
        raise ValueError("mask shape mismatch")
    if arrays["state_vector"].ndim != 2 or arrays["next_state_vector"].shape != arrays["state_vector"].shape:
        raise ValueError("vector shape mismatch")
    if arrays["state_depth"].ndim < 2 or arrays["next_state_depth"].shape != arrays["state_depth"].shape:
        raise ValueError("depth shape mismatch")
    if not np.isfinite(arrays["reward"]).all():
        raise ValueError("multi-action replay contains non-finite rewards")
    for field in ("state_vector", "state_depth", "next_state_vector", "next_state_depth"):
        if not np.isfinite(arrays[field]).all():
            raise ValueError("multi-action replay contains non-finite {}".format(field))
    actions = arrays["action"].astype(np.int64, copy=False)
    masks = arrays["mask"].astype(bool, copy=False)
    if np.any(actions < 0) or np.any(actions >= action_dim):
        raise ValueError("action outside mask dimension")
    if not np.all(masks[np.arange(count), actions]):
        raise ValueError("action is outside mask")
    for field in ("state_id", "mission_id", "episode_id", "route_id", "action_source", "next_state_id", "transition_id"):
        if any(not str(value).strip() for value in arrays[field].tolist()):
            raise ValueError("{} contains an empty identity".format(field))
    sources = {str(value) for value in arrays["action_source"].tolist()}
    if not sources.issubset(_ACTION_SOURCE_SET):
        raise ValueError("unknown action_source")
    if len(set(arrays["transition_id"].tolist())) != count:
        raise ValueError("duplicate transition_id")
    pairs = list(zip(arrays["state_id"].tolist(), actions.tolist()))
    if len(set(pairs)) != count:
        raise ValueError("duplicate state/action pair")
    if int(metadata.get("transition_count", count)) != count:
        raise ValueError("metadata transition_count mismatch")
    if int(metadata.get("action_count", action_dim)) != action_dim:
        raise ValueError("metadata action_count mismatch")


def load_multi_action_replay(directory: Path, *, validate: bool = True) -> Dict[str, Any]:
    """Load the diagnostic artifact with ``allow_pickle=False``."""

    arrays, metadata = _load_arrays(Path(directory).expanduser().resolve())
    if validate:
        _validate_arrays(arrays, metadata)
    result = dict(arrays)
    result["metadata"] = metadata
    return result


def audit_multi_action_replay(
    directory: Path, *, min_states: int = 500, min_transitions: int = 2500
) -> Dict[str, Any]:
    """Return an auditable report without changing the artifact."""

    loaded = load_multi_action_replay(directory, validate=True)
    actions = loaded["action"].astype(np.int64, copy=False)
    state_ids = loaded["state_id"].astype(str)
    sources = loaded["action_source"].astype(str)
    transition_count = int(actions.size)
    state_count = int(np.unique(state_ids).size)
    _, state_action_counts = np.unique(state_ids, return_counts=True)
    state_action_counts = np.asarray(state_action_counts, dtype=np.int64)
    source_counts = {source: int(np.count_nonzero(sources == source)) for source in ACTION_SOURCES}
    failures = []
    if state_count < int(min_states):
        failures.append("state_count_below_minimum")
    if transition_count < int(min_transitions):
        failures.append("transition_count_below_minimum")
    return {
        "status": "PASS" if not failures else "FAIL",
        "schema_id": MULTI_ACTION_REPLAY_SCHEMA_ID,
        "state_count": state_count,
        "transition_count": transition_count,
        "mean_actions_per_state": float(transition_count / state_count) if state_count else 0.0,
        "median_actions_per_state": float(np.median(state_action_counts)) if state_count else 0.0,
        "min_actions_per_state": int(state_action_counts.min()) if state_count else 0,
        "max_actions_per_state": int(state_action_counts.max()) if state_count else 0,
        "unique_action_ratio": float(transition_count / transition_count) if transition_count else 0.0,
        "unique_state_action_pairs": transition_count,
        "unique_action_count": int(np.unique(actions).size),
        "primitive_diversity_ratio": float(np.unique(actions).size / int(loaded["metadata"].get("action_count", 0))) if int(loaded["metadata"].get("action_count", 0)) else 0.0,
        "identity_complete_ratio": 1.0,
        "source_counts": source_counts,
        "source_ratios": {
            source: float(count / transition_count) if transition_count else 0.0
            for source, count in source_counts.items()
        },
        "done_count": int(np.count_nonzero(loaded["done"])),
        "failures": failures,
        "production_replay": bool(loaded["metadata"].get("production_replay", True)),
        "training_consumed": bool(loaded["metadata"].get("training_consumed", True)),
    }


def plan_action_sources(
    mask: Any,
    *,
    bc_action: int,
    teacher_action: Optional[int] = None,
    neighbor_actions: Sequence[int] = (),
    random_count: int = 3,
    seed: int = 0,
) -> List[Tuple[str, int]]:
    """Plan deterministic legal action candidates for one frozen state.

    The planner only chooses action IDs.  It never evaluates an action or
    changes a policy; the collector is responsible for executing each action
    from a fresh replayed prefix and recording the resulting transition.
    """

    valid = _as_mask(mask, "mask")
    selected = set()
    result: List[Tuple[str, int]] = []

    def add(source: str, value: int) -> None:
        action = int(value)
        if action < 0 or action >= valid.size or not bool(valid[action]) or action in selected:
            return
        selected.add(action)
        result.append((source, action))

    add("BC", int(bc_action))
    if teacher_action is not None:
        add("TEACHER", int(teacher_action))
    for neighbor in neighbor_actions:
        add("NEIGHBOR", int(neighbor))
    if int(random_count) < 0:
        raise ValueError("random_count must be non-negative")
    available = np.asarray([action for action in np.flatnonzero(valid) if int(action) not in selected])
    rng = np.random.default_rng(int(seed))
    if available.size and int(random_count):
        order = rng.permutation(available.size)[: int(random_count)]
        for index in order.tolist():
            add("RANDOM", int(available[index]))
    return result


__all__ = [
    "ACTION_SOURCES",
    "MULTI_ACTION_REPLAY_SCHEMA_ID",
    "audit_multi_action_replay",
    "load_multi_action_replay",
    "plan_action_sources",
    "write_multi_action_replay",
]
