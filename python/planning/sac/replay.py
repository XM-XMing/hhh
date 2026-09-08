"""Identity-bearing persistent replay owned by the SAC experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from planning.common.atomic import write_json_atomic
from planning.sac.contract import SAC_REPLAY_CONTRACT_ID


_NUMERIC_FIELDS = {
    "depth": np.float16,
    "vector": np.float32,
    "action_mask": np.uint8,
    "action": np.int16,
    "reward": np.float32,
    "next_depth": np.float16,
    "next_vector": np.float32,
    "next_action_mask": np.uint8,
    "done": np.uint8,
}
_IDENTITY_FIELDS = (
    "mission_id",
    "episode_id",
    "step_id",
    "behavior_policy_version",
    "termination_reason",
    "runtime_instance_id",
    "transition_id",
    "actor_log_prob",
    "policy_entropy",
    "bc_kl",
    "realized_return",
    "argmax_flip",
    "valid_action_count",
    "action_mask_valid",
)


def _array_shapes(capacity: int, depth_shape: Tuple[int, ...], vector_dim: int, action_dim: int):
    return {
        "depth": (capacity, 1) + tuple(depth_shape),
        "vector": (capacity, vector_dim),
        "action_mask": (capacity, action_dim),
        "action": (capacity,),
        "reward": (capacity,),
        "next_depth": (capacity, 1) + tuple(depth_shape),
        "next_vector": (capacity, vector_dim),
        "next_action_mask": (capacity, action_dim),
        "done": (capacity,),
    }


def _validate_identity(record: Mapping[str, Any], *, path: str = "transition") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for field in _IDENTITY_FIELDS:
        if field not in record:
            raise ValueError("{} missing {}".format(path, field))
        result[field] = record[field]
    for field in ("mission_id", "episode_id", "behavior_policy_version", "runtime_instance_id", "transition_id"):
        if not str(result[field]).strip():
            raise ValueError("{} {} must be non-empty".format(path, field))
        result[field] = str(result[field])
    for field in ("step_id", "valid_action_count"):
        value = int(result[field])
        if value < 0:
            raise ValueError("{} {} must be non-negative".format(path, field))
        result[field] = value
    for field in ("actor_log_prob", "policy_entropy", "bc_kl", "realized_return"):
        value = float(result[field])
        if not np.isfinite(value):
            raise ValueError("{} {} must be finite".format(path, field))
        result[field] = value
    result["argmax_flip"] = bool(result["argmax_flip"])
    result["action_mask_valid"] = bool(result["action_mask_valid"])
    return result


class SACReplayBuffer:
    """Append-only episode-commit replay with explicit identity sidecar."""

    def __init__(self, directory: Path, metadata: Mapping[str, Any], arrays: Mapping[str, Any], *, read_only: bool):
        self.directory = Path(directory).expanduser().resolve()
        self.metadata = dict(metadata)
        self.arrays = dict(arrays)
        self.read_only = bool(read_only)
        self.capacity = int(self.metadata["capacity"])
        self.size = int(self.metadata.get("size", 0))
        self.total_added = int(self.metadata.get("total_added", self.size))
        self.depth_shape = tuple(int(value) for value in self.metadata["depth_shape"])
        self.vector_dim = int(self.metadata["vector_dim"])
        self.action_dim = int(self.metadata["action_dim"])
        self._identity_records = []
        self._transition_ids = set()
        self._load_identity_sidecar()

    @classmethod
    def create(
        cls,
        directory: Path,
        *,
        capacity: int,
        depth_shape: Sequence[int],
        vector_dim: int,
        action_dim: int,
        provenance: Mapping[str, Any],
    ) -> "SACReplayBuffer":
        root = Path(directory).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("SAC replay directory is not empty: {}".format(root))
        root.mkdir(parents=True, exist_ok=True)
        capacity = int(capacity)
        depth_shape = tuple(int(value) for value in depth_shape)
        vector_dim = int(vector_dim)
        action_dim = int(action_dim)
        if capacity <= 0 or not depth_shape or vector_dim <= 0 or action_dim <= 0:
            raise ValueError("invalid SAC replay shape/capacity")
        shapes = _array_shapes(capacity, depth_shape, vector_dim, action_dim)
        arrays = {
            name: np.lib.format.open_memmap(
                str(root / (name + ".npy")),
                mode="w+",
                dtype=_NUMERIC_FIELDS[name],
                shape=shape,
            )
            for name, shape in shapes.items()
        }
        metadata = {
            "contract_id": SAC_REPLAY_CONTRACT_ID,
            "capacity": capacity,
            "size": 0,
            "total_added": 0,
            "position": 0,
            "depth_shape": list(depth_shape),
            "vector_dim": vector_dim,
            "action_dim": action_dim,
            "depth_dtype": "float16",
            "identity_sidecar": "identities.jsonl",
            **dict(provenance),
        }
        write_json_atomic(root / "metadata.json", metadata, trailing_newline=True)
        (root / "identities.jsonl").touch()
        for value in arrays.values():
            value.flush()
        return cls(root, metadata, arrays, read_only=False)

    @classmethod
    def open(cls, directory: Path, *, read_only: bool = False) -> "SACReplayBuffer":
        root = Path(directory).expanduser().resolve()
        metadata_path = root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("SAC replay metadata is missing: {}".format(metadata_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("contract_id", "")) != SAC_REPLAY_CONTRACT_ID:
            raise ValueError("SAC replay contract mismatch")
        shapes = _array_shapes(
            int(metadata["capacity"]),
            tuple(int(value) for value in metadata["depth_shape"]),
            int(metadata["vector_dim"]),
            int(metadata["action_dim"]),
        )
        mode = "r" if read_only else "r+"
        arrays = {
            name: np.lib.format.open_memmap(
                str(root / (name + ".npy")), mode=mode, dtype=_NUMERIC_FIELDS[name], shape=shape
            )
            for name, shape in shapes.items()
        }
        return cls(root, metadata, arrays, read_only=read_only)

    def _load_identity_sidecar(self) -> None:
        path = self.directory / "identities.jsonl"
        if not path.exists():
            if self.size:
                raise ValueError("SAC replay identity sidecar is missing")
            return
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = _validate_identity(
                    json.loads(line), path="identities.jsonl:{}".format(line_number)
                )
                transition_id = record["transition_id"]
                if transition_id in self._transition_ids:
                    raise ValueError("duplicate SAC transition_id {}".format(transition_id))
                self._transition_ids.add(transition_id)
                self._identity_records.append(record)
        if len(self._identity_records) != self.size:
            raise ValueError(
                "SAC replay identity rows {} != metadata size {}".format(
                    len(self._identity_records), self.size
                )
            )

    def _validate_transition(self, transition: Mapping[str, Any], index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        identity = _validate_identity(transition, path="transition[{}]".format(index))
        numeric: Dict[str, Any] = {}
        for name, dtype in _NUMERIC_FIELDS.items():
            if name not in transition:
                raise ValueError("transition[{}] missing {}".format(index, name))
            value = np.asarray(transition[name])
            expected = self.arrays[name].shape[1:]
            if tuple(value.shape) != tuple(expected):
                raise ValueError(
                    "transition[{}] {} shape {} != {}".format(index, name, value.shape, expected)
                )
            value = value.astype(dtype, copy=False)
            if name not in ("action_mask", "next_action_mask", "done") and not np.isfinite(value).all():
                raise ValueError("transition[{}] {} is non-finite".format(index, name))
            numeric[name] = value
        mask = numeric["action_mask"].astype(bool)
        next_mask = numeric["next_action_mask"].astype(bool)
        action = int(numeric["action"])
        if action < 0 or action >= self.action_dim or not bool(mask[action]):
            raise ValueError("transition[{}] action is outside current mask".format(index))
        if not bool(numeric["done"]) and not bool(next_mask.any()):
            raise ValueError("transition[{}] non-terminal next mask is empty".format(index))
        if not identity["action_mask_valid"]:
            raise ValueError("transition[{}] action_mask_valid is false".format(index))
        if identity["transition_id"] in self._transition_ids:
            raise ValueError("duplicate SAC transition_id {}".format(identity["transition_id"]))
        return identity, numeric

    def add_batch(self, transitions: Iterable[Mapping[str, Any]]) -> None:
        if self.read_only:
            raise PermissionError("SAC replay is read-only")
        values = list(transitions)
        if not values:
            raise ValueError("SAC replay cannot commit an empty episode")
        if self.size + len(values) > self.capacity:
            raise RuntimeError("SAC replay capacity exceeded")
        identities = []
        numeric_values = []
        seen = set()
        for index, transition in enumerate(values):
            identity, numeric = self._validate_transition(transition, index)
            if identity["transition_id"] in seen:
                raise ValueError("duplicate transition_id within SAC episode")
            seen.add(identity["transition_id"])
            identities.append(identity)
            numeric_values.append(numeric)
        start = self.size
        end = start + len(values)
        for name in _NUMERIC_FIELDS:
            self.arrays[name][start:end] = np.stack([item[name] for item in numeric_values], axis=0)
            self.arrays[name].flush()
        sidecar = self.directory / "identities.jsonl"
        with sidecar.open("a", encoding="utf-8") as stream:
            for identity in identities:
                stream.write(json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._identity_records.extend(identities)
        self._transition_ids.update(item["transition_id"] for item in identities)
        self.size = end
        self.total_added += len(values)
        self.metadata["size"] = int(self.size)
        self.metadata["total_added"] = int(self.total_added)
        self.metadata["position"] = int(self.size)
        write_json_atomic(self.directory / "metadata.json", self.metadata, trailing_newline=True)

    def sample_indices(self, batch_size: int, *, rng: np.random.RandomState) -> np.ndarray:
        count = int(batch_size)
        if count <= 0 or self.size < count:
            raise ValueError("SAC replay is smaller than requested batch")
        return np.asarray(rng.randint(0, self.size, size=count), dtype=np.int64)

    def batch_from_indices(self, indices: Sequence[int], *, torch, device) -> Dict[str, Any]:
        values_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if values_indices.size <= 0 or np.any(values_indices < 0) or np.any(values_indices >= self.size):
            raise ValueError("SAC replay batch indices are outside committed rows")
        result: Dict[str, Any] = {}
        for name in _NUMERIC_FIELDS:
            values = np.asarray(self.arrays[name][values_indices]).copy()
            if name in ("action_mask", "next_action_mask", "done"):
                dtype = torch.bool if name != "done" else torch.float32
            elif name == "action":
                dtype = torch.int64
            else:
                dtype = torch.float32
            result[name] = torch.from_numpy(values).to(device=device, dtype=dtype)
        return result

    def sample(self, batch_size: int, *, rng: np.random.RandomState, torch, device) -> Dict[str, Any]:
        return self.batch_from_indices(
            self.sample_indices(batch_size, rng=rng), torch=torch, device=device
        )

    def audit(self) -> Dict[str, Any]:
        identity_count = len(self._identity_records)
        duplicate_ids = identity_count - len(self._transition_ids)
        nonfinite = 0
        invalid_actions = 0
        for index in range(self.size):
            for name in ("depth", "vector", "reward", "next_depth", "next_vector"):
                if not np.isfinite(np.asarray(self.arrays[name][index])).all():
                    nonfinite += 1
                    break
            mask = np.asarray(self.arrays["action_mask"][index]).astype(bool)
            action = int(self.arrays["action"][index])
            if action < 0 or action >= self.action_dim or not bool(mask[action]):
                invalid_actions += 1
        return {
            "contract_id": SAC_REPLAY_CONTRACT_ID,
            "rows": int(self.size),
            "total_added": int(self.total_added),
            "identity_rows": int(identity_count),
            "duplicate_transition_id_count": int(duplicate_ids),
            "nonfinite_row_count": int(nonfinite),
            "invalid_action_count": int(invalid_actions),
            "unique_episode_count": len({item["episode_id"] for item in self._identity_records}),
            "unique_mission_count": len({item["mission_id"] for item in self._identity_records}),
            "behavior_policy_versions": sorted({item["behavior_policy_version"] for item in self._identity_records}),
            "identity_complete": bool(identity_count == self.size and duplicate_ids == 0),
        }

    @property
    def identity_records(self):
        return tuple(dict(item) for item in self._identity_records)

    def metadata_sha256(self) -> str:
        import hashlib

        return hashlib.sha256((self.directory / "metadata.json").read_bytes()).hexdigest()

    def flush(self) -> None:
        for value in self.arrays.values():
            value.flush()

    def close(self) -> None:
        self.flush()
        self.arrays.clear()


__all__ = ["SACReplayBuffer"]
