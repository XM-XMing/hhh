"""Persistent uint8-depth replay owned by the discrete AWAC trainer.

The replay contract contains only policy/training fields.  Candidate proposal,
trust-gate and rollback evidence belongs to historical diagnostics and is not
part of a formal AWAC transition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from planning.awac.contract import (
    AWAC_REPLAY_CONTRACT_ID,
    AWAC_REWARD_SCALE,
    AWAC_REWARD_SCALE_OWNER,
    replay_contract_sha256,
)
from planning.awac.checkpoint import (
    CALIBRATION_REPLAY_PENDING_GENERATION_SCHEMA_ID,
    remove_calibration_transaction_file,
    write_calibration_transaction_json,
)
from planning.awac.interaction import behavior_source_contract, validate_behavior_source
from planning.common.atomic import write_json_atomic
from planning.common.hashing import file_sha256
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.reward import (
    REWARD_CONTRACT_ID,
    reward_contract_sha256 as current_reward_contract_sha256,
)


_FIELD_SPECS = {
    "depth": np.uint8,
    "vector": np.float32,
    "action_mask": np.uint8,
    "action": np.int16,
    "reward": np.float32,
    "next_depth": np.uint8,
    "next_vector": np.float32,
    "next_action_mask": np.uint8,
    "done": np.uint8,
    "behavior_source": np.uint8,
}


def move_awac_batch_to_device(batch: Mapping[str, Any], *, device) -> Dict[str, Any]:
    """Move a transient AWAC computation batch without changing replay storage."""

    if not isinstance(batch, Mapping) or not batch:
        raise ValueError("AWAC computation batch must be a non-empty mapping")
    moved = {}
    for name, value in batch.items():
        move = getattr(value, "to", None)
        if not callable(move):
            raise TypeError("AWAC computation batch field {} is not a tensor".format(name))
        moved[name] = move(device=device)
    return moved


def _array_shapes(*, capacity: int, depth_shape: Tuple[int, ...], vector_dim: int, action_dim: int):
    return {
        "depth": (capacity,) + depth_shape,
        "vector": (capacity, vector_dim),
        "action_mask": (capacity, action_dim),
        "action": (capacity,),
        "reward": (capacity,),
        "next_depth": (capacity,) + depth_shape,
        "next_vector": (capacity, vector_dim),
        "next_action_mask": (capacity, action_dim),
        "done": (capacity,),
        "behavior_source": (capacity,),
    }


def _validate_observation_metadata(metadata: Mapping[str, Any]) -> None:
    if str(metadata.get("observation_contract", "")) != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("AWAC replay observation contract mismatch")
    if str(metadata.get("observation_source", "")) != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("AWAC replay observation source mismatch")
    if str(metadata.get("observation_semantics", "")) != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("AWAC replay observation semantics mismatch")
    if int(metadata.get("legacy_replay_transition_count", -1)) != 0:
        raise ValueError("AWAC replay contains legacy transitions")
    required = (
        "bc_checkpoint_sha256", "task_contract_id", "task_contract_sha256",
        "mpl_contract_sha256", "run_identity", "run_contract_sha256",
        "reward_contract_sha256", "reward_scale", "reward_scale_owner",
    )
    missing = [name for name in required if not str(metadata.get(name, ""))]
    if missing:
        raise ValueError("AWAC replay missing formal provenance: {}".format(", ".join(missing)))
    if str(metadata.get("reward_contract_id", "")) != REWARD_CONTRACT_ID:
        raise ValueError("AWAC replay reward contract mismatch")
    if str(metadata.get("reward_contract_sha256", "")) != current_reward_contract_sha256():
        raise ValueError("AWAC replay reward contract SHA mismatch")
    if float(metadata.get("reward_scale", -1.0)) != float(AWAC_REWARD_SCALE):
        raise ValueError("AWAC replay reward scale mismatch")
    if str(metadata.get("reward_scale_owner", "")) != AWAC_REWARD_SCALE_OWNER:
        raise ValueError("AWAC replay reward scale owner mismatch")
    if str(metadata.get("reward_storage_semantics", "")) != "raw_environment_reward":
        raise ValueError("AWAC replay reward storage semantics mismatch")
    if str(metadata.get("replay_contract_sha256", "")) != replay_contract_sha256():
        raise ValueError("AWAC replay contract SHA mismatch")


def _validate_behavior_metadata(metadata: Mapping[str, Any]) -> None:
    if dict(metadata.get("behavior_source_enum", {})) != behavior_source_contract():
        raise ValueError("AWAC replay behavior source enum mismatch")
    phase = str(metadata.get("behavior_source_phase", "unspecified"))
    if phase not in ("unspecified", "critic_calibration", "awac_training"):
        raise ValueError("AWAC replay behavior source phase mismatch")
    if int(metadata.get("privileged_field_count", 0)) != 0:
        raise ValueError("AWAC replay contains privileged fields")


class AWACReplayBuffer:
    """Fixed-capacity persistent ring buffer for formal AWAC transitions."""

    def __init__(
        self,
        directory: Path,
        metadata: Dict,
        arrays: Dict[str, np.memmap],
        *,
        read_only: bool = False,
    ):
        self.directory = Path(directory)
        self.metadata = dict(metadata)
        self.arrays = dict(arrays)
        self._read_only = bool(read_only)
        self.capacity = int(metadata["capacity"])
        self.depth_shape = tuple(int(value) for value in metadata["depth_shape"])
        self.vector_dim = int(metadata["vector_dim"])
        self.action_dim = int(metadata["action_dim"])
        self.size = int(metadata.get("size", 0))
        self.position = int(metadata.get("position", 0))
        self.total_added = int(metadata.get("total_added", self.size))
        self._calibration_checkpoint_recovery_state = None
        if not 0 <= self.size <= self.capacity:
            raise ValueError("AWAC replay size is outside capacity")
        if not 0 <= self.position < self.capacity:
            raise ValueError("AWAC replay position is outside capacity")

    @classmethod
    def create(
        cls,
        directory: Path,
        *,
        capacity: int,
        depth_shape: Sequence[int],
        vector_dim: int,
        action_dim: int,
        training_config_sha256: str = "",
        observation_semantics: str = EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        observation_contract: str = EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        observation_source: str = EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        bc_checkpoint_sha256: str = "",
        task_contract_id: str = "",
        task_contract_sha256: str = "",
        mpl_contract_sha256: str = "",
        run_identity: str = "",
        run_contract_sha256: str = "",
        mission_source_sha256: str = "",
        mission_index_sha256: str = "",
        legacy_replay_transition_count: int = 0,
        reliable_v4_transition_count: int = 0,
        behavior_source_phase: str = "unspecified",
        reward_contract_sha256: str = "",
        reward_scale: float = AWAC_REWARD_SCALE,
        reward_scale_owner: str = AWAC_REWARD_SCALE_OWNER,
    ) -> "AWACReplayBuffer":
        root = Path(directory).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("replay directory is not empty: {}".format(root))
        root.mkdir(parents=True, exist_ok=True)
        capacity = int(capacity)
        depth_shape = tuple(int(value) for value in depth_shape)
        vector_dim = int(vector_dim)
        action_dim = int(action_dim)
        if capacity <= 0 or not depth_shape or vector_dim <= 0 or action_dim <= 0:
            raise ValueError("invalid AWAC replay shape/capacity")
        if str(observation_contract) != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
            raise ValueError("formal AWAC replay requires exact endpoint observations")
        if str(observation_source) != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
            raise ValueError("formal AWAC replay requires exact endpoint source")
        if str(observation_semantics) != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
            raise ValueError("formal AWAC replay observation semantics mismatch")
        if int(legacy_replay_transition_count) != 0:
            raise ValueError("formal AWAC replay cannot contain legacy transitions")
        if int(reliable_v4_transition_count) < 0:
            raise ValueError("reliable transition count must be non-negative")
        if str(reward_contract_sha256 or current_reward_contract_sha256()) != current_reward_contract_sha256():
            raise ValueError("formal AWAC replay reward contract SHA mismatch")
        if float(reward_scale) != float(AWAC_REWARD_SCALE):
            raise ValueError("formal AWAC replay reward scale is fixed at 0.10")
        if str(reward_scale_owner) != AWAC_REWARD_SCALE_OWNER:
            raise ValueError("formal AWAC replay reward scale owner mismatch")
        behavior_source_phase = str(behavior_source_phase)
        if behavior_source_phase not in (
            "unspecified",
            "critic_calibration",
            "awac_training",
        ):
            raise ValueError("unknown AWAC behavior source phase")
        required = {
            "bc_checkpoint_sha256": bc_checkpoint_sha256,
            "task_contract_id": task_contract_id,
            "task_contract_sha256": task_contract_sha256,
            "mpl_contract_sha256": mpl_contract_sha256,
            "run_identity": run_identity,
            "run_contract_sha256": run_contract_sha256,
        }
        missing = [name for name, value in required.items() if not str(value)]
        if missing:
            raise ValueError("formal AWAC replay provenance missing: {}".format(", ".join(missing)))
        shapes = _array_shapes(
            capacity=capacity, depth_shape=depth_shape,
            vector_dim=vector_dim, action_dim=action_dim,
        )
        arrays = {
            name: np.lib.format.open_memmap(
                str(root / "{}.npy".format(name)), mode="w+",
                dtype=_FIELD_SPECS[name], shape=shape,
            )
            for name, shape in shapes.items()
        }
        metadata = {
            "contract_id": AWAC_REPLAY_CONTRACT_ID,
            "replay_contract_sha256": replay_contract_sha256(),
            "capacity": capacity,
            "depth_shape": list(depth_shape),
            "vector_dim": vector_dim,
            "action_dim": action_dim,
            "size": 0,
            "position": 0,
            "total_added": 0,
            "training_config_sha256": str(training_config_sha256),
            "observation_semantics": str(observation_semantics),
            "observation_contract": str(observation_contract),
            "observation_source": str(observation_source),
            "bc_checkpoint_sha256": str(bc_checkpoint_sha256),
            "task_contract_id": str(task_contract_id),
            "task_contract_sha256": str(task_contract_sha256),
            "mpl_contract_sha256": str(mpl_contract_sha256),
            "run_identity": str(run_identity),
            "run_contract_sha256": str(run_contract_sha256),
            "mission_source_sha256": str(mission_source_sha256),
            "mission_index_sha256": str(
                mission_index_sha256 or mission_source_sha256
            ),
            "legacy_replay_transition_count": 0,
            "reliable_v4_transition_count": int(reliable_v4_transition_count),
            "reward_contract_id": REWARD_CONTRACT_ID,
            "reward_contract_sha256": current_reward_contract_sha256(),
            "reward_storage_semantics": "raw_environment_reward",
            "reward_scale": float(AWAC_REWARD_SCALE),
            "reward_scale_owner": AWAC_REWARD_SCALE_OWNER,
            "behavior_source_enum": behavior_source_contract(),
            "behavior_source_phase": behavior_source_phase,
            "privileged_field_count": 0,
        }
        buffer = cls(root, metadata, arrays)
        buffer.flush()
        return buffer

    @classmethod
    def open(
        cls, directory: Path, *, read_only: bool = False
    ) -> "AWACReplayBuffer":
        root = Path(directory).expanduser().resolve()
        metadata_path = root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("AWAC replay metadata missing: {}".format(metadata_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("contract_id", "")) != AWAC_REPLAY_CONTRACT_ID:
            raise ValueError("AWAC replay contract mismatch")
        _validate_behavior_metadata(metadata)
        _validate_observation_metadata(metadata)
        shapes = _array_shapes(
            capacity=int(metadata["capacity"]),
            depth_shape=tuple(int(value) for value in metadata["depth_shape"]),
            vector_dim=int(metadata["vector_dim"]), action_dim=int(metadata["action_dim"]),
        )
        arrays = {}
        for name, shape in shapes.items():
            path = root / "{}.npy".format(name)
            if not path.is_file():
                raise FileNotFoundError("AWAC replay field missing: {}".format(path))
            array = np.load(str(path), mmap_mode="r" if read_only else "r+")
            if tuple(array.shape) != tuple(shape) or array.dtype != np.dtype(_FIELD_SPECS[name]):
                raise ValueError("AWAC replay field mismatch: {}".format(name))
            arrays[name] = array
        return cls(root, metadata, arrays, read_only=bool(read_only))

    @classmethod
    def clone_from(
        cls,
        source: Union["AWACReplayBuffer", Path],
        directory: Path,
        *,
        run_identity: str,
        run_contract_sha256: str,
        training_config_sha256: str,
        behavior_source_phase: str = "awac_training",
        capacity: Optional[int] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> "AWACReplayBuffer":
        """Create an independent sparse replay snapshot from a source.

        Only the committed prefix is copied.  New ``.npy`` files retain the
        source's sparse/preallocated shape, so the clone never aliases source
        pages and subsequent online writes cannot mutate the calibration
        replay.  A path source is opened read-only for this operation.
        """

        owns_source = not isinstance(source, AWACReplayBuffer)
        source_buffer = (
            cls.open(Path(source), read_only=True) if owns_source else source
        )
        if not isinstance(source_buffer, AWACReplayBuffer):
            raise TypeError("replay clone source must be an AWACReplayBuffer or path")
        try:
            if not str(run_identity).strip():
                raise ValueError("standard replay run identity is required")
            if not str(run_contract_sha256).strip():
                raise ValueError("standard replay run contract SHA is required")
            if not str(training_config_sha256).strip():
                raise ValueError("standard replay training config SHA is required")
            if str(source_buffer.metadata.get("behavior_source_phase", "")) not in (
                "critic_calibration",
                "awac_training",
            ):
                raise ValueError("replay clone source phase is invalid")
            target_capacity = (
                int(source_buffer.capacity) if capacity is None else int(capacity)
            )
            if target_capacity != int(source_buffer.capacity):
                raise ValueError(
                    "independent replay clone must preserve source capacity"
                )

            source_metadata_path = (
                Path(source_buffer.directory).expanduser().resolve()
                / "metadata.json"
            )
            source_metadata_sha256 = file_sha256(source_metadata_path)
            clone = cls.create(
                Path(directory),
                capacity=target_capacity,
                depth_shape=source_buffer.depth_shape,
                vector_dim=source_buffer.vector_dim,
                action_dim=source_buffer.action_dim,
                training_config_sha256=str(training_config_sha256),
                observation_semantics=str(
                    source_buffer.metadata["observation_semantics"]
                ),
                observation_contract=str(
                    source_buffer.metadata["observation_contract"]
                ),
                observation_source=str(source_buffer.metadata["observation_source"]),
                bc_checkpoint_sha256=str(
                    source_buffer.metadata["bc_checkpoint_sha256"]
                ),
                task_contract_id=str(source_buffer.metadata["task_contract_id"]),
                task_contract_sha256=str(
                    source_buffer.metadata["task_contract_sha256"]
                ),
                mpl_contract_sha256=str(source_buffer.metadata["mpl_contract_sha256"]),
                run_identity=str(run_identity),
                run_contract_sha256=str(run_contract_sha256),
                mission_source_sha256=str(
                    source_buffer.metadata.get("mission_source_sha256", "")
                ),
                mission_index_sha256=str(
                    source_buffer.metadata.get("mission_index_sha256", "")
                ),
                legacy_replay_transition_count=int(
                    source_buffer.metadata.get("legacy_replay_transition_count", 0)
                ),
                reliable_v4_transition_count=int(
                    source_buffer.metadata.get("reliable_v4_transition_count", 0)
                ),
                behavior_source_phase=str(behavior_source_phase),
                reward_contract_sha256=str(
                    source_buffer.metadata["reward_contract_sha256"]
                ),
                reward_scale=float(source_buffer.metadata["reward_scale"]),
                reward_scale_owner=str(source_buffer.metadata["reward_scale_owner"]),
            )
            copy_count = (
                int(source_buffer.size)
                if int(source_buffer.size) < int(source_buffer.capacity)
                else int(source_buffer.capacity)
            )
            for name in _FIELD_SPECS:
                if copy_count:
                    clone.arrays[name][:copy_count] = source_buffer.arrays[name][
                        :copy_count
                    ]
            clone.size = int(source_buffer.size)
            clone.position = int(source_buffer.position)
            clone.total_added = int(source_buffer.total_added)
            clone.metadata.update(
                {
                    "source_replay_run_identity": str(
                        source_buffer.metadata.get("run_identity", "")
                    ),
                    "source_replay_metadata_sha256": source_metadata_sha256,
                    "source_replay_size": int(source_buffer.size),
                    "source_replay_position": int(source_buffer.position),
                    "source_replay_total_added": int(source_buffer.total_added),
                    "standard_online_starting_replay_size": int(source_buffer.size),
                    "standard_online_starting_replay_total_added": int(
                        source_buffer.total_added
                    ),
                    **dict(extra_metadata or {}),
                }
            )
            clone.flush()
            return clone
        finally:
            if owns_source:
                source_buffer.close()

    @staticmethod
    def _encode_depth(depth: np.ndarray, expected_shape: Tuple[int, ...]) -> np.ndarray:
        value = np.asarray(depth, dtype=np.float32)
        if tuple(value.shape) != tuple(expected_shape):
            raise ValueError("depth shape {} != {}".format(value.shape, expected_shape))
        if not np.isfinite(value).all():
            raise ValueError("non-finite AWAC replay depth")
        return np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)

    def add(self, **transition: Any) -> None:
        self.add_batch([transition])

    def _encode_transition(self, transition: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(transition, Mapping):
            raise TypeError("AWAC replay transition must be a mapping")
        required = set(_FIELD_SPECS)
        missing = sorted(required.difference(transition))
        if missing:
            raise KeyError("AWAC replay transition missing fields: {}".format(missing))
        unknown = sorted(set(transition).difference(required))
        if unknown:
            raise ValueError("AWAC replay transition contains unsupported fields: {}".format(unknown))
        vector = np.asarray(transition["vector"], dtype=np.float32).reshape(self.vector_dim)
        next_vector = np.asarray(transition["next_vector"], dtype=np.float32).reshape(self.vector_dim)
        action_mask = np.asarray(transition["action_mask"], dtype=np.bool_).reshape(self.action_dim)
        next_action_mask = np.asarray(transition["next_action_mask"], dtype=np.bool_).reshape(self.action_dim)
        action = int(transition["action"])
        reward = float(transition["reward"])
        done = bool(transition["done"])
        behavior_source = validate_behavior_source(int(transition["behavior_source"]))
        if action < 0 or action >= self.action_dim or not bool(action_mask[action]):
            raise ValueError("AWAC replay action is invalid under the stored mask")
        if not done and int(next_action_mask.sum()) == 0:
            raise ValueError("non-terminal AWAC transition has an empty next action mask")
        if not np.isfinite(vector).all() or not np.isfinite(next_vector).all() or not np.isfinite(reward):
            raise ValueError("non-finite AWAC replay transition")
        return {
            "depth": self._encode_depth(transition["depth"], self.depth_shape),
            "vector": vector,
            "action_mask": action_mask.astype(np.uint8),
            "action": np.int16(action),
            "reward": np.float32(reward),
            "next_depth": self._encode_depth(transition["next_depth"], self.depth_shape),
            "next_vector": next_vector,
            "next_action_mask": next_action_mask.astype(np.uint8),
            "done": np.uint8(done),
            "behavior_source": np.uint8(behavior_source),
        }

    def add_batch(self, transitions: Sequence[Mapping[str, Any]]) -> None:
        if self._read_only:
            raise RuntimeError("AWAC replay opened read-only")
        rows = list(transitions)
        if not rows:
            return
        if len(rows) > self.capacity:
            raise ValueError("AWAC replay batch exceeds capacity")
        encoded = [self._encode_transition(row) for row in rows]
        count = len(encoded)
        encoded_batch = {
            name: np.asarray([row[name] for row in encoded], dtype=dtype)
            for name, dtype in _FIELD_SPECS.items()
        }
        old_size, old_position, old_total = self.size, self.position, self.total_added
        self._record_calibration_pending_generation_write(
            count=count,
            old_total_added=old_total,
        )
        indices = (old_position + np.arange(count, dtype=np.int64)) % self.capacity
        backups = {name: np.array(self.arrays[name][indices], copy=True) for name in _FIELD_SPECS}
        try:
            for name in _FIELD_SPECS:
                self.arrays[name][indices] = encoded_batch[name]
            self.position = (old_position + count) % self.capacity
            self.size = min(self.capacity, old_size + count)
            self.total_added = old_total + count
            if self.metadata["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT:
                self.metadata["reliable_v4_transition_count"] = int(
                    self.metadata.get("reliable_v4_transition_count", 0)
                ) + count
        except BaseException:
            for name in _FIELD_SPECS:
                self.arrays[name][indices] = backups[name]
            self.size, self.position, self.total_added = old_size, old_position, old_total
            raise

    def sample(self, batch_size: int, *, rng: np.random.RandomState, torch, device) -> Dict:
        if self.size <= 0:
            raise RuntimeError("cannot sample an empty AWAC replay")
        count = int(batch_size)
        if count <= 0:
            raise ValueError("batch_size must be positive")
        return self.sample_indices(rng.randint(0, self.size, size=count), torch=torch, device=device)

    def sample_indices(self, indices, *, torch, device) -> Dict:
        if self.size <= 0:
            raise RuntimeError("cannot sample an empty AWAC replay")
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size <= 0 or bool((indices < 0).any()) or bool((indices >= self.size).any()):
            raise IndexError("AWAC replay sample index is outside committed buffer")
        def tensor(name, dtype):
            return torch.from_numpy(np.asarray(self.arrays[name][indices], dtype=dtype))
        batch = move_awac_batch_to_device({
            "depth": tensor("depth", np.float32),
            "vector": tensor("vector", np.float32),
            "action_mask": tensor("action_mask", np.bool_),
            "action": tensor("action", np.int64).long(),
            "reward": tensor("reward", np.float32),
            "next_depth": tensor("next_depth", np.float32),
            "next_vector": tensor("next_vector", np.float32),
            "next_action_mask": tensor("next_action_mask", np.bool_),
            "done": tensor("done", np.float32),
            "behavior_source": tensor("behavior_source", np.int64).long(),
        }, device=device)
        batch["depth"] = batch["depth"] / 255.0
        batch["next_depth"] = batch["next_depth"] / 255.0
        return batch

    def flush(self) -> None:
        if self._read_only:
            return
        for array in self.arrays.values():
            flush = getattr(array, "flush", None)
            if callable(flush):
                flush()
        self.metadata.update({
            "size": int(self.size),
            "position": int(self.position),
            "total_added": int(self.total_added),
        })
        write_json_atomic(self.directory / "metadata.json", self.metadata)

    @property
    def _calibration_pending_generation_path(self) -> Path:
        return self.directory / "calibration_replay_pending_generation.json"

    def _load_calibration_pending_generation(self) -> Dict[str, Any]:
        path = self._calibration_pending_generation_path
        if not path.is_file():
            raise FileNotFoundError(
                "calibration replay pending generation is missing: {}".format(path)
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("calibration replay pending generation is invalid") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("calibration replay pending generation is invalid")
        required = (
            "schema_id",
            "generation",
            "base_metadata",
            "base_metadata_sha256",
            "max_safe_total_added",
            "latest_planned_total_added",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                "calibration replay pending generation missing: {}".format(
                    ", ".join(missing)
                )
            )
        if payload["schema_id"] != CALIBRATION_REPLAY_PENDING_GENERATION_SCHEMA_ID:
            raise ValueError("calibration replay pending generation schema mismatch")
        if int(payload["generation"]) <= 0:
            raise ValueError("calibration replay pending generation is invalid")
        if not isinstance(payload["base_metadata"], Mapping):
            raise ValueError("calibration replay pending generation metadata is invalid")
        return dict(payload)

    def activate_calibration_checkpoint_generation(
        self,
        *,
        generation: int,
        committed_metadata: Mapping[str, Any],
    ) -> None:
        """Start the bounded rollback window after a committed checkpoint."""

        if self._read_only:
            raise RuntimeError("AWAC replay opened read-only")

        if int(generation) <= 0:
            raise ValueError("calibration checkpoint generation is invalid")
        expected = dict(committed_metadata)
        self.flush()
        if self.metadata != expected:
            raise ValueError("calibration replay committed metadata mismatch")
        metadata_path = self.directory / "metadata.json"
        state = {
            "schema_id": CALIBRATION_REPLAY_PENDING_GENERATION_SCHEMA_ID,
            "generation": int(generation),
            "base_metadata": expected,
            "base_metadata_sha256": file_sha256(metadata_path),
            "max_safe_total_added": int(
                int(expected["total_added"])
                + int(self.capacity)
                - int(expected["size"])
            ),
            "latest_planned_total_added": int(expected["total_added"]),
        }
        write_calibration_transaction_json(
            self._calibration_pending_generation_path, state
        )
        self._calibration_checkpoint_recovery_state = state

    def _record_calibration_pending_generation_write(
        self,
        *,
        count: int,
        old_total_added: int,
    ) -> None:
        state = self._calibration_checkpoint_recovery_state
        if state is None:
            return
        planned_total = int(old_total_added) + int(count)
        if planned_total > int(state["max_safe_total_added"]):
            raise RuntimeError(
                "calibration exact-resume replay would overwrite a committed "
                "generation before the next checkpoint"
            )
        state = dict(state)
        state["latest_planned_total_added"] = planned_total
        write_calibration_transaction_json(
            self._calibration_pending_generation_path, state
        )
        self._calibration_checkpoint_recovery_state = state

    def _validate_calibration_pending_tail(
        self,
        *,
        pending: Mapping[str, Any],
        committed_metadata: Mapping[str, Any],
        committed_metadata_sha256: str,
    ) -> None:
        """Allow only the journaled append-only tail to be rolled back.

        A generation recovery must not silently turn arbitrary metadata damage
        into a valid resume.  Before the next checkpoint this owner permits
        only rows announced in its durable pending-generation sidecar; the
        capacity guard in ``_record_calibration_pending_generation_write``
        guarantees those rows did not overwrite a committed row.
        """

        base = dict(committed_metadata)
        if str(pending["base_metadata_sha256"]) != str(
            committed_metadata_sha256
        ):
            raise ValueError(
                "calibration replay pending generation does not match committed checkpoint"
            )
        if dict(pending["base_metadata"]) != base:
            raise ValueError(
                "calibration replay pending generation does not match committed checkpoint"
            )
        base_total = int(base["total_added"])
        latest_total = int(pending["latest_planned_total_added"])
        max_safe_total = int(pending["max_safe_total_added"])
        if latest_total < base_total or latest_total > max_safe_total:
            raise ValueError(
                "calibration replay pending generation does not match committed checkpoint"
            )
        metadata_path = self.directory / "metadata.json"
        try:
            current = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                "calibration replay metadata does not match committed checkpoint"
            ) from exc
        if not isinstance(current, Mapping):
            raise ValueError(
                "calibration replay metadata does not match committed checkpoint"
            )
        current = dict(current)
        current_total = int(current.get("total_added", -1))
        appended = current_total - base_total
        if appended < 0 or current_total > latest_total:
            raise ValueError(
                "calibration replay metadata does not match committed checkpoint"
            )
        expected = dict(base)
        expected["size"] = min(self.capacity, int(base["size"]) + appended)
        expected["position"] = (int(base["position"]) + appended) % self.capacity
        expected["total_added"] = current_total
        if (
            expected["observation_contract"]
            == EXACT_ENDPOINT_OBSERVATION_CONTRACT
        ):
            expected["reliable_v4_transition_count"] = int(
                base.get("reliable_v4_transition_count", 0)
            ) + appended
        if current != expected:
            raise ValueError(
                "calibration replay metadata does not match committed checkpoint"
            )

    def restore_calibration_checkpoint_generation(
        self,
        *,
        generation: int,
        committed_metadata: Mapping[str, Any],
        committed_metadata_sha256: str,
    ) -> None:
        """Recover the replay view to the last committed, non-overwritten state."""

        if self._read_only:
            raise RuntimeError("AWAC replay opened read-only")

        expected = dict(committed_metadata)
        expected_generation = int(generation)
        pending_path = self._calibration_pending_generation_path
        if pending_path.exists():
            pending = self._load_calibration_pending_generation()
            pending_generation = int(pending["generation"])
            if pending_generation > expected_generation:
                raise ValueError(
                    "calibration replay pending generation is newer than its checkpoint"
                )
            if pending_generation == expected_generation:
                if int(pending["latest_planned_total_added"]) > int(
                    pending["max_safe_total_added"]
                ):
                    raise ValueError(
                        "calibration replay cannot recover after committed-row overwrite"
                    )
                self._validate_calibration_pending_tail(
                    pending=pending,
                    committed_metadata=expected,
                    committed_metadata_sha256=committed_metadata_sha256,
                )
                self.metadata = expected
                self.size = int(expected["size"])
                self.position = int(expected["position"])
                self.total_added = int(expected["total_added"])
                self.flush()
            # A lower-generation sidecar survived a crash after the newer
            # manifest commit.  It is not authoritative and is safe to drop.
            remove_calibration_transaction_file(pending_path)
        metadata_path = self.directory / "metadata.json"
        if file_sha256(metadata_path) != str(committed_metadata_sha256):
            raise ValueError(
                "calibration replay metadata does not match committed checkpoint"
            )
        self.activate_calibration_checkpoint_generation(
            generation=expected_generation,
            committed_metadata=expected,
        )

    def close(self) -> None:
        self.flush()
        self.arrays.clear()
