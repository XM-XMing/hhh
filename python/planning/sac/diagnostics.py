"""Deterministic replay and transaction diagnostics for experimental SAC.

This module is deliberately SAC-local.  It does not alter AWAC, BC, replay
semantics, or the Unity runtime contract; it only makes the state needed to
audit an SAC update durable and hashable.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np


def _update_digest(digest: "hashlib._Hash", value: Any) -> None:
    """Hash nested checkpoint/RNG/optimizer values deterministically."""

    if value is None:
        digest.update(b"N;")
        return
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"A;")
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b";")
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"M;")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_digest(digest, repr(key))
            _update_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"L;")
        for item in value:
            _update_digest(digest, item)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        digest.update(b"B;")
        digest.update(bytes(value))
        return
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(b":")
    digest.update(repr(value).encode("utf-8"))
    digest.update(b";")


def stable_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def optimizer_state_sha256(optimizer: Any) -> str:
    return stable_fingerprint(optimizer.state_dict())


def rng_fingerprint(value: Any) -> str:
    return stable_fingerprint(value)


def batch_indices_sha256(indices: Iterable[int]) -> str:
    values = np.asarray(list(indices), dtype=np.int64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def action_mask_sha256(mask: Any) -> str:
    values = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def cuda_determinism_metadata(torch: Any) -> Mapping[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "gpu_model": str(torch.cuda.get_device_name(0)) if cuda_available else None,
        "cuda_available": cuda_available,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def capture_rng_state(torch: Any, replay_rng: Any, *, worker_state: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    """Capture every RNG owner used by the SAC diagnostic runtime."""

    state = {
        "python_random": random.getstate(),
        "numpy_global": np.random.get_state(),
        "replay_sampler": replay_rng.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if worker_state is not None:
        state["workers"] = dict(worker_state)
    return state


def restore_rng_state(torch: Any, replay_rng: Any, state: Mapping[str, Any]) -> None:
    """Restore the deterministic RNG owners captured by ``capture_rng_state``.

    Worker snapshots are evidence carried by the state manifest.  Worker
    lifecycle restoration remains the runtime owner's responsibility; Actor
    proposals themselves do not consume worker policy RNG.  The replay
    sampler and process-local RNGs are restored here so every trust-region
    factor starts from the same transaction boundary.
    """

    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_global" in state:
        np.random.set_state(state["numpy_global"])
    if "replay_sampler" in state:
        replay_rng.set_state(state["replay_sampler"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class SACUpdateJournal:
    """Append-only, fsync-backed update journal.

    JSONL is intentionally the canonical durable representation.  A compact
    NPZ index is generated at run finalization for replay tooling; a truncated
    JSONL tail is ignored only after the last newline and is never treated as
    a committed update.
    """

    schema_id = "bc_initialized_discrete_sac_update_journal_v2"

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+", encoding="utf-8")
        self._stream.seek(0, os.SEEK_END)
        self.offset = int(self._stream.tell())
        self.record_count = self._count_committed_records()

    def _count_committed_records(self) -> int:
        count = 0
        with self.path.open("rb") as stream:
            for line in stream:
                if line.endswith(b"\n") and line.strip():
                    try:
                        json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError):
                        continue
                    count += 1
        return count

    def append(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = {"schema_id": self.schema_id, "journal_sequence": self.record_count, **dict(record)}
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default) + "\n").encode("utf-8")
        self._stream.write(encoded.decode("utf-8"))
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self.offset += len(encoded)
        self.record_count += 1
        return payload

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {"path": str(self.path), "offset": int(self.offset), "record_count": int(self.record_count)}

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError("value is not JSON serializable: {}".format(type(value).__name__))


def write_batch_index_npz(journal_path: Path, output_path: Path) -> int:
    """Materialize committed journal batch indices without trusting a tail."""

    records = []
    with Path(journal_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n") or not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if value.get("schema_id") != SACUpdateJournal.schema_id:
                continue
            records.append(value)
    max_batch = max((len(item.get("batch_indices", [])) for item in records), default=0)
    batches = np.full((len(records), max_batch), -1, dtype=np.int64)
    for row, record in enumerate(records):
        values = np.asarray(record.get("batch_indices", []), dtype=np.int64)
        batches[row, : values.size] = values
    np.savez_compressed(
        str(Path(output_path)),
        journal_sequence=np.asarray([item.get("journal_sequence", i) for i, item in enumerate(records)], dtype=np.int64),
        update_kind=np.asarray([str(item.get("update_kind", "")) for item in records]),
        update_index=np.asarray([int(item.get("update_index", -1)) for item in records], dtype=np.int64),
        replay_size=np.asarray([int(item.get("replay_size", -1)) for item in records], dtype=np.int64),
        batch_indices=batches,
    )
    return len(records)


__all__ = [
    "SACUpdateJournal",
    "action_mask_sha256",
    "batch_indices_sha256",
    "capture_rng_state",
    "restore_rng_state",
    "cuda_determinism_metadata",
    "optimizer_state_sha256",
    "rng_fingerprint",
    "stable_fingerprint",
    "write_batch_index_npz",
]
