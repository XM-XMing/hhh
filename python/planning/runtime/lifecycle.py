"""Runtime-incarnation identity for reliable-v4 training workers.

This is a launch-contract seam.  It has no Unity, ZMQ, replay, or learner
dependency, so it can be checked before a child process is started.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from planning.common import canonical_json_sha256, write_json_atomic
from planning.runtime.worker import WorkerRuntimeSpec


RUNTIME_INSTANCE_LIFECYCLE_SCHEMA = "reliable_v4_runtime_instance_lifecycle_v1"
STARTUP_RUNTIME_INSTANCE_ID_REUSE = "STARTUP_RUNTIME_INSTANCE_ID_REUSE"


class RuntimeInstanceLifecycleError(RuntimeError):
    """Runtime incarnations cannot be proven unique for a training lineage."""


def _fail(message: str) -> None:
    raise RuntimeInstanceLifecycleError(
        "{}: {}".format(STARTUP_RUNTIME_INSTANCE_ID_REUSE, message)
    )


def resume_topology_sha256(topology: Mapping[str, object]) -> str:
    """Hash stable worker topology, excluding Unity-incarnation identity."""

    normalized = dict(topology)
    workers = []
    for worker in topology.get("workers", []):
        projected = dict(worker)
        projected.pop("runtime_instance_id", None)
        projected.pop("training_run_id", None)
        projected.pop("runtime_launch_nonce", None)
        workers.append(projected)
    normalized["workers"] = workers
    return canonical_json_sha256(normalized)


def allow_resume_lifecycle_migration(
    *,
    previous_config: Mapping[str, object],
    current_config: Mapping[str, object],
    previous_source_sha256: str,
    current_source_sha256: str,
    allowed_previous_source_sha256: Sequence[str],
) -> bool:
    """Allow only the explicit source/config delta for runtime lifecycle.

    This protects the canonical pre-actor checkpoint while refusing any
    accidental schedule, reward, normalizer, or learner change.
    """

    if str(previous_source_sha256) not in {
        str(value) for value in allowed_previous_source_sha256
    }:
        return False
    if not str(current_source_sha256) or previous_source_sha256 == current_source_sha256:
        return False
    ignored = {"source_code_sha256", "worker_topology_sha256"}
    prior = {key: value for key, value in previous_config.items() if key not in ignored}
    current = {key: value for key, value in current_config.items() if key not in ignored}
    return prior == current


def _segment_mapping(segment_id: str, specs: Sequence[WorkerRuntimeSpec]) -> dict:
    return {
        "segment_id": str(segment_id),
        "workers": [
            {
                "worker_id": int(spec.worker_id),
                "runtime_instance_id": str(spec.runtime_instance_id),
                "runtime_launch_nonce": str(spec.runtime_launch_nonce),
            }
            for spec in specs
        ],
    }


def register_runtime_segment(
    path: Path,
    *,
    training_run_id: str,
    segment_id: str,
    worker_specs: Sequence[WorkerRuntimeSpec],
) -> dict:
    """Persist a pre-launch runtime incarnation set, rejecting prior reuse."""

    destination = Path(path)
    specs = tuple(worker_specs)
    if not specs:
        _fail("worker specs are empty")
    if not str(training_run_id) or not str(segment_id):
        _fail("training_run_id and segment_id are required")
    if any(str(spec.training_run_id) != str(training_run_id) for spec in specs):
        _fail("worker spec training_run_id disagrees with lifecycle")
    ids = [str(spec.runtime_instance_id) for spec in specs]
    if len(ids) != len(set(ids)):
        _fail("new segment contains duplicate runtime_instance_id")

    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("schema") != RUNTIME_INSTANCE_LIFECYCLE_SCHEMA:
            _fail("lifecycle schema mismatch")
        if str(payload.get("training_run_id", "")) != str(training_run_id):
            _fail("training_run_id differs from existing lifecycle")
    else:
        payload = {
            "schema": RUNTIME_INSTANCE_LIFECYCLE_SCHEMA,
            "training_run_id": str(training_run_id),
            "segments": [],
        }
    segments = list(payload.get("segments", []))
    seen = {
        str(worker.get("runtime_instance_id", ""))
        for segment in segments
        for worker in segment.get("workers", [])
    }
    duplicate = sorted(set(ids) & seen)
    if duplicate:
        _fail("runtime_instance_id already registered: {}".format(",".join(duplicate)))
    segments.append(_segment_mapping(str(segment_id), specs))
    payload["segments"] = segments
    write_json_atomic(destination, payload, trailing_newline=True)
    return payload


def bootstrap_lifecycle_from_transition_ledger(
    path: Path, *, training_run_id: str, ledger_path: Path
) -> dict:
    """Reserve runtime IDs observed before lifecycle persistence existed.

    A canonical checkpoint may predate this lifecycle contract.  Its ledger is
    still authoritative for already-used IDs, so backfill it before a resumed
    segment gets a chance to launch a Unity process.
    """

    destination = Path(path)
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("schema") != RUNTIME_INSTANCE_LIFECYCLE_SCHEMA:
            _fail("lifecycle schema mismatch")
        if str(payload.get("training_run_id", "")) != str(training_run_id):
            _fail("training_run_id differs from existing lifecycle")
    else:
        payload = {
            "schema": RUNTIME_INSTANCE_LIFECYCLE_SCHEMA,
            "training_run_id": str(training_run_id),
            "segments": [],
        }
    known = {
        str(worker.get("runtime_instance_id", ""))
        for segment in payload.get("segments", [])
        for worker in segment.get("workers", [])
    }
    observed: dict[tuple[int, str], dict] = {}
    ledger = Path(ledger_path)
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            runtime_id = str(record.get("runtime_instance_id", ""))
            if not runtime_id:
                raise ValueError("ledger record has no runtime_instance_id")
            worker_id = int(record["worker_id"])
            observed[(worker_id, runtime_id)] = {
                "worker_id": worker_id,
                "runtime_instance_id": runtime_id,
                "runtime_launch_nonce": "ledger-backfill",
            }
    missing = [
        item for _key, item in sorted(observed.items())
        if item["runtime_instance_id"] not in known
    ]
    if missing:
        payload["segments"] = list(payload.get("segments", [])) + [{
            "segment_id": "ledger-backfill-{:03d}".format(
                len(payload.get("segments", []))
            ),
            "workers": missing,
        }]
        write_json_atomic(destination, payload, trailing_newline=True)
    elif not destination.exists():
        write_json_atomic(destination, payload, trailing_newline=True)
    return payload


def build_runtime_segment(
    *,
    training_run_id: str,
    segment_id: str,
    worker_specs: Sequence[WorkerRuntimeSpec],
    launch_nonce: str,
) -> Tuple[WorkerRuntimeSpec, ...]:
    """Return one new runtime incarnation for every logical worker.

    ``worker_id`` is deliberately preserved.  The runtime ID includes an
    explicit launch nonce, so an execution counter may safely start at zero in
    the new Unity process.
    """

    if not str(training_run_id):
        raise ValueError("training_run_id is required")
    if not str(segment_id):
        raise ValueError("segment_id is required")
    if not str(launch_nonce):
        raise ValueError("launch_nonce is required")
    return tuple(
        replace(
            spec,
            training_run_id=str(training_run_id),
            runtime_launch_nonce=str(launch_nonce),
            runtime_instance_id=(
                "{}-worker-{:02d}-runtime-{}".format(
                    training_run_id, int(spec.worker_id), launch_nonce
                )
            ),
        )
        for spec in worker_specs
    )


def validate_transition_ledger(
    records: Sequence[Mapping[str, object]], *, reject_conflicts: bool = False
) -> dict:
    """Validate run-wide execution identity provenance without transport I/O."""

    by_execution: dict[tuple[str, int], set[str]] = {}
    runtime_ids_by_worker: dict[str, set[str]] = {}
    for record in records:
        runtime_id = str(record.get("runtime_instance_id", ""))
        command_hash = str(record.get("command_sequence_hash", ""))
        if not runtime_id or not command_hash:
            raise ValueError("ledger record needs runtime_instance_id and command_sequence_hash")
        execution_id = int(record["execution_id"])
        worker_id = str(int(record["worker_id"]))
        by_execution.setdefault((runtime_id, execution_id), set()).add(command_hash)
        runtime_ids_by_worker.setdefault(worker_id, set()).add(runtime_id)

    duplicate_same_hash_count = sum(
        0
        for _key, _hashes in by_execution.items()
    )
    # A ledger record is the authoritative occurrence count.  Re-scan to
    # distinguish harmless repeated evidence from conflicting reuse.
    occurrences: dict[tuple[str, int, str], int] = {}
    for record in records:
        key = (
            str(record["runtime_instance_id"]),
            int(record["execution_id"]),
            str(record["command_sequence_hash"]),
        )
        occurrences[key] = occurrences.get(key, 0) + 1
    duplicate_same_hash_count = sum(value - 1 for value in occurrences.values())
    duplicate_different_hash_count = sum(
        max(0, len(hashes) - 1) for hashes in by_execution.values()
    )
    report = {
        "transition_count": len(records),
        "unique_runtime_execution_keys": len(by_execution),
        "duplicate_same_hash_count": duplicate_same_hash_count,
        "duplicate_different_hash_count": duplicate_different_hash_count,
        "runtime_instance_count": len(
            {runtime for runtime, _execution in by_execution}
        ),
        "runtime_segments_per_worker": {
            worker: len(runtime_ids)
            for worker, runtime_ids in sorted(runtime_ids_by_worker.items())
        },
    }
    if reject_conflicts and duplicate_different_hash_count:
        raise RuntimeInstanceLifecycleError(
            "PROTOCOL_ERROR: runtime/execution identity has conflicting command hashes"
        )
    return report


def _attach_json_metadata(path: Path, lifecycle: Mapping[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("lifecycle metadata target must be a JSON object")
    payload["runtime_instance_lifecycle"] = dict(lifecycle)
    write_json_atomic(path, payload, trailing_newline=True)


def _attach_checkpoint_metadata(path: Path, lifecycle: Mapping[str, Any]) -> None:
    """Add launch evidence after a segment without changing learner code."""

    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    payload["runtime_instance_lifecycle"] = dict(lifecycle)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, str(temporary))
    temporary.replace(path)


def persist_runtime_lifecycle_metadata(
    *,
    lifecycle_path: Path,
    manifest_path: Path,
    run_contract_path: Path,
    checkpoint_paths: Sequence[Path],
) -> dict:
    """Bind one lifecycle registry into all continuation artifacts."""

    lifecycle = json.loads(Path(lifecycle_path).read_text(encoding="utf-8"))
    if lifecycle.get("schema") != RUNTIME_INSTANCE_LIFECYCLE_SCHEMA:
        _fail("lifecycle schema mismatch while persisting metadata")
    _attach_json_metadata(Path(manifest_path), lifecycle)
    _attach_json_metadata(Path(run_contract_path), lifecycle)
    for checkpoint_path in checkpoint_paths:
        path = Path(checkpoint_path)
        if path.is_file():
            _attach_checkpoint_metadata(path, lifecycle)
    return lifecycle


__all__ = [
    "RUNTIME_INSTANCE_LIFECYCLE_SCHEMA",
    "STARTUP_RUNTIME_INSTANCE_ID_REUSE",
    "RuntimeInstanceLifecycleError",
    "build_runtime_segment",
    "bootstrap_lifecycle_from_transition_ledger",
    "allow_resume_lifecycle_migration",
    "register_runtime_segment",
    "resume_topology_sha256",
    "persist_runtime_lifecycle_metadata",
    "validate_transition_ledger",
]
