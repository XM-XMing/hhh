"""P0-M2 v4 single-worker validation runner.

The dry-run path is intentionally small and deterministic.  The real runtime
path is added below this ledger seam; both paths emit the same per-primitive
record and checkpoint contract.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import faulthandler
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import zmq
from zmq.utils.monitor import recv_monitor_message

from planning.common import file_sha256
from planning.protocol.constants import PRIMITIVE_FRAME_COUNT, PROTOCOL_VERSION
from planning.protocol.msgpack import messagepack_bytes, messagepack_unpack
from planning.protocol.result import build_result_commit_wire, build_result_ready_wire
from planning.runtime.primitive_execution_observation_store import (
    IndexedEndpointObservationStore,
)
from planning.runtime.primitive_execution_command_transport import (
    ZmqPrimitiveExecutionCommandClient,
)
from planning.protocol.primitive_execution_command_v4 import build_command_wire
from planning.protocol.primitive_execution_result_consumer import (
    PrimitiveExecutionResultConsumer,
)
from planning.protocol.primitive_execution_result_receipt import (
    build_result_receipt_ack_wire,
)
from planning.runtime.reliable_endpoint_snapshot_provider import (
    BridgeSnapshotEndpointProvider,
    ZmqBridgeSnapshotRetriever,
)
from planning.runtime.bridge_identity import (
    bridge_identity_manifest,
    planning_bridge_binary,
)
from planning.runtime.ports import DirectRuntimePortProfile
_schema = import_module("planning.protocol.primitive_execution_schema" + "_" + "v4")
canonical_command_sequence_hash = _schema.canonical_command_sequence_hash


RUNNER_CONTRACT_ID = "P0-M2-v4-single-worker-validation"
LEDGER_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
FRAME_COUNT = PRIMITIVE_FRAME_COUNT
TERMINAL_STATUSES = frozenset({"COMPLETE", "CANCELLED", "FAILED", "REJECTED"})
DIAGNOSTIC_EXECUTION_ID = 44_000_000_000_054

REQUIRED_LEDGER_FIELDS = (
    "execution_id",
    "runtime_instance_id",
    "terminal_status",
    "requested_frame_count",
    "applied_frame_count",
    "last_applied_frame_index",
    "endpoint_state_id",
    "depth_id",
    "bridge_accepted",
    "bridge_duplicate",
    "bridge_conflict",
    "ack_status",
    "python_receive",
    "endpoint_lookup",
    "transition_commit",
    "replay_append",
    "duplicate_transition",
    "protocol_error",
    "pending_result",
)

_BOOL_FIELDS = frozenset(
    {
        "bridge_accepted",
        "bridge_duplicate",
        "bridge_conflict",
        "python_receive",
        "endpoint_lookup",
        "transition_commit",
        "replay_append",
        "duplicate_transition",
        "protocol_error",
        "pending_result",
    }
)


def current_memory_usage_bytes() -> int:
    """Return the process high-water RSS in bytes on Linux."""

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB.  Keep the conversion local to this
    # runner so checkpoint values are comparable across one validation run.
    return int(usage.ru_maxrss) * 1024


def validate_ledger_record(record: Mapping[str, Any]) -> None:
    """Validate one public ledger record against the P0-M2 schema."""

    missing = [field for field in REQUIRED_LEDGER_FIELDS if field not in record]
    if missing:
        raise ValueError("ledger record missing fields: {}".format(",".join(missing)))
    execution_id = record["execution_id"]
    if isinstance(execution_id, bool) or not isinstance(execution_id, int) or execution_id < 0:
        raise ValueError("execution_id must be a non-negative integer")
    runtime_id = record["runtime_instance_id"]
    if not isinstance(runtime_id, str) or not runtime_id:
        raise ValueError("runtime_instance_id must be a non-empty string")
    status = record["terminal_status"]
    if status not in TERMINAL_STATUSES:
        raise ValueError("invalid terminal_status: {}".format(status))
    requested = record["requested_frame_count"]
    applied = record["applied_frame_count"]
    last_frame = record["last_applied_frame_index"]
    if requested != FRAME_COUNT:
        raise ValueError("requested_frame_count must be 25")
    if isinstance(applied, bool) or not isinstance(applied, int) or not 0 <= applied <= requested:
        raise ValueError("applied_frame_count must be in [0, 25]")
    expected_last = applied - 1 if applied else -1
    if last_frame != expected_last:
        raise ValueError("last_applied_frame_index does not match applied_frame_count")
    if status == "COMPLETE" and applied != FRAME_COUNT:
        raise ValueError("COMPLETE requires 25 applied frames")
    if status == "COMPLETE" and (
        record["endpoint_state_id"] is None or record["depth_id"] is None
    ):
        raise ValueError("COMPLETE requires endpoint state and depth identities")
    if not isinstance(record["ack_status"], str) or not record["ack_status"]:
        raise ValueError("ack_status must be a non-empty string")
    for field in _BOOL_FIELDS:
        if not isinstance(record[field], bool):
            raise ValueError("{} must be boolean".format(field))


def _hash_for_dry_run(execution_id: int) -> str:
    return hashlib.sha256(
        "{}:{}".format(RUNNER_CONTRACT_ID, execution_id).encode("ascii")
    ).hexdigest()


def build_dry_run_record(
    primitive_index: int,
    *,
    runtime_instance_id: str,
) -> Dict[str, Any]:
    """Build one deterministic record for the non-network smoke seam."""

    if primitive_index < 1:
        raise ValueError("primitive_index must be positive")
    execution_id = 10_000_000_000 + primitive_index
    endpoint_state_id = 1000 + primitive_index * FRAME_COUNT + (FRAME_COUNT - 1)
    return {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "execution_index": primitive_index,
        "execution_id": execution_id,
        "runtime_instance_id": runtime_instance_id,
        "terminal_status": "COMPLETE",
        "requested_frame_count": FRAME_COUNT,
        "applied_frame_count": FRAME_COUNT,
        "last_applied_frame_index": FRAME_COUNT - 1,
        "endpoint_state_id": endpoint_state_id,
        "depth_id": "depth-{}".format(endpoint_state_id),
        "result_payload_hash": _hash_for_dry_run(execution_id),
        "bridge_accepted": True,
        "bridge_duplicate": False,
        "bridge_conflict": False,
        "ack_status": "DURABLE_RECEIVED",
        "python_receive": True,
        "endpoint_lookup": True,
        "transition_commit": True,
        "replay_append": True,
        "duplicate_transition": False,
        "protocol_error": False,
        "pending_result": False,
    }


class _Counters:
    def __init__(self) -> None:
        self.values: Dict[str, int] = {
            "execution_total": 0,
            "complete_total": 0,
            "cancelled_total": 0,
            "failed_total": 0,
            "rejected_total": 0,
            "transition_commit_total": 0,
            "replay_append_total": 0,
            "duplicate_transition_total": 0,
            "endpoint_lookup_failure_total": 0,
            "endpoint_mismatch_total": 0,
            "protocol_error_total": 0,
            "duplicate_result_total": 0,
            "pending_result_final": 0,
        }

    def add(self, record: Mapping[str, Any]) -> None:
        status = str(record["terminal_status"]).lower()
        self.values["execution_total"] += 1
        self.values["{}_total".format(status)] += 1
        self.values["transition_commit_total"] += int(record["transition_commit"])
        self.values["replay_append_total"] += int(record["replay_append"])
        self.values["duplicate_transition_total"] += int(record["duplicate_transition"])
        self.values["endpoint_lookup_failure_total"] += int(
            not record["endpoint_lookup"]
        )
        self.values["endpoint_mismatch_total"] += int(
            record.get("endpoint_mismatch", False)
        )
        self.values["protocol_error_total"] += int(record["protocol_error"])
        self.values["duplicate_result_total"] += int(record["bridge_duplicate"])
        self.values["pending_result_final"] = int(record["pending_result"])

    def summary(self) -> Dict[str, int]:
        return dict(self.values)


class ValidationLedger:
    """Streaming JSONL ledger with bounded in-memory accounting."""

    def __init__(
        self,
        out_dir: Path,
        *,
        checkpoint_interval: int = 1000,
        mode: str,
        runtime_instance_id: str,
    ) -> None:
        if checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.out_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.mode = mode
        self.runtime_instance_id = runtime_instance_id
        self._counters = _Counters()
        self._ledger_file = (self.out_dir / "ledger.jsonl").open(
            "w", encoding="utf-8"
        )
        self._closed_summary: Optional[Dict[str, int]] = None
        self._write_json(
            self.out_dir / "runtime_manifest.json",
            {
                "contract_id": RUNNER_CONTRACT_ID,
                "ledger_schema_version": LEDGER_SCHEMA_VERSION,
                "mode": mode,
                "runtime_instance_id": runtime_instance_id,
                "checkpoint_interval": checkpoint_interval,
            },
        )

    def write_manifest(self, payload: Mapping[str, Any]) -> None:
        """Persist resolved runtime identity without retaining it in memory."""

        self._write_json(self.out_dir / "runtime_manifest.json", payload)

    def append(self, record: Mapping[str, Any]) -> None:
        validate_ledger_record(record)
        self._ledger_file.write(
            json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._ledger_file.flush()
        self._counters.add(record)
        processed = self._counters.values["execution_total"]
        if processed % self.checkpoint_interval == 0:
            self.write_checkpoint()

    def write_checkpoint(self) -> Dict[str, Any]:
        summary = self._counters.summary()
        checkpoint = {
            "contract_id": RUNNER_CONTRACT_ID,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "processed": summary["execution_total"],
            "success": summary["complete_total"],
            "failure": summary["execution_total"] - summary["complete_total"],
            "pending": summary["pending_result_final"],
            "memory_usage_bytes": current_memory_usage_bytes(),
            "metrics": summary,
        }
        path = self.checkpoint_dir / "checkpoint_{:06d}.json".format(
            checkpoint["processed"]
        )
        self._write_json(path, checkpoint)
        return checkpoint

    def close(self, *, extra_summary: Optional[Mapping[str, Any]] = None) -> Dict[str, int]:
        if self._closed_summary is not None:
            return dict(self._closed_summary)
        self._ledger_file.close()
        summary = self._counters.summary()
        summary_payload: Dict[str, Any] = {
            "contract_id": RUNNER_CONTRACT_ID,
            "mode": self.mode,
            **summary,
        }
        if extra_summary:
            summary_payload.update(dict(extra_summary))
        self._write_json(self.out_dir / "summary.json", summary_payload)
        self._closed_summary = dict(summary)
        return summary

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def run_dry_run(
    *,
    out_dir: Path,
    primitive_count: int = 1,
    checkpoint_interval: int = 1000,
    runtime_instance_id: str = "worker-00-p0-m2-dry-run",
) -> Dict[str, int]:
    """Emit deterministic ledger/checkpoint evidence without opening sockets."""

    if primitive_count < 1:
        raise ValueError("primitive_count must be positive")
    ledger = ValidationLedger(
        out_dir,
        checkpoint_interval=checkpoint_interval,
        mode="dry-run",
        runtime_instance_id=runtime_instance_id,
    )
    try:
        for primitive_index in range(1, primitive_count + 1):
            ledger.append(
                build_dry_run_record(
                    primitive_index,
                    runtime_instance_id=runtime_instance_id,
                )
            )
    except BaseException:
        ledger.close()
        raise
    return ledger.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_tcp(
    port: int,
    timeout_s: float,
    *,
    process: Optional[subprocess.Popen] = None,
    label: str = "TCP endpoint",
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                "{} process exited with code {}".format(label, process.returncode)
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            continue
    raise RuntimeError("{} did not become ready: {}".format(label, port))


def _wait_for_zmq_connected(
    socket_obj: Any,
    timeout_s: float,
    *,
    process: Optional[subprocess.Popen] = None,
    label: str = "ZMQ socket",
) -> None:
    monitor = socket_obj.get_monitor_socket()
    _wait_for_zmq_monitors_connected(
        [(label, monitor)], timeout_s, process=process
    )


def _wait_for_zmq_monitors_connected(
    monitors: Iterable[Tuple[str, Any]],
    timeout_s: float,
    *,
    process: Optional[subprocess.Popen] = None,
    close_monitors: bool = True,
) -> None:
    monitored = list(monitors)
    monitor_labels = {monitor: label for label, monitor in monitored}
    pending = set(monitor_labels)
    deadline = time.monotonic() + timeout_s
    try:
        while pending and time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    "{} process exited with code {}".format(
                        ", ".join(monitor_labels[monitor] for monitor in pending),
                        process.returncode,
                    )
                )
            poller = zmq.Poller()
            for monitor in pending:
                poller.register(monitor, zmq.POLLIN)
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            events = dict(poller.poll(min(remaining_ms, 100)))
            for monitor in list(pending):
                if monitor not in events:
                    continue
                event = recv_monitor_message(monitor, flags=zmq.NOBLOCK)
                if event["event"] == zmq.EVENT_CONNECTED:
                    pending.remove(monitor)
        if pending:
            raise RuntimeError(
                "{} did not connect before timeout".format(
                    ", ".join(monitor_labels[monitor] for monitor in pending)
                )
            )
    finally:
        if close_monitors:
            for _, monitor in monitored:
                monitor.close(0)


def _terminate_process(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8)


def _git_head(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "NOT_AVAILABLE"
    return result.stdout.strip()


class PrimitiveStageLedger:
    """Synchronous stage evidence for one primitive boundary."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")

    def record(
        self,
        stage: str,
        *,
        execution_id: int,
        primitive_index: int,
        child_liveness: Optional[Mapping[str, Any]] = None,
        **fields: Any,
    ) -> None:
        row: Dict[str, Any] = {
            "stage": str(stage),
            "execution_id": int(execution_id),
            "primitive_index": int(primitive_index),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
        }
        if child_liveness is not None:
            row["child_liveness"] = dict(child_liveness)
        row.update(fields)
        self._stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


class ProcessLifecycleSupervisor:
    """Write process observations from a process independent of the runner."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.out_dir / "process_lifecycle_registry.jsonl"
        self.output_path = self.out_dir / "process_lifecycle.jsonl"
        self.registry_path.touch()
        python_root = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(python_root) + os.pathsep + env.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "planning.runtime.process_supervisor",
                "--registry",
                str(self.registry_path),
                "--output",
                str(self.output_path),
                "--runner-pid",
                str(os.getpid()),
            ],
            cwd=str(self.out_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.register(
            "SPAWN",
            role="runner",
            pid=os.getpid(),
            command=[sys.executable, *sys.argv],
            spawn_monotonic_ns=time.monotonic_ns(),
        )

    def _append(self, event: Mapping[str, Any]) -> None:
        with self.registry_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event), sort_keys=True) + "\n")

    def register(self, event: str, **fields: Any) -> None:
        self._append({"event": event, **fields})

    def register_spawn(self, role: str, process: subprocess.Popen, command: Any) -> None:
        self.register(
            "SPAWN",
            role=role,
            pid=process.pid,
            command=list(command),
            spawn_monotonic_ns=time.monotonic_ns(),
        )

    def record_liveness(self, *, primitive_index: int, execution_id: int, processes: Mapping[str, Any]) -> None:
        self.register(
            "LIVENESS",
            primitive_index=int(primitive_index),
            execution_id=int(execution_id),
            processes=dict(processes),
            monotonic_ns=time.monotonic_ns(),
        )

    def record_exit(
        self,
        role: str,
        process: Optional[subprocess.Popen],
        *,
        cleanup_requested: bool,
        exit_type: str,
        log_paths: Iterable[Path] = (),
    ) -> None:
        if process is None:
            return
        returncode = process.returncode
        self.register(
            "EXIT",
            role=role,
            pid=process.pid,
            exit_monotonic_ns=time.monotonic_ns(),
            returncode=returncode,
            signal=(-returncode if returncode is not None and returncode < 0 else None),
            exit_type=exit_type,
            cleanup_requested=bool(cleanup_requested),
            last_output_lines={str(path): self._tail(path) for path in log_paths},
        )

    @staticmethod
    def _tail(path: Path, count: int = 5) -> list:
        try:
            return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
        except OSError:
            return []

    def close(self) -> None:
        if self.process.poll() is None:
            self.register("STOP", monotonic_ns=time.monotonic_ns())
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # The observer is diagnostic-only; do not kill it or any child.
                pass


def _result_mapping(raw: bytes) -> Dict[str, Any]:
    value = messagepack_unpack(raw)
    if not isinstance(value, dict):
        raise ValueError("PrimitiveExecutionResult must be a map")
    result = dict(value)
    for field in ("command_sequence_hash", "result_payload_hash"):
        if isinstance(result.get(field), bytes):
            result[field] = result[field].hex()
    return result


class ExecutionResultBuffer:
    """Demultiplex result messages by execution identity.

    The Unity result socket also carries commit acknowledgements.  A result
    for a later execution can therefore be read while the runner is waiting
    for an earlier commit acknowledgement.  This buffer keeps that result
    available without invoking transition accounting out of order.
    """

    def __init__(self) -> None:
        self._results: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _execution_id(result: Mapping[str, Any]) -> int:
        try:
            return int(result["execution_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("result must contain an integer execution_id") from error

    @staticmethod
    def _identity(result: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (
            int(result["execution_id"]),
            str(result.get("runtime_instance_id", "")),
            str(result.get("result_payload_hash", "")),
            str(result.get("command_sequence_hash", "")),
        )

    def offer(self, result: Mapping[str, Any]) -> str:
        """Buffer one result, returning BUFFERED or idempotent DUPLICATE."""

        execution_id = self._execution_id(result)
        candidate = dict(result)
        existing = self._results.get(execution_id)
        if existing is None:
            self._results[execution_id] = candidate
            return "BUFFERED"
        if self._identity(existing) == self._identity(candidate):
            return "DUPLICATE"
        raise ValueError(
            "conflicting result for execution_id={}".format(execution_id)
        )

    def same_identity(
        self, first: Mapping[str, Any], second: Mapping[str, Any]
    ) -> bool:
        """Return whether two results are the same immutable result payload."""

        return self._identity(first) == self._identity(second)

    def wait_for_result(self, execution_id: int) -> Optional[Dict[str, Any]]:
        """Return and remove the cached result for one execution, if present."""

        return self._results.pop(int(execution_id), None)

    @property
    def pending_execution_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._results))


class ExecutionHandoffDiagnostics:
    """Targeted, append-only evidence for one Python result handoff."""

    def __init__(self, path: Path, *, target_execution_id: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.path.with_suffix(".summary.json")
        self.target_execution_id = int(target_execution_id)
        self._stream = self.path.open("w", encoding="utf-8")
        self._events = []
        self._closed = False
        self._summary: Optional[Dict[str, Any]] = None

    def record(
        self,
        stage: str,
        event: str,
        *,
        execution_id: int,
        **fields: Any,
    ) -> None:
        if int(execution_id) != self.target_execution_id or self._closed:
            return
        row: Dict[str, Any] = {
            "timestamp_ns": time.time_ns(),
            "execution_id": self.target_execution_id,
            "stage": str(stage),
            "event": str(event),
        }
        row.update(fields)
        self._events.append(row)
        self._stream.write(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._stream.flush()

    def record_socket_recv(
        self,
        *,
        wait_execution_id: int,
        raw_message: bytes,
        decoded_execution_id: Optional[int],
        decoded_message: Mapping[str, Any],
    ) -> None:
        if (
            int(wait_execution_id) != self.target_execution_id
            and decoded_execution_id != self.target_execution_id
        ):
            return
        self.record(
            "PYTHON_SOCKET",
            "RECV",
            execution_id=self.target_execution_id,
            wait_execution_id=int(wait_execution_id),
            decoded_execution_id=decoded_execution_id,
            message_type=decoded_message.get("message_type"),
            status=decoded_message.get("status"),
            commit_status=decoded_message.get("commit_status"),
            raw_message_hex=bytes(raw_message).hex(),
            raw_message_size=len(raw_message),
        )

    def _first_loss_layer(self) -> str:
        events = self._events
        socket_received = any(
            row["stage"] == "PYTHON_SOCKET"
            and row["event"] == "RECV"
            and row.get("decoded_execution_id") == self.target_execution_id
            for row in events
        )
        if not socket_received:
            return "PYTHON_SOCKET_RECEIVE_LOSS"

        if any(
            row["stage"] == "BUFFER" and row["event"] == "conflict"
            for row in events
        ):
            return "PYTHON_BUFFER_LOSS"

        buffer_inserted = any(
            row["stage"] == "BUFFER" and row["event"] == "insert"
            for row in events
        )
        cache_hit = any(
            row["stage"] == "WAIT" and row["event"] == "cache_hit"
            for row in events
        )
        result_selected = any(
            row["stage"] == "WAIT" and row["event"] == "result_selected"
            for row in events
        )
        if buffer_inserted and not cache_hit and not result_selected:
            return "PYTHON_BUFFER_LOSS"
        if not result_selected:
            return "PYTHON_WAITER_LOSS"

        consumer_received = any(
            row["stage"] == "CONSUMER" and row["event"] == "received"
            for row in events
        )
        if not consumer_received:
            return "PYTHON_COMMIT_LOSS"
        outcomes = [
            row
            for row in events
            if row["stage"] == "CONSUMER" and row["event"] == "outcome"
        ]
        if not outcomes or outcomes[-1].get("status") not in {
            "COMMITTED",
            "NO_TRANSITION",
        }:
            return "PYTHON_COMMIT_LOSS"
        return "NONE"

    def close(self) -> Dict[str, Any]:
        if self._summary is not None:
            return dict(self._summary)
        self._stream.close()
        self._closed = True
        self._summary = {
            "target_execution_id": self.target_execution_id,
            "event_count": len(self._events),
            "first_loss_layer": self._first_loss_layer(),
            "events": list(self._events),
        }
        self.summary_path.write_text(
            json.dumps(self._summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return dict(self._summary)


class EndpointObservationLifecycleDiagnostics:
    """Append-only observation evidence without changing lookup semantics."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self._closed = False

    def _write(self, execution_id: int, event: str, **fields: Any) -> None:
        if self._closed:
            return
        row: Dict[str, Any] = {
            "execution_id": int(execution_id),
            "event": str(event),
            "timestamp_ns": time.time_ns(),
        }
        row.update(fields)
        self._stream.write(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._stream.flush()

    def record_execution_started(self, execution_id: int) -> None:
        self._write(execution_id, "EXECUTION_STARTED")

    def record_result_received(
        self, execution_id: int, result: Mapping[str, Any]
    ) -> None:
        observation_ref = result.get("endpoint_observation_ref")
        observation_ref = observation_ref if isinstance(observation_ref, Mapping) else {}
        self._write(
            execution_id,
            "RESULT_RECEIVED",
            endpoint_state_id=result.get("endpoint_state_id"),
            depth_id=observation_ref.get("depth_id", observation_ref.get("capture_id")),
            result_receive_time_ns=time.time_ns(),
        )

    def record_telemetry(
        self, execution_id: int, event: str, **fields: Any
    ) -> None:
        event_name = str(event)
        time_field = (
            "state_receive_time_ns"
            if event_name.startswith("STATE")
            else "depth_receive_time_ns"
        )
        self._write(execution_id, event_name, **fields, **{time_field: time.time_ns()})

    def record_store_event(
        self, execution_id: int, event: str, **fields: Any
    ) -> None:
        event_name = str(event)
        extra: Dict[str, Any] = {}
        if event_name == "STATE_INSERT":
            extra["state_insert_time_ns"] = time.time_ns()
        elif event_name == "DEPTH_INSERT":
            extra["depth_insert_time_ns"] = time.time_ns()
        elif event_name.startswith("LOOKUP_"):
            extra["lookup_time_ns"] = time.time_ns()
            extra["lookup_result"] = "HIT" if event_name == "LOOKUP_HIT" else "MISS"
        self._write(execution_id, event_name, **fields, **extra)

    def close(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True


class TelemetryTransportDiagnostics:
    """Append-only raw SUB receipt ledger; it has no effect on observation use."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self._sequence = 0
        self._closed = False

    def _write(self, row: Mapping[str, Any]) -> None:
        if self._closed:
            return
        self._stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        self._stream.flush()

    def record_socket_configuration(
        self, stream: str, socket_obj: Any, *, subscription: bytes
    ) -> None:
        try:
            conflate = int(socket_obj.getsockopt(zmq.CONFLATE))
        except zmq.ZMQError:
            conflate = None
        self._write(
            {
                "contract_id": "[DEBUG-P0M2-PYTHON-SUB-9c41]",
                "event": "SUB_CONFIGURATION",
                "side": "python_direct_sub",
                "stream": str(stream),
                "thread_id": threading.get_ident(),
                "rcvhwm": int(socket_obj.getsockopt(zmq.RCVHWM)),
                "conflate": conflate,
                "subscription_hex": bytes(subscription).hex(),
                "timestamp_ns": time.time_ns(),
            }
        )

    def record(
        self,
        stream: str,
        *,
        state_id: Optional[int],
        depth_id: Optional[str],
        execution_id: Optional[int],
        frame_index: Optional[int],
        raw_size: int,
        payload_size: Optional[int] = None,
        decode_ok: bool,
        receive_started_monotonic_ns: Optional[int] = None,
    ) -> int:
        if self._closed:
            return -1
        self._sequence += 1
        row: Dict[str, Any] = {
            "contract_id": "[DEBUG-P0M2-TELEMETRY-e542]",
            "side": "python_direct_sub",
            "stream": str(stream),
            "sequence": self._sequence,
            "state_id": state_id,
            "depth_id": depth_id,
            "execution_id": execution_id,
            "frame_index": frame_index,
            "raw_size": int(raw_size),
            "payload_size": payload_size,
            "receive_timestamp_ns": time.time_ns(),
            "receive_started_monotonic_ns": receive_started_monotonic_ns,
            "receive_thread_id": threading.get_ident(),
            "decode_ok": bool(decode_ok),
        }
        self._write(row)
        return self._sequence

    def record_processing(
        self,
        sequence: int,
        stream: str,
        *,
        processing_started_monotonic_ns: int,
    ) -> None:
        if sequence < 0:
            return
        completed = time.monotonic_ns()
        self._write(
            {
                "contract_id": "[DEBUG-P0M2-PYTHON-SUB-9c41]",
                "event": "TELEMETRY_PROCESSED",
                "side": "python_direct_sub",
                "stream": str(stream),
                "sequence": int(sequence),
                "thread_id": threading.get_ident(),
                "processing_completed_monotonic_ns": completed,
                "processing_duration_ns": completed - int(processing_started_monotonic_ns),
            }
        )

    def record_result_processing(
        self,
        event: str,
        *,
        execution_id: int,
        processing_started_monotonic_ns: int,
    ) -> None:
        completed = time.monotonic_ns()
        self._write(
            {
                "contract_id": "[DEBUG-P0M2-PYTHON-SUB-9c41]",
                "event": str(event),
                "side": "python_direct_sub",
                "execution_id": int(execution_id),
                "thread_id": threading.get_ident(),
                "telemetry_consumer_blocked": True,
                "processing_started_monotonic_ns": int(processing_started_monotonic_ns),
                "processing_completed_monotonic_ns": completed,
                "processing_duration_ns": completed - int(processing_started_monotonic_ns),
            }
        )

    def record_poll_cycle(
        self,
        stage: str,
        *,
        execution_id: int,
        poll_started_monotonic_ns: int,
        events: Mapping[Any, Any],
        state_socket: Any,
        depth_socket: Any,
        result_socket: Any,
    ) -> None:
        completed = time.monotonic_ns()
        self._write(
            {
                "contract_id": "[DEBUG-P0M2-PYTHON-SUB-9c41]",
                "event": "POLL_CYCLE",
                "side": "python_direct_sub",
                "stage": str(stage),
                "execution_id": int(execution_id),
                "thread_id": threading.get_ident(),
                "poll_started_monotonic_ns": int(poll_started_monotonic_ns),
                "poll_completed_monotonic_ns": completed,
                "poll_duration_ns": completed - int(poll_started_monotonic_ns),
                "state_pollin": state_socket in events,
                "depth_pollin": depth_socket in events,
                "result_pollin": result_socket in events,
            }
        )

    def close(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True


def _zmq_monitor_event_name(event: int) -> str:
    names = {
        int(getattr(zmq, "EVENT_CONNECTED", -1)): "CONNECTED",
        int(getattr(zmq, "EVENT_CONNECT_DELAYED", -1)): "CONNECT_DELAYED",
        int(getattr(zmq, "EVENT_CONNECT_RETRIED", -1)): "CONNECT_RETRIED",
        int(getattr(zmq, "EVENT_DISCONNECTED", -1)): "DISCONNECTED",
        int(getattr(zmq, "EVENT_HANDSHAKE_SUCCEEDED", -1)): "HANDSHAKE_SUCCEEDED",
        int(getattr(zmq, "EVENT_HANDSHAKE_FAILED_NO_DETAIL", -1)):
            "HANDSHAKE_FAILED_NO_DETAIL",
        int(getattr(zmq, "EVENT_HANDSHAKE_FAILED_PROTOCOL", -1)):
            "HANDSHAKE_FAILED_PROTOCOL",
        int(getattr(zmq, "EVENT_HANDSHAKE_FAILED_AUTH", -1)):
            "HANDSHAKE_FAILED_AUTH",
    }
    return names.get(int(event), "UNKNOWN")


class DealerIdentityLifecycleDiagnostics:
    """Append-only evidence for the P0-M2 Python DEALER route identity."""

    def __init__(
        self,
        path: Path,
        *,
        runtime_instance_id: str,
        dealer_identity: bytes,
        socket_owner_thread_id: int,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_instance_id = str(runtime_instance_id)
        self.dealer_identity = bytes(dealer_identity)
        self.socket_owner_thread_id = int(socket_owner_thread_id)
        self._stream = self.path.open("w", encoding="utf-8")
        self._connected_once = False
        self._closed = False
        self._record(
            "DEALER_SOCKET_CREATED",
            socket_creation_thread_id=self.socket_owner_thread_id,
            **self._socket_thread_fields(),
        )
        self._record(
            "DEALER_IDENTITY",
            dealer_identity_hex=self.dealer_identity.hex(),
            dealer_identity_length=len(self.dealer_identity),
            runtime_instance_id=self.runtime_instance_id,
        )

    def _record(self, event: str, **fields: Any) -> None:
        if self._closed:
            return
        row = {"timestamp_monotonic_ns": time.monotonic_ns(), "event": event}
        row.update(fields)
        self._stream.write(json.dumps(row, sort_keys=True) + "\n")
        self._stream.flush()

    def _socket_thread_fields(self) -> Dict[str, Any]:
        access_thread_id = threading.get_ident()
        return {
            "socket_owner_thread_id": self.socket_owner_thread_id,
            "socket_access_thread_id": access_thread_id,
            "socket_owner_match": access_thread_id == self.socket_owner_thread_id,
        }

    def record_connected(self, endpoint: str) -> None:
        self._connected_once = True
        self._record(
            "DEALER_CONNECTED",
            endpoint=str(endpoint),
            dealer_identity_hex=self.dealer_identity.hex(),
            dealer_identity_length=len(self.dealer_identity),
        )

    def record_ready_sent(self) -> None:
        self._record(
            "DEALER_READY_SENT",
            dealer_identity_hex=self.dealer_identity.hex(),
            dealer_identity_length=len(self.dealer_identity),
            **self._socket_thread_fields(),
        )

    def record_lifecycle_state(self, state: str) -> None:
        self._record("DEALER_LIFECYCLE", state=str(state))

    def record_receive_contract(self) -> None:
        self._record(
            "DEALER_RECEIVE_CONTRACT",
            dealer_identity_hex=self.dealer_identity.hex(),
            dealer_identity_length=len(self.dealer_identity),
            expected_dealer_receive_frame_count=1,
            route_identity_frame_visible=False,
        )

    def record_received_frame(self, frame: bytes, *, has_more: bool) -> None:
        self._record(
            "DEALER_RECEIVE_FRAME",
            dealer_identity_hex=self.dealer_identity.hex(),
            dealer_identity_length=len(self.dealer_identity),
            frame_index=0,
            frame_size=len(frame),
            frame_hex=bytes(frame).hex(),
            has_more=bool(has_more),
            **self._socket_thread_fields(),
        )

    def record_receipt_ack_send(self, *, execution_id: int, payload: bytes) -> None:
        self._record(
            "DEALER_RECEIPT_ACK_SEND",
            execution_id=int(execution_id),
            ack_payload_size=len(payload),
            ack_payload_hex=bytes(payload).hex(),
            **self._socket_thread_fields(),
        )

    def record_commit_send(self, *, execution_id: int, payload: bytes) -> None:
        self._record(
            "DEALER_COMMIT_SEND",
            execution_id=int(execution_id),
            commit_payload_size=len(payload),
            **self._socket_thread_fields(),
        )

    def record_monitor_event(self, event: int, value: int, endpoint: str) -> None:
        event_name = _zmq_monitor_event_name(int(event))
        phase = "CONNECT"
        if event_name == "CONNECTED" and self._connected_once:
            phase = "RECONNECT"
        if event_name == "CONNECTED":
            self._connected_once = True
        self._record(
            "DEALER_MONITOR",
            monitor_event=event_name,
            monitor_event_id=int(event),
            monitor_value=int(value),
            endpoint=str(endpoint),
            connection_phase=phase,
            dealer_identity_hex=self.dealer_identity.hex(),
            dealer_identity_length=len(self.dealer_identity),
        )

    def poll_monitor(self, monitor: Any) -> None:
        while not self._closed:
            try:
                event = recv_monitor_message(monitor, flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            self.record_monitor_event(
                int(event["event"]),
                int(event.get("value", 0)),
                str(event.get("endpoint", "")),
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream.close()


class ResultDealerLifecycle:
    """Own one runner DEALER and recreate it after a result-channel timeout.

    The bridge owns receipt retransmission.  This class never retries a
    primitive or rebuilds a result: it only replaces a broken DEALER pipe with
    the same runtime identity and sends the existing READY control message.
    """

    CONNECTED = "CONNECTED"
    STALE = "STALE"
    RECONNECTING = "RECONNECTING"
    READY_SENT = "READY_SENT"

    def __init__(
        self,
        context: Any,
        *,
        endpoint: str,
        dealer_identity: bytes,
        connect_timeout_s: float,
        bridge_process: Optional[subprocess.Popen] = None,
        diagnostics: Optional[DealerIdentityLifecycleDiagnostics] = None,
        owns_context: bool = False,
    ) -> None:
        self._context = context
        self._owns_context = bool(owns_context)
        self._endpoint = str(endpoint)
        self._dealer_identity = bytes(dealer_identity)
        self._connect_timeout_s = float(connect_timeout_s)
        self._bridge_process = bridge_process
        self._diagnostics = diagnostics
        self.socket: Any = None
        self.monitor: Any = None
        self.state = self.STALE
        self.state_history = [self.STALE]

    def _set_state(self, state: str) -> None:
        self.state = state
        self.state_history.append(state)
        if self._diagnostics is not None:
            self._diagnostics.record_lifecycle_state(state)

    def connect_ready(self, *, wait_for_connect: bool = True) -> None:
        if self.socket is not None:
            raise RuntimeError("result DEALER is already open")
        self._set_state(self.RECONNECTING)
        socket_obj = self._context.socket(zmq.DEALER)
        socket_obj.setsockopt(zmq.LINGER, 0)
        socket_obj.setsockopt(zmq.IDENTITY, self._dealer_identity)
        monitor = socket_obj.get_monitor_socket()
        socket_obj.connect(self._endpoint)
        if wait_for_connect:
            try:
                _wait_for_zmq_monitors_connected(
                    [("bridge Python result endpoint", monitor)],
                    self._connect_timeout_s,
                    process=self._bridge_process,
                    close_monitors=False,
                )
            except BaseException:
                monitor.close(0)
                socket_obj.close(0)
                raise
        self.socket = socket_obj
        self.monitor = monitor
        if self._diagnostics is not None:
            self._diagnostics.record_connected(self._endpoint)
            self._diagnostics.record_receive_contract()
        self._set_state(self.READY_SENT)
        self.socket.send(build_result_ready_wire())
        if self._diagnostics is not None:
            self._diagnostics.record_ready_sent()
        # READY has no reply message. During recovery it must be queued before
        # waiting for a transport monitor event: the bridge's next immutable
        # result is the registration proof, and its receipt ACK closes the
        # pending transaction.
        self._set_state(self.CONNECTED)

    def recover_after_result_timeout(self) -> None:
        if self.socket is None:
            raise RuntimeError("cannot recover a closed result DEALER")
        self._set_state(self.STALE)
        self._close_socket()
        if self._owns_context:
            # The result socket is deliberately isolated from the runner's
            # command/state/depth context.  A fresh context gives the new
            # same-identity DEALER a new I/O pipe after an unreceipted result
            # has made the bridge peer STALE.
            self._context.term()
            self._context = zmq.Context()
        self.connect_ready(wait_for_connect=False)

    def poll_monitor(self) -> None:
        if self._diagnostics is None or self.monitor is None:
            return
        self._diagnostics.poll_monitor(self.monitor)

    def _close_socket(self) -> None:
        if self.monitor is not None:
            self.monitor.close(0)
            self.monitor = None
        if self.socket is not None:
            # A close alone can leave the old ROUTER pipe alive long enough
            # for a same-identity replacement READY to remain only in the
            # new DEALER's local queue.  Make the peer teardown explicit
            # before recreating that identity; the bridge handover remains
            # responsible for accepting the subsequent READY.
            try:
                self.socket.disconnect(self._endpoint)
            except zmq.ZMQError:
                # The socket may already have detached after a transport
                # failure.  Closing it still completes local ownership
                # cleanup, and no primitive/result is retried here.
                pass
            self.socket.close(0)
            self.socket = None

    def close(self) -> None:
        self._close_socket()
        if self._owns_context:
            self._context.term()


def _command_wire(
    execution_id: int,
    command_ids: Iterable[int],
    runtime_instance_id: str = "__runtime_instance_id__",
) -> (bytes, str):
    frames = []
    for frame_index, command_id in enumerate(command_ids):
        action = [0.0, 0.0, 0.0, 0.0]
        frames.append(
            {
                "frame_index": frame_index,
                "command_id": command_id,
                "action": action,
            }
        )
    command, wire = build_command_wire(
        runtime_instance_id=runtime_instance_id,
        execution_id=execution_id,
        frames=frames,
    )
    return wire, command.command_sequence_hash


def _commit_wire(result: Mapping[str, Any]) -> bytes:
    return build_result_commit_wire(result)


class _TransitionSink:
    def __init__(self) -> None:
        self.transitions = []

    def commit(self, transition: Any) -> None:
        self.transitions.append(transition)


class _ReplaySink:
    def __init__(self) -> None:
        self.transitions = []

    def append_once(self, transition: Any) -> None:
        self.transitions.append(transition)


class _RealRuntime:
    """One uninterrupted Unity/bridge/Python result path."""

    def __init__(
        self,
        *,
        out_dir: Path,
        unity_binary: Path,
        bridge_binary: Path,
        runtime_instance_id: str,
        episode_id: str,
        reset_id: str,
        startup_timeout_s: float,
        primitive_timeout_s: float,
        telemetry_buffer_size: int,
        port_profile: Optional[DirectRuntimePortProfile] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.unity_binary = Path(unity_binary)
        self.bridge_binary = Path(bridge_binary)
        self.runtime_instance_id = runtime_instance_id
        self.episode_id = episode_id
        self.reset_id = reset_id
        self.startup_timeout_s = float(startup_timeout_s)
        self.primitive_timeout_s = float(primitive_timeout_s)
        self.telemetry_buffer_size = int(telemetry_buffer_size)
        self.port_profile = port_profile
        if self.port_profile is not None and (
            self.port_profile.runtime_instance_id != self.runtime_instance_id
        ):
            raise ValueError(
                "direct port profile runtime_instance_id does not match runtime"
            )
        self.context = None
        self.result_context = None
        self.command_context = None
        self.command = None
        self.command_client: Optional[ZmqPrimitiveExecutionCommandClient] = None
        self.state = None
        self.depth = None
        self.result_receiver = None
        self.result_receiver_owner_thread_id = None
        self.result_receiver_monitor = None
        self.result_dealer: Optional[ResultDealerLifecycle] = None
        self.snapshot_retriever = None
        self._result_buffer = ExecutionResultBuffer()
        self.handoff_diagnostics = ExecutionHandoffDiagnostics(
            self.out_dir / "handoff_diagnostics.jsonl",
            target_execution_id=DIAGNOSTIC_EXECUTION_ID,
        )
        self.endpoint_diagnostics = EndpointObservationLifecycleDiagnostics(
            self.out_dir / "endpoint_observation_diagnostics.jsonl"
        )
        self.telemetry_transport_diagnostics = TelemetryTransportDiagnostics(
            self.out_dir / "python_telemetry_transport_diagnostics.jsonl"
        )
        self.roscore = None
        self.bridge = None
        self.unity = None
        self.metrics_path = self.out_dir / "bridge_result_metrics.json"
        self.command_audit_path = self.out_dir / "bridge_command_audit.jsonl"
        self.unity_execution_transport_audit_path = (
            self.out_dir / "unity_execution_transport_audit.json"
        )
        self.bridge_diagnostics_path = (
            self.out_dir / "bridge_python_result_diagnostics.jsonl"
        )
        self.dealer_diagnostics = None
        self._logs = []
        self.ports: Dict[str, int] = {}
        self.startup_state = []
        self._startup_monitors = []
        self._cleaned_up = False
        self.stage_ledger = PrimitiveStageLedger(
            self.out_dir / "primitive_stage_ledger.jsonl"
        )
        self.process_supervisor = ProcessLifecycleSupervisor(self.out_dir)
        self.faulthandler_stream = (self.out_dir / "runner_faulthandler.log").open(
            "a", encoding="utf-8"
        )
        faulthandler.enable(file=self.faulthandler_stream, all_threads=True)

    def _child_liveness(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for role, process in (
            ("runner", None),
            ("roscore", self.roscore),
            ("bridge", self.bridge),
            ("unity", self.unity),
        ):
            if process is None:
                values[role] = {"alive": False, "returncode": None}
            else:
                values[role] = {
                    "alive": process.poll() is None,
                    "returncode": process.returncode,
                }
        values["runner"] = {"alive": True, "returncode": None}
        return values

    def _stage(
        self,
        stage: str,
        *,
        execution_id: int,
        primitive_index: int,
        **fields: Any,
    ) -> None:
        self.stage_ledger.record(
            stage,
            execution_id=execution_id,
            primitive_index=primitive_index,
            child_liveness=self._child_liveness(),
            **fields,
        )

    def __enter__(self) -> "_RealRuntime":
        try:
            return self._start()
        except BaseException:
            self._cleanup()
            raise

    def _start(self) -> "_RealRuntime":
        if not self.unity_binary.is_file():
            raise FileNotFoundError("Unity binary not found: {}".format(self.unity_binary))
        if not self.bridge_binary.is_file():
            raise FileNotFoundError("bridge binary not found: {}".format(self.bridge_binary))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ros_logs = self.out_dir / "ros_logs"
        ros_logs.mkdir(parents=True, exist_ok=True)
        if self.port_profile is None:
            self.port_profile = DirectRuntimePortProfile.allocate(
                runtime_instance_id=self.runtime_instance_id
            )
        self.ports = {
            "master": self.port_profile.master_port,
            "command": self.port_profile.command_port,
            "python_command": self.port_profile.python_command_port,
            "unity_command": self.port_profile.unity_command_port,
            "state": self.port_profile.state_port,
            "depth": self.port_profile.depth_port,
            "execution_result": self.port_profile.unity_result_port,
            "python_result": self.port_profile.python_result_port,
            "python_snapshot": self.port_profile.python_snapshot_port,
            "observation_snapshot": self.port_profile.unity_snapshot_port,
        }
        self.startup_state.append("ENDPOINTS_ALLOCATED")
        env = dict(os.environ)
        env.update(
            {
                "ROS_MASTER_URI": "http://127.0.0.1:{}".format(self.ports["master"]),
                "ROS_LOG_DIR": str(ros_logs),
            }
        )
        self._logs = [
            (self.out_dir / "roscore.log").open("w", encoding="utf-8"),
            (self.out_dir / "bridge.log").open("w", encoding="utf-8"),
            (self.out_dir / "unity.log").open("w", encoding="utf-8"),
        ]
        self.roscore = subprocess.Popen(
            ["roscore", "-p", str(self.ports["master"])],
            cwd=str(self.out_dir),
            env=env,
            stdout=self._logs[0],
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.process_supervisor.register_spawn("roscore", self.roscore, self.roscore.args)
        self.startup_state.append("ROSCORE_STARTED")
        _wait_for_tcp(
            self.ports["master"],
            self.startup_timeout_s,
            process=self.roscore,
            label="roscore",
        )
        self.startup_state.append("ROS_MASTER_READY")

        self.context = zmq.Context()
        self.command = self.context.socket(zmq.PUB)
        self.state = self.context.socket(zmq.SUB)
        self.depth = self.context.socket(zmq.SUB)
        self.result_receiver_owner_thread_id = threading.get_ident()
        for socket_obj in (self.command, self.state, self.depth):
            socket_obj.setsockopt(zmq.LINGER, 0)
        self.state.setsockopt(zmq.SUBSCRIBE, b"")
        self.depth.setsockopt(zmq.SUBSCRIBE, b"")
        self.telemetry_transport_diagnostics.record_socket_configuration(
            "state", self.state, subscription=b""
        )
        self.telemetry_transport_diagnostics.record_socket_configuration(
            "depth", self.depth, subscription=b""
        )
        bridge_args = self.port_profile.bridge_launch_args()
        self.bridge = subprocess.Popen(
            [
                str(self.bridge_binary),
                "_unity_host:=127.0.0.1",
                *(
                    "_{}:={}".format(name, value)
                    for name, value in bridge_args.items()
                ),
                "_python_result_bind_host:=127.0.0.1",
                "_python_snapshot_bind_host:=127.0.0.1",
                "_command_retry_interval_s:=0.5",
                "_execution_result_metrics_path:={}".format(self.metrics_path),
                "_command_audit_path:={}".format(self.command_audit_path),
                "_python_result_diagnostics_path:={}".format(
                    self.bridge_diagnostics_path
                ),
                "_telemetry_transport_audit_path:={}".format(
                    self.out_dir / "bridge_telemetry_transport_diagnostics.jsonl"
                ),
                "_spin_hz:=200",
            ],
            cwd=str(self.out_dir),
            env=env,
            stdout=self._logs[1],
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.process_supervisor.register_spawn("bridge", self.bridge, self.bridge.args)
        self.startup_state.append("BRIDGE_STARTED")
        self.command.connect("tcp://127.0.0.1:{}".format(self.ports["command"]))
        self.state.connect("tcp://127.0.0.1:{}".format(self.ports["state"]))
        self.depth.connect("tcp://127.0.0.1:{}".format(self.ports["depth"]))
        # This is the bridge-ready boundary: the Python DEALER can connect
        # only after the bridge ROUTER has successfully bound its endpoint.
        self.dealer_diagnostics = DealerIdentityLifecycleDiagnostics(
            self.out_dir / "python_dealer_diagnostics.jsonl",
            runtime_instance_id=self.runtime_instance_id,
            dealer_identity=self.runtime_instance_id.encode("utf-8"),
            socket_owner_thread_id=self.result_receiver_owner_thread_id,
        )
        self.result_context = zmq.Context()
        self.command_context = zmq.Context()
        self.result_dealer = ResultDealerLifecycle(
            self.result_context,
            endpoint="tcp://127.0.0.1:{}".format(self.ports["python_result"]),
            dealer_identity=self.runtime_instance_id.encode("utf-8"),
            connect_timeout_s=self.startup_timeout_s,
            bridge_process=self.bridge,
            diagnostics=self.dealer_diagnostics,
            owns_context=True,
        )
        self.result_dealer.connect_ready()
        self.result_receiver = self.result_dealer.socket
        self.result_receiver_monitor = self.result_dealer.monitor
        self.command_client = ZmqPrimitiveExecutionCommandClient(
            self.command_context,
            endpoint="tcp://127.0.0.1:{}".format(self.ports["python_command"]),
            dealer_identity=self.runtime_instance_id.encode("utf-8"),
            connect_timeout_s=self.startup_timeout_s,
        )
        self.snapshot_retriever = ZmqBridgeSnapshotRetriever(
            self.command_context,
            "tcp://127.0.0.1:{}".format(self.ports["python_snapshot"]),
            timeout_ms=100,
        )
        self.startup_state.append("PYTHON_SOCKETS_CONNECTED")
        self.startup_state.append("BRIDGE_READY")
        self.startup_state.append("PYTHON_RESULT_READY_SENT")
        unity_monitors = []
        for socket_obj, label in (
            (self.command, "Unity command endpoint"),
            (self.state, "Unity state endpoint"),
            (self.depth, "Unity depth endpoint"),
        ):
            # Create every monitor before launching Unity so a fast bind cannot
            # race monitor creation and lose its CONNECTED event.
            monitor = socket_obj.get_monitor_socket()
            unity_monitors.append((label, monitor))
        self._startup_monitors = [monitor for _, monitor in unity_monitors]
        self.unity = subprocess.Popen(
            [
                str(self.unity_binary),
                "-screen-width", "320",
                "-screen-height", "240",
                "-screen-fullscreen", "0",
                *self.port_profile.unity_launch_argv(),
                "-primitiveResultSchema", str(PROTOCOL_VERSION),
                "-endpointEpisodeId", self.episode_id,
                "-endpointResetId", self.reset_id,
                "-executionTransportAuditPath",
                str(self.unity_execution_transport_audit_path),
                "-logFile", str(self.out_dir / "unity-player.log"),
            ],
            cwd=str(self.unity_binary.parent),
            env=env,
            stdout=self._logs[2],
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.process_supervisor.register_spawn("unity", self.unity, self.unity.args)
        self.startup_state.append("UNITY_STARTED")
        try:
            _wait_for_zmq_monitors_connected(
                unity_monitors,
                self.startup_timeout_s,
                process=self.unity,
            )
        finally:
            self._startup_monitors = []
        # The Unity result DEALER connects to the bridge ROUTER internally;
        # the Python result endpoint above is the runner-visible readiness
        # boundary. The first result receipt remains the execution-level
        # proof for that channel.
        self.startup_state.append("UNITY_FACING_SOCKETS_READY")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._cleanup(
            exit_type="NORMAL_RETURN" if exc_type is None else "PYTHON_EXCEPTION"
        )

    def _cleanup(self, *, exit_type: str = "PYTHON_EXCEPTION") -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if self.snapshot_retriever is not None:
            self.snapshot_retriever.close()
            self.snapshot_retriever = None
        if self.command_client is not None:
            self.command_client.close()
            self.command_client = None
        if self.command_context is not None:
            self.command_context.term()
            self.command_context = None
        if self.result_dealer is not None:
            self.result_dealer.close()
            self.result_dealer = None
        self.result_receiver = None
        self.result_receiver_monitor = None
        for socket_obj in (self.command, self.state, self.depth):
            if socket_obj is not None:
                socket_obj.close(0)
        for monitor in self._startup_monitors:
            monitor.close(0)
        self._startup_monitors = []
        if self.context is not None:
            self.context.term()
        _terminate_process(self.unity)
        self.process_supervisor.record_exit(
            "unity",
            self.unity,
            cleanup_requested=True,
            exit_type="CLEANUP_TERMINATION",
            log_paths=(self.out_dir / "unity.log", self.out_dir / "unity-player.log"),
        )
        _terminate_process(self.bridge)
        self.process_supervisor.record_exit(
            "bridge",
            self.bridge,
            cleanup_requested=True,
            exit_type="CLEANUP_TERMINATION",
            log_paths=(self.out_dir / "bridge.log",),
        )
        _terminate_process(self.roscore)
        self.process_supervisor.record_exit(
            "roscore",
            self.roscore,
            cleanup_requested=True,
            exit_type="CLEANUP_TERMINATION",
            log_paths=(self.out_dir / "roscore.log",),
        )
        for stream in self._logs:
            stream.close()
        self.handoff_diagnostics.close()
        self.endpoint_diagnostics.close()
        self.telemetry_transport_diagnostics.close()
        if self.dealer_diagnostics is not None:
            self.dealer_diagnostics.close()
        self.process_supervisor.register(
            "EXIT",
            role="runner",
            pid=os.getpid(),
            exit_monotonic_ns=time.monotonic_ns(),
            returncode=None,
            signal=None,
            exit_type=exit_type,
            cleanup_requested=True,
        )
        self.process_supervisor.close()
        faulthandler.disable()
        self.faulthandler_stream.close()
        self.stage_ledger.close()
        self.startup_state.append("CLEANED_UP")

    def _poll_result_receiver_monitor(self) -> None:
        if self.result_dealer is None:
            return
        self.result_dealer.poll_monitor()

    def _recover_result_receiver(self) -> None:
        if self.result_dealer is None:
            raise RuntimeError("result DEALER lifecycle is not initialized")
        self.result_dealer.recover_after_result_timeout()
        self.result_receiver = self.result_dealer.socket
        self.result_receiver_monitor = self.result_dealer.monitor
        self.startup_state.append("PYTHON_RESULT_DEALER_RECOVERED")

    def _recv_result_receiver_frame(self) -> bytes:
        raw_message = self.result_receiver.recv()
        has_more = bool(self.result_receiver.getsockopt(zmq.RCVMORE))
        if self.dealer_diagnostics is not None:
            self.dealer_diagnostics.record_received_frame(
                raw_message,
                has_more=has_more,
            )
        return raw_message

    def _send_result_receipt_ack(self, result: Mapping[str, Any]) -> None:
        payload = build_result_receipt_ack_wire(result)
        if self.dealer_diagnostics is not None:
            self.dealer_diagnostics.record_receipt_ack_send(
                execution_id=int(result["execution_id"]),
                payload=payload,
            )
        self.result_receiver.send(payload)

    def _send_result_commit(self, result: Mapping[str, Any]) -> None:
        payload = _commit_wire(result)
        if self.dealer_diagnostics is not None:
            self.dealer_diagnostics.record_commit_send(
                execution_id=int(result["execution_id"]),
                payload=payload,
            )
        self.result_receiver.send(payload)

    def _wait_for_execution_result(
        self,
        *,
        execution_id: int,
        primitive_index: int,
        store: IndexedEndpointObservationStore,
        applied_frames: Any,
        current_state_ids: Any,
        pending_depths: Any,
        result: Optional[Dict[str, Any]],
        deadline: float,
    ) -> Tuple[Optional[Dict[str, Any]], Any, Any]:
        """Wait for one result without resubmitting its physical primitive."""

        endpoint_observation = None
        poller = zmq.Poller()
        poller.register(self.state, zmq.POLLIN)
        poller.register(self.depth, zmq.POLLIN)
        poller.register(self.result_receiver, zmq.POLLIN)
        while time.monotonic() < deadline:
            self._poll_result_receiver_monitor()
            poll_started = time.monotonic_ns()
            events = dict(poller.poll(100))
            self.telemetry_transport_diagnostics.record_poll_cycle(
                "WAIT_RESULT",
                execution_id=execution_id,
                poll_started_monotonic_ns=poll_started,
                events=events,
                state_socket=self.state,
                depth_socket=self.depth,
                result_socket=self.result_receiver,
            )
            if self.state in events:
                self._read_state(
                    store,
                    applied_frames,
                    execution_id,
                    current_state_ids,
                    pending_depths,
                )
            if self.depth in events:
                self._read_depth(
                    store, current_state_ids, pending_depths, execution_id
                )
            if self.result_receiver in events:
                result_processing_started = time.monotonic_ns()
                raw_message = self._recv_result_receiver_frame()
                candidate = _result_mapping(raw_message)
                self.handoff_diagnostics.record_socket_recv(
                    wait_execution_id=execution_id,
                    raw_message=raw_message,
                    decoded_execution_id=candidate.get("execution_id"),
                    decoded_message=candidate,
                )
                if "status" in candidate and "execution_id" in candidate:
                    self._send_result_receipt_ack(candidate)
                    candidate_execution_id = int(candidate["execution_id"])
                    if candidate_execution_id == execution_id:
                        self._stage(
                            "RESULT_RECEIVED",
                            execution_id=execution_id,
                            primitive_index=primitive_index,
                            result_status=candidate.get("status"),
                        )
                        self.endpoint_diagnostics.record_result_received(
                            execution_id, candidate
                        )
                        if result is not None and not self._result_buffer.same_identity(
                            result, candidate
                        ):
                            self.handoff_diagnostics.record(
                                "BUFFER", "conflict", execution_id=execution_id
                            )
                            raise ValueError(
                                "conflicting result for execution_id={}".format(
                                    execution_id
                                )
                            )
                        result = candidate
                        self.handoff_diagnostics.record(
                            "WAIT",
                            "result_selected",
                            execution_id=execution_id,
                            source="socket",
                        )
                    else:
                        try:
                            buffer_outcome = self._result_buffer.offer(candidate)
                        except ValueError:
                            self.handoff_diagnostics.record(
                                "BUFFER",
                                "conflict",
                                execution_id=candidate_execution_id,
                            )
                            raise
                        self.handoff_diagnostics.record(
                            "BUFFER",
                            {"BUFFERED": "insert", "DUPLICATE": "duplicate"}.get(
                                buffer_outcome, "unknown"
                            ),
                            execution_id=candidate_execution_id,
                        )
                self.telemetry_transport_diagnostics.record_result_processing(
                    "RESULT_HANDLER_COMPLETED",
                    execution_id=execution_id,
                    processing_started_monotonic_ns=result_processing_started,
                )
            if result is not None:
                # PUB/SUB state/depth is audit telemetry only.  COMPLETE now
                # resolves its endpoint through the bridge snapshot provider.
                break
        return result, endpoint_observation, poller

    def execute_primitive(self, primitive_index: int) -> Dict[str, Any]:
        execution_id = 44_000_000_000_000 + primitive_index
        self._stage(
            "PRIMITIVE_BEGIN",
            execution_id=execution_id,
            primitive_index=primitive_index,
        )
        self.process_supervisor.record_liveness(
            primitive_index=primitive_index,
            execution_id=execution_id,
            processes=self._child_liveness(),
        )
        command_ids = [
            10_000_000_000_000 + primitive_index * FRAME_COUNT + frame_index
            for frame_index in range(FRAME_COUNT)
        ]
        command_wire, command_hash = _command_wire(
            execution_id, command_ids, self.runtime_instance_id
        )
        self._stage(
            "COMMAND_SERIALIZED",
            execution_id=execution_id,
            primitive_index=primitive_index,
            command_size=len(command_wire),
            command_sequence_hash=command_hash,
        )
        store = IndexedEndpointObservationStore(
            runtime_instance_id=self.runtime_instance_id,
            episode_id=self.episode_id,
            reset_id=self.reset_id,
            diagnostic_sink=lambda event, **fields: self.endpoint_diagnostics.record_store_event(
                execution_id, event, **fields
            ),
        )
        self.endpoint_diagnostics.record_execution_started(execution_id)
        applied_frames = set()
        current_state_ids = set()
        pending_depths = OrderedDict()
        self._poll_result_receiver_monitor()
        result = self._result_buffer.wait_for_result(execution_id)
        self.handoff_diagnostics.record(
            "WAIT",
            "cache_hit" if result is not None else "cache_miss",
            execution_id=execution_id,
        )
        if result is not None:
            self.handoff_diagnostics.record(
                "WAIT",
                "result_selected",
                execution_id=execution_id,
                source="cache",
            )
        endpoint_observation = None
        self._stage(
            "COMMAND_SEND_BEGIN",
            execution_id=execution_id,
            primitive_index=primitive_index,
            command_sequence_hash=command_hash,
            command_transport="zmq_dealer_bridge_router_unity_dealer",
        )
        if self.command_client is None:
            raise RuntimeError("reliable v4 command client is not initialized")
        self.command_client.send(command_wire)
        try:
            command_receipt = self.command_client.wait_for_receipt(
                execution_id=execution_id,
                command_sequence_hash=bytes.fromhex(command_hash),
                timeout_s=self.primitive_timeout_s,
            )
        except TimeoutError as error:
            self._stage(
                "COMMAND_RECEIPT_TIMEOUT",
                execution_id=execution_id,
                primitive_index=primitive_index,
            )
            raise RuntimeError(str(error))
        self._stage(
            "COMMAND_SEND_OK",
            execution_id=execution_id,
            primitive_index=primitive_index,
            command_size=len(command_wire),
            command_sequence_hash=command_hash,
            command_transport="zmq_dealer_bridge_router_unity_dealer",
            command_receipt_status=command_receipt.get("ack_status"),
        )
        self._stage(
            "WAIT_RESULT_BEGIN",
            execution_id=execution_id,
            primitive_index=primitive_index,
        )
        deadline = time.monotonic() + self.primitive_timeout_s
        result, endpoint_observation, poller = self._wait_for_execution_result(
            execution_id=execution_id,
            primitive_index=primitive_index,
            store=store,
            applied_frames=applied_frames,
            current_state_ids=current_state_ids,
            pending_depths=pending_depths,
            result=result,
            deadline=deadline,
        )
        if result is None:
            self.handoff_diagnostics.record(
                "DEALER", "result_timeout_recovery", execution_id=execution_id
            )
            self._recover_result_receiver()
            result, endpoint_observation, poller = self._wait_for_execution_result(
                execution_id=execution_id,
                primitive_index=primitive_index,
                store=store,
                applied_frames=applied_frames,
                current_state_ids=current_state_ids,
                pending_depths=pending_depths,
                result=None,
                deadline=time.monotonic() + self.primitive_timeout_s,
            )
        if result is None:
            self.handoff_diagnostics.record(
                "WAIT",
                "timeout",
                execution_id=execution_id,
            )
            failure_record = self._failure_record(
                primitive_index,
                execution_id,
                applied_frames,
                command_hash,
                "RESULT_TIMEOUT",
            )
            self._stage(
                "PRIMITIVE_RESULT_TIMEOUT",
                execution_id=execution_id,
                primitive_index=primitive_index,
            )
            self._stage(
                "PRIMITIVE_DONE",
                execution_id=execution_id,
                primitive_index=primitive_index,
                terminal_status=failure_record["terminal_status"],
            )
            self.process_supervisor.record_liveness(
                primitive_index=primitive_index,
                execution_id=execution_id,
                processes=self._child_liveness(),
            )
            return failure_record

        transition_sink = _TransitionSink()
        replay_sink = _ReplaySink()
        consumer = PrimitiveExecutionResultConsumer(
            endpoint_provider=BridgeSnapshotEndpointProvider(self.snapshot_retriever),
            transition_committer=transition_sink,
            replay_appender=replay_sink,
        )
        self.handoff_diagnostics.record(
            "CONSUMER",
            "received",
            execution_id=execution_id,
            result_payload_hash=result.get("result_payload_hash"),
        )
        self._stage(
            "CONSUMER_ENTER",
            execution_id=execution_id,
            primitive_index=primitive_index,
        )
        self._stage(
            "SNAPSHOT_FETCH_BEGIN",
            execution_id=execution_id,
            primitive_index=primitive_index,
        )
        outcome = consumer.consume(
            result,
            expected_runtime_instance_id=self.runtime_instance_id,
            expected_execution_id=execution_id,
            expected_command_sequence_hash=command_hash,
        )
        while outcome.status == "ENDPOINT_UNAVAILABLE" and time.monotonic() < deadline:
            outcome = consumer.consume(
                result,
                expected_runtime_instance_id=self.runtime_instance_id,
                expected_execution_id=execution_id,
                expected_command_sequence_hash=command_hash,
            )
        if outcome.status != "ENDPOINT_UNAVAILABLE":
            self._stage(
                "SNAPSHOT_FETCH_OK",
                execution_id=execution_id,
                primitive_index=primitive_index,
                consumer_status=outcome.status,
            )
        self.handoff_diagnostics.record(
            "CONSUMER",
            "outcome",
            execution_id=execution_id,
            status=outcome.status,
            transition_committed=outcome.transition_committed,
            replay_appended=outcome.replay_appended,
            error=outcome.error,
        )
        commit_ack_status = "NOT_SENT"
        if outcome.transition_committed or outcome.status == "NO_TRANSITION":
            if outcome.transition_committed:
                self._stage(
                    "TRANSITION_COMMIT",
                    execution_id=execution_id,
                    primitive_index=primitive_index,
                )
            self.handoff_diagnostics.record(
                "COMMIT",
                "sent",
                execution_id=execution_id,
            )
            self._send_result_commit(result)
            commit_ack_status = self._wait_for_commit_ack(
                poller,
                store,
                applied_frames,
                execution_id,
                consumer,
                result,
                deadline,
                current_state_ids,
                pending_depths,
            )
            self.handoff_diagnostics.record(
                "COMMIT",
                "ack",
                execution_id=execution_id,
                status=commit_ack_status,
            )
        record = self._record_from_result(
            primitive_index=primitive_index,
            result=result,
            command_hash=command_hash,
            applied_frames=applied_frames,
            outcome=outcome,
            commit_ack_status=commit_ack_status,
            transition_sink=transition_sink,
            replay_sink=replay_sink,
        )
        if outcome.status == "COMMITTED" and not transition_sink.transitions:
            raise RuntimeError("consumer committed without a snapshot transition")
        self._stage(
            "PRIMITIVE_DONE",
            execution_id=execution_id,
            primitive_index=primitive_index,
            terminal_status=record["terminal_status"],
        )
        self.process_supervisor.record_liveness(
            primitive_index=primitive_index,
            execution_id=execution_id,
            processes=self._child_liveness(),
        )
        return record

    def _read_state(
        self,
        store: Any,
        applied_frames: Any,
        execution_id: int,
        current_state_ids: Any,
        pending_depths: Any,
    ) -> None:
        # Parse FRAME_APPLIED telemetry at its public boundary.  The result
        # receipt remains the execution-success authority.
        processing_started = time.monotonic_ns()
        sequence = -1
        raw = self.state.recv()
        message = messagepack_unpack(raw)
        if not isinstance(message, list) or len(message) < 3:
            sequence = self.telemetry_transport_diagnostics.record(
                "state",
                state_id=None,
                depth_id=None,
                execution_id=None,
                frame_index=None,
                raw_size=len(raw),
                decode_ok=False,
                receive_started_monotonic_ns=processing_started,
            )
            self.telemetry_transport_diagnostics.record_processing(
                sequence, "state", processing_started_monotonic_ns=processing_started
            )
            return
        state_id = int(message[1])
        sim_time_ns = int(message[2])
        message_execution_id = int(message[10]) if len(message) >= 11 else -1
        frame_index = int(message[11]) if len(message) >= 12 else -1
        sequence = self.telemetry_transport_diagnostics.record(
            "state",
            state_id=state_id,
            depth_id=None,
            execution_id=message_execution_id,
            frame_index=frame_index,
            raw_size=len(raw),
            decode_ok=True,
            receive_started_monotonic_ns=processing_started,
        )
        episode_id = str(message[14]) if len(message) >= 17 else self.episode_id
        reset_id = str(message[15]) if len(message) >= 17 else self.reset_id
        runtime_id = str(message[16]) if len(message) >= 17 else self.runtime_instance_id
        if message_execution_id != execution_id or frame_index < 0:
            self.telemetry_transport_diagnostics.record_processing(
                sequence, "state", processing_started_monotonic_ns=processing_started
            )
            return
        self.endpoint_diagnostics.record_telemetry(
            execution_id,
            "STATE_TELEMETRY_RECEIVED",
            state_id=state_id,
            physics_time_ns=sim_time_ns,
            frame_index=frame_index,
        )
        state = {
            "runtime_instance_id": runtime_id,
            "state_id": state_id,
            "sim_time_ns": sim_time_ns,
            "physics_time_ns": sim_time_ns,
            "episode_id": episode_id,
            "reset_id": reset_id,
            "message": message,
        }
        try:
            store.put_state(state_id, state)
        except ValueError:
            self.telemetry_transport_diagnostics.record_processing(
                sequence, "state", processing_started_monotonic_ns=processing_started
            )
            return
        current_state_ids.add(state_id)
        applied_frames.add(frame_index)
        for capture_id, depth in list(pending_depths.items()):
            if depth["state_id"] != state_id:
                continue
            try:
                store.put_depth(capture_id, depth)
            except ValueError:
                pass
            else:
                self.endpoint_diagnostics.record_telemetry(
                    execution_id,
                    "DEPTH_PENDING_MATCHED",
                    state_id=depth["state_id"],
                    depth_id=capture_id,
                )
            pending_depths.pop(capture_id, None)
        self.telemetry_transport_diagnostics.record_processing(
            sequence, "state", processing_started_monotonic_ns=processing_started
        )

    def _read_depth(
        self,
        store: Any,
        current_state_ids: Any,
        pending_depths: Any,
        execution_id: int,
    ) -> None:
        processing_started = time.monotonic_ns()
        sequence = -1
        parts = self.depth.recv_multipart()
        if len(parts) != 2:
            sequence = self.telemetry_transport_diagnostics.record(
                "depth",
                state_id=None,
                depth_id=None,
                execution_id=None,
                frame_index=None,
                raw_size=sum(len(part) for part in parts),
                payload_size=None,
                decode_ok=False,
                receive_started_monotonic_ns=processing_started,
            )
            self.telemetry_transport_diagnostics.record_processing(
                sequence, "depth", processing_started_monotonic_ns=processing_started
            )
            return
        message = messagepack_unpack(parts[0])
        if not isinstance(message, list) or len(message) < 16:
            sequence = self.telemetry_transport_diagnostics.record(
                "depth",
                state_id=None,
                depth_id=None,
                execution_id=None,
                frame_index=None,
                raw_size=len(parts[0]),
                payload_size=len(parts[1]),
                decode_ok=False,
                receive_started_monotonic_ns=processing_started,
            )
            self.telemetry_transport_diagnostics.record_processing(
                sequence, "depth", processing_started_monotonic_ns=processing_started
            )
            return
        capture_id = "depth-{}".format(message[1])
        depth = {
            "runtime_instance_id": str(message[18]) if len(message) >= 19 else self.runtime_instance_id,
            "depth_id": capture_id,
            "capture_id": capture_id,
            "state_id": int(message[13]),
            "physics_time_ns": int(message[14]),
            "capture_time_ns": int(message[15]),
            "sim_time_ns": int(message[15]),
            "episode_id": str(message[16]) if len(message) >= 19 else self.episode_id,
            "reset_id": str(message[17]) if len(message) >= 19 else self.reset_id,
            "data": parts[1],
        }
        sequence = self.telemetry_transport_diagnostics.record(
            "depth",
            state_id=depth["state_id"],
            depth_id=capture_id,
            execution_id=None,
            frame_index=None,
            raw_size=len(parts[0]),
            payload_size=len(parts[1]),
            decode_ok=True,
            receive_started_monotonic_ns=processing_started,
        )
        self.endpoint_diagnostics.record_telemetry(
            execution_id,
            "DEPTH_TELEMETRY_RECEIVED",
            state_id=depth["state_id"],
            depth_id=capture_id,
            physics_time_ns=depth["physics_time_ns"],
            capture_time_ns=depth["capture_time_ns"],
        )
        if depth["state_id"] in current_state_ids:
            try:
                store.put_depth(capture_id, depth)
            except ValueError:
                pass
            self.telemetry_transport_diagnostics.record_processing(
                sequence, "depth", processing_started_monotonic_ns=processing_started
            )
            return
        pending_depths[capture_id] = depth
        self.endpoint_diagnostics.record_telemetry(
            execution_id,
            "DEPTH_PENDING",
            state_id=depth["state_id"],
            depth_id=capture_id,
        )
        while len(pending_depths) > self.telemetry_buffer_size:
            evicted_capture_id, evicted_depth = pending_depths.popitem(last=False)
            self.endpoint_diagnostics.record_telemetry(
                execution_id,
                "DEPTH_PENDING_EVICTED",
                state_id=evicted_depth["state_id"],
                depth_id=evicted_capture_id,
            )
        self.telemetry_transport_diagnostics.record_processing(
            sequence, "depth", processing_started_monotonic_ns=processing_started
        )

    def _wait_for_commit_ack(
        self,
        poller: Any,
        store: Any,
        applied_frames: Any,
        execution_id: int,
        consumer: PrimitiveExecutionResultConsumer,
        result: Mapping[str, Any],
        deadline: float,
        current_state_ids: Any,
        pending_depths: Any,
    ) -> str:
        while time.monotonic() < deadline:
            self._poll_result_receiver_monitor()
            poll_started = time.monotonic_ns()
            events = dict(poller.poll(100))
            self.telemetry_transport_diagnostics.record_poll_cycle(
                "WAIT_COMMIT_ACK",
                execution_id=execution_id,
                poll_started_monotonic_ns=poll_started,
                events=events,
                state_socket=self.state,
                depth_socket=self.depth,
                result_socket=self.result_receiver,
            )
            if self.state in events:
                self._read_state(
                    store,
                    applied_frames,
                    execution_id,
                    current_state_ids,
                    pending_depths,
                )
            if self.depth in events:
                self._read_depth(
                    store, current_state_ids, pending_depths, execution_id
                )
            if self.result_receiver not in events:
                continue
            result_processing_started = time.monotonic_ns()
            raw_message = self._recv_result_receiver_frame()
            candidate = messagepack_unpack(raw_message)
            decoded_execution_id = (
                candidate.get("execution_id")
                if isinstance(candidate, dict)
                else None
            )
            self.handoff_diagnostics.record_socket_recv(
                wait_execution_id=execution_id,
                raw_message=raw_message,
                decoded_execution_id=decoded_execution_id,
                decoded_message=candidate if isinstance(candidate, dict) else {},
            )
            if isinstance(candidate, dict) and "commit_status" in candidate:
                self.telemetry_transport_diagnostics.record_result_processing(
                    "COMMIT_HANDLER_COMPLETED",
                    execution_id=execution_id,
                    processing_started_monotonic_ns=result_processing_started,
                )
                return str(candidate["commit_status"])
            if isinstance(candidate, dict) and "status" in candidate:
                candidate_result = _result_mapping(
                    messagepack_bytes(candidate)
                )
                self._send_result_receipt_ack(candidate_result)
                candidate_execution_id = int(candidate_result["execution_id"])
                if candidate_execution_id == execution_id:
                    if not self._result_buffer.same_identity(result, candidate_result):
                        self.handoff_diagnostics.record(
                            "BUFFER",
                            "conflict",
                            execution_id=execution_id,
                        )
                        raise ValueError(
                            "conflicting result for execution_id={}".format(
                                execution_id
                            )
                        )
                    continue
                try:
                    buffer_outcome = self._result_buffer.offer(candidate_result)
                except ValueError:
                    self.handoff_diagnostics.record(
                        "BUFFER",
                        "conflict",
                        execution_id=candidate_execution_id,
                    )
                    raise
                self.handoff_diagnostics.record(
                    "BUFFER",
                    {
                        "BUFFERED": "insert",
                        "DUPLICATE": "duplicate",
                    }.get(buffer_outcome, "unknown"),
                    execution_id=candidate_execution_id,
                )
            self.telemetry_transport_diagnostics.record_result_processing(
                "COMMIT_HANDLER_COMPLETED",
                execution_id=execution_id,
                processing_started_monotonic_ns=result_processing_started,
            )
        return "COMMIT_ACK_TIMEOUT"

    def _record_from_result(
        self,
        *,
        primitive_index: int,
        result: Mapping[str, Any],
        command_hash: str,
        applied_frames: Any,
        outcome: Any,
        commit_ack_status: str,
        transition_sink: _TransitionSink,
        replay_sink: _ReplaySink,
    ) -> Dict[str, Any]:
        endpoint_ref = result.get("endpoint_observation_ref") or {}
        status = str(result.get("status", "FAILED"))
        if status not in TERMINAL_STATUSES:
            status = "FAILED"
        applied = int(result.get("applied_frame_count", len(applied_frames)))
        endpoint_state_id = result.get("endpoint_state_id")
        depth_id = endpoint_ref.get("depth_id", endpoint_ref.get("capture_id"))
        return {
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "execution_index": primitive_index,
            "execution_id": int(result["execution_id"]),
            "runtime_instance_id": str(result["runtime_instance_id"]),
            "terminal_status": status,
            "requested_frame_count": int(result.get("requested_frame_count", FRAME_COUNT)),
            "applied_frame_count": applied,
            "last_applied_frame_index": int(result.get("last_applied_frame_index", applied - 1)),
            "endpoint_state_id": endpoint_state_id,
            "depth_id": depth_id,
            "result_payload_hash": str(result.get("result_payload_hash", "")),
            "command_sequence_hash": str(result.get("command_sequence_hash", command_hash)),
            "bridge_accepted": True,
            "bridge_duplicate": False,
            "bridge_conflict": False,
            "ack_status": "DURABLE_RECEIVED",
            "python_receive": True,
            "endpoint_lookup": outcome.status not in {
                "ENDPOINT_UNAVAILABLE",
                "ENDPOINT_MISMATCH",
            },
            "endpoint_mismatch": outcome.status == "ENDPOINT_MISMATCH",
            "transition_commit": bool(transition_sink.transitions),
            "replay_append": bool(replay_sink.transitions),
            "duplicate_transition": outcome.status == "DUPLICATE",
            "protocol_error": outcome.status == "PROTOCOL_ERROR",
            "pending_result": commit_ack_status != "COMMITTED",
            "commit_ack_status": commit_ack_status,
            "telemetry_frame_count": len(applied_frames),
            "telemetry_frame_indices": sorted(applied_frames),
        }

    @staticmethod
    def _failure_record(
        primitive_index: int,
        execution_id: int,
        applied_frames: Any,
        command_hash: str,
        reason: str,
    ) -> Dict[str, Any]:
        applied = len(applied_frames)
        return {
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "execution_index": primitive_index,
            "execution_id": execution_id,
            "runtime_instance_id": "unknown",
            "terminal_status": "FAILED",
            "requested_frame_count": FRAME_COUNT,
            "applied_frame_count": applied,
            "last_applied_frame_index": applied - 1 if applied else -1,
            "endpoint_state_id": None,
            "depth_id": None,
            "result_payload_hash": "",
            "command_sequence_hash": command_hash,
            "bridge_accepted": False,
            "bridge_duplicate": False,
            "bridge_conflict": False,
            "ack_status": reason,
            "python_receive": False,
            "endpoint_lookup": False,
            "transition_commit": False,
            "replay_append": False,
            "duplicate_transition": False,
            "protocol_error": True,
            "pending_result": True,
        }


def _runtime_manifest(
    *,
    unity_binary: Path,
    bridge_binary: Path,
    runtime_instance_id: str,
    episode_id: str,
    reset_id: str,
    ports: Mapping[str, int],
    startup_state: Iterable[str],
    checkpoint: Optional[Path],
    out_dir: Optional[Path] = None,
    port_profile: Optional[DirectRuntimePortProfile] = None,
) -> Dict[str, Any]:
    planning_root = Path(__file__).resolve().parents[3]
    unity_root = Path("/home/xm/XM/XMflight")
    assembly = unity_binary.parent / "{}_Data/Managed/Assembly-CSharp.dll".format(
        unity_binary.stem
    )
    try:
        from planning.contracts.collection import source_tree_sha256

        planning_source_sha = source_tree_sha256(planning_root)
    except Exception:
        planning_source_sha = "NOT_AVAILABLE"
    manifest = {
        "contract_id": RUNNER_CONTRACT_ID,
        "mode": "real",
        "schema_version": PROTOCOL_VERSION,
        "runtime_instance_id": runtime_instance_id,
        "episode_id": episode_id,
        "reset_id": reset_id,
        "planning_source_tree_sha256": planning_source_sha,
        "planning_git_sha": "NOT_AVAILABLE",
        "unity_source_git_sha": _git_head(unity_root),
        "unity_player": {
            "path": str(unity_binary),
            "sha256": file_sha256(unity_binary),
        },
        "assembly_csharp": {
            "path": str(assembly),
            "sha256": file_sha256(assembly) if assembly.is_file() else "NOT_AVAILABLE",
        },
        "bridge": {
            "path": str(bridge_binary),
            "sha256": file_sha256(bridge_binary),
            **bridge_identity_manifest(planning_root),
        },
        "protocol": {
            "owner": "include/planning/protocol/xm_protocol.hpp",
            "schema_version": PROTOCOL_VERSION,
        },
        "schema": {
            "path": str(planning_root / ("primitive_execution_schema" + "_" + "v4.md")),
            "sha256": file_sha256(
                planning_root / ("primitive_execution_schema" + "_" + "v4.md")
            ),
        },
        "checkpoint": (
            {"path": str(checkpoint), "sha256": file_sha256(checkpoint)}
            if checkpoint is not None and checkpoint.is_file()
            else {"path": "NOT_USED", "sha256": "NOT_USED"}
        ),
        "ports": dict(ports),
        "port_profile": (
            {
                "profile": port_profile.profile,
                "runtime_instance_id": port_profile.runtime_instance_id,
                "ports": port_profile.port_mapping(),
            }
            if port_profile is not None
            else {"profile": "unknown"}
        ),
        "startup_state": list(startup_state),
        "fixed_frame_count": FRAME_COUNT,
        "telemetry_transport": "PUB/SUB unchanged",
        "result_transport": "reliable result channel",
    }
    if out_dir is not None:
        root = Path(out_dir)
        manifest["artifacts"] = {
            "runtime_manifest": str(root / "runtime_manifest.json"),
            "ledger": str(root / "ledger.jsonl"),
            "summary": str(root / "summary.json"),
            "bridge_result_metrics": str(root / "bridge_result_metrics.json"),
            "bridge_command_audit": str(root / "bridge_command_audit.jsonl"),
            "unity_execution_transport_audit": str(
                root / "unity_execution_transport_audit.json"
            ),
        }
    return manifest


def run_real(
    *,
    out_dir: Path,
    unity_binary: Path,
    bridge_binary: Path,
    primitive_count: int = 10000,
    checkpoint_interval: int = 1000,
    runtime_instance_id: str = "worker-00-p0-m2",
    episode_id: str = "p0-m2-episode-00",
    reset_id: str = "p0-m2-reset-00",
    startup_timeout_s: float = 30.0,
    primitive_timeout_s: float = 15.0,
    telemetry_buffer_size: int = 256,
    checkpoint: Optional[Path] = None,
    port_profile: Optional[DirectRuntimePortProfile] = None,
) -> Dict[str, Any]:
    """Run one uninterrupted real Unity worker and emit P0-M2 evidence."""

    if primitive_count < 1:
        raise ValueError("primitive_count must be positive")
    if telemetry_buffer_size < 1:
        raise ValueError("telemetry_buffer_size must be positive")
    ledger = ValidationLedger(
        out_dir,
        checkpoint_interval=checkpoint_interval,
        mode="real",
        runtime_instance_id=runtime_instance_id,
    )
    failure_path = Path(out_dir) / "failure_episodes.jsonl"
    failures = failure_path.open("w", encoding="utf-8")
    bridge_metrics: Dict[str, Any] = {}
    summary: Optional[Dict[str, int]] = None
    extra: Dict[str, Any] = {}
    phase = "RUNTIME_STARTUP"
    try:
        with _RealRuntime(
            out_dir=Path(out_dir),
            unity_binary=Path(unity_binary),
            bridge_binary=Path(bridge_binary),
            runtime_instance_id=runtime_instance_id,
            episode_id=episode_id,
            reset_id=reset_id,
            startup_timeout_s=startup_timeout_s,
            primitive_timeout_s=primitive_timeout_s,
            telemetry_buffer_size=telemetry_buffer_size,
            port_profile=port_profile
            or DirectRuntimePortProfile.allocate(
                runtime_instance_id=runtime_instance_id
            ),
        ) as runtime:
            ledger.write_manifest(
                _runtime_manifest(
                    unity_binary=Path(unity_binary),
                    bridge_binary=Path(bridge_binary),
                    runtime_instance_id=runtime_instance_id,
                    episode_id=episode_id,
                    reset_id=reset_id,
                    ports=runtime.ports,
                    startup_state=runtime.startup_state,
                    checkpoint=checkpoint,
                    out_dir=Path(out_dir),
                    port_profile=runtime.port_profile,
                )
            )
            phase = "RUNTIME_EXECUTION"
            for primitive_index in range(1, primitive_count + 1):
                try:
                    record = runtime.execute_primitive(primitive_index)
                    record["runtime_instance_id"] = runtime_instance_id
                    ledger.append(record)
                except Exception as error:
                    failure = {
                        "execution_index": primitive_index,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    failures.write(json.dumps(failure, sort_keys=True) + "\n")
                    failures.flush()
                    raise
        phase = "FINAL_ACCOUNTING"
        if runtime.metrics_path.is_file():
            bridge_metrics = json.loads(runtime.metrics_path.read_text(encoding="utf-8"))
        with failure_path.open("r", encoding="utf-8") as failure_stream:
            failure_count = sum(1 for _ in failure_stream)
        extra = {
            "bridge_metrics": bridge_metrics,
            "failure_episode_total": failure_count,
            "pending_result_final": int(bridge_metrics.get("pending", 0)),
            "snapshot_request_total": int(bridge_metrics.get("snapshot_request", 0)),
            "snapshot_response_total": int(bridge_metrics.get("snapshot_response", 0)),
            "snapshot_hash_match_total": int(bridge_metrics.get("snapshot_hash_match", 0)),
            "snapshot_cache_hit_total": int(bridge_metrics.get("snapshot_cache_hit", 0)),
            "snapshot_missing_total": int(bridge_metrics.get("snapshot_missing", 0)),
            "pending_snapshot_final": int(bridge_metrics.get("pending_snapshot", 0)),
            # The correctness path uses BridgeSnapshotEndpointProvider.  State/depth
            # PUB/SUB remains audit-only and has no transition-store lookup.
            "telemetry_lookup_count": 0,
            "duplicate_result_total": int(bridge_metrics.get("duplicate", 0)),
            "protocol_error_total": max(
                int(bridge_metrics.get("protocol_error", 0)),
                int(ledger._counters.values["protocol_error_total"]),
            ),
            "validation_status": "COMPLETED",
        }
        for metric_name in (
            "command_accepted_total",
            "command_forward_total",
            "unity_command_receipt_total",
            "command_duplicate_total",
            "command_receipt_duplicate_total",
            "command_retry_total",
            "command_conflict_total",
            "command_protocol_error_total",
            "pending_command_final",
        ):
            extra[metric_name] = int(bridge_metrics.get(metric_name, 0))
        unity_audit = {}
        unity_audit_path = getattr(
            runtime, "unity_execution_transport_audit_path", None
        )
        if unity_audit_path is not None and Path(unity_audit_path).is_file():
            unity_audit = json.loads(
                Path(unity_audit_path).read_text(encoding="utf-8")
            )
        extra["physical_execution_total"] = int(
            unity_audit.get(
                "physical_execution_total",
                bridge_metrics.get("physical_execution_total", 0),
            )
        )
        extra["unity_physical_execution_total"] = extra[
            "physical_execution_total"
        ]
    except BaseException as error:
        try:
            with failure_path.open("r", encoding="utf-8") as failure_stream:
                failure_count = sum(1 for _ in failure_stream)
        except OSError:
            failure_count = 0
        extra = {
            **extra,
            "failure_episode_total": failure_count,
            "validation_status": (
                "PARTIAL"
                if ledger._counters.values["execution_total"] > 0
                else "FAILED"
            ),
            "failure_stage": phase,
            "failure_error_type": type(error).__name__,
            "failure_error": str(error),
        }
        raise
    finally:
        failures.close()
        summary = ledger.close(extra_summary=extra)
    return {**summary, **extra}


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--primitives", type=int, default=10000)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument(
        "--runtime-instance-id",
        default="worker-00-p0-m2",
    )
    parser.add_argument("--episode-id", default="p0-m2-episode-00")
    parser.add_argument("--reset-id", default="p0-m2-reset-00")
    parser.add_argument(
        "--unity-binary",
        type=Path,
        default=Path(os.environ.get("PLANNING_UNITY_BINARY", "/home/xm/XM/xm_ws/src/unity/XMflight.x86_64")),
    )
    parser.add_argument(
        "--bridge-binary",
        type=Path,
        default=Path(
            os.environ.get(
                "PLANNING_BRIDGE_BINARY", str(planning_bridge_binary())
            )
        ),
    )
    parser.add_argument("--startup-timeout-s", type=float, default=30.0)
    parser.add_argument("--primitive-timeout-s", type=float, default=15.0)
    parser.add_argument("--telemetry-buffer-size", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        summary = run_dry_run(
            out_dir=args.out_dir,
            primitive_count=args.primitives,
            checkpoint_interval=args.checkpoint_interval,
            runtime_instance_id=args.runtime_instance_id,
        )
    else:
        summary = run_real(
            out_dir=args.out_dir,
            unity_binary=args.unity_binary,
            bridge_binary=args.bridge_binary,
            primitive_count=args.primitives,
            checkpoint_interval=args.checkpoint_interval,
            runtime_instance_id=args.runtime_instance_id,
            episode_id=args.episode_id,
            reset_id=args.reset_id,
            startup_timeout_s=args.startup_timeout_s,
            primitive_timeout_s=args.primitive_timeout_s,
            telemetry_buffer_size=args.telemetry_buffer_size,
            checkpoint=args.checkpoint,
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
