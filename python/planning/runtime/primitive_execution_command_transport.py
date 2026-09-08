"""Reliable v4 primitive-command DEALER seam for the validation runner."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time
import threading
from typing import Any, Dict, Optional, Tuple

import zmq

from planning.protocol.constants import PROTOCOL_VERSION
from planning.protocol.msgpack import messagepack_unpack
from planning.protocol.command import build_command_ready_wire
from planning.protocol.reset import build_reset_received_ack_wire

class PrimitiveExecutionCommandReceiptError(RuntimeError):
    """Raised when Unity/bridge rejects or corrupts a command receipt."""


class ZmqPrimitiveExecutionCommandClient:
    """Own one DEALER and wait only for exact command receipt identities."""

    def __init__(
        self,
        context: zmq.Context,
        *,
        endpoint: str,
        dealer_identity: bytes,
        connect_timeout_s: float,
        diagnostics_path: Optional[str] = None,
    ) -> None:
        self._context = context
        self._endpoint = str(endpoint)
        self._dealer_identity = bytes(dealer_identity)
        self._runtime_instance_id = bytes(dealer_identity).decode("utf-8")
        self._connect_timeout_s = float(connect_timeout_s)
        self._receipts: Dict[int, Dict[str, Any]] = OrderedDict()
        self._reset_completions: Dict[Tuple[str, str, str], Dict[str, Any]] = OrderedDict()
        self._known_hashes: Dict[int, bytes] = {}
        self._closed = False
        self._socket = None
        self._monitor_socket = None
        self._socket_epoch = 0
        self._worker_id = self._read_worker_id()
        configured_path = diagnostics_path or os.environ.get(
            "P3_COMMAND_RECEIPT_DIAGNOSTICS_PATH"
        )
        if configured_path and "{worker_id}" in configured_path:
            configured_path = configured_path.format(
                worker_id=(
                    self._worker_id if self._worker_id is not None else "unknown"
                )
            )
        self._diagnostics_stream = self._open_diagnostics_stream(configured_path)
        self._open_socket()

    @staticmethod
    def _read_worker_id() -> Optional[int]:
        raw = os.environ.get("PLANNING_WORKER_ID")
        try:
            return None if raw is None else int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _open_diagnostics_stream(path: Optional[str]):
        if not path:
            return None
        try:
            target = Path(path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            return target.open("a", encoding="utf-8", buffering=1)
        except OSError:
            return None

    def _emit(self, event: str, **fields: Any) -> None:
        if self._diagnostics_stream is None:
            return
        record: Dict[str, Any] = {
            "event": str(event),
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "worker_id": self._worker_id,
            "runtime_instance_id": self._runtime_instance_id,
            "dealer_identity_hex": self._dealer_identity.hex(),
            "dealer_identity_length": len(self._dealer_identity),
            "socket_epoch": self._socket_epoch,
        }
        record.update(fields)
        try:
            self._diagnostics_stream.write(json.dumps(record, sort_keys=True) + "\n")
            self._diagnostics_stream.flush()
        except (OSError, TypeError, ValueError):
            # Diagnostics must never change transport behavior.
            pass

    @staticmethod
    def _pid_snapshot(pid: Any) -> Dict[str, Any]:
        """Return best-effort liveness evidence without affecting transport."""

        try:
            normalized = int(pid)
        except (TypeError, ValueError):
            return {"pid": pid, "alive": False, "invalid_pid": True}
        if normalized <= 0:
            return {"pid": normalized, "alive": False, "invalid_pid": True}
        try:
            os.kill(normalized, 0)
        except OSError:
            return {"pid": normalized, "alive": False}
        return {"pid": normalized, "alive": True}

    def _timeout_liveness(self, execution_id: int) -> None:
        """Emit process evidence at the receipt-timeout boundary only.

        The external P3 launcher owns the actual processes and writes their
        root PIDs to this optional manifest.  This probe is deliberately
        best-effort: a malformed/unavailable artifact may not change receipt
        polling, error propagation, reconnects, or retries.
        """

        worker = self._pid_snapshot(os.getpid())
        learner = self._pid_snapshot(os.getppid())
        processes: Dict[str, Any] = {}
        manifest_error = ""
        path = os.environ.get("P3_COMMAND_RECEIPT_PROCESS_MANIFEST_PATH", "")
        if path and self._worker_id is not None:
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                workers = payload.get("workers", {})
                entry = workers.get(str(self._worker_id), {})
                if not isinstance(entry, dict):
                    raise ValueError("worker entry is not a map")
                for label in ("bridge", "unity", "roscore"):
                    processes["{}_process".format(label)] = self._pid_snapshot(
                        entry.get("{}_pid".format(label))
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                manifest_error = "{}: {}".format(type(error).__name__, error)
        self._emit(
            "COMMAND_RECEIPT_TIMEOUT_LIVENESS",
            execution_id=int(execution_id),
            worker_process=worker,
            learner_process=learner,
            manifest_path=path,
            manifest_error=manifest_error,
            **processes,
        )

    def _drain_monitor(self) -> None:
        if self._monitor_socket is None:
            return
        try:
            from zmq.utils.monitor import recv_monitor_message

            while self._monitor_socket.poll(0):
                event = recv_monitor_message(
                    self._monitor_socket, flags=zmq.DONTWAIT
                )
                self._emit(
                    "SOCKET_MONITOR_EVENT",
                    monitor_event=event.get("event"),
                    monitor_value=event.get("value"),
                    monitor_endpoint=event.get("endpoint"),
                )
        except (zmq.ZMQError, TypeError, ValueError) as error:
            self._emit(
                "SOCKET_MONITOR_ERROR",
                monitor_error_type=type(error).__name__,
                monitor_error=str(error),
            )

    def _open_socket(self) -> None:
        self._socket_epoch += 1
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IDENTITY, self._dealer_identity)
        try:
            self._monitor_socket = self._socket.get_monitor_socket(zmq.EVENT_ALL)
        except zmq.ZMQError:
            self._monitor_socket = None
        self._socket.connect(self._endpoint)
        ready_payload = build_command_ready_wire(self._runtime_instance_id)
        self._emit(
            "SOCKET_CREATED", endpoint=self._endpoint, socket_type="DEALER"
        )
        self._emit(
            "READY_SEND_BEGIN",
            message_type="PrimitiveExecutionCommandReady",
            payload_size=len(ready_payload),
        )
        ready_rc = self._socket.send(ready_payload)
        self._emit(
            "READY_SEND_OK",
            message_type="PrimitiveExecutionCommandReady",
            payload_size=len(ready_payload),
            send_return=ready_rc,
        )
        self._drain_monitor()

    def reconnect(self) -> None:
        """Replace the DEALER pipe while preserving its identity.

        The caller owns the immutable command payload and may send those same
        bytes again after this READY handshake.  This method never creates an
        execution id or rebuilds a primitive.
        """
        if self._closed:
            raise RuntimeError("command client is closed")
        self._emit("RECONNECT_BEGIN")
        if self._socket is not None:
            self._socket.close(0)
        if self._monitor_socket is not None:
            self._monitor_socket.close(0)
            self._monitor_socket = None
        self._open_socket()
        self._emit("RECONNECT_READY")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._drain_monitor()
        self._emit("SOCKET_CLOSE_BEGIN")
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
        if self._monitor_socket is not None:
            self._monitor_socket.close(0)
            self._monitor_socket = None
        if self._diagnostics_stream is not None:
            try:
                self._diagnostics_stream.close()
            except OSError:
                pass

    def send(self, payload: bytes) -> None:
        if self._closed or self._socket is None:
            raise RuntimeError("command client is closed")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("command payload must be non-empty bytes")
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        self._emit(
            "COMMAND_SEND_BEGIN",
            message_type="PrimitiveExecutionCommand",
            payload_size=len(payload),
            payload_sha256=payload_sha256,
        )
        send_return = self._socket.send(payload)
        self._emit(
            "COMMAND_SEND_OK",
            message_type="PrimitiveExecutionCommand",
            payload_size=len(payload),
            payload_sha256=payload_sha256,
            send_return=send_return,
        )
        self._drain_monitor()

    def send_reset(self, payload: bytes) -> None:
        """Send an immutable v4 reset payload on the existing command peer."""

        self.send(payload)

    def _recv_packet(self) -> Dict[str, Any]:
        """Receive one packet and emit raw-before-demux evidence."""

        self._drain_monitor()
        raw = self._socket.recv()
        try:
            value = messagepack_unpack(raw)
        except Exception:
            self._emit(
                "RAW_PACKET_RECEIVED",
                payload_size=len(raw),
                raw_message_hex=raw.hex(),
                decoded_discriminator=None,
                decode_error=True,
            )
            raise
        fields: Dict[str, Any] = {
            "payload_size": len(raw),
            "raw_message_hex": raw.hex(),
            "decoded_discriminator": (
                value.get("message_type") if isinstance(value, dict) else None
            ),
        }
        if isinstance(value, dict):
            packet_hash = value.get("command_sequence_hash")
            fields.update(
                {
                    "execution_id": value.get("execution_id"),
                    "episode_id": value.get("episode_id"),
                    "reset_id": value.get("reset_id"),
                    "packet_runtime_instance_id": value.get("runtime_instance_id"),
                    "command_sequence_hash": (
                        packet_hash.hex()
                        if isinstance(packet_hash, bytes)
                        else packet_hash
                    ),
                }
            )
        self._emit("RAW_PACKET_RECEIVED", **fields)
        return value

    def wait_for_reset(
        self,
        *,
        episode_id: str,
        reset_id: str,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Wait for an exact reset completion and receipt-ack it once received."""

        reset_key = (self._runtime_instance_id, str(episode_id), str(reset_id))
        cached = self._reset_completions.pop(reset_key, None)
        if cached is not None:
            self._emit(
                "RESET_COMPLETION_BUFFER_HIT",
                message_type="PrimitiveResetComplete",
                episode_id=episode_id,
                reset_id=reset_id,
            )
            return self._accept_reset_completion(cached, episode_id, reset_id)
        deadline = time.monotonic() + (
            self._connect_timeout_s if timeout_s is None else float(timeout_s)
        )
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000.0))
            events = dict(poller.poll(remaining_ms))
            self._drain_monitor()
            if self._socket not in events:
                continue
            try:
                value = self._recv_packet()
            except Exception as error:
                raise PrimitiveExecutionCommandReceiptError(
                    "reset completion is not MessagePack: {}".format(error)
                ) from error
            if not isinstance(value, dict):
                raise PrimitiveExecutionCommandReceiptError(
                    "reset completion is not a map"
                )
            if value.get("message_type") == "PrimitiveExecutionCommandReceiptAck":
                self._buffer_receipt(value)
                continue
            if value.get("message_type") != "PrimitiveResetComplete":
                raise PrimitiveExecutionCommandReceiptError(
                    "unexpected reliable command message while waiting for reset"
                )
            self._buffer_reset_completion(value)
            if (
                value.get("runtime_instance_id") == self._runtime_instance_id
                and value.get("episode_id") == episode_id
                and value.get("reset_id") == reset_id
            ):
                self._reset_completions.pop(reset_key, None)
                self._emit(
                    "RESET_COMPLETION_WAITER_MATCH",
                    message_type="PrimitiveResetComplete",
                    episode_id=episode_id,
                    reset_id=reset_id,
                )
                return self._accept_reset_completion(value, episode_id, reset_id)
        self._emit(
            "RESET_COMPLETION_TIMEOUT",
            message_type="PrimitiveResetComplete",
            episode_id=episode_id,
            reset_id=reset_id,
        )
        raise TimeoutError(
            "reset completion timeout for episode_id={} reset_id={}".format(
                episode_id, reset_id
            )
        )

    def _buffer_receipt(self, value: Dict[str, Any]) -> None:
        candidate_id = value.get("execution_id")
        if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
            self._emit(
                "COMMAND_RECEIPT_INVALID",
                message_type="PrimitiveExecutionCommandReceiptAck",
                invalid_field="execution_id",
                received_execution_id=candidate_id,
            )
            raise PrimitiveExecutionCommandReceiptError(
                "command receipt execution_id is invalid"
            )
        candidate_hash = value.get("command_sequence_hash")
        if not isinstance(candidate_hash, bytes) or len(candidate_hash) != 32:
            self._emit(
                "COMMAND_RECEIPT_INVALID",
                message_type="PrimitiveExecutionCommandReceiptAck",
                invalid_field="command_sequence_hash",
                execution_id=candidate_id,
            )
            raise PrimitiveExecutionCommandReceiptError(
                "command receipt hash is invalid"
            )
        previous_hash = self._known_hashes.get(candidate_id)
        if previous_hash is not None and previous_hash != candidate_hash:
            self._emit(
                "COMMAND_RECEIPT_INVALID",
                message_type="PrimitiveExecutionCommandReceiptAck",
                invalid_field="conflicting_identity",
                execution_id=candidate_id,
                command_sequence_hash=candidate_hash.hex(),
            )
            raise PrimitiveExecutionCommandReceiptError(
                "conflicting command receipt identity"
            )
        self._known_hashes[candidate_id] = candidate_hash
        previous = self._receipts.get(candidate_id)
        self._emit(
            "COMMAND_RECEIPT_BUFFER_DUPLICATE"
            if previous is not None
            else "COMMAND_RECEIPT_BUFFER_INSERT",
            message_type="PrimitiveExecutionCommandReceiptAck",
            execution_id=candidate_id,
            packet_runtime_instance_id=value.get("runtime_instance_id"),
            command_sequence_hash=candidate_hash.hex(),
            ack_status=value.get("ack_status"),
        )
        self._receipts[candidate_id] = value

    def _buffer_reset_completion(self, value: Dict[str, Any]) -> None:
        runtime_instance_id = value.get("runtime_instance_id")
        episode_id = value.get("episode_id")
        reset_id = value.get("reset_id")
        if not all(isinstance(item, str) and item for item in (
            runtime_instance_id, episode_id, reset_id
        )):
            raise PrimitiveExecutionCommandReceiptError(
                "reset completion identity is invalid"
            )
        key = (runtime_instance_id, episode_id, reset_id)
        previous = self._reset_completions.get(key)
        if previous is not None and previous != value:
            raise PrimitiveExecutionCommandReceiptError(
                "conflicting reset completion identity"
            )
        self._emit(
            "RESET_COMPLETION_BUFFER_DUPLICATE"
            if previous is not None
            else "RESET_COMPLETION_BUFFER_INSERT",
            message_type="PrimitiveResetComplete",
            observation_runtime_instance_id=runtime_instance_id,
            episode_id=episode_id,
            reset_id=reset_id,
        )
        self._reset_completions[key] = value

    def _accept_reset_completion(
        self,
        value: Dict[str, Any],
        episode_id: str,
        reset_id: str,
    ) -> Dict[str, Any]:
        if (
            value.get("schema_version") != PROTOCOL_VERSION
            or value.get("message_type") != "PrimitiveResetComplete"
            or value.get("runtime_instance_id") != self._runtime_instance_id
            or value.get("episode_id") != episode_id
            or value.get("reset_id") != reset_id
        ):
            raise PrimitiveExecutionCommandReceiptError(
                "reset completion identity mismatch"
            )
        self._socket.send(
            build_reset_received_ack_wire(
                runtime_instance_id=self._runtime_instance_id,
                episode_id=episode_id,
                reset_id=reset_id,
            )
        )
        return value

    def wait_for_receipt(
        self,
        *,
        execution_id: int,
        command_sequence_hash: bytes,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        cached = self._receipts.pop(int(execution_id), None)
        if cached is not None:
            self._emit(
                "COMMAND_RECEIPT_BUFFER_HIT",
                message_type="PrimitiveExecutionCommandReceiptAck",
                execution_id=int(execution_id),
            )
            self._validate(cached, execution_id, command_sequence_hash)
            self._emit(
                "COMMAND_RECEIPT_WAITER_MATCH",
                message_type="PrimitiveExecutionCommandReceiptAck",
                execution_id=int(execution_id),
            )
            return cached
        deadline = time.monotonic() + (
            self._connect_timeout_s if timeout_s is None else float(timeout_s)
        )
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000.0))
            events = dict(poller.poll(remaining_ms))
            self._drain_monitor()
            if self._socket not in events:
                continue
            try:
                value = self._recv_packet()
            except Exception as error:
                raise PrimitiveExecutionCommandReceiptError(
                    "command receipt is not MessagePack: {}".format(error)
                ) from error
            if not isinstance(value, dict):
                raise PrimitiveExecutionCommandReceiptError("command receipt is not a map")
            if value.get("message_type") == "PrimitiveResetComplete":
                self._buffer_reset_completion(value)
                continue
            if value.get("message_type") != "PrimitiveExecutionCommandReceiptAck":
                raise PrimitiveExecutionCommandReceiptError(
                    "unexpected reliable command message while waiting for receipt"
                )
            candidate_id = value.get("execution_id")
            self._buffer_receipt(value)
            if candidate_id != int(execution_id):
                self._emit(
                    "COMMAND_RECEIPT_WAITER_SKIP",
                    message_type="PrimitiveExecutionCommandReceiptAck",
                    expected_execution_id=int(execution_id),
                    received_execution_id=candidate_id,
                )
                continue
            self._validate(value, execution_id, command_sequence_hash)
            self._emit(
                "COMMAND_RECEIPT_WAITER_MATCH",
                message_type="PrimitiveExecutionCommandReceiptAck",
                execution_id=int(execution_id),
            )
            return value
        self._emit(
            "COMMAND_RECEIPT_TIMEOUT",
            message_type="PrimitiveExecutionCommandReceiptAck",
            execution_id=int(execution_id),
            expected_command_sequence_hash=bytes(command_sequence_hash).hex(),
        )
        self._timeout_liveness(int(execution_id))
        raise TimeoutError(
            "command receipt timeout for execution_id={}".format(execution_id)
        )

    def _validate(
        self, value: Dict[str, Any], execution_id: int, command_sequence_hash: bytes
    ) -> None:
        if value.get("schema_version") != PROTOCOL_VERSION:
            raise PrimitiveExecutionCommandReceiptError("command receipt schema mismatch")
        if value.get("message_type") != "PrimitiveExecutionCommandReceiptAck":
            raise PrimitiveExecutionCommandReceiptError("command receipt type mismatch")
        if value.get("runtime_instance_id") != self._runtime_instance_id:
            raise PrimitiveExecutionCommandReceiptError(
                "command receipt runtime identity mismatch"
            )
        if value.get("execution_id") != int(execution_id):
            raise PrimitiveExecutionCommandReceiptError("command receipt execution mismatch")
        if value.get("command_sequence_hash") != bytes(command_sequence_hash):
            raise PrimitiveExecutionCommandReceiptError("command receipt hash mismatch")
        if value.get("ack_status") != "RECEIVED":
            raise PrimitiveExecutionCommandReceiptError(
                "command rejected: {}".format(value.get("reason_code"))
            )


__all__ = [
    "PrimitiveExecutionCommandReceiptError",
    "ZmqPrimitiveExecutionCommandClient",
]
