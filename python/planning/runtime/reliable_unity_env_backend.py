"""Public reliable-v4 execution seam used by :mod:`planning.runtime.unity_env`.

The environment owns reward and termination semantics.  A backend owns only
the transport transaction and returns the exact endpoint observation selected
by the authoritative v4 snapshot.  This small seam also gives tests a
transport-free way to exercise ``UnityForestEnv.step_primitive`` without
reaching into private transport methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import time
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import zmq

from planning.protocol.constants import PRIMITIVE_FRAME_COUNT, PROTOCOL_VERSION
from planning.protocol.msgpack import messagepack_unpack
from planning.protocol.reset import build_reset_request_wire
from planning.protocol.result import build_result_commit_wire, build_result_ready_wire
from planning.protocol.primitive_execution_command_v4 import build_command_wire
from planning.runtime.primitive_execution_command_transport import (
    ZmqPrimitiveExecutionCommandClient,
)
from planning.protocol.primitive_execution_result_receipt import (
    build_result_receipt_ack_wire,
)
from planning.protocol.primitive_execution_schema_v4 import validate_execution_result
from planning.runtime.reliable_endpoint_snapshot_provider import BridgeSnapshotEndpointProvider


@dataclass(frozen=True)
class ReliableV4StepResult:
    """One immutable terminal result plus its exact endpoint observation."""

    status: str
    execution_id: int
    result: Any
    observation: Any = None


@dataclass(frozen=True)
class ReliableV4ResetResult:
    """One idempotent reset completion and its exact initial observation."""

    episode_id: str
    reset_id: str
    observation: Any


class ReliableV4ResetBackend(Protocol):
    """Public reset seam paired with the reliable v4 primitive backend."""

    def reset(
        self,
        *,
        start: Sequence[float],
        goal: Sequence[float],
        episode_id: str,
        reset_id: str,
    ) -> ReliableV4ResetResult:
        ...


class ReliableV4ResetTransport(Protocol):
    """Transport seam for the reset request/complete exchange."""

    def submit(
        self,
        payload: bytes,
        *,
        episode_id: str,
        reset_id: str,
    ) -> None:
        ...

    def receive(
        self,
        *,
        episode_id: str,
        reset_id: str,
    ) -> Mapping[str, Any]:
        ...


class ReliableV4EnvironmentBackend(Protocol):
    """Execute one already-selected motion primitive through v4."""

    def execute(
        self,
        *,
        action_id: int,
        primitive_commands: Sequence[Sequence[float]],
    ) -> ReliableV4StepResult:
        ...


class ReliableV4BackendError(RuntimeError):
    """Raised when a reliable-v4 environment transaction cannot commit."""


def _canonical_reset_request_bytes(
    *,
    runtime_instance_id: str,
    episode_id: str,
    reset_id: str,
    start: Sequence[float],
    goal: Sequence[float],
) -> bytes:
    """Encode a reset request once so retries retain identical bytes."""

    start_values = [float(value) for value in start]
    goal_values = [float(value) for value in goal]
    if len(start_values) < 3 or len(goal_values) < 3:
        raise ReliableV4BackendError("reset pose and goal require xyz")
    try:
        return build_reset_request_wire(
            runtime_instance_id=runtime_instance_id,
            episode_id=episode_id,
            reset_id=reset_id,
            start=start_values[:3],
            goal=goal_values[:3],
        )
    except ValueError as error:
        raise ReliableV4BackendError(str(error)) from error


class ReliableV4CommandTransport(Protocol):
    def submit(
        self,
        payload: bytes,
        *,
        execution_id: int,
        command_sequence_hash: str,
    ) -> None:
        ...


class ReliableV4ResultTransport(Protocol):
    def receive(
        self,
        *,
        execution_id: int,
        command_sequence_hash: str,
    ) -> Mapping[str, Any]:
        ...

    def commit(self, result: Mapping[str, Any]) -> None:
        ...


class ReliableV4ExecutionBackend:
    """Transport-neutral v4 transaction orchestrator.

    The command and result transports are injected at this seam.  Retries
    belong below it and must resend the immutable bytes owned by those
    transports.  This class maps one action to one command identity and
    exposes one endpoint snapshot to the environment; it never computes
    reward, termination, or policy features.
    """

    def __init__(
        self,
        *,
        runtime_instance_id: str,
        command_transport: ReliableV4CommandTransport,
        result_transport: ReliableV4ResultTransport,
        endpoint_provider: Any,
        execution_id_factory: Callable[[], int] = None,
        reset_transport: Optional[ReliableV4ResetTransport] = None,
    ) -> None:
        if not isinstance(runtime_instance_id, str) or not runtime_instance_id:
            raise ValueError("runtime_instance_id is required")
        self._runtime_instance_id = runtime_instance_id
        self._command_transport = command_transport
        self._result_transport = result_transport
        self._endpoint_provider = endpoint_provider
        self._execution_ids = execution_id_factory or count(44_000_000_000_000).__next__
        # The cache is a one-transaction idempotency seam.  It deliberately
        # retains at most the last completed value so an immediate duplicate
        # call can replay the committed result without retaining the full
        # collection history.
        self._completed: dict[int, ReliableV4StepResult] = {}
        self._reset_transport = reset_transport
        # Reset retries need the exact request bytes until their transaction
        # completes.  These maps are also one-transaction caches, not a
        # collection ledger.
        self._completed_resets: dict[tuple[str, str], ReliableV4ResetResult] = {}
        self._reset_payloads: dict[tuple[str, str], bytes] = {}

    def close(self) -> None:
        """Release the bounded idempotency caches at backend shutdown."""

        self._completed.clear()
        self._completed_resets.clear()
        self._reset_payloads.clear()

    def reset(
        self,
        *,
        start: Sequence[float],
        goal: Sequence[float],
        episode_id: str,
        reset_id: str,
    ) -> ReliableV4ResetResult:
        """Submit one exact reset and return its authoritative observation.

        Reset transport retries are intentionally below this seam.  A retry
        invokes ``submit`` with the exact cached bytes and the same identity;
        it never advances the episode/reset identity or applies a second reset
        locally.
        """

        if self._reset_transport is None:
            raise ReliableV4BackendError("reliable reset transport is not configured")
        if not isinstance(episode_id, str) or not episode_id:
            raise ReliableV4BackendError("episode_id is required")
        if not isinstance(reset_id, str) or not reset_id:
            raise ReliableV4BackendError("reset_id is required")
        key = (episode_id, reset_id)
        payload = _canonical_reset_request_bytes(
            runtime_instance_id=self._runtime_instance_id,
            episode_id=episode_id,
            reset_id=reset_id,
            start=start,
            goal=goal,
        )
        cached = self._completed_resets.get(key)
        if cached is not None:
            previous_payload = self._reset_payloads.get(key)
            if previous_payload is not None and previous_payload != payload:
                raise ReliableV4BackendError(
                    "conflicting reset payload for episode_id={} reset_id={}".format(
                        episode_id, reset_id
                    )
                )
            return cached
        if key not in self._reset_payloads:
            self._completed_resets.clear()
            self._reset_payloads.clear()
        previous_payload = self._reset_payloads.get(key)
        if previous_payload is not None and previous_payload != payload:
            raise ReliableV4BackendError(
                "conflicting reset payload for episode_id={} reset_id={}".format(
                    episode_id, reset_id
                )
            )
        self._reset_payloads[key] = payload

        self._reset_transport.submit(
            payload, episode_id=episode_id, reset_id=reset_id
        )
        response = self._reset_transport.receive(
            episode_id=episode_id, reset_id=reset_id
        )
        if not isinstance(response, Mapping):
            raise ReliableV4BackendError("reliable reset completion must be a mapping")
        if (
            response.get("schema_version") != PROTOCOL_VERSION
            or response.get("message_type") != "PrimitiveResetComplete"
            or response.get("runtime_instance_id") != self._runtime_instance_id
            or response.get("episode_id") != episode_id
            or response.get("reset_id") != reset_id
        ):
            raise ReliableV4BackendError("reliable reset completion identity mismatch")
        observation = response.get("observation")
        if observation is None:
            observation_ref = response.get("observation_ref")
            lookup_reset = getattr(self._endpoint_provider, "lookup_reset", None)
            if observation_ref is None or not callable(lookup_reset):
                raise ReliableV4BackendError(
                    "reliable reset completion is missing authoritative observation"
                )
            observation = lookup_reset(
                observation_ref,
                runtime_instance_id=self._runtime_instance_id,
                episode_id=episode_id,
                reset_id=reset_id,
            )
        if observation is None:
            raise ReliableV4BackendError(
                "authoritative reset observation is unavailable"
            )
        completed = ReliableV4ResetResult(
            episode_id=episode_id,
            reset_id=reset_id,
            observation=observation,
        )
        self._completed_resets.clear()
        self._completed_resets[key] = completed
        return completed

    def execute(
        self,
        *,
        action_id: int,
        primitive_commands: Sequence[Sequence[float]],
    ) -> ReliableV4StepResult:
        execution_id = int(self._execution_ids())
        cached = self._completed.get(execution_id)
        if cached is not None:
            return cached
        if self._completed:
            self._completed.clear()
        frames = tuple(
            {
                "frame_index": index,
                "command_id": execution_id * PRIMITIVE_FRAME_COUNT + index,
                "action": [float(value) for value in command],
            }
            for index, command in enumerate(primitive_commands)
        )
        command, payload = build_command_wire(
            runtime_instance_id=self._runtime_instance_id,
            execution_id=execution_id,
            frames=frames,
        )
        self._command_transport.submit(
            payload,
            execution_id=execution_id,
            command_sequence_hash=command.command_sequence_hash,
        )
        raw_result = self._result_transport.receive(
            execution_id=execution_id,
            command_sequence_hash=command.command_sequence_hash,
        )
        result = validate_execution_result(
            raw_result,
            expected_execution_id=execution_id,
            expected_command_sequence_hash=command.command_sequence_hash,
        )
        result_mapping = result.canonical_payload()
        from planning.protocol.primitive_execution_schema_v4 import canonical_result_payload_hash

        result_mapping["result_payload_hash"] = (
            result.result_payload_hash
            or canonical_result_payload_hash(result)
        )
        observation = None
        if result.status == "COMPLETE" or result.endpoint_observation_ref is not None:
            observation = self._endpoint_provider.lookup(result)
            if observation is None:
                raise ReliableV4BackendError(
                    "authoritative terminal endpoint snapshot is unavailable for "
                    "execution_id={}".format(execution_id)
                )
        self._result_transport.commit(result_mapping)
        completed = ReliableV4StepResult(
            status=result.status,
            execution_id=execution_id,
            result=result_mapping,
            observation=observation,
        )
        self._completed.clear()
        self._completed[execution_id] = completed
        return completed


class _ZmqCommandTransport:
    def __init__(self, client: ZmqPrimitiveExecutionCommandClient) -> None:
        self._client = client

    def submit(
        self,
        payload: bytes,
        *,
        execution_id: int,
        command_sequence_hash: str,
    ) -> None:
        self._client.send(payload)
        self._client.wait_for_receipt(
            execution_id=execution_id,
            command_sequence_hash=bytes.fromhex(command_sequence_hash),
        )


class _ZmqResetTransport:
    def __init__(self, client: ZmqPrimitiveExecutionCommandClient) -> None:
        self._client = client

    def submit(self, payload: bytes, *, episode_id: str, reset_id: str) -> None:
        self._client.send_reset(payload)

    def receive(self, *, episode_id: str, reset_id: str) -> Mapping[str, Any]:
        return self._client.wait_for_reset(
            episode_id=episode_id,
            reset_id=reset_id,
        )


def _normalize_result(raw: Any) -> dict:
    if not isinstance(raw, Mapping):
        raise ReliableV4BackendError("v4 result must be a map")
    result = dict(raw)
    for field in ("command_sequence_hash", "result_payload_hash"):
        if isinstance(result.get(field), bytes):
            result[field] = result[field].hex()
    if "status" in result and not result.get("result_payload_hash"):
        # Older v4 test endpoints may omit the derived field; the semantic
        # result hash is deterministic and is added only for ACK identity.
        validated = validate_execution_result(result)
        from planning.protocol.primitive_execution_schema_v4 import canonical_result_payload_hash

        result["result_payload_hash"] = canonical_result_payload_hash(validated)
    return result


class _ZmqResultTransport:
    def __init__(
        self,
        context: zmq.Context,
        *,
        endpoint: str,
        runtime_instance_id: str,
        timeout_s: float,
    ) -> None:
        self._socket = context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IDENTITY, runtime_instance_id.encode("utf-8"))
        self._socket.connect(str(endpoint))
        self._timeout_s = float(timeout_s)
        self._runtime_instance_id = str(runtime_instance_id)
        self._results: dict[int, dict] = {}
        self._hashes: dict[int, tuple[str, str]] = {}
        self._socket.send(build_result_ready_wire())

    def close(self) -> None:
        self._results.clear()
        self._hashes.clear()
        self._socket.close(0)

    @staticmethod
    def _identity(result: Mapping[str, Any]) -> tuple[str, int, str, str]:
        return (
            str(result.get("runtime_instance_id", "")),
            int(result["execution_id"]),
            str(result.get("result_payload_hash", "")),
            str(result.get("command_sequence_hash", "")),
        )

    def _offer(self, result: Mapping[str, Any]) -> None:
        candidate = dict(result)
        execution_id = int(candidate["execution_id"])
        identity = self._identity(candidate)
        previous = self._hashes.get(execution_id)
        if previous is not None and previous != (identity[2], identity[3]):
            raise ReliableV4BackendError(
                "conflicting result identity for execution_id={}".format(execution_id)
            )
        self._hashes[execution_id] = (identity[2], identity[3])
        existing = self._results.get(execution_id)
        if existing is None:
            if len(self._results) >= self._max_buffered_results:
                raise ReliableV4BackendError(
                    "reliable v4 result buffer exceeded {} entries".format(
                        self._max_buffered_results
                    )
                )
            self._results[execution_id] = candidate
        elif self._identity(existing) != identity:
            raise ReliableV4BackendError(
                "conflicting duplicate result for execution_id={}".format(execution_id)
            )

    def _send_receipt(self, result: Mapping[str, Any]) -> None:
        self._socket.send(build_result_receipt_ack_wire(result))

    def _recv(self, timeout_s: float) -> Optional[dict]:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        events = dict(poller.poll(max(1, int(float(timeout_s) * 1000.0))))
        if self._socket not in events:
            return None
        value = messagepack_unpack(self._socket.recv())
        if not isinstance(value, Mapping):
            raise ReliableV4BackendError("v4 result channel message must be a map")
        return dict(value)

    def receive(
        self,
        *,
        execution_id: int,
        command_sequence_hash: str,
    ) -> Mapping[str, Any]:
        cached = self._results.pop(int(execution_id), None)
        if cached is not None:
            self._hashes.pop(int(execution_id), None)
            return cached
        deadline = time.monotonic() + self._timeout_s
        while time.monotonic() < deadline:
            value = self._recv(deadline - time.monotonic())
            if value is None:
                continue
            if "status" not in value or "execution_id" not in value:
                continue
            result = _normalize_result(value)
            validate_execution_result(result)
            self._send_receipt(result)
            if int(result["execution_id"]) == int(execution_id):
                return result
            self._offer(result)
        raise ReliableV4BackendError(
            "reliable v4 result timeout for execution_id={}".format(execution_id)
        )

    @staticmethod
    def _commit_wire(result: Mapping[str, Any]) -> bytes:
        return build_result_commit_wire(result)

    def commit(self, result: Mapping[str, Any]) -> None:
        self._socket.send(self._commit_wire(result))
        deadline = time.monotonic() + self._timeout_s
        while time.monotonic() < deadline:
            value = self._recv(deadline - time.monotonic())
            if value is None:
                continue
            if "status" in value and "execution_id" in value:
                candidate = _normalize_result(value)
                validate_execution_result(candidate)
                self._send_receipt(candidate)
                if int(candidate["execution_id"]) == int(result["execution_id"]):
                    if self._identity(candidate) != self._identity(result):
                        raise ReliableV4BackendError(
                            "conflicting committed result for execution_id={}".format(
                                result["execution_id"]
                            )
                        )
                    # The result being committed is terminal.  It must not be
                    # reinserted into the out-of-order cache merely because
                    # the bridge repeated it while delivering the commit ACK.
                    self._results.pop(int(result["execution_id"]), None)
                    self._hashes.pop(int(result["execution_id"]), None)
                    continue
                self._offer(candidate)
                continue
            if value.get("message_type") not in (
                "PrimitiveExecutionResultCommitAck",
                "PrimitiveExecutionResultCommit",
                None,
            ):
                continue
            if int(value.get("execution_id", -1)) != int(result["execution_id"]):
                continue
            if value.get("runtime_instance_id") not in (None, result["runtime_instance_id"]):
                raise ReliableV4BackendError("commit ACK runtime identity mismatch")
            for field in ("command_sequence_hash", "result_payload_hash"):
                value_hash = value.get(field)
                if isinstance(value_hash, bytes):
                    value_hash = value_hash.hex()
                if value_hash not in (None, result[field]):
                    raise ReliableV4BackendError("commit ACK {} mismatch".format(field))
            if value.get("commit_status") != "COMMITTED":
                raise ReliableV4BackendError(
                    "reliable v4 commit rejected: {}".format(value.get("commit_status"))
                )
            return
        raise ReliableV4BackendError(
            "reliable v4 commit ACK timeout for execution_id={}".format(
                result["execution_id"]
            )
        )


class ZmqReliableV4EnvironmentBackend(ReliableV4EnvironmentBackend):
    """Real Python v4 backend for a running bridge and Unity worker."""

    def __init__(
        self,
        context: zmq.Context,
        *,
        runtime_instance_id: str,
        command_endpoint: str,
        result_endpoint: str,
        snapshot_provider: BridgeSnapshotEndpointProvider,
        timeout_s: float = 10.0,
    ) -> None:
        self._command_context = context
        self._command_client = ZmqPrimitiveExecutionCommandClient(
            context,
            endpoint=command_endpoint,
            dealer_identity=runtime_instance_id.encode("utf-8"),
            connect_timeout_s=float(timeout_s),
        )
        self._result_transport = _ZmqResultTransport(
            context,
            endpoint=result_endpoint,
            runtime_instance_id=runtime_instance_id,
            timeout_s=float(timeout_s),
        )
        self._backend = ReliableV4ExecutionBackend(
            runtime_instance_id=runtime_instance_id,
            command_transport=_ZmqCommandTransport(self._command_client),
            result_transport=self._result_transport,
            endpoint_provider=snapshot_provider,
            reset_transport=_ZmqResetTransport(self._command_client),
        )

    def execute(
        self,
        *,
        action_id: int,
        primitive_commands: Sequence[Sequence[float]],
    ) -> ReliableV4StepResult:
        return self._backend.execute(
            action_id=action_id,
            primitive_commands=primitive_commands,
        )

    def reset(
        self,
        *,
        start: Sequence[float],
        goal: Sequence[float],
        episode_id: str,
        reset_id: str,
    ) -> ReliableV4ResetResult:
        return self._backend.reset(
            start=start,
            goal=goal,
            episode_id=episode_id,
            reset_id=reset_id,
        )

    def close(self) -> None:
        self._backend.close()
        self._result_transport.close()
        self._command_client.close()


__all__ = [
    "ReliableV4ResetBackend",
    "ReliableV4ResetResult",
    "ReliableV4BackendError",
    "ReliableV4CommandTransport",
    "ReliableV4EnvironmentBackend",
    "ReliableV4ExecutionBackend",
    "ReliableV4ResultTransport",
    "ReliableV4StepResult",
    "ZmqReliableV4EnvironmentBackend",
]
