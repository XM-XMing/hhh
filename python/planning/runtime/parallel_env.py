"""Spawn-isolated Unity environment clients for a central parallel learner.

The pool deliberately owns only the Python environment clients.  A supervisor
must start one ROS master, Unity process, and bridge for every worker before
``ready`` is called.  Each child sets its worker-specific ROS environment
*before* lazily importing :mod:`planning.runtime.unity_env`.

Only the creating parent process may issue requests.  This keeps every Pipe a
single-writer channel and leaves replay storage and learner updates in the
central process.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union


PARALLEL_ENV_PROTOCOL_ID = "parallel_env_request_reply"
PARALLEL_ENV_PROTOCOL_VERSION = 1

_COMMANDS = frozenset(("ready", "reset", "step", "stop", "close"))
_PROTECTED_ENVIRONMENT_KEYS = frozenset(
    ("ROS_MASTER_URI", "ROS_HOME", "PLANNING_WORKER_ID")
)

EnvFactory = Callable[[int, Mapping[str, Any]], Any]
EnvFactoryReference = Union[str, EnvFactory]


class ParallelEnvError(RuntimeError):
    """Base class for parallel environment protocol failures."""


class ParallelEnvTimeout(TimeoutError, ParallelEnvError):
    """Raised when one or more environment workers miss a request deadline."""


class ParallelEnvWorkerError(ParallelEnvError):
    """A worker reported an exception while serving a request."""

    def __init__(
        self,
        worker_id: int,
        command: str,
        error_type: str,
        error_message: str,
        remote_traceback: str = "",
    ) -> None:
        self.worker_id = int(worker_id)
        self.command = str(command)
        self.error_type = str(error_type)
        self.error_message = str(error_message)
        self.remote_traceback = str(remote_traceback)
        super().__init__(
            "worker {} failed command {!r}: {}: {}".format(
                self.worker_id,
                self.command,
                self.error_type,
                self.error_message,
            )
        )


@dataclass(frozen=True)
class EnvWorkerSpec:
    """Serializable configuration for one ROS-isolated actor process."""

    worker_id: int
    ros_master_uri: str
    ros_home: str
    env_kwargs: Mapping[str, Any] = field(default_factory=dict)
    extra_environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvRequestHandle:
    """Opaque handle for one request currently executing in all selected envs."""

    request_id: int
    command: str
    worker_ids: Tuple[int, ...]
    deadline_monotonic: float
    pool_token: int = field(repr=False)


def _resolve_factory(reference: Optional[EnvFactoryReference]) -> EnvFactory:
    if reference is None:
        return _default_unity_env_factory
    if callable(reference):
        return reference
    if not isinstance(reference, str) or not reference.strip():
        raise TypeError("env_factory must be callable or 'module:qualname'")

    module_name, separator, qualname = reference.partition(":")
    if not separator or not module_name or not qualname:
        raise ValueError("env_factory string must use 'module:qualname'")
    value: Any = importlib.import_module(module_name)
    for name in qualname.split("."):
        value = getattr(value, name)
    if not callable(value):
        raise TypeError("resolved env_factory {!r} is not callable".format(reference))
    return value


def _default_unity_env_factory(
    worker_id: int,
    env_kwargs: Mapping[str, Any],
) -> Any:
    """Construct UnityForestEnv after the child ROS environment is installed."""

    # These imports must remain inside the spawned child.  Importing rospy in
    # the central process would bind it to the wrong ROS master.
    from planning.runtime.unity_env import EnvConfig, UnityForestEnv

    kwargs = dict(env_kwargs)
    reliable_v4_runtime = kwargs.pop("reliable_v4_runtime", None)
    if reliable_v4_runtime is not None:
        from planning.runtime.reliable_training import build_reliable_v4_backend

        kwargs["reliable_v4_backend"] = build_reliable_v4_backend(
            reliable_v4_runtime
        )
        kwargs["legacy_command_path_enabled"] = False
    config_kwargs = kwargs.pop("config_kwargs", None)
    if config_kwargs is not None:
        if "config" in kwargs:
            raise ValueError("env_kwargs cannot contain both config and config_kwargs")
        if not isinstance(config_kwargs, Mapping):
            raise TypeError("config_kwargs must be a mapping")
        kwargs["config"] = EnvConfig(**dict(config_kwargs))
    kwargs.setdefault("node_name", "planning_actor_{:02d}".format(int(worker_id)))
    kwargs.setdefault("anonymous", False)
    return UnityForestEnv(**kwargs)


def _response(
    worker_id: int,
    request_id: int,
    command: str,
    status: str,
    **fields: Any
) -> Dict[str, Any]:
    message: Dict[str, Any] = {
        "protocol_id": PARALLEL_ENV_PROTOCOL_ID,
        "protocol_version": PARALLEL_ENV_PROTOCOL_VERSION,
        "worker_id": int(worker_id),
        "request_id": int(request_id),
        "command": str(command),
        "status": str(status),
    }
    message.update(fields)
    return message


def _safe_stop_environment(env: Any) -> None:
    if env is None:
        return
    stop = getattr(env, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception:
            pass


def _close_environment(env: Any) -> Any:
    close = getattr(env, "close", None)
    result = close() if callable(close) else None
    if not callable(close):
        stop = getattr(env, "stop", None)
        if callable(stop):
            result = stop()
    backend = getattr(env, "_reliable_v4_backend", None)
    backend_close = getattr(backend, "close", None)
    if callable(backend_close):
        backend_close()
    return result


class _WorkerEnvironmentSession:
    """Child-only state binding action selection to the same sensor frame."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.observation = None
        self.action_mask = None
        self.action_mask_info = None
        self.needs_reset = True

    def _require_reset(self) -> None:
        self.needs_reset = True
        self.observation = None
        self.action_mask = None
        self.action_mask_info = None

    def _capture_observation(self, observation: Any) -> Dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise TypeError("Unity environment observation must be a mapping")
        action_mask, action_mask_info = self.env.get_action_mask(
            observation, return_info=True
        )
        self.observation = observation
        self.action_mask = action_mask
        self.action_mask_info = action_mask_info
        return {
            "observation": observation,
            "action_mask": action_mask,
            "action_mask_info": action_mask_info,
        }

    def _materialize_valid_terminal_abort(
        self,
        error: Exception,
        *,
        action_id: int,
    ) -> Optional[Tuple[Any, float, bool, Any]]:
        """Return one normal step only for a prevalidated collision abort.

        ``UnityForestEnv`` already classifies every failed primitive receipt
        against the submitted command sequence before it raises the typed
        exception.  The training worker consumes that result; it deliberately
        does not reinterpret a partial receipt or turn a protocol error into
        an environment terminal transition.
        """

        # Keep Unity/ROS imports out of CPU-only parents and ordinary worker
        # errors.  The actual type check below prevents a look-alike exception
        # from taking the terminal-abort path.
        if type(error).__name__ != "PrimitiveExecutionAbortedError":
            return None
        from planning.contracts.primitive_execution import (
            PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT,
        )
        from planning.runtime.unity_env import PrimitiveExecutionAbortedError

        if not isinstance(error, PrimitiveExecutionAbortedError):
            return None
        execution_result = error.execution_result
        if (
            not isinstance(execution_result, Mapping)
            or execution_result.get("kind")
            != PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT
        ):
            return None
        return self.env.materialize_terminal_abort(
            error,
            action_id=int(action_id),
            obs_before=self.observation,
            precomputed_mask=self.action_mask,
            precomputed_mask_info=self.action_mask_info,
        )

    def serve(self, command: str, payload: Mapping[str, Any]) -> Any:
        try:
            if command == "ready":
                timeout_s = float(payload["timeout_s"])
                return self.env.wait_until_ready(timeout_s=timeout_s)
            if command == "reset":
                kwargs = payload.get("kwargs", {})
                if not isinstance(kwargs, Mapping):
                    raise TypeError("reset kwargs must be a mapping")
                response = self._capture_observation(
                    self.env.reset(**dict(kwargs))
                )
                self.needs_reset = False
                return response
            if command == "step":
                if self.needs_reset:
                    raise RuntimeError(
                        "step requires a successful reset after initialization, "
                        "terminal, stop, or error"
                    )
                action_id = int(payload["action_id"])
                try:
                    result = self.env.step_primitive(
                        action_id,
                        obs_before=self.observation,
                        precomputed_mask=self.action_mask,
                        precomputed_mask_info=self.action_mask_info,
                    )
                except Exception as error:
                    result = self._materialize_valid_terminal_abort(
                        error,
                        action_id=action_id,
                    )
                    if result is None:
                        raise
                if not isinstance(result, tuple) or len(result) != 4:
                    raise TypeError(
                        "step_primitive must return (obs, reward, done, info)"
                    )
                next_observation, reward, done, info = result
                # Continuous execution deliberately leaves the last primitive
                # command active between non-terminal decisions.  A terminal
                # worker, however, may remain idle in the parent pool until the
                # rest of the parallel episode wave finishes.  Stop it here, in
                # the child that owns its ROS connection, before doing any
                # post-processing or waiting for the next parent request.
                terminal = bool(done)
                if terminal:
                    self.env.stop()
                response = self._capture_observation(next_observation)
                response.update(
                    {
                        "reward": reward,
                        "done": done,
                        "info": info,
                        # Linux monotonic clocks are shared across processes;
                        # the parent uses this to audit reply-to-next-command
                        # controller gaps and parallel-wave completion skew.
                        "worker_step_completed_monotonic_s": time.monotonic(),
                    }
                )
                if terminal:
                    self._require_reset()
                return response
            if command == "stop":
                try:
                    return self.env.stop()
                finally:
                    self._require_reset()
            if command == "close":
                try:
                    return _close_environment(self.env)
                finally:
                    self._require_reset()
            raise ValueError("unsupported command {!r}".format(command))
        except Exception:
            self._require_reset()
            raise


def _worker_main(
    spec: EnvWorkerSpec,
    connection: Connection,
    env_factory: Optional[EnvFactoryReference],
) -> None:
    """Child entry point; intentionally importable by multiprocessing spawn."""

    worker_id = int(spec.worker_id)
    env = None
    session = None
    closed = False
    try:
        for key, value in dict(spec.extra_environment).items():
            os.environ[str(key)] = str(value)
        os.environ["ROS_MASTER_URI"] = str(spec.ros_master_uri)
        os.environ["ROS_HOME"] = str(spec.ros_home)
        os.environ["PLANNING_WORKER_ID"] = str(worker_id)
        Path(spec.ros_home).mkdir(parents=True, exist_ok=True)

        factory = _resolve_factory(env_factory)
        env = factory(worker_id, dict(spec.env_kwargs))
        session = _WorkerEnvironmentSession(env)
        connection.send(
            _response(
                worker_id,
                request_id=0,
                command="startup",
                status="ok",
                result={
                    "pid": int(os.getpid()),
                    "ros_master_uri": str(os.environ["ROS_MASTER_URI"]),
                    "ros_home": str(os.environ["ROS_HOME"]),
                },
            )
        )

        while True:
            request = connection.recv()
            request_id = -1
            command = "unknown"
            try:
                if not isinstance(request, Mapping):
                    raise TypeError("request must be a mapping")
                request_id = int(request.get("request_id", -1))
                command = str(request.get("command", "unknown"))
                if request.get("protocol_id") != PARALLEL_ENV_PROTOCOL_ID:
                    raise ValueError("request protocol_id mismatch")
                if int(request.get("protocol_version", -1)) != PARALLEL_ENV_PROTOCOL_VERSION:
                    raise ValueError("request protocol_version mismatch")
                request_worker_id = int(request.get("worker_id", -1))
                if request_worker_id != worker_id:
                    raise ValueError(
                        "request worker_id {} does not match worker {}".format(
                            request_worker_id, worker_id
                        )
                    )
                if command not in _COMMANDS:
                    raise ValueError("unsupported command {!r}".format(command))
                payload = request.get("payload", {})
                if not isinstance(payload, Mapping):
                    raise TypeError("request payload must be a mapping")

                result = session.serve(command, payload)
                connection.send(
                    _response(
                        worker_id,
                        request_id=request_id,
                        command=command,
                        status="ok",
                        result=result,
                    )
                )
                if command == "close":
                    closed = True
                    return
            except Exception as exc:
                remote_traceback = traceback.format_exc()
                if session is not None:
                    session._require_reset()
                # A failed command may have left continuous motion active.
                # Publish stop before reporting the error to the parent so the
                # learner can never observe an error while this worker flies on.
                _safe_stop_environment(env)
                connection.send(
                    _response(
                        worker_id,
                        request_id=request_id,
                        command=command,
                        status="error",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        traceback=remote_traceback,
                    )
                )
    except EOFError:
        return
    except Exception as exc:
        try:
            connection.send(
                _response(
                    worker_id,
                    request_id=0,
                    command="startup",
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback=traceback.format_exc(),
                )
            )
        except Exception:
            pass
    finally:
        if not closed:
            _safe_stop_environment(env)
        try:
            connection.close()
        except Exception:
            pass


class ParallelEnvPool:
    """Request/reply pool for ROS-isolated parallel actor environments.

    The pool starts processes immediately.  Calls are serialized and every
    broadcast sends all requests before waiting, so environment steps proceed
    concurrently while the parent remains the only Pipe writer.
    """

    def __init__(
        self,
        worker_specs: Sequence[EnvWorkerSpec],
        *,
        env_factory: Optional[EnvFactoryReference] = None,
        startup_timeout_s: float = 30.0,
        request_timeout_s: float = 30.0,
    ) -> None:
        self._owner_pid = int(os.getpid())
        self._lock = threading.RLock()
        self._closed = False
        self._broken = False
        self._request_id = 0
        self._request_timeout_s = self._positive_timeout(
            request_timeout_s, "request_timeout_s"
        )
        self._pool_token = int(id(self))
        self._inflight: Optional[EnvRequestHandle] = None
        startup_timeout = self._positive_timeout(startup_timeout_s, "startup_timeout_s")
        self._specs = self._validate_specs(worker_specs)
        self._worker_ids = tuple(spec.worker_id for spec in self._specs)
        # Never use fork here: a parent that has imported rospy would carry a
        # ROS client bound to the wrong master into every worker.
        self._context = mp.get_context("spawn")
        self._connections: Dict[int, Connection] = {}
        self._processes: Dict[int, mp.Process] = {}
        self._startup: Dict[int, Any] = {}

        try:
            for spec in self._specs:
                parent_connection, child_connection = self._context.Pipe(duplex=True)
                process = self._context.Process(
                    target=_worker_main,
                    args=(spec, child_connection, env_factory),
                    name="planning-env-worker-{:02d}".format(spec.worker_id),
                )
                process.daemon = False
                try:
                    process.start()
                except Exception:
                    parent_connection.close()
                    child_connection.close()
                    raise
                child_connection.close()
                self._connections[spec.worker_id] = parent_connection
                self._processes[spec.worker_id] = process

            self._startup = self._collect_replies(
                worker_ids=self._worker_ids,
                request_id=0,
                command="startup",
                timeout_s=startup_timeout,
            )
        except Exception:
            self._broken = True
            self._shutdown_processes(grace_timeout_s=0.25)
            raise

    @staticmethod
    def _positive_timeout(value: float, name: str) -> float:
        timeout = float(value)
        if not timeout > 0.0:
            raise ValueError("{} must be > 0".format(name))
        return timeout

    @staticmethod
    def _validate_specs(
        worker_specs: Sequence[EnvWorkerSpec],
    ) -> Tuple[EnvWorkerSpec, ...]:
        specs = tuple(worker_specs)
        if not specs:
            raise ValueError("worker_specs must not be empty")
        if not all(isinstance(spec, EnvWorkerSpec) for spec in specs):
            raise TypeError("worker_specs must contain EnvWorkerSpec values")

        normalized = tuple(
            EnvWorkerSpec(
                worker_id=int(spec.worker_id),
                ros_master_uri=str(spec.ros_master_uri),
                ros_home=str(spec.ros_home),
                env_kwargs=dict(spec.env_kwargs),
                extra_environment={
                    str(key): str(value)
                    for key, value in dict(spec.extra_environment).items()
                },
            )
            for spec in specs
        )
        worker_ids = tuple(spec.worker_id for spec in normalized)
        if worker_ids != tuple(range(len(normalized))):
            raise ValueError(
                "worker_specs must be ordered with contiguous worker_id values 0..{}".format(
                    len(normalized) - 1
                )
            )
        master_uris = tuple(spec.ros_master_uri for spec in normalized)
        ros_homes = tuple(spec.ros_home for spec in normalized)
        if any(not value for value in master_uris):
            raise ValueError("ros_master_uri must not be empty")
        if len(set(master_uris)) != len(master_uris):
            raise ValueError("each worker requires a unique ros_master_uri")
        if any(not value for value in ros_homes):
            raise ValueError("ros_home must not be empty")
        if len(set(ros_homes)) != len(ros_homes):
            raise ValueError("each worker requires a unique ros_home")
        for spec in normalized:
            protected = _PROTECTED_ENVIRONMENT_KEYS.intersection(
                spec.extra_environment.keys()
            )
            if protected:
                raise ValueError(
                    "extra_environment cannot override {}".format(
                        ", ".join(sorted(protected))
                    )
                )
        return normalized

    @property
    def worker_ids(self) -> Tuple[int, ...]:
        return self._worker_ids

    @property
    def startup_metadata(self) -> Dict[int, Any]:
        return dict(self._startup)

    @property
    def is_closed(self) -> bool:
        return bool(self._closed)

    def _check_owner(self) -> None:
        if int(os.getpid()) != self._owner_pid:
            raise ParallelEnvError(
                "environment pool may only be used by its creating parent process"
            )
        if self._closed:
            raise ParallelEnvError("environment pool is closed")
        if self._broken:
            raise ParallelEnvError(
                "environment pool is unusable after a protocol or timeout failure"
            )

    def _normalize_worker_ids(
        self, worker_ids: Optional[Iterable[int]]
    ) -> Tuple[int, ...]:
        if worker_ids is None:
            return self._worker_ids
        selected = tuple(int(worker_id) for worker_id in worker_ids)
        if not selected:
            raise ValueError("worker_ids must not be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("worker_ids must be unique")
        unknown = sorted(set(selected).difference(self._worker_ids))
        if unknown:
            raise ValueError("unknown worker_id values: {}".format(unknown))
        return selected

    def _next_request_id(self) -> int:
        self._request_id += 1
        return int(self._request_id)

    def _collect_replies(
        self,
        *,
        worker_ids: Sequence[int],
        request_id: int,
        command: str,
        timeout_s: float,
    ) -> Dict[int, Any]:
        deadline = time.monotonic() + float(timeout_s)
        pending = {
            self._connections[int(worker_id)]: int(worker_id)
            for worker_id in worker_ids
        }
        results: Dict[int, Any] = {}
        errors = []

        while pending:
            remaining = deadline - time.monotonic()
            # A reply that reached the Pipe at the deadline is complete work,
            # not a timeout.  Always perform one non-blocking drain when the
            # budget is exhausted before classifying the remaining workers.
            ready_connections = wait(
                tuple(pending.keys()), timeout=max(0.0, remaining)
            )
            if not ready_connections:
                self._broken = True
                raise ParallelEnvTimeout(
                    "command {!r} timed out after {:.3f}s; pending workers={}".format(
                        command,
                        float(timeout_s),
                        sorted(pending.values()),
                    )
                )
            for connection in ready_connections:
                expected_worker_id = pending.pop(connection)
                try:
                    reply = connection.recv()
                except EOFError as exc:
                    errors.append(
                        ParallelEnvWorkerError(
                            expected_worker_id,
                            command,
                            "EOFError",
                            "worker exited before replying",
                        )
                    )
                    continue
                try:
                    result = self._validate_reply(
                        reply,
                        expected_worker_id=expected_worker_id,
                        expected_request_id=request_id,
                        expected_command=command,
                    )
                    results[expected_worker_id] = result
                except Exception as exc:
                    errors.append(exc)

        if errors:
            self._broken = True
            raise errors[0]
        return {worker_id: results[worker_id] for worker_id in worker_ids}

    @staticmethod
    def _validate_reply(
        reply: Any,
        *,
        expected_worker_id: int,
        expected_request_id: int,
        expected_command: str,
    ) -> Any:
        if not isinstance(reply, Mapping):
            raise ParallelEnvError(
                "worker {} returned a non-mapping reply".format(expected_worker_id)
            )
        if reply.get("protocol_id") != PARALLEL_ENV_PROTOCOL_ID:
            raise ParallelEnvError(
                "worker {} reply protocol_id mismatch".format(expected_worker_id)
            )
        if int(reply.get("protocol_version", -1)) != PARALLEL_ENV_PROTOCOL_VERSION:
            raise ParallelEnvError(
                "worker {} reply protocol_version mismatch".format(expected_worker_id)
            )
        actual_worker_id = int(reply.get("worker_id", -1))
        if actual_worker_id != int(expected_worker_id):
            raise ParallelEnvError(
                "reply worker_id {} does not match expected {}".format(
                    actual_worker_id, expected_worker_id
                )
            )
        actual_request_id = int(reply.get("request_id", -1))
        if actual_request_id != int(expected_request_id):
            raise ParallelEnvError(
                "worker {} reply request_id {} does not match expected {}".format(
                    expected_worker_id, actual_request_id, expected_request_id
                )
            )
        actual_command = str(reply.get("command", ""))
        if actual_command != str(expected_command):
            raise ParallelEnvError(
                "worker {} reply command {!r} does not match {!r}".format(
                    expected_worker_id, actual_command, expected_command
                )
            )
        status = str(reply.get("status", ""))
        if status == "error":
            raise ParallelEnvWorkerError(
                expected_worker_id,
                expected_command,
                str(reply.get("error_type", "RemoteError")),
                str(reply.get("error_message", "")),
                str(reply.get("traceback", "")),
            )
        if status != "ok":
            raise ParallelEnvError(
                "worker {} returned invalid status {!r}".format(
                    expected_worker_id, status
                )
            )
        return reply.get("result")

    def _begin_request(
        self,
        command: str,
        payloads: Mapping[int, Mapping[str, Any]],
        *,
        timeout_s: Optional[float] = None,
    ) -> EnvRequestHandle:
        with self._lock:
            self._check_owner()
            if self._inflight is not None:
                raise ParallelEnvError(
                    "request {} ({!r}) is still in flight".format(
                        self._inflight.request_id, self._inflight.command
                    )
                )
            if command not in _COMMANDS:
                raise ValueError("unsupported command {!r}".format(command))
            worker_ids = self._normalize_worker_ids(payloads.keys())
            request_timeout = self._request_timeout_s
            if timeout_s is not None:
                request_timeout = self._positive_timeout(timeout_s, "timeout_s")
            request_id = self._next_request_id()

            try:
                for worker_id in worker_ids:
                    process = self._processes[worker_id]
                    if not process.is_alive():
                        raise ParallelEnvWorkerError(
                            worker_id,
                            command,
                            "WorkerExited",
                            "worker process exitcode={}".format(process.exitcode),
                        )
                    payload = payloads[worker_id]
                    if not isinstance(payload, Mapping):
                        raise TypeError(
                            "payload for worker {} must be a mapping".format(worker_id)
                        )
                    self._connections[worker_id].send(
                        {
                            "protocol_id": PARALLEL_ENV_PROTOCOL_ID,
                            "protocol_version": PARALLEL_ENV_PROTOCOL_VERSION,
                            "worker_id": int(worker_id),
                            "request_id": int(request_id),
                            "command": str(command),
                            "payload": dict(payload),
                        }
                    )
            except Exception:
                # Some workers may already have received this request.  No
                # subsequent command can safely share these Pipes.
                self._broken = True
                raise

            handle = EnvRequestHandle(
                request_id=request_id,
                command=str(command),
                worker_ids=tuple(worker_ids),
                deadline_monotonic=time.monotonic() + request_timeout,
                pool_token=self._pool_token,
            )
            self._inflight = handle
            return handle

    def _finish_request(
        self,
        handle: EnvRequestHandle,
        *,
        timeout_s: Optional[float] = None,
    ) -> Dict[int, Any]:
        with self._lock:
            self._check_owner()
            if not isinstance(handle, EnvRequestHandle):
                raise TypeError("request handle must be EnvRequestHandle")
            if handle.pool_token != self._pool_token or handle != self._inflight:
                raise ParallelEnvError(
                    "request handle does not match this pool's in-flight request"
                )
            remaining = max(0.0, float(handle.deadline_monotonic) - time.monotonic())
            if timeout_s is not None:
                remaining = min(
                    remaining,
                    self._positive_timeout(timeout_s, "timeout_s"),
                )
            try:
                return self._collect_replies(
                    worker_ids=handle.worker_ids,
                    request_id=handle.request_id,
                    command=handle.command,
                    timeout_s=remaining,
                )
            finally:
                self._inflight = None

    def _request(
        self,
        command: str,
        payloads: Mapping[int, Mapping[str, Any]],
        *,
        timeout_s: Optional[float] = None,
    ) -> Dict[int, Any]:
        handle = self._begin_request(command, payloads, timeout_s=timeout_s)
        return self._finish_request(handle)

    def ready(
        self,
        worker_ids: Optional[Iterable[int]] = None,
        *,
        worker_ready_timeout_s: float = 30.0,
        timeout_s: Optional[float] = None,
    ) -> Dict[int, Any]:
        """Wait for the selected workers' Unity/ROS observations."""

        selected = self._normalize_worker_ids(worker_ids)
        ready_timeout = self._positive_timeout(
            worker_ready_timeout_s, "worker_ready_timeout_s"
        )
        request_timeout = timeout_s
        if request_timeout is None:
            request_timeout = ready_timeout + 1.0
        return self._request(
            "ready",
            {worker_id: {"timeout_s": ready_timeout} for worker_id in selected},
            timeout_s=request_timeout,
        )

    def reset(
        self,
        reset_kwargs_by_worker: Mapping[int, Mapping[str, Any]],
        *,
        timeout_s: Optional[float] = None,
    ) -> Dict[int, Any]:
        """Reset workers and return observation/action_mask/action_mask_info."""

        return self._request(
            "reset",
            {
                int(worker_id): {"kwargs": dict(kwargs)}
                for worker_id, kwargs in reset_kwargs_by_worker.items()
            },
            timeout_s=timeout_s,
        )

    def step(
        self,
        actions_by_worker: Mapping[int, int],
        *,
        timeout_s: Optional[float] = None,
    ) -> Dict[int, Any]:
        """Return observation/mask/reward/done/info for concurrent actions."""

        handle = self.begin_step(actions_by_worker, timeout_s=timeout_s)
        return self.finish_step(handle)

    def begin_step(
        self,
        actions_by_worker: Mapping[int, int],
        *,
        timeout_s: Optional[float] = None,
    ) -> EnvRequestHandle:
        """Start concurrent Unity steps and return without waiting for them."""

        return self._begin_request(
            "step",
            {
                int(worker_id): {"action_id": int(action_id)}
                for worker_id, action_id in actions_by_worker.items()
            },
            timeout_s=timeout_s,
        )

    def finish_step(
        self,
        handle: EnvRequestHandle,
        *,
        timeout_s: Optional[float] = None,
    ) -> Dict[int, Any]:
        """Collect the exact step started by :meth:`begin_step`."""

        if handle.command != "step":
            raise ParallelEnvError("finish_step requires a step request handle")
        return self._finish_request(handle, timeout_s=timeout_s)

    def stop(
        self,
        worker_ids: Optional[Iterable[int]] = None,
        *,
        timeout_s: Optional[float] = None,
    ) -> Dict[int, Any]:
        """Publish stop through each selected environment without exiting."""

        selected = self._normalize_worker_ids(worker_ids)
        return self._request(
            "stop",
            {worker_id: {} for worker_id in selected},
            timeout_s=timeout_s,
        )

    def close(self, timeout_s: float = 3.0) -> None:
        """Close all workers, terminating stragglers; safe to call repeatedly."""

        if int(os.getpid()) != self._owner_pid or self._closed:
            return
        with self._lock:
            if self._closed:
                return
            graceful_timeout = self._positive_timeout(timeout_s, "timeout_s")
            if self._inflight is not None:
                # Sending close behind an unfinished step would make reply
                # matching ambiguous.  Terminate instead.
                self._broken = True
            if not self._broken:
                alive = tuple(
                    worker_id
                    for worker_id, process in self._processes.items()
                    if process.is_alive()
                )
                if alive:
                    try:
                        self._request(
                            "close",
                            {worker_id: {} for worker_id in alive},
                            timeout_s=graceful_timeout,
                        )
                    except Exception:
                        self._broken = True
            self._closed = True
            self._shutdown_processes(grace_timeout_s=graceful_timeout)

    def _shutdown_processes(self, grace_timeout_s: float) -> None:
        deadline = time.monotonic() + max(0.0, float(grace_timeout_s))
        for process in self._processes.values():
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)
        for process in self._processes.values():
            if process.is_alive():
                process.terminate()
        for process in self._processes.values():
            if process.is_alive():
                process.join(timeout=1.0)
        for process in self._processes.values():
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
        for connection in self._connections.values():
            try:
                connection.close()
            except Exception:
                pass

    def __enter__(self) -> "ParallelEnvPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
