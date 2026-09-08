"""Shared owner for one or more managed ROS/Unity/Bridge runtimes.

``ParallelEnvPool`` owns only Python environment workers.  This module owns
the external processes those workers depend on, so evaluation and AWAC
calibration use the same startup, readiness, identity, and cleanup contract.
The default implementation launches each process in its own process group and
only ever terminates process groups created by this instance.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from planning.common import file_sha256, write_json_atomic
from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    LEGACY_ASYNC_OBSERVATION_CONTRACT,
)
from planning.contracts.task import task_contract_sha256 as task_contract_sha256_for_steps
from planning.protocol.constants import SCHEMA_VERSION
from planning.runtime.ports import (
    WorkerRuntimeSpec,
    is_tcp_port_available,
    validate_worker_runtime_specs,
)


MANAGED_RUNTIME_LIFECYCLE_SCHEMA = "planning_managed_runtime_lifecycle_v1"
MANAGED_RUNTIME_STARTUP_ERROR = "MANAGED_RUNTIME_STARTUP_ERROR"


class ManagedRuntimeError(RuntimeError):
    """A managed external runtime cannot satisfy its lifecycle contract."""


ProcessFactory = Callable[..., Any]
ReadinessChecker = Callable[[str, WorkerRuntimeSpec, Mapping[str, Any], Mapping[str, str], float], Any]


def _default_process_factory(argv: Sequence[str], **kwargs: Any) -> Any:
    return subprocess.Popen(list(argv), **kwargs)


def _default_mpl_contract_sha256() -> str:
    try:
        from planning.primitives.library import MotionPrimitiveLibrary

        return str(MotionPrimitiveLibrary().contract_sha256)
    except Exception as exc:
        raise ManagedRuntimeError(
            "{}: cannot resolve formal MPL contract: {}".format(
                MANAGED_RUNTIME_STARTUP_ERROR, exc
            )
        ) from exc


def _required_executable(path: Path, name: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise ManagedRuntimeError(
            "{}: {} does not exist: {}".format(
                MANAGED_RUNTIME_STARTUP_ERROR, name, value
            )
        )
    if not os.access(str(value), os.X_OK):
        raise ManagedRuntimeError(
            "{}: {} is not executable: {}".format(
                MANAGED_RUNTIME_STARTUP_ERROR, name, value
            )
        )
    return value


def _required_file(path: Path, name: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise ManagedRuntimeError(
            "{}: {} does not exist: {}".format(
                MANAGED_RUNTIME_STARTUP_ERROR, name, value
            )
        )
    return value


def _assembly_path(unity_bin: Path) -> Path:
    return unity_bin.parent / (unity_bin.stem + "_Data") / "Managed" / "Assembly-CSharp.dll"


def _terminate_process_group(process: Any, *, use_process_groups: bool, grace_s: float) -> None:
    """Terminate one owned process without matching unrelated processes."""

    poll = getattr(process, "poll", None)
    if callable(poll) and poll() is not None:
        return

    pid = getattr(process, "pid", None)
    signalled_group = False
    if use_process_groups and pid is not None:
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            signalled_group = True
        except (OSError, ProcessLookupError):
            signalled_group = False
    if not signalled_group:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
            except Exception:
                pass

    wait = getattr(process, "wait", None)
    if callable(wait):
        try:
            wait(timeout=max(0.0, float(grace_s)))
            return
        except Exception:
            pass

    if use_process_groups and pid is not None:
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    kill = getattr(process, "kill", None)
    if callable(kill):
        try:
            kill()
        except Exception:
            pass
    if callable(wait):
        try:
            wait(timeout=1.0)
        except Exception:
            pass


class ManagedRuntimePool:
    """Own the external ROS master, Unity, and Bridge for each worker.

    The interface is intentionally small: construct with canonical worker
    specs, call :meth:`start`, use the resulting runtime with
    ``ParallelEnvPool``, and call :meth:`close` (or use a context manager).
    Readiness and process launch are injectable system adapters for CPU-only
    lifecycle tests; production uses ``Popen`` and ROS command probes.
    """

    def __init__(
        self,
        worker_specs: Sequence[WorkerRuntimeSpec],
        *,
        unity_bin: Optional[Path] = None,
        bridge_bin: Optional[Path] = None,
        workspace: Optional[Path] = None,
        log_dir: Path,
        task_contract_sha256: str = "",
        mpl_contract_sha256: str = "",
        observation_contract: str = EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        max_steps: int = 45,
        startup_timeout_s: float = 60.0,
        window_width: int = 320,
        window_height: int = 240,
        unity_extra_args: Sequence[str] = (),
        bridge_extra_args: Sequence[str] = (),
        reliable_v4: bool = True,
        process_factory: Optional[ProcessFactory] = None,
        readiness_checker: Optional[ReadinessChecker] = None,
        port_available: Optional[Callable[[int], bool]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        try:
            self._worker_specs = validate_worker_runtime_specs(tuple(worker_specs))
        except Exception as exc:
            raise ManagedRuntimeError(
                "{}: invalid worker runtime specs: {}".format(
                    MANAGED_RUNTIME_STARTUP_ERROR, exc
                )
            ) from exc
        self._workspace = (
            Path(workspace).expanduser().resolve()
            if workspace is not None
            else Path.cwd().resolve()
        )
        default_unity = self._workspace / "src" / "unity" / "XMflight.x86_64"
        self._unity_bin = Path(unity_bin or default_unity).expanduser().resolve()
        if bridge_bin is None:
            from planning.runtime.bridge_identity import planning_bridge_binary

            bridge_value = str(os.environ.get("PLANNING_BRIDGE_BINARY", "")).strip()
            bridge_bin = Path(bridge_value) if bridge_value else planning_bridge_binary()
        self._bridge_bin = Path(bridge_bin).expanduser().resolve()
        self._log_dir = Path(log_dir).expanduser().resolve()
        self._task_contract_sha256 = str(task_contract_sha256).strip().lower()
        if not self._task_contract_sha256:
            self._task_contract_sha256 = task_contract_sha256_for_steps(int(max_steps))
        self._mpl_contract_sha256 = str(mpl_contract_sha256).strip().lower()
        if not self._mpl_contract_sha256:
            self._mpl_contract_sha256 = _default_mpl_contract_sha256().lower()
        self._reliable_mode = bool(reliable_v4)
        self._observation_contract = str(observation_contract).strip()
        expected_observation_contract = (
            EXACT_ENDPOINT_OBSERVATION_CONTRACT
            if self._reliable_mode
            else LEGACY_ASYNC_OBSERVATION_CONTRACT
        )
        if self._observation_contract != expected_observation_contract:
            raise ManagedRuntimeError(
                "{}: managed runtime mode requires {}".format(
                    MANAGED_RUNTIME_STARTUP_ERROR,
                    expected_observation_contract,
                )
            )
        self._max_steps = int(max_steps)
        if self._max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._startup_timeout_s = float(startup_timeout_s)
        if self._startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be positive")
        self._window_width = int(window_width)
        self._window_height = int(window_height)
        self._unity_extra_args = tuple(str(value) for value in unity_extra_args)
        self._bridge_extra_args = tuple(str(value) for value in bridge_extra_args)
        self._process_factory = process_factory or _default_process_factory
        self._use_process_groups = process_factory is None
        self._readiness_checker = readiness_checker or self._default_readiness_checker
        self._port_available = port_available or is_tcp_port_available
        self._sleep = sleep_fn
        self._owned_processes: list[Tuple[int, str, Any]] = []
        self._readiness: list[Dict[str, Any]] = []
        self._identity: Optional[Dict[str, Any]] = None
        self._started = False
        self._closed = False

    @property
    def worker_specs(self) -> Tuple[WorkerRuntimeSpec, ...]:
        return self._worker_specs

    @property
    def worker_count(self) -> int:
        return len(self._worker_specs)

    @property
    def runtime_identity(self) -> Dict[str, Any]:
        if self._identity is None:
            raise ManagedRuntimeError("managed runtime has not started")
        return copy.deepcopy(self._identity)

    @property
    def is_started(self) -> bool:
        return bool(self._started and not self._closed)

    def environment_for_worker(self, worker_id: int) -> Dict[str, str]:
        spec = self._spec_for_worker(worker_id)
        if not self._started:
            raise ManagedRuntimeError("managed runtime has not started")
        return self._environment_for_spec(spec)

    def start(self) -> Dict[str, Any]:
        if self._closed:
            raise ManagedRuntimeError("managed runtime is closed")
        if self._started:
            return self.runtime_identity
        self._validate_startup_inputs()
        self._preflight_ports()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        try:
            for spec in self._worker_specs:
                self._start_worker_runtime(spec)
            self._identity = self._build_identity()
            write_json_atomic(
                self._log_dir / "managed_runtime_manifest.json",
                self._identity,
                trailing_newline=True,
            )
            self._started = True
            return self.runtime_identity
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _worker_id, _role, process in reversed(self._owned_processes):
            _terminate_process_group(
                process,
                use_process_groups=self._use_process_groups,
                grace_s=2.0,
            )
        self._owned_processes = []
        self._started = False

    def run_command(
        self,
        argv: Sequence[str],
        *,
        worker_id: int = 0,
        environment: Optional[Mapping[str, str]] = None,
        cwd: Optional[Path] = None,
    ) -> int:
        """Run one caller command inside an owned runtime process group.

        The command is deliberately a child of this owner.  This gives the
        evaluator and any future managed caller identical interrupt and
        cleanup semantics without putting caller-specific training logic into
        the lifecycle module.
        """

        if not self.is_started:
            raise ManagedRuntimeError("managed runtime has not started")
        command = tuple(str(value) for value in argv)
        if not command:
            raise ValueError("managed runtime command must not be empty")
        command_environment = self.environment_for_worker(int(worker_id))
        if environment is not None:
            command_environment.update(
                {str(key): str(value) for key, value in dict(environment).items()}
            )
        process = subprocess.Popen(
            list(command),
            cwd=str(
                Path(cwd).expanduser().resolve()
                if cwd is not None
                else self._workspace
            ),
            env=command_environment,
            start_new_session=True,
        )
        self._owned_processes.append((int(worker_id), "caller", process))
        try:
            return int(process.wait())
        except KeyboardInterrupt:
            _terminate_process_group(
                process,
                use_process_groups=self._use_process_groups,
                grace_s=2.0,
            )
            raise

    def __enter__(self) -> "ManagedRuntimePool":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _spec_for_worker(self, worker_id: int) -> WorkerRuntimeSpec:
        for spec in self._worker_specs:
            if int(spec.worker_id) == int(worker_id):
                return spec
        raise ManagedRuntimeError("unknown managed worker_id={}".format(worker_id))

    def _validate_startup_inputs(self) -> None:
        _required_executable(self._unity_bin, "Unity executable")
        _required_file(_assembly_path(self._unity_bin), "Unity Assembly-CSharp.dll")
        _required_executable(self._bridge_bin, "Bridge executable")
        if not self._task_contract_sha256:
            raise ManagedRuntimeError("{}: missing Task Contract SHA".format(MANAGED_RUNTIME_STARTUP_ERROR))
        if not self._mpl_contract_sha256:
            raise ManagedRuntimeError("{}: missing MPL Contract SHA".format(MANAGED_RUNTIME_STARTUP_ERROR))

    def _preflight_ports(self) -> None:
        for spec in self._worker_specs:
            for port in spec.all_ports:
                try:
                    available = bool(self._port_available(int(port)))
                except Exception as exc:
                    raise ManagedRuntimeError(
                        "{}: port preflight failed for {}: {}".format(
                            MANAGED_RUNTIME_STARTUP_ERROR, port, exc
                        )
                    ) from exc
                if not available:
                    raise ManagedRuntimeError(
                        "{}: port {} is already occupied or unavailable".format(
                            MANAGED_RUNTIME_STARTUP_ERROR, port
                        )
                    )

    def _environment_for_spec(self, spec: WorkerRuntimeSpec) -> Dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "ROS_MASTER_URI": str(spec.ros_master_uri),
                "ROS_HOSTNAME": "127.0.0.1",
                "ROS_HOME": str(spec.ros_home),
                "PLANNING_WORKER_ID": str(spec.worker_id),
                "PLANNING_RUNTIME_INSTANCE_ID": str(spec.runtime_instance_id),
                "PLANNING_TRAINING_RUN_ID": str(spec.training_run_id),
                "PLANNING_RUNTIME_LAUNCH_NONCE": str(spec.runtime_launch_nonce),
            }
        )
        return environment

    def _start_worker_runtime(self, spec: WorkerRuntimeSpec) -> None:
        worker_dir = self._log_dir / "worker_{:02d}".format(int(spec.worker_id))
        worker_dir.mkdir(parents=True, exist_ok=True)
        Path(spec.ros_home).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        environment = self._environment_for_spec(spec)

        roscore = self._launch(
            spec,
            "ros_master",
            ["roscore", "-p", str(spec.master_port)],
            environment,
            worker_dir / "roscore.log",
        )
        self._wait_ready("ros_master", spec, {"ros_master": roscore}, environment)

        unity = self._launch(
            spec,
            "unity",
            self._unity_argv(spec),
            environment,
            worker_dir / "unity.log",
        )
        bridge = self._launch(
            spec,
            "bridge",
            self._bridge_argv(spec),
            environment,
            worker_dir / "bridge.log",
        )
        self._wait_ready(
            "bridge",
            spec,
            {"ros_master": roscore, "unity": unity, "bridge": bridge},
            environment,
        )

    def _launch(
        self,
        spec: WorkerRuntimeSpec,
        role: str,
        argv: Sequence[str],
        environment: Mapping[str, str],
        log_path: Path,
    ) -> Any:
        stream = log_path.open("w", encoding="utf-8")
        try:
            process = self._process_factory(
                tuple(str(value) for value in argv),
                cwd=str(self._workspace),
                env=dict(environment),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            stream.close()
            raise
        finally:
            stream.close()
        self._owned_processes.append((int(spec.worker_id), str(role), process))
        self._assert_alive(process, spec, role)
        return process

    def _assert_alive(self, process: Any, spec: WorkerRuntimeSpec, role: str) -> None:
        poll = getattr(process, "poll", None)
        if callable(poll):
            status = poll()
            if status is not None:
                raise ManagedRuntimeError(
                    "{}: worker {} {} exited before readiness with code {}".format(
                        MANAGED_RUNTIME_STARTUP_ERROR,
                        spec.worker_id,
                        role,
                        status,
                    )
                )

    def _wait_ready(
        self,
        stage: str,
        spec: WorkerRuntimeSpec,
        process_handles: Mapping[str, Any],
        environment: Mapping[str, str],
    ) -> None:
        for role, process in process_handles.items():
            self._assert_alive(process, spec, role)
        result = self._readiness_checker(
            str(stage),
            spec,
            dict(process_handles),
            dict(environment),
            float(self._startup_timeout_s),
        )
        self._readiness.append(
            {
                "stage": str(stage),
                "worker_id": int(spec.worker_id),
                "result": result,
            }
        )
        for role, process in process_handles.items():
            self._assert_alive(process, spec, role)

    def _unity_argv(self, spec: WorkerRuntimeSpec) -> Tuple[str, ...]:
        args = [
            str(self._unity_bin),
            "-screen-width",
            str(self._window_width),
            "-screen-height",
            str(self._window_height),
            "-screen-fullscreen",
            "0",
            "-cmdSubPort",
            str(spec.command_port),
            "-statePubPort",
            str(spec.state_port),
            "-depthPubPort",
            str(spec.depth_port),
        ]
        if self._reliable_mode:
            args.extend(
                (
                    "-executionResultPort",
                    str(spec.unity_result_port),
                    "-observationSnapshotPort",
                    str(spec.unity_snapshot_port),
                    "-reliableCommandPort",
                    str(spec.unity_command_port),
                    "-primitiveResultSchema",
                    str(SCHEMA_VERSION),
                    "-runtimeInstanceId",
                    str(spec.runtime_instance_id),
                    "-endpointEpisodeId",
                    "episode-0",
                    "-endpointResetId",
                    "reset-0",
                )
            )
        args.extend(self._unity_extra_args)
        return tuple(args)

    def _bridge_argv(self, spec: WorkerRuntimeSpec) -> Tuple[str, ...]:
        values = spec.bridge_launch_args()
        args = [
            "roslaunch",
            "planning",
            "sim_realtime.launch",
            "cmd_port:={}".format(values["cmd_port"]),
            "state_port:={}".format(values["state_port"]),
            "depth_port:={}".format(values["depth_port"]),
            "python_command_port:={}".format(values["python_command_port"]),
            "unity_command_port:={}".format(values["unity_command_port"]),
            "command_runtime_instance_id:={}".format(
                values["command_runtime_instance_id"]
            ),
            "execution_result_port:={}".format(values["execution_result_port"]),
            "observation_snapshot_port:={}".format(
                values["observation_snapshot_port"]
            ),
            "python_result_port:={}".format(values["python_result_port"]),
            "python_snapshot_port:={}".format(values["python_snapshot_port"]),
            "python_result_bind_host:=127.0.0.1",
            "python_snapshot_bind_host:=127.0.0.1",
            "visualization:=false",
            "rviz:=false",
        ]
        args.extend(self._bridge_extra_args)
        return tuple(args)

    def _build_identity(self) -> Dict[str, Any]:
        assembly = _assembly_path(self._unity_bin)
        artifacts = {
            "unity_player_path": str(self._unity_bin),
            "unity_player_sha256": file_sha256(self._unity_bin),
            "runtime_assembly_path": str(assembly),
            "runtime_assembly_sha256": file_sha256(assembly),
            "bridge_path": str(self._bridge_bin),
            "bridge_sha256": file_sha256(self._bridge_bin),
        }
        workers = []
        for spec in self._worker_specs:
            workers.append(
                {
                    "worker_id": int(spec.worker_id),
                    "runtime_instance_id": str(spec.runtime_instance_id),
                    "runtime_launch_nonce": str(spec.runtime_launch_nonce),
                    "training_run_id": str(spec.training_run_id),
                    "ros_master_uri": str(spec.ros_master_uri),
                    "ros_home": str(spec.ros_home),
                    "namespace": "/",
                    "ports": spec.port_mapping(),
                    "unity_executable_identity": {
                        "path": artifacts["unity_player_path"],
                        "sha256": artifacts["unity_player_sha256"],
                    },
                    "bridge_executable_identity": {
                        "path": artifacts["bridge_path"],
                        "sha256": artifacts["bridge_sha256"],
                    },
                    "task_contract_sha256": self._task_contract_sha256,
                    "observation_contract": self._observation_contract,
                    "mpl_contract_sha256": self._mpl_contract_sha256,
                }
            )
        return {
            "schema": MANAGED_RUNTIME_LIFECYCLE_SCHEMA,
            "mode": "managed",
            "workspace": str(self._workspace),
            "artifacts": artifacts,
            "contracts": {
                "task_contract_sha256": self._task_contract_sha256,
                "observation_contract": self._observation_contract,
                "mpl_contract_sha256": self._mpl_contract_sha256,
                "max_steps": int(self._max_steps),
            },
            "reliable_v4": self._reliable_mode,
            "workers": workers,
            "readiness": list(self._readiness),
            "process_cleanup": "owned_process_groups_only",
        }

    def _default_readiness_checker(
        self,
        stage: str,
        spec: WorkerRuntimeSpec,
        process_handles: Mapping[str, Any],
        environment: Mapping[str, str],
        timeout_s: float,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            for role, process in process_handles.items():
                self._assert_alive(process, spec, role)
            if stage == "ros_master":
                ready = self._probe_ros_master(environment)
            elif stage == "bridge":
                ready = self._probe_bridge_topics(environment)
            else:
                raise ManagedRuntimeError(
                    "{}: unknown readiness stage {}".format(
                        MANAGED_RUNTIME_STARTUP_ERROR, stage
                    )
                )
            if ready:
                return {"stage": str(stage), "ready": True}
            self._sleep(0.25)
        raise ManagedRuntimeError(
            "{}: worker {} {} readiness timed out after {:.3f}s".format(
                MANAGED_RUNTIME_STARTUP_ERROR,
                spec.worker_id,
                stage,
                float(timeout_s),
            )
        )

    @staticmethod
    def _probe_ros_master(environment: Mapping[str, str]) -> bool:
        result = subprocess.run(
            ["rosparam", "list"],
            env=dict(environment),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    @staticmethod
    def _probe_bridge_topics(environment: Mapping[str, str]) -> bool:
        result = subprocess.run(
            ["rostopic", "list"],
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        topics = {line.strip() for line in str(result.stdout).splitlines()}
        return "/xm/state" in topics and "/xm/depth/image_raw" in topics


__all__ = [
    "MANAGED_RUNTIME_LIFECYCLE_SCHEMA",
    "MANAGED_RUNTIME_STARTUP_ERROR",
    "ManagedRuntimeError",
    "ManagedRuntimePool",
]
