"""One ROS + Unity + Bridge runtime worker for teacher collection."""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from planning.runtime.health import wait_for_ros_master
from planning.runtime.ports import WorkerRuntimeSpec
from planning.runtime.process import ManagedProcess
from planning.common.paths import planning_package_root
from planning.data.rollout_shard import worker_shard_dir
from planning.protocol.constants import PROTOCOL_VERSION
from planning.common.logging import (
    append_log_event,
    bridge_log_path,
    roscore_log_path,
    unity_log_path,
    worker_log_path,
)


class RuntimeWorker:
    def __init__(
        self,
        *,
        spec: WorkerRuntimeSpec,
        out_dir: Path,
        unity_bin: Path,
        window_width: int,
        window_height: int,
        startup_timeout_s: float,
        bridge_binary: Optional[Path] = None,
    ) -> None:
        self.spec = spec
        self.out_dir = Path(out_dir)
        self.unity_bin = Path(unity_bin)
        self.window_width = int(window_width)
        self.window_height = int(window_height)
        self.startup_timeout_s = float(startup_timeout_s)
        self.bridge_binary = Path(bridge_binary).resolve() if bridge_binary is not None else None
        self.roscore: Optional[ManagedProcess] = None
        self.unity: Optional[ManagedProcess] = None
        self.bridge: Optional[ManagedProcess] = None
        self.collector: Optional[ManagedProcess] = None

    @property
    def logs_dir(self) -> Path:
        return self.out_dir / "logs"

    @property
    def worker_dir(self) -> Path:
        return worker_shard_dir(self.out_dir / "workers", self.spec.worker_id)

    @property
    def worker_log_path(self) -> Path:
        return worker_log_path(self.out_dir, self.spec.worker_id)

    @property
    def unity_log_path(self) -> Path:
        return unity_log_path(self.out_dir, self.spec.worker_id)

    @property
    def bridge_log_path(self) -> Path:
        return bridge_log_path(self.out_dir, self.spec.worker_id)

    @property
    def roscore_log_path(self) -> Path:
        return roscore_log_path(self.out_dir, self.spec.worker_id)

    @property
    def unity_session_log_path(self) -> Path:
        """Choose a new Unity file on resume so Unity cannot truncate history."""

        base = self.unity_log_path
        if not base.exists() or base.stat().st_size == 0:
            return base
        index = 1
        while True:
            candidate = base.with_name(
                "{}_resume_{:02d}{}".format(base.stem, index, base.suffix)
            )
            if not candidate.exists():
                return candidate
            index += 1

    def _event(self, message: str, *, level: str = "INFO", **fields) -> None:
        append_log_event(self.worker_log_path, message, level=level, **fields)

    def ros_environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "ROS_MASTER_URI": self.spec.ros_master_uri,
                "ROS_HOSTNAME": "127.0.0.1",
                "ROS_HOME": str(self.spec.ros_home),
            }
        )
        return env

    def start_ros_master(self) -> None:
        Path(self.spec.ros_home).mkdir(parents=True, exist_ok=True)
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["ROS_HOME"] = str(self.spec.ros_home)
        self.roscore = ManagedProcess.start(
            name=f"roscore-{self.spec.worker_id}",
            argv=("roscore", "-p", str(self.spec.master_port)),
            env=env,
            log_path=self.roscore_log_path,
            append_log=True,
        )
        self._event(
            "ROSCORE STARTED",
            pid=self.roscore.pid,
            worker_id=self.spec.worker_id,
            master_port=self.spec.master_port,
            log_path=str(self.roscore_log_path),
        )

    def wait_master_ready(self) -> None:
        wait_for_ros_master(env=self.ros_environment(), timeout_s=self.startup_timeout_s)

    def start_runtime(self) -> None:
        worker = self.spec.worker_id
        bridge_args = self.spec.bridge_launch_args()
        unity_log = self.unity_session_log_path
        self.unity = ManagedProcess.start(
            name=f"unity-{worker}",
            argv=(
                str(self.unity_bin),
                "-screen-width",
                str(self.window_width),
                "-screen-height",
                str(self.window_height),
                "-screen-fullscreen",
                "0",
                *self.spec.unity_launch_argv(),
                "-primitiveResultSchema",
                str(PROTOCOL_VERSION),
                "-logFile",
                str(unity_log),
            ),
            log_path=unity_log.with_name(unity_log.stem + ".launcher.log"),
            append_log=True,
        )
        self.bridge = ManagedProcess.start(
            name=f"bridge-{worker}",
            argv=(
                "roslaunch",
                str(planning_package_root() / "launch" / "sim_realtime.launch"),
                *("{}:={}".format(name, value) for name, value in bridge_args.items()),
                "visualization:=false",
                "rviz:=false",
            ),
            env=self.ros_environment(),
            log_path=self.bridge_log_path,
            append_log=True,
        )
        self._event(
            "RUNTIME STARTED",
            worker_id=worker,
            unity_pid=self.unity.pid,
            bridge_pid=self.bridge.pid,
            unity_log=str(unity_log),
            bridge_log=str(self.bridge_log_path),
            bridge_binary=str(self.bridge_binary or "roslaunch:planning/unity_bridge_node"),
            ports=self.spec.port_mapping(),
            runtime_instance_id=self.spec.runtime_instance_id,
        )

    def wait_runtime_ready(self) -> None:
        # Formal collection readiness is proven by the reliable command/result
        # and snapshot transactions.  Requiring latest ROS telemetry here
        # would reintroduce the forbidden observation path.
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if (
                self.unity is not None
                and self.bridge is not None
                and self.unity.is_alive()
                and self.bridge.is_alive()
            ):
                return
            time.sleep(0.25)
        raise TimeoutError(
            "managed reliable runtime did not remain alive for worker {}".format(
                self.spec.worker_id
            )
        )

    def start_collector(
        self,
        *,
        argv: Sequence[str],
        environment: Mapping[str, str],
    ) -> None:
        env = self.ros_environment()
        env.update({str(key): str(value) for key, value in environment.items()})
        self.collector = ManagedProcess.start(
            name=f"collector-{self.spec.worker_id}",
            argv=argv,
            env=env,
            log_path=self.worker_log_path,
            append_log=True,
        )
        self._event(
            "COLLECTOR STARTED",
            worker_id=self.spec.worker_id,
            collector_pid=self.collector.pid,
            mission_index=_argv_value(argv, "--index"),
            worker_shard=self.spec.worker_id,
            worker_count=_argv_value(argv, "--num-workers"),
            log_path=str(self.worker_log_path),
        )

    def collector_alive(self) -> bool:
        return self.collector is not None and self.collector.is_alive()

    def wait_collector(self) -> int:
        if self.collector is None:
            return 0
        return_code = self.collector.wait()
        progress = {}
        progress_path = self.worker_dir / "collection_progress.json"
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
            progress = {}
        self._event(
            "COLLECTOR EXITED",
            worker_id=self.spec.worker_id,
            exit_code=return_code,
            attempted=int(progress.get("attempted_total", 0) or 0),
            accepted=int(progress.get("accepted_total", 0) or 0),
            reliable_rows=int(progress.get("reliable_rows", 0) or 0),
            legacy_rows=int(progress.get("legacy_rows", 0) or 0),
            telemetry_lookup_count=int(progress.get("telemetry_lookup_count", 0) or 0),
            snapshot_missing_count=int(progress.get("snapshot_missing_count", 0) or 0),
        )
        return return_code

    def stop(self) -> None:
        self._event(
            "CLEANUP START",
            worker_id=self.spec.worker_id,
            collector_pid=_process_pid(self.collector),
            bridge_pid=_process_pid(self.bridge),
            unity_pid=_process_pid(self.unity),
            roscore_pid=_process_pid(self.roscore),
        )
        processes = (self.collector, self.bridge, self.unity, self.roscore)
        for process in processes:
            if process is not None:
                process.terminate()
        for process in processes:
            if process is None:
                continue
            try:
                process.wait(timeout=2.0)
            except Exception:
                process.kill()
        for process in processes:
            if process is None:
                continue
            try:
                process.wait(timeout=2.0)
            except Exception:
                pass
        self._event("CLEANUP COMPLETE", worker_id=self.spec.worker_id)


def _process_pid(process: Optional[ManagedProcess]) -> Optional[int]:
    if process is None:
        return None
    try:
        return int(process.pid)
    except (AttributeError, TypeError, ValueError):
        return None


def _argv_value(argv: Sequence[str], option: str) -> str:
    values = [str(value) for value in argv]
    try:
        return values[values.index(option) + 1]
    except (ValueError, IndexError):
        return ""
