"""Multi-worker runtime lifecycle orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from planning.runtime.collection_worker import RuntimeWorker
from planning.runtime.ports import (
    is_tcp_port_available,
    WorkerRuntimeSpec,
    validate_worker_runtime_specs,
)


class RuntimeSupervisor:
    def __init__(
        self,
        *,
        specs: Sequence[WorkerRuntimeSpec],
        out_dir: Path,
        unity_bin: Path,
        bridge_binary: Path | None = None,
        window_width: int,
        window_height: int,
        startup_timeout_s: float,
    ) -> None:
        self.specs = tuple(specs)
        # Validate the complete canonical profile before any child starts.
        validate_worker_runtime_specs(self.specs)
        self.workers = tuple(
            RuntimeWorker(
                spec=spec,
                out_dir=out_dir,
                unity_bin=unity_bin,
                bridge_binary=bridge_binary,
                window_width=window_width,
                window_height=window_height,
                startup_timeout_s=startup_timeout_s,
            )
            for spec in self.specs
        )
        self._stopped = False

    def validate_ports_available(self) -> None:
        for spec in self.specs:
            for port in spec.all_ports:
                if not is_tcp_port_available(port):
                    raise RuntimeError(
                        f"port already in use: {port} (worker {spec.worker_id}); "
                        "stop old Unity/ROS processes before starting collection"
                    )

    def start(self) -> None:
        self.validate_ports_available()
        try:
            for worker in self.workers:
                worker.start_ros_master()
            for worker in self.workers:
                worker.wait_master_ready()
            for worker in self.workers:
                worker.start_runtime()
            for worker in self.workers:
                worker.wait_runtime_ready()
        except Exception:
            self.stop()
            raise

    def active_collectors(self) -> int:
        return sum(worker.collector_alive() for worker in self.workers)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for worker in reversed(self.workers):
            worker.stop()

    def __enter__(self) -> "RuntimeSupervisor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
