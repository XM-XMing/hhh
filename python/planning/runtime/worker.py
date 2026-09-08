"""Persistence and CLI compatibility layer for the canonical port owner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from planning.common import write_json_atomic
from planning.runtime.ports import (
    STARTUP_WORKER_ENDPOINT_MISMATCH,
    WORKER_RUNTIME_SPEC_SCHEMA,
    WorkerEndpointMismatchError,
    WorkerRuntimeSpec,
    build_worker_runtime_specs,
    resolve_runtime_port_profile,
    validate_worker_runtime_specs,
)


def write_worker_runtime_spec_file(
    path: Path, specs: Sequence[WorkerRuntimeSpec]
) -> Path:
    normalized = validate_worker_runtime_specs(specs)
    destination = Path(path)
    payload = {
        "schema": WORKER_RUNTIME_SPEC_SCHEMA,
        "worker_count": len(normalized),
        "workers": [spec.to_mapping() for spec in normalized],
    }
    write_json_atomic(destination, payload, trailing_newline=True)
    return destination


def load_worker_runtime_spec_file(path: Path) -> Tuple[WorkerRuntimeSpec, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != WORKER_RUNTIME_SPEC_SCHEMA:
        raise WorkerEndpointMismatchError(
            "{}: worker spec schema mismatch".format(
                STARTUP_WORKER_ENDPOINT_MISMATCH
            )
        )
    workers = payload.get("workers")
    if not isinstance(workers, list):
        raise WorkerEndpointMismatchError(
            "{}: worker spec workers must be a list".format(
                STARTUP_WORKER_ENDPOINT_MISMATCH
            )
        )
    specs = tuple(
        resolve_runtime_port_profile(worker, mode="managed") for worker in workers
    )
    return validate_worker_runtime_specs(specs)


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-count", type=int, required=True)
    # Optional overrides are inputs to the canonical resolver.  Omitting them
    # deliberately selects the managed defaults in runtime.ports; the CLI
    # does not own a second set of production port defaults.
    parser.add_argument("--master-port-base", type=int, default=None)
    parser.add_argument("--command-port-base", type=int, default=None)
    parser.add_argument("--depth-port-base", type=int, default=None)
    parser.add_argument("--port-stride", type=int, default=None)
    parser.add_argument("--runtime-instance-template", required=True)
    parser.add_argument("--training-run-id", default="legacy-unspecified")
    parser.add_argument("--runtime-launch-nonce", default="legacy-unspecified")
    parser.add_argument("--ros-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    specs = build_worker_runtime_specs(
        worker_count=args.worker_count,
        master_port_base=args.master_port_base,
        command_port_base=args.command_port_base,
        depth_port_base=args.depth_port_base,
        port_stride=args.port_stride,
        runtime_instance_template=args.runtime_instance_template,
        ros_root=args.ros_root,
        training_run_id=args.training_run_id,
        runtime_launch_nonce=args.runtime_launch_nonce,
    )
    write_worker_runtime_spec_file(args.output, specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "STARTUP_WORKER_ENDPOINT_MISMATCH",
    "WORKER_RUNTIME_SPEC_SCHEMA",
    "WorkerEndpointMismatchError",
    "WorkerRuntimeSpec",
    "build_worker_runtime_specs",
    "load_worker_runtime_spec_file",
    "validate_worker_runtime_specs",
    "write_worker_runtime_spec_file",
]
