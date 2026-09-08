#!/usr/bin/env python3
"""Run one command under the canonical managed ROS/Unity/Bridge owner.

This is a thin process-boundary entrypoint.  It owns no evaluation or
training logic; :class:`planning.runtime.managed_runtime.ManagedRuntimePool`
owns external startup, readiness, identity, and cleanup.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_managed_runtime.py",
        allow_abbrev=False,
        description=(
            "Start canonical per-worker ROS/Unity/Bridge runtimes and run "
            "one command inside an isolated worker environment."
        ),
    )
    parser.add_argument("--worker-spec-file", required=True, type=Path)
    parser.add_argument("--worker-count", required=True, type=int)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--unity-bin", type=Path, default=None)
    parser.add_argument("--bridge-bin", type=Path, default=None)
    parser.add_argument("--task-contract-sha256", default="")
    parser.add_argument("--mpl-contract-sha256", default="")
    parser.add_argument(
        "--observation-contract",
        default="reliable_exact_endpoint_snapshot",
    )
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--window-width", type=int, default=320)
    parser.add_argument("--window-height", type=int, default=240)
    parser.add_argument("--reliable-v4", action="store_true")
    parser.add_argument(
        "--unity-arg",
        action="append",
        default=[],
        help="one additional Unity argument; may be repeated",
    )
    parser.add_argument(
        "--bridge-arg",
        action="append",
        default=[],
        help="one additional roslaunch argument; may be repeated",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run after --",
    )
    return parser


def _command_after_separator(values: Sequence[str]) -> tuple[str, ...]:
    command = tuple(str(value) for value in values)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("run_managed_runtime.py requires a command after --")
    return command


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = _command_after_separator(args.command)
    if int(args.worker_count) <= 0:
        raise ValueError("worker-count must be positive")
    if not 0 <= int(args.worker_id) < int(args.worker_count):
        raise ValueError("worker-id must be within worker-count")

    from planning.common.paths import planning_workspace_root
    from planning.runtime.managed_runtime import ManagedRuntimePool
    from planning.runtime.ports import validate_worker_runtime_specs
    from planning.runtime.worker import load_worker_runtime_spec_file

    all_specs = load_worker_runtime_spec_file(
        Path(args.worker_spec_file).expanduser().resolve()
    )
    if int(args.worker_count) > len(all_specs):
        raise ValueError(
            "worker spec file contains {} workers, requested {}".format(
                len(all_specs), int(args.worker_count)
            )
        )
    specs = validate_worker_runtime_specs(
        tuple(all_specs[: int(args.worker_count)])
    )
    workspace = (
        Path(args.workspace).expanduser().resolve()
        if args.workspace is not None
        else planning_workspace_root()
    )
    runtime = ManagedRuntimePool(
        specs,
        workspace=workspace,
        log_dir=Path(args.log_dir).expanduser().resolve(),
        unity_bin=args.unity_bin,
        bridge_bin=args.bridge_bin,
        task_contract_sha256=str(args.task_contract_sha256),
        mpl_contract_sha256=str(args.mpl_contract_sha256),
        observation_contract=str(args.observation_contract),
        max_steps=int(args.max_steps),
        startup_timeout_s=float(args.startup_timeout),
        window_width=int(args.window_width),
        window_height=int(args.window_height),
        reliable_v4=bool(args.reliable_v4),
        unity_extra_args=tuple(str(value) for value in args.unity_arg),
        bridge_extra_args=tuple(str(value) for value in args.bridge_arg),
    )
    try:
        runtime.start()
        return runtime.run_command(command, worker_id=int(args.worker_id))
    except KeyboardInterrupt:
        return 130
    finally:
        runtime.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

