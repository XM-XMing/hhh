#!/usr/bin/env python3
"""Canonical formal mission-preparation entry point with run logging."""

from planning.mission.preparation import main
from planning.mission.preparation import build_argument_parser
from planning.common.logging import (
    mirror_command_output,
    preparation_log_path,
    prepare_log_directories,
)
from planning.common.paths import planning_package_root

import sys
from pathlib import Path


def _resolved_candidate_index(values):
    parsed, _ = build_argument_parser().parse_known_args(values)
    path = Path(parsed.candidate_index).expanduser()
    if not path.is_absolute():
        path = planning_package_root() / path
    return path.resolve(), int(parsed.workers)


def cli_main(argv=None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    # Help is a read-only parser action and should not create a run directory.
    if "--help" in values or "-h" in values:
        return main()
    candidate_index, workers = _resolved_candidate_index(values)
    run_dir = candidate_index.parent
    prepare_log_directories(run_dir, worker_count=workers)
    with mirror_command_output(
        preparation_log_path(run_dir),
        run_id=run_dir.name,
        command="scripts/prepare_teacher_missions.py",
    ):
        return main()


if __name__ == "__main__":
    raise SystemExit(cli_main())
