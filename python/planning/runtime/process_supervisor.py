"""Out-of-process lifecycle observer for the P0-M2 diagnostic runner.

This helper deliberately observes processes only.  It never signals or kills a
registered process.  The runner writes registration/explicit-exit events to a
small append-only registry; this process writes observations directly to the
surviving process_lifecycle.jsonl artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict


def _write(stream, event: Dict[str, Any]) -> None:
    stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()


def _alive(pid: int) -> bool:
    proc_stat = Path("/proc") / str(pid) / "stat"
    try:
        fields = proc_stat.read_text(encoding="utf-8").split()
    except (FileNotFoundError, OSError):
        return False
    return len(fields) < 3 or fields[2] != "Z"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--observation-timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    registered: Dict[str, Dict[str, Any]] = {}
    explicit_exit = set()
    shutdown_requested = False
    started = time.monotonic()
    with args.output.open("a", encoding="utf-8") as output:
        _write(
            output,
            {
                "event": "SUPERVISOR_STARTED",
                "observer_pid": os.getpid(),
                "runner_pid": args.runner_pid,
                "monotonic_ns": time.monotonic_ns(),
            },
        )
        while True:
            try:
                with args.registry.open("r", encoding="utf-8") as registry:
                    registry.seek(offset)
                    while True:
                        line = registry.readline()
                        if not line:
                            break
                        offset = registry.tell()
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        event_name = event.get("event")
                        role = event.get("role")
                        if event_name == "SPAWN" and role:
                            registered[str(role)] = dict(event)
                            _write(
                                output,
                                {
                                    "event": "PROCESS_SPAWN_REGISTERED",
                                    "source": "supervisor",
                                    **event,
                                },
                            )
                        elif event_name == "EXIT" and role:
                            explicit_exit.add(str(role))
                            _write(
                                output,
                                {
                                    "event": "PROCESS_EXIT_REGISTERED",
                                    "source": "runner",
                                    **event,
                                },
                            )
                        elif event_name == "STOP":
                            shutdown_requested = True
                            _write(
                                output,
                                {
                                    "event": "SUPERVISOR_STOP_REQUESTED",
                                    "source": "runner",
                                    **event,
                                },
                            )
                        elif event_name in {"LIVENESS", "SIGNAL"}:
                            _write(output, {"source": "runner", **event})
            except FileNotFoundError:
                pass

            for role, process in list(registered.items()):
                if role in explicit_exit or process.get("pid") is None:
                    continue
                pid = int(process["pid"])
                if not _alive(pid):
                    _write(
                        output,
                        {
                            "event": "PROCESS_EXIT_OBSERVED",
                            "source": "supervisor",
                            "role": role,
                            "pid": pid,
                            "exit_type": "EXTERNAL_KILL_OR_DISAPPEAR",
                            "returncode": None,
                            "signal": None,
                            "cleanup_requested": False,
                            "observed_monotonic_ns": time.monotonic_ns(),
                        },
                    )
                    explicit_exit.add(role)

            runner_registered = registered.get("runner")
            runner_gone = bool(runner_registered) and not _alive(args.runner_pid)
            all_children_done = all(
                role == "runner" or role in explicit_exit
                for role in registered
            )
            if shutdown_requested and all_children_done:
                _write(
                    output,
                    {"event": "SUPERVISOR_STOPPED", "monotonic_ns": time.monotonic_ns()},
                )
                return 0
            if runner_gone and all_children_done:
                _write(
                    output,
                    {
                        "event": "RUNNER_DISAPPEARED",
                        "exit_type": "EXTERNAL_KILL_OR_DISAPPEAR",
                        "runner_pid": args.runner_pid,
                        "monotonic_ns": time.monotonic_ns(),
                    },
                )
                return 0
            if time.monotonic() - started >= args.observation_timeout_s:
                _write(
                    output,
                    {
                        "event": "SUPERVISOR_OBSERVATION_TIMEOUT",
                        "monotonic_ns": time.monotonic_ns(),
                    },
                )
                return 2
            time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
