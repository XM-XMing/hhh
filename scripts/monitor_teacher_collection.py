#!/usr/bin/env python3
"""Monitor a formal Teacher collection without mutating its artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from planning.teacher.collection_monitor import run_collection_monitor


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor formal Teacher collection progress."
    )
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--interval-sec", type=float, default=10.0)
    parser.add_argument(
        "--once",
        action="store_true",
        help="print one bounded snapshot (for tests and diagnostics)",
    )
    return parser


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        return run_collection_monitor(
            Path(args.rollout_dir),
            interval_sec=float(args.interval_sec),
            once=bool(args.once),
        )
    except KeyboardInterrupt:
        print("MONITOR_STOP=CTRL_C")
        return 0
    except Exception as error:
        print("COLLECTION_MONITOR=FAIL")
        print("ERROR={}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
