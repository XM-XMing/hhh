#!/usr/bin/env python3
"""Print an AWAC rolling-checkpoint GC plan without deleting anything."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planning.awac.storage import (
    DEFAULT_ROLLING_CHECKPOINT_RETENTION,
    GIB,
    checkpoint_retention_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan checkpoint_last retention; never deletes files."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--retention",
        type=int,
        default=DEFAULT_ROLLING_CHECKPOINT_RETENTION,
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    plan = checkpoint_retention_plan(
        Path(args.checkpoint_dir),
        keep_latest_committed=int(args.retention),
    )
    print(
        json.dumps(
            {
                "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
                "committed_generation": plan["committed_generation"],
                "retention": plan["retention"],
                "keep_files": plan["keep_files"],
                "preserved_uncommitted_files": plan["preserved_uncommitted_files"],
                "safe_delete_candidates": plan["delete_candidates"],
                "estimated_release_gib": float(
                    plan["delete_candidate_physical_bytes"]
                )
                / GIB,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

