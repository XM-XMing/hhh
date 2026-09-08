#!/usr/bin/env python3
"""Audit a diagnostic multi-action replay artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from planning.diagnostics.multi_action_replay import audit_multi_action_replay


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--min-states", type=int, default=500)
    parser.add_argument("--min-transitions", type=int, default=2500)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_multi_action_replay(
        args.replay_dir,
        min_states=int(args.min_states),
        min_transitions=int(args.min_transitions),
    )
    Path(args.output).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).expanduser().resolve().write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print("MULTI_ACTION_REPLAY_AUDIT={}".format(report["status"]))
    print("STATE_COUNT={}".format(report["state_count"]))
    print("TRANSITION_COUNT={}".format(report["transition_count"]))
    print("MEAN_ACTIONS_PER_STATE={}".format(report["mean_actions_per_state"]))
    print("UNIQUE_ACTION_RATIO={}".format(report["unique_action_ratio"]))
    print("SOURCE_COUNTS={}".format(json.dumps(report["source_counts"], sort_keys=True)))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
