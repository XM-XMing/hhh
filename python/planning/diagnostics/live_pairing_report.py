#!/usr/bin/env python3
"""Diagnose live-selected state/depth reachability across isolated evaluators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from planning.diagnostics.live_pairing_reachability import summarize_live_reachability


def _load_trace(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = list(payload.get("episodes", []))
    if len(episodes) != 1:
        raise ValueError("live reachability trace must contain exactly one episode: {}".format(path))
    trace = dict(episodes[0])
    trace["repeat"] = path.parent.name
    return trace


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--min-repeats", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    paths = sorted(args.root.glob("repeat_*/first_divergence_trace.json"))
    if len(paths) < int(args.min_repeats):
        raise ValueError(
            "live reachability requires at least {} repeats, found {}".format(
                args.min_repeats, len(paths)
            )
        )
    summary = summarize_live_reachability([_load_trace(path) for path in paths])
    report_path = args.root / "live_pairing_reachability_report.json"
    report_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        reset = summary["reset"]
        print(
            "LIVE_PAIRING_REACHABILITY classification={} repeats={} "
            "reset_pairs={} reset_depths={} reset_logits={} reset_actions={}".format(
                summary["classification"],
                summary["repeat_count"],
                reset["unique_selected_pair_metadata_count"],
                reset["unique_depth_fingerprint_count"],
                reset["unique_logits_fingerprint_count"],
                reset["unique_top1_action_count"],
            )
        )
        print(
            "divergence content={} mask={} action={} terminal={} outcome={}".format(
                summary["any_live_content_divergence"],
                summary["any_live_mask_divergence"],
                summary["any_live_action_divergence"],
                summary["any_live_terminal_reward_divergence"],
                summary["any_live_outcome_divergence"],
            )
        )
        print("report={}".format(report_path))
        print("RESULT={}".format("RED" if summary["classification"] == "T1" else "NO_L1_DIVERGENCE"))
    return 1 if summary["classification"] == "T1" else 0


if __name__ == "__main__":
    raise SystemExit(main())
