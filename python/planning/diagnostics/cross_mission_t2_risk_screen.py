"""Aggregate completed isolated Layer-2 risk-screen mission repeats."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from planning.diagnostics.cross_mission_t2_risk import (
    summarize_cross_mission_screen,
    summarize_mission_repeats,
)


def _load_repeat(path: Path):
    trace_payload = json.loads((path / "first_divergence_trace.json").read_text(encoding="utf-8"))
    episodes = list(trace_payload.get("episodes", []))
    if len(episodes) != 1:
        raise ValueError("repeat must contain exactly one trace episode: {}".format(path))
    with (path / "rollout_index.csv").open(encoding="utf-8", newline="") as stream:
        rollouts = list(csv.DictReader(stream))
    if len(rollouts) != 1:
        raise ValueError("repeat must contain exactly one rollout row: {}".format(path))
    return {"repeat": path.name, "trace": episodes[0], "rollout": rollouts[0]}


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args(argv)

    selection = json.loads(args.selection.read_text(encoding="utf-8"))["selection"]
    completed = []
    for mission in selection:
        episode_id = int(mission["episode_id"])
        mission_root = args.root / "missions" / "episode_{:06d}".format(episode_id)
        repeat_paths = sorted(mission_root.glob("repeat_*/first_divergence_trace.json"))
        if not repeat_paths:
            continue
        if len(repeat_paths) != int(args.repeats):
            raise ValueError(
                "episode {} has {} completed repeats, expected {}".format(
                    episode_id, len(repeat_paths), args.repeats
                )
            )
        completed.append(
            summarize_mission_repeats(
                mission=mission,
                repeats=[_load_repeat(path.parent) for path in repeat_paths],
            )
        )
    if bool(args.require_all) and len(completed) != len(selection):
        raise ValueError("only {} of {} selected missions are complete".format(len(completed), len(selection)))
    report = summarize_cross_mission_screen(completed)
    report_path = args.root / "cross_mission_t2_risk_report.json"
    args.root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    classes = report["mission_classification_counts"]
    print(
        "CROSS_MISSION_T2 classification={} missions={} M0={} M1={} M2={} M3={} M4={}".format(
            report["classification"], report["missions"], classes["M0"], classes["M1"],
            classes["M2"], classes["M3"], classes["M4"]
        )
    )
    print("report={}".format(report_path))
    print("RESULT={}".format("RED" if report["classification"] == "T1" else "NO_T1_DIVERGENCE"))
    return 1 if report["classification"] == "T1" else 0


if __name__ == "__main__":
    raise SystemExit(main())
