#!/usr/bin/env python3
"""Run the read-only hierarchical task-formulation audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.diagnostics.hierarchical_task_formulation import run_hierarchical_task_formulation_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=Path("data/awac/diagnostics/dev100_n5_confirmation/bc"),
    )
    parser.add_argument(
        "--motion-primitives-json",
        type=Path,
        default=Path("data/motion_primitives/motion_primitives_105.json"),
    )
    parser.add_argument(
        "--motion-primitives-npz",
        type=Path,
        default=Path("data/motion_primitives/motion_primitives_105.npz"),
    )
    parser.add_argument(
        "--action-formulation-root",
        type=Path,
        default=Path("data/awac/diagnostics/action_formulation_audit_v1"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/awac/diagnostics/hierarchical_task_formulation_audit_v1"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_hierarchical_task_formulation_audit(
        rollout_root=args.rollout_root,
        motion_primitives_json=args.motion_primitives_json,
        motion_primitives_npz=args.motion_primitives_npz,
        action_formulation_root=args.action_formulation_root,
        out_dir=args.out_dir,
    )
    counts = result["trajectory_pattern"]["outcome_counts"]
    failure = result["failure_mode_analysis"]
    print("HIERARCHICAL_TASK_FORMULATION_AUDIT_V1=PASS")
    print("ACTION_COUNT={}".format(result["action_clusters"]["summary"]["action_count"]))
    print("EPISODE_COUNT={}".format(result["trajectory_pattern"]["episode_count"]))
    print("SUCCESS_COUNT={}".format(counts.get("success", 0)))
    print("DEAD_END_COUNT={}".format(counts.get("dead_end", 0)))
    print("COLLISION_COUNT={}".format(counts.get("collision", 0)))
    print("HIGH_LEVEL_DECISION_EVIDENCE={}".format(failure["high_level_decision_error"]["status"]))
    print("LOW_LEVEL_CONTROL_EVIDENCE={}".format(failure["low_level_control_error"]["status"]))
    print("PRODUCTION_MODIFIED=NO")
    print("RL_TRAINING_EXECUTED=NO")
    print("OUT_DIR={}".format(result["out_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

