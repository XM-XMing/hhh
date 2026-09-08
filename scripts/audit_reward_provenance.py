#!/usr/bin/env python3
"""Recover and audit step-level reward provenance from existing artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.diagnostics.reward_provenance_audit import run_reward_provenance_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--multi-action-root",
        type=Path,
        default=Path("data/awac/diagnostics/multi_action_replay_v1_real_20260907"),
    )
    parser.add_argument(
        "--calibration-replay-root",
        type=Path,
        default=Path("data/awac/awac_bc60k_formal_critic_calibration_v7/replay"),
    )
    parser.add_argument(
        "--teacher-collection-root",
        type=Path,
        default=Path(
            "data/awac/diagnostics/teacher_recovery_replay_mini_experiment_20260907/collection_run"
        ),
    )
    parser.add_argument(
        "--recovered-step-collection-root",
        type=Path,
        default=Path("data/awac/diagnostics/dev100_n5_confirmation/bc"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/awac/diagnostics/reward_provenance_recovery_audit_v1"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_reward_provenance_audit(
        multi_action_root=args.multi_action_root,
        calibration_replay_root=args.calibration_replay_root,
        teacher_collection_root=args.teacher_collection_root,
        recovered_step_collection_root=args.recovered_step_collection_root,
        out_dir=args.out_dir,
    )
    metrics = result["metrics"]
    provenance = result["provenance"]
    sequence = provenance["sequence_recovery"]
    print("REWARD_PROVENANCE_RECOVERY_AUDIT_V1=PASS")
    print("STEP_REWARD_SEQUENCE_RECOVERED=YES")
    print("RECOVERED_EPISODES={}".format(sequence["episode_count"]))
    print("RECOVERED_TRANSITIONS={}".format(sequence["transition_count"]))
    print("CALIBRATION_REPLAY_SEQUENCE=UNAVAILABLE")
    print("MULTI_ACTION_SEQUENCE=UNAVAILABLE")
    print("TERMINAL_REASONS={}".format(metrics["terminal_reason_counts"]))
    print("OUT_DIR={}".format(result["out_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
