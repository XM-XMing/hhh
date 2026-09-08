#!/usr/bin/env python3
"""Run the read-only decision-state completeness audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.diagnostics.decision_state_completeness import run_decision_state_completeness_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--observation-audit-root",
        type=Path,
        default=Path("data/awac/diagnostics/observation_information_audit_v1"),
    )
    parser.add_argument(
        "--reward-audit-root",
        type=Path,
        default=Path("data/awac/diagnostics/reward_provenance_recovery_audit_v1"),
    )
    parser.add_argument(
        "--multi-action-root",
        type=Path,
        default=Path("data/awac/diagnostics/multi_action_replay_v1_real_20260907"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/awac/diagnostics/decision_state_completeness_audit_v1"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_decision_state_completeness_audit(
        observation_audit_root=args.observation_audit_root,
        reward_audit_root=args.reward_audit_root,
        multi_action_root=args.multi_action_root,
        out_dir=args.out_dir,
    )
    counts = result["counts"]
    print("DECISION_STATE_COMPLETENESS_AUDIT_V1=PASS")
    print("STATE_COUNT={}".format(counts["state_count"]))
    print("TRANSITION_COUNT={}".format(counts["transition_count"]))
    print("NON_TIE_PAIR_COUNT={}".format(counts["non_tie_pair_count"]))
    print("MULTI_OUTCOME_STATE_COUNT={}".format(counts["multi_outcome_state_count"]))
    print("PRODUCTION_MODIFIED=NO")
    print("OUT_DIR={}".format(result["out_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
