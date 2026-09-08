#!/usr/bin/env python3
"""Run the offline reward horizon / credit assignment audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.diagnostics.reward_horizon_audit import run_reward_horizon_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--observation-audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_reward_horizon_audit(
        dataset_root=args.dataset,
        state_audit_root=args.state_audit,
        observation_audit_root=args.observation_audit,
        out_dir=args.out_dir,
        seed=int(args.seed),
        bootstrap_repeats=int(args.bootstrap_repeats),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
    )
    identity = result["identity"]
    print("REWARD_HORIZON_CREDIT_ASSIGNMENT_AUDIT_V1=PASS")
    print("DATASET_ID={}".format(identity["dataset_id"]))
    print("BRANCH_COUNT={}".format(identity["branch_count"]))
    print("STATE_COUNT={}".format(identity["state_count"]))
    print("MISSION_COUNT={}".format(identity["mission_count"]))
    print("EXACT_G_T_AVAILABLE={}".format(result["availability"]["exact_future_return_available"]))
    print("OUT_DIR={}".format(result["out_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
