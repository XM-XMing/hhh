#!/usr/bin/env python3
"""Run the read-only offline RL feasibility upper-bound audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.diagnostics.offline_rl_feasibility import run_offline_rl_feasibility_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", type=Path, default=Path("data/awac/diagnostics/multi_action_replay_v1_real_20260907"))
    parser.add_argument("--bc-dev-root", type=Path, default=Path("data/awac/diagnostics/dev100_n5_confirmation/bc"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/awac/diagnostics/offline_rl_feasibility_audit_v1"))
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_offline_rl_feasibility_audit(
        dataset_root=args.dataset,
        bc_dev_root=args.bc_dev_root,
        out_dir=args.out_dir,
        seed=args.seed,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    payload = result["payload"]
    print("OFFLINE_RL_FEASIBILITY_AUDIT_V1=PASS")
    print("STATE_COUNT={}".format(payload["counts"]["state_count"]))
    print("ORACLE_RETURN_DELTA_MEAN={:.9f}".format(payload["oracle_upper_bound"]["return_delta_oracle_minus_bc"]["mean"]))
    print("ORACLE_SUCCESS_DELTA_MEAN={:.9f}".format(payload["oracle_upper_bound"]["success_delta_oracle_minus_bc"]["mean"]))
    print("RETURN_DELTA_CI95={}".format(payload["mission_bootstrap"]["return_delta"]["ci95"]))
    print("CONCLUSION={}".format(payload["conclusion"]["result"]))
    print("PRODUCTION_MODIFIED=NO")
    print("RL_TRAINING_EXECUTED=NO")
    print("OUT_DIR={}".format(result["out_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

