#!/usr/bin/env python3
"""Run the diagnostic historical-memory state-sufficiency audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.diagnostics.memory_state_sufficiency import run_memory_state_sufficiency_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", type=Path, default=Path("data/awac/diagnostics/multi_action_replay_v1_real_20260907"))
    parser.add_argument("--oracle", type=Path, default=Path("data/awac/diagnostics/oracle_action_imitation_feasibility_v1"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/awac/diagnostics/memory_state_sufficiency_audit_v1"))
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--action-batch-size", type=int, default=64)
    parser.add_argument("--pair-batch-size", type=int, default=128)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_memory_state_sufficiency_audit(
        dataset_root=args.dataset,
        oracle_root=args.oracle,
        out_dir=args.out_dir,
        seed=args.seed,
        fold_count=args.fold_count,
        bootstrap_repeats=args.bootstrap_repeats,
        epochs=args.epochs,
        action_batch_size=args.action_batch_size,
        pair_batch_size=args.pair_batch_size,
    )
    best_name = max(result["pair_metrics"]["models"], key=lambda name: result["pair_metrics"]["models"][name]["accuracy"])
    print("MEMORY_STATE_SUFFICIENCY_AUDIT_V1=PASS")
    print("STATE_COUNT={}".format(result["source"]["state_count"]))
    print("MISSION_COUNT={}".format(result["source"]["mission_count"]))
    print("BEST_MODEL={}".format(best_name))
    print("BEST_PAIRWISE_ACCURACY={:.9f}".format(result["pair_metrics"]["models"][best_name]["accuracy"]))
    print("BEST_PAIRWISE_AUC={}".format(result["pair_metrics"]["models"][best_name]["auc"]))
    print("PRODUCTION_MODIFIED=NO")
    print("RL_TRAINING_EXECUTED=NO")
    print("OUT_DIR={}".format(result["out_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
