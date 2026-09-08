#!/usr/bin/env python3
"""Run the diagnostic observed-action oracle imitation audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.diagnostics.oracle_action_imitation import run_oracle_action_imitation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", type=Path, default=Path("data/awac/diagnostics/multi_action_replay_v1_real_20260907"))
    parser.add_argument("--feasibility", type=Path, default=Path("data/awac/diagnostics/offline_rl_feasibility_audit_v1"))
    parser.add_argument("--bc-checkpoint", type=Path, default=Path("data/teach/2026_6w/bc_training/checkpoint_best_soft.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/awac/diagnostics/oracle_action_imitation_feasibility_v1"))
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_oracle_action_imitation(
        dataset_root=args.dataset,
        feasibility_root=args.feasibility,
        bc_checkpoint=args.bc_checkpoint,
        out_dir=args.out_dir,
        seed=args.seed,
        fold_count=args.fold_count,
        bootstrap_repeats=args.bootstrap_repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    models = result["metrics"]["models"]
    best = max((models["B1_oracle_ce"], models["B2_oracle_advantage_weighted"]), key=lambda item: item["action_accuracy"])
    print("ORACLE_ACTION_IMITATION_FEASIBILITY_V1=PASS")
    print("STATE_COUNT={}".format(result["source"]["state_count"]))
    print("MISSION_COUNT={}".format(result["source"]["mission_count"]))
    print("B0_ACTION_ACCURACY={:.9f}".format(models["B0_original_bc"]["action_accuracy"]))
    print("B1_ACTION_ACCURACY={:.9f}".format(models["B1_oracle_ce"]["action_accuracy"]))
    print("B2_ACTION_ACCURACY={:.9f}".format(models["B2_oracle_advantage_weighted"]["action_accuracy"]))
    print("BEST_LEARNED_MODEL={}".format(best["model"]))
    print("BEST_LEARNED_RETURN_COVERAGE={:.9f}".format(best["observed_action_return_coverage"]))
    print("PRODUCTION_MODIFIED=NO")
    print("RL_TRAINING_EXECUTED=NO")
    print("OUT_DIR={}".format(result["out_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
