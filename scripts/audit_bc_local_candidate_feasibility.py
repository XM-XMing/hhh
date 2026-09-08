#!/usr/bin/env python3
"""Run the read-only BC-local candidate feasibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planning.diagnostics.bc_local_candidate_feasibility import run_bc_local_candidate_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit BC-local primitive candidate coverage without RL training.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--action-audit", type=Path, required=True)
    parser.add_argument("--motion-primitives-json", type=Path, required=True)
    parser.add_argument("--motion-primitives-npz", type=Path, required=True)
    parser.add_argument("--motion-primitives-config", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--random-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_bc_local_candidate_audit(
        dataset_root=args.dataset,
        oracle_root=args.oracle,
        action_root=args.action_audit,
        motion_primitives_json=args.motion_primitives_json,
        motion_primitives_npz=args.motion_primitives_npz,
        motion_primitives_config=args.motion_primitives_config,
        bc_checkpoint=args.bc_checkpoint,
        out_dir=args.out_dir,
        seed=args.seed,
        random_repeats=args.random_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    oracle = result["oracle_coverage"]["families"]
    gains = result["gain_retention"]["families"]
    random_control = result["random_control"]["families"]
    print("BC_LOCAL_CANDIDATE_RL_FEASIBILITY_AUDIT_V1=PASS")
    print("STATE_COUNT={}".format(result["input_identity"]["state_count"]))
    print("VALID_ACTION_BRANCHING_MEAN={:.9f}".format(result["valid_action_branching"]["mean"]))
    for family in ("K0_BC_ONLY", "K2_LOCAL", "K4_LOCAL", "K8_LOCAL", "K16_LOCAL"):
        print("{}_ORACLE_COVERAGE={:.9f}".format(family, oracle[family]["mission_macro_coverage"]))
        print("{}_GAIN_RETENTION={}".format(family, gains[family]["aggregate_gain_retention_ratio"]))
        print("{}_CANDIDATE_SIZE_MEAN={:.9f}".format(family, result["candidate_sizes"][family]["size"]["mean"]))
    for family in ("K4_LOCAL", "K8_LOCAL", "K16_LOCAL"):
        print("LOCAL_MINUS_RANDOM_{}={:.9f}".format(family.split("_")[0], random_control[family]["local_minus_random"]))
    print("RECOMMENDED_K={}".format(result["decision"]["recommended_k"]))
    print("NEXT_DECISION={}".format(result["decision"]["next_decision"]))
    print("ACTOR_OPTIMIZER_STEPS=0")
    print("CRITIC_OPTIMIZER_STEPS=0")
    print("NEW_ENVIRONMENT_STEPS=0")
    print("OUTPUT_DIR={}".format(Path(args.out_dir).expanduser().resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
