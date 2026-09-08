#!/usr/bin/env python3
"""Run the read-only high-level action abstraction audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.diagnostics.high_level_action_abstraction import run_high_level_action_abstraction_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", type=Path, default=Path("data/awac/diagnostics/multi_action_replay_v1_real_20260907"))
    parser.add_argument("--hierarchical-root", type=Path, default=Path("data/awac/diagnostics/hierarchical_task_formulation_audit_v1"))
    parser.add_argument("--mission-split", type=Path, default=Path("data/awac/diagnostics/action_formulation_audit_v1/mission_split.json"))
    parser.add_argument("--motion-primitives-json", type=Path, default=Path("data/motion_primitives/motion_primitives_105.json"))
    parser.add_argument("--motion-primitives-npz", type=Path, default=Path("data/motion_primitives/motion_primitives_105.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/awac/diagnostics/high_level_action_abstraction_audit_v1"))
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_high_level_action_abstraction_audit(
        dataset_root=args.dataset,
        hierarchical_root=args.hierarchical_root,
        mission_split_path=args.mission_split,
        motion_primitives_json=args.motion_primitives_json,
        motion_primitives_npz=args.motion_primitives_npz,
        out_dir=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    ranking = result["ranking"]["models"]
    high = {name: ranking[name]["oof_metrics"]["micro"]["accuracy"] for name in ranking if name != "original_105"}
    best_name = max(high, key=high.get) if high else "UNKNOWN"
    print("HIGH_LEVEL_ACTION_ABSTRACTION_AUDIT_V1=PASS")
    print("DATASET_STATES={}".format(result["abstraction"]["dataset_identity"]["state_count"]))
    print("DATASET_TRANSITIONS={}".format(result["abstraction"]["dataset_identity"]["transition_count"]))
    print("PAIR_ROWS_ALL={}".format(result["ranking"]["pair_count_all"]))
    print("PAIR_ROWS_NON_TIE={}".format(result["ranking"]["pair_count_non_tie"]))
    print("BEST_HIGH_LEVEL_CANDIDATE={}".format(best_name))
    print("BEST_HIGH_LEVEL_MICRO_ACCURACY={:.9f}".format(high.get(best_name, float("nan"))))
    print("PRODUCTION_MODIFIED=NO")
    print("RL_TRAINING_EXECUTED=NO")
    print("OUT_DIR={}".format(result["out_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

