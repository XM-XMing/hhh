#!/usr/bin/env python3
"""Run the read-only observation augmentation ranking audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.diagnostics.observation_augmentation_ranking import (
    run_observation_augmentation_ranking_audit,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/awac/diagnostics/multi_action_replay_v1_real_20260907"
DEFAULT_OBSERVATION_AUDIT = ROOT / "data/awac/diagnostics/observation_information_audit_v1"
DEFAULT_OUTPUT = ROOT / "data/awac/diagnostics/observation_augmentation_ranking_audit_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit S0-S4 non-privileged observation ranking variants")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--observation-audit-root", type=Path, default=DEFAULT_OBSERVATION_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=202609071)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=45)
    args = parser.parse_args()
    result = run_observation_augmentation_ranking_audit(
        args.dataset_root,
        args.out_dir,
        observation_audit_root=args.observation_audit_root,
        seed=args.seed,
        bootstrap_repeats=args.bootstrap_repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )
    print("OBSERVATION_AUGMENTATION_RANKING_AUDIT_V1=PASS")
    print("STATE_COUNT={}".format(result["identity"]["state_count"]))
    print("NON_TIE_PAIR_COUNT={}".format(result["identity"]["non_tie_pair_count"]))
    print("MISSION_COUNT={}".format(result["split"]["mission_count"]))
    print("FOLDS={}".format(result["split"]["fold_count"]))
    print("BOOTSTRAP_REPEATS={}".format(result["bootstrap"]["repeats"]))
    print("PRODUCTION_MODIFIED=NO")
    print("OUT_DIR={}".format(result["output_root"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
