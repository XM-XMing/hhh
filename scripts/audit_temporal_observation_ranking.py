#!/usr/bin/env python3
"""Run the causal S0/S1/S2/S3 temporal observation ranking audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planning.diagnostics.temporal_observation_ranking import (
    run_temporal_observation_ranking_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit whether short causal observation history improves action ranking."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--observation-audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_temporal_observation_ranking_audit(
        args.dataset,
        args.out_dir,
        state_audit_root=args.state_audit,
        observation_audit_root=args.observation_audit,
        seed=args.seed,
        fold_count=args.folds,
        bootstrap_repeats=args.bootstrap_repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    best_variant = max(
        result["variant_metrics"]["variants"],
        key=lambda value: result["variant_metrics"]["variants"][value]["mission_macro_accuracy"],
    )
    print("TEMPORAL_OBSERVATION_RANKING_AUDIT=PASS")
    print("BEST_VARIANT={}".format(best_variant))
    print(
        "BEST_MISSION_MACRO_ACCURACY={:.9f}".format(
            result["variant_metrics"]["variants"][best_variant]["mission_macro_accuracy"]
        )
    )
    print("HISTORY_COVERAGE={}".format(json.dumps(result["coverage"], sort_keys=True)))
    print("OUTPUT_DIR={}".format(result["output_root"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
