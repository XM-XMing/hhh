#!/usr/bin/env python3
"""Run the read-only observation information audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planning.diagnostics.observation_information_audit import run_observation_information_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit which observation components predict same-state action outcomes."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_observation_information_audit(
        args.dataset,
        args.out_dir,
        state_audit_root=args.state_audit,
        seed=args.seed,
        fold_count=args.folds,
        bootstrap_repeats=args.bootstrap_repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    canonical = result["component_metrics"]["canonical"]["oof_metrics"]["micro"]
    print("OBSERVATION_INFORMATION_AUDIT=PASS")
    print("CANONICAL_PAIRWISE_ACCURACY={:.9f}".format(canonical["accuracy"]))
    print("OUTPUT_DIR={}".format(result["output_root"]))
    print(json.dumps({"q_features_used": result["q_features_used"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
