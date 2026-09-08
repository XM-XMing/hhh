#!/usr/bin/env python3
"""Run the independent state sufficiency audit on the diagnostic replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planning.diagnostics.state_sufficiency import run_state_sufficiency_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit whether observation state plus action predicts recorded outcomes."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="multi-action diagnostic dataset root")
    parser.add_argument("--out-dir", type=Path, required=True, help="new audit artifact directory")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_state_sufficiency_audit(
        args.dataset,
        args.out_dir,
        seed=args.seed,
        fold_count=args.folds,
        bootstrap_repeats=args.bootstrap_repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("STATE_SUFFICIENCY_AUDIT=PASS")
    print(json.dumps(result["state_sufficiency_result"], sort_keys=True))
    print("OUTPUT_DIR={}".format(result["output_root"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
