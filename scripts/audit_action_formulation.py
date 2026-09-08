#!/usr/bin/env python3
"""Run the read-only 105-action versus primitive-descriptor audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planning.diagnostics.action_formulation_audit import run_action_formulation_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit semantic action structure and primitive-level ranking."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--temporal-audit", type=Path, required=True)
    parser.add_argument("--motion-primitives-json", type=Path, required=True)
    parser.add_argument("--motion-primitives-npz", type=Path, required=True)
    parser.add_argument("--motion-primitives-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_action_formulation_audit(
        args.dataset,
        args.out_dir,
        state_audit_root=args.state_audit,
        temporal_audit_root=args.temporal_audit,
        motion_primitives_json=args.motion_primitives_json,
        motion_primitives_npz=args.motion_primitives_npz,
        motion_primitives_config=args.motion_primitives_config,
        seed=args.seed,
        bootstrap_repeats=args.bootstrap_repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    model = result["primitive_metrics"]["models"]
    print("ACTION_FORMULATION_AUDIT=PASS")
    print(
        "ONE_HOT_MISSION_MACRO_ACCURACY={:.9f}".format(
            model["105_one_hot"]["oof_metrics"]["mission_macro"]["accuracy"]
        )
    )
    print(
        "DESCRIPTOR_MISSION_MACRO_ACCURACY={:.9f}".format(
            model["primitive_descriptor"]["oof_metrics"]["mission_macro"]["accuracy"]
        )
    )
    print("OUTPUT_DIR={}".format(result["output_root"]))
    print(json.dumps(result["action_analysis"]["equivalence"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
