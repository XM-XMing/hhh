#!/usr/bin/env python3
"""Build an episode-disjoint AWAC critic-calibration train/holdout split."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.evaluation.mission_split import build_awac_calibration_split


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic calibration split excluding BC, final-test, and dev missions."
    )
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--exclude-index", required=True, action="append")
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_awac_calibration_split(
        source_index_path=Path(args.source_index),
        exclusion_index_paths=tuple(Path(path) for path in args.exclude_index),
        output_path=Path(args.out_manifest),
        seed=int(args.seed),
        holdout_fraction=float(args.holdout_fraction),
        overwrite=bool(args.overwrite),
    )
    print("AWAC_CALIBRATION_SPLIT")
    print("  eligible_missions:", manifest["eligible_mission_count"])
    print("  train_missions:", manifest["train_mission_count"])
    print("  holdout_missions:", manifest["holdout_mission_count"])
    print("  manifest_sha256:", manifest["manifest_sha256"])
    print("  out_manifest:", Path(args.out_manifest).expanduser().resolve())
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
