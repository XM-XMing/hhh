#!/usr/bin/env python3
"""Build an explicit mission-disjoint train/dev/final split for AWAC."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.evaluation.mission_split import build_awac_mission_split


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--dev-index", required=True)
    parser.add_argument("--final-holdout-index", required=True)
    parser.add_argument("--out-train-index", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = build_awac_mission_split(
        source_index_path=Path(args.source_index),
        dev_index_path=Path(args.dev_index),
        final_holdout_index_path=Path(args.final_holdout_index),
        train_index_path=Path(args.out_train_index),
        manifest_path=Path(args.out_manifest),
        overwrite=bool(args.overwrite),
    )
    print("AWAC_MISSION_SPLIT_SUMMARY")
    for key in ("source", "train", "dev", "final_holdout"):
        print("  {}_missions: {}".format(key, manifest[key]["mission_count"]))
        print("  {}_sha256: {}".format(key, manifest[key]["file_sha256"]))
    print("  intersections:", manifest["intersections"])
    print("  manifest_sha256:", manifest["manifest_sha256"])
    print("  train_index:", Path(args.out_train_index).expanduser().resolve())
    print("  manifest:", Path(args.out_manifest).expanduser().resolve())
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
