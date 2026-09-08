#!/usr/bin/env python3
"""Build and validate the independent Phase-0 AWAC development mission set."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.evaluation.mission_split import build_awac_dev_set


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic AWAC DEV set disjoint from BC and final-test missions."
    )
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--bc-train-index", required=True, action="append")
    parser.add_argument("--final-test-index", required=True, action="append")
    parser.add_argument("--out-dev-index", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_awac_dev_set(
        source_index_path=Path(args.source_index),
        bc_train_index_paths=tuple(Path(path) for path in args.bc_train_index),
        final_test_index_paths=tuple(Path(path) for path in args.final_test_index),
        dev_index_path=Path(args.out_dev_index),
        manifest_path=Path(args.out_manifest),
        count=int(args.count),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
    )
    print("AWAC_DEV_SET")
    print("  source_missions:", manifest["source"]["mission_count"])
    print("  candidate_missions:", manifest["candidate_count"])
    print("  dev_missions:", manifest["dev"]["mission_count"])
    print("  seed:", manifest["seed"])
    print("  intersections:", manifest["intersections"])
    print("  manifest_sha256:", manifest["manifest_sha256"])
    print("  out_dev_index:", Path(args.out_dev_index).expanduser().resolve())
    print("  out_manifest:", Path(args.out_manifest).expanduser().resolve())
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
