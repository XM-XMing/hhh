#!/usr/bin/env python3
"""Validate the final rollout, label, depth-mask, and BC mmap chain."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from planning.data.bc_mmap import validate_bc_dataset_artifacts


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate formal Pre-BC artifact provenance and accounting."
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--depth-action-masks", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--bc-mmap", required=True)
    parser.add_argument("--expected-observation-contract", required=True)
    parser.add_argument("--mode", choices=("fast", "full"), default="full")
    parser.add_argument("--sample-count", type=int, default=10000)
    return parser


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = validate_bc_dataset_artifacts(
            index_path=Path(args.index),
            labels_path=Path(args.labels),
            depth_masks_path=Path(args.depth_action_masks),
            dataset_audit_path=Path(args.dataset_audit),
            bc_mmap_dir=Path(args.bc_mmap),
            expected_observation_contract=args.expected_observation_contract,
            mode=args.mode,
            sample_count=int(args.sample_count),
        )
    except Exception as error:
        print("BC_DATASET_VALIDATION=FAIL")
        print("ERROR={}".format(error), file=sys.stderr)
        return 1
    for key in (
        "row_count",
        "episode_count",
        "observation_contract",
        "rollout_index_sha256",
        "rollout_manifest_sha256",
        "labels_sha256",
        "depth_masks_sha256",
        "bc_mmap_manifest_sha256",
    ):
        print("{}={}".format(key.upper(), result[key]))
    print("BC_DATASET_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
