#!/usr/bin/env python3
"""Validate formal reliable-exact Teacher collection artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from planning.teacher.collection_validation import validate_teacher_collection


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate formal Teacher rollout provenance and accounting."
    )
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--missions", required=True)
    parser.add_argument("--route-store-prefix", required=True)
    parser.add_argument("--min-accepted", type=int, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--expected-observation-contract", required=True)
    parser.add_argument(
        "--require-runtime-quality",
        action="store_true",
        help="also require zero runtime collector errors and all runtime quality gates",
    )
    return parser


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = validate_teacher_collection(
            rollout_dir=Path(args.rollout_dir),
            missions=Path(args.missions),
            route_store_prefix=Path(args.route_store_prefix),
            min_accepted=int(args.min_accepted),
            expected_workers=int(args.expected_workers),
            max_steps=int(args.max_steps),
            expected_observation_contract=args.expected_observation_contract,
            require_runtime_quality=bool(args.require_runtime_quality),
        )
    except Exception as error:
        print("COLLECTION_VALIDATION=FAIL")
        print("ERROR={}".format(error), file=sys.stderr)
        return 1
    for key in (
        "accepted_count",
        "source_accepted_count",
        "index_row_count",
        "reliable_rows",
        "transition_count",
        "npz_transition_count",
        "mission_count",
        "route_count",
        "worker_count",
        "legacy_rows",
        "telemetry_lookup_count",
        "snapshot_missing_count",
        "state_depth_skew_max_ns",
        "frame_contract_failures",
        "collector_error_total",
        "source_collector_error_count",
        "runtime_quality_pass",
        "accepted_dataset_quality_pass",
        "observation_contract",
    ):
        print("{}={}".format(key.upper(), result[key]))
    print("COLLECTION_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
