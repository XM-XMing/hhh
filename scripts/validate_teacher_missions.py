#!/usr/bin/env python3
"""Validate prepared formal missions and their immutable RouteStore."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from planning.mission.validation import validate_teacher_missions


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate formal V2 Teacher missions and RouteStore provenance."
    )
    parser.add_argument("--candidate-index", required=True)
    parser.add_argument("--missions", required=True)
    parser.add_argument("--route-store-prefix", required=True)
    parser.add_argument("--expected-passing", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    return parser


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = validate_teacher_missions(
            candidate_index=Path(args.candidate_index),
            missions=Path(args.missions),
            route_store_prefix=Path(args.route_store_prefix),
            expected_passing=int(args.expected_passing),
            max_steps=int(args.max_steps),
            expected_seed=int(args.expected_seed),
        )
    except Exception as error:
        print("MISSION_VALIDATION=FAIL")
        print("ERROR={}".format(error), file=sys.stderr)
        return 1
    for key in (
        "mission_count",
        "candidate_count",
        "candidate_index_row_count",
        "candidate_csv_row_count",
        "preparation_candidate_count",
        "routed_count",
        "audited_count",
        "passing_count",
        "rejected_route_count",
        "sampling_attempts",
        "route_count",
        "mission_sha256",
    ):
        print("{}={}".format(key.upper(), result[key]))
    print("MISSION_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
