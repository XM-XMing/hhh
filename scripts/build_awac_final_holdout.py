#!/usr/bin/env python3
"""Select a deterministic final AWAC holdout unseen in BC training or dev."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.evaluation.mission_split import (
    build_awac_evaluation_holdouts,
    build_awac_final_holdout,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--bc-train-index", required=True, action="append")
    dev_group = parser.add_mutually_exclusive_group(required=True)
    dev_group.add_argument("--dev-index")
    dev_group.add_argument("--out-dev-index")
    parser.add_argument("--dev-count", type=int)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-index", required=True)
    parser.add_argument("--out-selection", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.out_dev_index:
        if args.dev_count is None:
            raise ValueError("--out-dev-index requires --dev-count")
        selection = build_awac_evaluation_holdouts(
            source_index_path=Path(args.source_index),
            bc_train_index_path=Path(args.bc_train_index[0]),
            dev_index_path=Path(args.out_dev_index),
            final_holdout_index_path=Path(args.out_index),
            selection_path=Path(args.out_selection),
            dev_count=int(args.dev_count),
            final_count=int(args.count),
            seed=int(args.seed),
            additional_bc_train_index_paths=tuple(
                Path(path) for path in args.bc_train_index[1:]
            ),
            overwrite=bool(args.overwrite),
        )
        print("AWAC_EVALUATION_HOLDOUT_PAIR_SELECTION")
        print("  source_sha256:", selection["source"]["file_sha256"])
        print("  bc_train_sha256:", selection["bc_train"]["file_sha256"])
        print("  bc_train_indexes:", 1 + len(selection["additional_bc_train"]))
        print("  seed:", selection["seed"])
        print("  candidate_count:", selection["candidate_count"])
        print("  dev_count:", selection["dev"]["mission_count"])
        print("  final_holdout_count:", selection["final_holdout"]["mission_count"])
        print("  intersections:", selection["intersections"])
        print("  selection_sha256:", selection["selection_sha256"])
        print("  out_dev_index:", Path(args.out_dev_index).expanduser().resolve())
        print("  out_final_index:", Path(args.out_index).expanduser().resolve())
        print("  out_selection:", Path(args.out_selection).expanduser().resolve())
        print("RESULT=PASS")
        return 0
    if args.dev_count is not None:
        raise ValueError("--dev-count is only valid with --out-dev-index")
    if len(args.bc_train_index) != 1:
        raise ValueError("repeated --bc-train-index is only valid with --out-dev-index")
    selection = build_awac_final_holdout(
        source_index_path=Path(args.source_index),
        bc_train_index_path=Path(args.bc_train_index[0]),
        dev_index_path=Path(args.dev_index),
        final_holdout_index_path=Path(args.out_index),
        selection_path=Path(args.out_selection),
        count=int(args.count),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
    )
    print("AWAC_FINAL_HOLDOUT_SELECTION")
    print("  source_sha256:", selection["source"]["file_sha256"])
    print("  bc_train_sha256:", selection["bc_train"]["file_sha256"])
    print("  dev_sha256:", selection["dev"]["file_sha256"])
    print("  seed:", selection["seed"])
    print("  requested_count:", selection["requested_count"])
    print("  candidate_count:", selection["candidate_count"])
    print("  ordered_mission_ids_sha256:", selection["final_holdout"]["ordered_mission_ids_sha256"])
    print("  intersections:", selection["intersections"])
    print("  selection_sha256:", selection["selection_sha256"])
    print("  out_index:", Path(args.out_index).expanduser().resolve())
    print("  out_selection:", Path(args.out_selection).expanduser().resolve())
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
