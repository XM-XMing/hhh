#!/usr/bin/env python3
"""Select an AWAC checkpoint using the manifest-bound development split."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.awac.dev_selection import select_awac_dev_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--incumbent-checkpoint", required=True)
    parser.add_argument("--incumbent-summary", required=True)
    parser.add_argument("--out-checkpoint", required=True)
    parser.add_argument("--out-decision", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    decision = select_awac_dev_checkpoint(
        split_manifest_path=Path(args.split_manifest),
        candidate_checkpoint_path=Path(args.candidate_checkpoint),
        candidate_summary_path=Path(args.candidate_summary),
        incumbent_checkpoint_path=Path(args.incumbent_checkpoint),
        incumbent_summary_path=Path(args.incumbent_summary),
        out_checkpoint_path=Path(args.out_checkpoint),
        out_decision_path=Path(args.out_decision),
    )
    print("AWAC_DEV_CHECKPOINT_SELECTION")
    print("  candidate_selected:", decision["candidate_selected"])
    print("  selected_source:", decision["selected_source"])
    print("  candidate_score:", decision["candidate"]["score"])
    print("  incumbent_score:", decision["incumbent"]["score"])
    print("  gates:", decision["gates"])
    print("  selected_checkpoint_sha256:", decision["selected_checkpoint_sha256"])
    print("  decision:", Path(args.out_decision).expanduser().resolve())
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
