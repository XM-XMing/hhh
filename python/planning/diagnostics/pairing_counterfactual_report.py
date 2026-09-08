#!/usr/bin/env python3
"""Summarize a captured admissible state/depth counterfactual replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from planning.diagnostics.pairing_counterfactual import (
    PAIRING_COUNTERFACTUAL_CONTRACT_ID,
    risk_rankings,
    summarize_counterfactual_steps,
)


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if str(report.get("contract_id", "")) != PAIRING_COUNTERFACTUAL_CONTRACT_ID:
        raise ValueError("pairing counterfactual contract mismatch")
    steps = list(report.get("steps", []))
    max_skew_ns = int(report["max_sensor_skew_ns"])
    summary = summarize_counterfactual_steps(
        steps, max_sensor_skew_ns=max_skew_ns
    )
    summary.update({
        "total_captured_state_frames": len(report.get("states", [])),
        "total_captured_depth_frames": len(report.get("depths", [])),
        "risk_rankings": risk_rankings(steps, limit=10),
    })
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "PAIRING_COUNTERFACTUAL_DIAGNOSIS classification={} decisions={} "
            "pairs={} max_pairs={}".format(
                summary["classification"],
                summary["policy_decision_count"],
                summary["total_admissible_pairs"],
                summary["admissible_pair_count_distribution"]["max"],
            )
        )
        print(
            "content_steps={} mask_steps={} logits_steps={} action_steps={} terminal_steps={}".format(
                summary["steps_with_multiple_observation_fingerprints"],
                summary["steps_with_multiple_masks"],
                summary["steps_with_multiple_logits_fingerprints"],
                summary["steps_with_multiple_top1_actions"],
                summary["steps_with_terminal_reward_input_changes"],
            )
        )
        print(
            "max_logits_linf={} max_mask_hamming={} threshold_margin_min_ns={}".format(
                summary["maximum_logits_linf_delta"],
                summary["maximum_mask_hamming_distance"],
                summary["threshold_margin_min_ns"],
            )
        )
        print("RESULT={}".format("RED" if summary["classification"] == "T2-B" else "NO_ACTION_DIVERGENCE"))
    return 1 if summary["classification"] == "T2-B" else 0


if __name__ == "__main__":
    raise SystemExit(main())
