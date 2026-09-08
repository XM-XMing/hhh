"""Create an immutable, risk-enriched Layer-2 mission manifest from BC audit data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from planning.diagnostics.cross_mission_t2_risk import (
    CROSS_MISSION_T2_RISK_CONTRACT_ID,
    build_risk_table,
    select_risk_enriched_missions,
)


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--pairing-counts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args(argv)

    trace_payload = json.loads(args.trace.read_text(encoding="utf-8"))
    traces = list(trace_payload.get("episodes", []))
    with args.rollout.open(encoding="utf-8", newline="") as stream:
        rollout_by_episode = {
            int(row["episode_id"]): row for row in csv.DictReader(stream)
        }
    pairing_payload = json.loads(args.pairing_counts.read_text(encoding="utf-8"))
    pairing_counts_by_episode = {
        int(row["episode_id"]): list(row.get("boundaries", []))
        for row in pairing_payload
    }
    table = build_risk_table(traces, rollout_by_episode, pairing_counts_by_episode)
    if len(table) != 100:
        raise ValueError("Layer-2 baseline audit must contain exactly 100 missions")
    selection = select_risk_enriched_missions(table, count=int(args.count))
    output = {
        "contract_id": CROSS_MISSION_T2_RISK_CONTRACT_ID,
        "source_trace": str(args.trace),
        "source_rollout": str(args.rollout),
        "source_pairing_counts": str(args.pairing_counts),
        "risk_table": table,
        "selection": selection,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "CROSS_MISSION_T2_SELECTION missions={} episodes={} out={}".format(
            len(selection), ",".join(str(row["episode_id"]) for row in selection), args.out
        )
    )
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
