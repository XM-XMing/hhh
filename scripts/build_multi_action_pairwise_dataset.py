#!/usr/bin/env python3
"""Build same-state return pairs from completed multi-action branches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser


def _load_completed(path: Path) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("invalid branch JSON at line {}".format(line_number)) from error
            if str(row.get("status", "")) != "completed":
                continue
            state_id = str(row.get("state_id", "")).strip()
            if not state_id:
                raise ValueError("completed branch has no state_id")
            episode_return = float(row.get("episode_return"))
            if not np.isfinite(episode_return):
                raise ValueError("completed branch has non-finite episode_return")
            grouped.setdefault(state_id, []).append(dict(row))
    return [row for state_rows in grouped.values() for row in state_rows]


def _as_bool(value: Any) -> int:
    return 1 if bool(value) else 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    branches = _load_completed(args.branches)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in branches:
        grouped.setdefault(str(row["state_id"]), []).append(row)

    pair_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))
    effective_count = 0
    random_correct = 0
    tie_count = 0
    for state_id in sorted(grouped):
        rows = sorted(
            grouped[state_id],
            key=lambda row: (int(row.get("action", -1)), str(row.get("action_source", ""))),
        )
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                left = rows[left_index]
                right = rows[right_index]
                left_return = float(left["episode_return"])
                right_return = float(right["episode_return"])
                tie = bool(left_return == right_return)
                if tie:
                    higher_action = ""
                    tie_count += 1
                elif left_return > right_return:
                    higher_action = str(left["action"])
                    effective_count += 1
                    random_correct += int(bool(rng.integers(0, 2) == 0))
                else:
                    higher_action = str(right["action"])
                    effective_count += 1
                    random_correct += int(bool(rng.integers(0, 2) == 1))
                pair_rows.append(
                    {
                        "state_id": state_id,
                        "mission_id": str(left.get("mission_id", "")),
                        "step_id": int(left.get("step_id", -1)),
                        "action1": int(left["action"]),
                        "action2": int(right["action"]),
                        "action_source1": str(left.get("action_source", "")),
                        "action_source2": str(right.get("action_source", "")),
                        "return1": left_return,
                        "return2": right_return,
                        "success1": _as_bool(left.get("success", False)),
                        "success2": _as_bool(right.get("success", False)),
                        "collision1": _as_bool(left.get("collision", False)),
                        "collision2": _as_bool(right.get("collision", False)),
                        "dead_end1": _as_bool(left.get("dead_end", False)),
                        "dead_end2": _as_bool(right.get("dead_end", False)),
                        "timeout1": _as_bool(left.get("timeout", False)),
                        "timeout2": _as_bool(right.get("timeout", False)),
                        "terminal_reason1": str(left.get("terminal_reason", "")),
                        "terminal_reason2": str(right.get("terminal_reason", "")),
                        "higher_return_action": higher_action,
                        "return_tie": int(tie),
                    }
                )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "state_id", "mission_id", "step_id", "action1", "action2",
        "action_source1", "action_source2", "return1", "return2",
        "success1", "success2", "collision1", "collision2",
        "dead_end1", "dead_end2", "timeout1", "timeout2",
        "terminal_reason1", "terminal_reason2", "higher_return_action", "return_tie",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)

    report = {
        "schema_id": "multi_action_pairwise_dataset",
        "source_branch_file": str(Path(args.branches).expanduser().resolve()),
        "seed": int(args.seed),
        "state_count": int(len(grouped)),
        "completed_branch_count": int(len(branches)),
        "pair_count": int(len(pair_rows)),
        "effective_pair_count": int(effective_count),
        "tie_count": int(tie_count),
        "random_pair_correct_count": int(random_correct),
        "random_pair_accuracy": float(random_correct / effective_count) if effective_count else None,
        "production_replay_modified": False,
        "training_executed": False,
    }
    audit_output = Path(args.audit_output).expanduser().resolve()
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("MULTI_ACTION_PAIRWISE_DATASET=PASS")
    print("STATE_COUNT={}".format(report["state_count"]))
    print("PAIR_COUNT={}".format(report["pair_count"]))
    print("RANDOM_PAIR_ACCURACY={}".format(report["random_pair_accuracy"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
