#!/usr/bin/env python3
"""Compare two fixed-holdout policy evaluations on identical mission rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from planning.common.hashing import file_sha256
from planning.contracts.feature import policy_input_contract_sha256
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.task import (
    TASK_CONTRACT_ID,
    TASK_CONTRACT_SCHEMA_VERSION,
    task_contract_sha256,
)
from planning.primitives.library import MotionPrimitiveLibrary


OUTCOMES = ("success", "collision", "dead_end", "timeout", "far", "hard_altitude")
SUMMARY_COUNT_KEYS = {
    "success": "success_count",
    "collision": "collision_count",
    "dead_end": "dead_end_count",
    "timeout": "timeout_count",
    "far": "far_count",
    "hard_altitude": "altitude_violation_count",
}


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_evaluation(path: Path) -> Tuple[Dict, List[Dict]]:
    summary_path = path / "summary.json"
    index_path = path / "rollout_index.csv"
    if not summary_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("evaluation requires summary.json and rollout_index.csv: {}".format(path))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if int(summary.get("episodes", -1)) != len(rows):
        raise ValueError("summary/index episode count mismatch: {}".format(path))
    return summary, rows


def _load_checkpoint_identity(eval_dir: Path, summary: Mapping) -> Dict[str, str]:
    """Read frozen checkpoint identity without modifying the checkpoint."""

    import torch

    checkpoint_path = Path(str(summary["checkpoint"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = eval_dir / checkpoint_path
    try:
        checkpoint = torch.load(str(checkpoint_path.resolve()), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path.resolve()), map_location="cpu")
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": str(summary["checkpoint_sha256"]),
        "policy_input_contract_sha256": str(checkpoint.get("policy_input_contract_sha256", "")),
        "mpl_contract_sha256": str(checkpoint.get("mpl_contract_sha256", "")),
    }


def _outcome(row: Mapping) -> str:
    active = [name for name in OUTCOMES if _bool(row.get(name, False))]
    if len(active) != 1:
        raise ValueError("row must have exactly one terminal outcome: {}".format(row))
    return active[0]


def _validate_contract(summary: Mapping, rows: Sequence[Mapping], label: str) -> None:
    expected_sha = task_contract_sha256(45)
    for field, expected in (
        ("task_contract_id", TASK_CONTRACT_ID),
        ("task_contract_schema_version", TASK_CONTRACT_SCHEMA_VERSION),
        ("task_contract_sha256", expected_sha),
        ("max_primitive_steps", 45),
    ):
        if int(summary[field]) != int(expected) if isinstance(expected, int) else summary[field] != expected:
            raise ValueError("{} summary task contract mismatch: {}".format(label, field))
    for row in rows:
        if row.get("task_contract_id") != TASK_CONTRACT_ID:
            raise ValueError("{} row task contract id mismatch".format(label))
        if int(row.get("task_contract_schema_version", -1)) != TASK_CONTRACT_SCHEMA_VERSION:
            raise ValueError("{} row task contract schema mismatch".format(label))
        if row.get("task_contract_sha256") != expected_sha or int(row.get("max_primitive_steps", -1)) != 45:
            raise ValueError("{} row task contract hash/horizon mismatch".format(label))
        if row.get("observation_contract") != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
            raise ValueError("{} row observation contract mismatch".format(label))
        if row.get("observation_source") != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
            raise ValueError("{} row observation source mismatch".format(label))
    if summary.get("observation_contract") != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("{} summary observation contract mismatch".format(label))
    if summary.get("observation_source") != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError("{} summary observation source mismatch".format(label))
    for outcome, key in SUMMARY_COUNT_KEYS.items():
        count = sum(_outcome(row) == outcome for row in rows)
        if int(summary.get(key, -1)) != count:
            raise ValueError("{} summary outcome count mismatch: {}".format(label, outcome))


def _exact_mcnemar_p(regressed: int, recovered: int) -> float:
    total = int(regressed) + int(recovered)
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(regressed, recovered) + 1))
    return min(1.0, 2.0 * float(tail) / float(2 ** total))


def _classification(success_rate: float) -> str:
    percent = 100.0 * float(success_rate)
    if percent >= 72.0:
        return "IMPROVED"
    if percent >= 66.0:
        return "STABLE_FLAT"
    if percent >= 59.0:
        return "DEGRADED"
    return "COLLAPSED"


def _write_json(path: Path, value: Mapping) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, result: Mapping) -> None:
    bc = result["aggregate"]["bc"]
    awac = result["aggregate"]["awac"]
    paired = result["paired"]
    mcnemar = result["mcnemar"]
    lines = [
        "# Paired BC60K vs AWAC10K Dev100",
        "",
        "Fixed holdout rows are aligned by exact `episode_id` and `mission_id` order.",
        "",
        "| policy | success | collision | dead_end | timeout | mean return |",
        "|---|---:|---:|---:|---:|---:|",
        "| BC60K | {success_count} | {collision_count} | {dead_end_count} | {timeout_count} | {episode_return_mean:.6f} |".format(**bc),
        "| AWAC10K | {success_count} | {collision_count} | {dead_end_count} | {timeout_count} | {episode_return_mean:.6f} |".format(**awac),
        "",
        "## Paired outcomes",
        "",
        "- BC success → AWAC success: `{bc_success_awac_success}`",
        "- BC success → AWAC failure: `{bc_success_awac_fail}`",
        "- BC failure → AWAC success (recovered): `{bc_fail_awac_success}`",
        "- BC failure → AWAC failure: `{bc_fail_awac_fail}`",
        "- Regressed: `{regressed}`; recovered: `{recovered}`; net: `{net_success_change}`",
        "- Exact two-sided McNemar p-value: `{p_value:.8f}`",
        "",
        "## Classification",
        "",
        "- AWAC success-rate classification: `{awac_success_rate_classification}`",
        "- Success-rate delta (AWAC - BC): `{delta_success_pp:.2f} pp`",
        "- Historical BC result match (69/9/22/0): `{historical_bc_match}`",
        "",
        "## Provenance",
        "",
        "- Task contract: `{task_contract_id}` schema `{task_contract_schema_version}` SHA `{task_contract_sha256}`",
        "- Observation contract: `{observation_contract}`",
        "- Policy input contract SHA: `{policy_input_contract_sha256}`",
        "- MPL contract SHA: `{mpl_contract_sha256}`",
        "- BC checkpoint SHA: `{bc_checkpoint_sha256}`",
        "- AWAC checkpoint SHA: `{awac_checkpoint_sha256}`",
        "- Mission index SHA: `{mission_index_sha256}`",
    ]
    format_values = dict(paired)
    format_values.update(mcnemar)
    format_values.update(result["identity"])
    format_values.update(result["provenance"])
    format_values.update(result["classification"])
    lines = [line.format(**format_values) for line in lines]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparison(bc_dir: Path, awac_dir: Path) -> Tuple[Dict, List[Dict]]:
    bc_summary, bc_rows = _read_evaluation(bc_dir)
    awac_summary, awac_rows = _read_evaluation(awac_dir)
    if len(bc_rows) != len(awac_rows):
        raise ValueError("BC/AWAC episode counts differ")
    _validate_contract(bc_summary, bc_rows, "BC")
    _validate_contract(awac_summary, awac_rows, "AWAC")
    bc_checkpoint = _load_checkpoint_identity(bc_dir, bc_summary)
    awac_checkpoint = _load_checkpoint_identity(awac_dir, awac_summary)
    expected_policy_input_sha = policy_input_contract_sha256()
    expected_mpl_sha = MotionPrimitiveLibrary().contract_sha256
    for label, identity in (("BC", bc_checkpoint), ("AWAC", awac_checkpoint)):
        if identity["policy_input_contract_sha256"] != expected_policy_input_sha:
            raise ValueError("{} policy input contract mismatch".format(label))
        if identity["mpl_contract_sha256"] != expected_mpl_sha:
            raise ValueError("{} MPL contract mismatch".format(label))
    if bc_checkpoint["policy_input_contract_sha256"] != awac_checkpoint["policy_input_contract_sha256"]:
        raise ValueError("BC/AWAC policy input contract mismatch")
    if bc_checkpoint["mpl_contract_sha256"] != awac_checkpoint["mpl_contract_sha256"]:
        raise ValueError("BC/AWAC MPL contract mismatch")
    bc_identity = [(str(row["episode_id"]), str(row["mission_id"])) for row in bc_rows]
    awac_identity = [(str(row["episode_id"]), str(row["mission_id"])) for row in awac_rows]
    if bc_identity != awac_identity:
        raise ValueError("BC/AWAC episode or mission order mismatch")

    paired_rows = []
    counts = Counter()
    reason_deltas = {}
    for bc_row, awac_row in zip(bc_rows, awac_rows):
        bc_outcome = _outcome(bc_row)
        awac_outcome = _outcome(awac_row)
        bc_success = bc_outcome == "success"
        awac_success = awac_outcome == "success"
        key = (
            "bc_success_awac_success" if bc_success and awac_success else
            "bc_success_awac_fail" if bc_success else
            "bc_fail_awac_success" if awac_success else
            "bc_fail_awac_fail"
        )
        counts[key] += 1
        paired_rows.append({
            "episode_id": bc_row["episode_id"],
            "mission_id": bc_row["mission_id"],
            "bc_outcome": bc_outcome,
            "awac_outcome": awac_outcome,
            "bc_success": bc_success,
            "awac_success": awac_success,
            "bc_return": float(bc_row["return"]),
            "awac_return": float(awac_row["return"]),
        })
    for outcome in OUTCOMES:
        bc_count = int(bc_summary[SUMMARY_COUNT_KEYS[outcome]])
        awac_count = int(awac_summary[SUMMARY_COUNT_KEYS[outcome]])
        reason_deltas[outcome] = {
            "bc_count": bc_count,
            "awac_count": awac_count,
            "delta_awac_minus_bc": awac_count - bc_count,
        }
    regressed = counts["bc_success_awac_fail"]
    recovered = counts["bc_fail_awac_success"]
    mission_index_sha = bc_summary.get("mission_index_sha256")
    if mission_index_sha != awac_summary.get("mission_index_sha256"):
        raise ValueError("BC/AWAC mission index SHA mismatch")
    aggregate = {
        "bc": {key: bc_summary.get(key) for key in (
            "episodes", "success_count", "collision_count", "dead_end_count",
            "timeout_count", "far_count", "altitude_violation_count",
            "success_rate", "collision_rate", "dead_end_rate", "timeout_rate",
            "episode_return_mean", "episode_steps_mean",
        )},
        "awac": {key: awac_summary.get(key) for key in (
            "episodes", "success_count", "collision_count", "dead_end_count",
            "timeout_count", "far_count", "altitude_violation_count",
            "success_rate", "collision_rate", "dead_end_rate", "timeout_rate",
            "episode_return_mean", "episode_steps_mean",
        )},
    }
    result = {
        "comparison_schema": "paired_policy_evaluation_v1",
        "identity": {
            "episode_count": len(bc_rows),
            "episode_order_exact": True,
            "episode_identity_exact": True,
            "mission_identity_exact": True,
            "task_contract_id": TASK_CONTRACT_ID,
            "task_contract_schema_version": TASK_CONTRACT_SCHEMA_VERSION,
            "task_contract_sha256": task_contract_sha256(45),
            "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            "policy_input_contract_sha256": expected_policy_input_sha,
            "mpl_contract_sha256": expected_mpl_sha,
        },
        "provenance": {
            "bc_checkpoint": bc_checkpoint["checkpoint_path"],
            "bc_checkpoint_sha256": bc_checkpoint["checkpoint_sha256"],
            "awac_checkpoint": awac_checkpoint["checkpoint_path"],
            "awac_checkpoint_sha256": awac_checkpoint["checkpoint_sha256"],
            "policy_input_contract_sha256": expected_policy_input_sha,
            "mpl_contract_sha256": expected_mpl_sha,
            "bc_normalizer_checkpoint": bc_summary.get("normalizer_checkpoint"),
            "awac_normalizer_checkpoint": awac_summary.get("normalizer_checkpoint"),
            "mission_index_sha256": mission_index_sha,
            "bc_summary": str((bc_dir / "summary.json").resolve()),
            "awac_summary": str((awac_dir / "summary.json").resolve()),
        },
        "aggregate": aggregate,
        "paired": {
            **dict(counts),
            "regressed": regressed,
            "recovered": recovered,
            "net_success_change": recovered - regressed,
            "delta_success_pp": 100.0 * (float(awac_summary["success_rate"]) - float(bc_summary["success_rate"])),
            "reason_deltas": reason_deltas,
        },
        "mcnemar": {
            "regressed": regressed,
            "recovered": recovered,
            "p_value": _exact_mcnemar_p(regressed, recovered),
        },
        "classification": {
            "awac_success_rate_classification": _classification(awac_summary["success_rate"]),
            "historical_bc_match": bool(
                int(bc_summary["success_count"]) == 69
                and int(bc_summary["collision_count"]) == 9
                and int(bc_summary["dead_end_count"]) == 22
                and int(bc_summary["timeout_count"]) == 0
            ),
        },
        "validation": {
            "bc_outcome_partition_valid": bool(bc_summary.get("outcome_partition_valid")),
            "awac_outcome_partition_valid": bool(awac_summary.get("outcome_partition_valid")),
            "bc_snapshot_missing_count": int(bc_summary.get("snapshot_missing_count", -1)),
            "awac_snapshot_missing_count": int(awac_summary.get("snapshot_missing_count", -1)),
            "bc_telemetry_lookup_count": int(bc_summary.get("telemetry_lookup_count", -1)),
            "awac_telemetry_lookup_count": int(awac_summary.get("telemetry_lookup_count", -1)),
            "bc_runtime_contract_override": bool(bc_summary.get("runtime_contract_override")),
            "awac_runtime_contract_override": bool(awac_summary.get("runtime_contract_override")),
        },
    }
    return result, paired_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-dir", required=True, type=Path)
    parser.add_argument("--awac-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result, paired_rows = build_comparison(args.bc_dir.resolve(), args.awac_dir.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "comparison.json", result)
    _write_markdown(args.out_dir / "comparison.md", result)
    with (args.out_dir / "per_mission_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)
    print("PAIRED_POLICY_COMPARISON=PASS")
    print("EPISODES={}".format(result["identity"]["episode_count"]))
    print("BC_SUCCESS={}".format(result["aggregate"]["bc"]["success_count"]))
    print("AWAC_SUCCESS={}".format(result["aggregate"]["awac"]["success_count"]))
    print("MCNEMAR_P_VALUE={:.8f}".format(result["mcnemar"]["p_value"]))
    print("CLASSIFICATION={}".format(result["classification"]["awac_success_rate_classification"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
