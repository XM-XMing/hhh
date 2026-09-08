#!/usr/bin/env python3
"""Aggregate the fixed-holdout BC data-scale closed-loop evaluations.

This is a report-only utility.  It validates that all model result directories
refer to the same mission set and reliable-exact runtime contract, then writes
paired per-mission rows and scale-level statistics.  It does not alter model,
mission, rollout, or evaluator artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


MODEL_NAMES = ("bc20k", "bc40k", "bc60k")
EXPECTED_OBSERVATION_CONTRACT = "reliable_exact_endpoint_snapshot"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float_or_none(value: object):
    text = str(value).strip()
    return None if not text else float(text)


def _int_or_none(value: object):
    text = str(value).strip()
    return None if not text else int(float(text))


def _read_rows(path: Path) -> List[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty evaluation rollout index: {}".format(path))
    required = {
        "episode_id",
        "mission_id",
        "success",
        "collision",
        "dead_end",
        "timeout",
        "far",
        "hard_altitude",
        "stop_reason",
        "steps",
        "final_distance_xy",
    }
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError("{} missing columns: {}".format(path, ", ".join(missing)))
    normalized = []
    for row in rows:
        item = dict(row)
        item["episode_id"] = str(row["episode_id"]).strip()
        item["mission_id"] = str(row["mission_id"]).strip()
        if not item["mission_id"]:
            raise ValueError("empty mission_id in {}".format(path))
        for field in ("success", "collision", "dead_end", "timeout", "far", "hard_altitude"):
            item[field] = _bool(row[field])
        item["steps"] = _int_or_none(row["steps"])
        item["final_distance_xy"] = _float_or_none(row["final_distance_xy"])
        normalized.append(item)
    mission_ids = [row["mission_id"] for row in normalized]
    if len(mission_ids) != len(set(mission_ids)):
        raise ValueError("duplicate mission_id in {}".format(path))
    return normalized


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: {}".format(path))
    return value


def _validate_runtime(summary: Mapping[str, object], manifest: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> dict:
    contract_fields = (
        "observation_contract",
        "observation_source",
        "checkpoint_observation_contract",
        "checkpoint_observation_source",
        "expected_observation_contract",
        "runtime_observation_contract",
        "runtime_observation_source",
    )
    contracts_ok = all(
        summary.get(field) == EXPECTED_OBSERVATION_CONTRACT for field in contract_fields
    )
    row_contracts_ok = all(
        row.get("observation_contract") == EXPECTED_OBSERVATION_CONTRACT
        and row.get("observation_source") == EXPECTED_OBSERVATION_CONTRACT
        and row.get("checkpoint_observation_contract") == EXPECTED_OBSERVATION_CONTRACT
        and row.get("checkpoint_observation_source") == EXPECTED_OBSERVATION_CONTRACT
        for row in rows
    )
    manifest_contracts_ok = all(
        manifest.get(field) == EXPECTED_OBSERVATION_CONTRACT
        for field in (
            "observation_contract",
            "observation_source",
            "checkpoint_observation_contract",
            "checkpoint_observation_source",
            "expected_observation_contract",
            "runtime_observation_contract",
            "runtime_observation_source",
        )
    )
    counters_ok = (
        int(summary.get("snapshot_missing_count", -1)) == 0
        and int(summary.get("telemetry_lookup_count", -1)) == 0
        and int(manifest.get("snapshot_missing_count", -1)) == 0
        and int(manifest.get("telemetry_lookup_count", -1)) == 0
    )
    reliable_flags_ok = (
        _bool(summary.get("reliable_v4"))
        and _bool(summary.get("reliable_execution"))
        and _bool(manifest.get("reliable_execution"))
        and _bool(summary.get("state_depth_exact_endpoint_binding"))
        and _bool(manifest.get("state_depth_exact_endpoint_binding"))
        and not _bool(summary.get("telemetry_observation"))
        and not _bool(summary.get("telemetry_fallback_enabled"))
        and not _bool(manifest.get("telemetry_observation"))
        and not _bool(manifest.get("telemetry_fallback_enabled"))
        and not _bool(summary.get("runtime_contract_override"))
        and not _bool(manifest.get("runtime_contract_override"))
    )
    return {
        "reliable_exact_quality": bool(
            contracts_ok
            and row_contracts_ok
            and manifest_contracts_ok
            and counters_ok
            and reliable_flags_ok
            and int(summary.get("unclassified_count", -1)) == 0
            and int(summary.get("ambiguous_outcome_count", -1)) == 0
            and int(summary.get("legacy_done_timeout_count", -1)) == 0
            and _bool(summary.get("outcome_partition_valid"))
        ),
        "contracts_ok": contracts_ok and row_contracts_ok and manifest_contracts_ok,
        "counters_ok": counters_ok,
        "reliable_flags_ok": reliable_flags_ok,
    }


def _load_model(
    name: str,
    result_dir: Path,
    expected_mission_sha: str,
    expected_episodes: int = 100,
) -> dict:
    summary_path = result_dir / "summary.json"
    manifest_path = result_dir / "runtime_observation_manifest.json"
    index_path = result_dir / "rollout_index.csv"
    for path in (summary_path, manifest_path, index_path):
        if not path.exists():
            raise FileNotFoundError(path)
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    rows = _read_rows(index_path)
    if int(summary.get("episodes", -1)) != len(rows):
        raise ValueError("{} summary/index episode count mismatch".format(name))
    if int(summary.get("episodes", -1)) != int(expected_episodes):
        raise ValueError(
            "{} does not contain the required {} episodes".format(
                name, expected_episodes
            )
        )
    if summary.get("mission_index_sha256") != expected_mission_sha:
        raise ValueError("{} mission index SHA mismatch".format(name))
    if manifest.get("mission_index_sha256") not in (None, expected_mission_sha):
        raise ValueError("{} runtime mission index SHA mismatch".format(name))
    if summary.get("task_contract_sha256") != rows[0].get("task_contract_sha256"):
        raise ValueError("{} task contract differs between summary and rows".format(name))
    outcomes = {
        "success": sum(bool(row["success"]) for row in rows),
        "collision": sum(bool(row["collision"]) for row in rows),
        "dead_end": sum(bool(row["dead_end"]) for row in rows),
        "timeout": sum(bool(row["timeout"]) for row in rows),
        "far": sum(bool(row["far"]) for row in rows),
        "hard_altitude": sum(bool(row["hard_altitude"]) for row in rows),
    }
    summary_fields = {
        "success": "success",
        "collision": "collision",
        "dead_end": "dead_end",
        "timeout": "timeout",
        "far": "far",
        "hard_altitude": "altitude_violation",
    }
    for field, count in outcomes.items():
        summary_field = summary_fields[field]
        if int(summary.get(summary_field + "_count", -1)) != count:
            raise ValueError("{} {} count differs from rollout rows".format(name, field))
    if sum(outcomes.values()) != len(rows):
        raise ValueError("{} outcome flags are not a partition".format(name))
    runtime = _validate_runtime(summary, manifest, rows)
    return {
        "name": name,
        "result_dir": str(result_dir),
        "summary": summary,
        "manifest": manifest,
        "rows": rows,
        "rows_by_mission": {row["mission_id"]: row for row in rows},
        "checkpoint_sha256": summary.get("checkpoint_sha256", ""),
        "mission_index_sha256": summary.get("mission_index_sha256", ""),
        "runtime": runtime,
        "outcomes": outcomes,
        "success_rate": outcomes["success"] / float(len(rows)),
    }


def wilson_ci95(successes: int, total: int) -> Tuple[float, float]:
    if total <= 0:
        raise ValueError("CI requires a positive sample count")
    z = 1.959963984540054
    p = successes / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _exact_mcnemar_pvalue(left_fail_right_success: int, left_success_right_fail: int) -> float:
    discordant = left_fail_right_success + left_success_right_fail
    if discordant == 0:
        return 1.0
    smaller = min(left_fail_right_success, left_success_right_fail)
    tail = sum(
        math.comb(discordant, k) for k in range(smaller + 1)
    ) / float(2 ** discordant)
    return min(1.0, 2.0 * tail)


def _paired(left: Mapping[str, dict], right: Mapping[str, dict]) -> dict:
    if set(left) != set(right):
        raise ValueError("paired model results do not contain the same mission IDs")
    fail_success = sum(not left[key]["success"] and right[key]["success"] for key in left)
    success_fail = sum(left[key]["success"] and not right[key]["success"] for key in left)
    return {
        "left_fail_right_success": fail_success,
        "left_success_right_fail": success_fail,
        "discordant_total": fail_success + success_fail,
        "mcnemar_exact_two_sided_p": _exact_mcnemar_pvalue(fail_success, success_fail),
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def aggregate(test_manifest_path: Path, results_root: Path, out_csv: Path, out_json: Path) -> dict:
    test_manifest_path = Path(test_manifest_path).resolve()
    results_root = Path(results_root).resolve()
    test_manifest = _read_json(test_manifest_path)
    mission_count = int(test_manifest.get("mission_count", -1))
    expected_mission_sha = test_manifest.get("artifacts", {}).get("missions", {}).get("sha256")
    if mission_count != 100 or not expected_mission_sha:
        raise ValueError("test manifest is not the frozen 100-mission holdout")
    models = {
        name: _load_model(name, results_root / ("results_" + name), expected_mission_sha)
        for name in MODEL_NAMES
    }
    mission_sets = [set(model["rows_by_mission"]) for model in models.values()]
    if any(value != mission_sets[0] for value in mission_sets[1:]):
        raise ValueError("model evaluations are not on the same mission set")

    paired_rows = []
    for mission_id in sorted(mission_sets[0], key=lambda value: int(float(value)) if value.replace(".", "", 1).isdigit() else value):
        output = {"mission_id": mission_id}
        for name in MODEL_NAMES:
            row = models[name]["rows_by_mission"][mission_id]
            output["{}_episode_id".format(name)] = row["episode_id"]
            output["{}_success".format(name)] = row["success"]
            output["{}_failure_reason".format(name)] = "" if row["success"] else row["stop_reason"]
            output["{}_steps".format(name)] = row["steps"]
            # The evaluator emits final distance, not path length.  Preserve
            # the distinction instead of relabeling final distance as path.
            output["{}_path_length".format(name)] = ""
            output["{}_final_distance_xy".format(name)] = row["final_distance_xy"]
        paired_rows.append(output)
    paired_fields = ["mission_id"]
    for name in MODEL_NAMES:
        paired_fields.extend(
            [
                "{}_episode_id".format(name),
                "{}_success".format(name),
                "{}_failure_reason".format(name),
                "{}_steps".format(name),
                "{}_path_length".format(name),
                "{}_final_distance_xy".format(name),
            ]
        )
    _write_csv(out_csv, paired_rows, paired_fields)

    model_stats = {}
    for name, model in models.items():
        total = len(model["rows"])
        ci_low, ci_high = wilson_ci95(model["outcomes"]["success"], total)
        success_steps = [row["steps"] for row in model["rows"] if row["success"] and row["steps"] is not None]
        model_stats[name] = {
            "total_episodes": total,
            "success_count": model["outcomes"]["success"],
            "success_rate": model["success_rate"],
            "success_ci95": [ci_low, ci_high],
            "collision_count": model["outcomes"]["collision"],
            "dead_end_count": model["outcomes"]["dead_end"],
            "timeout_count": model["outcomes"]["timeout"],
            "far_count": model["outcomes"]["far"],
            "hard_altitude_count": model["outcomes"]["hard_altitude"],
            "mean_steps_success": (
                sum(success_steps) / float(len(success_steps)) if success_steps else None
            ),
            "checkpoint_sha256": model["checkpoint_sha256"],
            "reliable_exact_quality": model["runtime"]["reliable_exact_quality"],
            "quality_gate_pass": _bool(model["summary"].get("quality_pass")),
            "runtime_validation": model["runtime"],
        }

    rates = [model_stats[name]["success_rate"] for name in MODEL_NAMES]
    deltas = {
        "20k_to_40k": rates[1] - rates[0],
        "40k_to_60k": rates[2] - rates[1],
        "20k_to_60k": rates[2] - rates[0],
    }
    paired = {
        "20k_vs_40k": _paired(models["bc20k"]["rows_by_mission"], models["bc40k"]["rows_by_mission"]),
        "40k_vs_60k": _paired(models["bc40k"]["rows_by_mission"], models["bc60k"]["rows_by_mission"]),
        "20k_vs_60k": _paired(models["bc20k"]["rows_by_mission"], models["bc60k"]["rows_by_mission"]),
    }
    if rates[0] < rates[1] < rates[2] and deltas["20k_to_40k"] >= 0.10:
        conclusion = "A_DATA_LIMITED"
    elif deltas["20k_to_40k"] >= 0.10 and abs(deltas["40k_to_60k"]) <= 0.05:
        conclusion = "B_APPROXIMATE_SATURATION_FROM_40K"
    elif max(rates) - min(rates) <= 0.05:
        conclusion = "C_NOT_OBVIOUSLY_DATA_LIMITED"
    elif rates[1] - rates[2] >= 0.10:
        conclusion = "D_60K_LOWER_REQUIRES_CHECKPOINT_REVIEW"
    else:
        conclusion = "INCONCLUSIVE_WITH_100_MISSIONS"
    small_difference = abs(deltas["40k_to_60k"]) <= 0.05 or any(
        abs(value) <= 0.05 for value in deltas.values()
    )
    report = {
        "schema": "bc_data_scale_closed_loop_comparison_v1",
        "study": "BC_DATA_SCALE_20K_40K_60K_CLOSED_LOOP_EVAL_V1",
        "test_manifest": str(test_manifest_path),
        "test_manifest_sha256": file_sha256(test_manifest_path),
        "mission_index_sha256": expected_mission_sha,
        "test_missions": mission_count,
        "ci95_method": "Wilson score interval, z=1.959963984540054",
        "paired_csv": str(Path(out_csv).resolve()),
        "models": model_stats,
        "deltas": deltas,
        "paired": paired,
        "data_scale_conclusion": conclusion,
        "small_difference_warning": bool(small_difference),
        "recommend_extend_eval_to_300_or_500": bool(small_difference),
        "path_length_note": "Evaluator output has final_distance_xy but no path_length field; paired path_length cells are intentionally blank.",
    }
    _write_json(out_json, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    report = aggregate(
        Path(args.test_manifest),
        Path(args.results_root),
        Path(args.out_csv),
        Path(args.out_json),
    )
    print("BC_SCALE_COMPARISON=PASS")
    for name in MODEL_NAMES:
        stats = report["models"][name]
        print(
            "{} success={}/{} rate={:.6f} ci95=[{:.6f},{:.6f}] reliable_exact={}".format(
                name,
                stats["success_count"],
                stats["total_episodes"],
                stats["success_rate"],
                stats["success_ci95"][0],
                stats["success_ci95"][1],
                "PASS" if stats["reliable_exact_quality"] else "FAIL",
            )
        )
    print("DATA_SCALE_CONCLUSION={}".format(report["data_scale_conclusion"]))
    print("RECOMMEND_EXTEND_EVAL_TO_300_OR_500={}".format(
        "YES" if report["recommend_extend_eval_to_300_or_500"] else "NO"
    ))
    print("PAIRED_CSV={}".format(Path(args.out_csv).resolve()))
    print("SUMMARY_JSON={}".format(Path(args.out_json).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
