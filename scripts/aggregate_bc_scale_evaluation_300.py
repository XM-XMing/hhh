#!/usr/bin/env python3
"""Aggregate the frozen 100 + 200 mission BC closed-loop evaluations.

The utility is report-only.  It validates the two mission parts and their
reliable exact evaluator artifacts, joins the same mission IDs for the 40K
and 60K checkpoints, computes paired statistics, and writes a new report.
It never edits the source holdouts, checkpoints, or evaluator outputs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import math
import os
import statistics
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


_BASE_DIR = Path(__file__).resolve().parent
_COMMON_PATH = _BASE_DIR / "aggregate_bc_scale_evaluation.py"
_COMMON_SPEC = importlib.util.spec_from_file_location(
    "aggregate_bc_scale_evaluation_common", _COMMON_PATH
)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise RuntimeError("cannot load common evaluation aggregation helpers")
_COMMON = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON)


EXPECTED_SCHEMA = "bc_closed_loop_eval_300_manifest"
EXPECTED_OBSERVATION_CONTRACT = "reliable_exact_endpoint_snapshot"
MODEL_NAMES = ("bc40k", "bc60k")
PART_NAMES = ("old100", "new200")


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: {}".format(path))
    return value


def _read_mission_ids(path: Path) -> set[str]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty mission index: {}".format(path))
    ids = [str(row.get("mission_id", "")).strip() for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("mission IDs are not unique: {}".format(path))
    return set(ids)


def _sort_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


def _verify_part(manifest: Mapping[str, object], name: str, root: Path, expected_count: int) -> set[str]:
    parts = manifest.get("parts")
    if not isinstance(parts, Mapping) or name not in parts:
        raise ValueError("combined manifest is missing part {}".format(name))
    part = parts[name]
    if not isinstance(part, Mapping):
        raise ValueError("invalid part {}".format(name))
    mission_path = Path(root) / "missions.csv"
    artifacts = part.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("part {} is missing artifacts".format(name))
    mission_artifact = artifacts.get("missions")
    if not isinstance(mission_artifact, Mapping):
        raise ValueError("part {} is missing mission artifact identity".format(name))
    expected_sha = mission_artifact.get("sha256")
    if _COMMON.file_sha256(mission_path) != expected_sha:
        raise ValueError("part {} mission SHA mismatch".format(name))
    ids = _read_mission_ids(mission_path)
    if len(ids) != int(expected_count):
        raise ValueError("part {} mission count mismatch".format(name))
    listed_ids = {str(value) for value in part.get("mission_ids", [])}
    if ids != listed_ids:
        raise ValueError("part {} mission IDs differ from manifest".format(name))
    if int(part.get("mission_count", -1)) != int(expected_count):
        raise ValueError("part {} manifest count mismatch".format(name))
    return ids


def _row_path_length(result_dir: Path, row: Mapping[str, object]) -> float:
    raw_path = str(row.get("step_csv", "")).strip()
    if not raw_path:
        raise ValueError("evaluation row has no step CSV")
    step_path = Path(raw_path)
    if not step_path.is_absolute():
        step_path = Path(result_dir) / step_path
    if not step_path.is_file():
        raise FileNotFoundError(step_path)
    required = {
        "episode_id",
        "x_before",
        "y_before",
        "z_before",
        "x_after",
        "y_after",
        "z_after",
    }
    total = 0.0
    with step_path.open("r", newline="", encoding="utf-8") as handle:
        steps = list(csv.DictReader(handle))
    expected_episode = str(row["episode_id"])
    for step in steps:
        missing = sorted(required.difference(step))
        if missing:
            raise ValueError("{} missing step columns: {}".format(step_path, ", ".join(missing)))
        if str(step["episode_id"]) != expected_episode:
            raise ValueError("step episode identity mismatch: {}".format(step_path))
        before = tuple(float(step[field]) for field in ("x_before", "y_before", "z_before"))
        after = tuple(float(step[field]) for field in ("x_after", "y_after", "z_after"))
        segment = math.sqrt(sum((end - start) ** 2 for start, end in zip(before, after)))
        if not math.isfinite(segment):
            raise ValueError("non-finite path segment: {}".format(step_path))
        total += segment
    return total


def _failure_reason(row: Mapping[str, object]) -> str:
    if bool(row["success"]):
        return ""
    stop_reason = str(row.get("stop_reason", "")).strip().lower()
    if bool(row["collision"]) or stop_reason == "collision":
        return "collision"
    if bool(row["dead_end"]) or stop_reason == "dead_end":
        return "dead_end"
    if bool(row["timeout"]) or stop_reason == "timeout":
        return "timeout"
    if bool(row["hard_altitude"]) or stop_reason in {
        "hard_altitude",
        "altitude_violation",
    }:
        return "hard_altitude"
    return stop_reason or "other"


def _load_part_model(name: str, root: Path, mission_sha: str, expected_episodes: int) -> dict:
    return _COMMON._load_model(
        name,
        Path(root) / ("results_" + name),
        mission_sha,
        expected_episodes=expected_episodes,
    )


def _merge_model(
    name: str,
    old_root: Path,
    new_root: Path,
    old_ids: set[str],
    new_ids: set[str],
    old_sha: str,
    new_sha: str,
) -> dict:
    old = _load_part_model(name, old_root, old_sha, 100)
    new = _load_part_model(name, new_root, new_sha, 200)
    if set(old["rows_by_mission"]) != old_ids:
        raise ValueError("{} old result mission set mismatch".format(name))
    if set(new["rows_by_mission"]) != new_ids:
        raise ValueError("{} new result mission set mismatch".format(name))
    if old["checkpoint_sha256"] != new["checkpoint_sha256"]:
        raise ValueError("{} checkpoint changed between evaluation parts".format(name))

    rows = []
    for part_name, part in (("old100", old), ("new200", new)):
        for source_row in part["rows"]:
            row = dict(source_row)
            row["_test_part"] = part_name
            row["_result_dir"] = part["result_dir"]
            row["_path_length"] = _row_path_length(Path(part["result_dir"]), source_row)
            row["_failure_reason"] = _failure_reason(source_row)
            rows.append(row)
    rows_by_mission = {row["mission_id"]: row for row in rows}
    expected_ids = old_ids | new_ids
    if set(rows_by_mission) != expected_ids or len(rows) != len(expected_ids):
        raise ValueError("{} merged result mission set mismatch".format(name))
    return {
        "name": name,
        "rows": rows,
        "rows_by_mission": rows_by_mission,
        "checkpoint_sha256": old["checkpoint_sha256"],
        "source_parts": {"old100": old, "new200": new},
        "runtime": {
            "old100": old["runtime"],
            "new200": new["runtime"],
            "reliable_exact_quality": bool(
                old["runtime"]["reliable_exact_quality"]
                and new["runtime"]["reliable_exact_quality"]
            ),
        },
    }


def _counts(rows: Sequence[Mapping[str, object]]) -> dict:
    result = {
        "success": sum(bool(row["success"]) for row in rows),
        "collision": sum(bool(row["collision"]) for row in rows),
        "dead_end": sum(bool(row["dead_end"]) for row in rows),
        "timeout": sum(bool(row["timeout"]) for row in rows),
        "far": sum(bool(row["far"]) for row in rows),
        "hard_altitude": sum(bool(row["hard_altitude"]) for row in rows),
    }
    if sum(result.values()) != len(rows):
        raise ValueError("merged evaluator flags are not a partition")
    result["other_failure"] = sum(
        not bool(row["success"])
        and not bool(row["collision"])
        and not bool(row["dead_end"])
        and not bool(row["timeout"])
        and not bool(row["hard_altitude"])
        for row in rows
    )
    return result


def _model_stats(model: Mapping[str, object]) -> dict:
    rows = list(model["rows"])
    counts = _counts(rows)
    success_rows = [row for row in rows if bool(row["success"])]
    all_steps = [int(row["steps"]) for row in rows if row["steps"] is not None]
    success_steps = [int(row["steps"]) for row in success_rows if row["steps"] is not None]
    success_paths = [float(row["_path_length"]) for row in success_rows]
    total = len(rows)
    ci_low, ci_high = _COMMON.wilson_ci95(counts["success"], total)
    source_stats = {}
    for part_name, part in model["source_parts"].items():
        part_counts = part["outcomes"]
        source_stats[part_name] = {
            "attempted": len(part["rows"]),
            "success_count": part_counts["success"],
            "success_rate": part["success_rate"],
            "collision_count": part_counts["collision"],
            "dead_end_count": part_counts["dead_end"],
            "timeout_count": part_counts["timeout"],
            "hard_altitude_count": part_counts["hard_altitude"],
            "quality_gate_pass": _COMMON._bool(part["summary"].get("quality_pass")),
            "reliable_exact_quality": part["runtime"]["reliable_exact_quality"],
        }
    return {
        "attempted": total,
        "success_count": counts["success"],
        "success_rate": counts["success"] / float(total),
        "success_ci95": [ci_low, ci_high],
        "collision_count": counts["collision"],
        "dead_end_count": counts["dead_end"],
        "timeout_count": counts["timeout"],
        "hard_altitude_count": counts["hard_altitude"],
        "far_count": counts["far"],
        "other_failure_count": counts["other_failure"],
        "mean_steps_all": sum(all_steps) / float(len(all_steps)) if all_steps else None,
        "mean_steps_success": sum(success_steps) / float(len(success_steps)) if success_steps else None,
        "median_steps_success": statistics.median(success_steps) if success_steps else None,
        "mean_path_length_success": sum(success_paths) / float(len(success_paths)) if success_paths else None,
        "median_path_length_success": statistics.median(success_paths) if success_paths else None,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "reliable_exact_quality": model["runtime"]["reliable_exact_quality"],
        "source_parts": source_stats,
    }


def _failure_reason_analysis(left: Mapping[str, dict], right: Mapping[str, dict]) -> dict:
    fail_success = [
        (left[key], right[key])
        for key in left
        if not bool(left[key]["success"]) and bool(right[key]["success"])
    ]
    success_fail = [
        (left[key], right[key])
        for key in left
        if bool(left[key]["success"]) and not bool(right[key]["success"])
    ]
    both_fail = [
        (left[key], right[key])
        for key in left
        if not bool(left[key]["success"]) and not bool(right[key]["success"])
    ]

    def reason_counter(rows: Iterable[Mapping[str, object]]) -> dict:
        return dict(sorted(collections.Counter(row["_failure_reason"] for row in rows).items()))

    joint = collections.Counter(
        "{}|{}".format(left_row["_failure_reason"], right_row["_failure_reason"])
        for left_row, right_row in both_fail
    )
    return {
        "bc40k_fail_bc60k_success": {
            "count": len(fail_success),
            "bc40k_failure_reason_counts": reason_counter(row[0] for row in fail_success),
        },
        "bc40k_success_bc60k_fail": {
            "count": len(success_fail),
            "bc60k_failure_reason_counts": reason_counter(row[1] for row in success_fail),
        },
        "both_fail": {
            "count": len(both_fail),
            "joint_failure_reason_counts": dict(sorted(joint.items())),
            "bc40k_failure_reason_counts": reason_counter(row[0] for row in both_fail),
            "bc60k_failure_reason_counts": reason_counter(row[1] for row in both_fail),
        },
    }


def classify_conclusion(rate_40k: float, rate_60k: float, paired: Mapping[str, object]) -> str:
    delta = float(rate_60k) - float(rate_40k)
    p_value = float(paired["mcnemar_exact_two_sided_p"])
    left_to_right = int(paired["left_fail_right_success"])
    right_to_left = int(paired["left_success_right_fail"])
    if abs(delta) <= 0.03 and p_value >= 0.05:
        return "40K_60K_STATISTICALLY_SIMILAR_DATA_SATURATION"
    if delta >= 0.05 and p_value < 0.05 and left_to_right > right_to_left:
        return "60K_CLEARLY_BETTER"
    if delta <= -0.05 and p_value < 0.05 and right_to_left > left_to_right:
        return "40K_OUTPERFORMS_60K"
    return "INCONCLUSIVE_AT_300"


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def aggregate(
    combined_manifest_path: Path,
    old_root: Path,
    new_root: Path,
    out_csv: Path,
    out_json: Path,
) -> dict:
    combined_manifest_path = Path(combined_manifest_path).resolve()
    old_root = Path(old_root).resolve()
    new_root = Path(new_root).resolve()
    manifest = _read_json(combined_manifest_path)
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unexpected combined manifest schema")
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unexpected combined manifest schema version")
    if manifest.get("observation_contract") != EXPECTED_OBSERVATION_CONTRACT:
        raise ValueError("combined observation contract mismatch")
    combined = manifest.get("combined", {})
    if not isinstance(combined, Mapping):
        raise ValueError("combined manifest is missing combined counts")
    if int(combined.get("mission_count", -1)) != 300:
        raise ValueError("combined manifest is not a 300 mission holdout")
    if int(combined.get("old_new_duplicate_mission_count", -1)) != 0:
        raise ValueError("old/new mission overlap is nonzero")
    if int(combined.get("train_new_exact_geometry_duplicate_count", -1)) != 0:
        raise ValueError("training/new geometry overlap is nonzero")

    old_ids = _verify_part(manifest, "old100", old_root, 100)
    new_ids = _verify_part(manifest, "new200", new_root, 200)
    if old_ids & new_ids:
        raise ValueError("old and new mission IDs overlap")
    if len(old_ids | new_ids) != 300:
        raise ValueError("combined mission set does not contain 300 IDs")

    parts = manifest["parts"]
    old_sha = str(parts["old100"]["artifacts"]["missions"]["sha256"])
    new_sha = str(parts["new200"]["artifacts"]["missions"]["sha256"])
    models = {
        name: _merge_model(name, old_root, new_root, old_ids, new_ids, old_sha, new_sha)
        for name in MODEL_NAMES
    }
    mission_sets = [set(model["rows_by_mission"]) for model in models.values()]
    if mission_sets[0] != mission_sets[1]:
        raise ValueError("40K and 60K merged results are not paired on the same missions")

    output_rows = []
    for mission_id in sorted(mission_sets[0], key=_sort_key):
        left = models["bc40k"]["rows_by_mission"][mission_id]
        right = models["bc60k"]["rows_by_mission"][mission_id]
        output_rows.append(
            {
                "mission_id": mission_id,
                "test_part": left["_test_part"],
                "bc40k_episode_id": left["episode_id"],
                "bc40k_success": left["success"],
                "bc40k_failure_reason": left["_failure_reason"],
                "bc40k_steps": left["steps"],
                "bc40k_path_length": left["_path_length"],
                "bc40k_final_distance_xy": left["final_distance_xy"],
                "bc60k_episode_id": right["episode_id"],
                "bc60k_success": right["success"],
                "bc60k_failure_reason": right["_failure_reason"],
                "bc60k_steps": right["steps"],
                "bc60k_path_length": right["_path_length"],
                "bc60k_final_distance_xy": right["final_distance_xy"],
            }
        )
    fields = [
        "mission_id",
        "test_part",
        "bc40k_episode_id",
        "bc40k_success",
        "bc40k_failure_reason",
        "bc40k_steps",
        "bc40k_path_length",
        "bc40k_final_distance_xy",
        "bc60k_episode_id",
        "bc60k_success",
        "bc60k_failure_reason",
        "bc60k_steps",
        "bc60k_path_length",
        "bc60k_final_distance_xy",
    ]
    _write_csv(out_csv, output_rows, fields)

    left_by_mission = models["bc40k"]["rows_by_mission"]
    right_by_mission = models["bc60k"]["rows_by_mission"]
    paired = _COMMON._paired(left_by_mission, right_by_mission)
    paired["both_success"] = sum(
        bool(left_by_mission[key]["success"]) and bool(right_by_mission[key]["success"])
        for key in left_by_mission
    )
    paired["both_fail"] = sum(
        not bool(left_by_mission[key]["success"])
        and not bool(right_by_mission[key]["success"])
        for key in left_by_mission
    )
    model_stats = {name: _model_stats(model) for name, model in models.items()}
    rate_40k = model_stats["bc40k"]["success_rate"]
    rate_60k = model_stats["bc60k"]["success_rate"]
    conclusion = classify_conclusion(rate_40k, rate_60k, paired)
    report = {
        "schema": "bc_scale_40k_60k_closed_loop_300_summary",
        "schema_version": 1,
        "study": "BC_40K_60K_EXTEND_EVAL_TO_300_V1",
        "combined_test_manifest": str(combined_manifest_path),
        "combined_test_manifest_sha256": _COMMON.file_sha256(combined_manifest_path),
        "test_missions": 300,
        "old_test_missions": 100,
        "new_test_missions": 200,
        "mission_index_sha256": {
            "old100": old_sha,
            "new200": new_sha,
        },
        "old_new_duplicate_mission_count": 0,
        "train_test_exact_duplicate_count": int(
            combined.get("train_new_exact_geometry_duplicate_count", -1)
        ),
        "ci95_method": "Wilson score interval, z=1.959963984540054",
        "paired_csv": str(Path(out_csv).resolve()),
        "paired_csv_sha256": _COMMON.file_sha256(Path(out_csv)),
        "models": model_stats,
        "deltas": {
            "success_60k_minus_40k": rate_60k - rate_40k,
            "success_40k_to_60k": rate_60k - rate_40k,
        },
        "paired": paired,
        "failure_reason_paired": _failure_reason_analysis(left_by_mission, right_by_mission),
        "data_scale_conclusion": conclusion,
        "formal_bc_baseline": (
            "60K"
            if conclusion in {
                "40K_60K_STATISTICALLY_SIMILAR_DATA_SATURATION",
                "60K_CLEARLY_BETTER",
            }
            else "40K"
            if conclusion == "40K_OUTPERFORMS_60K"
            else "UNDECIDED"
        ),
        "recommend_extend_eval_to_500": conclusion == "INCONCLUSIVE_AT_300",
        "reliable_exact_quality_40k": model_stats["bc40k"]["reliable_exact_quality"],
        "reliable_exact_quality_60k": model_stats["bc60k"]["reliable_exact_quality"],
        "old_100_test_set_modified": False,
        "source_artifacts_modified": False,
        "path_length_definition": (
            "sum of 3D Euclidean x_before/y_before/z_before to "
            "x_after/y_after/z_after segments from each evaluator step CSV"
        ),
        "runtime_identity": manifest.get("runtime_identity", {}),
        "task_contract_sha256": manifest.get("task_contract_sha256"),
        "observation_contract": manifest.get("observation_contract"),
    }
    _COMMON._write_json(Path(out_json), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-manifest", required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    report = aggregate(
        Path(args.combined_manifest),
        Path(args.old_root),
        Path(args.new_root),
        Path(args.out_csv),
        Path(args.out_json),
    )
    print("BC_40K_60K_AGGREGATION=PASS")
    for name in MODEL_NAMES:
        stats = report["models"][name]
        print(
            "{} success={}/{} rate={:.6f} ci95=[{:.6f},{:.6f}] reliable_exact={}".format(
                name,
                stats["success_count"],
                stats["attempted"],
                stats["success_rate"],
                stats["success_ci95"][0],
                stats["success_ci95"][1],
                "PASS" if stats["reliable_exact_quality"] else "FAIL",
            )
        )
    print("DATA_SCALE_CONCLUSION={}".format(report["data_scale_conclusion"]))
    print("FORMAL_BC_BASELINE={}".format(report["formal_bc_baseline"]))
    print(
        "RECOMMEND_EXTEND_EVAL_TO_500={}".format(
            "YES" if report["recommend_extend_eval_to_500"] else "NO"
        )
    )
    print("PAIRED_CSV={}".format(Path(args.out_csv).resolve()))
    print("SUMMARY_JSON={}".format(Path(args.out_json).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
