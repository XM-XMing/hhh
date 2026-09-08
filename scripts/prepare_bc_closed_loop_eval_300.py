#!/usr/bin/env python3
"""Freeze a no-copy manifest for the 100 + 200 mission BC holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Set

from planning.mission.spec import mission_id_from_values


OBSERVATION_CONTRACT = "reliable_exact_endpoint_snapshot"
TASK_SHA = "2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> List[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty mission index: {}".format(path))
    return rows


def _canonical_identity(row: Mapping[str, str]) -> str:
    return mission_id_from_values(
        row["start_x"],
        row["start_y"],
        row["start_z"],
        row.get("start_yaw_deg", 0.0),
        row["goal_x"],
        row["goal_y"],
        row["goal_z"],
    )


def _mission_set(path: Path) -> Set[str]:
    rows = _read_rows(path)
    ids = [str(row.get("mission_id", "")).strip() for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("mission IDs are not nonempty and unique: {}".format(path))
    if any(str(row.get("task_contract_sha256", "")) != TASK_SHA for row in rows):
        raise ValueError("Task Contract SHA mismatch: {}".format(path))
    if any(int(float(row.get("max_primitive_steps", -1))) != 45 for row in rows):
        raise ValueError("max_steps mismatch: {}".format(path))
    return {_canonical_identity(row) for row in rows}


def _artifact(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _part(name: str, root: Path, expected_count: int, result_names: Iterable[str]) -> dict:
    root = Path(root).resolve()
    mission_path = root / "missions.csv"
    candidate_path = root / "mission_candidates.csv"
    route_meta = root / "mission_routes.meta.json"
    route_points = root / "mission_routes.points.bin"
    route_offsets = root / "mission_routes.offsets.bin"
    rows = _read_rows(mission_path)
    if len(rows) != int(expected_count):
        raise ValueError("{} mission count {} != {}".format(name, len(rows), expected_count))
    mission_ids = _mission_set(mission_path)
    result_artifacts = {}
    for result_name in result_names:
        summary = root / result_name / "summary.json"
        if summary.exists():
            result_artifacts[result_name] = _artifact(summary)
    return {
        "name": name,
        "seed": 3026 if name == "old100" else 3027,
        "mission_count": len(rows),
        "mission_ids": sorted(str(row["mission_id"]) for row in rows),
        "canonical_geometry_id_count": len(mission_ids),
        "artifacts": {
            "missions": _artifact(mission_path),
            "candidate_index": _artifact(candidate_path),
            "route_store_meta": _artifact(route_meta),
            "route_store_points": _artifact(route_points),
            "route_store_offsets": _artifact(route_offsets),
        },
        "result_summaries": result_artifacts,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--training-index", required=True)
    parser.add_argument("--old-test-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    old_root = Path(args.old_dir).expanduser().resolve()
    new_root = Path(args.new_dir).expanduser().resolve()
    training_index = Path(args.training_index).expanduser().resolve()
    old_manifest_path = Path(args.old_test_manifest).expanduser().resolve()
    old = _part("old100", old_root, 100, ("results_bc40k", "results_bc60k"))
    new = _part("new200", new_root, 200, ())
    old_geometry = set(_mission_set(old_root / "missions.csv"))
    new_geometry = set(_mission_set(new_root / "missions.csv"))
    training_geometry = _mission_set(training_index)
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_runtime = dict(old_manifest.get("runtime_identity", {}))

    combined_identity_payload = {
        "old100": old["mission_ids"],
        "new200": new["mission_ids"],
    }
    combined_identity_sha = hashlib.sha256(
        json.dumps(combined_identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    old_checkpoint_summaries = {}
    for name in ("results_bc40k", "results_bc60k"):
        summary_path = old_root / name / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint_path = (old_root / name / summary["checkpoint"]).resolve()
        if not checkpoint_path.is_file():
            checkpoint_path = Path(summary["checkpoint"]).resolve()
        old_checkpoint_summaries[name] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "selection_rule": "best_soft",
            "evaluation_summary": _artifact(summary_path),
        }

    payload = {
        "schema": "bc_closed_loop_eval_300_manifest",
        "schema_version": 1,
        "study": "BC_40K_60K_EXTEND_EVAL_TO_300_V1",
        "observation_contract": OBSERVATION_CONTRACT,
        "task_contract_schema_version": 2,
        "task_contract_sha256": TASK_SHA,
        "max_steps": 45,
        "goal_distance_m": 40.0,
        "voxel_size_m": 0.10,
        "inflate_radius_m": 0.35,
        "max_route_stretch": 1.15,
        "mission_workers": 20,
        "max_inflight_results": 40,
        "collision_threads": 1,
        "teacher_config": {
            "beam_depth": 3,
            "beam_width": 8,
            "beam_branching": 4,
            "beam_discount": 0.95,
        },
        "evaluation_contract": {
            "safety_mask": "depth",
            "execution_mode": "continuous",
            "policy_input": "depth + state + final_goal + previous_action",
            "reliable_exact": True,
            "no_global_map_or_teacher_inputs": True,
        },
        "parts": {"old100": old, "new200": new},
        "combined": {
            "mission_count": 300,
            "old100_count": 100,
            "new200_count": 200,
            "canonical_identity_sha256": combined_identity_sha,
            "old_new_duplicate_mission_count": len(old_geometry.intersection(new_geometry)),
            "train_new_exact_geometry_duplicate_count": len(training_geometry.intersection(new_geometry)),
            "no_source_artifact_copies": True,
        },
        "training_index": _artifact(training_index),
        "old_test_manifest": _artifact(old_manifest_path),
        "old_100_test_set_modified": False,
        "runtime_identity": old_runtime,
        "frozen_checkpoints": old_checkpoint_summaries,
        "expected_new_result_dirs": {
            "bc40k": str((new_root / "results_bc40k").resolve()),
            "bc60k": str((new_root / "results_bc60k").resolve()),
        },
    }
    out_path = Path(args.out).expanduser().resolve()
    _write_json(out_path, payload)
    print("COMBINED_MANIFEST={}".format(out_path))
    print("OLD_TEST_MISSIONS={}".format(old["mission_count"]))
    print("NEW_TEST_MISSIONS={}".format(new["mission_count"]))
    print("TOTAL_TEST_MISSIONS={}".format(payload["combined"]["mission_count"]))
    print("OLD_NEW_DUPLICATE_MISSION_COUNT={}".format(payload["combined"]["old_new_duplicate_mission_count"]))
    print("TRAIN_TEST_EXACT_DUPLICATE_COUNT={}".format(payload["combined"]["train_new_exact_geometry_duplicate_count"]))
    print("COMBINED_CANONICAL_IDENTITY_SHA256={}".format(combined_identity_sha))
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
