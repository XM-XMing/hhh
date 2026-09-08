#!/usr/bin/env python3
"""Regression test for merged rollout statistics and planning contracts."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


from planning.contracts.collection import canonical_sha256
from planning.data.rollout_merge import main as merge_main


def main() -> int:
    shared_collection_config = {
        "collection_config_contract_id": (
            "teacher_rollout_collection_resolved_config"
        ),
        "resolved_cli_args": {
            "max_actual_path_length_m": 46.0,
            "max_endpoint_error_m": 0.6,
            "max_sensor_skew_ms": 80.0,
            "max_steps": 45,
            "max_stream_drift_m": 0.35,
            "num_workers": 2,
            "stream_horizon": 2,
        },
        "mission_index_sha256": "c" * 64,
        "collision_cache_sha256": "d" * 64,
        "code_version_sha256": "e" * 64,
    }
    contract = {
        "teacher_planning_contract_id": "global_route_bounded_beam",
        "teacher_config": {"beam_depth": 3, "beam_width": 8},
        "mpl_duration_s": 0.5,
        "mpl_forward_distance_m": 1.5,
        "mpl_contract_sha256": "a" * 64,
        "offline_relabel_required": True,
        "asynchronous_prefetch": True,
        "teacher_path_length_contract_id": (
            "teacher_plan44_actual46_observed_polyline"
        ),
        "max_actual_path_length_m": 46.0,
        "max_steps": 45,
        "stream_horizon": 2,
        "max_sensor_skew_ms": 80.0,
        "max_endpoint_error_m": 0.6,
        "max_stream_drift_m": 0.35,
        "collection_config_contract_id": (
            "teacher_rollout_collection_resolved_config"
        ),
        "resolved_config_sha256": canonical_sha256(
            shared_collection_config
        ),
        "mission_index_sha256": "c" * 64,
        "collision_cache_sha256": "d" * 64,
        "code_version_sha256": "e" * 64,
        "shared_collection_config": shared_collection_config,
    }
    fields = (
        "episode_id",
        "execute_ok",
        "success",
        "collision",
        "dead_end",
        "hard_altitude",
        "global_collision_mask_enabled",
        "stop_reason",
        "dataset_npz",
        "path_length_exceeded",
        "actual_path_length_m",
        "error",
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        workers = root / "workers"
        output = root / "merged"
        for worker_id in range(2):
            worker = workers / "worker_{:02d}".format(worker_id)
            worker.mkdir(parents=True)
            rows = [
                {
                    "episode_id": worker_id,
                    "execute_ok": worker_id == 0,
                    "success": worker_id == 0,
                    "collision": False,
                    "dead_end": worker_id == 1,
                    "hard_altitude": False,
                    "global_collision_mask_enabled": True,
                    "stop_reason": (
                        "success" if worker_id == 0 else "actual_dead_end"
                    ),
                    "dataset_npz": "",
                    "path_length_exceeded": False,
                    "actual_path_length_m": 45.5 if worker_id == 0 else 10.0,
                    "error": "",
                }
            ]
            if worker_id == 1:
                rows.append(
                    {
                        "episode_id": 2,
                        "execute_ok": False,
                        "success": False,
                        "collision": False,
                        "dead_end": False,
                        "hard_altitude": False,
                        "global_collision_mask_enabled": True,
                        "stop_reason": "collector_error",
                        "dataset_npz": "",
                        "path_length_exceeded": False,
                        "actual_path_length_m": "",
                        "error": (
                            "RuntimeError('no global route from (1, 2) "
                            "to (3, 4)')"
                        ),
                    }
                )
            with (worker / "collection_report.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            (worker / "collection_summary.json").write_text(
                json.dumps(
                    {
                        **contract,
                        "worker_id": worker_id,
                        "workers": 2,
                        "resolved_cli_args": {
                            "max_steps": 45,
                            "num_workers": 2,
                            "worker_id": worker_id,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (worker / "resolved_collection_config.json").write_text(
                json.dumps(
                    {
                        "collection_config_contract_id": contract[
                            "collection_config_contract_id"
                        ],
                        "resolved_config_sha256": contract[
                            "resolved_config_sha256"
                        ],
                        "shared_collection_config": contract[
                            "shared_collection_config"
                        ],
                        "resolved_cli_args": {
                            "max_steps": 45,
                            "num_workers": 2,
                            "worker_id": worker_id,
                        },
                    }
                ),
                encoding="utf-8",
            )

        old_argv = sys.argv
        try:
            sys.argv = [
                "merge_teacher_rollouts.py",
                "--workers-dir",
                str(workers),
                "--out-dir",
                str(output),
                "--expected-workers",
                "2",
            ]
            result = merge_main()
        finally:
            sys.argv = old_argv
        summary = json.loads((output / "collection_summary.json").read_text())
        quality_gate_passed = (
            summary["teacher_collection_quality_contract_id"]
            == "teacher_collection_quality_gate"
            and summary["quality_pass"]
            and all(
                gate["pass"] for gate in summary["quality_gates"].values()
            )
        )

        collision_report = workers / "worker_00" / "collection_report.csv"
        collision_rows = list(csv.DictReader(collision_report.open()))
        collision_rows[0]["collision"] = True
        with collision_report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(collision_rows)
        quality_rejected_output = root / "quality-rejected"
        try:
            sys.argv = [
                "merge_teacher_rollouts.py",
                "--workers-dir",
                str(workers),
                "--out-dir",
                str(quality_rejected_output),
                "--expected-workers",
                "2",
            ]
            quality_result = merge_main()
        finally:
            sys.argv = old_argv
        quality_rejected_summary = json.loads(
            (
                quality_rejected_output / "collection_summary.json"
            ).read_text(encoding="utf-8")
        )
        quality_failure_is_explicit = (
            quality_result == 1
            and not quality_rejected_summary["quality_pass"]
            and not quality_rejected_summary["quality_gates"][
                "collision_zero"
            ]["pass"]
            and not (
                quality_rejected_output / "rollout_index.csv"
            ).exists()
        )
        collision_rows[0]["collision"] = False
        with collision_report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(collision_rows)

        mismatched_worker = workers / "worker_01"
        mismatched_summary_path = (
            mismatched_worker / "collection_summary.json"
        )
        mismatched_config_path = (
            mismatched_worker / "resolved_collection_config.json"
        )
        mismatched_summary = json.loads(
            mismatched_summary_path.read_text(encoding="utf-8")
        )
        mismatched_config = json.loads(
            mismatched_config_path.read_text(encoding="utf-8")
        )
        mismatched_summary["resolved_config_sha256"] = "f" * 64
        mismatched_config["resolved_config_sha256"] = "f" * 64
        mismatched_summary_path.write_text(
            json.dumps(mismatched_summary), encoding="utf-8"
        )
        mismatched_config_path.write_text(
            json.dumps(mismatched_config), encoding="utf-8"
        )
        rejected_output = root / "rejected"
        mismatch_rejected = False
        try:
            sys.argv = [
                "merge_teacher_rollouts.py",
                "--workers-dir",
                str(workers),
                "--out-dir",
                str(rejected_output),
                "--expected-workers",
                "2",
            ]
            merge_main()
        except RuntimeError:
            mismatch_rejected = True
        finally:
            sys.argv = old_argv
        rejected_output_absent = not any(
            (rejected_output / name).exists()
            for name in (
                "collection_report.csv",
                "collection_summary.json",
                "rollout_index.csv",
            )
        )

    checks = {
        "merge_exit_success": result == 0,
        "accepted_count": summary["accepted_total"] == 1,
        "attempted_dead_end_count": summary["dead_end_total"] == 1,
        "legacy_route_error_is_nonfatal": (
            summary["raw_collector_error_total"] == 1
            and summary["mission_route_unavailable_total"] == 1
            and summary["collector_error_total"] == 0
        ),
        "accepted_path_max_observed": abs(
            summary["accepted_actual_path_length_max_m"] - 45.5
        )
        < 1.0e-9,
        "quality_gate_passed": quality_gate_passed,
        "quality_failure_is_explicit": quality_failure_is_explicit,
        "planning_contract_preserved": summary["teacher_planning_contract_id"]
        == contract["teacher_planning_contract_id"],
        "teacher_config_preserved": summary["teacher_config"]
        == contract["teacher_config"],
        "path_contract_preserved": summary["teacher_path_length_contract_id"]
        == contract["teacher_path_length_contract_id"],
        "resolved_config_preserved": summary["resolved_config_sha256"]
        == contract["resolved_config_sha256"],
        "input_hashes_preserved": (
            summary["mission_index_sha256"]
            == contract["mission_index_sha256"]
            and summary["collision_cache_sha256"]
            == contract["collision_cache_sha256"]
        ),
        "config_mismatch_rejected": mismatch_rejected,
        "rejected_merge_publishes_nothing": rejected_output_absent,
    }
    print("MERGE_TEACHER_ROLLOUTS_TEST")
    for name, passed in checks.items():
        print("  {}: {}".format(name, passed))
    passed = all(checks.values())
    print("RESULT={}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
