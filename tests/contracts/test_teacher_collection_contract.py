#!/usr/bin/env python3
"""Contract checks for resolved teacher-collection configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from planning.contracts.collection import (
    COLLECTION_CONFIG_CONTRACT_ID,
    build_resolved_collection_config,
    canonical_json,
    resolve_source_tree_sha256,
    source_tree_sha256,
)


def _args(worker_id: int, out_dir: str, max_steps: int = 45):
    return argparse.Namespace(
        index="data/teach/flight/missions.csv",
        out_dir=out_dir,
        worker_id=worker_id,
        num_workers=2,
        max_episodes=0,
        target_accepted=0,
        max_steps=max_steps,
        stream_horizon=2,
        max_endpoint_error_m=0.6,
        max_sensor_skew_ms=80.0,
        collision_cache="data/map_data/forest_voxels_10cm.npz",
        stop_file="data/teach/flight/.collection_stop",
        disable_async_prefetch=False,
    )


def _build(args, root: Path):
    return build_resolved_collection_config(
        args,
        mission_index_sha256="a" * 64,
        collision_cache_sha256="b" * 64,
        code_version_sha256="c" * 64,
        teacher_config={"beam_depth": 3},
        mpl_contract_sha256="d" * 64,
        async_prefetch_enabled=True,
        package_root=root,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    previous = {
        name: os.environ.get(name)
        for name in (
            "PLANNING_COLLECTION_RUN_ID",
            "PLANNING_COLLECTION_WORKERS_REQUESTED",
            "PLANNING_COLLECTION_GLOBAL_MAX_EPISODES",
            "PLANNING_COLLECTION_GLOBAL_TARGET_ACCEPTED",
            "PLANNING_COLLECTION_CODE_SHA256",
        )
    }
    os.environ.update(
        {
            "PLANNING_COLLECTION_RUN_ID": "contract-run",
            "PLANNING_COLLECTION_WORKERS_REQUESTED": "2",
            "PLANNING_COLLECTION_GLOBAL_MAX_EPISODES": "0",
            "PLANNING_COLLECTION_GLOBAL_TARGET_ACCEPTED": "10000",
            "PLANNING_COLLECTION_CODE_SHA256": "e" * 64,
        }
    )
    try:
        worker0 = _build(_args(0, "data/teach/run/workers/worker_00"), root)
        worker1 = _build(_args(1, "data/teach/run/workers/worker_01"), root)
        changed = _build(
            _args(1, "data/teach/run/workers/worker_01", max_steps=20), root
        )
        resolved_code_hash = resolve_source_tree_sha256()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    parallel_entrypoint = (
        root / "scripts" / "collect_rollouts_parallel.py"
    ).read_text(encoding="utf-8")
    collection_config_source = (
        root / "python" / "planning" / "teacher" / "collection_config.py"
    ).read_text(encoding="utf-8")
    parallel_collection_source = (
        root / "python" / "planning" / "teacher" / "parallel_collection.py"
    ).read_text(encoding="utf-8")
    collector_source = (
        root
        / "python"
        / "planning"
        / "teacher"
        / "rollout_collector.py"
    ).read_text(encoding="utf-8")
    checks = {
        "contract_id": worker0["collection_config_contract_id"]
        == COLLECTION_CONFIG_CONTRACT_ID,
        "worker_identity_excluded": worker0["resolved_config_sha256"]
        == worker1["resolved_config_sha256"],
        "data_setting_changes_hash": worker0["resolved_config_sha256"]
        != changed["resolved_config_sha256"],
        "all_cli_args_recorded": set(worker0["resolved_cli_args"])
        == set(vars(_args(0, "data/teach/run/workers/worker_00"))),
        "resolved_cli_args_log_serializable": json.loads(
            canonical_json(worker0["resolved_cli_args"])
        )
        == worker0["resolved_cli_args"],
        "worker_runtime_recorded": worker0["worker_runtime"]["worker_id"] == 0,
        "source_tree_hash": len(source_tree_sha256(root)) == 64,
        "launcher_code_hash_reused": resolved_code_hash == "e" * 64,
        "launcher_rejects_managed_overrides": (
            "COLLECTOR_EXTRA_ARGS may not override managed option"
            in collection_config_source
            and "PROTECTED_COLLECTOR_OPTIONS" in collection_config_source
        ),
        "argparse_abbreviation_disabled": "allow_abbrev=False"
        in collector_source,
        "formal_collector_is_reliable_exact_only": (
            "diagnostic_legacy_async" not in collector_source
            and "return _run_reliable_exact(args)" in collector_source
            and "ThreadPoolExecutor" not in collector_source
        ),
        "progress_reads_atomic_json": (
            'workers_root.glob("worker_*/collection_progress.json")'
            in parallel_collection_source
            and 'workers_root.glob("worker_*/collection_report.csv")'
            in parallel_collection_source
            and "_count_progress(config.out_dir)" in parallel_collection_source
        ),
        "partial_index_resume_boundary_is_present": (
            "persist_partial_rollout_index()" in collector_source
            and "persist_partial_rollout_index(force=True)" in collector_source
        ),
        "cpu_collector_disables_gpu_sampling": "sample_gpu=False"
        in collector_source,
        "canonical_python_entrypoint": (
            "planning.teacher.parallel_collection import main" in parallel_entrypoint
            and not (root / "scripts" / "collect_rollouts_parallel.sh").exists()
        ),
    }
    print("TEACHER_COLLECTION_CONFIG_CONTRACT")
    for name, passed in checks.items():
        print("  {}: {}".format(name, passed))
    passed = all(checks.values())
    print("RESULT={}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
