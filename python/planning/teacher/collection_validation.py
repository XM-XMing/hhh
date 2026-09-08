"""Canonical post-validation for formal reliable-exact collection artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from planning.common import file_sha256, read_csv, read_json
from planning.common.config import parse_bool
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.pipeline_provenance import (
    validate_collection_summary,
    validate_rollout_provenance,
)
from planning.contracts.task import validate_task_contract
from planning.data.mission_routes import MissionRouteStore, validate_route_references
from planning.mission.spec import validate_mission_rows


def _integer(value: Any, field: str, path: str) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError("{} invalid {}".format(path, field)) from error
    if parsed < 0:
        raise ValueError("{} negative {}".format(path, field))
    return parsed


def _required_csv(path: Path, label: str) -> list[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("missing {}: {}".format(label, path))
    rows = [dict(row) for row in read_csv(path)]
    if not rows:
        raise ValueError("{} is empty: {}".format(label, path))
    return rows


def validate_teacher_collection(
    *,
    rollout_dir: Path,
    missions: Path,
    route_store_prefix: Path,
    min_accepted: int,
    expected_workers: int,
    max_steps: int,
    expected_observation_contract: str,
    require_runtime_quality: bool = False,
) -> Dict[str, Any]:
    """Validate the merged rollout selection and all immutable identities."""

    root = Path(rollout_dir).expanduser().resolve()
    index_path = root / "rollout_index.csv"
    mission_path = Path(missions).expanduser().resolve()
    if str(expected_observation_contract).strip() != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError(
            "formal collection requires observation contract {}".format(
                EXACT_ENDPOINT_OBSERVATION_CONTRACT
            )
        )
    if int(min_accepted) <= 0 or int(expected_workers) <= 0 or int(max_steps) <= 0:
        raise ValueError("collection validation limits must be positive")

    provenance = validate_rollout_provenance(index_path)
    root_summary = validate_collection_summary(
        provenance.root_metadata, path=str(provenance.manifest_path)
    )
    if int(root_summary.get("max_steps", root_summary["max_primitive_steps"])) != int(max_steps):
        raise ValueError("collection max_steps does not match requested max_steps")
    source_accepted = _integer(
        root_summary.get("accepted_total", -1),
        "accepted_total",
        str(provenance.manifest_path),
    )
    accepted = _integer(
        root_summary.get("accepted_dataset_count", source_accepted),
        "accepted_dataset_count",
        str(provenance.manifest_path),
    )
    if accepted < int(min_accepted):
        raise ValueError(
            "accepted_dataset_count {} < minimum {}".format(
                accepted, int(min_accepted)
            )
        )
    if accepted != len(provenance.rows):
        raise ValueError("accepted_dataset_count does not match rollout index row count")
    if "accepted_dataset_quality_pass" in root_summary and not parse_bool(
        root_summary.get("accepted_dataset_quality_pass")
    ):
        raise ValueError("accepted-only dataset quality gate failed")
    for field in (
        "accepted_dataset_invalid_npz_count",
        "accepted_dataset_missing_npz_count",
        "accepted_dataset_duplicate_episode_count",
        "accepted_dataset_duplicate_transition_count",
    ):
        if field in root_summary and _integer(
            root_summary.get(field), field, str(provenance.manifest_path)
        ) != 0:
            raise ValueError("{} is non-zero".format(field))
    runtime_quality_pass = parse_bool(
        root_summary.get(
            "runtime_quality_pass", root_summary.get("quality_pass", False)
        )
    )
    if require_runtime_quality and not runtime_quality_pass:
        raise ValueError("runtime collection quality gate failed")

    mission_rows = _required_csv(mission_path, "mission index")
    validate_mission_rows(
        mission_rows,
        expected_max_primitive_steps=int(max_steps),
    )
    if str(root_summary.get("mission_index_sha256", "")) != file_sha256(mission_path):
        raise ValueError("mission index SHA does not match collection summary")
    by_episode: Dict[int, Mapping[str, Any]] = {}
    for row in mission_rows:
        episode_id = _integer(row.get("episode_id", ""), "episode_id", str(mission_path))
        if episode_id in by_episode:
            raise ValueError("mission index contains duplicate episode_id {}".format(episode_id))
        by_episode[episode_id] = row
    for row in provenance.rows:
        episode_id = _integer(row.get("episode_id", ""), "episode_id", str(index_path))
        mission = by_episode.get(episode_id)
        if mission is None:
            raise ValueError("rollout references unknown mission episode_id {}".format(episode_id))
        if str(row.get("mission_id", "")) != str(mission.get("mission_id", "")):
            raise ValueError("rollout mission_id disagrees with mission index")
        for field in (
            "global_route_index",
            "global_route_contract_id",
            "mission_route_store_contract_id",
            "mission_route_store_schema_version",
            "mission_route_store_candidate_index_sha256",
            "mission_route_store_points_sha256",
            "mission_route_store_offsets_sha256",
        ):
            if str(row.get(field, "")) != str(mission.get(field, "")):
                raise ValueError("rollout {} disagrees with mission index".format(field))
        validate_task_contract(
            row,
            expected_max_primitive_steps=int(max_steps),
            path="{} task contract".format(index_path),
        )

    worker_dirs = sorted(path for path in (root / "workers").glob("worker_*") if path.is_dir())
    worker_ids = []
    runtime_identity = root_summary["runtime_artifact_identity"]
    for worker_dir in worker_dirs:
        try:
            worker_id = int(worker_dir.name.split("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError("invalid worker directory {}".format(worker_dir)) from error
        worker_ids.append(worker_id)
        summary_path = worker_dir / "collection_summary.json"
        summary = validate_collection_summary(
            read_json(summary_path), path=str(summary_path)
        )
        for field in ("unity_player_sha256", "runtime_assembly_sha256", "bridge_sha256"):
            if str(summary.get(field, "")) != str(root_summary.get(field, "")):
                raise ValueError("worker runtime identity mismatch for {}".format(field))
            if str(summary["runtime_artifact_identity"].get(field, "")) != str(runtime_identity.get(field, "")):
                raise ValueError("worker runtime artifact identity mismatch for {}".format(field))
    if worker_ids != list(range(int(expected_workers))):
        raise ValueError(
            "worker directories {} != expected {}".format(
                worker_ids, list(range(int(expected_workers)))
            )
        )
    if int(root_summary.get("workers", -1)) != int(expected_workers):
        raise ValueError("collection summary worker count mismatch")

    route_store = MissionRouteStore.open(
        Path(route_store_prefix).expanduser().resolve(), validate=True
    )
    try:
        candidate_sha = str(route_store.metadata["candidate_index_sha256"])
        validate_route_references(
            mission_rows,
            route_store,
            candidate_index_sha256=candidate_sha,
            require_provenance=True,
        )
        validate_route_references(
            [dict(row) for row in provenance.rows],
            route_store,
            candidate_index_sha256=candidate_sha,
            require_provenance=True,
        )
        route_count = int(route_store.mission_count)
    finally:
        route_store.close()

    transition_count = sum(
        _integer(row.get("reliable_rows", ""), "reliable_rows", str(index_path))
        for row in provenance.rows
    )
    if transition_count != int(provenance.accepted_reliable_rows):
        raise ValueError("rollout index transition accounting mismatch")
    if int(root_summary.get("reliable_rows", -1)) < transition_count:
        raise ValueError("root reliable_rows is below accepted transition count")
    return {
        "accepted_count": accepted,
        "source_accepted_count": source_accepted,
        "index_row_count": len(provenance.rows),
        "reliable_rows": int(provenance.root_reliable_rows),
        "transition_count": transition_count,
        "npz_transition_count": int(provenance.accepted_reliable_rows),
        "mission_count": len(mission_rows),
        "route_count": route_count,
        "worker_count": len(worker_dirs),
        "runtime_identity": dict(runtime_identity),
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "task_contract_schema_version": int(root_summary["task_contract_schema_version"]),
        "task_contract_sha256": str(root_summary["task_contract_sha256"]),
        "max_steps": int(max_steps),
        "mission_sha256": file_sha256(mission_path),
        "rollout_index_sha256": provenance.index_sha256,
        "rollout_manifest_sha256": provenance.manifest_sha256,
        "legacy_rows": int(root_summary.get("legacy_rows", 0)),
        "telemetry_lookup_count": int(root_summary.get("telemetry_lookup_count", 0)),
        "snapshot_missing_count": int(root_summary.get("snapshot_missing_count", 0)),
        "state_depth_skew_max_ns": int(root_summary.get("state_depth_skew_max_ns", 0)),
        "frame_contract_failures": int(root_summary.get("frame_contract_failures", 0)),
        "collector_error_total": int(root_summary.get("collector_error_total", 0)),
        "source_collector_error_count": int(
            root_summary.get(
                "source_collector_error_count",
                root_summary.get("raw_collector_error_total", 0),
            )
        ),
        "runtime_quality_pass": bool(runtime_quality_pass),
        "accepted_dataset_quality_pass": bool(
            parse_bool(root_summary.get("accepted_dataset_quality_pass", True))
        ),
        "endpoint_identity_chain_valid": bool(root_summary.get("endpoint_identity_chain_valid")),
    }


__all__ = ["validate_teacher_collection"]
