"""Canonical validation of prepared formal Teacher mission artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from planning.common import file_sha256, read_csv, read_json
from planning.data.mission_routes import (
    MissionRouteStore,
    validate_route_references,
)
from planning.mission.global_route import (
    FORMAL_GLOBAL_ROUTE_BACKEND,
    GLOBAL_ROUTE_CONTRACT_ID,
    GLOBAL_ROUTE_SOURCE_ID,
    global_route_identity,
)
from planning.mission.spec import validate_mission_rows


def _required_rows(path: Path, label: str) -> list[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("missing {}: {}".format(label, path))
    rows = [dict(row) for row in read_csv(path)]
    if not rows:
        raise ValueError("{} is empty: {}".format(label, path))
    return rows


def _unique_ids(rows: list[Mapping[str, Any]], field: str, label: str) -> None:
    values = [str(row.get(field, "")).strip() for row in rows]
    if any(not value for value in values):
        raise ValueError("{} contains an empty {}".format(label, field))
    if len(values) != len(set(values)):
        raise ValueError("{} contains duplicate {}".format(label, field))


def _episode_order(rows: list[Mapping[str, Any]], label: str) -> list[int]:
    try:
        values = [int(float(row.get("episode_id", ""))) for row in rows]
    except (TypeError, ValueError) as error:
        raise ValueError("{} has an invalid episode_id".format(label)) from error
    if values != list(range(len(values))):
        raise ValueError("{} episode order is not canonical".format(label))
    return values


def _required_count(metadata: Mapping[str, Any], field: str, label: str) -> int:
    if field not in metadata or isinstance(metadata.get(field), bool):
        raise ValueError("{} missing or invalid {}".format(label, field))
    try:
        value = int(metadata[field])
    except (TypeError, ValueError) as error:
        raise ValueError("{} has invalid {}".format(label, field)) from error
    if value < 0:
        raise ValueError("{} {} must be non-negative".format(label, field))
    return value


def _required_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("missing {}: {}".format(label, path))
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a mapping".format(label))
    return value


def validate_teacher_missions(
    *,
    candidate_index: Path,
    missions: Path,
    route_store_prefix: Path,
    expected_passing: int,
    max_steps: int,
    expected_seed: int,
) -> Dict[str, Any]:
    """Fail closed unless candidates, missions, routes, and preparation agree."""

    candidate_path = Path(candidate_index).expanduser().resolve()
    mission_path = Path(missions).expanduser().resolve()
    candidates = _required_rows(candidate_path, "candidate index")
    mission_rows = _required_rows(mission_path, "mission index")
    if int(expected_passing) <= 0:
        raise ValueError("expected_passing must be positive")
    if len(mission_rows) != int(expected_passing):
        raise ValueError(
            "mission count {} != expected {}".format(
                len(mission_rows), int(expected_passing)
            )
        )
    validate_mission_rows(
        mission_rows,
        expected_max_primitive_steps=int(max_steps),
    )
    _unique_ids(candidates, "mission_id", "candidate index")
    _unique_ids(mission_rows, "mission_id", "mission index")
    _episode_order(mission_rows, "mission index")
    candidate_route_ids = [str(row.get("global_route_index", "")).strip() for row in candidates]
    if any(not value for value in candidate_route_ids):
        raise ValueError("candidate index has a missing global_route_index")
    if len(candidate_route_ids) != len(set(candidate_route_ids)):
        raise ValueError("candidate index has duplicate global_route_index")
    candidate_by_mission = {str(row["mission_id"]): row for row in candidates}
    if any(str(row["mission_id"]) not in candidate_by_mission for row in mission_rows):
        raise ValueError("mission index references an unknown candidate mission_id")

    preparation_path = candidate_path.with_suffix(".preparation.json")
    preparation = _required_mapping(preparation_path, "preparation metadata")
    progress_path = mission_path.with_suffix(".preparation.progress.json")
    progress = _required_mapping(progress_path, "preparation progress metadata")
    if int(preparation.get("seed", -1)) != int(expected_seed):
        raise ValueError("preparation seed does not match expected seed")
    candidate_sha = file_sha256(candidate_path)
    mission_sha = file_sha256(mission_path)
    if str(preparation.get("candidate_sha256", "")) != candidate_sha:
        raise ValueError("candidate index SHA does not match preparation metadata")
    if str(preparation.get("mission_sha256", "")) != mission_sha:
        raise ValueError("mission index SHA does not match preparation metadata")
    preparation_candidate_count = _required_count(
        preparation, "candidate_count", "preparation metadata"
    )
    routed_count = _required_count(preparation, "routed_count", "preparation metadata")
    preparation_passing_count = _required_count(
        preparation, "passing_count", "preparation metadata"
    )
    rejected_route_count = _required_count(
        preparation, "rejected_route_count", "preparation metadata"
    )
    sampling_attempts = _required_count(
        preparation, "sampling_attempts", "preparation metadata"
    )
    if sampling_attempts < preparation_candidate_count:
        raise ValueError("sampling attempts are fewer than candidate count")
    if preparation_candidate_count < routed_count:
        raise ValueError("preparation candidate count is below routed count")
    if routed_count < preparation_passing_count:
        raise ValueError("preparation routed count is below passing count")
    if preparation_candidate_count - routed_count != rejected_route_count:
        raise ValueError("route rejection accounting mismatch")
    if len(candidates) != routed_count:
        raise ValueError(
            "candidate index row count {} != routed count {}".format(
                len(candidates), routed_count
            )
        )
    if preparation_passing_count != len(mission_rows):
        raise ValueError("preparation passing count mismatch")
    progress_counts = {
        field: _required_count(progress, field, "preparation progress metadata")
        for field in (
            "candidate_count",
            "routed_count",
            "audited_count",
            "passing_count",
            "rejected_route_count",
        )
    }
    if progress_counts["candidate_count"] != preparation_candidate_count:
        raise ValueError("progress candidate count mismatch")
    if progress_counts["routed_count"] != routed_count:
        raise ValueError("progress routed count mismatch")
    if progress_counts["audited_count"] != routed_count:
        raise ValueError("preparation audited count mismatch")
    if progress_counts["passing_count"] != preparation_passing_count:
        raise ValueError("progress passing count mismatch")
    if progress_counts["rejected_route_count"] != rejected_route_count:
        raise ValueError("progress route rejection count mismatch")
    if str(preparation.get("global_route_backend", "")) != FORMAL_GLOBAL_ROUTE_BACKEND:
        raise ValueError("preparation global route backend is not cpp_native")
    route_identity = global_route_identity()
    for field, expected in (
        ("global_route_contract_id", GLOBAL_ROUTE_CONTRACT_ID),
        ("global_route_source_id", GLOBAL_ROUTE_SOURCE_ID),
    ):
        if str(preparation.get(field, "")) != expected:
            raise ValueError("preparation {} mismatch".format(field))
        if str(route_identity[field]) != expected:
            raise ValueError("current global route {} mismatch".format(field))
    if int(preparation.get("generation_astar_calls", -1)) != preparation_candidate_count:
        raise ValueError("generation A* call count does not match candidates")
    if int(preparation.get("audit_astar_call_count", -1)) != 0:
        raise ValueError("mission audit must not perform a second A* call")

    store = MissionRouteStore.open(
        Path(route_store_prefix).expanduser().resolve(), validate=True
    )
    route_count = int(store.mission_count)
    try:
        if route_count != routed_count:
            raise ValueError(
                "route count {} != routed count {}".format(
                    route_count, routed_count
                )
            )
        if str(store.metadata.get("candidate_index_sha256", "")) != candidate_sha:
            raise ValueError("route store candidate index SHA mismatch")
        if str(store.metadata.get("global_route_contract_id", "")) != GLOBAL_ROUTE_CONTRACT_ID:
            raise ValueError("route store global route contract mismatch")
        validate_route_references(
            candidates,
            store,
            candidate_index_sha256=candidate_sha,
            require_provenance=False,
        )
        validate_route_references(
            mission_rows,
            store,
            candidate_index_sha256=candidate_sha,
            require_provenance=True,
        )
    finally:
        store.close()

    return {
        "mission_count": len(mission_rows),
        "candidate_count": preparation_candidate_count,
        "candidate_index_row_count": len(candidates),
        "candidate_csv_row_count": len(candidates),
        "preparation_candidate_count": preparation_candidate_count,
        "route_count": route_count,
        "routed_count": routed_count,
        "audited_count": progress_counts["audited_count"],
        "passing_count": preparation_passing_count,
        "rejected_route_count": rejected_route_count,
        "sampling_attempts": sampling_attempts,
        "candidate_sha256": candidate_sha,
        "mission_sha256": mission_sha,
        "seed": int(expected_seed),
        "task_contract_schema_version": 2,
        "max_steps": int(max_steps),
        "global_route_backend": FORMAL_GLOBAL_ROUTE_BACKEND,
        "global_route_contract_id": GLOBAL_ROUTE_CONTRACT_ID,
        "global_route_source_id": GLOBAL_ROUTE_SOURCE_ID,
        "preparation_metadata": str(preparation_path),
    }


__all__ = ["validate_teacher_missions"]
