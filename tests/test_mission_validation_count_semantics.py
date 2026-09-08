"""Regression tests for integrated mission-preparation count semantics."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from planning.common import file_sha256, write_csv_atomic, write_json_atomic
from planning.contracts.task import task_contract_fields
from planning.data.mission_routes import (
    MissionRouteStoreWriter,
    route_store_provenance,
)
from planning.mission.global_route import (
    FORMAL_GLOBAL_ROUTE_BACKEND,
    GLOBAL_ROUTE_CONTRACT_ID,
    GLOBAL_ROUTE_SOURCE_ID,
)
from planning.mission.validation import validate_teacher_missions


pytestmark = pytest.mark.unit


def _build_fixture(
    tmp_path: Path,
    *,
    preparation_candidate_count: int = 10,
    routed_count: int = 8,
    passing_count: int = 6,
    route_store_count: int = 8,
    mission_route_index_override: int | None = None,
    candidate_sha_override: str | None = None,
    preparation_passing_count: int | None = None,
) -> tuple[Path, Path, Path]:
    candidate_path = tmp_path / "mission_candidates.csv"
    mission_path = tmp_path / "missions.csv"
    route_prefix = tmp_path / "mission_routes"

    candidate_rows = []
    for index in range(routed_count):
        start_x = float(index)
        candidate_rows.append(
            {
                "episode_id": index,
                "mission_id": "mission-{}".format(index),
                "start_x": start_x,
                "start_y": 0.0,
                "start_z": 2.0,
                "goal_x": start_x + 40.0,
                "goal_y": 0.0,
                "goal_z": 2.0,
                "global_route_contract_id": GLOBAL_ROUTE_CONTRACT_ID,
                "global_route_index": index,
            }
        )
    write_csv_atomic(candidate_path, candidate_rows)
    candidate_sha = file_sha256(candidate_path)

    writer = MissionRouteStoreWriter(
        route_prefix,
        candidate_index_sha256=candidate_sha,
        global_route_contract_id=GLOBAL_ROUTE_CONTRACT_ID,
        route_resolution_m=0.25,
        tracking_margin_m=0.0,
        collision_map_identity={"sha256": "b" * 64},
        source_config_identity={"seed": 2026},
    )
    for index in range(route_store_count):
        start_x = float(index)
        writer.append(
            np.asarray(
                [[start_x, 0.0, 2.0], [start_x + 40.0, 0.0, 2.0]],
                dtype=np.float32,
            ),
            goal=[start_x + 40.0, 0.0, 2.0],
        )
    store = writer.commit()
    route_provenance = route_store_provenance(store, artifact_path=mission_path)
    store.close()

    mission_rows = []
    task_fields = task_contract_fields(45)
    for index, candidate in enumerate(candidate_rows[:passing_count]):
        row = copy.deepcopy(candidate)
        row.update(task_fields)
        row["episode_id"] = index
        if mission_route_index_override is not None and index == 0:
            row["global_route_index"] = mission_route_index_override
        row.update(route_provenance)
        mission_rows.append(row)
    write_csv_atomic(mission_path, mission_rows)

    preparation = {
        "seed": 2026,
        "candidate_sha256": candidate_sha_override or candidate_sha,
        "mission_sha256": file_sha256(mission_path),
        "candidate_count": preparation_candidate_count,
        "routed_count": routed_count,
        "passing_count": (
            preparation_passing_count
            if preparation_passing_count is not None
            else passing_count
        ),
        "rejected_route_count": preparation_candidate_count - routed_count,
        "sampling_attempts": preparation_candidate_count,
        "generation_astar_calls": preparation_candidate_count,
        "audit_astar_call_count": 0,
        "global_route_backend": FORMAL_GLOBAL_ROUTE_BACKEND,
        "global_route_contract_id": GLOBAL_ROUTE_CONTRACT_ID,
        "global_route_source_id": GLOBAL_ROUTE_SOURCE_ID,
    }
    write_json_atomic(candidate_path.with_suffix(".preparation.json"), preparation)
    write_json_atomic(
        mission_path.with_suffix(".preparation.progress.json"),
        {
            "sampling_attempts": preparation_candidate_count,
            "candidate_count": preparation_candidate_count,
            "routed_count": routed_count,
            "audited_count": routed_count,
            "passing_count": passing_count,
            "rejected_route_count": preparation_candidate_count - routed_count,
        },
    )
    return candidate_path, mission_path, route_prefix


def test_integrated_preparation_allows_candidate_count_above_routed_count(tmp_path):
    paths = _build_fixture(tmp_path)

    result = validate_teacher_missions(
        candidate_index=paths[0],
        missions=paths[1],
        route_store_prefix=paths[2],
        expected_passing=6,
        max_steps=45,
        expected_seed=2026,
    )

    assert result["preparation_candidate_count"] == 10
    assert result["candidate_index_row_count"] == 8
    assert result["candidate_csv_row_count"] == 8
    assert result["routed_count"] == 8
    assert result["audited_count"] == 8
    assert result["mission_count"] == 6


def test_route_store_count_must_match_routed_count(tmp_path):
    paths = _build_fixture(tmp_path, route_store_count=10)

    with pytest.raises(ValueError, match="route count 10 != routed count 8"):
        validate_teacher_missions(
            candidate_index=paths[0],
            missions=paths[1],
            route_store_prefix=paths[2],
            expected_passing=6,
            max_steps=45,
            expected_seed=2026,
        )


def test_final_mission_route_index_must_be_in_route_store(tmp_path):
    paths = _build_fixture(tmp_path, mission_route_index_override=8)

    with pytest.raises(IndexError, match="route index out of range"):
        validate_teacher_missions(
            candidate_index=paths[0],
            missions=paths[1],
            route_store_prefix=paths[2],
            expected_passing=6,
            max_steps=45,
            expected_seed=2026,
        )


def test_preparation_passing_count_must_match_final_missions(tmp_path):
    paths = _build_fixture(tmp_path, preparation_passing_count=7)

    with pytest.raises(ValueError, match="preparation passing count mismatch"):
        validate_teacher_missions(
            candidate_index=paths[0],
            missions=paths[1],
            route_store_prefix=paths[2],
            expected_passing=6,
            max_steps=45,
            expected_seed=2026,
        )


def test_preparation_candidate_sha_must_match_candidate_index(tmp_path):
    paths = _build_fixture(tmp_path, candidate_sha_override="0" * 64)

    with pytest.raises(ValueError, match="candidate index SHA"):
        validate_teacher_missions(
            candidate_index=paths[0],
            missions=paths[1],
            route_store_prefix=paths[2],
            expected_passing=6,
            max_steps=45,
            expected_seed=2026,
        )
