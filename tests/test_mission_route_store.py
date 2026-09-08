"""Contract tests for the bounded, immutable mission route artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from planning.data.mission_routes import (
    MISSION_ROUTE_STORE_CONTRACT_ID,
    MISSION_ROUTE_STORE_SCHEMA_VERSION,
    MissionRouteStore,
    MissionRouteStoreWriter,
    route_store_provenance,
)


def _writer(tmp_path: Path) -> MissionRouteStoreWriter:
    return MissionRouteStoreWriter(
        tmp_path / "mission_routes",
        candidate_index_sha256="a" * 64,
        global_route_contract_id="route-contract",
        route_resolution_m=0.25,
        tracking_margin_m=0.0,
        collision_map_identity={"sha256": "b" * 64},
        source_config_identity={"seed": 55},
    )


def test_route_store_round_trip_is_read_only_and_preserves_offsets(tmp_path):
    writer = _writer(tmp_path)
    first = np.asarray(
        [[0.0, 0.0, 1.5], [0.25, 0.0, 1.5], [1.0, 0.0, 1.5]],
        dtype=np.float32,
    )
    second = np.asarray(
        [[-1.0, 2.0, 1.2], [-0.5, 2.0, 1.8]],
        dtype=np.float32,
    )
    assert writer.append(first) == 0
    assert writer.append(second) == 1
    store = writer.commit()

    assert store.metadata["contract_id"] == MISSION_ROUTE_STORE_CONTRACT_ID
    assert store.metadata["schema_version"] == MISSION_ROUTE_STORE_SCHEMA_VERSION
    assert store.metadata["mission_count"] == 2
    assert store.metadata["total_route_points"] == 5
    assert store.points.dtype == np.dtype(np.float32)
    assert store.offsets.dtype == np.dtype(np.int64)
    assert not store.points.flags.writeable
    assert not store.offsets.flags.writeable
    np.testing.assert_array_equal(store.route(0), first)
    np.testing.assert_array_equal(store.route(1), second)
    np.testing.assert_array_equal(store.offsets, np.asarray([0, 3, 5], dtype=np.int64))
    store.validate()
    store.close()


def test_route_store_metadata_contains_binary_hashes_and_goal_validation(tmp_path):
    writer = _writer(tmp_path)
    route = np.asarray([[0, 0, 1.5], [1, 0, 1.5]], dtype=np.float32)
    writer.append(route)
    store = writer.commit()

    metadata = json.loads((tmp_path / "mission_routes.meta.json").read_text())
    assert metadata["points_sha256"]
    assert metadata["offsets_sha256"]
    np.testing.assert_array_equal(store.validate(0, goal=[1.0, 0.0, 1.5]), route)
    with pytest.raises(ValueError, match="endpoint"):
        store.validate(0, goal=[1.0, 0.0, 1.6])
    store.close()


@pytest.mark.parametrize(
    "route, message",
    [
        (np.asarray([0.0, 0.0, 1.0], dtype=np.float32), "Nx3"),
        (np.asarray([[0.0, 0.0, np.nan], [1.0, 0.0, 1.0]], dtype=np.float32), "finite"),
        (np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), "N >= 2"),
    ],
)
def test_route_store_rejects_invalid_routes(tmp_path, route, message):
    writer = _writer(tmp_path)
    with pytest.raises(ValueError, match=message):
        writer.append(route)
    writer.close()


def test_route_store_requires_committed_artifact_for_open(tmp_path):
    writer = _writer(tmp_path)
    writer.append(np.asarray([[0, 0, 1], [1, 0, 1]], dtype=np.float32))
    writer.commit()
    store = MissionRouteStore.open(tmp_path / "mission_routes", validate=True)
    assert store.mission_count == 1
    store.close()


def test_route_store_provenance_carries_global_route_contract(tmp_path):
    writer = _writer(tmp_path)
    writer.append(np.asarray([[0, 0, 1], [1, 0, 1]], dtype=np.float32))
    store = writer.commit()

    provenance = route_store_provenance(
        store, artifact_path=tmp_path / "rollout_index.csv"
    )
    assert provenance["global_route_contract_id"] == "route-contract"
    store.close()
