"""Regression seams for the formal collection acceptance-collapse fixes."""

from __future__ import annotations

import json

import pytest

from planning.contracts.observation import exact_endpoint_metadata
from planning.contracts.pipeline_provenance import validate_collection_row
from planning.contracts.task import task_contract_fields


def _formal_row():
    return {
        "episode_id": "1",
        "mission_id": "mission-1",
        "collection_run_id": "run-1",
        "runtime_instance_id": "runtime-1",
        "unity_player_sha256": "a" * 64,
        "runtime_assembly_sha256": "b" * 64,
        "bridge_sha256": "c" * 64,
        **exact_endpoint_metadata(),
        **task_contract_fields(),
        "reliable_rows": 0,
        "legacy_rows": 0,
        "telemetry_lookup_count": 0,
        "snapshot_missing_count": 0,
        "state_depth_skew_max_ns": 0,
        "frame_contract_failures": 0,
        "endpoint_identity_chain_valid": True,
        "transition_ids": json.dumps([]),
        "global_collision_mask_enabled": True,
        "path_length_exceeded": False,
        "success": False,
        "collision": False,
        "dead_end": False,
        "hard_altitude": False,
        "execute_ok": False,
        "async_prefetch_enabled": False,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    (
        "global_collision_mask_enabled",
        "path_length_exceeded",
        "success",
        "collision",
        "dead_end",
        "hard_altitude",
        "execute_ok",
        "async_prefetch_enabled",
    ),
)
def test_formal_collection_row_rejects_blank_required_boolean(field):
    row = _formal_row()
    row[field] = ""
    with pytest.raises(ValueError, match=field):
        validate_collection_row(row, path="worker_00 episode 1")


@pytest.mark.unit
def test_formal_report_producer_initializes_every_required_boolean_explicitly():
    from planning.teacher.rollout_collector import formal_report_boolean_defaults

    defaults = formal_report_boolean_defaults()
    assert set(defaults) == {
        "reliable_execution",
        "telemetry_observation",
        "endpoint_identity_available",
        "asynchronous_prefetch",
        "endpoint_identity_chain_valid",
        "global_collision_mask_enabled",
        "path_length_exceeded",
        "success",
        "collision",
        "dead_end",
        "hard_altitude",
        "execute_ok",
        "async_prefetch_enabled",
    }
    assert all(isinstance(value, bool) for value in defaults.values())
    assert defaults["reliable_execution"] is True
    assert defaults["global_collision_mask_enabled"] is True
    assert defaults["execute_ok"] is False
