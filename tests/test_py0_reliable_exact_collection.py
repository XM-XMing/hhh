"""PY0 RED tests for the formal reliable-exact collection contract.

These tests intentionally exercise public seams only.  They do not start ROS,
Unity, ZMQ, or a formal collection run.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    exact_endpoint_metadata,
    validate_reliable_exact_metadata,
)
from planning.teacher.collection_run import (
    CollectionResumeLedger,
    aggregate_target_reached,
    aggregate_target_status,
)
from planning.teacher.rollout_session import ReliableExactRolloutSession
from planning.runtime.identity import (
    RuntimeIdentityMismatchError,
    runtime_artifact_identity,
)


def _exact_observation(*, state_id: int, sim_time_ns: int, depth_id: str = "depth-0"):
    metadata = exact_endpoint_metadata()
    return {
        **metadata,
        "depth": np.zeros((90, 160), dtype=np.float32),
        "state": {
            "state_id": state_id,
            "sim_time_ns": sim_time_ns,
            "position": np.zeros(3, dtype=np.float32),
            "velocity": np.zeros(3, dtype=np.float32),
            "yaw": 0.0,
        },
        "sensor_time": {
            "state_stamp_ns": sim_time_ns,
            "depth_stamp_ns": sim_time_ns,
            "skew_ns": 0,
        },
        "episode_id": "episode-1",
        "reset_id": "reset-1",
        "endpoint_identity": {
            "runtime_instance_id": "runtime-1",
            "episode_id": "episode-1",
            "reset_id": "reset-1",
            "state_id": state_id,
            "depth_id": depth_id,
            "sim_time_ns": sim_time_ns,
        },
    }


def _exact_info(*, execution_id: int = 1001, frame_count: int = 25):
    reference = {
        "runtime_instance_id": "runtime-1",
        "episode_id": "episode-1",
        "reset_id": "reset-1",
        "state_id": 1,
        "depth_id": "depth-1",
        "sim_time_ns": 20,
    }
    return {
        "reliable_v4": True,
        "reliable_v4_runtime_instance_id": "runtime-1",
        "reliable_v4_execution_id": execution_id,
        "reliable_v4_observation_ref": reference,
        "telemetry_lookup_count": 0,
        "telemetry_observation": False,
        "primitive_execution": {
            "runtime_instance_id": "runtime-1",
            "execution_id": execution_id,
            "requested_frame_count": frame_count,
            "applied_frame_count": frame_count,
            "endpoint_observation_ref": reference,
        },
    }


def test_exact_metadata_is_explicit_and_fail_closed():
    metadata = exact_endpoint_metadata()
    assert metadata["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    assert validate_reliable_exact_metadata(metadata)["observation_source"] == (
        EXACT_ENDPOINT_OBSERVATION_CONTRACT
    )

    missing = copy.deepcopy(metadata)
    missing.pop("observation_contract")
    with pytest.raises(ValueError, match="contract"):
        validate_reliable_exact_metadata(missing)

    unknown = copy.deepcopy(metadata)
    unknown["observation_contract"] = "future-v9"
    with pytest.raises(ValueError, match="unknown|mismatch"):
        validate_reliable_exact_metadata(unknown)

    mismatched = copy.deepcopy(metadata)
    mismatched["observation_source"] = "legacy_async_telemetry"
    with pytest.raises(ValueError, match="mismatch"):
        validate_reliable_exact_metadata(mismatched)


def test_reliable_exact_session_maps_only_exact_endpoint_transition():
    before = _exact_observation(state_id=0, sim_time_ns=0, depth_id="depth-0")
    after = _exact_observation(state_id=1, sim_time_ns=20, depth_id="depth-1")
    session = ReliableExactRolloutSession(
        runtime_instance_id="runtime-1", depth_shape=(90, 160)
    )
    session.reset(before)
    record = session.record_transition(
        before=before,
        after=after,
        info=_exact_info(),
        action_id=7,
        previous_action=-1,
        command_frame_count=25,
    )
    assert record.transition_id == "runtime-1:1001"
    assert record.before_identity[3] == 0
    assert record.after_identity[3] == 1
    assert session.counters.reliable_rows == 1
    assert session.counters.legacy_rows == 0
    assert session.counters.telemetry_lookup_count == 0
    assert session.counters.state_depth_skew_max_ns == 0


def test_resume_allocates_transition_namespace_after_runtime_restart():
    rows = [
        {
            **exact_endpoint_metadata(),
            "episode_id": "1",
            "mission_id": "mission-1",
            "collection_run_id": "run-1",
            "execute_ok": "True",
            "runtime_instance_id": "runtime-1",
            "transition_ids": '["runtime-1:10", "runtime-1:12"]',
        }
    ]
    ledger = CollectionResumeLedger.from_rows(
        rows, runtime_instance_id="runtime-1", run_id="run-1"
    )
    assert ledger.next_transition_id_offset() == 13
    before = _exact_observation(state_id=0, sim_time_ns=0)
    after = _exact_observation(state_id=1, sim_time_ns=20, depth_id="depth-1")
    restarted_session = ReliableExactRolloutSession(
        runtime_instance_id="runtime-1",
        depth_shape=(90, 160),
        transition_id_offset=ledger.next_transition_id_offset(),
    )
    restarted_session.reset(before)
    record = restarted_session.record_transition(
        before=before,
        after=after,
        info=_exact_info(execution_id=0),
        action_id=7,
        previous_action=-1,
        command_frame_count=25,
    )
    assert record.transition_id == "runtime-1:13"


def test_reliable_exact_session_fails_closed_for_snapshot_telemetry_and_frames():
    before = _exact_observation(state_id=0, sim_time_ns=0)
    after = _exact_observation(state_id=1, sim_time_ns=20, depth_id="depth-1")
    session = ReliableExactRolloutSession(
        runtime_instance_id="runtime-1", depth_shape=(90, 160)
    )
    session.reset(before)
    with pytest.raises(ValueError, match="snapshot"):
        session.record_transition(
            before=before,
            after=None,
            info=_exact_info(),
            action_id=7,
            previous_action=-1,
            command_frame_count=25,
        )

    with pytest.raises(ValueError, match="telemetry"):
        session.record_transition(
            before=before,
            after=after,
            info=dict(_exact_info(), telemetry_lookup_count=1),
            action_id=7,
            previous_action=-1,
            command_frame_count=25,
        )

    with pytest.raises(ValueError, match="frame count"):
        session.record_transition(
            before=before,
            after=after,
            info=_exact_info(frame_count=24),
            action_id=7,
            previous_action=-1,
            command_frame_count=25,
        )
    assert session.counters.snapshot_missing_count == 1


@pytest.mark.parametrize(
    "mutator,pattern",
    [
        (lambda obs: obs.pop("observation_contract"), "contract"),
        (lambda obs: obs.__setitem__("observation_source", "legacy_async_telemetry"), "mismatch"),
        (lambda obs: obs["sensor_time"].__setitem__("skew_ns", 1), "co-timestamped|skew"),
        (lambda obs: obs["endpoint_identity"].__setitem__("runtime_instance_id", "wrong"), "identity"),
    ],
)
def test_reliable_exact_session_rejects_non_exact_transition(mutator, pattern):
    before = _exact_observation(state_id=0, sim_time_ns=0)
    after = _exact_observation(state_id=1, sim_time_ns=20, depth_id="depth-1")
    mutator(after)
    session = ReliableExactRolloutSession(
        runtime_instance_id="runtime-1", depth_shape=(90, 160)
    )
    session.reset(before)
    with pytest.raises(ValueError, match=pattern):
        session.record_transition(
            before=before,
            after=after,
            info=_exact_info(),
            action_id=7,
            previous_action=-1,
            command_frame_count=25,
        )


def test_aggregate_target_is_global_and_exhaustion_is_explicit():
    assert aggregate_target_reached(9, 10) is False
    assert aggregate_target_reached(10, 10) is True
    assert aggregate_target_reached(11, 10) is True
    assert aggregate_target_reached(3, 0) is False
    assert aggregate_target_status(10, 10, active_workers=2) == "TARGET_REACHED"
    assert aggregate_target_status(9, 10, active_workers=0) == "TARGET_UNREACHED"
    assert aggregate_target_status(9, 10, active_workers=1) == "RUNNING"


def test_resume_ledger_rejects_duplicate_or_wrong_runtime_identity():
    rows = [
        {
            **exact_endpoint_metadata(),
            "episode_id": "1",
            "mission_id": "mission-1",
            "collection_run_id": "run-1",
            "execute_ok": "True",
            "runtime_instance_id": "runtime-1",
            "transition_ids": '["runtime-1:1"]',
        }
    ]
    ledger = CollectionResumeLedger.from_rows(
        rows, runtime_instance_id="runtime-1"
    )
    assert ledger.contains_episode(1)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.add_row(rows[0])
    wrong = dict(rows[0], runtime_instance_id="runtime-2", episode_id="2")
    with pytest.raises(ValueError, match="runtime"):
        ledger.add_row(wrong)


def test_runtime_artifact_identity_is_hashed_and_fail_closed(tmp_path: Path):
    player = tmp_path / "XMflight.x86_64"
    assembly = tmp_path / "Assembly-CSharp.dll"
    bridge = tmp_path / "unity_bridge_node"
    player.write_bytes(b"player")
    assembly.write_bytes(b"assembly")
    bridge.write_bytes(b"bridge")
    identity = runtime_artifact_identity(
        player=player, assembly=assembly, bridge=bridge
    )
    assert len(identity["unity_player_sha256"]) == 64
    assert identity["runtime_assembly_sha256"] == identity["assembly_csharp_sha256"]
    assert len(identity["bridge_sha256"]) == 64

    with pytest.raises(RuntimeIdentityMismatchError, match="missing runtime artifacts"):
        runtime_artifact_identity(
            player=player, assembly=assembly, bridge=tmp_path / "missing-bridge"
        )


def test_formal_collection_source_has_no_legacy_stream_or_telemetry_path():
    root = Path(__file__).resolve().parents[1]
    collector = (
        root / "python" / "planning" / "teacher" / "rollout_collector.py"
    ).read_text(encoding="utf-8")
    parallel = (
        root / "python" / "planning" / "teacher" / "parallel_collection.py"
    ).read_text(encoding="utf-8")
    formal_source = collector.split("def _run_reliable_exact", 1)[1].split(
        "if __name__", 1
    )[0]
    assert "execute_primitive_stream(" not in formal_source
    assert "diagnostic_legacy_async" not in collector
    assert "ThreadPoolExecutor" not in collector
    assert "def _run_reliable_exact" in collector
    assert "build_collection_worker_specs(" not in parallel
    assert "CollectionWorkerRuntimeSpec" not in parallel
    assert formal_source.index("resume_ledger.add_row(outcome)") < formal_source.index(
        "report_journal.append(outcome)"
    )
