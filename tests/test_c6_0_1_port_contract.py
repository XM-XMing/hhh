"""C6.0.1 port-owner and fail-closed contract tests.

These tests pin the final port seam owned by ``planning.runtime.ports``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planning.runtime.ports import (
    DirectRuntimePortProfile,
    PortContractError,
    WorkerRuntimeSpec,
    build_worker_runtime_specs,
    resolve_runtime_port_profile,
    validate_port_profiles,
)


pytestmark = pytest.mark.unit


def _managed_specs(count: int = 1):
    return build_worker_runtime_specs(
        worker_count=count,
        master_port_base=11621,
        command_port_base=10553,
        depth_port_base=12554,
        port_stride=20,
        runtime_instance_template="worker-{worker_id:02d}-c6-0-1",
        ros_root=Path("/tmp/xm-c6-0-1-workers"),
        training_run_id="c6-0-1",
        runtime_launch_nonce="test",
    )


def _direct_profile() -> DirectRuntimePortProfile:
    return DirectRuntimePortProfile(
        profile="direct",
        runtime_instance_id="direct-c6-0-1",
        master_port=11691,
        command_port=10253,
        state_port=10254,
        depth_port=11254,
        python_command_port=10259,
        unity_command_port=10260,
        unity_result_port=10255,
        python_result_port=10256,
        unity_snapshot_port=10257,
        python_snapshot_port=10258,
    )


def test_worker_spec_is_the_exact_source_for_all_process_projections():
    spec = _managed_specs()[0]

    assert spec.unity_launch_argv() == (
        "-cmdSubPort", "10553",
        "-statePubPort", "10554",
        "-depthPubPort", "12554",
        "-executionResultPort", "10555",
        "-observationSnapshotPort", "10557",
        "-reliableCommandPort", "10560",
        "-runtimeInstanceId", spec.runtime_instance_id,
    )
    assert spec.bridge_launch_args() == {
        "cmd_port": 10553,
        "state_port": 10554,
        "depth_port": 12554,
        "python_command_port": 10559,
        "unity_command_port": 10560,
        "command_runtime_instance_id": spec.runtime_instance_id,
        "execution_result_port": 10555,
        "observation_snapshot_port": 10557,
        "python_result_port": 10556,
        "python_snapshot_port": 10558,
    }
    assert spec.python_backend_config()["endpoints"] == {
        "command": "tcp://127.0.0.1:10559",
        "result": "tcp://127.0.0.1:10556",
        "snapshot": "tcp://127.0.0.1:10558",
        "reset": "tcp://127.0.0.1:10559",
    }


def test_managed_profile_round_trip_keeps_all_ports_consistent():
    spec = _managed_specs()[0]
    mapping = {"profile": "managed", **spec.to_mapping()}

    resolved = resolve_runtime_port_profile(mapping, mode="managed")

    assert isinstance(resolved, WorkerRuntimeSpec)
    assert resolved.port_mapping() == spec.port_mapping()
    assert resolved.to_mapping()["profile"] == "managed"


def test_direct_profile_round_trip_has_one_explicit_port_owner():
    profile = _direct_profile()
    resolved = resolve_runtime_port_profile(
        profile.to_mapping(), mode="direct"
    )

    assert isinstance(resolved, DirectRuntimePortProfile)
    assert resolved.port_mapping() == profile.port_mapping()
    assert resolved.unity_launch_args()["depthPubPort"] == 11254
    assert resolved.bridge_launch_args()["depth_port"] == 11254
    assert resolved.python_backend_config()["endpoints"]["command"] == (
        "tcp://127.0.0.1:10259"
    )


def test_missing_required_managed_port_fails_closed_before_startup():
    mapping = {"profile": "managed", **_managed_specs()[0].to_mapping()}
    del mapping["python_snapshot_port"]

    with pytest.raises(PortContractError, match="required port"):
        resolve_runtime_port_profile(mapping, mode="managed")


def test_unknown_or_mixed_profile_fails_closed():
    mapping = {"profile": "managed", **_managed_specs()[0].to_mapping()}
    mapping["profile"] = "direct"

    with pytest.raises(PortContractError, match="profile"):
        resolve_runtime_port_profile(mapping, mode="managed")

    direct = _direct_profile().to_mapping()
    direct["profile"] = "managed"
    with pytest.raises(PortContractError, match="profile"):
        resolve_runtime_port_profile(direct, mode="direct")

    missing = {key: value for key, value in mapping.items() if key != "profile"}
    with pytest.raises(PortContractError, match="profile"):
        resolve_runtime_port_profile(missing, mode="managed")


def test_duplicate_ports_are_rejected_before_startup():
    first = _direct_profile()
    with pytest.raises(PortContractError, match="collision"):
        DirectRuntimePortProfile(
            **{
                **first.__dict__,
                "runtime_instance_id": "direct-c6-0-1-b",
                "depth_port": first.command_port,
            }
        )

    second = DirectRuntimePortProfile(
        profile="direct",
        runtime_instance_id="direct-c6-0-1-b",
        master_port=first.command_port,
        command_port=12691,
        state_port=12692,
        depth_port=12693,
        python_command_port=12694,
        unity_command_port=12695,
        unity_result_port=12696,
        python_result_port=12697,
        unity_snapshot_port=12698,
        python_snapshot_port=12699,
    )
    with pytest.raises(PortContractError, match="collision"):
        validate_port_profiles((first, second))


def test_workers_zero_through_eleven_have_no_cross_profile_collisions():
    specs = _managed_specs(12)

    assert validate_port_profiles(specs) == tuple(specs)
    ports = [port for spec in specs for port in spec.all_ports]
    assert len(ports) == len(set(ports))


def test_bridge_source_has_no_reliable_port_derivation_fallback():
    source = (Path(__file__).resolve().parents[1] / "src/bridge/bridge_node.cpp").read_text(
        encoding="utf-8"
    )

    assert "transport_config_.command_port + 2" not in source
    assert "transport_config_.command_port + 3" not in source
    assert "transport_config_.command_port + 6" not in source
    assert "transport_config_.command_port + 7" not in source
    assert "execution_result_port + 1" not in source
    assert "python_result_port + 1" not in source
