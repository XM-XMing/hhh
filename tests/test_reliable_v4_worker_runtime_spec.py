"""TDD tests for the authoritative reliable-v4 worker endpoint spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from planning.runtime.worker import (
    WorkerEndpointMismatchError,
    WorkerRuntimeSpec,
    build_worker_runtime_specs,
)


pytestmark = pytest.mark.unit


def _specs(count: int = 1):
    return build_worker_runtime_specs(
        worker_count=count,
        master_port_base=11621,
        command_port_base=10553,
        depth_port_base=12554,
        port_stride=20,
        runtime_instance_template="worker-{worker_id:02d}-awac-p3",
        ros_root=Path("/tmp/p3-worker-spec-tests"),
    )


def test_worker_zero_has_the_frozen_python_and_unity_port_directions():
    spec = _specs()[0]

    assert spec.worker_id == 0
    assert spec.command_port == 10553
    assert spec.state_port == 10554
    assert spec.depth_port == 12554
    assert spec.python_command_port == 10559
    assert spec.unity_command_port == 10560
    assert spec.unity_result_port == 10555
    assert spec.python_result_port == 10556
    assert spec.unity_snapshot_port == 10557
    assert spec.python_snapshot_port == 10558
    assert spec.to_mapping()["reliable_command_port"] == spec.python_command_port
    assert spec.to_mapping()["reliable_result_port"] == spec.python_result_port
    assert spec.to_mapping()["reliable_snapshot_port"] == spec.python_snapshot_port
    assert spec.python_reset_endpoint == spec.python_command_endpoint
    assert spec.unity_reset_endpoint == spec.unity_command_endpoint


def test_worker_n_mapping_is_deterministic_and_stride_bound():
    spec = _specs(12)[7]

    assert spec.worker_id == 7
    assert spec.command_port == 10553 + 7 * 20
    assert spec.python_command_port == 10559 + 7 * 20
    assert spec.unity_command_port == 10560 + 7 * 20
    assert spec.unity_result_port == 10555 + 7 * 20
    assert spec.python_result_port == 10556 + 7 * 20
    assert spec.unity_snapshot_port == 10557 + 7 * 20
    assert spec.python_snapshot_port == 10558 + 7 * 20


def test_twelve_workers_have_no_port_collisions():
    specs = _specs(12)
    ports = [port for spec in specs for port in spec.all_ports]

    assert len(ports) == len(set(ports))
    assert [spec.worker_id for spec in specs] == list(range(12))


def test_twenty_four_managed_workers_have_no_port_collisions():
    specs = _specs(24)
    ports = [port for spec in specs for port in spec.all_ports]

    assert len(ports) == len(set(ports))
    assert [spec.worker_id for spec in specs] == list(range(24))
    assert len({spec.runtime_instance_id for spec in specs}) == 24


def test_managed_worker_count_above_explicit_collection_bound_fails_closed():
    with pytest.raises(
        WorkerEndpointMismatchError,
        match=r"worker_count must be in \[1,24\]",
    ):
        _specs(25)


def test_generated_spec_round_trips_bridge_unity_and_python_arguments():
    spec = _specs()[0]
    bridge = spec.bridge_launch_args()
    unity = spec.unity_launch_args()
    python = spec.python_backend_config()

    assert bridge["python_command_port"] == spec.python_command_port
    assert bridge["unity_command_port"] == spec.unity_command_port
    assert bridge["execution_result_port"] == spec.unity_result_port
    assert bridge["python_result_port"] == spec.python_result_port
    assert bridge["observation_snapshot_port"] == spec.unity_snapshot_port
    assert bridge["python_snapshot_port"] == spec.python_snapshot_port
    assert unity["reliableCommandPort"] == spec.unity_command_port
    assert unity["executionResultPort"] == spec.unity_result_port
    assert unity["observationSnapshotPort"] == spec.unity_snapshot_port
    assert python["command_endpoint"] == spec.python_command_endpoint
    assert python["result_endpoint"] == spec.python_result_endpoint
    assert python["snapshot_endpoint"] == spec.python_snapshot_endpoint

    restored = WorkerRuntimeSpec.from_mapping(spec.to_mapping())
    assert restored.to_mapping() == spec.to_mapping()


def test_wrong_python_facing_port_fails_before_process_start():
    mapping = _specs()[0].to_mapping()
    mapping["python_command_port"] += 1

    with pytest.raises(
        WorkerEndpointMismatchError,
        match="STARTUP_WORKER_ENDPOINT_MISMATCH",
    ):
        WorkerRuntimeSpec.from_mapping(mapping)


def test_wrong_legacy_reliable_alias_fails_before_process_start():
    mapping = _specs()[0].to_mapping()
    mapping["reliable_result_port"] += 1

    with pytest.raises(
        WorkerEndpointMismatchError,
        match="STARTUP_WORKER_ENDPOINT_MISMATCH",
    ):
        WorkerRuntimeSpec.from_mapping(mapping)
