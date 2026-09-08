"""Public lifecycle contracts for resumed reliable-v4 Unity workers."""

from __future__ import annotations

from pathlib import Path

import pytest

from planning.runtime.worker import build_worker_runtime_specs


pytestmark = pytest.mark.unit


def test_restarted_worker_gets_a_new_runtime_instance_id(tmp_path: Path):
    """A logical worker may restart, but its Unity incarnation may not repeat."""

    from planning.runtime.lifecycle import build_runtime_segment

    first = build_runtime_segment(
        training_run_id="p3-canonical",
        segment_id="step-009000",
        worker_specs=build_worker_runtime_specs(
            worker_count=1,
            master_port_base=11621,
            command_port_base=10553,
            depth_port_base=12554,
            port_stride=20,
            runtime_instance_template="worker-{worker_id:02d}-legacy",
            ros_root=tmp_path / "first",
        ),
        launch_nonce="launch-a",
    )
    second = build_runtime_segment(
        training_run_id="p3-canonical",
        segment_id="step-009250",
        worker_specs=build_worker_runtime_specs(
            worker_count=1,
            master_port_base=11621,
            command_port_base=10553,
            depth_port_base=12554,
            port_stride=20,
            runtime_instance_template="worker-{worker_id:02d}-legacy",
            ros_root=tmp_path / "second",
        ),
        launch_nonce="launch-b",
    )

    assert first[0].worker_id == second[0].worker_id == 0
    assert first[0].training_run_id == second[0].training_run_id == "p3-canonical"
    assert first[0].runtime_instance_id != second[0].runtime_instance_id


def _segment(tmp_path: Path, *, nonce: str, count: int = 1):
    from planning.runtime.lifecycle import build_runtime_segment

    return build_runtime_segment(
        training_run_id="p3-canonical",
        segment_id="segment-{}".format(nonce),
        worker_specs=build_worker_runtime_specs(
            worker_count=count,
            master_port_base=11621,
            command_port_base=10553,
            depth_port_base=12554,
            port_stride=20,
            runtime_instance_template="legacy-{worker_id}",
            ros_root=tmp_path / nonce,
        ),
        launch_nonce=nonce,
    )


def test_twelve_workers_get_distinct_runtime_incarnations(tmp_path: Path):
    specs = _segment(tmp_path, nonce="launch-12", count=12)

    assert [spec.worker_id for spec in specs] == list(range(12))
    assert len({spec.runtime_instance_id for spec in specs}) == 12


def test_reusing_a_prior_runtime_id_fails_before_runtime_launch(tmp_path: Path):
    from planning.runtime.lifecycle import (
        RuntimeInstanceLifecycleError,
        register_runtime_segment,
    )

    registry = tmp_path / "runtime_instance_lifecycle.json"
    first = _segment(tmp_path, nonce="launch-a")
    register_runtime_segment(
        registry,
        training_run_id="p3-canonical",
        segment_id="step-009000",
        worker_specs=first,
    )

    with pytest.raises(
        RuntimeInstanceLifecycleError,
        match="STARTUP_RUNTIME_INSTANCE_ID_REUSE",
    ):
        register_runtime_segment(
            registry,
            training_run_id="p3-canonical",
            segment_id="step-009250",
            worker_specs=first,
        )


def test_run_wide_ledger_allows_counter_restart_only_for_new_runtime(tmp_path: Path):
    from planning.runtime.lifecycle import validate_transition_ledger

    first, second, third = (
        _segment(tmp_path, nonce="a", count=2),
        _segment(tmp_path, nonce="b", count=2),
        _segment(tmp_path, nonce="c", count=2),
    )
    rows = [
        {
            "worker_id": spec.worker_id,
            "runtime_instance_id": spec.runtime_instance_id,
            "execution_id": 44000000000000,
            "command_sequence_hash": "a" * 64,
        }
        for spec in first + second + third
    ]

    report = validate_transition_ledger(rows)

    assert report["transition_count"] == 6
    assert report["unique_runtime_execution_keys"] == 6
    assert report["duplicate_different_hash_count"] == 0
    assert report["runtime_instance_count"] == 6
    assert report["runtime_segments_per_worker"] == {"0": 3, "1": 3}


def test_same_runtime_reused_execution_with_different_hash_is_rejected():
    from planning.runtime.lifecycle import (
        RuntimeInstanceLifecycleError,
        validate_transition_ledger,
    )

    with pytest.raises(RuntimeInstanceLifecycleError, match="PROTOCOL_ERROR"):
        validate_transition_ledger(
            [
                {
                    "worker_id": 0,
                    "runtime_instance_id": "runtime-a",
                    "execution_id": 7,
                    "command_sequence_hash": "a" * 64,
                },
                {
                    "worker_id": 0,
                    "runtime_instance_id": "runtime-a",
                    "execution_id": 7,
                    "command_sequence_hash": "b" * 64,
                },
            ],
            reject_conflicts=True,
        )


def test_manifest_contract_and_checkpoint_persist_runtime_lifecycle(tmp_path: Path):
    import json
    import torch

    from planning.runtime.lifecycle import (
        persist_runtime_lifecycle_metadata,
        register_runtime_segment,
    )

    specs = _segment(tmp_path, nonce="launch-a")
    lifecycle = tmp_path / "runtime_instance_lifecycle.json"
    register_runtime_segment(
        lifecycle,
        training_run_id="p3-canonical",
        segment_id="step-009000",
        worker_specs=specs,
    )
    manifest = tmp_path / "runtime_manifest.json"
    run_contract = tmp_path / "run_contract.json"
    checkpoint = tmp_path / "checkpoint.pt"
    manifest.write_text("{}", encoding="utf-8")
    run_contract.write_text("{}", encoding="utf-8")
    torch.save({"global_step": 8999}, checkpoint)

    persist_runtime_lifecycle_metadata(
        lifecycle_path=lifecycle,
        manifest_path=manifest,
        run_contract_path=run_contract,
        checkpoint_paths=(checkpoint,),
    )

    expected = json.loads(lifecycle.read_text(encoding="utf-8"))
    assert json.loads(manifest.read_text(encoding="utf-8"))["runtime_instance_lifecycle"] == expected
    assert json.loads(run_contract.read_text(encoding="utf-8"))["runtime_instance_lifecycle"] == expected
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)[
        "runtime_instance_lifecycle"
    ] == expected


def test_prior_ledger_runtime_ids_are_reserved_before_a_resume_launch(tmp_path: Path):
    import json

    from planning.runtime.lifecycle import (
        RuntimeInstanceLifecycleError,
        bootstrap_lifecycle_from_transition_ledger,
        register_runtime_segment,
    )

    ledger = tmp_path / "reliable_v4_transition_ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "worker_id": 0,
                "runtime_instance_id": "old-runtime",
                "execution_id": 44000000000000,
                "command_sequence_hash": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "runtime_instance_lifecycle.json"
    bootstrap_lifecycle_from_transition_ledger(
        registry,
        training_run_id="p3-canonical",
        ledger_path=ledger,
    )
    reused = _segment(tmp_path, nonce="fresh")[0]
    reused = reused.__class__(
        **{**reused.__dict__, "runtime_instance_id": "old-runtime"}
    )

    with pytest.raises(RuntimeInstanceLifecycleError, match="STARTUP_RUNTIME_INSTANCE_ID_REUSE"):
        register_runtime_segment(
            registry,
            training_run_id="p3-canonical",
            segment_id="step-009000",
            worker_specs=(reused,),
        )


def test_runtime_incarnation_does_not_change_resume_topology_contract():
    from planning.runtime.lifecycle import resume_topology_sha256

    first = {
        "worker_count": 1,
        "workers": [{
            "worker_id": 0,
            "ros_master_uri": "http://127.0.0.1:11621",
            "runtime_instance_id": "runtime-a",
            "reliable_command_port": 10559,
        }],
    }
    second = {
        **first,
        "workers": [{**first["workers"][0], "runtime_instance_id": "runtime-b"}],
    }

    assert resume_topology_sha256(first) == resume_topology_sha256(second)


def test_only_declared_runtime_lifecycle_migration_can_resume_canonical_checkpoint():
    from planning.runtime.lifecycle import allow_resume_lifecycle_migration

    old = {"actor_learning_starts": 9000, "source_code_sha256": "old", "worker_topology_sha256": "old-topology"}
    new = {"actor_learning_starts": 9000, "source_code_sha256": "new", "worker_topology_sha256": "new-topology"}

    assert allow_resume_lifecycle_migration(
        previous_config=old,
        current_config=new,
        previous_source_sha256="old",
        current_source_sha256="new",
        allowed_previous_source_sha256=("old",),
    )
    assert not allow_resume_lifecycle_migration(
        previous_config={**old, "actor_learning_starts": 8999},
        current_config=new,
        previous_source_sha256="old",
        current_source_sha256="new",
        allowed_previous_source_sha256=("old",),
    )

