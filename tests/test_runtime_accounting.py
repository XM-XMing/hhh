"""P3-A.1 public seams for critic-start scheduling and runtime evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_critic_start_boundary_is_explicit():
    from planning.awac.optimization import critic_update_budget_increment

    kwargs = {
        "learning_starts": 5000,
        "updates_per_step": 0.50,
    }
    assert critic_update_budget_increment(
        replay_total_before=4998, replay_total_after=4999, **kwargs
    ) == pytest.approx(0.0)
    assert critic_update_budget_increment(
        replay_total_before=4999, replay_total_after=5000, **kwargs
    ) == pytest.approx(0.0)
    assert critic_update_budget_increment(
        replay_total_before=5000, replay_total_after=5001, **kwargs
    ) == pytest.approx(1.0)


def test_first_critic_update_evidence_is_recorded_once():
    from planning.awac.optimization import first_critic_update_evidence

    assert first_critic_update_evidence(
        global_step=5000,
        critic_update_step=0,
        metrics={},
    ) is None

    metrics = {
        "critic_loss": 1.25,
        "target_q_mean": 0.5,
        "target_q_std": 0.25,
        "q_mean": 0.4,
        "q_std": 0.2,
        "critic_gradient_norm": 2.0,
        "critic_parameter_delta_norm": 0.01,
        "parameters_finite": 1.0,
    }
    assert first_critic_update_evidence(
        global_step=5001,
        critic_update_step=1,
        metrics=metrics,
    ) == {
        "global_step": 5001,
        "critic_update_step": 1,
        **metrics,
    }


def test_runtime_accounting_uses_authoritative_unity_physical_count(tmp_path):
    from planning.diagnostics.runtime_accounting import aggregate_runtime_accounting

    bridge = {
        "command_accepted_total": 2,
        "command_forward_total": 2,
        "unity_command_receipt_total": 2,
        "command_retry_total": 1,
        "command_conflict_total": 0,
        "pending_command_final": 0,
        "accepted": 2,
        "ack": 2,
        "commit": 2,
        "pending": 0,
        "pending_receipt": 0,
        "snapshot_request": 2,
        "snapshot_response": 2,
        "snapshot_hash_match": 2,
        "snapshot_cache_hit": 2,
        "snapshot_missing": 0,
        "pending_snapshot": 0,
        "protocol_error": 0,
        "command_protocol_error_total": 0,
    }
    unity = {"physical_execution_total": 2}

    accounting = aggregate_runtime_accounting(
        bridge_metrics=[bridge],
        unity_audits=[unity],
        telemetry_lookup_count=0,
    )

    assert accounting["command_accepted_total"] == 2
    assert accounting["unity_command_receipt_total"] == 2
    assert accounting["physical_execution_total"] == 2
    assert accounting["result_accepted_total"] == 2
    assert accounting["result_ack_total"] == 2
    assert accounting["result_commit_total"] == 2
    assert accounting["snapshot_request_total"] == 2
    assert accounting["pending_command_final"] == 0
    assert accounting["pending_result_final"] == 0
    assert accounting["pending_snapshot_final"] == 0
    assert accounting["telemetry_lookup_count"] == 0


def test_runtime_accounting_does_not_infer_physical_execution_from_sends():
    from planning.diagnostics.runtime_accounting import aggregate_runtime_accounting

    accounting = aggregate_runtime_accounting(
        bridge_metrics=[
            {
                "command_accepted_total": 3,
                "unity_command_receipt_total": 3,
                "accepted": 3,
                "ack": 3,
                "commit": 3,
            }
        ],
        unity_audits=[],
        telemetry_lookup_count=0,
    )

    assert accounting["command_accepted_total"] == 3
    assert accounting["physical_execution_total"] is None


def test_training_artifact_merge_persists_accounting_in_summary_and_checkpoint(
    tmp_path,
):
    from planning.diagnostics.runtime_accounting import merge_training_artifacts

    summary_path = tmp_path / "summary.json"
    checkpoint_path = tmp_path / "checkpoint_latest.pt"
    summary_path.write_text(json.dumps({"global_step": 5001}), encoding="utf-8")

    # The implementation must preserve the opaque checkpoint payload while
    # adding the accounting namespace.  This tiny JSON stand-in keeps the seam
    # CPU-only and independent of torch serialization details.
    checkpoint_path.write_text(json.dumps({"global_step": 5001}), encoding="utf-8")
    accounting = {"physical_execution_total": 1, "protocol_error_total": 0}

    merge_training_artifacts(
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
        accounting=accounting,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert summary["runtime_accounting"] == accounting
    assert checkpoint["runtime_accounting"] == accounting


def test_reliable_training_launch_wires_bridge_result_metrics_path():
    launch_dir = Path(__file__).resolve().parents[1] / "launch"
    sim_realtime = (launch_dir / "sim_realtime.launch").read_text(encoding="utf-8")
    unity_bridge = (launch_dir / "unity_bridge.launch").read_text(encoding="utf-8")

    assert 'arg name="execution_result_metrics_path"' in sim_realtime
    assert (
        'arg name="execution_result_metrics_path" '
        'value="$(arg execution_result_metrics_path)"' in sim_realtime
    )
    assert 'arg name="execution_result_metrics_path"' in unity_bridge
    assert 'param name="execution_result_metrics_path"' in unity_bridge
