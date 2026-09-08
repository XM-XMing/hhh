"""P0-M2 validation-runner contract tests."""

from __future__ import annotations

import json
import socket
import threading

import pytest
import zmq


def _result(execution_id, result_payload_hash=None):
    return {
        "execution_id": execution_id,
        "runtime_instance_id": "worker-00-p0-m2",
        "result_payload_hash": result_payload_hash
        or "{:064x}".format(execution_id),
        "status": "COMPLETE",
    }


def _dry_record():
    from planning.diagnostics.reliable_single_worker import build_dry_run_record

    return build_dry_run_record(
        1,
        runtime_instance_id="worker-00-p0-m2-dry-run",
    )

def test_dry_run_emits_one_v4_ledger_record_and_summary(tmp_path):
    from planning.diagnostics.reliable_single_worker import run_dry_run

    summary = run_dry_run(
        out_dir=tmp_path,
        primitive_count=1,
        checkpoint_interval=1,
        runtime_instance_id="worker-00-p0-m2-dry-run",
    )

    assert summary["execution_total"] == 1
    assert summary["complete_total"] == 1
    assert summary["transition_commit_total"] == 1
    assert summary["replay_append_total"] == 1
    assert summary["pending_result_final"] == 0
    assert (tmp_path / "ledger.jsonl").is_file()
    assert (tmp_path / "checkpoints" / "checkpoint_000001.json").is_file()


def _install_fake_real_runtime(monkeypatch, *, mode="success"):
    """Exercise run_real artifact finalization without opening processes."""

    from planning.diagnostics import reliable_single_worker as runner

    class FakeRealRuntime:
        def __init__(self, *, out_dir, runtime_instance_id, **_kwargs):
            self.out_dir = out_dir
            self.runtime_instance_id = runtime_instance_id
            self.metrics_path = self.out_dir / "bridge_result_metrics.json"
            self.ports = {"fake": 1}
            # Keep the fake's constructor contract aligned with the real
            # runtime: run_real persists the allocated port profile in its
            # manifest even when process startup is replaced by this fixture.
            self.port_profile = _kwargs.get("port_profile")
            self.startup_state = ["FAKE_READY"]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            if mode == "final_accounting_error":
                self.metrics_path.write_text("not json", encoding="utf-8")
            else:
                self.metrics_path.write_text(
                    json.dumps(
                        {
                            "accepted": 1,
                            "ack": 1,
                            "commit": 1,
                            "pending": 0,
                            "snapshot_request": 1,
                            "snapshot_response": 1,
                            "snapshot_hash_match": 1,
                            "snapshot_cache_hit": 1,
                            "snapshot_missing": 0,
                            "pending_snapshot": 0,
                            "duplicate": 0,
                            "protocol_error": 0,
                            "command_accepted_total": 1,
                            "command_forward_total": 1,
                            "unity_command_receipt_total": 1,
                            "command_duplicate_total": 0,
                            "command_receipt_duplicate_total": 0,
                            "command_retry_total": 0,
                            "command_conflict_total": 0,
                            "command_protocol_error_total": 0,
                            "pending_command_final": 0,
                        }
                    ),
                    encoding="utf-8",
                )

        def execute_primitive(self, primitive_index):
            if mode == "execution_error":
                raise RuntimeError("synthetic execution failure")
            return runner.build_dry_run_record(
                primitive_index,
                runtime_instance_id=self.runtime_instance_id,
            )

    monkeypatch.setattr(runner, "_RealRuntime", FakeRealRuntime)
    monkeypatch.setattr(
        runner,
        "_runtime_manifest",
        lambda **_kwargs: {"contract_id": runner.RUNNER_CONTRACT_ID, "mode": "real"},
    )
    return runner


def test_real_runner_success_always_writes_completed_summary(tmp_path, monkeypatch):
    runner = _install_fake_real_runtime(monkeypatch)

    summary = runner.run_real(
        out_dir=tmp_path,
        unity_binary=tmp_path / "unity",
        bridge_binary=tmp_path / "bridge",
        primitive_count=1,
        checkpoint_interval=1,
    )

    summary_path = tmp_path / "summary.json"
    assert summary_path.is_file()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["validation_status"] == "COMPLETED"
    assert summary["execution_total"] == 1


def test_real_runner_exposes_command_accounting_metrics(tmp_path, monkeypatch):
    runner = _install_fake_real_runtime(monkeypatch)

    summary = runner.run_real(
        out_dir=tmp_path,
        unity_binary=tmp_path / "unity",
        bridge_binary=tmp_path / "bridge",
        primitive_count=1,
        checkpoint_interval=1,
    )

    assert summary["command_accepted_total"] == 1
    assert summary["unity_command_receipt_total"] == 1
    assert summary["physical_execution_total"] == 0
    assert summary["pending_command_final"] == 0


def test_real_runner_execution_failure_writes_failed_summary(tmp_path, monkeypatch):
    runner = _install_fake_real_runtime(monkeypatch, mode="execution_error")

    with pytest.raises(RuntimeError, match="synthetic execution failure"):
        runner.run_real(
            out_dir=tmp_path,
            unity_binary=tmp_path / "unity",
            bridge_binary=tmp_path / "bridge",
            primitive_count=1,
            checkpoint_interval=1,
        )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["validation_status"] == "FAILED"
    assert summary["failure_stage"] == "RUNTIME_EXECUTION"
    assert summary["execution_total"] == 0


def test_real_runner_final_accounting_error_writes_partial_summary(tmp_path, monkeypatch):
    runner = _install_fake_real_runtime(monkeypatch, mode="final_accounting_error")

    with pytest.raises(json.JSONDecodeError):
        runner.run_real(
            out_dir=tmp_path,
            unity_binary=tmp_path / "unity",
            bridge_binary=tmp_path / "bridge",
            primitive_count=1,
            checkpoint_interval=1,
        )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["validation_status"] == "PARTIAL"
    assert summary["failure_stage"] == "FINAL_ACCOUNTING"
    assert summary["execution_total"] == 1


def test_ledger_finalization_is_idempotent_after_summary_is_written(tmp_path):
    from planning.diagnostics.reliable_single_worker import ValidationLedger

    ledger = ValidationLedger(
        tmp_path,
        checkpoint_interval=1,
        mode="real",
        runtime_instance_id="worker-finalize-test",
    )
    ledger.append(
        _dry_record()
    )

    first = ledger.close(extra_summary={"validation_status": "COMPLETED"})
    second = ledger.close(extra_summary={"validation_status": "FAILED"})

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert first == second
    assert summary["validation_status"] == "COMPLETED"


def test_complete_record_cannot_claim_24_applied_frames():
    from planning.diagnostics.reliable_single_worker import validate_ledger_record

    record = _dry_record()
    record["applied_frame_count"] = 24
    record["last_applied_frame_index"] = 23

    with pytest.raises(ValueError, match="COMPLETE requires 25"):
        validate_ledger_record(record)


def test_checkpoint_has_required_progress_and_memory_fields(tmp_path):
    from planning.diagnostics.reliable_single_worker import run_dry_run

    run_dry_run(
        out_dir=tmp_path,
        primitive_count=2,
        checkpoint_interval=2,
    )

    checkpoint = json.loads(
        (tmp_path / "checkpoints" / "checkpoint_000002.json").read_text()
    )
    assert checkpoint["processed"] == 2
    assert checkpoint["success"] == 2
    assert checkpoint["failure"] == 0
    assert checkpoint["pending"] == 0
    assert isinstance(checkpoint["memory_usage_bytes"], int)
    assert checkpoint["metrics"]["transition_commit_total"] == 2


def test_ledger_jsonl_contains_one_schema_valid_record(tmp_path):
    from planning.diagnostics.reliable_single_worker import (
        run_dry_run,
        validate_ledger_record,
    )

    run_dry_run(out_dir=tmp_path, primitive_count=1)
    rows = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]

    assert len(rows) == 1
    validate_ledger_record(rows[0])


def test_default_checkpoint_contract_emits_each_thousand(tmp_path):
    from planning.diagnostics.reliable_single_worker import run_dry_run

    run_dry_run(out_dir=tmp_path, primitive_count=2000)

    assert (tmp_path / "checkpoints" / "checkpoint_001000.json").is_file()
    assert (tmp_path / "checkpoints" / "checkpoint_002000.json").is_file()


def test_out_of_order_results_are_demultiplexed_by_execution_id():
    from planning.diagnostics.reliable_single_worker import ExecutionResultBuffer

    buffer = ExecutionResultBuffer()
    result_b = _result(202)

    assert buffer.offer(result_b) == "BUFFERED"
    assert buffer.wait_for_result(101) is None
    assert buffer.wait_for_result(202) == result_b


def test_result_for_b_received_while_waiting_for_a_is_retained():
    from planning.diagnostics.reliable_single_worker import ExecutionResultBuffer

    buffer = ExecutionResultBuffer()
    result_b = _result(202)

    buffer.offer(result_b)

    assert buffer.wait_for_result(101) is None
    assert buffer.wait_for_result(202) == result_b
    assert buffer.pending_execution_ids == ()


def test_duplicate_result_does_not_create_a_second_buffered_delivery():
    from planning.diagnostics.reliable_single_worker import ExecutionResultBuffer

    buffer = ExecutionResultBuffer()
    result_b = _result(202)

    assert buffer.offer(result_b) == "BUFFERED"
    assert buffer.offer(dict(result_b)) == "DUPLICATE"
    assert buffer.pending_execution_ids == (202,)
    assert buffer.wait_for_result(202) == result_b
    assert buffer.wait_for_result(202) is None


def test_timeout_for_unknown_execution_does_not_corrupt_buffered_results():
    from planning.diagnostics.reliable_single_worker import ExecutionResultBuffer

    buffer = ExecutionResultBuffer()
    result_b = _result(202)
    buffer.offer(result_b)

    assert buffer.wait_for_result(999) is None
    assert buffer.pending_execution_ids == (202,)
    assert buffer.wait_for_result(202) == result_b


def test_endpoint_observation_diagnostics_records_result_and_lookup_timing(tmp_path):
    from planning.diagnostics.reliable_single_worker import (
        EndpointObservationLifecycleDiagnostics,
    )

    diagnostics = EndpointObservationLifecycleDiagnostics(
        tmp_path / "endpoint_observation_diagnostics.jsonl"
    )
    diagnostics.record_execution_started(54)
    diagnostics.record_result_received(
        54,
        {
            "endpoint_state_id": 1518,
            "endpoint_observation_ref": {"capture_id": "depth-1518"},
        },
    )
    diagnostics.record_store_event(
        54,
        "LOOKUP_MISS",
        endpoint_state_id=1518,
        depth_id="depth-1518",
        reason="STATE_MISSING",
    )
    diagnostics.close()

    rows = [
        json.loads(line)
        for line in (tmp_path / "endpoint_observation_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "EXECUTION_STARTED",
        "RESULT_RECEIVED",
        "LOOKUP_MISS",
    ]
    assert rows[1]["execution_id"] == 54
    assert rows[1]["endpoint_state_id"] == 1518
    assert rows[1]["depth_id"] == "depth-1518"
    assert rows[2]["lookup_result"] == "MISS"
    assert rows[2]["lookup_time_ns"] >= rows[1]["result_receive_time_ns"]


def test_handoff_diagnostics_records_target_execution_and_classifies_socket_loss(
    tmp_path,
):
    from planning.diagnostics.reliable_single_worker import (
        ExecutionHandoffDiagnostics,
    )

    diagnostics = ExecutionHandoffDiagnostics(
        tmp_path / "handoff.jsonl", target_execution_id=54
    )
    diagnostics.record_socket_recv(
        wait_execution_id=53,
        raw_message=b"result-54",
        decoded_execution_id=54,
        decoded_message={"status": "COMPLETE"},
    )
    diagnostics.record("WAIT", "timeout", execution_id=54)
    summary = diagnostics.close()

    assert summary["first_loss_layer"] == "PYTHON_WAITER_LOSS"
    rows = [
        json.loads(line)
        for line in (tmp_path / "handoff.jsonl").read_text().splitlines()
    ]
    assert rows[0]["event"] == "RECV"
    assert rows[0]["decoded_execution_id"] == 54
    assert rows[0]["raw_message_hex"] == b"result-54".hex()
    assert (tmp_path / "handoff.summary.json").is_file()


def test_handoff_diagnostics_classifies_buffer_loss_after_socket_receive(tmp_path):
    from planning.diagnostics.reliable_single_worker import (
        ExecutionHandoffDiagnostics,
    )

    diagnostics = ExecutionHandoffDiagnostics(
        tmp_path / "handoff.jsonl", target_execution_id=54
    )
    diagnostics.record_socket_recv(
        wait_execution_id=53,
        raw_message=b"result-54",
        decoded_execution_id=54,
        decoded_message={"status": "COMPLETE"},
    )
    diagnostics.record("BUFFER", "insert", execution_id=54)
    diagnostics.record("WAIT", "cache_miss", execution_id=54)
    diagnostics.record("WAIT", "timeout", execution_id=54)

    assert diagnostics.close()["first_loss_layer"] == "PYTHON_BUFFER_LOSS"


def test_handoff_diagnostics_ignores_other_execution_ids(tmp_path):
    from planning.diagnostics.reliable_single_worker import (
        ExecutionHandoffDiagnostics,
    )

    diagnostics = ExecutionHandoffDiagnostics(
        tmp_path / "handoff.jsonl", target_execution_id=54
    )
    diagnostics.record("BUFFER", "insert", execution_id=53)
    diagnostics.record("WAIT", "cache_miss", execution_id=53)

    summary = diagnostics.close()

    assert summary["event_count"] == 0
    assert summary["first_loss_layer"] == "PYTHON_SOCKET_RECEIVE_LOSS"


def test_dealer_identity_lifecycle_diagnostics_emits_identity_and_connection_events(
    tmp_path,
):
    from planning.diagnostics.reliable_single_worker import (
        DealerIdentityLifecycleDiagnostics,
    )

    socket_owner_thread_id = threading.get_ident()
    diagnostics = DealerIdentityLifecycleDiagnostics(
        tmp_path / "python_dealer_diagnostics.jsonl",
        runtime_instance_id="worker-00-p0-m2",
        dealer_identity=b"worker-00-p0-m2",
        socket_owner_thread_id=socket_owner_thread_id,
    )
    diagnostics.record_connected("tcp://127.0.0.1:60001")
    diagnostics.record_receive_contract()
    diagnostics.record_received_frame(b"result-payload", has_more=False)
    diagnostics.record_receipt_ack_send(
        execution_id=44_000_000_000_054,
        payload=b"receipt-ack",
    )
    diagnostics.record_monitor_event(
        zmq.EVENT_DISCONNECTED,
        0,
        "tcp://127.0.0.1:60001",
    )
    diagnostics.record_monitor_event(
        zmq.EVENT_CONNECTED,
        0,
        "tcp://127.0.0.1:60001",
    )
    diagnostics.close()

    rows = [
        json.loads(line)
        for line in (tmp_path / "python_dealer_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0] == {
        "event": "DEALER_SOCKET_CREATED",
        "socket_access_thread_id": socket_owner_thread_id,
        "socket_creation_thread_id": socket_owner_thread_id,
        "socket_owner_match": True,
        "socket_owner_thread_id": socket_owner_thread_id,
        "timestamp_monotonic_ns": rows[0]["timestamp_monotonic_ns"],
    }
    assert rows[1] == {
        "dealer_identity_hex": b"worker-00-p0-m2".hex(),
        "dealer_identity_length": len(b"worker-00-p0-m2"),
        "event": "DEALER_IDENTITY",
        "runtime_instance_id": "worker-00-p0-m2",
        "timestamp_monotonic_ns": rows[1]["timestamp_monotonic_ns"],
    }
    assert rows[2]["event"] == "DEALER_CONNECTED"
    assert rows[3] == {
        "dealer_identity_hex": b"worker-00-p0-m2".hex(),
        "dealer_identity_length": len(b"worker-00-p0-m2"),
        "event": "DEALER_RECEIVE_CONTRACT",
        "expected_dealer_receive_frame_count": 1,
        "route_identity_frame_visible": False,
        "timestamp_monotonic_ns": rows[3]["timestamp_monotonic_ns"],
    }
    assert rows[4]["event"] == "DEALER_RECEIVE_FRAME"
    assert rows[4]["frame_index"] == 0
    assert rows[4]["frame_size"] == len(b"result-payload")
    assert rows[4]["frame_hex"] == b"result-payload".hex()
    assert rows[4]["has_more"] is False
    assert rows[4]["socket_owner_thread_id"] == socket_owner_thread_id
    assert rows[4]["socket_access_thread_id"] == socket_owner_thread_id
    assert rows[4]["socket_owner_match"] is True
    assert rows[5] == {
        "ack_payload_hex": b"receipt-ack".hex(),
        "ack_payload_size": len(b"receipt-ack"),
        "event": "DEALER_RECEIPT_ACK_SEND",
        "execution_id": 44_000_000_000_054,
        "socket_access_thread_id": socket_owner_thread_id,
        "socket_owner_match": True,
        "socket_owner_thread_id": socket_owner_thread_id,
        "timestamp_monotonic_ns": rows[5]["timestamp_monotonic_ns"],
    }
    assert rows[6]["monitor_event"] == "DISCONNECTED"
    assert rows[7]["monitor_event"] == "CONNECTED"
    assert rows[7]["connection_phase"] == "RECONNECT"


def test_zmq_startup_monitor_can_be_retained_for_dealer_lifecycle_observation():
    from planning.diagnostics.reliable_single_worker import (
        _wait_for_zmq_monitors_connected,
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    dealer = context.socket(zmq.DEALER)
    monitor = None
    try:
        router.bind("tcp://127.0.0.1:{}".format(port))
        monitor = dealer.get_monitor_socket()
        dealer.connect("tcp://127.0.0.1:{}".format(port))
        _wait_for_zmq_monitors_connected(
            [("test dealer", monitor)],
            timeout_s=2.0,
            close_monitors=False,
        )
        assert not monitor.closed
    finally:
        if monitor is not None:
            monitor.close(0)
        dealer.close(0)
        router.close(0)
        context.term()
