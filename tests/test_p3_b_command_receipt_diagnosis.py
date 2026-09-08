"""Deterministic P3-B command-receipt handoff diagnostics.

These tests intentionally exercise only the Python command-channel seam.  A
passing receipt here does not prove a live bridge delivery; the diagnostic
events are the evidence boundary needed for the multi-worker reproduction.
"""

import json

import msgpack
import pytest
import zmq

from planning.runtime.primitive_execution_command_transport import (
    ZmqPrimitiveExecutionCommandClient,
)


def _receipt(runtime, execution_id, command_hash):
    return msgpack.packb(
        {
            "schema_version": 4,
            "message_type": "PrimitiveExecutionCommandReceiptAck",
            "runtime_instance_id": runtime,
            "execution_id": execution_id,
            "ack_status": "RECEIVED",
            "reason_code": "NONE",
            "command_sequence_hash": command_hash,
        },
        use_bin_type=True,
    )


@pytest.mark.unit
def test_receipt_handoff_diagnostics_capture_bridge_to_waiter_lifecycle(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PLANNING_WORKER_ID", "4")
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    endpoint = "inproc://p3-b-receipt-diagnostics"
    router.bind(endpoint)
    path = tmp_path / "worker-04-command-receipts.jsonl"
    client = ZmqPrimitiveExecutionCommandClient(
        context,
        endpoint=endpoint,
        dealer_identity=b"worker-04-awac-p3",
        connect_timeout_s=1.0,
        diagnostics_path=str(path),
    )
    try:
        assert router.poll(1000) & zmq.POLLIN
        identity, ready = router.recv_multipart()
        assert identity == b"worker-04-awac-p3"
        assert msgpack.unpackb(ready, raw=False)["message_type"] == (
            "PrimitiveExecutionCommandReady"
        )
        command_hash = b"a" * 32
        client.send(b"immutable-command")
        router.recv_multipart()
        router.send_multipart([identity, _receipt("worker-04-awac-p3", 28, command_hash)])
        value = client.wait_for_receipt(
            execution_id=28,
            command_sequence_hash=command_hash,
            timeout_s=1.0,
        )
        assert value["execution_id"] == 28
    finally:
        client.close()
        router.close(0)
        context.term()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    names = [event["event"] for event in events]
    assert "SOCKET_CREATED" in names
    assert "READY_SEND_OK" in names
    assert "RAW_PACKET_RECEIVED" in names
    assert "COMMAND_RECEIPT_BUFFER_INSERT" in names
    assert "COMMAND_RECEIPT_WAITER_MATCH" in names
    receipt_events = [
        event for event in events if event["event"] == "RAW_PACKET_RECEIVED"
    ]
    assert receipt_events[-1]["worker_id"] == 4
    assert receipt_events[-1]["runtime_instance_id"] == "worker-04-awac-p3"
    assert receipt_events[-1]["execution_id"] == 28


@pytest.mark.unit
def test_receipt_timeout_records_process_liveness_from_diagnostic_manifest(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PLANNING_WORKER_ID", "4")
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    endpoint = "inproc://p3-b-receipt-timeout-liveness"
    router.bind(endpoint)
    path = tmp_path / "worker-04-command-receipts.jsonl"
    liveness_path = tmp_path / "runtime-processes.json"
    liveness_path.write_text(
        json.dumps(
            {
                "workers": {
                    "4": {
                        "bridge_pid": 999999,
                        "unity_pid": 999998,
                        "roscore_pid": 999997,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "P3_COMMAND_RECEIPT_PROCESS_MANIFEST_PATH", str(liveness_path)
    )
    client = ZmqPrimitiveExecutionCommandClient(
        context,
        endpoint=endpoint,
        dealer_identity=b"worker-04-awac-p3",
        connect_timeout_s=0.01,
        diagnostics_path=str(path),
    )
    try:
        assert router.poll(1000) & zmq.POLLIN
        router.recv_multipart()
        with pytest.raises(TimeoutError, match="execution_id=28"):
            client.wait_for_receipt(
                execution_id=28,
                command_sequence_hash=b"z" * 32,
                timeout_s=0.01,
            )
    finally:
        client.close()
        router.close(0)
        context.term()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    liveness = [
        event for event in events
        if event["event"] == "COMMAND_RECEIPT_TIMEOUT_LIVENESS"
    ]
    assert len(liveness) == 1
    assert liveness[0]["execution_id"] == 28
    assert liveness[0]["worker_process"]["pid"] == liveness[0]["pid"]
    assert liveness[0]["learner_process"]["pid"] > 0
    assert liveness[0]["bridge_process"]["pid"] == 999999
    assert liveness[0]["bridge_process"]["alive"] is False


@pytest.mark.unit
def test_same_execution_id_isolated_by_independent_worker_clients():
    context = zmq.Context()
    routers = []
    clients = []
    try:
        for worker in range(12):
            router = context.socket(zmq.ROUTER)
            router.setsockopt(zmq.LINGER, 0)
            router.bind("inproc://p3-b-worker-{}".format(worker))
            client = ZmqPrimitiveExecutionCommandClient(
                context,
                endpoint="inproc://p3-b-worker-{}".format(worker),
                dealer_identity=("worker-{:02d}".format(worker)).encode(),
                connect_timeout_s=1.0,
            )
            assert router.poll(1000) & zmq.POLLIN
            identity, ready = router.recv_multipart()
            assert identity == ("worker-{:02d}".format(worker)).encode()
            assert msgpack.unpackb(ready, raw=False)["message_type"] == (
                "PrimitiveExecutionCommandReady"
            )
            routers.append((router, identity))
            clients.append(client)

        for worker, (router, identity) in enumerate(routers):
            clients[worker].send("command-{}".format(worker).encode())
            router.recv_multipart()
            command_hash = bytes([65 + worker]) * 32
            router.send_multipart([
                identity,
                _receipt("worker-{:02d}".format(worker), 28, command_hash),
            ])

        for worker, client in enumerate(clients):
            command_hash = bytes([65 + worker]) * 32
            value = client.wait_for_receipt(
                execution_id=28,
                command_sequence_hash=command_hash,
                timeout_s=1.0,
            )
            assert value["runtime_instance_id"] == "worker-{:02d}".format(worker)
    finally:
        for client in clients:
            client.close()
        for router, _identity in routers:
            router.close(0)
        context.term()


@pytest.mark.unit
def test_reset_completions_and_receipts_are_isolated_across_workers():
    context = zmq.Context()
    routers = []
    clients = []
    try:
        for worker in (0, 1):
            router = context.socket(zmq.ROUTER)
            router.setsockopt(zmq.LINGER, 0)
            router.bind("inproc://p3-b-reset-worker-{}".format(worker))
            runtime = "worker-{:02d}".format(worker)
            client = ZmqPrimitiveExecutionCommandClient(
                context,
                endpoint="inproc://p3-b-reset-worker-{}".format(worker),
                dealer_identity=runtime.encode(),
                connect_timeout_s=1.0,
            )
            assert router.poll(1000) & zmq.POLLIN
            identity, _ready = router.recv_multipart()
            routers.append((router, identity, runtime))
            clients.append(client)

        reset_payload = lambda runtime, worker: msgpack.packb(
            {
                "schema_version": 4,
                "message_type": "PrimitiveResetComplete",
                "runtime_instance_id": runtime,
                "episode_id": "episode-{}".format(worker),
                "reset_id": "reset-{}".format(worker),
            },
            use_bin_type=True,
        )
        for worker, (router, identity, runtime) in enumerate(routers):
            command_hash = bytes([90 - worker]) * 32
            # Deliberately interleave each worker's reset completion and
            # command receipt in the opposite order.
            if worker == 0:
                router.send_multipart([
                    identity,
                    _receipt(runtime, 28, command_hash),
                ])
                router.send_multipart([
                    identity,
                    reset_payload(runtime, worker),
                ])
            else:
                router.send_multipart([
                    identity,
                    reset_payload(runtime, worker),
                ])
                router.send_multipart([
                    identity,
                    _receipt(runtime, 28, command_hash),
                ])

        for worker, client in enumerate(clients):
            runtime = "worker-{:02d}".format(worker)
            reset = client.wait_for_reset(
                episode_id="episode-{}".format(worker),
                reset_id="reset-{}".format(worker),
                timeout_s=1.0,
            )
            receipt = client.wait_for_receipt(
                execution_id=28,
                command_sequence_hash=bytes([90 - worker]) * 32,
                timeout_s=1.0,
            )
            assert reset["runtime_instance_id"] == runtime
            assert receipt["runtime_instance_id"] == runtime
    finally:
        for client in clients:
            client.close()
        for router, _identity, _runtime in routers:
            router.close(0)
        context.term()


@pytest.mark.unit
def test_one_thousand_interleaved_receipts_across_twelve_workers():
    context = zmq.Context()
    routers = []
    clients = []
    try:
        for worker in range(12):
            router = context.socket(zmq.ROUTER)
            router.setsockopt(zmq.LINGER, 0)
            router.bind("inproc://p3-b-interleaved-{}".format(worker))
            client = ZmqPrimitiveExecutionCommandClient(
                context,
                endpoint="inproc://p3-b-interleaved-{}".format(worker),
                dealer_identity=("worker-{:02d}".format(worker)).encode(),
                connect_timeout_s=1.0,
            )
            assert router.poll(1000) & zmq.POLLIN
            identity, _ready = router.recv_multipart()
            routers.append((router, identity))
            clients.append(client)

        expected = {}
        for index in range(1000):
            worker = index % 12
            execution_id = index // 12
            command_hash = bytes([index % 251]) * 32
            expected[(worker, execution_id)] = command_hash
            routers[worker][0].send_multipart([
                routers[worker][1],
                _receipt("worker-{:02d}".format(worker), execution_id, command_hash),
            ])

        for (worker, execution_id), command_hash in expected.items():
            value = clients[worker].wait_for_receipt(
                execution_id=execution_id,
                command_sequence_hash=command_hash,
                timeout_s=1.0,
            )
            assert value["runtime_instance_id"] == "worker-{:02d}".format(worker)
    finally:
        for client in clients:
            client.close()
        for router, _identity in routers:
            router.close(0)
        context.term()
