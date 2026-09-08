"""Public DEALER receipt seam for the reliable primitive command channel."""

from contextlib import contextmanager

import msgpack
import pytest
import zmq

from planning.runtime.primitive_execution_command_transport import (
    PrimitiveExecutionCommandReceiptError,
    ZmqPrimitiveExecutionCommandClient,
)


@contextmanager
def _client_and_router():
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.setsockopt(zmq.ROUTER_HANDOVER, 1)
    router.bind("inproc://primitive-command-receipt-test")
    client = ZmqPrimitiveExecutionCommandClient(
        context,
        endpoint="inproc://primitive-command-receipt-test",
        dealer_identity=b"worker-00",
        connect_timeout_s=1.0,
    )
    try:
        assert router.poll(1000) & zmq.POLLIN
        identity, ready = router.recv_multipart()
        assert identity == b"worker-00"
        assert msgpack.unpackb(ready, raw=False)["message_type"] == (
            "PrimitiveExecutionCommandReady"
        )
        yield client, router, identity
    finally:
        client.close()
        router.close(0)
        context.term()


def _receipt(runtime, execution_id, command_hash, status="RECEIVED"):
    return msgpack.packb(
        {
            "schema_version": 4,
            "message_type": "PrimitiveExecutionCommandReceiptAck",
            "runtime_instance_id": runtime,
            "execution_id": execution_id,
            "ack_status": status,
            "reason_code": "NONE",
            "command_sequence_hash": command_hash,
        },
        use_bin_type=True,
    )


def _reset_complete(runtime="worker-00", episode_id="episode-21", reset_id="reset-21"):
    return msgpack.packb(
        {
            "schema_version": 4,
            "message_type": "PrimitiveResetComplete",
            "runtime_instance_id": runtime,
            "episode_id": episode_id,
            "reset_id": reset_id,
            "observation_ref": {
                "schema_version": 4,
                "runtime_instance_id": runtime,
                "episode_id": episode_id,
                "reset_id": reset_id,
                "state_id": 101,
                "depth_id": "depth-101",
                "sim_time_ns": 50500000000,
            },
        },
        use_bin_type=True,
    )


@pytest.mark.unit
def test_out_of_order_receipts_are_buffered_by_execution_identity():
    with _client_and_router() as (client, router, identity):
        client.send(b"command-a")
        _, _ = router.recv_multipart()
        client.send(b"command-b")
        _, _ = router.recv_multipart()
        hash_a = b"a" * 32
        hash_b = b"b" * 32
        router.send_multipart([identity, _receipt("worker-00", 2, hash_b)])
        router.send_multipart([identity, _receipt("worker-00", 1, hash_a)])

        first = client.wait_for_receipt(
            execution_id=1, command_sequence_hash=hash_a, timeout_s=1.0
        )
        second = client.wait_for_receipt(
            execution_id=2, command_sequence_hash=hash_b, timeout_s=0.01
        )
        assert first["execution_id"] == 1
        assert second["execution_id"] == 2


@pytest.mark.unit
def test_conflicting_duplicate_receipt_fails_closed():
    with _client_and_router() as (client, router, identity):
        client.send(b"command-a")
        router.recv_multipart()
        hash_b = b"b" * 32
        router.send_multipart([identity, _receipt("worker-00", 2, hash_b)])
        router.send_multipart([identity, _receipt("worker-00", 2, b"c" * 32)])
        with pytest.raises(PrimitiveExecutionCommandReceiptError, match="conflicting"):
            client.wait_for_receipt(
                execution_id=1, command_sequence_hash=b"a" * 32, timeout_s=1.0
            )


@pytest.mark.unit
def test_identical_duplicate_receipt_is_idempotent():
    with _client_and_router() as (client, router, identity):
        client.send(b"command-a")
        router.recv_multipart()
        payload = _receipt("worker-00", 1, b"a" * 32)
        router.send_multipart([identity, payload])
        router.send_multipart([identity, payload])

        first = client.wait_for_receipt(
            execution_id=1, command_sequence_hash=b"a" * 32, timeout_s=1.0
        )
        second = client.wait_for_receipt(
            execution_id=1, command_sequence_hash=b"a" * 32, timeout_s=1.0
        )
        assert first == second


@pytest.mark.unit
def test_receipt_runtime_identity_is_checked():
    with _client_and_router() as (client, router, identity):
        client.send(b"command-a")
        router.recv_multipart()
        router.send_multipart(
            [identity, _receipt("worker-01", 1, b"a" * 32)]
        )
        with pytest.raises(PrimitiveExecutionCommandReceiptError, match="runtime"):
            client.wait_for_receipt(
                execution_id=1, command_sequence_hash=b"a" * 32, timeout_s=1.0
            )


@pytest.mark.unit
def test_reset_completion_interleaved_before_command_receipt_is_not_lost():
    with _client_and_router() as (client, router, identity):
        client.send(b"command-a")
        router.recv_multipart()
        reset_payload = _reset_complete()
        receipt_payload = _receipt("worker-00", 1, b"a" * 32)
        router.send_multipart([identity, reset_payload])
        router.send_multipart([identity, receipt_payload])

        receipt = client.wait_for_receipt(
            execution_id=1, command_sequence_hash=b"a" * 32, timeout_s=1.0
        )
        assert receipt["execution_id"] == 1
        completion = client.wait_for_reset(
            episode_id="episode-21", reset_id="reset-21", timeout_s=0.01
        )
        assert completion["message_type"] == "PrimitiveResetComplete"


@pytest.mark.unit
def test_command_receipt_interleaved_before_reset_completion_is_not_lost():
    with _client_and_router() as (client, router, identity):
        client.send(b"command-a")
        router.recv_multipart()
        receipt_payload = _receipt("worker-00", 1, b"a" * 32)
        reset_payload = _reset_complete()
        router.send_multipart([identity, receipt_payload])
        router.send_multipart([identity, reset_payload])

        completion = client.wait_for_reset(
            episode_id="episode-21", reset_id="reset-21", timeout_s=1.0
        )
        assert completion["message_type"] == "PrimitiveResetComplete"
        receipt = client.wait_for_receipt(
            execution_id=1, command_sequence_hash=b"a" * 32, timeout_s=0.01
        )
        assert receipt["execution_id"] == 1


@pytest.mark.unit
def test_duplicate_reset_completion_is_idempotent():
    with _client_and_router() as (client, router, identity):
        client.send(b"command-a")
        router.recv_multipart()
        payload = _reset_complete()
        router.send_multipart([identity, payload])
        router.send_multipart([identity, payload])

        first = client.wait_for_reset(
            episode_id="episode-21", reset_id="reset-21", timeout_s=1.0
        )
        second = client.wait_for_reset(
            episode_id="episode-21", reset_id="reset-21", timeout_s=1.0
        )
        assert first == second


@pytest.mark.unit
def test_one_hundred_interleaved_messages_are_buffered_without_loss():
    with _client_and_router() as (client, router, identity):
        client.send(b"command-batch")
        router.recv_multipart()
        receipts = [(index, bytes([65 + index % 26]) * 32) for index in range(50)]
        resets = [
            ("episode-{}".format(index), "reset-{}".format(index))
            for index in range(50)
        ]
        for index, command_hash in receipts:
            router.send_multipart(
                [identity, _receipt("worker-00", index, command_hash)]
            )
        for episode_id, reset_id in resets:
            router.send_multipart(
                [identity, _reset_complete("worker-00", episode_id, reset_id)]
            )

        for index, command_hash in receipts:
            receipt = client.wait_for_receipt(
                execution_id=index,
                command_sequence_hash=command_hash,
                timeout_s=1.0 if index == 0 else 0.01,
            )
            assert receipt["execution_id"] == index
        for episode_id, reset_id in resets:
            completion = client.wait_for_reset(
                episode_id=episode_id,
                reset_id=reset_id,
                timeout_s=1.0 if episode_id == "episode-0" else 0.01,
            )
            assert completion["episode_id"] == episode_id


@pytest.mark.unit
def test_python_reconnect_reuses_identity_and_resends_exact_payload():
    with _client_and_router() as (client, router, identity):
        payload = b"immutable-command-payload"
        client.send(payload)
        first_identity, first_payload = router.recv_multipart()
        assert first_identity == identity
        assert first_payload == payload

        client.reconnect()
        new_identity, ready = router.recv_multipart()
        assert new_identity == identity
        assert msgpack.unpackb(ready, raw=False)["runtime_instance_id"] == (
            "worker-00"
        )

        client.send(payload)
        resend_identity, resend_payload = router.recv_multipart()
        assert resend_identity == identity
        assert resend_payload == payload
        router.send_multipart([identity, _receipt("worker-00", 1, b"a" * 32)])
        receipt = client.wait_for_receipt(
            execution_id=1, command_sequence_hash=b"a" * 32, timeout_s=1.0
        )
        assert receipt["execution_id"] == 1


@pytest.mark.unit
def test_buffered_receipt_survives_reconnect():
    with _client_and_router() as (client, router, identity):
        client.send(b"command-a")
        router.recv_multipart()
        router.send_multipart([identity, _receipt("worker-00", 1, b"a" * 32)])

        with pytest.raises(TimeoutError):
            client.wait_for_receipt(
                execution_id=2, command_sequence_hash=b"b" * 32, timeout_s=0.05
            )

        client.reconnect()
        new_identity, ready = router.recv_multipart()
        assert new_identity == identity
        assert msgpack.unpackb(ready, raw=False)["message_type"] == (
            "PrimitiveExecutionCommandReady"
        )
        receipt = client.wait_for_receipt(
            execution_id=1, command_sequence_hash=b"a" * 32, timeout_s=0.01
        )
        assert receipt["execution_id"] == 1
