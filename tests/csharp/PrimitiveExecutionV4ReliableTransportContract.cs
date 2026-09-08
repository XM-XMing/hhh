using System;
using System.Collections.Generic;
using XMflight;

public static class PrimitiveExecutionV4ReliableTransportContract
{
    private const string RuntimeId = "worker-00-runtime-test";
    private const ulong ExecutionId = 0x0102030405060708UL;

    private static int passed;

    private sealed class FakeTransport : IPrimitiveExecutionResultTransport
    {
        public bool Connected = true;
        public readonly List<byte[]> Payloads = new List<byte[]>();

        public bool IsConnected { get { return Connected; } }

        public void Send(PrimitiveExecutionResultTransmission result)
        {
            Payloads.Add(result.SerializedCanonicalResultBytes);
        }
    }

    private sealed class FakeAckSink : IPrimitiveExecutionResultAckSink
    {
        public readonly List<PrimitiveExecutionV4Ack> Acks =
            new List<PrimitiveExecutionV4Ack>();

        public void SendAck(PrimitiveExecutionV4Ack ack)
        {
            Acks.Add(ack);
        }
    }

    private sealed class ReceiverTransport : IPrimitiveExecutionResultTransport
    {
        private readonly ReliableResultReceiver receiver;
        public bool Connected = true;
        public readonly List<byte[]> Payloads = new List<byte[]>();

        public ReceiverTransport(ReliableResultReceiver receiver)
        {
            this.receiver = receiver;
        }

        public bool IsConnected { get { return Connected; } }

        public void Send(PrimitiveExecutionResultTransmission result)
        {
            Payloads.Add(result.SerializedCanonicalResultBytes);
            receiver.Receive(result);
        }
    }

    private static void Assert(bool condition, string label)
    {
        if (!condition) throw new Exception(label);
    }

    private static void Equal<T>(T actual, T expected, string label)
    {
        if (!EqualityComparer<T>.Default.Equals(actual, expected))
            throw new Exception(label + " expected " + expected + " got " + actual);
    }

    private static bool SameBytes(byte[] left, byte[] right)
    {
        if (left == null || right == null || left.Length != right.Length) return false;
        for (int index = 0; index < left.Length; ++index)
            if (left[index] != right[index]) return false;
        return true;
    }

    private static byte[] CommandHash()
    {
        var hash = new byte[32];
        for (int index = 0; index < hash.Length; ++index) hash[index] = (byte)(index + 1);
        return hash;
    }

    private static PrimitiveExecutionV4Ack Ack(PrimitiveExecutionResultTransmission result)
    {
        return new PrimitiveExecutionV4Ack {
            schema_version = 4,
            runtime_instance_id = result.RuntimeInstanceId,
            execution_id = result.ExecutionId,
            ack_status = "DURABLE_RECEIVED",
            result_payload_hash = result.ResultPayloadHash,
            command_sequence_hash = result.CommandSequenceHash,
        };
    }

    private static PrimitiveExecutionResultTransmission CompleteResult(
        PrimitiveExecutionResultLifecycle lifecycle,
        ulong endpointSimTimeNs = 5480000000UL)
    {
        byte[] commandHash = CommandHash();
        lifecycle.BeginExecution(RuntimeId, ExecutionId, commandHash);
        return lifecycle.FinalizeExecution(new PrimitiveExecutionV4Result {
            schema_version = 4,
            runtime_instance_id = RuntimeId,
            execution_id = ExecutionId,
            status = "COMPLETE",
            requested_frame_count = 25,
            applied_frame_count = 25,
            first_applied_state_id = 4000L,
            endpoint_state_id = 4024L,
            last_applied_frame_index = 24,
            reason_code = "NONE",
            command_sequence_hash = commandHash,
            endpoint_sim_time_ns = endpointSimTimeNs,
            endpoint_observation_ref = new PrimitiveExecutionV4ObservationRef {
                schema_version = 4,
                runtime_instance_id = RuntimeId,
                episode_id = "episode-v4-0001",
                reset_id = "reset-v4-0001",
                state_id = 4024L,
                depth_id = "depth-v4-0001",
                sim_time_ns = endpointSimTimeNs,
            },
            result_generation = 0,
        }, 0UL);
    }

    private static void CompleteResultIsSentAndAckClearsPending()
    {
        var lifecycle = new PrimitiveExecutionResultLifecycle(10UL);
        var transport = new FakeTransport();
        var adapter = new PrimitiveExecutionResultTransportAdapter(lifecycle, transport);
        PrimitiveExecutionResultTransmission result = CompleteResult(lifecycle);

        adapter.Record(result);

        Equal(transport.Payloads.Count, 1, "initial result send count");
        Equal(adapter.PendingResultCount, 1, "pending before ACK");
        Equal(lifecycle.PhysicalExecutionCount, 1, "physical execution count");
        Equal(lifecycle.ResultGeneratedCount, 1, "result generation count");
        Assert(SameBytes(transport.Payloads[0], result.SerializedCanonicalResultBytes),
            "initial payload bytes");

        Equal(adapter.ReceiveAck(Ack(result)), PrimitiveExecutionAckOutcome.ACCEPTED,
            "matching ACK outcome");
        Equal(adapter.PendingResultCount, 0, "pending after ACK");
        Equal(adapter.ReceiveAck(Ack(result)), PrimitiveExecutionAckOutcome.DUPLICATE,
            "duplicate ACK outcome");
        Equal(adapter.PendingResultCount, 0, "pending after duplicate ACK");
        Equal(lifecycle.PhysicalExecutionCount, 1, "physical count after ACK");
        Equal(lifecycle.ResultGeneratedCount, 1, "generation count after ACK");
        ++passed;
    }

    private static void DuplicateCompleteIsAckedWithoutNewReceiptSideEffect()
    {
        var lifecycle = new PrimitiveExecutionResultLifecycle(10UL);
        var ackSink = new FakeAckSink();
        var receiver = new ReliableResultReceiver(ackSink);
        PrimitiveExecutionResultTransmission result = CompleteResult(lifecycle);

        Equal(receiver.Receive(result), ReliableResultReceiveOutcome.ACCEPTED,
            "first receiver outcome");
        Equal(receiver.Receive(result), ReliableResultReceiveOutcome.DUPLICATE,
            "duplicate receiver outcome");
        Equal(receiver.NewResultCount, 1, "new durable result count");
        Equal(receiver.DuplicateResultCount, 1, "duplicate result count");
        Equal(receiver.ProtocolErrorCount, 0, "duplicate protocol error count");
        Equal(ackSink.Acks.Count, 2, "duplicate ACK count");
        Equal(ackSink.Acks[0].schema_version, (uint)4, "ACK schema version");
        Equal(ackSink.Acks[0].message_type, "PrimitiveExecutionResultAck", "ACK message type");
        Equal(ackSink.Acks[0].runtime_instance_id, RuntimeId, "ACK runtime identity");
        Equal(ackSink.Acks[0].ack_status, "DURABLE_RECEIVED", "ACK status");
        Assert(SameBytes(ackSink.Acks[0].result_payload_hash,
            ackSink.Acks[1].result_payload_hash), "duplicate ACK payload identity");
        Equal(ackSink.Acks[0].execution_id, ExecutionId, "duplicate ACK execution identity");
        Assert(SameBytes(ackSink.Acks[0].command_sequence_hash,
            result.CommandSequenceHash), "ACK command identity");
        ++passed;
    }

    private static void ConflictingDuplicateIsProtocolError()
    {
        var firstLifecycle = new PrimitiveExecutionResultLifecycle(10UL);
        var secondLifecycle = new PrimitiveExecutionResultLifecycle(10UL);
        var receiver = new ReliableResultReceiver(new FakeAckSink());
        PrimitiveExecutionResultTransmission first = CompleteResult(firstLifecycle);
        PrimitiveExecutionResultTransmission conflicting = CompleteResult(
            secondLifecycle, 5480000001UL);

        Equal(receiver.Receive(first), ReliableResultReceiveOutcome.ACCEPTED,
            "conflict first outcome");
        Equal(receiver.Receive(conflicting), ReliableResultReceiveOutcome.PROTOCOL_ERROR,
            "conflict outcome");
        Equal(receiver.NewResultCount, 1, "conflict new result count");
        Equal(receiver.ProtocolErrorCount, 1, "conflict protocol error count");
        ++passed;
    }

    private static void AckLossRetransmitsImmutablePayload()
    {
        var lifecycle = new PrimitiveExecutionResultLifecycle(10UL);
        var transport = new FakeTransport();
        var adapter = new PrimitiveExecutionResultTransportAdapter(lifecycle, transport);
        PrimitiveExecutionResultTransmission result = CompleteResult(lifecycle);

        adapter.Record(result);
        for (int retry = 1; retry <= 100; ++retry)
            Assert(adapter.Poll((ulong)(retry * 10)), "retry eligibility");

        Equal(adapter.SendAttemptCount, 101, "100 retransmission send count");
        Equal(transport.Payloads.Count, 101, "100 retransmission transport count");
        Equal(lifecycle.PhysicalExecutionCount, 1, "100 retry physical count");
        Equal(lifecycle.ResultGeneratedCount, 1, "100 retry result generation count");
        Equal(adapter.PendingResultCount, 1, "100 retry pending count");
        for (int attempt = 0; attempt < transport.Payloads.Count; ++attempt)
            Assert(SameBytes(transport.Payloads[attempt], transport.Payloads[0]),
                "retransmission payload identity");

        Equal(adapter.ReceiveAck(Ack(result)), PrimitiveExecutionAckOutcome.ACCEPTED,
            "100 retry final ACK");
        Equal(adapter.PendingResultCount, 0, "100 retry pending after ACK");
        ++passed;
    }

    private static void WrongAckVariantsPreservePending()
    {
        var lifecycle = new PrimitiveExecutionResultLifecycle(10UL);
        var transport = new FakeTransport();
        var adapter = new PrimitiveExecutionResultTransportAdapter(lifecycle, transport);
        PrimitiveExecutionResultTransmission result = CompleteResult(lifecycle);
        adapter.Record(result);

        PrimitiveExecutionV4Ack wrongRuntime = Ack(result);
        wrongRuntime.runtime_instance_id = "worker-01-runtime-test";
        PrimitiveExecutionV4Ack wrongExecution = Ack(result);
        wrongExecution.execution_id += 1UL;
        PrimitiveExecutionV4Ack wrongResultHash = Ack(result);
        wrongResultHash.result_payload_hash[0] ^= 0xff;
        PrimitiveExecutionV4Ack wrongCommandHash = Ack(result);
        wrongCommandHash.command_sequence_hash[0] ^= 0xff;

        Equal(adapter.ReceiveAck(wrongRuntime), PrimitiveExecutionAckOutcome.REJECTED,
            "wrong runtime ACK");
        Equal(adapter.PendingResultCount, 1, "wrong runtime pending");
        Equal(adapter.ReceiveAck(wrongExecution), PrimitiveExecutionAckOutcome.REJECTED,
            "wrong execution ACK");
        Equal(adapter.PendingResultCount, 1, "wrong execution pending");
        Equal(adapter.ReceiveAck(wrongResultHash), PrimitiveExecutionAckOutcome.REJECTED,
            "wrong result hash ACK");
        Equal(adapter.PendingResultCount, 1, "wrong result hash pending");
        Equal(adapter.ReceiveAck(wrongCommandHash), PrimitiveExecutionAckOutcome.REJECTED,
            "wrong command hash ACK");
        Equal(adapter.PendingResultCount, 1, "wrong command hash pending");
        ++passed;
    }

    private static void DisconnectReconnectResendsPendingResult()
    {
        var lifecycle = new PrimitiveExecutionResultLifecycle(10UL);
        var transport = new FakeTransport();
        var adapter = new PrimitiveExecutionResultTransportAdapter(lifecycle, transport);
        PrimitiveExecutionResultTransmission result = CompleteResult(lifecycle);
        adapter.Record(result);

        adapter.OnDisconnected();
        transport.Connected = false;
        Assert(!adapter.Poll(10UL), "disconnected retry must wait");
        Equal(transport.Payloads.Count, 1, "disconnected send count");
        Equal(adapter.PendingResultCount, 1, "disconnected pending count");

        transport.Connected = true;
        Assert(adapter.OnReconnected(), "reconnect must resend pending result");
        Equal(transport.Payloads.Count, 2, "reconnect send count");
        Assert(SameBytes(transport.Payloads[0], transport.Payloads[1]),
            "reconnect payload identity");
        Equal(adapter.PendingResultCount, 1, "reconnect pending before ACK");
        Equal(adapter.ReceiveAck(Ack(result)), PrimitiveExecutionAckOutcome.ACCEPTED,
            "reconnect ACK");
        Equal(adapter.PendingResultCount, 0, "reconnect pending after ACK");
        ++passed;
    }

    private static void AckLossMakesReceiverSeeDuplicateButNoNewReceipt()
    {
        var lifecycle = new PrimitiveExecutionResultLifecycle(10UL);
        var ackSink = new FakeAckSink();
        var receiver = new ReliableResultReceiver(ackSink);
        var transport = new ReceiverTransport(receiver);
        var adapter = new PrimitiveExecutionResultTransportAdapter(lifecycle, transport);
        PrimitiveExecutionResultTransmission result = CompleteResult(lifecycle);

        adapter.Record(result);
        Equal(receiver.NewResultCount, 1, "loopback first durable result count");
        Equal(ackSink.Acks.Count, 1, "loopback first ACK count");
        ackSink.Acks.Clear();

        Assert(adapter.Poll(10UL), "loopback retry eligibility");
        Equal(transport.Payloads.Count, 2, "loopback retry send count");
        Equal(receiver.NewResultCount, 1, "loopback retry new result count");
        Equal(receiver.DuplicateResultCount, 1, "loopback retry duplicate count");
        Equal(ackSink.Acks.Count, 1, "loopback retry ACK count");
        Equal(adapter.ReceiveAck(ackSink.Acks[0]), PrimitiveExecutionAckOutcome.ACCEPTED,
            "loopback retry ACK outcome");
        Equal(adapter.PendingResultCount, 0, "loopback retry pending count");
        Equal(lifecycle.PhysicalExecutionCount, 1, "loopback retry physical count");
        Equal(lifecycle.ResultGeneratedCount, 1, "loopback retry generation count");
        ++passed;
    }

    public static void Main(string[] args)
    {
        CompleteResultIsSentAndAckClearsPending();
        DuplicateCompleteIsAckedWithoutNewReceiptSideEffect();
        ConflictingDuplicateIsProtocolError();
        AckLossRetransmitsImmutablePayload();
        WrongAckVariantsPreservePending();
        DisconnectReconnectResendsPendingResult();
        AckLossMakesReceiverSeeDuplicateButNoNewReceipt();
        Console.WriteLine("C# reliable transport tests: {0} passed", passed);
    }
}
