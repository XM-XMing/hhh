using System;
using System.Collections.Generic;
using XMflight;

public static class PrimitiveExecutionV4LifecycleContract
{
    private const string RuntimeId = "worker-00-runtime-test";
    private const ulong ExecutionId = 0x0102030405060708UL;

    private static int passed;

    private static byte[] Hash(byte value)
    {
        byte[] result = new byte[32];
        for (int index = 0; index < result.Length; ++index) result[index] = value;
        return result;
    }

    private static byte[] Hex(string text)
    {
        byte[] bytes = new byte[text.Length / 2];
        for (int index = 0; index < bytes.Length; ++index)
            bytes[index] = Convert.ToByte(text.Substring(index * 2, 2), 16);
        return bytes;
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

    private static void EqualBytes(byte[] actual, byte[] expected, string label)
    {
        Assert(actual != null && expected != null && actual.Length == expected.Length,
            label + " length mismatch");
        for (int index = 0; index < actual.Length; ++index)
            if (actual[index] != expected[index]) throw new Exception(label + " byte mismatch");
    }

    private static void Throws(Action action, string label)
    {
        try
        {
            action();
        }
        catch (PrimitiveExecutionLifecycleException)
        {
            return;
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception(label + " did not throw");
    }

    private static PrimitiveExecutionV4Result Result(
        string status, byte[] commandHash, ulong executionId = ExecutionId)
    {
        var result = new PrimitiveExecutionV4Result {
            schema_version = 4,
            runtime_instance_id = RuntimeId,
            execution_id = executionId,
            status = status,
            requested_frame_count = 25,
            command_sequence_hash = (byte[])commandHash.Clone(),
            result_generation = 0,
        };
        if (status == "COMPLETE")
        {
            result.applied_frame_count = 25;
            result.first_applied_state_id = 4000;
            result.endpoint_state_id = 4024;
            result.last_applied_frame_index = 24;
            result.reason_code = "NONE";
            result.endpoint_sim_time_ns = 5000000000UL;
            result.endpoint_observation_ref = new PrimitiveExecutionV4ObservationRef {
                schema_version = 4,
                runtime_instance_id = RuntimeId,
                episode_id = "episode-v4-0001",
                reset_id = "reset-v4-0001",
                state_id = 4024,
                depth_id = "depth-v4-0001",
                sim_time_ns = 5000000000UL,
            };
        }
        else if (status == "REJECTED")
        {
            result.applied_frame_count = 0;
            result.last_applied_frame_index = -1;
            result.reason_code = "MALFORMED_COMMAND";
        }
        else if (status == "CANCELLED")
        {
            result.applied_frame_count = 11;
            result.first_applied_state_id = 7000;
            result.last_applied_frame_index = 10;
            result.reason_code = "STOP";
        }
        else
        {
            result.applied_frame_count = 7;
            result.first_applied_state_id = 8000;
            result.last_applied_frame_index = 6;
            // The final Unity protocol requires a COLLISION failure to carry
            // its endpoint observation.  This standalone lifecycle fixture
            // exercises an internal failure without an endpoint instead.
            result.reason_code = "INTERNAL_ERROR";
        }
        return result;
    }

    private static PrimitiveExecutionV4Ack Ack(PrimitiveExecutionResultTransmission transmission)
    {
        return new PrimitiveExecutionV4Ack {
            schema_version = 4,
            message_type = "PrimitiveExecutionResultAck",
            runtime_instance_id = transmission.RuntimeInstanceId,
            execution_id = transmission.ExecutionId,
            ack_status = "DURABLE_RECEIVED",
            result_payload_hash = transmission.ResultPayloadHash,
            command_sequence_hash = transmission.CommandSequenceHash,
        };
    }

    private static PrimitiveExecutionResultLifecycle NewLifecycle()
    {
        return new PrimitiveExecutionResultLifecycle(10UL);
    }

    private static PrimitiveExecutionResultTransmission Complete(
        PrimitiveExecutionResultLifecycle lifecycle)
    {
        lifecycle.BeginExecution(RuntimeId, ExecutionId, Hash(0x11));
        return lifecycle.FinalizeExecution(Result("COMPLETE", Hash(0x11)), 0UL);
    }

    private static void TestCompleteNormalPath()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission transmission = Complete(lifecycle);
        Equal(lifecycle.State, PrimitiveExecutionLifecycleState.FINAL_RESULT_PENDING_ACK,
            "complete state");
        Equal(lifecycle.PhysicalExecutionCount, 1, "complete physical execution count");
        Equal(lifecycle.AppliedFrameCount, (uint)25, "complete applied count");
        Equal(lifecycle.ResultGeneratedCount, 1, "complete result count");
        Equal(lifecycle.SendAttemptCount, 1, "complete initial send");
        Equal(transmission.ResultGeneration, (uint)0, "complete generation");
        EqualBytes(transmission.ResultPayloadHash,
            XMProtocolV4.Sha256Bytes(transmission.SerializedCanonicalResultBytes),
            "complete result hash");
        EqualBytes(transmission.SerializedCanonicalResultBytes,
            lifecycle.PendingResult.SerializedCanonicalResultBytes,
            "complete immutable bytes");
        ++passed;
    }

    private static void TestHundredRetransmissions()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission first = Complete(lifecycle);
        for (ulong now = 10UL; now <= 1000UL; now += 10UL)
        {
            PrimitiveExecutionResultTransmission retry = lifecycle.Poll(now);
            Assert(retry != null, "retry missing");
            EqualBytes(retry.SerializedCanonicalResultBytes,
                first.SerializedCanonicalResultBytes, "retry bytes");
            EqualBytes(retry.ResultPayloadHash, first.ResultPayloadHash, "retry hash");
        }
        Equal(lifecycle.PhysicalExecutionCount, 1, "retry physical execution count");
        Equal(lifecycle.ResultGeneratedCount, 1, "retry result count");
        Equal(lifecycle.SendAttemptCount, 101, "retry send count");
        ++passed;
    }

    private static void TestMatchingDurableReceivedClearsPending()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission transmission = Complete(lifecycle);
        Equal(lifecycle.Acknowledge(Ack(transmission)), PrimitiveExecutionAckOutcome.ACCEPTED,
            "matching ack outcome");
        Assert(!lifecycle.HasPendingResult, "matching ack pending");
        Equal(lifecycle.State, PrimitiveExecutionLifecycleState.ACKED, "matching ack state");
        ++passed;
    }

    private static void TestWrongExecutionAckKeepsPending()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission transmission = Complete(lifecycle);
        PrimitiveExecutionV4Ack ack = Ack(transmission);
        ack.execution_id += 1;
        Equal(lifecycle.Acknowledge(ack), PrimitiveExecutionAckOutcome.REJECTED,
            "wrong execution ack outcome");
        Assert(lifecycle.HasPendingResult, "wrong execution pending");
        ++passed;
    }

    private static void TestWrongRuntimeAckKeepsPending()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission transmission = Complete(lifecycle);
        PrimitiveExecutionV4Ack ack = Ack(transmission);
        ack.runtime_instance_id = "worker-01-runtime-test";
        Equal(lifecycle.Acknowledge(ack), PrimitiveExecutionAckOutcome.REJECTED,
            "wrong runtime ack outcome");
        Assert(lifecycle.HasPendingResult, "wrong runtime pending");
        ++passed;
    }

    private static void TestWrongResultHashAckKeepsPending()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission transmission = Complete(lifecycle);
        PrimitiveExecutionV4Ack ack = Ack(transmission);
        ack.result_payload_hash = Hash(0x22);
        Equal(lifecycle.Acknowledge(ack), PrimitiveExecutionAckOutcome.REJECTED,
            "wrong result hash outcome");
        Assert(lifecycle.HasPendingResult, "wrong result hash pending");
        ++passed;
    }

    private static void TestWrongCommandHashAckKeepsPending()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission transmission = Complete(lifecycle);
        PrimitiveExecutionV4Ack ack = Ack(transmission);
        ack.command_sequence_hash = Hash(0x22);
        Equal(lifecycle.Acknowledge(ack), PrimitiveExecutionAckOutcome.REJECTED,
            "wrong command hash outcome");
        Assert(lifecycle.HasPendingResult, "wrong command hash pending");
        ++passed;
    }

    private static void TestWrongSchemaAckKeepsPending()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission transmission = Complete(lifecycle);
        PrimitiveExecutionV4Ack ack = Ack(transmission);
        ack.schema_version = 3;
        Equal(lifecycle.Acknowledge(ack), PrimitiveExecutionAckOutcome.REJECTED,
            "wrong schema ack outcome");
        Assert(lifecycle.HasPendingResult, "wrong schema pending");
        ++passed;
    }

    private static void TestDuplicateMatchingAckIdempotent()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission transmission = Complete(lifecycle);
        PrimitiveExecutionV4Ack ack = Ack(transmission);
        Equal(lifecycle.Acknowledge(ack), PrimitiveExecutionAckOutcome.ACCEPTED,
            "first ack");
        Equal(lifecycle.Acknowledge(ack), PrimitiveExecutionAckOutcome.DUPLICATE,
            "duplicate ack");
        Equal(lifecycle.PhysicalExecutionCount, 1, "duplicate ack physical execution");
        Equal(lifecycle.ResultGeneratedCount, 1, "duplicate ack result generation");
        ++passed;
    }

    private static void TestIncompleteCompleteDoesNotGenerate()
    {
        var lifecycle = NewLifecycle();
        lifecycle.BeginExecution(RuntimeId, ExecutionId, Hash(0x11));
        PrimitiveExecutionV4Result invalid = Result("COMPLETE", Hash(0x11));
        invalid.applied_frame_count = 24;
        invalid.last_applied_frame_index = 23;
        Throws(() => lifecycle.FinalizeExecution(invalid, 0UL), "incomplete complete");
        Equal(lifecycle.ResultGeneratedCount, 0, "incomplete generated count");
        Equal(lifecycle.State, PrimitiveExecutionLifecycleState.EXECUTING,
            "incomplete state");
        ++passed;
    }

    private static void TestRejectedBeforeExecution()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission transmission = lifecycle.RejectBeforeExecution(
            Result("REJECTED", Hash(0x11)), 0UL);
        Equal(transmission.Status, "REJECTED", "rejected status");
        Equal(lifecycle.PhysicalExecutionCount, 0, "rejected physical execution");
        Equal(lifecycle.AppliedFrameCount, (uint)0, "rejected applied");
        Equal(lifecycle.ResultGeneratedCount, 1, "rejected generated");
        Assert(lifecycle.HasPendingResult, "rejected pending");
        ++passed;
    }

    private static void TestCancelledPrefix()
    {
        var lifecycle = NewLifecycle();
        lifecycle.BeginExecution(RuntimeId, ExecutionId, Hash(0x11));
        PrimitiveExecutionResultTransmission transmission = lifecycle.FinalizeExecution(
            Result("CANCELLED", Hash(0x11)), 0UL);
        Equal(transmission.Status, "CANCELLED", "cancelled status");
        Equal(lifecycle.AppliedFrameCount, (uint)11, "cancelled prefix");
        ++passed;
    }

    private static void TestFailedPrefix()
    {
        var lifecycle = NewLifecycle();
        lifecycle.BeginExecution(RuntimeId, ExecutionId, Hash(0x11));
        PrimitiveExecutionResultTransmission transmission = lifecycle.FinalizeExecution(
            Result("FAILED", Hash(0x11)), 0UL);
        Equal(transmission.Status, "FAILED", "failed status");
        Equal(lifecycle.AppliedFrameCount, (uint)7, "failed prefix");
        ++passed;
    }

    private static void TestConflictingFinalizationRejected()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission first = Complete(lifecycle);
        Throws(() => lifecycle.FinalizeExecution(Result("FAILED", Hash(0x11)), 1UL),
            "conflicting finalization");
        Equal(lifecycle.State, PrimitiveExecutionLifecycleState.FINAL_RESULT_PENDING_ACK,
            "conflicting state");
        EqualBytes(lifecycle.PendingResult.SerializedCanonicalResultBytes,
            first.SerializedCanonicalResultBytes, "conflicting immutable result");
        ++passed;
    }

    private static void TestSecondExecutionRejectedAndCannotClearA()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission first = Complete(lifecycle);
        Throws(() => lifecycle.BeginExecution(RuntimeId, ExecutionId + 1, Hash(0x22)),
            "second execution");
        PrimitiveExecutionV4Ack ack = Ack(first);
        ack.execution_id += 1;
        Equal(lifecycle.Acknowledge(ack), PrimitiveExecutionAckOutcome.REJECTED,
            "execution B ack");
        Assert(lifecycle.HasPendingResult, "execution A pending");
        ++passed;
    }

    private static void TestNextExecutionStartsOnlyAfterAck()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission first = Complete(lifecycle);
        Equal(lifecycle.Acknowledge(Ack(first)), PrimitiveExecutionAckOutcome.ACCEPTED,
            "first execution ack");
        lifecycle.BeginExecution(RuntimeId, ExecutionId + 1, Hash(0x22));
        PrimitiveExecutionResultTransmission second = lifecycle.FinalizeExecution(
            Result("FAILED", Hash(0x22), ExecutionId + 1), 0UL);
        Equal(second.ExecutionId, ExecutionId + 1UL, "second execution identity");
        Equal(lifecycle.PhysicalExecutionCount, 2, "sequential physical executions");
        Equal(lifecycle.ResultGeneratedCount, 2, "sequential generated results");
        ++passed;
    }

    private static void TestDeterministicRetryClock()
    {
        var lifecycle = NewLifecycle();
        Complete(lifecycle);
        Assert(lifecycle.Poll(9UL) == null, "retry before interval");
        Assert(lifecycle.Poll(10UL) != null, "retry at interval");
        Equal(lifecycle.SendAttemptCount, 2, "retry at interval count");
        Assert(lifecycle.Poll(1000000UL) != null, "large jump retry");
        Assert(lifecycle.Poll(1000000UL) == null, "large jump burst");
        Equal(lifecycle.SendAttemptCount, 3, "large jump count");
        ++passed;
    }

    private static void TestAckCodecValidatesFullIdentity()
    {
        var lifecycle = NewLifecycle();
        PrimitiveExecutionResultTransmission transmission = Complete(lifecycle);
        PrimitiveExecutionV4Ack ack = Ack(transmission);
        byte[] first = XMProtocolV4.CanonicalAckPayload(ack);
        byte[] second = XMProtocolV4.CanonicalAckPayload(ack);
        EqualBytes(first, second, "ack canonical stability");
        Assert(first.Length > 0 && first[0] == 0x87, "ack map header");
        EqualBytes(first, Hex(
            "87d90a61636b5f737461747573d91044555241424c455f5245434549564544" +
            "d915636f6d6d616e645f73657175656e63655f68617368c4201111111111111111111111111111111111111111111111111111111111111111" +
            "d90c657865637574696f6e5f6964cf0102030405060708" +
            "d90c6d6573736167655f74797065d91b5072696d6974697665457865637574696f6e526573756c7441636b" +
            "d913726573756c745f7061796c6f61645f68617368c420af248672b0884f0450cfe51abb0e538ecc80e07ef0f213490dadd29770dacb88" +
            "d91372756e74696d655f696e7374616e63655f6964d916776f726b65722d30302d72756e74696d652d74657374" +
            "d90e736368656d615f76657273696f6ece00000004"), "ack golden bytes");
        Equal(XMProtocolV4.Sha256Hex(first),
            "e4aeeb85e25bba6c2d1a13311c1733c16d1723b3974fa546bd5157127b7f2574",
            "ack golden hash");
        Assert(XMProtocolV4.Sha256Hex(first) == XMProtocolV4.Sha256Hex(second),
            "ack hash stability");
        ack.result_payload_hash = new byte[31];
        Throws(() => XMProtocolV4.CanonicalAckPayload(ack), "ack hash length");
        ack.result_payload_hash = transmission.ResultPayloadHash;
        ack.message_type = "WrongAck";
        Throws(() => XMProtocolV4.CanonicalAckPayload(ack), "ack message type");
        ack.message_type = "PrimitiveExecutionResultAck";
        ack.ack_status = "RECEIVED";
        Throws(() => XMProtocolV4.CanonicalAckPayload(ack), "ack status");
        ++passed;
    }

    public static void Main(string[] args)
    {
        TestCompleteNormalPath();
        TestHundredRetransmissions();
        TestMatchingDurableReceivedClearsPending();
        TestWrongExecutionAckKeepsPending();
        TestWrongRuntimeAckKeepsPending();
        TestWrongResultHashAckKeepsPending();
        TestWrongCommandHashAckKeepsPending();
        TestWrongSchemaAckKeepsPending();
        TestDuplicateMatchingAckIdempotent();
        TestIncompleteCompleteDoesNotGenerate();
        TestRejectedBeforeExecution();
        TestCancelledPrefix();
        TestFailedPrefix();
        TestConflictingFinalizationRejected();
        TestSecondExecutionRejectedAndCannotClearA();
        TestNextExecutionStartsOnlyAfterAck();
        TestDeterministicRetryClock();
        TestAckCodecValidatesFullIdentity();
        Console.WriteLine("C# lifecycle tests: {0} passed", passed);
    }
}
