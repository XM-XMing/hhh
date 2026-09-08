using System;
using System.Collections.Generic;
using XMflight;

public static class PrimitiveExecutionV4RuntimeIntegrationContract
{
    private const string RuntimeId = "worker-00-runtime-test";
    private const ulong ExecutionId = 0x0102030405060708UL;
    private const string ExpectedCommandHash =
        "82e7f39c8f9bb8b6ea5a42cee108cf1e33a3ee872ba47a51167c4ffd25bed62c";

    private static int passed;

    private sealed class InMemoryResultSink : IPrimitiveExecutionResultSink
    {
        public readonly List<PrimitiveExecutionResultTransmission> Results =
            new List<PrimitiveExecutionResultTransmission>();

        public void Record(PrimitiveExecutionResultTransmission result)
        {
            Results.Add(result);
        }
    }

    private sealed class InMemoryResultTransport : IPrimitiveExecutionResultTransport
    {
        public readonly List<byte[]> Payloads = new List<byte[]>();
        public bool IsConnected { get; set; }

        public InMemoryResultTransport()
        {
            IsConnected = true;
        }

        public void Send(PrimitiveExecutionResultTransmission result)
        {
            Payloads.Add(result.SerializedCanonicalResultBytes);
        }
    }

    // This is a test-only model of the final manager's best-effort telemetry
    // adapter.  It deliberately does not add a production callback seam to
    // PrimitiveExecutionRuntimeIntegration.
    private interface IPrimitiveExecutionFrameTelemetrySink
    {
        void PublishFrameApplied(uint frameIndex, long stateId, ulong simTimeNs);
    }

    private sealed class DropFrame24TelemetrySink : IPrimitiveExecutionFrameTelemetrySink
    {
        public readonly List<uint> Delivered = new List<uint>();

        public void PublishFrameApplied(uint frameIndex, long stateId, ulong simTimeNs)
        {
            if (frameIndex == 24U) return;
            Delivered.Add(frameIndex);
        }
    }

    private sealed class ThrowOnceResultTransport : IPrimitiveExecutionResultTransport
    {
        public readonly List<byte[]> Payloads = new List<byte[]>();
        public bool IsConnected { get { return true; } }
        private bool throwNext = true;

        public void Send(PrimitiveExecutionResultTransmission result)
        {
            if (throwNext) {
                throwNext = false;
                throw new InvalidOperationException("injected first result send failure");
            }
            Payloads.Add(result.SerializedCanonicalResultBytes);
        }
    }

    // The final Unity owner keeps frame advancement and result integration in
    // separate owners.  This adapter drives both owners so the contract tests
    // retain the old end-to-end assertions without restoring the removed
    // telemetry callback seam.
    private sealed class RuntimeHarness
    {
        private readonly PrimitiveExecutionRuntimeIntegration integration;
        private readonly PrimitiveExecutionController controller =
            new PrimitiveExecutionController();
        private readonly List<uint> appliedFrameIndices = new List<uint>();
        private IPrimitiveExecutionFrameTelemetrySink telemetrySink;

        public RuntimeHarness(
            string runtimeInstanceId,
            IPrimitiveExecutionResultSink resultSink,
            ulong retryIntervalMs)
        {
            integration = new PrimitiveExecutionRuntimeIntegration(
                runtimeInstanceId, resultSink, retryIntervalMs);
        }

        public PrimitiveExecutionLifecycleState State { get { return integration.State; } }
        public int ResultGeneratedCount { get { return integration.ResultGeneratedCount; } }
        public bool IsActive { get { return integration.IsActive; } }
        public int PendingResultCount { get { return integration.PendingResultCount; } }
        public uint AppliedFrameCount { get { return controller.AppliedFrameCount; } }
        public int PhysicalExecutionCount { get { return controller.PhysicalExecutionCount; } }
        public long FirstAppliedStateId { get { return controller.FirstAppliedStateId; } }
        public long EndpointStateId { get { return controller.EndpointStateId; } }
        public ulong EndpointSimTimeNs { get { return controller.EndpointSimTimeNs; } }
        public IList<uint> AppliedFrameIndices { get { return appliedFrameIndices; } }

        public void SetFrameTelemetrySink(IPrimitiveExecutionFrameTelemetrySink sink)
        {
            telemetrySink = sink;
        }

        public void AttachResultTransport(IPrimitiveExecutionResultTransport transport)
        {
            integration.AttachResultTransport(transport);
        }

        public PrimitiveExecutionAckOutcome ReceiveResultAck(PrimitiveExecutionV4Ack ack)
        {
            return integration.ReceiveResultAck(ack);
        }

        public bool PollResultTransport(ulong nowMs)
        {
            return integration.PollResultTransport(nowMs);
        }

        public void BeginExecution(
            ulong executionId,
            IList<PrimitiveExecutionV4Frame> frames,
            ulong nowMs)
        {
            if (executionId > long.MaxValue)
                throw new Exception("test execution id must fit Unity controller");
            byte[] commandHash = XMProtocolV4.Sha256Bytes(
                XMProtocolV4.CanonicalCommandSequence(frames));
            var controllerFrames = new PrimitiveExecutionFrameMsg[frames.Count];
            for (int index = 0; index < frames.Count; ++index)
            {
                controllerFrames[index] = new PrimitiveExecutionFrameMsg {
                    frame_index = (int)frames[index].frame_index,
                    command_id = frames[index].command_id,
                    action = (float[])frames[index].action.Clone(),
                };
            }
            controller.Begin((long)executionId, controllerFrames, commandHash);
            integration.BeginExecution(executionId, commandHash, nowMs);
        }

        public bool RecordAppliedFrame(
            uint frameIndex,
            long stateId,
            ulong simTimeNs,
            ulong nowMs)
        {
            PrimitiveExecutionFrameMsg frame;
            if (!controller.TryAdvanceFrame(out frame)) return false;
            if (frame.frame_index != (int)frameIndex)
                throw new Exception("controller frame order changed");
            controller.RecordAppliedState(stateId, simTimeNs);
            appliedFrameIndices.Add(frameIndex);
            if (telemetrySink != null)
                telemetrySink.PublishFrameApplied(frameIndex, stateId, simTimeNs);
            return true;
        }

        public bool RecordEndpointObservation(
            string captureId,
            ulong captureTimeNs,
            string episodeId,
            string resetId,
            string observationRuntimeInstanceId,
            ulong nowMs)
        {
            bool accepted = integration.RecordEndpointObservation(
                captureId,
                captureTimeNs,
                episodeId,
                resetId,
                observationRuntimeInstanceId,
                controller.ResultContext(),
                nowMs);
            if (accepted) controller.MarkResultReady();
            return accepted;
        }

        public bool RecordTerminalFailureObservation(
            string reason,
            long terminalStateId,
            ulong terminalSimTimeNs,
            string captureId,
            ulong captureTimeNs,
            string episodeId,
            string resetId,
            string observationRuntimeInstanceId,
            ulong nowMs)
        {
            bool accepted = integration.RecordTerminalFailureObservation(
                reason,
                terminalStateId,
                terminalSimTimeNs,
                captureId,
                captureTimeNs,
                episodeId,
                resetId,
                observationRuntimeInstanceId,
                controller.ResultContext(),
                nowMs);
            if (accepted) controller.MarkResultReady();
            return accepted;
        }

        public bool CancelActiveExecution(string reason, ulong nowMs)
        {
            bool accepted = integration.CancelActiveExecution(
                reason, controller.ResultContext(), nowMs);
            if (accepted) controller.MarkResultReady();
            return accepted;
        }

        public bool FailActiveExecution(string reason, ulong nowMs)
        {
            bool accepted = integration.FailActiveExecution(
                reason, controller.ResultContext(), nowMs);
            if (accepted) controller.MarkResultReady();
            return accepted;
        }

        public bool RejectOverlappingExecution(
            ulong executionId,
            IList<PrimitiveExecutionV4Frame> frames,
            ulong nowMs)
        {
            return integration.RejectOverlappingExecution(executionId, frames, nowMs);
        }

        public bool RejectExecutionIfBusy(
            ulong executionId,
            IList<PrimitiveExecutionV4Frame> frames,
            ulong nowMs)
        {
            return integration.RejectExecutionIfBusy(executionId, frames, nowMs);
        }

        public bool RejectBeforeExecution(
            ulong executionId,
            byte[] commandHash,
            string reason,
            ulong nowMs)
        {
            return integration.RejectBeforeExecution(
                executionId, commandHash, reason, nowMs);
        }
    }

    private static List<PrimitiveExecutionV4Frame> Frames()
    {
        return Frames(7000000000L);
    }

    private static List<PrimitiveExecutionV4Frame> Frames(long commandBase)
    {
        var frames = new List<PrimitiveExecutionV4Frame>();
        for (uint index = 0; index < 25; ++index)
        {
            frames.Add(new PrimitiveExecutionV4Frame {
                frame_index = index,
                command_id = commandBase + index,
                action = new[] { (float)index, -2.0f, 0.125f },
            });
        }
        return frames;
    }

    private static PrimitiveExecutionV4Ack AckFor(
        PrimitiveExecutionResultTransmission result)
    {
        return new PrimitiveExecutionV4Ack {
            schema_version = 4,
            message_type = "PrimitiveExecutionResultAck",
            runtime_instance_id = result.RuntimeInstanceId,
            execution_id = result.ExecutionId,
            ack_status = "DURABLE_RECEIVED",
            result_payload_hash = result.ResultPayloadHash,
            command_sequence_hash = result.CommandSequenceHash,
        };
    }

    private static void Equal<T>(T actual, T expected, string label)
    {
        if (!EqualityComparer<T>.Default.Equals(actual, expected))
            throw new Exception(label + " expected " + expected + " got " + actual);
    }

    private static void Assert(bool condition, string label)
    {
        if (!condition) throw new Exception(label);
    }

    private static string Hex(byte[] bytes)
    {
        var result = new System.Text.StringBuilder(bytes.Length * 2);
        foreach (byte value in bytes) result.Append(value.ToString("x2"));
        return result.ToString();
    }

    private static bool SameBytes(byte[] left, byte[] right)
    {
        if (left == null || right == null || left.Length != right.Length) return false;
        for (int index = 0; index < left.Length; ++index)
            if (left[index] != right[index]) return false;
        return true;
    }

    private static void FinalizeComplete(RuntimeHarness runtime)
    {
        Assert(runtime.RecordEndpointObservation(
            "depth-25",
            561224825UL,
            "m1-episode-00",
            "m1-reset-00",
            RuntimeId,
            25UL), "endpoint observation finalization accepted");
    }

    private static void NormalCompleteUsesRuntimeEvents()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);

        for (uint index = 0; index < 25; ++index)
        {
            runtime.RecordAppliedFrame(
                index,
                4000L + index,
                5000000000UL + index * 20000000UL,
                index);
        }
        FinalizeComplete(runtime);

        Equal(runtime.PhysicalExecutionCount, 1, "physical execution count");
        Equal(runtime.AppliedFrameCount, (uint)25, "applied frame count");
        Equal(runtime.AppliedFrameIndices.Count, 25, "unique frame count");
        for (int index = 0; index < 25; ++index)
            Equal(runtime.AppliedFrameIndices[index], (uint)index, "frame order");
        Equal(sink.Results.Count, 1, "result count");
        Equal(runtime.ResultGeneratedCount, 1, "generated result count");
        Equal(runtime.State, PrimitiveExecutionLifecycleState.FINAL_RESULT_PENDING_ACK,
            "lifecycle state");

        PrimitiveExecutionResultTransmission result = sink.Results[0];
        Equal(result.Status, "COMPLETE", "result status");
        Equal(result.ExecutionId, ExecutionId, "execution id");
        Equal(result.ResultGeneration, (uint)0, "result generation");
        Equal(Hex(result.CommandSequenceHash), ExpectedCommandHash, "command hash");
        Equal(runtime.FirstAppliedStateId, 4000L, "first state id");
        Equal(runtime.EndpointStateId, 4024L, "endpoint state id");
        Equal(runtime.EndpointSimTimeNs, 5480000000UL, "endpoint sim time");
        Assert(result.EndpointObservationRef != null, "complete observation ref");
        Equal(result.EndpointObservationRef.schema_version, (uint)4,
            "observation schema version");
        Equal(result.EndpointObservationRef.runtime_instance_id, RuntimeId,
            "observation runtime id");
        Equal(result.EndpointObservationRef.episode_id, "m1-episode-00",
            "observation episode id");
        Equal(result.EndpointObservationRef.reset_id, "m1-reset-00",
            "observation reset id");
        Equal(result.EndpointObservationRef.state_id, 4024L,
            "observation state id");
        Equal(result.EndpointObservationRef.depth_id, "depth-25",
            "observation depth id");
        Equal(result.EndpointObservationRef.sim_time_ns, 5480000000UL,
            "observation sim time");
        ++passed;
    }

    private static void TwentyFourFramesCannotComplete()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);

        for (uint index = 0; index < 24; ++index)
        {
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);
        }

        Equal(runtime.AppliedFrameCount, (uint)24, "partial applied frame count");
        Equal(sink.Results.Count, 0, "partial result count");
        Equal(runtime.ResultGeneratedCount, 0, "partial generated result count");
        Assert(runtime.IsActive, "partial execution remains active");
        ++passed;
    }

    private static void ExtraTickAfterCompleteDoesNotReapplyFrame()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 25; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);
        FinalizeComplete(runtime);

        bool applied = runtime.RecordAppliedFrame(24U, 4024L, 5480000000UL, 25UL);
        Assert(!applied, "extra tick must not apply");
        Equal(runtime.AppliedFrameCount, (uint)25, "extra tick applied count");
        Equal(sink.Results.Count, 1, "extra tick result count");
        Equal(runtime.PhysicalExecutionCount, 1, "extra tick physical execution count");
        Equal(runtime.ResultGeneratedCount, 1, "extra tick generated result count");
        ++passed;
    }

    private static void StopRecordsCancelledAppliedPrefix()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 11; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);

        Assert(runtime.CancelActiveExecution("STOP", 11UL), "stop cancellation accepted");
        Equal(sink.Results.Count, 1, "stop result count");
        PrimitiveExecutionResultTransmission result = sink.Results[0];
        Equal(result.Status, "CANCELLED", "stop status");
        Equal(result.ReasonCode, "STOP", "stop reason");
        Equal(result.AppliedFrameCount, (uint)11, "stop applied prefix");
        Equal(result.LastAppliedFrameIndex, 10, "stop last frame");
        Equal(runtime.State, PrimitiveExecutionLifecycleState.FINAL_RESULT_PENDING_ACK,
            "stop lifecycle state");
        Assert(!runtime.IsActive, "stop active execution cleared after result record");
        ++passed;
    }

    private static void ResetRecordsCancelledAppliedPrefix()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 7; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);

        Assert(runtime.CancelActiveExecution("RESET", 7UL), "reset cancellation accepted");
        Equal(sink.Results.Count, 1, "reset result count");
        PrimitiveExecutionResultTransmission result = sink.Results[0];
        Equal(result.Status, "CANCELLED", "reset status");
        Equal(result.ReasonCode, "RESET", "reset reason");
        Equal(result.AppliedFrameCount, (uint)7, "reset applied prefix");
        Equal(result.LastAppliedFrameIndex, 6, "reset last frame");
        ++passed;
    }

    private static void ExternalOverrideRecordsCancellation()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 5; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);

        Assert(runtime.CancelActiveExecution("EXTERNAL_OVERRIDE", 5UL),
            "external override cancellation accepted");
        Equal(sink.Results.Count, 1, "override result count");
        PrimitiveExecutionResultTransmission result = sink.Results[0];
        Equal(result.Status, "CANCELLED", "override status");
        Equal(result.ReasonCode, "EXTERNAL_OVERRIDE", "override reason");
        Equal(result.AppliedFrameCount, (uint)5, "override applied prefix");
        ++passed;
    }

    private static void OverlapRejectsBWithoutChangingA()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);

        Assert(runtime.RejectOverlappingExecution(ExecutionId + 1UL, Frames(), 0UL),
            "overlap rejection accepted");
        Equal(sink.Results.Count, 1, "overlap rejection result count");
        PrimitiveExecutionResultTransmission rejected = sink.Results[0];
        Equal(rejected.ExecutionId, ExecutionId + 1UL, "overlap B execution id");
        Equal(rejected.Status, "REJECTED", "overlap B status");
        Equal(rejected.ReasonCode, "OVERLAPPING_EXECUTION", "overlap B reason");
        Equal(rejected.AppliedFrameCount, (uint)0, "overlap B applied");
        Assert(runtime.IsActive, "overlap A remains active");
        Equal(runtime.PhysicalExecutionCount, 1, "overlap A physical count");

        for (uint index = 0; index < 25; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);
        FinalizeComplete(runtime);
        Equal(sink.Results.Count, 2, "overlap final result count");
        Equal(sink.Results[1].ExecutionId, ExecutionId, "overlap A execution id");
        Equal(sink.Results[1].Status, "COMPLETE", "overlap A status");
        ++passed;
    }

    private static void MalformedCommandRecordsImmediateRejection()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        Assert(runtime.RejectBeforeExecution(ExecutionId, new byte[32],
            "MALFORMED_COMMAND", 0UL), "malformed rejection accepted");
        Equal(sink.Results.Count, 1, "malformed result count");
        PrimitiveExecutionResultTransmission result = sink.Results[0];
        Equal(result.Status, "REJECTED", "malformed status");
        Equal(result.ReasonCode, "MALFORMED_COMMAND", "malformed reason");
        Equal(result.AppliedFrameCount, (uint)0, "malformed applied");
        Equal(result.LastAppliedFrameIndex, -1, "malformed last frame");
        Equal(runtime.PhysicalExecutionCount, 0, "malformed physical execution count");
        Equal(runtime.State, PrimitiveExecutionLifecycleState.FINAL_RESULT_PENDING_ACK,
            "malformed result pending state");
        ++passed;
    }

    private static void RejectedResultUsesReliableTransport()
    {
        var sink = new InMemoryResultSink();
        var transport = new InMemoryResultTransport();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.AttachResultTransport(transport);

        Assert(runtime.RejectBeforeExecution(ExecutionId, new byte[32],
            "MALFORMED_COMMAND", 0UL), "reliable rejection accepted");
        Equal(sink.Results.Count, 1, "reliable rejection result count");
        Equal(transport.Payloads.Count, 1, "reliable rejection first send count");
        Equal(runtime.PendingResultCount, 1, "reliable rejection pending count");
        Equal(runtime.PhysicalExecutionCount, 0, "reliable rejection physical count");

        PrimitiveExecutionResultTransmission result = sink.Results[0];
        Equal(runtime.ReceiveResultAck(AckFor(result)),
            PrimitiveExecutionAckOutcome.ACCEPTED, "reliable rejection ACK");
        Equal(runtime.PendingResultCount, 0, "reliable rejection pending after ACK");
        Equal(runtime.ReceiveResultAck(AckFor(result)),
            PrimitiveExecutionAckOutcome.DUPLICATE, "reliable rejection duplicate ACK");
        Equal(runtime.ResultGeneratedCount, 1, "reliable rejection generation count");
        ++passed;
    }

    private static void AllRejectedReasonsUseReliableTransport()
    {
        string[] reasons = {
            "MALFORMED_COMMAND",
            "INVALID_FRAME_COUNT",
            "CTRL_LATENCY_NOT_ZERO",
        };
        for (int index = 0; index < reasons.Length; ++index) {
            var sink = new InMemoryResultSink();
            var transport = new InMemoryResultTransport();
            var runtime = new RuntimeHarness(
                RuntimeId, sink, 10UL);
            runtime.AttachResultTransport(transport);
            Assert(runtime.RejectBeforeExecution(
                ExecutionId + (ulong)index + 10UL, new byte[32], reasons[index], 0UL),
                reasons[index] + " reliable rejection accepted");
            Equal(transport.Payloads.Count, 1,
                reasons[index] + " reliable first send count");
            Equal(runtime.PendingResultCount, 1,
                reasons[index] + " reliable pending count");
            Equal(runtime.ReceiveResultAck(AckFor(sink.Results[0])),
                PrimitiveExecutionAckOutcome.ACCEPTED,
                reasons[index] + " reliable ACK");
            Equal(runtime.PendingResultCount, 0,
                reasons[index] + " reliable pending after ACK");
        }

        var overlapSink = new InMemoryResultSink();
        var overlapTransport = new InMemoryResultTransport();
        var overlapRuntime = new RuntimeHarness(
            RuntimeId, overlapSink, 10UL);
        overlapRuntime.AttachResultTransport(overlapTransport);
        overlapRuntime.BeginExecution(ExecutionId + 20UL, Frames(), 0UL);
        Assert(overlapRuntime.RejectExecutionIfBusy(
            ExecutionId + 21UL, Frames(8100000000L), 1UL),
            "OVERLAPPING_EXECUTION reliable rejection accepted");
        Equal(overlapTransport.Payloads.Count, 1,
            "OVERLAPPING_EXECUTION reliable first send count");
        Equal(overlapRuntime.PendingResultCount, 1,
            "OVERLAPPING_EXECUTION reliable pending count");
        Equal(overlapRuntime.ReceiveResultAck(AckFor(overlapSink.Results[0])),
            PrimitiveExecutionAckOutcome.ACCEPTED,
            "OVERLAPPING_EXECUTION reliable ACK");
        Equal(overlapRuntime.PendingResultCount, 0,
            "OVERLAPPING_EXECUTION reliable pending after ACK");
        Equal(overlapRuntime.PhysicalExecutionCount, 1,
            "OVERLAPPING_EXECUTION physical count");
        ++passed;
    }

    private static void InvalidFrameCountRecordsImmediateRejection()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        Assert(runtime.RejectBeforeExecution(ExecutionId, new byte[32],
            "INVALID_FRAME_COUNT", 0UL), "frame count rejection accepted");
        Equal(sink.Results.Count, 1, "frame count result count");
        Equal(sink.Results[0].Status, "REJECTED", "frame count status");
        Equal(sink.Results[0].ReasonCode, "INVALID_FRAME_COUNT", "frame count reason");
        Equal(sink.Results[0].AppliedFrameCount, (uint)0, "frame count applied");
        ++passed;
    }

    private static void NonZeroControlLatencyRecordsImmediateRejection()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        Assert(runtime.RejectBeforeExecution(ExecutionId, new byte[32],
            "CTRL_LATENCY_NOT_ZERO", 0UL), "latency rejection accepted");
        Equal(sink.Results.Count, 1, "latency result count");
        Equal(sink.Results[0].Status, "REJECTED", "latency status");
        Equal(sink.Results[0].ReasonCode, "CTRL_LATENCY_NOT_ZERO", "latency reason");
        Equal(sink.Results[0].AppliedFrameCount, (uint)0, "latency applied");
        ++passed;
    }

    private static void InternalFailurePreservesAppliedPrefix()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 8; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);

        Assert(runtime.FailActiveExecution("INTERNAL_ERROR", 8UL),
            "internal failure accepted");
        Equal(sink.Results.Count, 1, "internal failure result count");
        PrimitiveExecutionResultTransmission result = sink.Results[0];
        Equal(result.Status, "FAILED", "internal failure status");
        Equal(result.ReasonCode, "INTERNAL_ERROR", "internal failure reason");
        Equal(result.AppliedFrameCount, (uint)8, "internal failure applied prefix");
        Equal(result.LastAppliedFrameIndex, 7, "internal failure last frame");
        ++passed;
    }

    private static void CollisionFailurePreservesAppliedPrefix()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.BeginExecution(ExecutionId, Frames(), 10UL);
        for (uint index = 0; index < 11; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, 10UL + index);

        Assert(runtime.RecordTerminalFailureObservation(
            "COLLISION", 4011L, 5220000000UL, "depth-terminal-1", 5221000000UL,
            "episode-v4-0001", "reset-v4-0001", RuntimeId, 21UL),
            "collision failure with exact observation accepted");
        Equal(sink.Results.Count, 1, "collision result count");
        PrimitiveExecutionResultTransmission result = sink.Results[0];
        Equal(result.Status, "FAILED", "collision status");
        Equal(result.ReasonCode, "COLLISION", "collision reason");
        Equal(result.AppliedFrameCount, (uint)11, "collision applied prefix");
        Equal(result.LastAppliedFrameIndex, 10, "collision last frame");
        Equal(result.EndpointStateId.Value, 4011L, "collision endpoint state");
        Assert(result.EndpointObservationRef != null,
            "collision must carry exact endpoint observation");
        ++passed;
    }

    private static void PhysicsCompleteObservationFailurePreservesAppliedFrames()
    {
        var sink = new InMemoryResultSink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 25; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);

        Assert(runtime.FailActiveExecution(
            "PHYSICS_COMPLETE_OBSERVATION_FAILED", 25UL),
            "observation failure accepted after physics completion");
        Equal(sink.Results.Count, 1, "observation failure result count");
        PrimitiveExecutionResultTransmission result = sink.Results[0];
        Equal(result.Status, "FAILED", "observation failure status");
        Equal(result.ReasonCode, "PHYSICS_COMPLETE_OBSERVATION_FAILED",
            "observation failure reason");
        Equal(result.AppliedFrameCount, (uint)25, "observation failure applied count");
        Equal(result.LastAppliedFrameIndex, 24, "observation failure last frame");
        Assert(result.EndpointObservationRef == null,
            "observation failure must not claim endpoint ref");
        Assert(!runtime.IsActive, "observation failure must clear active execution");
        Assert(!runtime.FailActiveExecution(
            "PHYSICS_COMPLETE_OBSERVATION_FAILED", 26UL),
            "observation failure must be terminal exactly once");
        Equal(sink.Results.Count, 1, "observation failure terminal count");
        ++passed;
    }

    private static void TelemetryDropDoesNotSuppressCompleteResult()
    {
        var sink = new InMemoryResultSink();
        var telemetry = new DropFrame24TelemetrySink();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.SetFrameTelemetrySink(telemetry);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 25; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);
        FinalizeComplete(runtime);

        Equal(telemetry.Delivered.Count, 24, "telemetry delivered count");
        Assert(!telemetry.Delivered.Contains(24U), "telemetry frame 24 dropped");
        Equal(runtime.AppliedFrameCount, (uint)25, "physical applied count");
        Equal(sink.Results.Count, 1, "telemetry-drop result count");
        Equal(sink.Results[0].Status, "COMPLETE", "telemetry-drop status");
        ++passed;
    }

    private static void PendingResultDoesNotBlockNextExecution()
    {
        var sink = new InMemoryResultSink();
        var transport = new InMemoryResultTransport();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.AttachResultTransport(transport);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 25; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);
        FinalizeComplete(runtime);
        PrimitiveExecutionResultTransmission firstResult = sink.Results[0];
        byte[] firstPayload = firstResult.SerializedCanonicalResultBytes;
        byte[] firstHash = firstResult.ResultPayloadHash;
        Equal(runtime.PendingResultCount, 1, "first result pending count");
        Equal(transport.Payloads.Count, 1, "first result send count");

        Assert(runtime.PollResultTransport(35UL), "first result retry eligibility");
        Equal(transport.Payloads.Count, 2, "first result retry count");
        Assert(SameBytes(transport.Payloads[0], transport.Payloads[1]),
            "first retry payload must be immutable");

        runtime.BeginExecution(ExecutionId + 1UL, Frames(8000000000L), 25UL);
        for (uint index = 0; index < 25; ++index)
            runtime.RecordAppliedFrame(index, 5000L + index,
                6000000000UL + index * 20000000UL, 25UL + index);
        Assert(runtime.RecordEndpointObservation(
            "depth-26", 661224825UL, "m1-episode-00", "m1-reset-00",
            RuntimeId, 50UL), "second endpoint observation accepted");

        Equal(runtime.PhysicalExecutionCount, 2, "two physical executions");
        Equal(runtime.ResultGeneratedCount, 2, "two result generations");
        Equal(sink.Results.Count, 2, "two terminal results");
        Equal(transport.Payloads.Count, 3, "second result initial send count");
        Equal(sink.Results[0].ExecutionId, ExecutionId,
            "first execution identity preserved");
        Equal(sink.Results[1].ExecutionId, ExecutionId + 1UL,
            "second execution identity isolated");
        Assert(Hex(sink.Results[0].CommandSequenceHash) !=
            Hex(sink.Results[1].CommandSequenceHash),
            "second command identity must differ");
        Assert(SameBytes(sink.Results[0].SerializedCanonicalResultBytes, firstPayload),
            "first pending payload must not be overwritten");
        Assert(SameBytes(sink.Results[0].ResultPayloadHash, firstHash),
            "first pending hash must not be overwritten");
        Equal(runtime.PendingResultCount, 2, "two pending result identities");
        Equal(runtime.ReceiveResultAck(AckFor(sink.Results[0])),
            PrimitiveExecutionAckOutcome.ACCEPTED, "first ACK outcome");
        Equal(runtime.PendingResultCount, 1, "second result remains pending");
        Equal(runtime.ReceiveResultAck(AckFor(sink.Results[0])),
            PrimitiveExecutionAckOutcome.DUPLICATE, "duplicate first ACK outcome");
        Equal(runtime.ReceiveResultAck(AckFor(sink.Results[1])),
            PrimitiveExecutionAckOutcome.ACCEPTED, "second ACK outcome");
        Equal(runtime.PendingResultCount, 0, "all results acknowledged");
        Equal(runtime.State, PrimitiveExecutionLifecycleState.ACKED,
            "delivery state after both ACKs");
        ++passed;
    }

    private static void Frame24PendingEndpointRejectsNewExecution()
    {
        var sink = new InMemoryResultSink();
        var transport = new InMemoryResultTransport();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.AttachResultTransport(transport);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 25; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);

        Assert(runtime.IsActive, "frame24 endpoint capture remains owned");
        Assert(runtime.RejectExecutionIfBusy(
            ExecutionId + 1UL, Frames(8000000000L), 25UL),
            "overlap during endpoint capture is rejected");
        Equal(runtime.PhysicalExecutionCount, 1,
            "endpoint-pending overlap physical count");
        Equal(sink.Results.Count, 1, "endpoint-pending overlap result count");
        Equal(sink.Results[0].ExecutionId, ExecutionId + 1UL,
            "endpoint-pending overlap execution id");
        Equal(sink.Results[0].Status, "REJECTED",
            "endpoint-pending overlap status");
        Equal(sink.Results[0].ReasonCode, "OVERLAPPING_EXECUTION",
            "endpoint-pending overlap reason");
        Assert(runtime.IsActive, "execution A remains active during endpoint capture");

        FinalizeComplete(runtime);
        Assert(!runtime.IsActive, "execution A becomes terminal after endpoint capture");
        runtime.BeginExecution(ExecutionId + 2UL, Frames(9000000000L), 50UL);
        Equal(runtime.PhysicalExecutionCount, 2,
            "new execution starts after A is terminal");
        ++passed;
    }

    private static void CompleteFirstSendFailureRetainsTerminalPending()
    {
        var sink = new InMemoryResultSink();
        var transport = new ThrowOnceResultTransport();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.AttachResultTransport(transport);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 25; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);

        bool accepted = false;
        try { accepted = runtime.RecordEndpointObservation(
            "depth-25", 561224825UL, "m1-episode-00", "m1-reset-00",
            RuntimeId, 25UL); }
        catch (Exception) { }
        Assert(accepted, "complete local terminal transition survives send failure");
        Assert(!runtime.IsActive, "complete send failure clears active execution");
        Equal(runtime.ResultGeneratedCount, 1, "complete send failure generation count");
        Equal(runtime.PendingResultCount, 1, "complete send failure pending count");
        Equal(transport.Payloads.Count, 0, "complete failed first send count");
        Assert(runtime.PollResultTransport(35UL), "complete retry after failed first send");
        Equal(transport.Payloads.Count, 1, "complete retry send count");
        Equal(runtime.ReceiveResultAck(AckFor(sink.Results[0])),
            PrimitiveExecutionAckOutcome.ACCEPTED, "complete retry ACK");
        Equal(runtime.PendingResultCount, 0, "complete pending after retry ACK");
        ++passed;
    }

    private static void FailedFirstSendFailureRetainsTerminalPending()
    {
        var sink = new InMemoryResultSink();
        var transport = new ThrowOnceResultTransport();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.AttachResultTransport(transport);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 8; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);

        bool accepted = false;
        try { accepted = runtime.FailActiveExecution("INTERNAL_ERROR", 8UL); }
        catch (Exception) { }
        Assert(accepted, "failed local terminal transition survives send failure");
        Assert(!runtime.IsActive, "failed send failure clears active execution");
        Equal(sink.Results[0].Status, "FAILED", "failed retry status");
        Equal(runtime.PendingResultCount, 1, "failed send failure pending count");
        Assert(runtime.PollResultTransport(18UL), "failed retry after failed first send");
        Equal(runtime.ReceiveResultAck(AckFor(sink.Results[0])),
            PrimitiveExecutionAckOutcome.ACCEPTED, "failed retry ACK");
        Equal(runtime.PendingResultCount, 0, "failed pending after retry ACK");
        ++passed;
    }

    private static void CancelledFirstSendFailureRetainsTerminalPending()
    {
        var sink = new InMemoryResultSink();
        var transport = new ThrowOnceResultTransport();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.AttachResultTransport(transport);
        runtime.BeginExecution(ExecutionId, Frames(), 0UL);
        for (uint index = 0; index < 8; ++index)
            runtime.RecordAppliedFrame(index, 4000L + index,
                5000000000UL + index * 20000000UL, index);

        bool accepted = false;
        try { accepted = runtime.CancelActiveExecution("STOP", 8UL); }
        catch (Exception) { }
        Assert(accepted, "cancel local terminal transition survives send failure");
        Assert(!runtime.IsActive, "cancel send failure clears active execution");
        Equal(sink.Results[0].Status, "CANCELLED", "cancel retry status");
        Equal(runtime.PendingResultCount, 1, "cancel send failure pending count");
        Assert(runtime.PollResultTransport(18UL), "cancel retry after failed first send");
        Equal(runtime.ReceiveResultAck(AckFor(sink.Results[0])),
            PrimitiveExecutionAckOutcome.ACCEPTED, "cancel retry ACK");
        Equal(runtime.PendingResultCount, 0, "cancel pending after retry ACK");
        ++passed;
    }

    private static void RejectedFirstSendFailureRetainsTerminalPending()
    {
        var sink = new InMemoryResultSink();
        var transport = new ThrowOnceResultTransport();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.AttachResultTransport(transport);

        bool accepted = false;
        try { accepted = runtime.RejectBeforeExecution(ExecutionId, new byte[32],
            "MALFORMED_COMMAND", 0UL); }
        catch (Exception) { }
        Assert(accepted, "reject local terminal transition survives send failure");
        Assert(!runtime.IsActive, "reject never becomes physically active");
        Equal(runtime.PhysicalExecutionCount, 0, "reject physical count");
        Equal(runtime.ResultGeneratedCount, 1, "reject generation count");
        Equal(runtime.PendingResultCount, 1, "reject send failure pending count");
        Assert(runtime.PollResultTransport(10UL), "reject retry after failed first send");
        Equal(runtime.ReceiveResultAck(AckFor(sink.Results[0])),
            PrimitiveExecutionAckOutcome.ACCEPTED, "reject retry ACK");
        Equal(runtime.PendingResultCount, 0, "reject pending after retry ACK");
        ++passed;
    }

    private static void DuplicateRejectedIdentityReusesImmutableResult()
    {
        var sink = new InMemoryResultSink();
        var transport = new InMemoryResultTransport();
        var runtime = new RuntimeHarness(
            RuntimeId, sink, 10UL);
        runtime.AttachResultTransport(transport);
        byte[] commandHash = new byte[32];
        commandHash[0] = 0x42;

        Assert(runtime.RejectBeforeExecution(
            ExecutionId, commandHash, "SCHEMA_MISMATCH", 0UL),
            "first rejection accepted");
        byte[] payload = sink.Results[0].SerializedCanonicalResultBytes;
        Assert(!runtime.RejectBeforeExecution(
            ExecutionId, commandHash, "MALFORMED_COMMAND", 1UL),
            "duplicate rejection must not generate a second result");

        Equal(runtime.ResultGeneratedCount, 1, "duplicate rejection generation count");
        Equal(runtime.PhysicalExecutionCount, 0, "duplicate rejection physical count");
        Equal(sink.Results.Count, 1, "duplicate rejection sink count");
        Equal(transport.Payloads.Count, 2, "duplicate rejection resend count");
        Assert(SameBytes(payload, transport.Payloads[1]),
            "duplicate rejection must resend immutable payload");

        bool conflictRejected = false;
        commandHash[0] = 0x43;
        try {
            runtime.RejectBeforeExecution(
                ExecutionId, commandHash, "SCHEMA_MISMATCH", 2UL);
        }
        catch (PrimitiveExecutionLifecycleException) {
            conflictRejected = true;
        }
        Assert(conflictRejected, "conflicting rejection identity must fail closed");
        Equal(runtime.ResultGeneratedCount, 1, "conflict generation count");
        ++passed;
    }

    public static void Main(string[] args)
    {
        NormalCompleteUsesRuntimeEvents();
        TwentyFourFramesCannotComplete();
        ExtraTickAfterCompleteDoesNotReapplyFrame();
        StopRecordsCancelledAppliedPrefix();
        ResetRecordsCancelledAppliedPrefix();
        ExternalOverrideRecordsCancellation();
        OverlapRejectsBWithoutChangingA();
        MalformedCommandRecordsImmediateRejection();
        InvalidFrameCountRecordsImmediateRejection();
        NonZeroControlLatencyRecordsImmediateRejection();
        InternalFailurePreservesAppliedPrefix();
        CollisionFailurePreservesAppliedPrefix();
        PhysicsCompleteObservationFailurePreservesAppliedFrames();
        TelemetryDropDoesNotSuppressCompleteResult();
        PendingResultDoesNotBlockNextExecution();
        RejectedResultUsesReliableTransport();
        AllRejectedReasonsUseReliableTransport();
        Frame24PendingEndpointRejectsNewExecution();
        CompleteFirstSendFailureRetainsTerminalPending();
        FailedFirstSendFailureRetainsTerminalPending();
        CancelledFirstSendFailureRetainsTerminalPending();
        RejectedFirstSendFailureRetainsTerminalPending();
        DuplicateRejectedIdentityReusesImmutableResult();
        Console.WriteLine("C# runtime integration tests: {0} passed", passed);
    }
}
