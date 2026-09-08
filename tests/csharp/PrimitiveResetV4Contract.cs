using System;
using XMflight;

public static class PrimitiveResetV4Contract
{
    private static void Equal(string actual, string expected, string label)
    {
        if (actual != expected) throw new Exception(label + " mismatch");
    }

    private static void Equal(float actual, float expected, string label)
    {
        if (Math.Abs(actual - expected) > 1e-6f) throw new Exception(label + " mismatch");
    }

    public static void Main()
    {
        var request = new PrimitiveResetV4Request {
            runtime_instance_id = "worker-00",
            episode_id = "episode-7",
            reset_id = "reset-7",
            start = new[] { 1.0f, 2.0f, 3.0f },
            goal = new[] { 4.0f, 5.0f, 6.0f },
        };
        byte[] requestBytes = PrimitiveResetV4WireCodec.SerializeRequest(request);
        PrimitiveResetV4Request decoded = PrimitiveResetV4WireCodec.DeserializeRequest(requestBytes);
        Equal(decoded.schema_version.ToString(), "4", "request schema");
        Equal(decoded.message_type, "PrimitiveResetRequest", "request type");
        Equal(decoded.runtime_instance_id, request.runtime_instance_id, "request runtime");
        Equal(decoded.episode_id, request.episode_id, "request episode");
        Equal(decoded.reset_id, request.reset_id, "request reset");
        Equal(decoded.start[2], 3.0f, "request start");
        Equal(decoded.goal[0], 4.0f, "request goal");

        var complete = new PrimitiveResetV4Complete {
            runtime_instance_id = "worker-00",
            episode_id = "episode-7",
            reset_id = "reset-7",
            observation_ref = new ObservationRefV4 {
                schema_version = 4,
                runtime_instance_id = "worker-00",
                episode_id = "episode-7",
                reset_id = "reset-7",
                state_id = 123,
                depth_id = "depth-123",
                sim_time_ns = 2460000000UL,
            },
        };
        byte[] completeBytes = PrimitiveResetV4WireCodec.SerializeComplete(complete);
        if (completeBytes == null || completeBytes.Length == 0)
            throw new Exception("complete bytes missing");

        var ack = new PrimitiveResetV4ReceivedAck {
            runtime_instance_id = "worker-00",
            episode_id = "episode-7",
            reset_id = "reset-7",
        };
        byte[] ackBytes = PrimitiveResetV4WireCodec.SerializeReceivedAck(ack);
        if (ackBytes == null || ackBytes.Length == 0)
            throw new Exception("ack bytes missing");

        byte[] secondRequestBytes = PrimitiveResetV4WireCodec.SerializeRequest(request);
        if (requestBytes.Length != secondRequestBytes.Length)
            throw new Exception("request serialization is not deterministic");
        for (int index = 0; index < requestBytes.Length; ++index)
            if (requestBytes[index] != secondRequestBytes[index])
                throw new Exception("request serialization is not deterministic");

        Console.WriteLine("C# reset contract tests: 4 passed");
    }
}
