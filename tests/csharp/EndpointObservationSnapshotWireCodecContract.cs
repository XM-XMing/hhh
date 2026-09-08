using System;
using System.Collections.Generic;
using MessagePack;
using XMflight;

public static class EndpointObservationSnapshotWireCodecContract
{
    private static void Require(bool value, string message)
    {
        if (!value) throw new Exception(message);
    }

    private static byte[] Hash(byte value)
    {
        byte[] hash = new byte[32];
        for (int index = 0; index < hash.Length; index++) hash[index] = value;
        return hash;
    }

    private static Dictionary<string, object> RefMap()
    {
        return new Dictionary<string, object> {
            { "schema_version", 4U }, { "runtime_instance_id", "worker-00-runtime-test" },
            { "episode_id", "episode-1" }, { "reset_id", "reset-1" },
            { "state_id", 4024L }, { "depth_id", "depth-4024" },
            { "sim_time_ns", 5000000000UL },
        };
    }

    private static void RequestRoundTrip()
    {
        byte[] raw = MessagePackSerializer.Serialize(new Dictionary<string, object> {
            { "schema_version", 4U }, { "message_type", "SnapshotRequest" },
            { "observation_ref", RefMap() }, { "execution_id", 9UL },
            { "result_payload_hash", Hash(0x11) }, { "command_sequence_hash", Hash(0x22) },
        });
        SnapshotRequestV4 request;
        Require(EndpointObservationSnapshotWireCodec.TryDeserializeRequest(raw, out request),
            "request must decode");
        Require(request.execution_id == 9UL && request.observation_ref.state_id == 4024L,
            "request identity mismatch");
    }

    private static void AckRoundTrip()
    {
        byte[] raw = MessagePackSerializer.Serialize(new Dictionary<string, object> {
            { "schema_version", 4U }, { "message_type", "SnapshotAck" },
            { "observation_ref", RefMap() }, { "snapshot_hash", Hash(0x33) },
        });
        SnapshotAckV4 ack;
        Require(EndpointObservationSnapshotWireCodec.TryDeserializeAck(raw, out ack), "ack must decode");
        Require(ack.snapshot_hash[0] == 0x33 && ack.observation_ref.depth_id == "depth-4024",
            "ack identity mismatch");
    }

    public static int Main()
    {
        RequestRoundTrip();
        AckRoundTrip();
        Console.WriteLine("C# endpoint snapshot wire codec tests: 2 passed");
        return 0;
    }
}
