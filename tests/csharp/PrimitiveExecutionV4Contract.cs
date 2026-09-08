using System;
using System.Collections.Generic;
using System.IO;
using XMflight;

public static class PrimitiveExecutionV4Contract
{
    private static byte[] Hex(string text)
    {
        text = text.Trim();
        byte[] bytes = new byte[text.Length / 2];
        for (int index = 0; index < bytes.Length; ++index)
            bytes[index] = Convert.ToByte(text.Substring(index * 2, 2), 16);
        return bytes;
    }

    private static void Equal(byte[] actual, byte[] expected, string label)
    {
        if (actual.Length != expected.Length) throw new Exception(label + " length mismatch");
        for (int index = 0; index < actual.Length; ++index)
            if (actual[index] != expected[index]) throw new Exception(label + " byte mismatch");
    }

    private static List<PrimitiveExecutionV4Frame> Frames()
    {
        var frames = new List<PrimitiveExecutionV4Frame>();
        for (uint index = 0; index < 25; ++index)
            frames.Add(new PrimitiveExecutionV4Frame {
                frame_index = index,
                command_id = 7000000000L + index,
                action = new[] { (float)index, -2.0f, 0.125f },
            });
        return frames;
    }

    private static PrimitiveExecutionV4Result Result(string status, byte[] commandHash)
    {
        var result = new PrimitiveExecutionV4Result {
            schema_version = 4,
            runtime_instance_id = "worker-00-runtime-test",
            execution_id = 0x0102030405060708UL,
            status = status,
            requested_frame_count = 25,
            command_sequence_hash = commandHash,
            result_generation = 0,
        };
        if (status == "COMPLETE") {
            result.applied_frame_count = 25;
            result.first_applied_state_id = 4000;
            result.endpoint_state_id = 4024;
            result.last_applied_frame_index = 24;
            result.reason_code = "NONE";
            result.endpoint_sim_time_ns = 5000000000UL;
            result.endpoint_observation_ref = new PrimitiveExecutionV4ObservationRef {
                schema_version = 4,
                runtime_instance_id = "worker-00-runtime-test",
                episode_id = "episode-v4-0001",
                reset_id = "reset-v4-0001",
                state_id = 4024,
                depth_id = "depth-v4-0001",
                sim_time_ns = 5000000000UL,
            };
        } else if (status == "REJECTED") {
            result.applied_frame_count = 0;
            result.last_applied_frame_index = -1;
            result.reason_code = "MALFORMED_COMMAND";
        } else if (status == "CANCELLED") {
            result.applied_frame_count = 11;
            result.first_applied_state_id = 7000;
            result.last_applied_frame_index = 10;
            result.reason_code = "STOP";
        } else {
            result.applied_frame_count = 7;
            result.first_applied_state_id = 8000;
            result.endpoint_state_id = 8007;
            result.last_applied_frame_index = 6;
            result.reason_code = "COLLISION";
            result.endpoint_sim_time_ns = 5140000000UL;
            result.endpoint_observation_ref = new PrimitiveExecutionV4ObservationRef {
                schema_version = 4,
                runtime_instance_id = "worker-00-runtime-test",
                episode_id = "episode-v4-0001",
                reset_id = "reset-v4-0001",
                state_id = 8007,
                depth_id = "depth-v4-failed-0001",
                sim_time_ns = 5140000000UL,
            };
        }
        return result;
    }

    public static void Main(string[] args)
    {
        string fixtureDir = args[0];
        byte[] commandBytes = XMProtocolV4.CanonicalCommandSequence(Frames());
        byte[] commandHash = Hex(File.ReadAllText(Path.Combine(
            fixtureDir, "complete.command_sequence_hash.hex")));
        Equal(commandBytes, Hex(File.ReadAllText(Path.Combine(
            fixtureDir, "complete.command_msgpack.hex"))), "command");
        if (XMProtocolV4.Sha256Hex(commandBytes) != File.ReadAllText(Path.Combine(
            fixtureDir, "complete.command_sequence_hash.sha256")).Trim())
            throw new Exception("command hash mismatch");
        foreach (string status in new[] { "complete", "rejected", "cancelled", "failed" }) {
            string wireStatus = status == "complete" ? "COMPLETE" :
                status == "rejected" ? "REJECTED" : status == "cancelled" ? "CANCELLED" : "FAILED";
            byte[] resultBytes = XMProtocolV4.CanonicalResultPayload(Result(wireStatus, commandHash));
            Equal(resultBytes, Hex(File.ReadAllText(Path.Combine(
                fixtureDir, status + ".result_msgpack.hex"))), status + " result");
            if (XMProtocolV4.Sha256Hex(resultBytes) != File.ReadAllText(Path.Combine(
                fixtureDir, status + ".result_payload_hash.sha256")).Trim())
                throw new Exception(status + " result hash mismatch");
        }
    }
}
