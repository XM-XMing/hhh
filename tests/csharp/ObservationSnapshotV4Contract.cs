using System;
using System.IO;
using XMflight;

public static class ObservationSnapshotV4Contract
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

    private static ObservationRefV4 ObservationRef()
    {
        return new ObservationRefV4 {
            schema_version = 4,
            runtime_instance_id = "worker-00-runtime-test",
            episode_id = "episode-54",
            reset_id = "reset-54",
            state_id = 1514,
            depth_id = "depth-1514",
            sim_time_ns = 559999987UL,
        };
    }

    private static EndpointObservationSnapshotV4 Snapshot(byte[] snapshotHash)
    {
        return new EndpointObservationSnapshotV4 {
            observation_ref = ObservationRef(),
            state_bytes = Hex("00ff1073746174652d3135313400"),
            depth_bytes = Hex("0102030405060708090a0b0c0d0e0f10"),
            snapshot_hash = snapshotHash,
        };
    }

    private static SnapshotRequestV4 Request()
    {
        return new SnapshotRequestV4 {
            observation_ref = ObservationRef(),
            execution_id = 44000000000054UL,
            result_payload_hash = Hex(new string('a', 64)),
            command_sequence_hash = Hex(new string('b', 64)),
        };
    }

    private static SnapshotAckV4 Ack(byte[] snapshotHash)
    {
        return new SnapshotAckV4 {
            observation_ref = ObservationRef(),
            snapshot_hash = snapshotHash,
        };
    }

    public static void Main(string[] args)
    {
        string fixtureDir = args[0];
        byte[] snapshotHash = Hex("a91d4155d93c6dbaddec2b7bf617065c92f04507fd60505cae4845db13684e5a");
        EndpointObservationSnapshotV4 snapshot = Snapshot(snapshotHash);
        byte[] refBytes = XMProtocolV4.CanonicalObservationRef(snapshot.observation_ref);
        byte[] snapshotBytes = XMProtocolV4.CanonicalEndpointObservationSnapshot(snapshot);
        byte[] requestBytes = XMProtocolV4.CanonicalSnapshotRequest(Request());
        byte[] ackBytes = XMProtocolV4.CanonicalSnapshotAck(Ack(snapshotHash));

        Equal(refBytes, Hex(File.ReadAllText(Path.Combine(
            fixtureDir, "observation_ref.msgpack.hex"))), "observation ref");
        Equal(snapshotBytes, Hex(File.ReadAllText(Path.Combine(
            fixtureDir, "snapshot.msgpack.hex"))), "snapshot");
        Equal(requestBytes, Hex(File.ReadAllText(Path.Combine(
            fixtureDir, "snapshot_request.msgpack.hex"))), "snapshot request");
        Equal(ackBytes, Hex(File.ReadAllText(Path.Combine(
            fixtureDir, "snapshot_ack.msgpack.hex"))), "snapshot ack");
        if (XMProtocolV4.Sha256Hex(refBytes) !=
            "98a5377a5bafba35c2b4ce54e6b6dd4419fa973bc1abbdc3be212d605dc364dc")
            throw new Exception("observation ref hash mismatch");
        if (XMProtocolV4.Sha256Hex(snapshotBytes) !=
            "a91d4155d93c6dbaddec2b7bf617065c92f04507fd60505cae4845db13684e5a")
            throw new Exception("snapshot hash mismatch");
        if (XMProtocolV4.Sha256Hex(requestBytes) !=
            "f5141c37a5c09d1f10be31898b151d5871ed68dce6b8d64261eece85087e5474")
            throw new Exception("snapshot request hash mismatch");
        if (XMProtocolV4.Sha256Hex(ackBytes) !=
            "a240ca1087508618158106ffdef1bfc0212c0d1a1eec7749d1d6ec422d71d80c")
            throw new Exception("snapshot ack hash mismatch");
        for (int index = 0; index < 100; ++index)
            Equal(XMProtocolV4.CanonicalEndpointObservationSnapshot(snapshot), snapshotBytes,
                "deterministic snapshot");

        var registry = new ObservationSnapshotV4IdentityRegistry();
        if (registry.Observe(snapshot) != ObservationSnapshotV4IdentityOutcome.First)
            throw new Exception("first snapshot identity outcome mismatch");
        if (registry.Observe(snapshot) != ObservationSnapshotV4IdentityOutcome.Duplicate)
            throw new Exception("duplicate snapshot identity outcome mismatch");
        EndpointObservationSnapshotV4 conflicting = Snapshot((byte[])snapshotHash.Clone());
        conflicting.snapshot_hash[0] ^= 0xff;
        try {
            registry.Observe(conflicting);
            throw new Exception("conflicting snapshot was accepted");
        } catch (ArgumentException) {
        }
    }
}
