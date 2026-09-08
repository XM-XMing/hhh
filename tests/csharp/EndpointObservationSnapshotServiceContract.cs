using System;
using XMflight;

public static class EndpointObservationSnapshotServiceContract
{
    private const string RuntimeId = "worker-00-runtime-test";

    private sealed class Sink : IEndpointObservationSnapshotSink
    {
        public int SnapshotCount;
        public int MissingCount;
        public SnapshotRequestV4 LastRequest;
        public EndpointObservationSnapshotV4 LastSnapshot;

        public void SendSnapshot(SnapshotRequestV4 request, EndpointObservationSnapshotV4 snapshot)
        {
            SnapshotCount++;
            LastRequest = request;
            LastSnapshot = snapshot;
        }

        public void SendMissing(SnapshotRequestV4 request) { MissingCount++; LastRequest = request; }
    }

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

    private static ObservationRefV4 Ref(string depthId = "depth-4024")
    {
        return new ObservationRefV4 {
            schema_version = 4,
            runtime_instance_id = RuntimeId,
            episode_id = "episode-1",
            reset_id = "reset-1",
            state_id = 4024,
            depth_id = depthId,
            sim_time_ns = 5000000000UL,
        };
    }

    private static SnapshotRequestV4 Request(ObservationRefV4 reference = null)
    {
        return new SnapshotRequestV4 {
            observation_ref = reference ?? Ref(),
            execution_id = 0x0102030405060708UL,
            result_payload_hash = Hash(0x11),
            command_sequence_hash = Hash(0x22),
        };
    }

    private static EndpointObservationSnapshotV4 Store(EndpointObservationCache cache)
    {
        return cache.StoreTerminalFrame24(
            RuntimeId, "episode-1", "reset-1", 4024L, "depth-4024",
            5000000000UL, new byte[] { 1, 2 }, new byte[] { 3, 4 }).snapshot;
    }

    private static void ExactRequestReturnsOnlyExactSnapshot()
    {
        EndpointObservationCache cache = new EndpointObservationCache();
        EndpointObservationSnapshotV4 stored = Store(cache);
        Sink sink = new Sink();
        EndpointObservationSnapshotService service =
            new EndpointObservationSnapshotService(cache, sink, RuntimeId);
        SnapshotRequestV4 request = Request();
        service.HandleRequest(request);
        Require(sink.SnapshotCount == 1 && sink.MissingCount == 0, "exact request must return one snapshot");
        Require(sink.LastSnapshot.snapshot_hash[0] == stored.snapshot_hash[0], "snapshot hash changed");
        Require(sink.LastRequest.execution_id == request.execution_id, "execution identity changed");
    }

    private static void MissingAndWrongRefDoNotFallback()
    {
        EndpointObservationCache cache = new EndpointObservationCache();
        Store(cache);
        Sink sink = new Sink();
        EndpointObservationSnapshotService service =
            new EndpointObservationSnapshotService(cache, sink, RuntimeId);
        service.HandleRequest(Request(Ref("depth-missing")));
        Require(sink.SnapshotCount == 0 && sink.MissingCount == 1, "missing ref must not fallback");
    }

    private static void DuplicateRequestReturnsImmutableDuplicate()
    {
        EndpointObservationCache cache = new EndpointObservationCache();
        Store(cache);
        Sink sink = new Sink();
        EndpointObservationSnapshotService service =
            new EndpointObservationSnapshotService(cache, sink, RuntimeId);
        SnapshotRequestV4 request = Request();
        service.HandleRequest(request);
        byte[] first = (byte[])sink.LastSnapshot.state_bytes.Clone();
        service.HandleRequest(request);
        Require(sink.SnapshotCount == 2, "duplicate request must be served idempotently");
        Require(first[0] == sink.LastSnapshot.state_bytes[0], "duplicate overwrote snapshot bytes");
    }

    private static void ExactAckReleasesAndWrongAckCannotRelease()
    {
        EndpointObservationCache cache = new EndpointObservationCache();
        EndpointObservationSnapshotV4 stored = Store(cache);
        Sink sink = new Sink();
        EndpointObservationSnapshotService service =
            new EndpointObservationSnapshotService(cache, sink, RuntimeId);
        bool rejected = false;
        byte[] wrong = (byte[])stored.snapshot_hash.Clone();
        wrong[0] ^= 1;
        try {
            service.HandleAck(new SnapshotAckV4 { observation_ref = Ref(), snapshot_hash = wrong });
        } catch (EndpointObservationCacheProtocolException) { rejected = true; }
        Require(rejected && cache.Count == 1, "wrong ACK must retain snapshot");
        service.HandleAck(new SnapshotAckV4 { observation_ref = Ref(), snapshot_hash = stored.snapshot_hash });
        Require(cache.Count == 0, "exact ACK must release snapshot");
    }

    public static int Main()
    {
        ExactRequestReturnsOnlyExactSnapshot();
        MissingAndWrongRefDoNotFallback();
        DuplicateRequestReturnsImmutableDuplicate();
        ExactAckReleasesAndWrongAckCannotRelease();
        Console.WriteLine("C# endpoint snapshot service tests: 4 passed");
        return 0;
    }
}
