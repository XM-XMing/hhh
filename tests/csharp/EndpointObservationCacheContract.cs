using System;
using XMflight;

public static class EndpointObservationCacheContract
{
    private static readonly byte[] StateBytes = {
        0x00, 0xff, 0x10, 0x73, 0x74, 0x61, 0x74, 0x65,
    };
    private static readonly byte[] DepthBytes = {
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    };

    private static void Assert(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }

    private static bool EqualBytes(byte[] left, byte[] right)
    {
        if (left == null || right == null || left.Length != right.Length) return false;
        for (int index = 0; index < left.Length; ++index)
            if (left[index] != right[index]) return false;
        return true;
    }

    private static ObservationRefV4 Ref()
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

    private static EndpointObservationCacheStoreResult Store(
        EndpointObservationCache cache, byte[] stateBytes, byte[] depthBytes)
    {
        return cache.StoreTerminalFrame24(
            "worker-00-runtime-test", "episode-54", "reset-54", 1514,
            "depth-1514", 559999987UL, stateBytes, depthBytes);
    }

    private static void SnapshotCreationAndExactIdentityBinding()
    {
        var cache = new EndpointObservationCache();
        byte[] stateInput = (byte[])StateBytes.Clone();
        byte[] depthInput = (byte[])DepthBytes.Clone();
        EndpointObservationCacheStoreResult stored = Store(cache, stateInput, depthInput);
        EndpointObservationSnapshotV4 snapshot = stored.snapshot;
        stateInput[0] ^= 0xff;
        depthInput[0] ^= 0xff;

        Assert(stored.outcome == EndpointObservationCacheStoreOutcome.Stored,
            "terminal snapshot was not stored");
        Assert(snapshot.observation_ref.runtime_instance_id == "worker-00-runtime-test",
            "runtime identity mismatch");
        Assert(snapshot.observation_ref.episode_id == "episode-54", "episode identity mismatch");
        Assert(snapshot.observation_ref.reset_id == "reset-54", "reset identity mismatch");
        Assert(snapshot.observation_ref.state_id == 1514, "state identity mismatch");
        Assert(snapshot.observation_ref.depth_id == "depth-1514", "depth identity mismatch");
        Assert(snapshot.observation_ref.sim_time_ns == 559999987UL, "time identity mismatch");
        Assert(EqualBytes(snapshot.snapshot_hash, XMProtocolV4.SnapshotHash(snapshot)),
            "snapshot hash does not bind immutable bytes");
        EndpointObservationSnapshotV4 retained;
        Assert(cache.TryGet(Ref(), out retained), "stored snapshot cannot be retrieved");
        Assert(EqualBytes(retained.state_bytes, StateBytes), "cache retained caller state buffer");
        Assert(EqualBytes(retained.depth_bytes, DepthBytes), "cache retained caller depth buffer");
        Assert(cache.Count == 1, "cache count mismatch after store");
    }

    private static void MissingStateBytesAreRejected()
    {
        var cache = new EndpointObservationCache();
        bool rejected = false;
        try {
            cache.StoreTerminalFrame24(
                "worker-00-runtime-test", "episode-54", "reset-54", 1514,
                "depth-1514", 559999987UL, new byte[0], DepthBytes);
        }
        catch (ArgumentException) {
            rejected = true;
        }
        Assert(rejected, "missing state bytes must not create a snapshot");
        Assert(cache.Count == 0, "missing state bytes changed cache");
    }

    private static void DeterministicHashAndDuplicateConflict()
    {
        var cache = new EndpointObservationCache();
        EndpointObservationCacheStoreResult first = Store(cache, StateBytes, DepthBytes);
        for (int index = 0; index < 100; ++index) {
            EndpointObservationCacheStoreResult duplicate = Store(cache, StateBytes, DepthBytes);
            Assert(duplicate.outcome == EndpointObservationCacheStoreOutcome.Duplicate,
                "same terminal snapshot was not duplicate");
            Assert(EqualBytes(first.snapshot.snapshot_hash, duplicate.snapshot.snapshot_hash),
                "same terminal snapshot changed hash");
        }
        try {
            Store(cache, StateBytes, new byte[] { 9, 8, 7 });
            throw new Exception("conflicting snapshot overwrite was accepted");
        } catch (EndpointObservationCacheProtocolException) {
        }
        Assert(cache.Count == 1, "conflict changed immutable cache count");
    }

    private static void ExactLookupAndMissingSnapshot()
    {
        var cache = new EndpointObservationCache();
        EndpointObservationCacheStoreResult stored = Store(cache, StateBytes, DepthBytes);
        EndpointObservationSnapshotV4 fetched;
        Assert(cache.TryGet(Ref(), out fetched), "exact snapshot lookup failed");
        fetched.state_bytes[0] ^= 0xff;
        EndpointObservationSnapshotV4 fetchedAgain;
        Assert(cache.TryGet(Ref(), out fetchedAgain), "second exact lookup failed");
        Assert(EqualBytes(fetchedAgain.state_bytes, StateBytes), "cache leaked mutable state bytes");

        ObservationRefV4 missing = Ref();
        missing.depth_id = "depth-1515";
        Assert(!cache.TryGet(missing, out fetched), "wrong ref used a fallback snapshot");
        Assert(EqualBytes(stored.snapshot.snapshot_hash, fetchedAgain.snapshot_hash),
            "exact snapshot identity changed");
    }

    private static void RetentionLifecycle()
    {
        var cache = new EndpointObservationCache();
        EndpointObservationCacheStoreResult stored = Store(cache, StateBytes, DepthBytes);
        byte[] wrongHash = (byte[])stored.snapshot.snapshot_hash.Clone();
        wrongHash[0] ^= 0xff;
        try {
            cache.Release(Ref(), wrongHash);
            throw new Exception("wrong release hash removed snapshot");
        } catch (EndpointObservationCacheProtocolException) {
        }
        Assert(cache.Count == 1, "wrong release hash changed cache");
        Assert(cache.Release(Ref(), stored.snapshot.snapshot_hash), "exact release failed");
        EndpointObservationSnapshotV4 missing;
        Assert(!cache.TryGet(Ref(), out missing), "released snapshot remained retrievable");
        Assert(cache.Count == 0, "cache retention leak after exact release");
    }

    public static void Main(string[] args)
    {
        SnapshotCreationAndExactIdentityBinding();
        MissingStateBytesAreRejected();
        DeterministicHashAndDuplicateConflict();
        ExactLookupAndMissingSnapshot();
        RetentionLifecycle();
    }
}
