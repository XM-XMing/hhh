using System;
using XMflight;

public static class EndpointObservationCaptureOwnershipContract
{
    private static int passed;

    private static void Assert(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }

    private static EndpointObservationCaptureIdentity Identity(
        long executionId = 53,
        long stateId = 1516,
        int sourceFrameIndex = 24)
    {
        return new EndpointObservationCaptureIdentity {
            runtime_instance_id = "worker-00-runtime-test",
            execution_id = executionId,
            episode_id = "episode-54",
            reset_id = "reset-54",
            endpoint_state_id = stateId,
            source_frame_index = sourceFrameIndex,
            capture_id = 1516,
        };
    }

    private static void CorrectCallbackIsAcceptedExactlyOnce()
    {
        var owner = new EndpointObservationCaptureOwnership();
        owner.Begin(Identity());
        Assert(owner.TryAccept(Identity()) == EndpointObservationCaptureAcceptOutcome.ACCEPTED,
            "correct callback must be accepted");
        Assert(!owner.HasOwner, "accepted callback must consume ownership");
        Assert(owner.TryAccept(Identity()) == EndpointObservationCaptureAcceptOutcome.STALE,
            "duplicate callback must be stale");
        ++passed;
    }

    private static void OldCallbackAfterCancelIsRejected()
    {
        var owner = new EndpointObservationCaptureOwnership();
        owner.Begin(Identity());
        owner.Invalidate();
        Assert(owner.TryAccept(Identity()) == EndpointObservationCaptureAcceptOutcome.STALE,
            "callback after cancel must be stale");
        ++passed;
    }

    private static void OldCallbackDuringNewExecutionIsRejected()
    {
        var owner = new EndpointObservationCaptureOwnership();
        owner.Begin(Identity(53, 1516));
        owner.Invalidate();
        owner.Begin(Identity(54, 1517));
        Assert(owner.TryAccept(Identity(53, 1516)) ==
            EndpointObservationCaptureAcceptOutcome.REJECTED,
            "old callback must not mutate new execution");
        Assert(owner.TryAccept(Identity(54, 1517)) ==
            EndpointObservationCaptureAcceptOutcome.ACCEPTED,
            "new callback must remain accepted");
        ++passed;
    }

    private static void StateAndFrameIdentityAreValidated()
    {
        var owner = new EndpointObservationCaptureOwnership();
        owner.Begin(Identity());
        Assert(owner.TryAccept(Identity(53, 1515)) ==
            EndpointObservationCaptureAcceptOutcome.REJECTED,
            "wrong state id must be rejected");
        Assert(owner.TryAccept(Identity(53, 1516, 23)) ==
            EndpointObservationCaptureAcceptOutcome.REJECTED,
            "wrong source frame must be rejected");
        Assert(owner.HasOwner, "rejected callbacks must not consume ownership");
        ++passed;
    }

    private static void AllBoundaryIdentityFieldsAreValidated()
    {
        var owner = new EndpointObservationCaptureOwnership();
        owner.Begin(Identity());
        EndpointObservationCaptureIdentity wrongRuntime = Identity();
        wrongRuntime.runtime_instance_id = "worker-01-runtime-test";
        Assert(owner.TryAccept(wrongRuntime) == EndpointObservationCaptureAcceptOutcome.REJECTED,
            "runtime identity must be validated");
        EndpointObservationCaptureIdentity wrongEpisode = Identity();
        wrongEpisode.episode_id = "episode-55";
        Assert(owner.TryAccept(wrongEpisode) == EndpointObservationCaptureAcceptOutcome.REJECTED,
            "episode identity must be validated");
        ++passed;
    }

    private static void ResetCaptureUsesExplicitNonPrimitiveFrameIdentity()
    {
        var owner = new EndpointObservationCaptureOwnership();
        EndpointObservationCaptureIdentity reset = Identity(0, 1516, -1);
        owner.Begin(reset);
        Assert(owner.TryAccept(Identity(0, 1516, -1)) ==
            EndpointObservationCaptureAcceptOutcome.ACCEPTED,
            "reset callback must accept its explicit non-primitive frame identity");
        ++passed;
    }

    public static void Main(string[] args)
    {
        CorrectCallbackIsAcceptedExactlyOnce();
        OldCallbackAfterCancelIsRejected();
        OldCallbackDuringNewExecutionIsRejected();
        StateAndFrameIdentityAreValidated();
        AllBoundaryIdentityFieldsAreValidated();
        ResetCaptureUsesExplicitNonPrimitiveFrameIdentity();
        Console.WriteLine("C# endpoint ownership tests: {0} passed", passed);
    }
}
