using System;
using XMflight;

public static class PrimitiveExecutionCommandAdmissionContract
{
    private static int passed;

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }

    private static PrimitiveExecutionV4Command Command(string runtime = "worker-00")
    {
        var command = new PrimitiveExecutionV4Command {
            runtime_instance_id = runtime,
            execution_id = 44000000000001UL,
        };
        for (uint index = 0; index < 25; ++index)
            command.frames.Add(new PrimitiveExecutionV4Frame {
                frame_index = index,
                command_id = 1000L + (long)index,
                action = new[] { 0.1f, 0.2f, 0.3f, 0.4f },
            });
        command.command_sequence_hash = XMProtocolV4.Sha256Bytes(
            XMProtocolV4.CanonicalCommandSequence(command.frames));
        return command;
    }

    private static void FirstCommandIsAcceptedOnce()
    {
        var admission = new PrimitiveExecutionCommandAdmission("worker-00");
        var result = admission.Register(Command());
        Require(result.Outcome == PrimitiveExecutionCommandAdmissionOutcome.ACCEPTED,
            "first command outcome");
        Require(admission.PhysicalExecutionCount == 1,
            "first physical execution count");
        ++passed;
    }

    private static void DuplicateCommandDoesNotExecuteAgain()
    {
        var admission = new PrimitiveExecutionCommandAdmission("worker-00");
        var command = Command();
        var first = admission.Register(command);
        var duplicate = admission.Register(command);
        Require(first.Outcome == PrimitiveExecutionCommandAdmissionOutcome.ACCEPTED,
            "duplicate first outcome");
        Require(duplicate.Outcome == PrimitiveExecutionCommandAdmissionOutcome.DUPLICATE,
            "duplicate outcome");
        Require(admission.PhysicalExecutionCount == 1,
            "duplicate physical execution count");
        Require(duplicate.Receipt.ack_status == "DUPLICATE", "duplicate receipt");
        ++passed;
    }

    private static void ConflictingCommandFailsClosed()
    {
        var admission = new PrimitiveExecutionCommandAdmission("worker-00");
        var first = Command();
        var conflict = Command();
        conflict.frames[10].action[0] += 1.0f;
        conflict.command_sequence_hash = XMProtocolV4.Sha256Bytes(
            XMProtocolV4.CanonicalCommandSequence(conflict.frames));
        admission.Register(first);
        var result = admission.Register(conflict);
        Require(result.Outcome == PrimitiveExecutionCommandAdmissionOutcome.PROTOCOL_ERROR,
            "conflict outcome");
        Require(admission.PhysicalExecutionCount == 1,
            "conflict physical execution count");
        ++passed;
    }

    private static void WrongRuntimeIsRejected()
    {
        var admission = new PrimitiveExecutionCommandAdmission("worker-00");
        var result = admission.Register(Command("worker-01"));
        Require(result.Outcome == PrimitiveExecutionCommandAdmissionOutcome.REJECTED,
            "wrong runtime outcome");
        Require(admission.PhysicalExecutionCount == 0,
            "wrong runtime physical execution count");
        ++passed;
    }

    private static void OneHundredDuplicatesRemainOneExecution()
    {
        var admission = new PrimitiveExecutionCommandAdmission("worker-00");
        var command = Command();
        admission.Register(command);
        for (int index = 0; index < 100; ++index)
            Require(admission.Register(command).Outcome ==
                PrimitiveExecutionCommandAdmissionOutcome.DUPLICATE,
                "repeated duplicate outcome");
        Require(admission.PhysicalExecutionCount == 1,
            "repeated duplicate physical execution count");
        ++passed;
    }

    public static void Main()
    {
        FirstCommandIsAcceptedOnce();
        DuplicateCommandDoesNotExecuteAgain();
        ConflictingCommandFailsClosed();
        WrongRuntimeIsRejected();
        OneHundredDuplicatesRemainOneExecution();
        Console.WriteLine("C# command admission tests: {0} passed", passed);
    }
}
