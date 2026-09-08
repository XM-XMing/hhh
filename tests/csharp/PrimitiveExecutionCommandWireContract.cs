using System;
using System.IO;
using XMflight;

public static class PrimitiveExecutionCommandWireContract
{
    private static void Require(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }

    public static void Main(string[] args)
    {
        Require(args != null && args.Length == 1, "wire fixture path is required");
        PrimitiveExecutionV4Command command =
            PrimitiveExecutionCommandWireCodec.DeserializeCommand(
                File.ReadAllBytes(args[0]));

        Require(command.schema_version == 4, "schema version");
        Require(command.message_type == "PrimitiveExecutionCommand", "message type");
        Require(command.runtime_instance_id == "worker-00", "runtime identity");
        Require(command.execution_id == 44000000000001UL, "execution identity");
        Require(command.frames.Count == 25, "frame count");
        Require(command.frames[0].frame_index == 0, "first frame index");
        Require(command.frames[24].frame_index == 24, "last frame index");
        Require(command.frames[10].action.Length == 4, "action width");
        Console.WriteLine("C# command wire tests: 1 passed");
    }
}
