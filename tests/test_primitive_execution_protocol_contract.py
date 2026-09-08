"""RED contract tests for physics-clock-bound primitive execution."""

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_command_wire_protocol_carries_primitive_execution_identity_and_frame_index():
    protocol = (ROOT / "include/planning/protocol/xm_protocol.hpp").read_text(encoding="utf-8")

    required_command_fields = {
        "ExecutionId",
        "ExecutionFrameIndex",
        "ExecutionFrameCount",
    }
    missing = sorted(
        field
        for field in required_command_fields
        if re.search(r"static constexpr int\s+{}\s*=".format(field), protocol) is None
    )

    assert missing == [], (
        "primitive command wire schema cannot identify an execution or bind a command "
        "frame within it; missing fields: {}".format(missing)
    )


@pytest.mark.unit
def test_deterministic_execution_wire_contract_has_a_new_schema_version():
    protocol = (ROOT / "include/planning/protocol/xm_protocol.hpp").read_text(encoding="utf-8")
    match = re.search(r"kSchemaVersion\s*=\s*(\d+)", protocol)

    assert match is not None
    assert int(match.group(1)) >= 3, (
        "adding execution/frame identity changes the wire layout and must not be "
        "silently accepted as schema v2"
    )


@pytest.mark.unit
def test_state_protocol_acknowledges_the_command_frame_applied_by_fixed_update():
    protocol = (ROOT / "include/planning/protocol/xm_protocol.hpp").read_text(encoding="utf-8")
    state_message = (ROOT / "msg/XMState.msg").read_text(encoding="utf-8")

    required_wire_fields = {
        "AppliedExecutionId",
        "AppliedExecutionFrameIndex",
        "AppliedCommandId",
    }
    missing_wire = sorted(
        field
        for field in required_wire_fields
        if re.search(r"static constexpr int\s+{}\s*=".format(field), protocol) is None
    )
    required_ros_fields = {
        "int64 applied_execution_id",
        "int32 applied_execution_frame_index",
        "int64 applied_command_id",
    }
    missing_ros = sorted(field for field in required_ros_fields if field not in state_message)

    assert missing_wire == [] and missing_ros == [], (
        "post-integration state cannot acknowledge which primitive command frame was "
        "actually applied; missing wire={} ROS={}".format(missing_wire, missing_ros)
    )
