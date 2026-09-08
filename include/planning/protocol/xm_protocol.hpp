#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace planning {
namespace xm_protocol {

static constexpr int kSchemaVersion = 3;
static constexpr int kModeVelocity = 0;
static constexpr int kModeTeleport = 1;
static constexpr int kModeStep = 2;
static constexpr int kModeTrajectory = 3;
static constexpr int kModePrimitiveExecution = 4;

static constexpr int kExecutionStatusNone = 0;
static constexpr int kExecutionStatusFrameApplied = 1;
static constexpr int kExecutionStatusComplete = 2;
static constexpr int kExecutionStatusFailed = 3;

static constexpr int kDepthEncoding16UC1 = 1;
static constexpr int kByteOrderLittleEndian = 0;
static constexpr int kFlagCollision = 1 << 0;
static constexpr int kFlagAltitudeViolation = 1 << 1;

static constexpr int kCommandFieldCount = 10;
static constexpr int kDynamicsStateFieldCount = 14;
static constexpr int kDepthFrameFieldCount = 13;
static constexpr int kCameraFieldCount = 8;
static constexpr int kDepthMetaFieldCount = 5;

namespace CommandIndex {
static constexpr int SchemaVersion = 0;
static constexpr int Mode = 1;
static constexpr int Action = 2;
static constexpr int Position = 3;
static constexpr int ClientTimeNs = 4;
static constexpr int CommandId = 5;
static constexpr int ExecutionId = 6;
static constexpr int ExecutionFrameIndex = 7;
static constexpr int ExecutionFrameCount = 8;
static constexpr int ExecutionFrames = 9;
}  // namespace CommandIndex

namespace DynamicsStateIndex {
static constexpr int SchemaVersion = 0;
static constexpr int StateId = 1;
static constexpr int SimTimeNs = 2;
static constexpr int Flags = 3;
static constexpr int MinClearance = 4;
static constexpr int CurrPos = 5;
static constexpr int CurrRot = 6;
static constexpr int CurrVel = 7;
static constexpr int CurrAcc = 8;
static constexpr int FrontClearances = 9;
static constexpr int AppliedExecutionId = 10;
static constexpr int AppliedExecutionFrameIndex = 11;
static constexpr int AppliedCommandId = 12;
static constexpr int ExecutionStatus = 13;
}  // namespace DynamicsStateIndex

namespace DepthFrameIndex {
static constexpr int SchemaVersion = 0;
static constexpr int CaptureId = 1;
static constexpr int SimTimeNs = 2;
static constexpr int Flags = 3;
static constexpr int CapturePos = 4;
static constexpr int CaptureRot = 5;
static constexpr int CaptureVel = 6;
static constexpr int CaptureAcc = 7;
static constexpr int CaptureForward = 8;
static constexpr int Camera = 9;
static constexpr int DepthMeta = 10;
static constexpr int MinClearance = 11;
static constexpr int FrontClearances = 12;
}  // namespace DepthFrameIndex

struct DynamicsState {
  int64_t state_id = 0;
  int64_t sim_time_ns = 0;
  int32_t flags = 0;
  float min_clearance = 0.0f;
  std::array<float, 3> pos{{0.0f, 0.0f, 0.0f}};
  std::array<float, 4> rot{{0.0f, 0.0f, 0.0f, 1.0f}};
  std::array<float, 3> vel{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> acc{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> front_clearances{{0.0f, 0.0f, 0.0f}};
  int64_t applied_execution_id = -1;
  int32_t applied_execution_frame_index = -1;
  int64_t applied_command_id = -1;
  int32_t execution_status = kExecutionStatusNone;
};

struct DepthFrameMeta {
  int64_t capture_id = 0;
  int64_t sim_time_ns = 0;
  int32_t flags = 0;
  std::array<float, 3> pos{{0.0f, 0.0f, 0.0f}};
  std::array<float, 4> rot{{0.0f, 0.0f, 0.0f, 1.0f}};
  std::array<float, 3> vel{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> acc{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> forward{{1.0f, 0.0f, 0.0f}};
  float fx = 0.0f;
  float fy = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;
  int width = 0;
  int height = 0;
  float min_depth_m = 0.0f;
  float max_depth_m = 0.0f;
  int encoding = kDepthEncoding16UC1;
  int row_step = 0;
  int byte_order = kByteOrderLittleEndian;
  float min_clearance = 0.0f;
  std::array<float, 3> front_clearances{{0.0f, 0.0f, 0.0f}};
};

namespace v4 {

static constexpr uint32_t kSchemaVersion = 4;
static constexpr uint32_t kRequestedFrameCount = 25;

struct PrimitiveExecutionFrame {
  uint32_t frame_index = 0;
  int64_t command_id = 0;
  std::vector<float> action;
};

// Reliable v4 command envelope.  The command sequence hash is the identity
// of the physical primitive; the canonical envelope bytes are the immutable
// payload retained by the bridge for retransmission.
struct PrimitiveExecutionCommand {
  uint32_t schema_version = kSchemaVersion;
  std::string message_type = "PrimitiveExecutionCommand";
  std::string runtime_instance_id;
  uint64_t execution_id = 0;
  std::array<uint8_t, 32> command_sequence_hash{{}};
  std::vector<PrimitiveExecutionFrame> frames;
};

struct PrimitiveExecutionCommandReceiptAck {
  uint32_t schema_version = kSchemaVersion;
  std::string message_type = "PrimitiveExecutionCommandReceiptAck";
  std::string runtime_instance_id;
  uint64_t execution_id = 0;
  std::string ack_status;
  std::string reason_code;
  std::array<uint8_t, 32> command_sequence_hash{{}};
};

struct EndpointObservationRef {
  uint32_t schema_version = kSchemaVersion;
  std::string runtime_instance_id;
  std::string episode_id;
  std::string reset_id;
  int64_t state_id = 0;
  std::string depth_id;
  uint64_t sim_time_ns = 0;
};

struct PrimitiveExecutionResult {
  uint32_t schema_version = kSchemaVersion;
  std::string message_type = "PrimitiveExecutionResult";
  std::string runtime_instance_id;
  uint64_t execution_id = 0;
  std::string status;
  uint32_t requested_frame_count = kRequestedFrameCount;
  uint32_t applied_frame_count = 0;
  bool has_first_applied_state_id = false;
  int64_t first_applied_state_id = 0;
  bool has_endpoint_state_id = false;
  int64_t endpoint_state_id = 0;
  int32_t last_applied_frame_index = -1;
  std::string reason_code;
  std::array<uint8_t, 32> command_sequence_hash{{}};
  bool has_endpoint_sim_time_ns = false;
  uint64_t endpoint_sim_time_ns = 0;
  bool has_endpoint_observation_ref = false;
  EndpointObservationRef endpoint_observation_ref;
  uint32_t result_generation = 0;
  // Transport identity. This is deliberately excluded from the canonical
  // payload bytes so the hash does not hash itself.
  std::array<uint8_t, 32> result_payload_hash{{}};
};

struct PrimitiveExecutionResultAck {
  uint32_t schema_version = kSchemaVersion;
  std::string message_type = "PrimitiveExecutionResultAck";
  std::string runtime_instance_id;
  uint64_t execution_id = 0;
  std::string ack_status;
  std::array<uint8_t, 32> result_payload_hash{{}};
  std::array<uint8_t, 32> command_sequence_hash{{}};
};

struct PrimitiveExecutionResultReceiptAck {
  uint32_t schema_version = kSchemaVersion;
  std::string message_type = "PrimitiveExecutionResultReceiptAck";
  std::string runtime_instance_id;
  uint64_t execution_id = 0;
  std::string receipt_status;
  std::array<uint8_t, 32> result_payload_hash{{}};
  std::array<uint8_t, 32> command_sequence_hash{{}};
};

// Exact endpoint identity used by reliable observation snapshot retrieval.
// It has the same seven semantic fields as EndpointObservationRef so a
// COMPLETE result can form an exact SnapshotRequest without context inference.
struct ObservationRef {
  uint32_t schema_version = kSchemaVersion;
  std::string runtime_instance_id;
  std::string episode_id;
  std::string reset_id;
  int64_t state_id = 0;
  std::string depth_id;
  uint64_t sim_time_ns = 0;
};

struct EndpointObservationSnapshot {
  ObservationRef observation_ref;
  std::vector<uint8_t> state_bytes;
  std::vector<uint8_t> depth_bytes;
  // This transport identity is excluded from canonical snapshot bytes.
  std::array<uint8_t, 32> snapshot_hash{{}};
};

struct SnapshotRequest {
  ObservationRef observation_ref;
  uint64_t execution_id = 0;
  std::array<uint8_t, 32> result_payload_hash{{}};
  std::array<uint8_t, 32> command_sequence_hash{{}};
};

struct SnapshotAck {
  ObservationRef observation_ref;
  std::array<uint8_t, 32> snapshot_hash{{}};
};

namespace detail {

inline void append_u8(std::vector<uint8_t>* output, uint8_t value) {
  output->push_back(value);
}

inline void append_u16(std::vector<uint8_t>* output, uint16_t value) {
  output->push_back(static_cast<uint8_t>((value >> 8) & 0xff));
  output->push_back(static_cast<uint8_t>(value & 0xff));
}

inline void append_u32(std::vector<uint8_t>* output, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    output->push_back(static_cast<uint8_t>((value >> shift) & 0xff));
}

inline void append_u64(std::vector<uint8_t>* output, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    output->push_back(static_cast<uint8_t>((value >> shift) & 0xff));
}

inline void append_map_header(std::vector<uint8_t>* output, uint32_t count) {
  if (count <= 15) {
    append_u8(output, static_cast<uint8_t>(0x80 | count));
  } else if (count <= 0xffff) {
    append_u8(output, 0xde);
    append_u16(output, static_cast<uint16_t>(count));
  } else {
    append_u8(output, 0xdf);
    append_u32(output, count);
  }
}

inline void append_array_header(std::vector<uint8_t>* output, uint32_t count) {
  if (count <= 15) {
    append_u8(output, static_cast<uint8_t>(0x90 | count));
  } else if (count <= 0xffff) {
    append_u8(output, 0xdc);
    append_u16(output, static_cast<uint16_t>(count));
  } else {
    append_u8(output, 0xdd);
    append_u32(output, count);
  }
}

inline void append_nil(std::vector<uint8_t>* output) { append_u8(output, 0xc0); }

inline void append_string(std::vector<uint8_t>* output, const std::string& value) {
  const size_t length = value.size();
  if (length <= 0xff) {
    append_u8(output, 0xd9);
    append_u8(output, static_cast<uint8_t>(length));
  } else if (length <= 0xffff) {
    append_u8(output, 0xda);
    append_u16(output, static_cast<uint16_t>(length));
  } else if (length <= 0xffffffffULL) {
    append_u8(output, 0xdb);
    append_u32(output, static_cast<uint32_t>(length));
  } else {
    throw std::invalid_argument("v4 string is too long");
  }
  output->insert(output->end(), value.begin(), value.end());
}

inline void append_binary32(
    std::vector<uint8_t>* output,
    const std::array<uint8_t, 32>& value) {
  append_u8(output, 0xc4);
  append_u8(output, 32);
  output->insert(output->end(), value.begin(), value.end());
}

inline void append_binary(std::vector<uint8_t>* output, const std::vector<uint8_t>& value) {
  const size_t length = value.size();
  if (length <= 0xff) {
    append_u8(output, 0xc4);
    append_u8(output, static_cast<uint8_t>(length));
  } else if (length <= 0xffff) {
    append_u8(output, 0xc5);
    append_u16(output, static_cast<uint16_t>(length));
  } else if (length <= 0xffffffffULL) {
    append_u8(output, 0xc6);
    append_u32(output, static_cast<uint32_t>(length));
  } else {
    throw std::invalid_argument("v4 binary field is too long");
  }
  output->insert(output->end(), value.begin(), value.end());
}

inline void append_uint32(std::vector<uint8_t>* output, uint32_t value) {
  append_u8(output, 0xce);
  append_u32(output, value);
}

inline void append_uint64(std::vector<uint8_t>* output, uint64_t value) {
  append_u8(output, 0xcf);
  append_u64(output, value);
}

inline void append_int32(std::vector<uint8_t>* output, int32_t value) {
  append_u8(output, 0xd2);
  append_u32(output, static_cast<uint32_t>(value));
}

inline void append_int64(std::vector<uint8_t>* output, int64_t value) {
  append_u8(output, 0xd3);
  append_u64(output, static_cast<uint64_t>(value));
}

inline void append_float32(std::vector<uint8_t>* output, float value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("v4 action values reject NaN and infinity");
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float32 must be four bytes");
  std::memcpy(&bits, &value, sizeof(bits));
  append_u8(output, 0xca);
  append_u32(output, bits);
}

inline void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

inline void append_result_optional_int64(
    std::vector<uint8_t>* output,
    bool present,
    int64_t value) {
  if (present) append_int64(output, value);
  else append_nil(output);
}

inline void append_result_optional_uint64(
    std::vector<uint8_t>* output,
    bool present,
    uint64_t value) {
  if (present) append_uint64(output, value);
  else append_nil(output);
}

inline void validate_result(const PrimitiveExecutionResult& result) {
  require(result.schema_version == kSchemaVersion, "schema_version must be 4");
  require(result.message_type == "PrimitiveExecutionResult", "message_type mismatch");
  require(!result.runtime_instance_id.empty(), "runtime_instance_id is required");
  require(result.requested_frame_count == kRequestedFrameCount, "requested_frame_count mismatch");
  require(result.result_generation == 0, "result_generation must remain zero");
  if (result.status == "COMPLETE") {
    require(result.applied_frame_count == kRequestedFrameCount, "COMPLETE applied count mismatch");
    require(result.last_applied_frame_index == 24, "COMPLETE last frame mismatch");
    require(result.has_first_applied_state_id && result.has_endpoint_state_id,
            "COMPLETE state ids are required");
    require(result.endpoint_state_id == result.first_applied_state_id + 24,
            "COMPLETE endpoint state mismatch");
    require(result.reason_code == "NONE", "COMPLETE reason must be NONE");
    require(result.has_endpoint_sim_time_ns && result.has_endpoint_observation_ref,
            "COMPLETE endpoint observation is required");
    require(result.endpoint_observation_ref.schema_version == kSchemaVersion,
            "COMPLETE observation schema_version mismatch");
    require(result.endpoint_observation_ref.runtime_instance_id == result.runtime_instance_id,
            "COMPLETE observation runtime identity mismatch");
    require(!result.endpoint_observation_ref.episode_id.empty(),
            "COMPLETE episode id is required");
    require(!result.endpoint_observation_ref.reset_id.empty(),
            "COMPLETE reset id is required");
    require(!result.endpoint_observation_ref.depth_id.empty(),
            "COMPLETE depth id is required");
    require(result.endpoint_observation_ref.state_id == result.endpoint_state_id,
            "COMPLETE observation state mismatch");
    require(result.endpoint_observation_ref.sim_time_ns == result.endpoint_sim_time_ns,
            "COMPLETE observation time mismatch");
  } else if (result.status == "REJECTED") {
    require(result.applied_frame_count == 0, "REJECTED applied count must be zero");
    require(result.last_applied_frame_index == -1, "REJECTED last frame must be -1");
    require(!result.has_first_applied_state_id && !result.has_endpoint_state_id,
            "REJECTED must not claim state ids");
    require(!result.has_endpoint_sim_time_ns && !result.has_endpoint_observation_ref,
            "REJECTED must not claim endpoint observation");
    require(result.reason_code != "" && result.reason_code != "NONE",
            "REJECTED reason is required");
  } else if (result.status == "CANCELLED" || result.status == "FAILED") {
    const bool physics_complete_observation_failed =
        result.status == "FAILED" &&
        result.reason_code == "PHYSICS_COMPLETE_OBSERVATION_FAILED";
    const bool environment_terminal_with_observation =
        result.status == "FAILED" && result.reason_code == "COLLISION";
    if (physics_complete_observation_failed) {
      require(result.applied_frame_count == kRequestedFrameCount,
              "observation failure must preserve all applied frames");
      require(result.last_applied_frame_index == 24,
              "observation failure last frame mismatch");
    } else {
      require(result.applied_frame_count < kRequestedFrameCount,
              "interrupted result cannot claim complete prefix");
    }
    require(result.last_applied_frame_index ==
                static_cast<int32_t>(result.applied_frame_count) - 1,
            "interrupted last frame mismatch");
    require(result.reason_code != "" && result.reason_code != "NONE",
            "interrupted reason is required");
    if (environment_terminal_with_observation) {
      require(result.has_endpoint_state_id && result.has_endpoint_sim_time_ns &&
                  result.has_endpoint_observation_ref,
              "environment terminal observation is required");
      require(result.endpoint_observation_ref.schema_version == kSchemaVersion,
              "terminal observation schema_version mismatch");
      require(result.endpoint_observation_ref.runtime_instance_id ==
                  result.runtime_instance_id,
              "terminal observation runtime identity mismatch");
      require(!result.endpoint_observation_ref.episode_id.empty(),
              "terminal observation episode id is required");
      require(!result.endpoint_observation_ref.reset_id.empty(),
              "terminal observation reset id is required");
      require(!result.endpoint_observation_ref.depth_id.empty(),
              "terminal observation depth id is required");
      require(result.endpoint_observation_ref.state_id == result.endpoint_state_id,
              "terminal observation state mismatch");
      require(result.endpoint_observation_ref.sim_time_ns == result.endpoint_sim_time_ns,
              "terminal observation time mismatch");
    } else {
      require(!result.has_endpoint_state_id && !result.has_endpoint_sim_time_ns &&
                  !result.has_endpoint_observation_ref,
              "interrupted result must not claim endpoint observation");
    }
    require(result.applied_frame_count == 0 || result.has_first_applied_state_id,
            "applied prefix requires first state id");
  } else {
    throw std::invalid_argument("unknown v4 result status");
  }
}

inline void validate_observation_ref(const ObservationRef& observation_ref) {
  require(observation_ref.schema_version == kSchemaVersion, "observation schema_version must be 4");
  require(!observation_ref.runtime_instance_id.empty(), "observation runtime_instance_id is required");
  require(!observation_ref.episode_id.empty(), "observation episode_id is required");
  require(!observation_ref.reset_id.empty(), "observation reset_id is required");
  require(observation_ref.state_id >= 0, "observation state_id must be non-negative");
  require(!observation_ref.depth_id.empty(), "observation depth_id is required");
}

}  // namespace detail

inline void append_observation_ref(
    std::vector<uint8_t>* output,
    const ObservationRef& observation_ref) {
  detail::validate_observation_ref(observation_ref);
  detail::append_map_header(output, 7);
  detail::append_string(output, "depth_id");
  detail::append_string(output, observation_ref.depth_id);
  detail::append_string(output, "episode_id");
  detail::append_string(output, observation_ref.episode_id);
  detail::append_string(output, "reset_id");
  detail::append_string(output, observation_ref.reset_id);
  detail::append_string(output, "runtime_instance_id");
  detail::append_string(output, observation_ref.runtime_instance_id);
  detail::append_string(output, "schema_version");
  detail::append_uint32(output, observation_ref.schema_version);
  detail::append_string(output, "sim_time_ns");
  detail::append_uint64(output, observation_ref.sim_time_ns);
  detail::append_string(output, "state_id");
  detail::append_int64(output, observation_ref.state_id);
}

inline std::vector<uint8_t> canonical_observation_ref_bytes(
    const ObservationRef& observation_ref) {
  std::vector<uint8_t> output;
  append_observation_ref(&output, observation_ref);
  return output;
}

inline std::vector<uint8_t> canonical_endpoint_observation_snapshot_bytes(
    const EndpointObservationSnapshot& snapshot) {
  detail::validate_observation_ref(snapshot.observation_ref);
  std::vector<uint8_t> output;
  detail::append_map_header(&output, 3);
  detail::append_string(&output, "depth_bytes");
  detail::append_binary(&output, snapshot.depth_bytes);
  detail::append_string(&output, "observation_ref");
  append_observation_ref(&output, snapshot.observation_ref);
  detail::append_string(&output, "state_bytes");
  detail::append_binary(&output, snapshot.state_bytes);
  return output;
}

inline std::vector<uint8_t> canonical_snapshot_request_bytes(
    const SnapshotRequest& request) {
  detail::validate_observation_ref(request.observation_ref);
  std::vector<uint8_t> output;
  detail::append_map_header(&output, 4);
  detail::append_string(&output, "command_sequence_hash");
  detail::append_binary32(&output, request.command_sequence_hash);
  detail::append_string(&output, "execution_id");
  detail::append_uint64(&output, request.execution_id);
  detail::append_string(&output, "observation_ref");
  append_observation_ref(&output, request.observation_ref);
  detail::append_string(&output, "result_payload_hash");
  detail::append_binary32(&output, request.result_payload_hash);
  return output;
}

inline std::vector<uint8_t> canonical_snapshot_ack_bytes(const SnapshotAck& ack) {
  detail::validate_observation_ref(ack.observation_ref);
  std::vector<uint8_t> output;
  detail::append_map_header(&output, 2);
  detail::append_string(&output, "observation_ref");
  append_observation_ref(&output, ack.observation_ref);
  detail::append_string(&output, "snapshot_hash");
  detail::append_binary32(&output, ack.snapshot_hash);
  return output;
}

enum class ObservationSnapshotIdentityOutcome { kFirst, kDuplicate };

class ObservationSnapshotIdentityRegistry {
 public:
  ObservationSnapshotIdentityOutcome Observe(const EndpointObservationSnapshot& snapshot) {
    const std::vector<uint8_t> key = canonical_observation_ref_bytes(snapshot.observation_ref);
    const auto existing = snapshot_hashes_.find(key);
    if (existing == snapshot_hashes_.end()) {
      snapshot_hashes_.emplace(key, snapshot.snapshot_hash);
      return ObservationSnapshotIdentityOutcome::kFirst;
    }
    if (existing->second != snapshot.snapshot_hash)
      throw std::invalid_argument("snapshot_hash conflict for observation_ref");
    return ObservationSnapshotIdentityOutcome::kDuplicate;
  }

 private:
  std::map<std::vector<uint8_t>, std::array<uint8_t, 32>> snapshot_hashes_;
};

inline std::vector<uint8_t> canonical_command_sequence_bytes(
    const std::vector<PrimitiveExecutionFrame>& frames) {
  detail::require(!frames.empty(), "command sequence must not be empty");
  std::vector<uint8_t> output;
  detail::append_array_header(&output, static_cast<uint32_t>(frames.size()));
  for (const PrimitiveExecutionFrame& frame : frames) {
    detail::append_map_header(&output, 3);
    detail::append_string(&output, "action");
    detail::append_array_header(&output, static_cast<uint32_t>(frame.action.size()));
    for (float value : frame.action) detail::append_float32(&output, value);
    detail::append_string(&output, "command_id");
    detail::append_int64(&output, frame.command_id);
    detail::append_string(&output, "frame_index");
    detail::append_uint32(&output, frame.frame_index);
  }
  return output;
}

inline void validate_command(const PrimitiveExecutionCommand& command) {
  detail::require(command.schema_version == kSchemaVersion,
                  "command schema_version must be 4");
  detail::require(command.message_type == "PrimitiveExecutionCommand",
                  "command message_type mismatch");
  detail::require(!command.runtime_instance_id.empty(),
                  "command runtime_instance_id is required");
  detail::require(command.frames.size() == kRequestedFrameCount,
                  "command frame count mismatch");
  for (size_t index = 0; index < command.frames.size(); ++index) {
    detail::require(command.frames[index].frame_index == index,
                    "command frame indices must be contiguous");
  }
}

inline std::vector<uint8_t> canonical_command_payload_bytes(
    const PrimitiveExecutionCommand& command) {
  validate_command(command);
  const std::vector<uint8_t> sequence =
      canonical_command_sequence_bytes(command.frames);
  std::vector<uint8_t> output;
  detail::append_map_header(&output, 6);
  detail::append_string(&output, "command_sequence_hash");
  detail::append_binary32(&output, command.command_sequence_hash);
  detail::append_string(&output, "execution_id");
  detail::append_uint64(&output, command.execution_id);
  detail::append_string(&output, "frames");
  output.insert(output.end(), sequence.begin(), sequence.end());
  detail::append_string(&output, "message_type");
  detail::append_string(&output, command.message_type);
  detail::append_string(&output, "runtime_instance_id");
  detail::append_string(&output, command.runtime_instance_id);
  detail::append_string(&output, "schema_version");
  detail::append_uint32(&output, command.schema_version);
  return output;
}

inline std::vector<uint8_t> canonical_command_receipt_ack_bytes(
    const PrimitiveExecutionCommandReceiptAck& ack) {
  detail::require(ack.schema_version == kSchemaVersion,
                  "command receipt schema_version must be 4");
  detail::require(
      ack.message_type == "PrimitiveExecutionCommandReceiptAck",
      "command receipt message_type mismatch");
  detail::require(!ack.runtime_instance_id.empty(),
                  "command receipt runtime_instance_id is required");
  detail::require(!ack.ack_status.empty(),
                  "command receipt ack_status is required");
  std::vector<uint8_t> output;
  detail::append_map_header(&output, 7);
  detail::append_string(&output, "ack_status");
  detail::append_string(&output, ack.ack_status);
  detail::append_string(&output, "command_sequence_hash");
  detail::append_binary32(&output, ack.command_sequence_hash);
  detail::append_string(&output, "execution_id");
  detail::append_uint64(&output, ack.execution_id);
  detail::append_string(&output, "message_type");
  detail::append_string(&output, ack.message_type);
  detail::append_string(&output, "reason_code");
  detail::append_string(&output, ack.reason_code);
  detail::append_string(&output, "runtime_instance_id");
  detail::append_string(&output, ack.runtime_instance_id);
  detail::append_string(&output, "schema_version");
  detail::append_uint32(&output, ack.schema_version);
  return output;
}

inline std::vector<uint8_t> canonical_result_payload_bytes(
    const PrimitiveExecutionResult& result) {
  detail::validate_result(result);
  std::vector<uint8_t> output;
  detail::append_map_header(&output, 15);
  detail::append_string(&output, "applied_frame_count");
  detail::append_uint32(&output, result.applied_frame_count);
  detail::append_string(&output, "command_sequence_hash");
  detail::append_binary32(&output, result.command_sequence_hash);
  detail::append_string(&output, "endpoint_observation_ref");
  if (!result.has_endpoint_observation_ref) {
    detail::append_nil(&output);
  } else {
    detail::append_map_header(&output, 7);
    detail::append_string(&output, "depth_id");
    detail::append_string(&output, result.endpoint_observation_ref.depth_id);
    detail::append_string(&output, "episode_id");
    detail::append_string(&output, result.endpoint_observation_ref.episode_id);
    detail::append_string(&output, "reset_id");
    detail::append_string(&output, result.endpoint_observation_ref.reset_id);
    detail::append_string(&output, "runtime_instance_id");
    detail::append_string(&output, result.endpoint_observation_ref.runtime_instance_id);
    detail::append_string(&output, "schema_version");
    detail::append_uint32(&output, result.endpoint_observation_ref.schema_version);
    detail::append_string(&output, "sim_time_ns");
    detail::append_uint64(&output, result.endpoint_observation_ref.sim_time_ns);
    detail::append_string(&output, "state_id");
    detail::append_int64(&output, result.endpoint_observation_ref.state_id);
  }
  detail::append_string(&output, "endpoint_sim_time_ns");
  detail::append_result_optional_uint64(
      &output, result.has_endpoint_sim_time_ns, result.endpoint_sim_time_ns);
  detail::append_string(&output, "endpoint_state_id");
  detail::append_result_optional_int64(
      &output, result.has_endpoint_state_id, result.endpoint_state_id);
  detail::append_string(&output, "execution_id");
  detail::append_uint64(&output, result.execution_id);
  detail::append_string(&output, "first_applied_state_id");
  detail::append_result_optional_int64(
      &output, result.has_first_applied_state_id, result.first_applied_state_id);
  detail::append_string(&output, "last_applied_frame_index");
  detail::append_int32(&output, result.last_applied_frame_index);
  detail::append_string(&output, "message_type");
  detail::append_string(&output, result.message_type);
  detail::append_string(&output, "reason_code");
  detail::append_string(&output, result.reason_code);
  detail::append_string(&output, "requested_frame_count");
  detail::append_uint32(&output, result.requested_frame_count);
  detail::append_string(&output, "result_generation");
  detail::append_uint32(&output, result.result_generation);
  detail::append_string(&output, "runtime_instance_id");
  detail::append_string(&output, result.runtime_instance_id);
  detail::append_string(&output, "schema_version");
  detail::append_uint32(&output, result.schema_version);
  detail::append_string(&output, "status");
  detail::append_string(&output, result.status);
  return output;
}

}  // namespace v4

}  // namespace xm_protocol
}  // namespace planning
