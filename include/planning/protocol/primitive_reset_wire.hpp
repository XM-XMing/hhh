#pragma once

#include <array>
#include <cstdint>
#include <msgpack.hpp>
#include <string>
#include <vector>

#include "planning/protocol/msgpack_value.hpp"
#include "planning/protocol/primitive_execution_result_wire.hpp"

namespace planning {
namespace primitive_reset_wire {

namespace xp = planning::xm_protocol;
namespace result_wire = planning::primitive_execution_result_wire;

static constexpr uint32_t kSchemaVersion = 4;
static constexpr const char* kRequestMessageType = "PrimitiveResetRequest";
static constexpr const char* kCompleteMessageType = "PrimitiveResetComplete";
static constexpr const char* kReceivedAckMessageType = "PrimitiveResetReceivedAck";

using planning::msgpack_value::AsFloat;

struct ResetRequest {
  uint32_t schema_version = kSchemaVersion;
  std::string runtime_instance_id;
  std::string episode_id;
  std::string reset_id;
  std::array<float, 3> start{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> goal{{0.0f, 0.0f, 0.0f}};
  std::vector<uint8_t> immutable_payload;
};

struct ResetComplete {
  uint32_t schema_version = kSchemaVersion;
  std::string runtime_instance_id;
  std::string episode_id;
  std::string reset_id;
  xp::v4::ObservationRef observation_ref;
  std::vector<uint8_t> immutable_payload;
};

struct ResetReceivedAck {
  uint32_t schema_version = kSchemaVersion;
  std::string runtime_instance_id;
  std::string episode_id;
  std::string reset_id;
};

inline void PackObservationRef(
    msgpack::packer<msgpack::sbuffer>* packer,
    const xp::v4::ObservationRef& ref) {
  packer->pack_map(7);
  packer->pack(std::string("depth_id")); packer->pack(ref.depth_id);
  packer->pack(std::string("episode_id")); packer->pack(ref.episode_id);
  packer->pack(std::string("reset_id")); packer->pack(ref.reset_id);
  packer->pack(std::string("runtime_instance_id")); packer->pack(ref.runtime_instance_id);
  packer->pack(std::string("schema_version")); packer->pack(ref.schema_version);
  packer->pack(std::string("sim_time_ns")); packer->pack(ref.sim_time_ns);
  packer->pack(std::string("state_id")); packer->pack(ref.state_id);
}

inline std::vector<uint8_t> SerializeRequest(const ResetRequest& request) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(7);
  packer.pack(std::string("episode_id")); packer.pack(request.episode_id);
  packer.pack(std::string("goal")); packer.pack_array(3);
  for (float value : request.goal) packer.pack(value);
  packer.pack(std::string("message_type")); packer.pack(kRequestMessageType);
  packer.pack(std::string("reset_id")); packer.pack(request.reset_id);
  packer.pack(std::string("runtime_instance_id")); packer.pack(request.runtime_instance_id);
  packer.pack(std::string("schema_version")); packer.pack(request.schema_version);
  packer.pack(std::string("start")); packer.pack_array(3);
  for (float value : request.start) packer.pack(value);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

inline std::vector<uint8_t> SerializeComplete(const ResetComplete& complete) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(6);
  packer.pack(std::string("episode_id")); packer.pack(complete.episode_id);
  packer.pack(std::string("message_type")); packer.pack(kCompleteMessageType);
  packer.pack(std::string("observation_ref")); PackObservationRef(&packer, complete.observation_ref);
  packer.pack(std::string("reset_id")); packer.pack(complete.reset_id);
  packer.pack(std::string("runtime_instance_id")); packer.pack(complete.runtime_instance_id);
  packer.pack(std::string("schema_version")); packer.pack(complete.schema_version);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

inline std::vector<uint8_t> SerializeReceivedAck(const ResetReceivedAck& ack) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(5);
  packer.pack(std::string("episode_id")); packer.pack(ack.episode_id);
  packer.pack(std::string("message_type")); packer.pack(kReceivedAckMessageType);
  packer.pack(std::string("reset_id")); packer.pack(ack.reset_id);
  packer.pack(std::string("runtime_instance_id")); packer.pack(ack.runtime_instance_id);
  packer.pack(std::string("schema_version")); packer.pack(ack.schema_version);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

inline bool ParseRequest(
    const char* data, size_t size, ResetRequest* request, std::string* error) {
  try {
    msgpack::object_handle handle = msgpack::unpack(data, size);
    const msgpack::object& root = handle.get();
    const msgpack::object* field = nullptr;
    std::string message_type;
    if (!result_wire::Required(root, "schema_version", &field, error) ||
        !result_wire::AsUInt32(*field, &request->schema_version, error) ||
        request->schema_version != kSchemaVersion ||
        !result_wire::Required(root, "message_type", &field, error) ||
        !result_wire::AsString(*field, &message_type, error) ||
        message_type != kRequestMessageType ||
        !result_wire::Required(root, "runtime_instance_id", &field, error) ||
        !result_wire::AsString(*field, &request->runtime_instance_id, error) ||
        !result_wire::Required(root, "episode_id", &field, error) ||
        !result_wire::AsString(*field, &request->episode_id, error) ||
        !result_wire::Required(root, "reset_id", &field, error) ||
        !result_wire::AsString(*field, &request->reset_id, error)) {
      if (error->empty()) *error = "reset request envelope is invalid";
      return false;
    }
    if (!result_wire::Required(root, "start", &field, error) ||
        field->type != msgpack::type::ARRAY || field->via.array.size != 3) {
      *error = "reset start must contain three values";
      return false;
    }
    for (size_t index = 0; index < 3; ++index) {
      if (!AsFloat(field->via.array.ptr[index], &request->start[index], error))
        return false;
    }
    if (!result_wire::Required(root, "goal", &field, error) ||
        field->type != msgpack::type::ARRAY || field->via.array.size != 3) {
      *error = "reset goal must contain three values";
      return false;
    }
    for (size_t index = 0; index < 3; ++index) {
      if (!AsFloat(field->via.array.ptr[index], &request->goal[index], error))
        return false;
    }
    return true;
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

inline bool ParseObservationRef(
    const msgpack::object& root,
    xp::v4::ObservationRef* ref,
    std::string* error) {
  const msgpack::object* field = nullptr;
  return result_wire::Required(root, "schema_version", &field, error) &&
      result_wire::AsUInt32(*field, &ref->schema_version, error) &&
      result_wire::Required(root, "runtime_instance_id", &field, error) &&
      result_wire::AsString(*field, &ref->runtime_instance_id, error) &&
      result_wire::Required(root, "episode_id", &field, error) &&
      result_wire::AsString(*field, &ref->episode_id, error) &&
      result_wire::Required(root, "reset_id", &field, error) &&
      result_wire::AsString(*field, &ref->reset_id, error) &&
      result_wire::Required(root, "state_id", &field, error) &&
      result_wire::AsInt64(*field, &ref->state_id, error) &&
      result_wire::Required(root, "depth_id", &field, error) &&
      result_wire::AsString(*field, &ref->depth_id, error) &&
      result_wire::Required(root, "sim_time_ns", &field, error) &&
      result_wire::AsUInt64(*field, &ref->sim_time_ns, error) &&
      ref->schema_version == kSchemaVersion;
}

inline bool ParseComplete(
    const char* data, size_t size, ResetComplete* complete, std::string* error) {
  try {
    msgpack::object_handle handle = msgpack::unpack(data, size);
    const msgpack::object& root = handle.get();
    const msgpack::object* field = nullptr;
    std::string message_type;
    if (!result_wire::Required(root, "schema_version", &field, error) ||
        !result_wire::AsUInt32(*field, &complete->schema_version, error) ||
        complete->schema_version != kSchemaVersion ||
        !result_wire::Required(root, "message_type", &field, error) ||
        !result_wire::AsString(*field, &message_type, error) ||
        message_type != kCompleteMessageType ||
        !result_wire::Required(root, "runtime_instance_id", &field, error) ||
        !result_wire::AsString(*field, &complete->runtime_instance_id, error) ||
        !result_wire::Required(root, "episode_id", &field, error) ||
        !result_wire::AsString(*field, &complete->episode_id, error) ||
        !result_wire::Required(root, "reset_id", &field, error) ||
        !result_wire::AsString(*field, &complete->reset_id, error) ||
        !result_wire::Required(root, "observation_ref", &field, error) ||
        !ParseObservationRef(*field, &complete->observation_ref, error)) {
      if (error->empty()) *error = "reset completion is invalid";
      return false;
    }
    return true;
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

inline bool ParseReceivedAck(
    const char* data, size_t size, ResetReceivedAck* ack, std::string* error) {
  try {
    msgpack::object_handle handle = msgpack::unpack(data, size);
    const msgpack::object& root = handle.get();
    const msgpack::object* field = nullptr;
    std::string message_type;
    return result_wire::Required(root, "schema_version", &field, error) &&
        result_wire::AsUInt32(*field, &ack->schema_version, error) &&
        ack->schema_version == kSchemaVersion &&
        result_wire::Required(root, "message_type", &field, error) &&
        result_wire::AsString(*field, &message_type, error) &&
        message_type == kReceivedAckMessageType &&
        result_wire::Required(root, "runtime_instance_id", &field, error) &&
        result_wire::AsString(*field, &ack->runtime_instance_id, error) &&
        result_wire::Required(root, "episode_id", &field, error) &&
        result_wire::AsString(*field, &ack->episode_id, error) &&
        result_wire::Required(root, "reset_id", &field, error) &&
        result_wire::AsString(*field, &ack->reset_id, error);
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

}  // namespace primitive_reset_wire
}  // namespace planning
