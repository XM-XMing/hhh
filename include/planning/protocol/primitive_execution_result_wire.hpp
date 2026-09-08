#pragma once

#include <array>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <msgpack.hpp>
#include <string>
#include <vector>

#include "planning/bridge/primitive_execution_result_broker.hpp"
#include "planning/protocol/msgpack_value.hpp"

namespace planning {
namespace primitive_execution_result_wire {

namespace xp = planning::xm_protocol;
namespace broker = planning::primitive_execution_result_broker;

static constexpr uint32_t kSchemaVersion = 4;
static constexpr const char* kResultMessageType = "PrimitiveExecutionResult";
static constexpr const char* kAckMessageType = "PrimitiveExecutionResultAck";
static constexpr const char* kReceiptAckMessageType = "PrimitiveExecutionResultReceiptAck";
static constexpr const char* kReadyMessageType = "PrimitiveExecutionResultReady";
static constexpr const char* kCommitMessageType = "PrimitiveExecutionResultCommit";

inline bool Find(
    const msgpack::object& object,
    const char* key,
    const msgpack::object** value) {
  if (object.type != msgpack::type::MAP) return false;
  for (uint32_t index = 0; index < object.via.map.size; ++index) {
    const msgpack::object_kv& entry = object.via.map.ptr[index];
    if (entry.key.type == msgpack::type::STR &&
        entry.key.via.str.size == std::strlen(key) &&
        std::strncmp(entry.key.via.str.ptr, key, entry.key.via.str.size) == 0) {
      *value = &entry.val;
      return true;
    }
  }
  return false;
}

inline bool Required(
    const msgpack::object& object,
    const char* key,
    const msgpack::object** value,
    std::string* error) {
  if (Find(object, key, value)) return true;
  *error = std::string("missing field: ") + key;
  return false;
}

inline bool AsString(
    const msgpack::object& object,
    std::string* value,
    std::string* error) {
  if (object.type != msgpack::type::STR) {
    *error = "expected string";
    return false;
  }
  value->assign(object.via.str.ptr, object.via.str.size);
  if (value->empty()) {
    *error = "empty string";
    return false;
  }
  return true;
}

inline bool AsUInt64(
    const msgpack::object& object,
    uint64_t* value,
    std::string* error) {
  if (object.type != msgpack::type::POSITIVE_INTEGER) {
    *error = "expected unsigned integer";
    return false;
  }
  *value = object.via.u64;
  return true;
}

inline bool AsUInt32(
    const msgpack::object& object,
    uint32_t* value,
    std::string* error) {
  uint64_t wide = 0;
  if (!AsUInt64(object, &wide, error) || wide > std::numeric_limits<uint32_t>::max()) {
    if (wide > std::numeric_limits<uint32_t>::max()) *error = "unsigned integer exceeds uint32";
    return false;
  }
  *value = static_cast<uint32_t>(wide);
  return true;
}

using planning::msgpack_value::AsInt64;

inline bool AsOptionalInt64(
    const msgpack::object& object,
    bool* present,
    int64_t* value,
    std::string* error) {
  if (object.type == msgpack::type::NIL) {
    *present = false;
    return true;
  }
  *present = true;
  return AsInt64(object, value, error);
}

inline bool AsOptionalUInt64(
    const msgpack::object& object,
    bool* present,
    uint64_t* value,
    std::string* error) {
  if (object.type == msgpack::type::NIL) {
    *present = false;
    return true;
  }
  *present = true;
  return AsUInt64(object, value, error);
}

inline bool AsHash(
    const msgpack::object& object,
    std::array<uint8_t, 32>* value,
    std::string* error) {
  if (object.type != msgpack::type::BIN || object.via.bin.size != value->size()) {
    *error = "expected binary sha256";
    return false;
  }
  std::copy(
      object.via.bin.ptr,
      object.via.bin.ptr + value->size(),
      value->begin());
  return true;
}

inline bool ParseObservation(
    const msgpack::object& object,
    bool* present,
    xp::v4::EndpointObservationRef* observation,
    std::string* error) {
  if (object.type == msgpack::type::NIL) {
    *present = false;
    return true;
  }
  *present = true;
  const msgpack::object* schema_version = nullptr;
  const msgpack::object* runtime_instance_id = nullptr;
  const msgpack::object* episode_id = nullptr;
  const msgpack::object* reset_id = nullptr;
  const msgpack::object* depth_id = nullptr;
  const msgpack::object* state_id = nullptr;
  const msgpack::object* sim_time = nullptr;
  if (!Required(object, "schema_version", &schema_version, error) ||
      !Required(object, "runtime_instance_id", &runtime_instance_id, error) ||
      !Required(object, "episode_id", &episode_id, error) ||
      !Required(object, "reset_id", &reset_id, error) ||
      !Required(object, "depth_id", &depth_id, error) ||
      !Required(object, "state_id", &state_id, error) ||
      !Required(object, "sim_time_ns", &sim_time, error) ||
      !AsUInt32(*schema_version, &observation->schema_version, error) ||
      !AsString(*runtime_instance_id, &observation->runtime_instance_id, error) ||
      !AsString(*episode_id, &observation->episode_id, error) ||
      !AsString(*reset_id, &observation->reset_id, error) ||
      !AsString(*depth_id, &observation->depth_id, error) ||
      !AsInt64(*state_id, &observation->state_id, error) ||
      !AsUInt64(*sim_time, &observation->sim_time_ns, error)) {
    return false;
  }
  return true;
}

inline bool ParseResult(
    const char* data,
    size_t size,
    xp::v4::PrimitiveExecutionResult* result,
    std::string* error) {
  try {
    msgpack::object_handle handle = msgpack::unpack(data, size);
    const msgpack::object& object = handle.get();
    const msgpack::object* field = nullptr;
    uint32_t schema_version = 0;
    if (!Required(object, "schema_version", &field, error) ||
        !AsUInt32(*field, &schema_version, error)) return false;
    if (schema_version != kSchemaVersion) {
      *error = "schema_version mismatch";
      return false;
    }

    result->schema_version = schema_version;
    if (!Required(object, "message_type", &field, error) ||
        !AsString(*field, &result->message_type, error) ||
        result->message_type != kResultMessageType) {
      if (error->empty()) *error = "message_type mismatch";
      return false;
    }
    if (!Required(object, "runtime_instance_id", &field, error) ||
        !AsString(*field, &result->runtime_instance_id, error) ||
        !Required(object, "execution_id", &field, error) ||
        !AsUInt64(*field, &result->execution_id, error) ||
        !Required(object, "status", &field, error) ||
        !AsString(*field, &result->status, error) ||
        !Required(object, "requested_frame_count", &field, error) ||
        !AsUInt32(*field, &result->requested_frame_count, error) ||
        !Required(object, "applied_frame_count", &field, error) ||
        !AsUInt32(*field, &result->applied_frame_count, error) ||
        !Required(object, "reason_code", &field, error) ||
        !AsString(*field, &result->reason_code, error) ||
        !Required(object, "command_sequence_hash", &field, error) ||
        !AsHash(*field, &result->command_sequence_hash, error) ||
        !Required(object, "result_generation", &field, error) ||
        !AsUInt32(*field, &result->result_generation, error) ||
        !Required(object, "result_payload_hash", &field, error) ||
        !AsHash(*field, &result->result_payload_hash, error)) {
      return false;
    }

    int64_t last_frame = 0;
    if (!Required(object, "last_applied_frame_index", &field, error) ||
        !AsInt64(*field, &last_frame, error) ||
        last_frame < std::numeric_limits<int32_t>::min() ||
        last_frame > std::numeric_limits<int32_t>::max()) {
      if (error->empty()) *error = "last_applied_frame_index exceeds int32";
      return false;
    }
    result->last_applied_frame_index = static_cast<int32_t>(last_frame);

    if (!Required(object, "first_applied_state_id", &field, error) ||
        !AsOptionalInt64(*field, &result->has_first_applied_state_id,
                         &result->first_applied_state_id, error) ||
        !Required(object, "endpoint_state_id", &field, error) ||
        !AsOptionalInt64(*field, &result->has_endpoint_state_id,
                         &result->endpoint_state_id, error) ||
        !Required(object, "endpoint_sim_time_ns", &field, error) ||
        !AsOptionalUInt64(*field, &result->has_endpoint_sim_time_ns,
                          &result->endpoint_sim_time_ns, error) ||
        !Required(object, "endpoint_observation_ref", &field, error) ||
        !ParseObservation(*field, &result->has_endpoint_observation_ref,
                          &result->endpoint_observation_ref, error)) {
      return false;
    }

    xp::v4::canonical_result_payload_bytes(*result);
    return true;
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

inline void PackHash(
    msgpack::packer<msgpack::sbuffer>* packer,
    const std::array<uint8_t, 32>& value) {
  packer->pack_bin(static_cast<uint32_t>(value.size()));
  packer->pack_bin_body(
      reinterpret_cast<const char*>(value.data()), value.size());
}

inline void PackOptionalInt64(
    msgpack::packer<msgpack::sbuffer>* packer,
    bool present,
    int64_t value) {
  if (present) packer->pack(value);
  else packer->pack_nil();
}

inline void PackOptionalUInt64(
    msgpack::packer<msgpack::sbuffer>* packer,
    bool present,
    uint64_t value) {
  if (present) packer->pack(value);
  else packer->pack_nil();
}

inline std::vector<uint8_t> SerializeResult(
    const xp::v4::PrimitiveExecutionResult& result) {
  xp::v4::canonical_result_payload_bytes(result);
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(16);
  packer.pack("applied_frame_count"); packer.pack(result.applied_frame_count);
  packer.pack("command_sequence_hash"); PackHash(&packer, result.command_sequence_hash);
  packer.pack("endpoint_observation_ref");
  if (result.has_endpoint_observation_ref) {
    packer.pack_map(7);
    packer.pack("depth_id"); packer.pack(result.endpoint_observation_ref.depth_id);
    packer.pack("episode_id"); packer.pack(result.endpoint_observation_ref.episode_id);
    packer.pack("reset_id"); packer.pack(result.endpoint_observation_ref.reset_id);
    packer.pack("runtime_instance_id"); packer.pack(result.endpoint_observation_ref.runtime_instance_id);
    packer.pack("schema_version"); packer.pack(result.endpoint_observation_ref.schema_version);
    packer.pack("sim_time_ns"); packer.pack(result.endpoint_observation_ref.sim_time_ns);
    packer.pack("state_id"); packer.pack(result.endpoint_observation_ref.state_id);
  } else {
    packer.pack_nil();
  }
  packer.pack("endpoint_sim_time_ns");
  PackOptionalUInt64(&packer, result.has_endpoint_sim_time_ns, result.endpoint_sim_time_ns);
  packer.pack("endpoint_state_id");
  PackOptionalInt64(&packer, result.has_endpoint_state_id, result.endpoint_state_id);
  packer.pack("execution_id"); packer.pack(result.execution_id);
  packer.pack("first_applied_state_id");
  PackOptionalInt64(&packer, result.has_first_applied_state_id, result.first_applied_state_id);
  packer.pack("last_applied_frame_index"); packer.pack(result.last_applied_frame_index);
  packer.pack("message_type"); packer.pack(result.message_type);
  packer.pack("reason_code"); packer.pack(result.reason_code);
  packer.pack("requested_frame_count"); packer.pack(result.requested_frame_count);
  packer.pack("result_generation"); packer.pack(result.result_generation);
  packer.pack("result_payload_hash"); PackHash(&packer, result.result_payload_hash);
  packer.pack("runtime_instance_id"); packer.pack(result.runtime_instance_id);
  packer.pack("schema_version"); packer.pack(result.schema_version);
  packer.pack("status"); packer.pack(result.status);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

inline std::vector<uint8_t> SerializeAck(
    const xp::v4::PrimitiveExecutionResultAck& ack) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(7);
  packer.pack("ack_status"); packer.pack(ack.ack_status);
  packer.pack("command_sequence_hash"); PackHash(&packer, ack.command_sequence_hash);
  packer.pack("execution_id"); packer.pack(ack.execution_id);
  packer.pack("message_type"); packer.pack(ack.message_type);
  packer.pack("result_payload_hash"); PackHash(&packer, ack.result_payload_hash);
  packer.pack("runtime_instance_id"); packer.pack(ack.runtime_instance_id);
  packer.pack("schema_version"); packer.pack(ack.schema_version);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

inline std::vector<uint8_t> SerializeReceiptAck(
    const xp::v4::PrimitiveExecutionResultReceiptAck& ack) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(7);
  packer.pack("command_sequence_hash"); PackHash(&packer, ack.command_sequence_hash);
  packer.pack("execution_id"); packer.pack(ack.execution_id);
  packer.pack("message_type"); packer.pack(ack.message_type);
  packer.pack("receipt_status"); packer.pack(ack.receipt_status);
  packer.pack("result_payload_hash"); PackHash(&packer, ack.result_payload_hash);
  packer.pack("runtime_instance_id"); packer.pack(ack.runtime_instance_id);
  packer.pack("schema_version"); packer.pack(ack.schema_version);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

inline std::vector<uint8_t> ReadyMessage() {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(2);
  packer.pack("message_type"); packer.pack(kReadyMessageType);
  packer.pack("schema_version"); packer.pack(kSchemaVersion);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

struct CommitMessage {
  broker::ResultKey key;
  std::array<uint8_t, 32> result_payload_hash{{}};
  std::array<uint8_t, 32> command_sequence_hash{{}};
};

inline std::vector<uint8_t> SerializeCommitAck(
    const CommitMessage& commit,
    const std::string& status) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(7);
  packer.pack("command_sequence_hash"); PackHash(&packer, commit.command_sequence_hash);
  packer.pack("commit_status"); packer.pack(status);
  packer.pack("execution_id"); packer.pack(commit.key.execution_id);
  packer.pack("message_type"); packer.pack("PrimitiveExecutionResultCommitAck");
  packer.pack("result_payload_hash"); PackHash(&packer, commit.result_payload_hash);
  packer.pack("runtime_instance_id"); packer.pack(commit.key.runtime_instance_id);
  packer.pack("schema_version"); packer.pack(kSchemaVersion);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

inline bool ParseReady(
    const char* data,
    size_t size,
    std::string* error) {
  try {
    msgpack::object_handle handle = msgpack::unpack(data, size);
    const msgpack::object& object = handle.get();
    const msgpack::object* field = nullptr;
    uint32_t schema = 0;
    std::string message_type;
    return Required(object, "schema_version", &field, error) &&
        AsUInt32(*field, &schema, error) && schema == kSchemaVersion &&
        Required(object, "message_type", &field, error) &&
        AsString(*field, &message_type, error) &&
        message_type == kReadyMessageType;
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

inline bool ParseReceiptAck(
    const char* data,
    size_t size,
    xp::v4::PrimitiveExecutionResultReceiptAck* ack,
    std::string* error) {
  try {
    msgpack::object_handle handle = msgpack::unpack(data, size);
    const msgpack::object& object = handle.get();
    const msgpack::object* field = nullptr;
    uint32_t schema = 0;
    if (!Required(object, "schema_version", &field, error) ||
        !AsUInt32(*field, &schema, error) || schema != kSchemaVersion ||
        !Required(object, "message_type", &field, error) ||
        !AsString(*field, &ack->message_type, error) ||
        ack->message_type != kReceiptAckMessageType ||
        !Required(object, "runtime_instance_id", &field, error) ||
        !AsString(*field, &ack->runtime_instance_id, error) ||
        !Required(object, "execution_id", &field, error) ||
        !AsUInt64(*field, &ack->execution_id, error) ||
        !Required(object, "receipt_status", &field, error) ||
        !AsString(*field, &ack->receipt_status, error) ||
        !Required(object, "result_payload_hash", &field, error) ||
        !AsHash(*field, &ack->result_payload_hash, error) ||
        !Required(object, "command_sequence_hash", &field, error) ||
        !AsHash(*field, &ack->command_sequence_hash, error)) {
      if (error->empty()) *error = "receipt ACK schema or field mismatch";
      return false;
    }
    ack->schema_version = schema;
    if (ack->receipt_status != "RECEIVED") {
      *error = "receipt_status must be RECEIVED";
      return false;
    }
    return true;
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

inline bool ParseCommit(
    const char* data,
    size_t size,
    CommitMessage* commit,
    std::string* error) {
  try {
    msgpack::object_handle handle = msgpack::unpack(data, size);
    const msgpack::object& object = handle.get();
    const msgpack::object* field = nullptr;
    uint32_t schema = 0;
    std::string message_type;
    std::string commit_status;
    if (!Required(object, "schema_version", &field, error) ||
        !AsUInt32(*field, &schema, error) || schema != kSchemaVersion ||
        !Required(object, "message_type", &field, error) ||
        !AsString(*field, &message_type, error) ||
        message_type != kCommitMessageType ||
        !Required(object, "commit_status", &field, error) ||
        !AsString(*field, &commit_status, error) ||
        commit_status != "COMMITTED" ||
        !Required(object, "runtime_instance_id", &field, error) ||
        !AsString(*field, &commit->key.runtime_instance_id, error) ||
        !Required(object, "execution_id", &field, error) ||
        !AsUInt64(*field, &commit->key.execution_id, error) ||
        !Required(object, "command_sequence_hash", &field, error) ||
        !AsHash(*field, &commit->command_sequence_hash, error) ||
        !Required(object, "result_payload_hash", &field, error) ||
        !AsHash(*field, &commit->result_payload_hash, error)) {
      if (schema != kSchemaVersion && error->empty()) *error = "schema_version mismatch";
      return false;
    }
    return true;
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

}  // namespace primitive_execution_result_wire
}  // namespace planning
