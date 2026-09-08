#pragma once

#include <cstdint>
#include <msgpack.hpp>
#include <string>
#include <vector>

#include "planning/bridge/primitive_execution_command_broker.hpp"
#include "planning/protocol/msgpack_value.hpp"
#include "planning/protocol/primitive_execution_result_wire.hpp"

namespace planning {
namespace primitive_execution_command_wire {

namespace xp = planning::xm_protocol;
namespace command_broker = planning::primitive_execution_command_broker;
namespace result_wire = planning::primitive_execution_result_wire;

static constexpr uint32_t kSchemaVersion = 4;
static constexpr const char* kCommandMessageType = "PrimitiveExecutionCommand";
static constexpr const char* kReceiptMessageType =
    "PrimitiveExecutionCommandReceiptAck";
static constexpr const char* kReadyMessageType = "PrimitiveExecutionCommandReady";

using planning::msgpack_value::AsFloat;

inline std::vector<uint8_t> SerializeCommand(
    const xp::v4::PrimitiveExecutionCommand& command) {
  return xp::v4::canonical_command_payload_bytes(command);
}

inline std::vector<uint8_t> SerializeReceipt(
    const xp::v4::PrimitiveExecutionCommandReceiptAck& receipt) {
  return xp::v4::canonical_command_receipt_ack_bytes(receipt);
}

inline std::vector<uint8_t> SerializeReady(const std::string& runtime_id) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(3);
  packer.pack("message_type");
  packer.pack(kReadyMessageType);
  packer.pack("runtime_instance_id");
  packer.pack(runtime_id);
  packer.pack("schema_version");
  packer.pack(kSchemaVersion);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

inline bool ParseReady(
    const char* data, size_t size, std::string* runtime_id, std::string* error) {
  try {
    msgpack::object_handle handle = msgpack::unpack(data, size);
    const msgpack::object& root = handle.get();
    const msgpack::object* field = nullptr;
    uint32_t schema = 0;
    std::string message_type;
    return result_wire::Required(root, "schema_version", &field, error) &&
        result_wire::AsUInt32(*field, &schema, error) &&
        schema == kSchemaVersion &&
        result_wire::Required(root, "message_type", &field, error) &&
        result_wire::AsString(*field, &message_type, error) &&
        message_type == kReadyMessageType &&
        result_wire::Required(root, "runtime_instance_id", &field, error) &&
        result_wire::AsString(*field, runtime_id, error);
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

inline bool ParseCommand(
    const char* data,
    size_t size,
    xp::v4::PrimitiveExecutionCommand* command,
    std::string* error) {
  try {
    msgpack::object_handle handle = msgpack::unpack(data, size);
    const msgpack::object& root = handle.get();
    const msgpack::object* field = nullptr;
    uint32_t schema = 0;
    if (!result_wire::Required(root, "schema_version", &field, error) ||
        !result_wire::AsUInt32(*field, &schema, error) ||
        schema != kSchemaVersion ||
        !result_wire::Required(root, "message_type", &field, error) ||
        !result_wire::AsString(*field, &command->message_type, error) ||
        command->message_type != kCommandMessageType ||
        !result_wire::Required(root, "runtime_instance_id", &field, error) ||
        !result_wire::AsString(*field, &command->runtime_instance_id, error) ||
        !result_wire::Required(root, "execution_id", &field, error) ||
        !result_wire::AsUInt64(*field, &command->execution_id, error) ||
        !result_wire::Required(root, "command_sequence_hash", &field, error) ||
        !result_wire::AsHash(*field, &command->command_sequence_hash, error) ||
        !result_wire::Required(root, "frames", &field, error)) {
      if (error->empty()) *error = "command envelope is invalid";
      return false;
    }
    command->schema_version = schema;
    if (field->type != msgpack::type::ARRAY ||
        field->via.array.size != xp::v4::kRequestedFrameCount) {
      *error = "command frame count mismatch";
      return false;
    }
    command->frames.clear();
    command->frames.reserve(field->via.array.size);
    for (uint32_t index = 0; index < field->via.array.size; ++index) {
      const msgpack::object& frame_object = field->via.array.ptr[index];
      const msgpack::object* frame_field = nullptr;
      xp::v4::PrimitiveExecutionFrame frame;
      uint32_t frame_index = 0;
      if (!result_wire::Required(frame_object, "frame_index", &frame_field, error) ||
          !result_wire::AsUInt32(*frame_field, &frame_index, error) ||
          !result_wire::Required(frame_object, "command_id", &frame_field, error) ||
          !result_wire::AsInt64(*frame_field, &frame.command_id, error) ||
          !result_wire::Required(frame_object, "action", &frame_field, error) ||
          frame_field->type != msgpack::type::ARRAY) {
        if (error->empty()) *error = "command frame is invalid";
        return false;
      }
      frame.frame_index = frame_index;
      frame.action.reserve(frame_field->via.array.size);
      for (uint32_t action_index = 0;
           action_index < frame_field->via.array.size; ++action_index) {
        float action = 0.0f;
        if (!AsFloat(frame_field->via.array.ptr[action_index], &action, error))
          return false;
        frame.action.push_back(action);
      }
      command->frames.push_back(frame);
    }
    xp::v4::validate_command(*command);
    return true;
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

inline bool ParseReceipt(
    const char* data,
    size_t size,
    xp::v4::PrimitiveExecutionCommandReceiptAck* receipt,
    std::string* error) {
  try {
    msgpack::object_handle handle = msgpack::unpack(data, size);
    const msgpack::object& root = handle.get();
    const msgpack::object* field = nullptr;
    uint32_t schema = 0;
    if (!result_wire::Required(root, "schema_version", &field, error) ||
        !result_wire::AsUInt32(*field, &schema, error) ||
        schema != kSchemaVersion ||
        !result_wire::Required(root, "message_type", &field, error) ||
        !result_wire::AsString(*field, &receipt->message_type, error) ||
        receipt->message_type != kReceiptMessageType ||
        !result_wire::Required(root, "runtime_instance_id", &field, error) ||
        !result_wire::AsString(*field, &receipt->runtime_instance_id, error) ||
        !result_wire::Required(root, "execution_id", &field, error) ||
        !result_wire::AsUInt64(*field, &receipt->execution_id, error) ||
        !result_wire::Required(root, "ack_status", &field, error) ||
        !result_wire::AsString(*field, &receipt->ack_status, error) ||
        !result_wire::Required(root, "reason_code", &field, error) ||
        !result_wire::AsString(*field, &receipt->reason_code, error) ||
        !result_wire::Required(root, "command_sequence_hash", &field, error) ||
        !result_wire::AsHash(*field, &receipt->command_sequence_hash, error)) {
      if (error->empty()) *error = "command receipt is invalid";
      return false;
    }
    receipt->schema_version = schema;
    return true;
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

}  // namespace primitive_execution_command_wire
}  // namespace planning
