#include <planning/protocol/endpoint_observation_snapshot_wire.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include <msgpack.hpp>

#include <planning/protocol/msgpack_value.hpp>
#include <planning/protocol/primitive_execution_result_wire.hpp>

namespace xp = planning::xm_protocol;
namespace result_wire = planning::primitive_execution_result_wire;

namespace planning::endpoint_observation_snapshot_wire {

constexpr uint32_t kSchemaVersion = 4;

void PackObservationRef(
    msgpack::packer<msgpack::sbuffer>* pk,
    const xp::v4::ObservationRef& ref) {
  pk->pack_map(7);
  pk->pack(std::string("depth_id")); pk->pack(ref.depth_id);
  pk->pack(std::string("episode_id")); pk->pack(ref.episode_id);
  pk->pack(std::string("reset_id")); pk->pack(ref.reset_id);
  pk->pack(std::string("runtime_instance_id")); pk->pack(ref.runtime_instance_id);
  pk->pack(std::string("schema_version")); pk->pack(ref.schema_version);
  pk->pack(std::string("sim_time_ns")); pk->pack(ref.sim_time_ns);
  pk->pack(std::string("state_id")); pk->pack(ref.state_id);
}

bool ParseObservationRef(
    const msgpack::object& object,
    xp::v4::ObservationRef* ref,
    std::string* error) {
  bool present = false;
  xp::v4::EndpointObservationRef result_ref;
  if (!result_wire::ParseObservation(object, &present, &result_ref, error) || !present) {
    if (present == false && error->empty()) *error = "observation_ref is required";
    return false;
  }
  ref->schema_version = result_ref.schema_version;
  ref->runtime_instance_id = result_ref.runtime_instance_id;
  ref->episode_id = result_ref.episode_id;
  ref->reset_id = result_ref.reset_id;
  ref->state_id = result_ref.state_id;
  ref->depth_id = result_ref.depth_id;
  ref->sim_time_ns = result_ref.sim_time_ns;
  try {
    xp::v4::canonical_observation_ref_bytes(*ref);
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

bool ParseRequestFields(
    const msgpack::object& root,
    xp::v4::SnapshotRequest* request,
    std::string* error) {
  const msgpack::object* observation_ref = nullptr;
  const msgpack::object* execution_id = nullptr;
  const msgpack::object* result_hash = nullptr;
  const msgpack::object* command_hash = nullptr;
  return result_wire::Required(root, "observation_ref", &observation_ref, error) &&
      result_wire::Required(root, "execution_id", &execution_id, error) &&
      result_wire::Required(root, "result_payload_hash", &result_hash, error) &&
      result_wire::Required(root, "command_sequence_hash", &command_hash, error) &&
      ParseObservationRef(*observation_ref, &request->observation_ref, error) &&
      result_wire::AsUInt64(*execution_id, &request->execution_id, error) &&
      result_wire::AsHash(*result_hash, &request->result_payload_hash, error) &&
      result_wire::AsHash(*command_hash, &request->command_sequence_hash, error);
}

bool ParseEnvelope(
    const char* data,
    size_t size,
    const char* expected_type,
    msgpack::object_handle* handle,
    const msgpack::object** root,
    std::string* error) {
  try {
    *handle = msgpack::unpack(data, size);
    *root = &handle->get();
    const msgpack::object* schema = nullptr;
    const msgpack::object* type = nullptr;
    uint32_t schema_version = 0;
    std::string message_type;
    if (!result_wire::Required(**root, "schema_version", &schema, error) ||
        !result_wire::Required(**root, "message_type", &type, error) ||
        !result_wire::AsUInt32(*schema, &schema_version, error) ||
        !result_wire::AsString(*type, &message_type, error)) return false;
    if (schema_version != kSchemaVersion || message_type != expected_type) {
      *error = "snapshot envelope schema or message_type mismatch";
      return false;
    }
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

bool ParseReady(const char* data, size_t size, std::string* runtime_id, std::string* error) {
  msgpack::object_handle handle;
  const msgpack::object* root = nullptr;
  const msgpack::object* runtime = nullptr;
  return ParseEnvelope(data, size, "EndpointObservationSnapshotReady", &handle, &root, error) &&
      result_wire::Required(*root, "runtime_instance_id", &runtime, error) &&
      result_wire::AsString(*runtime, runtime_id, error);
}

bool ParseSnapshotResponse(
    const char* data,
    size_t size,
    xp::v4::SnapshotRequest* request,
    xp::v4::EndpointObservationSnapshot* snapshot,
    std::string* error) {
  msgpack::object_handle handle;
  const msgpack::object* root = nullptr;
  const msgpack::object* state = nullptr;
  const msgpack::object* depth = nullptr;
  const msgpack::object* snapshot_hash = nullptr;
  if (!ParseEnvelope(data, size, "EndpointObservationSnapshot", &handle, &root, error) ||
      !ParseRequestFields(*root, request, error) ||
      !result_wire::Required(*root, "state_bytes", &state, error) ||
      !result_wire::Required(*root, "depth_bytes", &depth, error) ||
      !result_wire::Required(*root, "snapshot_hash", &snapshot_hash, error) ||
      !msgpack_value::AsBytes(*state, &snapshot->state_bytes) ||
      !msgpack_value::AsBytes(*depth, &snapshot->depth_bytes) ||
      !result_wire::AsHash(*snapshot_hash, &snapshot->snapshot_hash, error)) {
    if (error->empty()) *error = "snapshot response bytes are invalid";
    return false;
  }
  snapshot->observation_ref = request->observation_ref;
  return true;
}

bool ParseMissing(
    const char* data,
    size_t size,
    xp::v4::SnapshotRequest* request,
    std::string* error) {
  msgpack::object_handle handle;
  const msgpack::object* root = nullptr;
  return ParseEnvelope(data, size, "SnapshotMissing", &handle, &root, error) &&
      ParseRequestFields(*root, request, error);
}

bool ParsePythonRequest(
    const char* data, size_t size, xp::v4::SnapshotRequest* request,
    std::string* error) {
  msgpack::object_handle handle;
  const msgpack::object* root = nullptr;
  return ParseEnvelope(data, size, "BridgeSnapshotRequest", &handle, &root, error) &&
      ParseRequestFields(*root, request, error);
}

std::vector<uint8_t> SerializeRequest(const xp::v4::SnapshotRequest& request) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> pk(&buffer);
  pk.pack_map(6);
  pk.pack(std::string("command_sequence_hash")); pk.pack_bin(32); pk.pack_bin_body(reinterpret_cast<const char*>(request.command_sequence_hash.data()), 32);
  pk.pack(std::string("execution_id")); pk.pack(request.execution_id);
  pk.pack(std::string("message_type")); pk.pack(std::string("SnapshotRequest"));
  pk.pack(std::string("observation_ref")); PackObservationRef(&pk, request.observation_ref);
  pk.pack(std::string("result_payload_hash")); pk.pack_bin(32); pk.pack_bin_body(reinterpret_cast<const char*>(request.result_payload_hash.data()), 32);
  pk.pack(std::string("schema_version")); pk.pack(kSchemaVersion);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

std::vector<uint8_t> SerializeAck(const xp::v4::SnapshotAck& ack) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> pk(&buffer);
  pk.pack_map(4);
  pk.pack(std::string("message_type")); pk.pack(std::string("SnapshotAck"));
  pk.pack(std::string("observation_ref")); PackObservationRef(&pk, ack.observation_ref);
  pk.pack(std::string("schema_version")); pk.pack(kSchemaVersion);
  pk.pack(std::string("snapshot_hash")); pk.pack_bin(32); pk.pack_bin_body(reinterpret_cast<const char*>(ack.snapshot_hash.data()), 32);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

std::vector<uint8_t> SerializePythonSnapshot(
    const xp::v4::SnapshotRequest& request,
    const xp::v4::EndpointObservationSnapshot& snapshot) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> pk(&buffer);
  pk.pack_map(9);
  pk.pack(std::string("command_sequence_hash")); pk.pack_bin(32); pk.pack_bin_body(reinterpret_cast<const char*>(request.command_sequence_hash.data()), 32);
  pk.pack(std::string("depth_bytes")); pk.pack_bin(snapshot.depth_bytes.size()); pk.pack_bin_body(reinterpret_cast<const char*>(snapshot.depth_bytes.data()), snapshot.depth_bytes.size());
  pk.pack(std::string("execution_id")); pk.pack(request.execution_id);
  pk.pack(std::string("message_type")); pk.pack(std::string("BridgeSnapshotResponse"));
  pk.pack(std::string("observation_ref")); PackObservationRef(&pk, snapshot.observation_ref);
  pk.pack(std::string("result_payload_hash")); pk.pack_bin(32); pk.pack_bin_body(reinterpret_cast<const char*>(request.result_payload_hash.data()), 32);
  pk.pack(std::string("schema_version")); pk.pack(kSchemaVersion);
  pk.pack(std::string("snapshot_hash")); pk.pack_bin(32); pk.pack_bin_body(reinterpret_cast<const char*>(snapshot.snapshot_hash.data()), 32);
  pk.pack(std::string("state_bytes")); pk.pack_bin(snapshot.state_bytes.size()); pk.pack_bin_body(reinterpret_cast<const char*>(snapshot.state_bytes.data()), snapshot.state_bytes.size());
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

std::vector<uint8_t> SerializePythonMissing(const xp::v4::SnapshotRequest& request) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> pk(&buffer);
  pk.pack_map(6);
  pk.pack(std::string("command_sequence_hash")); pk.pack_bin(32); pk.pack_bin_body(reinterpret_cast<const char*>(request.command_sequence_hash.data()), 32);
  pk.pack(std::string("execution_id")); pk.pack(request.execution_id);
  pk.pack(std::string("message_type")); pk.pack(std::string("BridgeSnapshotMissing"));
  pk.pack(std::string("observation_ref")); PackObservationRef(&pk, request.observation_ref);
  pk.pack(std::string("result_payload_hash")); pk.pack_bin(32); pk.pack_bin_body(reinterpret_cast<const char*>(request.result_payload_hash.data()), 32);
  pk.pack(std::string("schema_version")); pk.pack(kSchemaVersion);
  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

}  // namespace planning::endpoint_observation_snapshot_wire
