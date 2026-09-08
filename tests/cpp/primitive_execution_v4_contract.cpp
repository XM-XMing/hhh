#include "planning/xm_protocol.hpp"

#include <openssl/sha.h>

#include <fstream>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using planning::xm_protocol::v4::EndpointObservationRef;
using planning::xm_protocol::v4::PrimitiveExecutionFrame;
using planning::xm_protocol::v4::PrimitiveExecutionResult;

std::string read_text(const std::string& path) {
  std::ifstream stream(path.c_str());
  if (!stream) throw std::runtime_error("cannot read " + path);
  std::ostringstream content;
  content << stream.rdbuf();
  std::string value = content.str();
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' '))
    value.pop_back();
  return value;
}

std::vector<uint8_t> decode_hex(const std::string& text) {
  if (text.size() % 2 != 0) throw std::runtime_error("odd hex length");
  std::vector<uint8_t> bytes;
  for (size_t i = 0; i < text.size(); i += 2) {
    unsigned int value = 0;
    std::istringstream stream(text.substr(i, 2));
    stream >> std::hex >> value;
    if (stream.fail()) throw std::runtime_error("invalid hex");
    bytes.push_back(static_cast<uint8_t>(value));
  }
  return bytes;
}

std::string sha256_hex(const std::vector<uint8_t>& bytes) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(bytes.data(), bytes.size(), digest);
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned char value : digest) output << std::setw(2) << static_cast<int>(value);
  return output.str();
}

std::array<uint8_t, 32> decode_hash(const std::string& text) {
  const std::vector<uint8_t> bytes = decode_hex(text);
  if (bytes.size() != 32) throw std::runtime_error("hash must contain 32 bytes");
  std::array<uint8_t, 32> value{{}};
  std::copy(bytes.begin(), bytes.end(), value.begin());
  return value;
}

std::vector<PrimitiveExecutionFrame> command_frames() {
  std::vector<PrimitiveExecutionFrame> frames;
  for (uint32_t index = 0; index < 25; ++index) {
    PrimitiveExecutionFrame frame;
    frame.frame_index = index;
    frame.command_id = 7000000000LL + static_cast<int64_t>(index);
    frame.action = {static_cast<float>(index), -2.0f, 0.125f};
    frames.push_back(frame);
  }
  return frames;
}

PrimitiveExecutionResult result_for(
    const std::string& status,
    const std::array<uint8_t, 32>& command_hash) {
  PrimitiveExecutionResult result;
  result.runtime_instance_id = "worker-00-runtime-test";
  result.execution_id = 0x0102030405060708ULL;
  result.status = status;
  result.requested_frame_count = 25;
  result.command_sequence_hash = command_hash;
  result.result_generation = 0;
  if (status == "COMPLETE") {
    result.applied_frame_count = 25;
    result.has_first_applied_state_id = true;
    result.first_applied_state_id = 4000;
    result.has_endpoint_state_id = true;
    result.endpoint_state_id = 4024;
    result.last_applied_frame_index = 24;
    result.reason_code = "NONE";
    result.has_endpoint_sim_time_ns = true;
    result.endpoint_sim_time_ns = 5000000000ULL;
    result.has_endpoint_observation_ref = true;
    result.endpoint_observation_ref = EndpointObservationRef{
        4, "worker-00-runtime-test", "episode-v4-0001", "reset-v4-0001",
        4024, "depth-v4-0001", 5000000000ULL};
  } else if (status == "REJECTED") {
    result.applied_frame_count = 0;
    result.last_applied_frame_index = -1;
    result.reason_code = "MALFORMED_COMMAND";
  } else if (status == "CANCELLED") {
    result.applied_frame_count = 11;
    result.has_first_applied_state_id = true;
    result.first_applied_state_id = 7000;
    result.last_applied_frame_index = 10;
    result.reason_code = "STOP";
  } else {
    result.applied_frame_count = 7;
    result.has_first_applied_state_id = true;
    result.first_applied_state_id = 8000;
    result.has_endpoint_state_id = true;
    result.endpoint_state_id = 8007;
    result.last_applied_frame_index = 6;
    result.reason_code = "COLLISION";
    result.has_endpoint_sim_time_ns = true;
    result.endpoint_sim_time_ns = 5140000000ULL;
    result.has_endpoint_observation_ref = true;
    result.endpoint_observation_ref = EndpointObservationRef{
        4, "worker-00-runtime-test", "episode-v4-0001", "reset-v4-0001",
        8007, "depth-v4-failed-0001", 5140000000ULL};
  }
  return result;
}

void assert_equal(const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected,
                  const std::string& label) {
  if (actual != expected) throw std::runtime_error(label + " bytes mismatch");
}

void assert_physics_complete_observation_failure_is_valid(
    const std::array<uint8_t, 32>& command_hash) {
  PrimitiveExecutionResult result = result_for("FAILED", command_hash);
  result.applied_frame_count = 25;
  result.has_first_applied_state_id = true;
  result.first_applied_state_id = 8000;
  result.has_endpoint_state_id = false;
  result.last_applied_frame_index = 24;
  result.reason_code = "PHYSICS_COMPLETE_OBSERVATION_FAILED";
  result.has_endpoint_sim_time_ns = false;
  result.has_endpoint_observation_ref = false;
  planning::xm_protocol::v4::canonical_result_payload_bytes(result);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string fixture_dir = argv[1];
  const std::vector<PrimitiveExecutionFrame> frames = command_frames();
  const std::vector<uint8_t> command_bytes =
      planning::xm_protocol::v4::canonical_command_sequence_bytes(frames);
  const std::array<uint8_t, 32> command_hash = decode_hash(
      read_text(fixture_dir + "/complete.command_sequence_hash.hex"));
  if (sha256_hex(command_bytes) != read_text(fixture_dir + "/complete.command_sequence_hash.sha256"))
    throw std::runtime_error("command hash mismatch");
  for (const std::string& status : {"complete", "rejected", "cancelled", "failed"}) {
    const std::string wire_status = status == "complete" ? "COMPLETE" :
        status == "rejected" ? "REJECTED" : status == "cancelled" ? "CANCELLED" : "FAILED";
    const PrimitiveExecutionResult result = result_for(wire_status, command_hash);
    const std::vector<uint8_t> result_bytes =
        planning::xm_protocol::v4::canonical_result_payload_bytes(result);
    assert_equal(result_bytes,
                 decode_hex(read_text(fixture_dir + "/" + status + ".result_msgpack.hex")),
                 status + " result");
    if (sha256_hex(result_bytes) != read_text(fixture_dir + "/" + status + ".result_payload_hash.sha256"))
      throw std::runtime_error(status + " result hash mismatch");
  }
  assert_physics_complete_observation_failure_is_valid(command_hash);
  return 0;
}
