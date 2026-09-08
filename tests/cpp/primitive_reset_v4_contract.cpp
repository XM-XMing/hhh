#include "planning/primitive_reset_wire.hpp"

#include <iostream>
#include <stdexcept>

namespace {

using planning::primitive_reset_wire::ResetComplete;
using planning::primitive_reset_wire::ResetReceivedAck;
using planning::primitive_reset_wire::ResetRequest;

planning::xm_protocol::v4::ObservationRef Ref() {
  planning::xm_protocol::v4::ObservationRef ref;
  ref.schema_version = 4;
  ref.runtime_instance_id = "worker-00";
  ref.episode_id = "episode-7";
  ref.reset_id = "reset-7";
  ref.state_id = 123;
  ref.depth_id = "depth-123";
  ref.sim_time_ns = 2460000000ULL;
  return ref;
}

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  ResetRequest request;
  request.runtime_instance_id = "worker-00";
  request.episode_id = "episode-7";
  request.reset_id = "reset-7";
  request.start = {{1.0f, 2.0f, 3.0f}};
  request.goal = {{4.0f, 5.0f, 6.0f}};
  const std::vector<uint8_t> request_bytes =
      planning::primitive_reset_wire::SerializeRequest(request);
  ResetRequest decoded_request;
  std::string error;
  Require(planning::primitive_reset_wire::ParseRequest(
              reinterpret_cast<const char*>(request_bytes.data()),
              request_bytes.size(), &decoded_request, &error),
          "reset request parse failed: " + error);
  Require(decoded_request.runtime_instance_id == request.runtime_instance_id,
          "request runtime mismatch");
  Require(decoded_request.episode_id == request.episode_id,
          "request episode mismatch");
  Require(decoded_request.reset_id == request.reset_id,
          "request reset mismatch");
  Require(decoded_request.start == request.start, "request start mismatch");
  Require(decoded_request.goal == request.goal, "request goal mismatch");
  Require(request_bytes == planning::primitive_reset_wire::SerializeRequest(request),
          "request serialization is not deterministic");

  ResetComplete complete;
  complete.runtime_instance_id = "worker-00";
  complete.episode_id = "episode-7";
  complete.reset_id = "reset-7";
  complete.observation_ref = Ref();
  const std::vector<uint8_t> complete_bytes =
      planning::primitive_reset_wire::SerializeComplete(complete);
  ResetComplete decoded_complete;
  Require(planning::primitive_reset_wire::ParseComplete(
              reinterpret_cast<const char*>(complete_bytes.data()),
              complete_bytes.size(), &decoded_complete, &error),
          "reset complete parse failed: " + error);
  Require(decoded_complete.observation_ref.state_id == 123,
          "complete state identity mismatch");
  Require(decoded_complete.observation_ref.depth_id == "depth-123",
          "complete depth identity mismatch");

  ResetReceivedAck ack;
  ack.runtime_instance_id = "worker-00";
  ack.episode_id = "episode-7";
  ack.reset_id = "reset-7";
  const std::vector<uint8_t> ack_bytes =
      planning::primitive_reset_wire::SerializeReceivedAck(ack);
  ResetReceivedAck decoded_ack;
  Require(planning::primitive_reset_wire::ParseReceivedAck(
              reinterpret_cast<const char*>(ack_bytes.data()), ack_bytes.size(),
              &decoded_ack, &error),
          "reset ACK parse failed: " + error);
  Require(decoded_ack.reset_id == ack.reset_id, "ACK reset identity mismatch");
  std::cout << "C++ reset contract tests: 4 passed\n";
  return 0;
}
