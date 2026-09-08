#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "planning/bridge/reset_gateway.hpp"
#include "planning/protocol/primitive_reset_wire.hpp"
#include "planning/transport/command_transport.hpp"

namespace {

namespace reset_wire = planning::primitive_reset_wire;

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

reset_wire::ResetRequest Request(uint64_t index) {
  reset_wire::ResetRequest request;
  request.runtime_instance_id = "worker-00-runtime-test";
  request.episode_id = "episode-" + std::to_string(index);
  request.reset_id = "reset-" + std::to_string(index);
  request.start = {{0.0f, 1.0f, 2.0f}};
  request.goal = {{3.0f, 4.0f, 5.0f}};
  return request;
}

reset_wire::ResetComplete Complete(const reset_wire::ResetRequest& request) {
  reset_wire::ResetComplete complete;
  complete.runtime_instance_id = request.runtime_instance_id;
  complete.episode_id = request.episode_id;
  complete.reset_id = request.reset_id;
  complete.observation_ref.schema_version = reset_wire::kSchemaVersion;
  complete.observation_ref.runtime_instance_id = request.runtime_instance_id;
  complete.observation_ref.episode_id = request.episode_id;
  complete.observation_ref.reset_id = request.reset_id;
  complete.observation_ref.state_id = 100;
  complete.observation_ref.depth_id = "depth-" + request.reset_id;
  complete.observation_ref.sim_time_ns = 200;
  return complete;
}

void ResetLifecycleIsBounded() {
  planning::transport::CommandTransport transport;
  planning::bridge::ResetGateway gateway(
      transport, planning::bridge::ResetGatewayConfig{
                     "worker-00-runtime-test"});
  const std::vector<uint8_t> unity_identity{0x01, 0x02};
  const std::vector<uint8_t> python_identity{0x03, 0x04};
  gateway.OnUnityReady("worker-00-runtime-test", unity_identity);

  for (uint64_t index = 1; index <= 10000; ++index) {
    const reset_wire::ResetRequest request = Request(index);
    const std::vector<uint8_t> request_bytes =
        reset_wire::SerializeRequest(request);
    const reset_wire::ResetComplete complete = Complete(request);
    const std::vector<uint8_t> complete_bytes =
        reset_wire::SerializeComplete(complete);
    reset_wire::ResetReceivedAck ack;
    ack.runtime_instance_id = request.runtime_instance_id;
    ack.episode_id = request.episode_id;
    ack.reset_id = request.reset_id;
    const std::vector<uint8_t> ack_bytes = reset_wire::SerializeReceivedAck(ack);

    gateway.ReceivePythonReset(python_identity, request_bytes);
    Require(gateway.pending_count() == 1, "reset must remain pending before complete");
    gateway.ReceiveUnityResetComplete(unity_identity, complete_bytes);
    gateway.ReceivePythonResetAck(ack_bytes);
    Require(gateway.pending_count() == 0, "reset must be removed after ACK");
    Require(gateway.current_reset_entries() == 0,
            "active reset entries must be empty after ACK");

    // An exact retry is answered from the bounded terminal identity record;
    // it must not recreate an active reset or forward a second Unity reset.
    gateway.ReceivePythonReset(python_identity, request_bytes);
    Require(gateway.pending_count() == 0,
            "terminal reset retry must not recreate pending state");
    gateway.ReceiveUnityResetComplete(unity_identity, complete_bytes);
    gateway.ReceivePythonResetAck(ack_bytes);
  }
  Require(gateway.peak_reset_entries() <= 1,
          "reset active peak must be bounded");
  Require(gateway.terminal_reset_count() <= 1024,
          "reset tombstones must be bounded");
}

}  // namespace

int main() {
  try {
    ResetLifecycleIsBounded();
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
  std::cout << "C++ reset gateway lifecycle memory tests: 1 passed" << std::endl;
  return 0;
}
