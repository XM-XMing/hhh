#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "planning/bridge/observation_retrieval_broker.hpp"
#include "planning/bridge/primitive_execution_command_broker.hpp"
#include "planning/bridge/primitive_execution_result_broker.hpp"

namespace {

namespace xp = planning::xm_protocol;
namespace observation = planning::observation_retrieval_broker;
namespace command = planning::primitive_execution_command_broker;
namespace result = planning::primitive_execution_result_broker;

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::array<uint8_t, 32> Hash(uint8_t value) {
  std::array<uint8_t, 32> hash{{}};
  hash.fill(value);
  return hash;
}

xp::v4::PrimitiveExecutionResult Result(uint64_t execution_id) {
  xp::v4::PrimitiveExecutionResult value;
  value.runtime_instance_id = "worker-00-runtime-test";
  value.execution_id = execution_id;
  value.status = "COMPLETE";
  value.applied_frame_count = 25;
  value.has_first_applied_state_id = true;
  value.first_applied_state_id = static_cast<int64_t>(execution_id * 25);
  value.has_endpoint_state_id = true;
  value.endpoint_state_id = static_cast<int64_t>(execution_id * 25 + 24);
  value.last_applied_frame_index = 24;
  value.reason_code = "NONE";
  value.command_sequence_hash = Hash(static_cast<uint8_t>(execution_id));
  value.has_endpoint_sim_time_ns = true;
  value.endpoint_sim_time_ns = execution_id * 1000;
  value.has_endpoint_observation_ref = true;
  value.endpoint_observation_ref.schema_version = 4;
  value.endpoint_observation_ref.runtime_instance_id = value.runtime_instance_id;
  value.endpoint_observation_ref.episode_id = "episode-" + std::to_string(execution_id);
  value.endpoint_observation_ref.reset_id = "reset-" + std::to_string(execution_id);
  value.endpoint_observation_ref.state_id = value.endpoint_state_id;
  value.endpoint_observation_ref.depth_id = "depth-" + std::to_string(execution_id);
  value.endpoint_observation_ref.sim_time_ns = value.endpoint_sim_time_ns;
  value.result_payload_hash = observation::ResultPayloadHash(value);
  return value;
}

xp::v4::EndpointObservationSnapshot Snapshot(
    const xp::v4::SnapshotRequest& request) {
  xp::v4::EndpointObservationSnapshot value;
  value.observation_ref = request.observation_ref;
  value.state_bytes.assign(1024, 0x01);
  value.depth_bytes.assign(4096, 0x02);
  value.snapshot_hash = observation::SnapshotHash(value);
  return value;
}

class FakeSnapshotRequester : public observation::IUnitySnapshotRequester {
 public:
  std::vector<xp::v4::SnapshotRequest> requests;

  bool Request(const xp::v4::SnapshotRequest& request) override {
    requests.push_back(request);
    return true;
  }
};

class FakeSnapshotAckSink : public observation::ISnapshotAckSink {
 public:
  size_t count = 0;

  void Send(const xp::v4::SnapshotAck&) override { ++count; }
};

class FakeResultAckSink : public result::IResultAckSink {
 public:
  void Send(const xp::v4::PrimitiveExecutionResultAck&) override {}
};

class FakeResultRelay : public result::IPythonResultRelay {
 public:
  bool Relay(const result::StoredResult&) override { return true; }
};

class FakeUnityRelay : public command::IUnityCommandRelay {
 public:
  bool Send(const command::StoredCommand&) override { return true; }
};

class FakePythonAckSink : public command::IPythonCommandAckSink {
 public:
  bool Send(const xp::v4::PrimitiveExecutionCommandReceiptAck&) override {
    return true;
  }
};

xp::v4::PrimitiveExecutionCommand Command(uint64_t execution_id) {
  xp::v4::PrimitiveExecutionCommand value;
  value.runtime_instance_id = "worker-00-runtime-test";
  value.execution_id = execution_id;
  for (uint32_t index = 0; index < 25; ++index) {
    xp::v4::PrimitiveExecutionFrame frame;
    frame.frame_index = index;
    frame.command_id = execution_id * 25 + index;
    frame.action = {0.1f, 0.2f, 0.3f, 0.4f};
    value.frames.push_back(frame);
  }
  value.command_sequence_hash = command::CommandSequenceHash(value.frames);
  return value;
}

void SnapshotLifecycleIsBounded() {
  observation::InMemorySnapshotCache cache;
  FakeSnapshotRequester requester;
  FakeSnapshotAckSink ack_sink;
  observation::ObservationRetrievalBroker broker(cache, requester, ack_sink, 10);

  for (uint64_t index = 1; index <= 10000; ++index) {
    const xp::v4::PrimitiveExecutionResult terminal = Result(index);
    Require(broker.ReceiveComplete(terminal, index) == observation::RequestOutcome::REQUESTED,
            "snapshot request must be accepted");
    const xp::v4::SnapshotRequest request = requester.requests.back();
    const xp::v4::EndpointObservationSnapshot snapshot = Snapshot(request);
    Require(broker.ReceiveSnapshot(request, snapshot) == observation::SnapshotOutcome::STORED,
            "snapshot must be stored");
    Require(broker.FinalizeRequest(request), "snapshot terminal cleanup");
    Require(cache.size() == 0, "active snapshot cache must be empty");
  }
  Require(broker.peak_snapshot_entries() <= 1, "snapshot peak must be bounded");
  Require(broker.terminal_snapshot_count() <= observation::ObservationRetrievalBroker::kTerminalTombstoneCapacity,
          "snapshot tombstones must be bounded");
}

void ResultLifecycleIsBounded() {
  result::InMemoryResultStore store;
  FakeResultAckSink ack_sink;
  FakeResultRelay relay;
  result::PrimitiveExecutionResultBroker broker(store, ack_sink, relay);
  for (uint64_t index = 1; index <= 10000; ++index) {
    const xp::v4::PrimitiveExecutionResult terminal = Result(index);
    const result::ResultKey key = result::KeyFor(terminal);
    Require(broker.Receive(terminal) == result::ReceiveOutcome::ACCEPTED,
            "result must be accepted");
    xp::v4::PrimitiveExecutionResultReceiptAck receipt;
    receipt.schema_version = xp::v4::kSchemaVersion;
    receipt.message_type = "PrimitiveExecutionResultReceiptAck";
    receipt.receipt_status = "RECEIVED";
    receipt.runtime_instance_id = terminal.runtime_instance_id;
    receipt.execution_id = terminal.execution_id;
    receipt.result_payload_hash = terminal.result_payload_hash;
    receipt.command_sequence_hash = terminal.command_sequence_hash;
    Require(broker.AcknowledgePythonReceipt(receipt) == result::ReceiptOutcome::RECEIVED,
            "result receipt must be accepted");
    Require(broker.CommitPython(key, terminal.result_payload_hash,
                                &terminal.command_sequence_hash) ==
                result::CommitOutcome::COMMITTED,
            "result commit must be accepted");
    Require(store.size() == 0, "active result store must be empty");
  }
  Require(store.peak_size() <= 1, "result peak must be bounded");
}

void CommandLifecycleIsBounded() {
  command::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  command::PrimitiveExecutionCommandBroker broker(
      store, unity, python, "worker-00-runtime-test");
  for (uint64_t index = 1; index <= 10000; ++index) {
    const xp::v4::PrimitiveExecutionCommand submitted = Command(index);
    const command::CommandKey key = command::KeyFor(submitted);
    Require(broker.Receive(submitted) == command::CommandReceiveOutcome::ACCEPTED,
            "command must be accepted");
    Require(broker.HandleUnityReceipt(command::ReceiptFor(submitted, "ACCEPTED")) ==
                command::CommandReceiptOutcome::RECEIVED,
            "command receipt must be accepted");
    Require(broker.FinalizeExecution(key, submitted.command_sequence_hash),
            "command terminal cleanup");
    Require(store.size() == 0, "active command store must be empty");
  }
  Require(store.peak_size() <= 1, "command peak must be bounded");
}

}  // namespace

int main() {
  try {
    SnapshotLifecycleIsBounded();
    ResultLifecycleIsBounded();
    CommandLifecycleIsBounded();
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
  std::cout << "C++ bridge lifecycle memory tests: 3 passed" << std::endl;
  return 0;
}
