#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "planning/primitive_execution_result_broker.hpp"

namespace {

namespace xp = planning::xm_protocol;
namespace broker = planning::primitive_execution_result_broker;

int passed = 0;

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::array<uint8_t, 32> HexBytes(const std::string& hex) {
  Require(hex.size() == 64, "hex hash length");
  std::array<uint8_t, 32> value{{}};
  for (size_t index = 0; index < value.size(); ++index) {
    value[index] = static_cast<uint8_t>(std::stoul(
        hex.substr(index * 2, 2), nullptr, 16));
  }
  return value;
}

xp::v4::PrimitiveExecutionResult CompleteResult() {
  xp::v4::PrimitiveExecutionResult result;
  result.runtime_instance_id = "worker-00-runtime-test";
  result.execution_id = 72623859790382856ULL;
  result.status = "COMPLETE";
  result.applied_frame_count = 25;
  result.has_first_applied_state_id = true;
  result.first_applied_state_id = 4000;
  result.has_endpoint_state_id = true;
  result.endpoint_state_id = 4024;
  result.last_applied_frame_index = 24;
  result.reason_code = "NONE";
  result.command_sequence_hash = HexBytes(
      "82e7f39c8f9bb8b6ea5a42cee108cf1e33a3ee872ba47a51167c4ffd25bed62c");
  result.has_endpoint_sim_time_ns = true;
  result.endpoint_sim_time_ns = 5000000000ULL;
  result.has_endpoint_observation_ref = true;
  result.endpoint_observation_ref.schema_version = 4;
  result.endpoint_observation_ref.runtime_instance_id = result.runtime_instance_id;
  result.endpoint_observation_ref.episode_id = "episode-v4-0001";
  result.endpoint_observation_ref.reset_id = "reset-v4-0001";
  result.endpoint_observation_ref.state_id = 4024;
  result.endpoint_observation_ref.depth_id = "depth-v4-0001";
  result.endpoint_observation_ref.sim_time_ns = 5000000000ULL;
  result.result_payload_hash = HexBytes(
      "8527192b92ff82f9c702811d632ea672670ce81f99ec7cb5a5bec073c7376e09");
  return result;
}

class FakeAckSink : public broker::IResultAckSink {
 public:
  std::vector<xp::v4::PrimitiveExecutionResultAck> acks;

  void Send(const xp::v4::PrimitiveExecutionResultAck& ack) override {
    acks.push_back(ack);
  }
};

class FakeRelay : public broker::IPythonResultRelay {
 public:
  bool available = true;
  std::vector<broker::StoredResult> relayed;

  bool Relay(const broker::StoredResult& result) override {
    if (!available) return false;
    relayed.push_back(result);
    return true;
  }
};

void FirstCompleteIsDurablyStoredAndAcked() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);

  const broker::ReceiveOutcome outcome = result_broker.Receive(CompleteResult());

  Require(outcome == broker::ReceiveOutcome::ACCEPTED, "first result outcome");
  Require(store.pending_count() == 1, "first pending count");
  Require(result_broker.first_receipt_count() == 1, "first receipt count");
  Require(ack_sink.acks.size() == 1, "first ACK count");
  Require(ack_sink.acks[0].ack_status == "DURABLE_RECEIVED", "ACK status");
  Require(ack_sink.acks[0].schema_version == 4, "ACK schema version");
  Require(ack_sink.acks[0].message_type == "PrimitiveExecutionResultAck",
          "ACK message type");
  Require(ack_sink.acks[0].execution_id == 72623859790382856ULL,
          "ACK execution identity");
  Require(ack_sink.acks[0].result_payload_hash ==
              CompleteResult().result_payload_hash,
          "ACK result identity");
  Require(ack_sink.acks[0].command_sequence_hash ==
              CompleteResult().command_sequence_hash,
          "ACK command identity");
  Require(relay.relayed.size() == 1, "first Python relay count");
  ++passed;
}

void DuplicateCompleteIsAckedWithoutSecondRelay() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  const xp::v4::PrimitiveExecutionResult result = CompleteResult();

  Require(result_broker.Receive(result) == broker::ReceiveOutcome::ACCEPTED,
          "duplicate first outcome");
  Require(result_broker.Receive(result) == broker::ReceiveOutcome::DUPLICATE,
          "duplicate second outcome");
  Require(store.pending_count() == 1, "duplicate pending count");
  Require(result_broker.first_receipt_count() == 1,
          "duplicate first receipt count");
  Require(result_broker.duplicate_count() == 1, "duplicate count");
  Require(ack_sink.acks.size() == 2, "duplicate ACK count");
  Require(relay.relayed.size() == 1, "duplicate Python side effect");
  ++passed;
}

void ConflictingDuplicateIsProtocolError() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  const xp::v4::PrimitiveExecutionResult first = CompleteResult();
  xp::v4::PrimitiveExecutionResult conflict = first;
  conflict.command_sequence_hash[0] ^= 0x01;
  conflict.result_payload_hash = broker::ResultPayloadHash(conflict);

  Require(result_broker.Receive(first) == broker::ReceiveOutcome::ACCEPTED,
          "conflict first outcome");
  Require(result_broker.Receive(conflict) ==
              broker::ReceiveOutcome::PROTOCOL_ERROR,
          "conflict outcome");
  Require(store.pending_count() == 1, "conflict pending count");
  Require(result_broker.conflict_count() == 1, "conflict count");
  Require(result_broker.protocol_error_count() == 1,
          "conflict protocol error count");
  Require(ack_sink.acks.size() == 1, "conflict must not be ACKed");
  Require(relay.relayed.size() == 1, "conflict relay count");
  ++passed;
}

void RuntimeIdentitySeparatesWorkers() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  const xp::v4::PrimitiveExecutionResult worker_zero = CompleteResult();
  xp::v4::PrimitiveExecutionResult worker_one = worker_zero;
  worker_one.runtime_instance_id = "worker-01-runtime-test";
  worker_one.endpoint_observation_ref.runtime_instance_id = worker_one.runtime_instance_id;
  worker_one.result_payload_hash = broker::ResultPayloadHash(worker_one);

  Require(result_broker.Receive(worker_zero) == broker::ReceiveOutcome::ACCEPTED,
          "worker zero outcome");
  Require(result_broker.Receive(worker_one) == broker::ReceiveOutcome::ACCEPTED,
          "worker one outcome");
  Require(store.pending_count() == 2, "worker isolation pending count");
  Require(result_broker.first_receipt_count() == 2,
          "worker isolation receipt count");
  Require(result_broker.conflict_count() == 0,
          "worker isolation conflict count");
  Require(relay.relayed.size() == 2, "worker isolation relay count");
  ++passed;
}

void AckLossAndUnityResendIsDuplicate() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  const xp::v4::PrimitiveExecutionResult result = CompleteResult();

  Require(result_broker.Receive(result) == broker::ReceiveOutcome::ACCEPTED,
          "ACK loss first outcome");
  ack_sink.acks.clear();
  Require(result_broker.Receive(result) == broker::ReceiveOutcome::DUPLICATE,
          "ACK loss resend outcome");
  Require(store.pending_count() == 1, "ACK loss pending count");
  Require(ack_sink.acks.size() == 1, "ACK loss re-ACK count");
  Require(result_broker.first_receipt_count() == 1,
          "ACK loss first receipt count");
  Require(result_broker.duplicate_count() == 1, "ACK loss duplicate count");
  Require(relay.relayed.size() == 1, "ACK loss relay count");
  ++passed;
}

void PythonRelayFailurePreservesPendingAndCanRetry() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  relay.available = false;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  const xp::v4::PrimitiveExecutionResult result = CompleteResult();
  const broker::ResultKey key = broker::KeyFor(result);

  Require(result_broker.Receive(result) == broker::ReceiveOutcome::ACCEPTED,
          "relay failure receipt outcome");
  Require(store.pending_count() == 1, "relay failure pending count");
  Require(result_broker.python_relay_failure_count() == 1,
          "relay failure count");
  relay.available = true;
  Require(result_broker.RetryPythonRelay(key), "relay retry outcome");
  Require(store.pending_count() == 1, "relay retry pending count");
  Require(relay.relayed.size() == 1, "relay retry count");
  ++passed;
}

void PythonCommitIsExactlyOnce() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  const xp::v4::PrimitiveExecutionResult result = CompleteResult();
  const broker::ResultKey key = broker::KeyFor(result);

  Require(result_broker.Receive(result) == broker::ReceiveOutcome::ACCEPTED,
          "commit receipt outcome");
  broker::ResultKey wrong_key = key;
  ++wrong_key.execution_id;
  Require(result_broker.CommitPython(wrong_key, result.result_payload_hash) ==
              broker::CommitOutcome::NOT_FOUND,
          "wrong execution id is rejected");
  Require(result_broker.python_commit_count() == 0,
          "wrong execution id must not increment commit count");
  std::array<uint8_t, 32> wrong_result_hash = result.result_payload_hash;
  wrong_result_hash[0] ^= 0x01;
  Require(result_broker.CommitPython(key, wrong_result_hash) ==
              broker::CommitOutcome::PROTOCOL_ERROR,
          "wrong result hash is rejected");
  Require(result_broker.python_commit_count() == 0,
          "wrong result hash must not increment commit count");
  std::array<uint8_t, 32> wrong_command_hash = result.command_sequence_hash;
  wrong_command_hash[0] ^= 0x01;
  Require(result_broker.CommitPython(key, result.result_payload_hash,
                                     &wrong_command_hash) ==
              broker::CommitOutcome::PROTOCOL_ERROR,
          "wrong command hash is rejected");
  Require(result_broker.python_commit_count() == 0,
          "wrong command hash must not increment commit count");
  Require(result_broker.CommitPython(key, result.result_payload_hash) ==
              broker::CommitOutcome::COMMITTED,
          "first Python commit");
  Require(store.pending_count() == 0, "commit pending count");
  Require(store.committed_count() == 1, "commit durable count");
  Require(result_broker.python_commit_count() == 1, "commit count");
  Require(result_broker.transition_side_effect_count() == 1,
          "first transition side effect");
  Require(result_broker.CommitPython(key, result.result_payload_hash) ==
              broker::CommitOutcome::DUPLICATE,
          "duplicate Python commit");
  Require(result_broker.duplicate_commit_count() == 1,
          "duplicate commit count");
  Require(result_broker.transition_side_effect_count() == 1,
          "duplicate transition side effect");
  ++passed;
}

void InvalidPayloadHashIsRejectedBeforeStorage() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  xp::v4::PrimitiveExecutionResult invalid = CompleteResult();
  invalid.result_payload_hash[0] ^= 0x01;

  Require(result_broker.Receive(invalid) == broker::ReceiveOutcome::PROTOCOL_ERROR,
          "invalid hash outcome");
  Require(store.pending_count() == 0, "invalid hash pending count");
  Require(ack_sink.acks.empty(), "invalid hash ACK count");
  Require(relay.relayed.empty(), "invalid hash relay count");
  Require(result_broker.protocol_error_count() == 1,
          "invalid hash protocol error count");
  ++passed;
}

}  // namespace

int main() {
  try {
    FirstCompleteIsDurablyStoredAndAcked();
    DuplicateCompleteIsAckedWithoutSecondRelay();
    ConflictingDuplicateIsProtocolError();
    RuntimeIdentitySeparatesWorkers();
    AckLossAndUnityResendIsDuplicate();
    PythonRelayFailurePreservesPendingAndCanRetry();
    PythonCommitIsExactlyOnce();
    InvalidPayloadHashIsRejectedBeforeStorage();
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
  std::cout << "C++ result broker tests: " << passed << " passed" << std::endl;
  return 0;
}
