#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "planning/primitive_execution_result_broker.hpp"
#include "planning/primitive_execution_result_wire.hpp"

namespace {

namespace xp = planning::xm_protocol;
namespace broker = planning::primitive_execution_result_broker;

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::array<uint8_t, 32> Hash(const std::string& hex) {
  Require(hex.size() == 64, "hash length");
  std::array<uint8_t, 32> value{{}};
  for (size_t index = 0; index < value.size(); ++index) {
    value[index] = static_cast<uint8_t>(std::stoul(
        hex.substr(index * 2, 2), nullptr, 16));
  }
  return value;
}

xp::v4::PrimitiveExecutionResult CompleteResult() {
  xp::v4::PrimitiveExecutionResult result;
  result.runtime_instance_id = "worker-00-receipt-test";
  result.execution_id = 9001;
  result.status = "COMPLETE";
  result.applied_frame_count = 25;
  result.has_first_applied_state_id = true;
  result.first_applied_state_id = 100;
  result.has_endpoint_state_id = true;
  result.endpoint_state_id = 124;
  result.last_applied_frame_index = 24;
  result.reason_code = "NONE";
  result.command_sequence_hash = Hash(
      "82e7f39c8f9bb8b6ea5a42cee108cf1e33a3ee872ba47a51167c4ffd25bed62c");
  result.has_endpoint_sim_time_ns = true;
  result.endpoint_sim_time_ns = 5000000000ULL;
  result.has_endpoint_observation_ref = true;
  result.endpoint_observation_ref.schema_version = 4;
  result.endpoint_observation_ref.runtime_instance_id = result.runtime_instance_id;
  result.endpoint_observation_ref.episode_id = "episode-receipt-0001";
  result.endpoint_observation_ref.reset_id = "reset-receipt-0001";
  result.endpoint_observation_ref.state_id = 124;
  result.endpoint_observation_ref.depth_id = "depth-receipt-0001";
  result.endpoint_observation_ref.sim_time_ns = 5000000000ULL;
  result.result_payload_hash = broker::ResultPayloadHash(result);
  return result;
}

class FakeAckSink : public broker::IResultAckSink {
 public:
  void Send(const xp::v4::PrimitiveExecutionResultAck&) override {}
};

class FakeRelay : public broker::IPythonResultRelay {
 public:
  std::vector<broker::StoredResult> relayed;

  bool Relay(const broker::StoredResult& result) override {
    relayed.push_back(result);
    return true;
  }
};

xp::v4::PrimitiveExecutionResultReceiptAck ReceiptAck(
    const xp::v4::PrimitiveExecutionResult& result) {
  xp::v4::PrimitiveExecutionResultReceiptAck ack;
  ack.runtime_instance_id = result.runtime_instance_id;
  ack.execution_id = result.execution_id;
  ack.receipt_status = "RECEIVED";
  ack.result_payload_hash = result.result_payload_hash;
  ack.command_sequence_hash = result.command_sequence_hash;
  return ack;
}

void SendSuccessWithoutReceiptAckRemainsPending() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  const xp::v4::PrimitiveExecutionResult result = CompleteResult();

  Require(result_broker.Receive(result) == broker::ReceiveOutcome::ACCEPTED,
          "first result must be accepted");
  Require(result_broker.receipt_pending_count() == 1,
          "send success without receipt ACK must remain pending");
  Require(relay.relayed.size() == 1, "initial relay count");

  Require(result_broker.RetryPendingPythonRelays() == 1,
          "pending result must be eligible for retransmission");
  Require(relay.relayed.size() == 2, "retransmission count");
  Require(relay.relayed[0].canonical_payload == relay.relayed[1].canonical_payload,
          "retransmission must reuse immutable payload");
}

void ReceiptAckClearsPendingAndStopsRetransmission() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  const xp::v4::PrimitiveExecutionResult result = CompleteResult();

  Require(result_broker.Receive(result) == broker::ReceiveOutcome::ACCEPTED,
          "receipt ACK first result");
  Require(result_broker.AcknowledgePythonReceipt(ReceiptAck(result)) ==
              broker::ReceiptOutcome::RECEIVED,
          "matching receipt ACK outcome");
  Require(result_broker.receipt_pending_count() == 0,
          "matching receipt ACK clears pending");
  Require(result_broker.RetryPendingPythonRelays() == 0,
          "receipt acknowledged result is not retransmitted");
  Require(relay.relayed.size() == 1, "receipt acknowledged relay count");
}

void ReceiptAckWireRoundTripPreservesIdentity() {
  const xp::v4::PrimitiveExecutionResult result = CompleteResult();
  const xp::v4::PrimitiveExecutionResultReceiptAck expected = ReceiptAck(result);
  const std::vector<uint8_t> wire =
      planning::primitive_execution_result_wire::SerializeReceiptAck(expected);
  xp::v4::PrimitiveExecutionResultReceiptAck actual;
  std::string error;
  Require(
      planning::primitive_execution_result_wire::ParseReceiptAck(
          reinterpret_cast<const char*>(wire.data()), wire.size(), &actual, &error),
      "receipt ACK wire parse: " + error);
  Require(actual.schema_version == expected.schema_version, "receipt schema");
  Require(actual.message_type == expected.message_type, "receipt message type");
  Require(actual.runtime_instance_id == expected.runtime_instance_id,
          "receipt runtime identity");
  Require(actual.execution_id == expected.execution_id, "receipt execution identity");
  Require(actual.receipt_status == expected.receipt_status, "receipt status");
  Require(actual.result_payload_hash == expected.result_payload_hash,
          "receipt result identity");
  Require(actual.command_sequence_hash == expected.command_sequence_hash,
          "receipt command identity");
}

void DuplicateReceiptAckIsIdempotent() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  const xp::v4::PrimitiveExecutionResult result = CompleteResult();
  const xp::v4::PrimitiveExecutionResultReceiptAck ack = ReceiptAck(result);

  result_broker.Receive(result);
  Require(result_broker.AcknowledgePythonReceipt(ack) ==
              broker::ReceiptOutcome::RECEIVED,
          "duplicate receipt first outcome");
  Require(result_broker.AcknowledgePythonReceipt(ack) ==
              broker::ReceiptOutcome::DUPLICATE,
          "duplicate receipt second outcome");
  Require(result_broker.receipt_count() == 1, "duplicate receipt count");
  Require(result_broker.duplicate_receipt_count() == 1,
          "duplicate receipt duplicate count");
  Require(result_broker.receipt_pending_count() == 0,
          "duplicate receipt pending count");
}

void ConflictingReceiptAckIsProtocolErrorAndKeepsPending() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  const xp::v4::PrimitiveExecutionResult result = CompleteResult();
  xp::v4::PrimitiveExecutionResultReceiptAck conflict = ReceiptAck(result);
  conflict.result_payload_hash[0] ^= 0x01;

  result_broker.Receive(result);
  Require(result_broker.AcknowledgePythonReceipt(conflict) ==
              broker::ReceiptOutcome::PROTOCOL_ERROR,
          "conflicting receipt outcome");
  Require(result_broker.receipt_pending_count() == 1,
          "conflicting receipt keeps pending");
  Require(result_broker.protocol_error_count() == 1,
          "conflicting receipt protocol error count");
}

void ReconnectRetryUsesSamePayloadUntilReceipt() {
  broker::InMemoryResultStore store;
  FakeAckSink ack_sink;
  FakeRelay relay;
  broker::PrimitiveExecutionResultBroker result_broker(store, ack_sink, relay);
  const xp::v4::PrimitiveExecutionResult result = CompleteResult();
  const broker::ResultKey key = broker::KeyFor(result);

  result_broker.Receive(result);
  Require(result_broker.RetryPythonRelay(key), "reconnect retry outcome");
  Require(relay.relayed.size() == 2, "reconnect relay count");
  Require(relay.relayed[0].canonical_payload == relay.relayed[1].canonical_payload,
          "reconnect immutable payload");
  Require(result_broker.AcknowledgePythonReceipt(ReceiptAck(result)) ==
              broker::ReceiptOutcome::RECEIVED,
          "reconnect receipt outcome");
  Require(result_broker.RetryPendingPythonRelays() == 0,
          "reconnect does not retry after receipt");
}

}  // namespace

int main() {
  try {
    SendSuccessWithoutReceiptAckRemainsPending();
    ReceiptAckClearsPendingAndStopsRetransmission();
    ReceiptAckWireRoundTripPreservesIdentity();
    DuplicateReceiptAckIsIdempotent();
    ConflictingReceiptAckIsProtocolErrorAndKeepsPending();
    ReconnectRetryUsesSamePayloadUntilReceipt();
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
  std::cout << "C++ result receipt tests: 6 passed" << std::endl;
  return 0;
}
