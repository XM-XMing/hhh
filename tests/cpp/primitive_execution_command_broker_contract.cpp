#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "planning/primitive_execution_command_broker.hpp"
#include "planning/primitive_execution_command_wire.hpp"

namespace broker = planning::primitive_execution_command_broker;
namespace command_wire = planning::primitive_execution_command_wire;
namespace xp = planning::xm_protocol;

namespace {

int passed = 0;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::array<uint8_t, 32> HashByte(uint8_t value) {
  std::array<uint8_t, 32> result{{}};
  result[0] = value;
  return result;
}

xp::v4::PrimitiveExecutionCommand Command(const std::string& runtime = "worker-00") {
  xp::v4::PrimitiveExecutionCommand command;
  command.runtime_instance_id = runtime;
  command.execution_id = 44000000000001ULL;
  for (uint32_t index = 0; index < 25; ++index) {
    xp::v4::PrimitiveExecutionFrame frame;
    frame.frame_index = index;
    frame.command_id = 1000 + index;
    frame.action = {0.1f, 0.2f, 0.3f, 0.4f};
    command.frames.push_back(frame);
  }
  command.command_sequence_hash = broker::CommandSequenceHash(command.frames);
  return command;
}

class FakeUnityRelay : public broker::IUnityCommandRelay {
 public:
  bool available = true;
  std::vector<broker::StoredCommand> sent;

  bool Send(const broker::StoredCommand& command) override {
    if (!available) return false;
    sent.push_back(command);
    return true;
  }
};

class FakePythonAckSink : public broker::IPythonCommandAckSink {
 public:
  bool available = true;
  std::vector<xp::v4::PrimitiveExecutionCommandReceiptAck> acks;

  bool Send(const xp::v4::PrimitiveExecutionCommandReceiptAck& ack) override {
    if (!available) return false;
    acks.push_back(ack);
    return true;
  }
};

void NormalCommandWaitsForUnityReceipt() {
  broker::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  broker::PrimitiveExecutionCommandBroker command_broker(
      store, unity, python, "worker-00");
  const xp::v4::PrimitiveExecutionCommand command = Command();

  Require(command_broker.Receive(command) == broker::CommandReceiveOutcome::ACCEPTED,
          "normal command must be accepted");
  Require(unity.sent.size() == 1, "normal command must be forwarded once");
  Require(command_broker.pending_count() == 1,
          "command remains pending until Unity receipt");
  Require(python.acks.empty(), "Python ACK must wait for Unity receipt");
  ++passed;
}

void LostUnityReceiptRetriesImmutableCommand() {
  broker::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  broker::PrimitiveExecutionCommandBroker command_broker(
      store, unity, python, "worker-00");
  const xp::v4::PrimitiveExecutionCommand command = Command();
  const broker::CommandKey key = broker::KeyFor(command);

  Require(command_broker.Receive(command) == broker::CommandReceiveOutcome::ACCEPTED,
          "retry first accept");
  const std::vector<uint8_t> original = unity.sent.front().canonical_payload;
  for (int retry = 0; retry < 100; ++retry) {
    Require(command_broker.RetryPending(key), "retry must be eligible");
  }
  Require(unity.sent.size() == 101, "100 retransmissions expected");
  for (const broker::StoredCommand& sent : unity.sent) {
    Require(sent.canonical_payload == original,
            "retransmission payload must be byte-identical");
  }
  Require(command_broker.physical_execution_count() == 0,
          "bridge retry cannot execute the primitive");
  ++passed;
}

void UnityDisconnectReconnectKeepsPendingCommand() {
  broker::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  broker::PrimitiveExecutionCommandBroker command_broker(
      store, unity, python, "worker-00");
  const xp::v4::PrimitiveExecutionCommand command = Command();
  const broker::CommandKey key = broker::KeyFor(command);

  unity.available = false;
  Require(command_broker.Receive(command) == broker::CommandReceiveOutcome::ACCEPTED,
          "disconnected Unity command remains accepted durably");
  Require(unity.sent.empty(), "unavailable Unity receives nothing");
  Require(command_broker.pending_count() == 1,
          "disconnected Unity command remains pending");

  unity.available = true;
  Require(command_broker.RetryPending(key),
          "reconnected Unity makes the command retryable");
  Require(unity.sent.size() == 1,
          "reconnected Unity receives the pending command");
  Require(unity.sent[0].canonical_payload ==
              xp::v4::canonical_command_payload_bytes(command),
          "reconnected Unity receives immutable bytes");
  Require(command_broker.HandleUnityReceipt(
              broker::ReceiptFor(command, "ACCEPTED")) ==
              broker::CommandReceiptOutcome::RECEIVED,
          "reconnected Unity receipt completes admission");
  Require(command_broker.pending_count() == 0,
          "reconnected Unity clears pending command");
  ++passed;
}

void UnityReceiptClearsPendingAndAcksPython() {
  broker::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  broker::PrimitiveExecutionCommandBroker command_broker(
      store, unity, python, "worker-00");
  const xp::v4::PrimitiveExecutionCommand command = Command();
  Require(command_broker.Receive(command) == broker::CommandReceiveOutcome::ACCEPTED,
          "receipt first accept");

  xp::v4::PrimitiveExecutionCommandReceiptAck receipt =
      broker::ReceiptFor(command, "ACCEPTED");
  Require(command_broker.HandleUnityReceipt(receipt) ==
              broker::CommandReceiptOutcome::RECEIVED,
          "matching Unity receipt");
  Require(command_broker.pending_count() == 0, "receipt clears command pending");
  Require(python.acks.size() == 1, "Python receives one command receipt");
  Require(python.acks[0].ack_status == "RECEIVED",
          "Python ACK means Unity receipt was proven");
  Require(command_broker.HandleUnityReceipt(receipt) ==
              broker::CommandReceiptOutcome::DUPLICATE,
          "duplicate Unity receipt is idempotent");
  Require(command_broker.physical_execution_count() == 0,
          "receipt handling never executes a primitive");
  ++passed;
}

void DuplicateCommandDoesNotCreateSecondPhysicalExecution() {
  broker::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  broker::PrimitiveExecutionCommandBroker command_broker(
      store, unity, python, "worker-00");
  const xp::v4::PrimitiveExecutionCommand command = Command();
  Require(command_broker.Receive(command) == broker::CommandReceiveOutcome::ACCEPTED,
          "duplicate first accept");
  Require(command_broker.Receive(command) == broker::CommandReceiveOutcome::DUPLICATE,
          "duplicate command outcome");
  Require(store.pending_count() == 1, "duplicate does not add pending entry");
  Require(unity.sent.size() == 2, "duplicate may resend immutable command");
  Require(unity.sent[0].canonical_payload == unity.sent[1].canonical_payload,
          "duplicate resend payload identity");
  Require(command_broker.physical_execution_count() == 0,
          "bridge duplicate does not execute");
  ++passed;
}

void DurableReceiptIsResentToReconnectedPython() {
  broker::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  broker::PrimitiveExecutionCommandBroker command_broker(
      store, unity, python, "worker-00");
  const xp::v4::PrimitiveExecutionCommand command = Command();

  Require(command_broker.Receive(command) == broker::CommandReceiveOutcome::ACCEPTED,
          "durable receipt first accept");
  Require(command_broker.HandleUnityReceipt(
              broker::ReceiptFor(command, "ACCEPTED")) ==
              broker::CommandReceiptOutcome::RECEIVED,
          "durable receipt first delivery");
  python.acks.clear();

  Require(command_broker.Receive(command) == broker::CommandReceiveOutcome::DUPLICATE,
          "reconnected Python duplicate outcome");
  Require(python.acks.size() == 1,
          "reconnected Python receives the durable receipt again");
  Require(python.acks[0].ack_status == "RECEIVED",
          "reconnected Python receipt status");
  Require(command_broker.pending_count() == 0,
          "durable receipt duplicate remains complete");
  ++passed;
}

void ConflictingCommandIsProtocolError() {
  broker::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  broker::PrimitiveExecutionCommandBroker command_broker(
      store, unity, python, "worker-00");
  xp::v4::PrimitiveExecutionCommand first = Command();
  xp::v4::PrimitiveExecutionCommand conflict = first;
  conflict.frames[10].action[0] += 1.0f;
  conflict.command_sequence_hash = broker::CommandSequenceHash(conflict.frames);
  Require(command_broker.Receive(first) == broker::CommandReceiveOutcome::ACCEPTED,
          "conflict first accept");
  Require(command_broker.Receive(conflict) ==
              broker::CommandReceiveOutcome::PROTOCOL_ERROR,
          "conflicting command must fail closed");
  Require(unity.sent.size() == 1, "conflict is not forwarded");
  Require(command_broker.protocol_error_count() == 1,
          "conflict protocol error count");
  ++passed;
}

void WrongRuntimeIsRejected() {
  broker::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  broker::PrimitiveExecutionCommandBroker command_broker(
      store, unity, python, "worker-00");
  Require(command_broker.Receive(Command("worker-01")) ==
              broker::CommandReceiveOutcome::REJECTED,
          "wrong runtime must be rejected");
  Require(unity.sent.empty(), "wrong runtime must not be forwarded");
  Require(command_broker.pending_count() == 0, "wrong runtime pending count");
  ++passed;
}

void PythonAckLossKeepsReceiptPendingUntilReconnect() {
  broker::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  broker::PrimitiveExecutionCommandBroker command_broker(
      store, unity, python, "worker-00");
  const xp::v4::PrimitiveExecutionCommand command = Command();
  Require(command_broker.Receive(command) == broker::CommandReceiveOutcome::ACCEPTED,
          "python reconnect first accept");
  python.available = false;
  Require(command_broker.HandleUnityReceipt(broker::ReceiptFor(command, "ACCEPTED")) ==
              broker::CommandReceiptOutcome::RECEIVED,
          "Unity receipt is durable even when Python is disconnected");
  Require(command_broker.python_ack_pending_count() == 1,
          "Python ACK remains pending");
  python.available = true;
  Require(command_broker.RetryPythonAcks(), "Python ACK retry after reconnect");
  Require(command_broker.python_ack_pending_count() == 0,
          "Python ACK pending clears after send");
  Require(python.acks.size() == 1, "one Python ACK after reconnect");
  ++passed;
}

void CommandWireRoundTripPreservesCanonicalIdentity() {
  const xp::v4::PrimitiveExecutionCommand command = Command();
  const std::vector<uint8_t> wire = command_wire::SerializeCommand(command);
  xp::v4::PrimitiveExecutionCommand parsed;
  std::string error;
  Require(command_wire::ParseCommand(
              reinterpret_cast<const char*>(wire.data()), wire.size(), &parsed,
              &error),
          "command wire parse");
  Require(parsed.runtime_instance_id == command.runtime_instance_id,
          "command wire runtime identity");
  Require(parsed.execution_id == command.execution_id,
          "command wire execution identity");
  Require(parsed.command_sequence_hash == command.command_sequence_hash,
          "command wire hash identity");
  Require(command_wire::SerializeCommand(parsed) == wire,
          "command wire canonical bytes");
  ++passed;
}

}  // namespace

int main() {
  NormalCommandWaitsForUnityReceipt();
  LostUnityReceiptRetriesImmutableCommand();
  UnityDisconnectReconnectKeepsPendingCommand();
  UnityReceiptClearsPendingAndAcksPython();
  DuplicateCommandDoesNotCreateSecondPhysicalExecution();
  DurableReceiptIsResentToReconnectedPython();
  ConflictingCommandIsProtocolError();
  WrongRuntimeIsRejected();
  PythonAckLossKeepsReceiptPendingUntilReconnect();
  CommandWireRoundTripPreservesCanonicalIdentity();
  return passed == 10 ? 0 : 1;
}
