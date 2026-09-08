#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "planning/primitive_execution_command_broker.hpp"

namespace broker = planning::primitive_execution_command_broker;
namespace xp = planning::xm_protocol;

namespace {

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

xp::v4::PrimitiveExecutionCommand Command() {
  xp::v4::PrimitiveExecutionCommand command;
  command.runtime_instance_id = "worker-00";
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
  bool Send(const broker::StoredCommand&) override { return true; }
};

class FakePythonAckSink : public broker::IPythonCommandAckSink {
 public:
  bool Send(const xp::v4::PrimitiveExecutionCommandReceiptAck&) override {
    return true;
  }
};

class RecordingAuditSink : public broker::ICommandAuditSink {
 public:
  std::vector<broker::CommandAuditRecord> records;

  void Record(const broker::CommandAuditRecord& record) override {
    records.push_back(record);
  }
};

void NormalCommandAccountingAndAudit() {
  broker::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  RecordingAuditSink audit;
  broker::PrimitiveExecutionCommandBroker command_broker(
      store, unity, python, "worker-00", &audit);
  const xp::v4::PrimitiveExecutionCommand command = Command();

  Require(command_broker.Receive(command) ==
              broker::CommandReceiveOutcome::ACCEPTED,
          "normal command must be accepted");
  Require(command_broker.accepted_count() == 1,
          "accepted transaction count");
  Require(command_broker.forward_count() == 1,
          "forward transaction count");
  Require(command_broker.pending_count() == 1,
          "pending command count before receipt");

  Require(command_broker.HandleUnityReceipt(
              broker::ReceiptFor(command, "ACCEPTED")) ==
              broker::CommandReceiptOutcome::RECEIVED,
          "matching Unity receipt");
  Require(command_broker.receipt_count() == 1,
          "Unity receipt transaction count");
  Require(command_broker.pending_count() == 0,
          "pending command count after receipt");
  Require(audit.records.size() == 5,
          "normal command must produce receipt lifecycle audit events");
  Require(audit.records[0].event == "COMMAND_ACCEPTED",
          "accepted audit event");
  Require(audit.records[1].event == "COMMAND_FORWARD",
          "forward audit event");
  Require(audit.records[2].event == "COMMAND_COMPLETE",
          "complete audit event");
  Require(audit.records[3].event == "COMMAND_RECEIPT_CREATED",
          "receipt created audit event");
  Require(audit.records[4].event == "COMMAND_RECEIPT_SEND_OK",
          "receipt send audit event");
  for (const broker::CommandAuditRecord& record : audit.records) {
    Require(record.runtime_instance_id == "worker-00",
            "audit runtime identity");
    Require(record.execution_id == command.execution_id,
            "audit execution identity");
    Require(record.command_sequence_hash == command.command_sequence_hash,
            "audit command hash");
    Require(record.monotonic_ns > 0, "audit monotonic timestamp");
  }
}

void DuplicateAndConflictAccountingAreDistinct() {
  broker::InMemoryCommandStore store;
  FakeUnityRelay unity;
  FakePythonAckSink python;
  RecordingAuditSink audit;
  broker::PrimitiveExecutionCommandBroker command_broker(
      store, unity, python, "worker-00", &audit);
  xp::v4::PrimitiveExecutionCommand command = Command();

  Require(command_broker.Receive(command) ==
              broker::CommandReceiveOutcome::ACCEPTED,
          "first command");
  Require(command_broker.Receive(command) ==
              broker::CommandReceiveOutcome::DUPLICATE,
          "same command duplicate");
  xp::v4::PrimitiveExecutionCommand conflict = command;
  conflict.frames[10].action[0] += 1.0f;
  conflict.command_sequence_hash = broker::CommandSequenceHash(conflict.frames);
  Require(command_broker.Receive(conflict) ==
              broker::CommandReceiveOutcome::PROTOCOL_ERROR,
          "conflicting command");
  Require(command_broker.duplicate_count() == 1,
          "duplicate command count");
  Require(command_broker.conflict_count() == 1,
          "conflict command count");
  Require(command_broker.pending_count() == 1,
          "conflict must not clear pending command");
  Require(audit.records.size() == 5,
          "duplicate/conflict audit events");
  Require(audit.records[2].event == "COMMAND_DUPLICATE",
          "duplicate audit event");
  Require(audit.records[4].event == "COMMAND_CONFLICT",
          "conflict audit event");
}

}  // namespace

int main() {
  NormalCommandAccountingAndAudit();
  DuplicateAndConflictAccountingAreDistinct();
  return 0;
}
