#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <openssl/sha.h>
#include <string>
#include <vector>

#include "planning/protocol/xm_protocol.hpp"

namespace planning {
namespace primitive_execution_command_broker {

namespace xp = planning::xm_protocol;

enum class CommandReceiveOutcome {
  ACCEPTED,
  DUPLICATE,
  PROTOCOL_ERROR,
  REJECTED,
};

enum class CommandReceiptOutcome {
  RECEIVED,
  DUPLICATE,
  PROTOCOL_ERROR,
  NOT_FOUND,
};

enum class CommandStorePutOutcome { STORED, DUPLICATE, CONFLICT };

struct CommandKey {
  std::string runtime_instance_id;
  uint64_t execution_id = 0;

  bool operator<(const CommandKey& other) const {
    if (runtime_instance_id != other.runtime_instance_id)
      return runtime_instance_id < other.runtime_instance_id;
    return execution_id < other.execution_id;
  }
};

struct StoredCommand {
  CommandKey key;
  xp::v4::PrimitiveExecutionCommand command;
  std::vector<uint8_t> canonical_payload;
  bool unity_receipt_received = false;
  bool python_ack_sent = false;
};

struct TerminalCommand {
  std::array<uint8_t, 32> command_sequence_hash{{}};
};

struct CommandAuditRecord {
  std::string event;
  std::string runtime_instance_id;
  uint64_t execution_id = 0;
  std::array<uint8_t, 32> command_sequence_hash{{}};
  uint64_t monotonic_ns = 0;
};

class ICommandAuditSink {
 public:
  virtual ~ICommandAuditSink() {}
  virtual void Record(const CommandAuditRecord& record) = 0;
};

class ICommandStore {
 public:
  virtual ~ICommandStore() {}
  virtual CommandStorePutOutcome put_if_absent(
      const CommandKey& key, const StoredCommand& value) = 0;
  virtual bool get(const CommandKey& key, StoredCommand* value) const = 0;
  virtual bool mark_unity_receipt(const CommandKey& key) = 0;
  virtual bool mark_python_ack_sent(const CommandKey& key) = 0;
  virtual bool erase(const CommandKey&) { return false; }
  virtual bool is_terminal(const CommandKey&) const { return false; }
  virtual bool terminal_matches(
      const CommandKey&, const std::array<uint8_t, 32>&) const {
    return false;
  }
  virtual std::vector<CommandKey> pending_keys() const = 0;
  virtual std::vector<CommandKey> python_ack_pending_keys() const = 0;
  virtual size_t pending_count() const = 0;
  virtual size_t python_ack_pending_count() const = 0;
  virtual size_t size() const { return 0; }
  virtual size_t peak_size() const { return size(); }
};

class InMemoryCommandStore : public ICommandStore {
 public:
  CommandStorePutOutcome put_if_absent(
      const CommandKey& key, const StoredCommand& value) override {
    const auto found = values_.find(key);
    if (found == values_.end()) {
      const auto terminal = terminal_.find(key);
      if (terminal != terminal_.end()) {
        return terminal->second.command_sequence_hash ==
                   value.command.command_sequence_hash
            ? CommandStorePutOutcome::DUPLICATE
            : CommandStorePutOutcome::CONFLICT;
      }
      values_.insert(std::make_pair(key, value));
      if (values_.size() > peak_size_) peak_size_ = values_.size();
      return CommandStorePutOutcome::STORED;
    }
    if (found->second.command.command_sequence_hash ==
            value.command.command_sequence_hash &&
        found->second.canonical_payload == value.canonical_payload) {
      return CommandStorePutOutcome::DUPLICATE;
    }
    return CommandStorePutOutcome::CONFLICT;
  }

  bool get(const CommandKey& key, StoredCommand* value) const override {
    const auto found = values_.find(key);
    if (found == values_.end()) return false;
    if (value != nullptr) *value = found->second;
    return true;
  }

  bool mark_unity_receipt(const CommandKey& key) override {
    auto found = values_.find(key);
    if (found == values_.end()) return false;
    found->second.unity_receipt_received = true;
    return true;
  }

  bool mark_python_ack_sent(const CommandKey& key) override {
    auto found = values_.find(key);
    if (found == values_.end()) return false;
    found->second.python_ack_sent = true;
    return true;
  }

  bool erase(const CommandKey& key) override {
    const auto found = values_.find(key);
    if (found == values_.end()) return false;
    terminal_[key] = TerminalCommand{found->second.command.command_sequence_hash};
    terminal_order_.push_back(key);
    while (terminal_order_.size() > kTerminalTombstoneCapacity) {
      terminal_.erase(terminal_order_.front());
      terminal_order_.pop_front();
    }
    values_.erase(found);
    return true;
  }

  bool is_terminal(const CommandKey& key) const override {
    return terminal_.find(key) != terminal_.end();
  }

  bool terminal_matches(
      const CommandKey& key,
      const std::array<uint8_t, 32>& command_sequence_hash) const override {
    const auto found = terminal_.find(key);
    return found != terminal_.end() &&
        found->second.command_sequence_hash == command_sequence_hash;
  }

  std::vector<CommandKey> pending_keys() const override {
    std::vector<CommandKey> result;
    for (const auto& entry : values_) {
      if (!entry.second.unity_receipt_received) result.push_back(entry.first);
    }
    return result;
  }

  std::vector<CommandKey> python_ack_pending_keys() const override {
    std::vector<CommandKey> result;
    for (const auto& entry : values_) {
      if (entry.second.unity_receipt_received &&
          !entry.second.python_ack_sent) {
        result.push_back(entry.first);
      }
    }
    return result;
  }

  size_t pending_count() const override { return pending_keys().size(); }
  size_t python_ack_pending_count() const override {
    return python_ack_pending_keys().size();
  }
  size_t size() const override { return values_.size(); }
  size_t peak_size() const override { return peak_size_; }
  size_t terminal_count() const { return terminal_.size(); }

 private:
  static constexpr size_t kTerminalTombstoneCapacity = 1024;
  std::map<CommandKey, StoredCommand> values_;
  std::map<CommandKey, TerminalCommand> terminal_;
  std::deque<CommandKey> terminal_order_;
  size_t peak_size_ = 0;
};

class IUnityCommandRelay {
 public:
  virtual ~IUnityCommandRelay() {}
  virtual bool Send(const StoredCommand& command) = 0;
};

class IPythonCommandAckSink {
 public:
  virtual ~IPythonCommandAckSink() {}
  virtual bool Send(
      const xp::v4::PrimitiveExecutionCommandReceiptAck& ack) = 0;
};

inline std::array<uint8_t, 32> Sha256(const std::vector<uint8_t>& payload) {
  std::array<uint8_t, 32> digest{{}};
  SHA256(payload.data(), payload.size(), digest.data());
  return digest;
}

inline std::array<uint8_t, 32> CommandSequenceHash(
    const std::vector<xp::v4::PrimitiveExecutionFrame>& frames) {
  return Sha256(xp::v4::canonical_command_sequence_bytes(frames));
}

inline CommandKey KeyFor(const xp::v4::PrimitiveExecutionCommand& command) {
  CommandKey key;
  key.runtime_instance_id = command.runtime_instance_id;
  key.execution_id = command.execution_id;
  return key;
}

inline xp::v4::PrimitiveExecutionCommandReceiptAck ReceiptFor(
    const xp::v4::PrimitiveExecutionCommand& command,
    const std::string& status) {
  xp::v4::PrimitiveExecutionCommandReceiptAck receipt;
  receipt.runtime_instance_id = command.runtime_instance_id;
  receipt.execution_id = command.execution_id;
  receipt.ack_status = status;
  receipt.reason_code = "NONE";
  receipt.command_sequence_hash = command.command_sequence_hash;
  return receipt;
}

class PrimitiveExecutionCommandBroker {
 public:
  PrimitiveExecutionCommandBroker(
      ICommandStore& store,
      IUnityCommandRelay& unity_relay,
      IPythonCommandAckSink& python_ack_sink,
      const std::string& expected_runtime_instance_id,
      ICommandAuditSink* audit_sink = nullptr)
      : store_(store),
        unity_relay_(unity_relay),
        python_ack_sink_(python_ack_sink),
        expected_runtime_instance_id_(expected_runtime_instance_id),
        audit_sink_(audit_sink) {}

  void SetAuditSink(ICommandAuditSink* audit_sink) { audit_sink_ = audit_sink; }

  void SetExpectedRuntimeInstanceId(const std::string& runtime_instance_id) {
    expected_runtime_instance_id_ = runtime_instance_id;
  }

  CommandReceiveOutcome Receive(
      const xp::v4::PrimitiveExecutionCommand& command) {
    if (!expected_runtime_instance_id_.empty() &&
        command.runtime_instance_id != expected_runtime_instance_id_) {
      ++rejected_count_;
      return CommandReceiveOutcome::REJECTED;
    }

    std::vector<uint8_t> sequence;
    std::vector<uint8_t> payload;
    try {
      xp::v4::validate_command(command);
      sequence = xp::v4::canonical_command_sequence_bytes(command.frames);
      payload = xp::v4::canonical_command_payload_bytes(command);
    } catch (...) {
      ++protocol_error_count_;
      return CommandReceiveOutcome::PROTOCOL_ERROR;
    }
    if (Sha256(sequence) != command.command_sequence_hash) {
      ++protocol_error_count_;
      return CommandReceiveOutcome::PROTOCOL_ERROR;
    }

    StoredCommand stored;
    stored.key = KeyFor(command);
    stored.command = command;
    stored.canonical_payload = payload;
    const CommandStorePutOutcome put = store_.put_if_absent(stored.key, stored);
    if (put == CommandStorePutOutcome::CONFLICT) {
      ++protocol_error_count_;
      ++conflict_count_;
      Audit("COMMAND_CONFLICT", stored.command);
      return CommandReceiveOutcome::PROTOCOL_ERROR;
    }
    if (put == CommandStorePutOutcome::DUPLICATE) {
      ++duplicate_count_;
      Audit("COMMAND_DUPLICATE", stored.command);
      StoredCommand existing;
      if (store_.get(stored.key, &existing)) {
        if (existing.unity_receipt_received) {
          // A Python reconnect may resend a command after the Unity receipt
          // was already durable.  Re-emit the exact receipt instead of
          // waiting for another physical admission.
          SendPythonReceipt(existing);
        } else {
          Forward(existing, false);
        }
      } else if (store_.terminal_matches(
                     stored.key, stored.command.command_sequence_hash)) {
        // A terminal tombstone is enough to answer a late duplicate without
        // retaining the complete command payload.
        SendPythonReceipt(stored);
      }
      return CommandReceiveOutcome::DUPLICATE;
    }

    ++accepted_count_;
    Audit("COMMAND_ACCEPTED", stored.command);
    Forward(stored, false);
    return CommandReceiveOutcome::ACCEPTED;
  }

  bool RetryPending(const CommandKey& key) {
    StoredCommand stored;
    if (!store_.get(key, &stored) || stored.unity_receipt_received) return false;
    ++retry_count_;
    Forward(stored, true);
    return true;
  }

  bool RetryPendingCommands() {
    bool attempted = false;
    for (const CommandKey& key : store_.pending_keys()) {
      attempted = RetryPending(key) || attempted;
    }
    return attempted;
  }

  CommandReceiptOutcome HandleUnityReceipt(
      const xp::v4::PrimitiveExecutionCommandReceiptAck& receipt) {
    if (receipt.schema_version != xp::v4::kSchemaVersion ||
        receipt.message_type != "PrimitiveExecutionCommandReceiptAck" ||
        (!expected_runtime_instance_id_.empty() &&
         receipt.runtime_instance_id != expected_runtime_instance_id_) ||
        (receipt.ack_status != "ACCEPTED" &&
         receipt.ack_status != "DUPLICATE")) {
      ++protocol_error_count_;
      return CommandReceiptOutcome::PROTOCOL_ERROR;
    }
    const CommandKey key{receipt.runtime_instance_id, receipt.execution_id};
    StoredCommand stored;
    if (!store_.get(key, &stored)) {
      if (store_.is_terminal(key) && store_.terminal_matches(
              key, receipt.command_sequence_hash)) {
        ++receipt_duplicate_count_;
        return CommandReceiptOutcome::DUPLICATE;
      }
      return CommandReceiptOutcome::NOT_FOUND;
    }
    if (stored.command.command_sequence_hash != receipt.command_sequence_hash) {
      ++protocol_error_count_;
      return CommandReceiptOutcome::PROTOCOL_ERROR;
    }
    if (stored.unity_receipt_received) {
      ++receipt_duplicate_count_;
      Audit("COMMAND_DUPLICATE", stored.command);
      SendPythonReceipt(stored);
      return CommandReceiptOutcome::DUPLICATE;
    }
    store_.mark_unity_receipt(key);
    ++receipt_count_;
    Audit("COMMAND_COMPLETE", stored.command);
    SendPythonReceipt(stored);
    return CommandReceiptOutcome::RECEIVED;
  }

  bool RetryPythonAcks() {
    bool sent = false;
    for (const CommandKey& key : store_.python_ack_pending_keys()) {
      StoredCommand stored;
      if (!store_.get(key, &stored)) continue;
      sent = SendPythonReceipt(stored) || sent;
    }
    return sent;
  }

  size_t pending_count() const { return store_.pending_count(); }
  size_t python_ack_pending_count() const {
    return store_.python_ack_pending_count();
  }
  uint64_t accepted_count() const { return accepted_count_; }
  uint64_t duplicate_count() const { return duplicate_count_; }
  uint64_t conflict_count() const { return conflict_count_; }
  uint64_t protocol_error_count() const { return protocol_error_count_; }
  uint64_t rejected_count() const { return rejected_count_; }
  uint64_t receipt_count() const { return receipt_count_; }
  uint64_t receipt_duplicate_count() const { return receipt_duplicate_count_; }
  uint64_t retry_count() const { return retry_count_; }
  uint64_t forward_count() const { return forward_count_; }
  uint64_t forward_failure_count() const { return forward_failure_count_; }
  uint64_t physical_execution_count() const { return 0; }

  bool FinalizeExecution(
      const CommandKey& key,
      const std::array<uint8_t, 32>& command_sequence_hash) {
    StoredCommand stored;
    if (!store_.get(key, &stored)) {
      return store_.is_terminal(key) &&
          store_.terminal_matches(key, command_sequence_hash);
    }
    if (stored.command.command_sequence_hash != command_sequence_hash) {
      ++protocol_error_count_;
      return false;
    }
    return store_.erase(key);
  }

 private:
  static uint64_t MonotonicNowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
  }

  void Audit(const char* event,
             const xp::v4::PrimitiveExecutionCommand& command) {
    if (audit_sink_ == nullptr) return;
    CommandAuditRecord record;
    record.event = event;
    record.runtime_instance_id = command.runtime_instance_id;
    record.execution_id = command.execution_id;
    record.command_sequence_hash = command.command_sequence_hash;
    record.monotonic_ns = MonotonicNowNs();
    audit_sink_->Record(record);
  }

  bool Forward(const StoredCommand& stored, bool retry) {
    Audit(retry ? "COMMAND_RETRY" : "COMMAND_FORWARD", stored.command);
    const bool sent = unity_relay_.Send(stored);
    if (sent) {
      ++forward_count_;
    } else {
      ++forward_failure_count_;
    }
    return sent;
  }

  bool SendPythonReceipt(const StoredCommand& stored) {
    xp::v4::PrimitiveExecutionCommandReceiptAck receipt =
        ReceiptFor(stored.command, "RECEIVED");
    Audit("COMMAND_RECEIPT_CREATED", stored.command);
    const bool sent = python_ack_sink_.Send(receipt);
    Audit(sent ? "COMMAND_RECEIPT_SEND_OK" : "COMMAND_RECEIPT_SEND_FAILED",
          stored.command);
    if (!sent) return false;
    store_.mark_python_ack_sent(stored.key);
    return true;
  }

  ICommandStore& store_;
  IUnityCommandRelay& unity_relay_;
  IPythonCommandAckSink& python_ack_sink_;
  std::string expected_runtime_instance_id_;
  ICommandAuditSink* audit_sink_ = nullptr;
  uint64_t accepted_count_ = 0;
  uint64_t duplicate_count_ = 0;
  uint64_t conflict_count_ = 0;
  uint64_t protocol_error_count_ = 0;
  uint64_t rejected_count_ = 0;
  uint64_t receipt_count_ = 0;
  uint64_t receipt_duplicate_count_ = 0;
  uint64_t retry_count_ = 0;
  uint64_t forward_count_ = 0;
  uint64_t forward_failure_count_ = 0;
};

}  // namespace primitive_execution_command_broker
}  // namespace planning
