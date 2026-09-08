#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <openssl/sha.h>
#include <string>
#include <vector>

#include "planning/protocol/xm_protocol.hpp"

namespace planning {
namespace primitive_execution_result_broker {

namespace xp = planning::xm_protocol;

enum class ReceiveOutcome {
  ACCEPTED,
  DUPLICATE,
  PROTOCOL_ERROR,
};

enum class StorePutOutcome {
  STORED,
  DUPLICATE,
  CONFLICT,
};

enum class CommitOutcome {
  COMMITTED,
  DUPLICATE,
  PROTOCOL_ERROR,
  NOT_FOUND,
};

enum class ReceiptOutcome {
  RECEIVED,
  DUPLICATE,
  PROTOCOL_ERROR,
  NOT_FOUND,
};

struct ResultKey {
  std::string runtime_instance_id;
  uint64_t execution_id = 0;

  bool operator<(const ResultKey& other) const {
    if (runtime_instance_id != other.runtime_instance_id)
      return runtime_instance_id < other.runtime_instance_id;
    return execution_id < other.execution_id;
  }
};

struct StoredResult {
  ResultKey key;
  xp::v4::PrimitiveExecutionResult result;
  std::vector<uint8_t> canonical_payload;
  bool python_receipt_received = false;
  bool committed = false;
};

struct TerminalResult {
  std::array<uint8_t, 32> result_payload_hash{{}};
  std::array<uint8_t, 32> command_sequence_hash{{}};
};

class IResultStore {
 public:
  virtual ~IResultStore() {}

  virtual StorePutOutcome put_if_absent(
      const ResultKey& key,
      const StoredResult& value) = 0;
  virtual bool get(const ResultKey& key, StoredResult* value) const = 0;
  virtual bool mark_receipt_received(const ResultKey& key) = 0;
  virtual bool is_receipt_received(const ResultKey& key) const = 0;
  virtual bool mark_committed(const ResultKey& key) = 0;
  virtual bool is_committed(const ResultKey& key) const = 0;
  virtual bool erase(const ResultKey&) { return false; }
  virtual bool is_terminal(const ResultKey&) const { return false; }
  virtual bool terminal_matches(
      const ResultKey&, const std::array<uint8_t, 32>&,
      const std::array<uint8_t, 32>*) const {
    return false;
  }
  virtual std::vector<ResultKey> pending_keys() const = 0;
  virtual std::vector<ResultKey> receipt_pending_keys() const = 0;
  virtual size_t pending_count() const = 0;
  virtual size_t receipt_pending_count() const = 0;
  virtual size_t committed_count() const = 0;
  virtual size_t size() const { return 0; }
  virtual size_t peak_size() const { return size(); }
};

class InMemoryResultStore : public IResultStore {
 public:
  // This store preserves the receipt for the lifetime of the bridge process.
  // Crash/restart recovery requires a durable backend and is deferred here.
  StorePutOutcome put_if_absent(
      const ResultKey& key,
      const StoredResult& value) override {
    const std::map<ResultKey, StoredResult>::iterator found = values_.find(key);
    if (found == values_.end()) {
      const std::map<ResultKey, TerminalResult>::const_iterator terminal =
          terminal_.find(key);
      if (terminal != terminal_.end()) {
        if (terminal->second.result_payload_hash ==
                value.result.result_payload_hash &&
            terminal->second.command_sequence_hash ==
                value.result.command_sequence_hash) {
          return StorePutOutcome::DUPLICATE;
        }
        return StorePutOutcome::CONFLICT;
      }
      values_.insert(std::make_pair(key, value));
      if (values_.size() > peak_size_) peak_size_ = values_.size();
      return StorePutOutcome::STORED;
    }
    if (found->second.result.result_payload_hash ==
            value.result.result_payload_hash &&
        found->second.canonical_payload == value.canonical_payload &&
        found->second.result.command_sequence_hash ==
            value.result.command_sequence_hash) {
      return StorePutOutcome::DUPLICATE;
    }
    return StorePutOutcome::CONFLICT;
  }

  bool get(const ResultKey& key, StoredResult* value) const override {
    const std::map<ResultKey, StoredResult>::const_iterator found = values_.find(key);
    if (found == values_.end()) return false;
    if (value != nullptr) *value = found->second;
    return true;
  }

  bool mark_committed(const ResultKey& key) override {
    std::map<ResultKey, StoredResult>::iterator found = values_.find(key);
    if (found == values_.end()) return false;
    if (found->second.committed) return true;
    found->second.committed = true;
    ++committed_total_;
    return true;
  }

  bool is_committed(const ResultKey& key) const override {
    StoredResult value;
    return get(key, &value) && value.committed;
  }

  bool mark_receipt_received(const ResultKey& key) override {
    std::map<ResultKey, StoredResult>::iterator found = values_.find(key);
    if (found == values_.end()) return false;
    found->second.python_receipt_received = true;
    return true;
  }

  bool is_receipt_received(const ResultKey& key) const override {
    StoredResult value;
    return get(key, &value) && value.python_receipt_received;
  }

  bool erase(const ResultKey& key) override {
    const std::map<ResultKey, StoredResult>::iterator found = values_.find(key);
    if (found == values_.end()) return false;
    TerminalResult terminal;
    terminal.result_payload_hash = found->second.result.result_payload_hash;
    terminal.command_sequence_hash = found->second.result.command_sequence_hash;
    terminal_[key] = terminal;
    terminal_order_.push_back(key);
    while (terminal_order_.size() > kTerminalTombstoneCapacity) {
      terminal_.erase(terminal_order_.front());
      terminal_order_.pop_front();
    }
    values_.erase(found);
    return true;
  }

  bool is_terminal(const ResultKey& key) const override {
    return terminal_.find(key) != terminal_.end();
  }

  bool terminal_matches(
      const ResultKey& key,
      const std::array<uint8_t, 32>& result_payload_hash,
      const std::array<uint8_t, 32>* command_sequence_hash) const override {
    const std::map<ResultKey, TerminalResult>::const_iterator found =
        terminal_.find(key);
    if (found == terminal_.end() ||
        found->second.result_payload_hash != result_payload_hash) {
      return false;
    }
    return command_sequence_hash == nullptr ||
        found->second.command_sequence_hash == *command_sequence_hash;
  }

  std::vector<ResultKey> pending_keys() const override {
    std::vector<ResultKey> keys;
    for (const std::pair<const ResultKey, StoredResult>& entry : values_) {
      if (!entry.second.committed) keys.push_back(entry.first);
    }
    return keys;
  }

  std::vector<ResultKey> receipt_pending_keys() const override {
    std::vector<ResultKey> keys;
    for (const std::pair<const ResultKey, StoredResult>& entry : values_) {
      if (!entry.second.python_receipt_received) keys.push_back(entry.first);
    }
    return keys;
  }

  size_t pending_count() const override {
    size_t count = 0;
    for (const std::pair<const ResultKey, StoredResult>& entry : values_) {
      if (!entry.second.committed) ++count;
    }
    return count;
  }

  size_t receipt_pending_count() const override {
    size_t count = 0;
    for (const std::pair<const ResultKey, StoredResult>& entry : values_) {
      if (!entry.second.python_receipt_received) ++count;
    }
    return count;
  }

  size_t committed_count() const override { return committed_total_; }
  size_t size() const override { return values_.size(); }
  size_t peak_size() const override { return peak_size_; }
  size_t terminal_count() const { return terminal_.size(); }

 private:
  static constexpr size_t kTerminalTombstoneCapacity = 1024;
  std::map<ResultKey, StoredResult> values_;
  std::map<ResultKey, TerminalResult> terminal_;
  std::deque<ResultKey> terminal_order_;
  size_t peak_size_ = 0;
  size_t committed_total_ = 0;
};

class IResultAckSink {
 public:
  virtual ~IResultAckSink() {}
  virtual void Send(const xp::v4::PrimitiveExecutionResultAck& ack) = 0;
};

class IPythonResultRelay {
 public:
  virtual ~IPythonResultRelay() {}
  virtual bool Relay(const StoredResult& result) = 0;
};

inline std::array<uint8_t, 32> Sha256(const std::vector<uint8_t>& payload) {
  std::array<uint8_t, 32> digest{{}};
  SHA256(payload.data(), payload.size(), digest.data());
  return digest;
}

inline std::array<uint8_t, 32> ResultPayloadHash(
    const xp::v4::PrimitiveExecutionResult& result) {
  return Sha256(xp::v4::canonical_result_payload_bytes(result));
}

inline ResultKey KeyFor(const xp::v4::PrimitiveExecutionResult& result) {
  ResultKey key;
  key.runtime_instance_id = result.runtime_instance_id;
  key.execution_id = result.execution_id;
  return key;
}

class PrimitiveExecutionResultBroker {
 public:
  PrimitiveExecutionResultBroker(
      IResultStore& store,
      IResultAckSink& ack_sink,
      IPythonResultRelay& relay)
      : store_(store), ack_sink_(ack_sink), relay_(relay) {}

  ReceiveOutcome Receive(const xp::v4::PrimitiveExecutionResult& result) {
    std::vector<uint8_t> canonical_payload;
    try {
      canonical_payload = xp::v4::canonical_result_payload_bytes(result);
    } catch (...) {
      ++protocol_error_count_;
      return ReceiveOutcome::PROTOCOL_ERROR;
    }
    if (Sha256(canonical_payload) != result.result_payload_hash) {
      ++protocol_error_count_;
      return ReceiveOutcome::PROTOCOL_ERROR;
    }

    const ResultKey key = KeyFor(result);
    StoredResult stored;
    stored.key = key;
    stored.result = result;
    stored.canonical_payload = canonical_payload;

    const StorePutOutcome put = store_.put_if_absent(key, stored);
    if (put == StorePutOutcome::CONFLICT) {
      ++conflict_count_;
      ++protocol_error_count_;
      return ReceiveOutcome::PROTOCOL_ERROR;
    }
    if (put == StorePutOutcome::DUPLICATE) {
      ++duplicate_count_;
      SendDurableAck(result);
      return ReceiveOutcome::DUPLICATE;
    }

    ++first_receipt_count_;
    SendDurableAck(result);
    ++python_relay_attempt_count_;
    if (!relay_.Relay(stored)) ++python_relay_failure_count_;
    return ReceiveOutcome::ACCEPTED;
  }

  bool RetryPythonRelay(const ResultKey& key) {
    StoredResult stored;
    if (!store_.get(key, &stored) || stored.python_receipt_received) return false;
    ++python_relay_attempt_count_;
    if (!relay_.Relay(stored)) {
      ++python_relay_failure_count_;
      return false;
    }
    return true;
  }

  ReceiptOutcome AcknowledgePythonReceipt(
      const xp::v4::PrimitiveExecutionResultReceiptAck& receipt) {
    if (receipt.schema_version != xp::v4::kSchemaVersion ||
        receipt.message_type != "PrimitiveExecutionResultReceiptAck" ||
        receipt.receipt_status != "RECEIVED") {
      ++protocol_error_count_;
      return ReceiptOutcome::PROTOCOL_ERROR;
    }

    ResultKey key;
    key.runtime_instance_id = receipt.runtime_instance_id;
    key.execution_id = receipt.execution_id;
    StoredResult stored;
    if (!store_.get(key, &stored)) {
      if (store_.is_terminal(key)) {
        if (!store_.terminal_matches(
                key, receipt.result_payload_hash,
                &receipt.command_sequence_hash)) {
          ++protocol_error_count_;
          return ReceiptOutcome::PROTOCOL_ERROR;
        }
        ++duplicate_receipt_count_;
        return ReceiptOutcome::DUPLICATE;
      }
      return ReceiptOutcome::NOT_FOUND;
    }
    if (stored.result.result_payload_hash != receipt.result_payload_hash ||
        stored.result.command_sequence_hash != receipt.command_sequence_hash) {
      ++protocol_error_count_;
      return ReceiptOutcome::PROTOCOL_ERROR;
    }
    if (store_.is_receipt_received(key)) {
      ++duplicate_receipt_count_;
      return ReceiptOutcome::DUPLICATE;
    }
    if (!store_.mark_receipt_received(key)) return ReceiptOutcome::NOT_FOUND;
    ++receipt_count_;
    return ReceiptOutcome::RECEIVED;
  }

  size_t RetryPendingPythonRelays() {
    size_t delivered = 0;
    const std::vector<ResultKey> keys = store_.receipt_pending_keys();
    for (const ResultKey& key : keys) {
      if (RetryPythonRelay(key)) ++delivered;
    }
    return delivered;
  }

  CommitOutcome CommitPython(
      const ResultKey& key,
      const std::array<uint8_t, 32>& result_payload_hash) {
    return CommitPython(key, result_payload_hash, nullptr);
  }

  CommitOutcome CommitPython(
      const ResultKey& key,
      const std::array<uint8_t, 32>& result_payload_hash,
      const std::array<uint8_t, 32>* command_sequence_hash) {
    StoredResult stored;
    if (!store_.get(key, &stored)) {
      if (store_.is_terminal(key)) {
        if (store_.terminal_matches(
                key, result_payload_hash, command_sequence_hash)) {
          ++duplicate_commit_count_;
          return CommitOutcome::DUPLICATE;
        }
        ++protocol_error_count_;
        return CommitOutcome::PROTOCOL_ERROR;
      }
      return CommitOutcome::NOT_FOUND;
    }
    if (stored.result.result_payload_hash != result_payload_hash) {
      ++protocol_error_count_;
      return CommitOutcome::PROTOCOL_ERROR;
    }
    if (command_sequence_hash != nullptr &&
        stored.result.command_sequence_hash != *command_sequence_hash) {
      ++protocol_error_count_;
      return CommitOutcome::PROTOCOL_ERROR;
    }
    if (stored.committed || store_.is_committed(key)) {
      ++duplicate_commit_count_;
      return CommitOutcome::DUPLICATE;
    }
    if (!store_.mark_committed(key)) return CommitOutcome::NOT_FOUND;
    ++python_commit_count_;
    ++transition_side_effect_count_;
    store_.erase(key);
    return CommitOutcome::COMMITTED;
  }

  size_t first_receipt_count() const { return first_receipt_count_; }
  size_t duplicate_count() const { return duplicate_count_; }
  size_t conflict_count() const { return conflict_count_; }
  size_t protocol_error_count() const { return protocol_error_count_; }
  size_t python_relay_attempt_count() const { return python_relay_attempt_count_; }
  size_t python_relay_failure_count() const { return python_relay_failure_count_; }
  size_t receipt_pending_count() const { return store_.receipt_pending_count(); }
  size_t receipt_count() const { return receipt_count_; }
  size_t duplicate_receipt_count() const { return duplicate_receipt_count_; }
  size_t python_commit_count() const { return python_commit_count_; }
  size_t duplicate_commit_count() const { return duplicate_commit_count_; }
  size_t transition_side_effect_count() const {
    return transition_side_effect_count_;
  }

 private:
  void SendDurableAck(const xp::v4::PrimitiveExecutionResult& result) {
    xp::v4::PrimitiveExecutionResultAck ack;
    ack.runtime_instance_id = result.runtime_instance_id;
    ack.execution_id = result.execution_id;
    ack.ack_status = "DURABLE_RECEIVED";
    ack.result_payload_hash = result.result_payload_hash;
    ack.command_sequence_hash = result.command_sequence_hash;
    ack_sink_.Send(ack);
  }

  IResultStore& store_;
  IResultAckSink& ack_sink_;
  IPythonResultRelay& relay_;
  size_t first_receipt_count_ = 0;
  size_t duplicate_count_ = 0;
  size_t conflict_count_ = 0;
  size_t protocol_error_count_ = 0;
  size_t python_relay_attempt_count_ = 0;
  size_t python_relay_failure_count_ = 0;
  size_t receipt_count_ = 0;
  size_t duplicate_receipt_count_ = 0;
  size_t python_commit_count_ = 0;
  size_t duplicate_commit_count_ = 0;
  size_t transition_side_effect_count_ = 0;
};

}  // namespace primitive_execution_result_broker
}  // namespace planning
