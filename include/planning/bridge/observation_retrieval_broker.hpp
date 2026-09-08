#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <openssl/sha.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "planning/protocol/xm_protocol.hpp"

namespace planning {
namespace observation_retrieval_broker {

namespace xp = planning::xm_protocol;

enum class RequestOutcome {
  REQUESTED,
  DUPLICATE,
  PROTOCOL_ERROR,
};

enum class SnapshotOutcome {
  STORED,
  DUPLICATE,
  MISSING,
  NOT_FOUND,
  PROTOCOL_ERROR,
};

enum class SnapshotStoreOutcome {
  STORED,
  DUPLICATE,
  CONFLICT,
};

struct ExecutionKey {
  std::string runtime_instance_id;
  uint64_t execution_id = 0;
  // Reset snapshot requests use the reserved execution identity (execution_id
  // 0 and zero hashes), so their exact ObservationRef must be part of the
  // request key. Primitive requests intentionally leave this empty and keep
  // the existing execution-identity deduplication semantics.
  std::vector<uint8_t> observation_ref_key;

  bool operator<(const ExecutionKey& other) const {
    if (runtime_instance_id != other.runtime_instance_id)
      return runtime_instance_id < other.runtime_instance_id;
    if (execution_id != other.execution_id)
      return execution_id < other.execution_id;
    return observation_ref_key < other.observation_ref_key;
  }
};

struct StoredSnapshot {
  xp::v4::EndpointObservationSnapshot snapshot;
  std::vector<uint8_t> canonical_payload;
};

class ISnapshotCache {
 public:
  virtual ~ISnapshotCache() {}

  virtual SnapshotStoreOutcome put_if_absent(
      const xp::v4::EndpointObservationSnapshot& snapshot,
      const std::vector<uint8_t>& canonical_payload) = 0;
  virtual bool get(
      const xp::v4::ObservationRef& observation_ref,
      xp::v4::EndpointObservationSnapshot* snapshot) const = 0;
  virtual bool erase(const xp::v4::ObservationRef&) { return false; }
  virtual size_t size() const = 0;
  virtual size_t peak_size() const { return size(); }
};

class InMemorySnapshotCache : public ISnapshotCache {
 public:
  SnapshotStoreOutcome put_if_absent(
      const xp::v4::EndpointObservationSnapshot& snapshot,
      const std::vector<uint8_t>& canonical_payload) override {
    const std::vector<uint8_t> key = xp::v4::canonical_observation_ref_bytes(
        snapshot.observation_ref);
    const std::map<std::vector<uint8_t>, StoredSnapshot>::iterator found =
        snapshots_.find(key);
    if (found == snapshots_.end()) {
      StoredSnapshot stored;
      stored.snapshot = snapshot;
      stored.canonical_payload = canonical_payload;
      snapshots_.insert(std::make_pair(key, stored));
      if (snapshots_.size() > peak_size_) peak_size_ = snapshots_.size();
      return SnapshotStoreOutcome::STORED;
    }
    if (found->second.snapshot.snapshot_hash == snapshot.snapshot_hash &&
        found->second.canonical_payload == canonical_payload) {
      return SnapshotStoreOutcome::DUPLICATE;
    }
    return SnapshotStoreOutcome::CONFLICT;
  }

  bool get(
      const xp::v4::ObservationRef& observation_ref,
      xp::v4::EndpointObservationSnapshot* snapshot) const override {
    const std::vector<uint8_t> key = xp::v4::canonical_observation_ref_bytes(
        observation_ref);
    const std::map<std::vector<uint8_t>, StoredSnapshot>::const_iterator found =
        snapshots_.find(key);
    if (found == snapshots_.end()) return false;
    if (snapshot != nullptr) *snapshot = found->second.snapshot;
    return true;
  }

  bool erase(const xp::v4::ObservationRef& observation_ref) override {
    const std::vector<uint8_t> key = xp::v4::canonical_observation_ref_bytes(
        observation_ref);
    return snapshots_.erase(key) != 0;
  }

  size_t size() const override { return snapshots_.size(); }
  size_t peak_size() const override { return peak_size_; }

 private:
  std::map<std::vector<uint8_t>, StoredSnapshot> snapshots_;
  size_t peak_size_ = 0;
};

class IUnitySnapshotRequester {
 public:
  virtual ~IUnitySnapshotRequester() {}
  virtual bool Request(const xp::v4::SnapshotRequest& request) = 0;
};

class ISnapshotAckSink {
 public:
  virtual ~ISnapshotAckSink() {}
  virtual void Send(const xp::v4::SnapshotAck& ack) = 0;
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

inline std::array<uint8_t, 32> SnapshotHash(
    const xp::v4::EndpointObservationSnapshot& snapshot) {
  return Sha256(xp::v4::canonical_endpoint_observation_snapshot_bytes(snapshot));
}

inline ExecutionKey KeyFor(const xp::v4::PrimitiveExecutionResult& result) {
  ExecutionKey key;
  key.runtime_instance_id = result.runtime_instance_id;
  key.execution_id = result.execution_id;
  return key;
}

inline bool IsReservedResetRequest(const xp::v4::SnapshotRequest& request) {
  if (request.execution_id != 0) return false;
  for (size_t index = 0; index < request.result_payload_hash.size(); ++index) {
    if (request.result_payload_hash[index] != 0 ||
        request.command_sequence_hash[index] != 0) {
      return false;
    }
  }
  return true;
}

inline ExecutionKey KeyFor(const xp::v4::SnapshotRequest& request) {
  ExecutionKey key;
  key.runtime_instance_id = request.observation_ref.runtime_instance_id;
  key.execution_id = request.execution_id;
  if (IsReservedResetRequest(request)) {
    key.observation_ref_key = xp::v4::canonical_observation_ref_bytes(
        request.observation_ref);
  }
  return key;
}

inline xp::v4::ObservationRef ObservationRefFor(
    const xp::v4::PrimitiveExecutionResult& result) {
  const xp::v4::EndpointObservationRef& source = result.endpoint_observation_ref;
  xp::v4::ObservationRef observation_ref;
  observation_ref.schema_version = source.schema_version;
  observation_ref.runtime_instance_id = source.runtime_instance_id;
  observation_ref.episode_id = source.episode_id;
  observation_ref.reset_id = source.reset_id;
  observation_ref.state_id = source.state_id;
  observation_ref.depth_id = source.depth_id;
  observation_ref.sim_time_ns = source.sim_time_ns;
  return observation_ref;
}

class ObservationRetrievalBroker {
 public:
  static constexpr size_t kTerminalTombstoneCapacity = 1024;

  ObservationRetrievalBroker(
      ISnapshotCache& cache,
      IUnitySnapshotRequester& requester,
      ISnapshotAckSink& ack_sink,
      uint64_t retry_interval_ms)
      : cache_(cache), requester_(requester), ack_sink_(ack_sink),
        retry_interval_ms_(retry_interval_ms) {
    if (retry_interval_ms_ == 0) throw std::invalid_argument("retry interval must be positive");
  }

  RequestOutcome ReceiveComplete(
      const xp::v4::PrimitiveExecutionResult& result,
      uint64_t now_ms) {
    xp::v4::SnapshotRequest request;
    try {
      if (result.status != "COMPLETE")
        throw std::invalid_argument("snapshot retrieval requires COMPLETE");
      return ReceiveResult(result, now_ms);
    } catch (...) {
      ++protocol_error_count_;
      return RequestOutcome::PROTOCOL_ERROR;
    }
  }

  // Environment-terminal FAILED results may carry an exact authoritative
  // terminal ObservationRef just like COMPLETE.  Keep the old named seam for
  // callers/tests, but share one identity/pending implementation.
  RequestOutcome ReceiveResult(
      const xp::v4::PrimitiveExecutionResult& result,
      uint64_t now_ms) {
    xp::v4::SnapshotRequest request;
    try {
      if (ResultPayloadHash(result) != result.result_payload_hash)
        throw std::invalid_argument("result payload hash mismatch");
      request.observation_ref = ObservationRefFor(result);
      xp::v4::canonical_observation_ref_bytes(request.observation_ref);
      request.execution_id = result.execution_id;
      request.result_payload_hash = result.result_payload_hash;
      request.command_sequence_hash = result.command_sequence_hash;
    } catch (...) {
      ++protocol_error_count_;
      return RequestOutcome::PROTOCOL_ERROR;
    }

    const ExecutionKey key = KeyFor(result);
    const std::vector<uint8_t> canonical_request =
        xp::v4::canonical_snapshot_request_bytes(request);
    const std::map<ExecutionKey, TerminalSnapshot>::const_iterator terminal =
        terminal_.find(key);
    if (terminal != terminal_.end()) {
      if (terminal->second.canonical_request != canonical_request) {
        ++protocol_error_count_;
        return RequestOutcome::PROTOCOL_ERROR;
      }
      ++duplicate_request_count_;
      return RequestOutcome::DUPLICATE;
    }
    const std::map<ExecutionKey, PendingRequest>::iterator found = pending_.find(key);
    if (found != pending_.end()) {
      if (found->second.canonical_request != canonical_request) {
        ++protocol_error_count_;
        return RequestOutcome::PROTOCOL_ERROR;
      }
      ++duplicate_request_count_;
      return RequestOutcome::DUPLICATE;
    }

    PendingRequest pending;
    pending.request = request;
    pending.canonical_request = canonical_request;
    pending.pending = true;
    pending_.insert(std::make_pair(key, pending));
    SendRequest(key, now_ms);
    ++request_count_;
    return RequestOutcome::REQUESTED;
  }

  // Reset completion has no primitive result identity.  It still uses the
  // same exact ObservationRef and immutable snapshot store, with a reserved
  // execution identity supplied by the caller.
  RequestOutcome ReceiveRequest(
      const xp::v4::SnapshotRequest& request, uint64_t now_ms) {
    std::vector<uint8_t> canonical_request;
    try {
      canonical_request = xp::v4::canonical_snapshot_request_bytes(request);
    } catch (...) {
      ++protocol_error_count_;
      return RequestOutcome::PROTOCOL_ERROR;
    }
    const ExecutionKey key = KeyFor(request);
    const std::map<ExecutionKey, TerminalSnapshot>::const_iterator terminal =
        terminal_.find(key);
    if (terminal != terminal_.end()) {
      if (terminal->second.canonical_request != canonical_request) {
        ++protocol_error_count_;
        return RequestOutcome::PROTOCOL_ERROR;
      }
      ++duplicate_request_count_;
      return RequestOutcome::DUPLICATE;
    }
    const std::map<ExecutionKey, PendingRequest>::iterator found = pending_.find(key);
    if (found != pending_.end()) {
      if (found->second.canonical_request != canonical_request) {
        ++protocol_error_count_;
        return RequestOutcome::PROTOCOL_ERROR;
      }
      ++duplicate_request_count_;
      return RequestOutcome::DUPLICATE;
    }
    PendingRequest pending;
    pending.request = request;
    pending.canonical_request = canonical_request;
    pending.pending = true;
    pending_.insert(std::make_pair(key, pending));
    SendRequest(key, now_ms);
    ++request_count_;
    return RequestOutcome::REQUESTED;
  }

  SnapshotOutcome ReceiveSnapshot(
      const xp::v4::SnapshotRequest& request,
      const xp::v4::EndpointObservationSnapshot& snapshot) {
    const ExecutionKey key = KeyFor(request);
    const std::map<ExecutionKey, PendingRequest>::iterator found = pending_.find(key);
    if (found == pending_.end()) {
      const std::map<ExecutionKey, TerminalSnapshot>::const_iterator terminal =
          terminal_.find(key);
      if (terminal == terminal_.end()) return SnapshotOutcome::NOT_FOUND;
      try {
        const std::vector<uint8_t> canonical_request =
            xp::v4::canonical_snapshot_request_bytes(request);
        const std::vector<uint8_t> canonical_snapshot =
            xp::v4::canonical_endpoint_observation_snapshot_bytes(snapshot);
        if (!terminal->second.has_snapshot ||
            terminal->second.canonical_request != canonical_request ||
            xp::v4::canonical_observation_ref_bytes(request.observation_ref) !=
                xp::v4::canonical_observation_ref_bytes(snapshot.observation_ref) ||
            SnapshotHash(snapshot) != terminal->second.snapshot_hash) {
          throw std::invalid_argument("terminal snapshot identity mismatch");
        }
        (void)canonical_snapshot;
      } catch (...) {
        ++protocol_error_count_;
        return SnapshotOutcome::PROTOCOL_ERROR;
      }
      SendSnapshotAck(snapshot);
      ++duplicate_snapshot_count_;
      return SnapshotOutcome::DUPLICATE;
    }

    std::vector<uint8_t> canonical_request;
    std::vector<uint8_t> canonical_snapshot;
    try {
      canonical_request = xp::v4::canonical_snapshot_request_bytes(request);
      canonical_snapshot = xp::v4::canonical_endpoint_observation_snapshot_bytes(snapshot);
      if (canonical_request != found->second.canonical_request ||
          xp::v4::canonical_observation_ref_bytes(request.observation_ref) !=
              xp::v4::canonical_observation_ref_bytes(snapshot.observation_ref) ||
          SnapshotHash(snapshot) != snapshot.snapshot_hash) {
        throw std::invalid_argument("snapshot response identity mismatch");
      }
    } catch (...) {
      ++protocol_error_count_;
      return SnapshotOutcome::PROTOCOL_ERROR;
    }

    const SnapshotStoreOutcome store = cache_.put_if_absent(snapshot, canonical_snapshot);
    if (store == SnapshotStoreOutcome::CONFLICT) {
      ++conflict_count_;
      ++protocol_error_count_;
      return SnapshotOutcome::PROTOCOL_ERROR;
    }

    found->second.pending = false;
    SendSnapshotAck(snapshot);
    if (store == SnapshotStoreOutcome::DUPLICATE) {
      ++duplicate_snapshot_count_;
      return SnapshotOutcome::DUPLICATE;
    }
    ++stored_snapshot_count_;
    return SnapshotOutcome::STORED;
  }

  // The Python result commit is the terminal consumer event for a primitive
  // endpoint snapshot.  Drop the full payload only after the exact request
  // has been served; keep a bounded identity tombstone so late duplicate
  // Unity responses can still be acknowledged without retaining bytes.
  bool FinalizeRequest(const xp::v4::SnapshotRequest& request) {
    const ExecutionKey key = KeyFor(request);
    std::vector<uint8_t> canonical_request;
    try {
      canonical_request = xp::v4::canonical_snapshot_request_bytes(request);
    } catch (...) {
      ++protocol_error_count_;
      return false;
    }
    const std::map<ExecutionKey, TerminalSnapshot>::const_iterator terminal =
        terminal_.find(key);
    if (terminal != terminal_.end())
      return terminal->second.canonical_request == canonical_request;

    const std::map<ExecutionKey, PendingRequest>::iterator found = pending_.find(key);
    if (found == pending_.end() || found->second.canonical_request != canonical_request)
      return false;

    TerminalSnapshot record;
    record.canonical_request = canonical_request;
    xp::v4::EndpointObservationSnapshot snapshot;
    if (cache_.get(request.observation_ref, &snapshot)) {
      record.has_snapshot = true;
      record.snapshot_hash = snapshot.snapshot_hash;
      if (!cache_.erase(request.observation_ref)) return false;
    }
    RememberTerminal(key, record);
    pending_.erase(found);
    return true;
  }

  SnapshotOutcome ReportMissingSnapshot(const xp::v4::SnapshotRequest& request) {
    const ExecutionKey key = KeyFor(request);
    const std::map<ExecutionKey, PendingRequest>::iterator found = pending_.find(key);
    if (found == pending_.end()) return SnapshotOutcome::NOT_FOUND;
    try {
      if (xp::v4::canonical_snapshot_request_bytes(request) !=
          found->second.canonical_request) {
        ++protocol_error_count_;
        return SnapshotOutcome::PROTOCOL_ERROR;
      }
    } catch (...) {
      ++protocol_error_count_;
      return SnapshotOutcome::PROTOCOL_ERROR;
    }
    ++missing_snapshot_count_;
    return SnapshotOutcome::MISSING;
  }

  size_t Poll(uint64_t now_ms) {
    if (!connected_) return 0;
    size_t attempts = 0;
    for (std::map<ExecutionKey, PendingRequest>::iterator entry = pending_.begin();
         entry != pending_.end(); ++entry) {
      if (!entry->second.pending || now_ms < entry->second.next_retry_at_ms) continue;
      SendRequest(entry->first, now_ms);
      ++attempts;
    }
    return attempts;
  }

  void OnUnityDisconnected() { connected_ = false; }

  size_t OnUnityReconnected(uint64_t now_ms) {
    connected_ = true;
    size_t attempts = 0;
    for (std::map<ExecutionKey, PendingRequest>::iterator entry = pending_.begin();
         entry != pending_.end(); ++entry) {
      if (!entry->second.pending) continue;
      SendRequest(entry->first, now_ms);
      ++attempts;
    }
    return attempts;
  }

  bool GetSnapshot(
      const xp::v4::SnapshotRequest& request,
      xp::v4::EndpointObservationSnapshot* snapshot) const {
    const std::map<ExecutionKey, PendingRequest>::const_iterator found =
        pending_.find(KeyFor(request));
    if (found == pending_.end()) return false;
    try {
      if (xp::v4::canonical_snapshot_request_bytes(request) !=
          found->second.canonical_request) {
        return false;
      }
    } catch (...) {
      return false;
    }
    return cache_.get(request.observation_ref, snapshot);
  }

  size_t pending_count() const {
    size_t count = 0;
    for (std::map<ExecutionKey, PendingRequest>::const_iterator entry = pending_.begin();
         entry != pending_.end(); ++entry)
      if (entry->second.pending) ++count;
    return count;
  }

  size_t current_snapshot_entries() const { return cache_.size(); }
  size_t peak_snapshot_entries() const { return cache_.peak_size(); }
  size_t terminal_snapshot_count() const { return terminal_.size(); }

  bool IsTerminalRequest(const xp::v4::SnapshotRequest& request) const {
    try {
      return terminal_.find(KeyFor(request)) != terminal_.end();
    } catch (...) {
      return false;
    }
  }

  size_t request_count() const { return request_count_; }
  size_t retry_count() const { return retry_count_; }
  size_t missing_snapshot_count() const { return missing_snapshot_count_; }
  size_t stored_snapshot_count() const { return stored_snapshot_count_; }
  size_t duplicate_request_count() const { return duplicate_request_count_; }
  size_t duplicate_snapshot_count() const { return duplicate_snapshot_count_; }
  size_t conflict_count() const { return conflict_count_; }
  size_t protocol_error_count() const { return protocol_error_count_; }

 private:
  struct TerminalSnapshot {
    std::vector<uint8_t> canonical_request;
    std::array<uint8_t, 32> snapshot_hash{{}};
    bool has_snapshot = false;
  };

  struct PendingRequest {
    xp::v4::SnapshotRequest request;
    std::vector<uint8_t> canonical_request;
    uint64_t next_retry_at_ms = 0;
    bool pending = false;
  };

  void SendRequest(const ExecutionKey& key, uint64_t now_ms) {
    std::map<ExecutionKey, PendingRequest>::iterator found = pending_.find(key);
    if (found == pending_.end() || !found->second.pending) return;
    if (connected_) requester_.Request(found->second.request);
    found->second.next_retry_at_ms = now_ms + retry_interval_ms_;
    ++retry_count_;
  }

  void SendSnapshotAck(const xp::v4::EndpointObservationSnapshot& snapshot) {
    xp::v4::SnapshotAck ack;
    ack.observation_ref = snapshot.observation_ref;
    ack.snapshot_hash = snapshot.snapshot_hash;
    ack_sink_.Send(ack);
  }

  void RememberTerminal(
      const ExecutionKey& key, const TerminalSnapshot& snapshot) {
    const std::map<ExecutionKey, TerminalSnapshot>::iterator found =
        terminal_.find(key);
    if (found != terminal_.end()) {
      found->second = snapshot;
      return;
    }
    terminal_.insert(std::make_pair(key, snapshot));
    terminal_order_.push_back(key);
    while (terminal_order_.size() > kTerminalTombstoneCapacity) {
      terminal_.erase(terminal_order_.front());
      terminal_order_.pop_front();
    }
  }

  ISnapshotCache& cache_;
  IUnitySnapshotRequester& requester_;
  ISnapshotAckSink& ack_sink_;
  uint64_t retry_interval_ms_;
  bool connected_ = true;
  std::map<ExecutionKey, PendingRequest> pending_;
  std::map<ExecutionKey, TerminalSnapshot> terminal_;
  std::deque<ExecutionKey> terminal_order_;
  size_t request_count_ = 0;
  size_t retry_count_ = 0;
  size_t missing_snapshot_count_ = 0;
  size_t stored_snapshot_count_ = 0;
  size_t duplicate_request_count_ = 0;
  size_t duplicate_snapshot_count_ = 0;
  size_t conflict_count_ = 0;
  size_t protocol_error_count_ = 0;
};

}  // namespace observation_retrieval_broker
}  // namespace planning
