#pragma once

#include <array>
#include <cstdint>
#include <cerrno>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <openssl/sha.h>
#include <zmq.h>

#include <planning/bridge/bridge_util.hpp>
#include <planning/bridge/observation_retrieval_broker.hpp>
#include <planning/bridge/primitive_execution_command_broker.hpp>
#include <planning/bridge/primitive_execution_result_broker.hpp>
#include <planning/protocol/endpoint_observation_snapshot_wire.hpp>
#include <planning/protocol/primitive_execution_command_wire.hpp>
#include <planning/protocol/primitive_execution_result_wire.hpp>
#include <planning/protocol/xm_protocol.hpp>

namespace planning::transport {

namespace xp = planning::xm_protocol;
namespace result_broker = planning::primitive_execution_result_broker;
namespace observation_broker = planning::observation_retrieval_broker;
namespace command_broker = planning::primitive_execution_command_broker;
namespace command_wire = planning::primitive_execution_command_wire;
namespace result_wire = planning::primitive_execution_result_wire;
namespace snapshot_wire = planning::endpoint_observation_snapshot_wire;

using planning::bridge_util::HexBytes;
using planning::bridge_util::RuntimeInstanceIdFromDealerIdentity;
using planning::bridge_util::SocketThreadDiagnosticFields;
using planning::bridge_util::WallTimeNs;
using planning::bridge_util::WritePythonDiagnostic;
using planning::bridge_util::ZmqSocketEvents;

class ZmqUnityResultAckSink : public result_broker::IResultAckSink {
 public:
  void Configure(void* router_socket, uint64_t* ack_count) {
    router_socket_ = router_socket;
    ack_count_ = ack_count;
  }

  void SetIdentity(const std::vector<uint8_t>& identity) {
    identity_ = identity;
  }

  void Send(const xp::v4::PrimitiveExecutionResultAck& ack) override {
    if (!router_socket_ || identity_.empty()) return;
    const std::vector<uint8_t> payload = result_wire::SerializeAck(ack);
    const int identity_rc = zmq_send(
        router_socket_, identity_.data(), identity_.size(), ZMQ_SNDMORE | ZMQ_DONTWAIT);
    if (identity_rc < 0) return;
    const int payload_rc = zmq_send(
        router_socket_, payload.data(), payload.size(), ZMQ_DONTWAIT);
    if (payload_rc >= 0 && ack_count_ != nullptr) ++(*ack_count_);
  }

 private:
  void* router_socket_ = nullptr;
  uint64_t* ack_count_ = nullptr;
  std::vector<uint8_t> identity_;
};

class ZmqUnitySnapshotRequester : public observation_broker::IUnitySnapshotRequester {
 public:
  void Configure(void* router_socket) { router_socket_ = router_socket; }

  void RegisterReady(
      const std::string& runtime_instance_id,
      const std::vector<uint8_t>& identity) {
    peers_[runtime_instance_id] = identity;
  }

  bool HasPeer(const std::string& runtime_instance_id) const {
    return peers_.find(runtime_instance_id) != peers_.end();
  }

  bool IdentityMatches(
      const std::string& runtime_instance_id,
      const std::vector<uint8_t>& identity) const {
    const std::map<std::string, std::vector<uint8_t>>::const_iterator found =
        peers_.find(runtime_instance_id);
    return found != peers_.end() && found->second == identity;
  }

  bool Request(const xp::v4::SnapshotRequest& request) override {
    const std::map<std::string, std::vector<uint8_t>>::const_iterator found =
        peers_.find(request.observation_ref.runtime_instance_id);
    if (!router_socket_ || found == peers_.end() || found->second.empty()) return false;
    const std::vector<uint8_t> payload = snapshot_wire::SerializeRequest(request);
    const int identity_rc = zmq_send(
        router_socket_, found->second.data(), found->second.size(),
        ZMQ_SNDMORE | ZMQ_DONTWAIT);
    if (identity_rc != static_cast<int>(found->second.size())) return false;
    return zmq_send(
        router_socket_, payload.data(), payload.size(), ZMQ_DONTWAIT) ==
        static_cast<int>(payload.size());
  }

  bool SendAck(const xp::v4::SnapshotAck& ack) {
    const std::map<std::string, std::vector<uint8_t>>::const_iterator found =
        peers_.find(ack.observation_ref.runtime_instance_id);
    if (!router_socket_ || found == peers_.end() || found->second.empty()) return false;
    const std::vector<uint8_t> payload = snapshot_wire::SerializeAck(ack);
    const int identity_rc = zmq_send(
        router_socket_, found->second.data(), found->second.size(),
        ZMQ_SNDMORE | ZMQ_DONTWAIT);
    if (identity_rc != static_cast<int>(found->second.size())) return false;
    return zmq_send(
        router_socket_, payload.data(), payload.size(), ZMQ_DONTWAIT) ==
        static_cast<int>(payload.size());
  }

 private:
  void* router_socket_ = nullptr;
  std::map<std::string, std::vector<uint8_t>> peers_;
};

class ZmqUnitySnapshotAckSink : public observation_broker::ISnapshotAckSink {
 public:
  void Configure(ZmqUnitySnapshotRequester* requester) { requester_ = requester; }

  void Send(const xp::v4::SnapshotAck& ack) override {
    if (requester_ != nullptr) requester_->SendAck(ack);
  }

 private:
  ZmqUnitySnapshotRequester* requester_ = nullptr;
};

class ZmqPythonResultRelay : public result_broker::IPythonResultRelay {
 public:
  enum class PeerState {
    READY,
    STALE,
  };

  struct PeerRegistration {
    std::vector<uint8_t> identity;
    std::string runtime_instance_id;
    uint64_t connection_epoch = 0;
    PeerState state = PeerState::STALE;
  };

  void Configure(
      void* router_socket,
      uint64_t* relay_count,
      std::ofstream* diagnostics_stream,
      const std::string& socket_owner_thread_id,
      int stale_retry_threshold) {
    router_socket_ = router_socket;
    relay_count_ = relay_count;
    diagnostics_stream_ = diagnostics_stream;
    socket_owner_thread_id_ = socket_owner_thread_id;
    stale_retry_threshold_ = stale_retry_threshold > 0
        ? stale_retry_threshold
        : 1;
  }

  bool ObserveIdentity(const std::vector<uint8_t>& identity) {
    const bool changed = observed_identity_ != identity;
    observed_identity_ = identity;
    return changed;
  }

  bool RegisterReady(const std::vector<uint8_t>& identity) {
    // A READY is a registration boundary.  For an existing identity it means
    // the ROUTER handover option has replaced the old DEALER pipe; never keep
    // using a previous pipe merely because its identity bytes are unchanged.
    const bool replaces_existing_identity =
        !peer_.identity.empty() && peer_.identity == identity;
    peer_.identity = identity;
    peer_.runtime_instance_id = RuntimeInstanceIdFromDealerIdentity(identity);
    peer_.connection_epoch = ++next_connection_epoch_;
    peer_.state = PeerState::READY;
    observed_identity_ = identity;
    unacknowledged_attempts_.clear();
    return replaces_existing_identity;
  }

  void MarkStale() {
    // Preserve the registration for audit, but invalidate the route until a
    // subsequent READY establishes a current DEALER pipe.
    peer_.state = PeerState::STALE;
  }

  bool is_ready() const { return peer_.state == PeerState::READY; }

  const PeerRegistration& peer() const { return peer_; }

  void AcknowledgeReceipt(
      const xp::v4::PrimitiveExecutionResultReceiptAck& receipt) {
    unacknowledged_attempts_.erase(
        PendingKey(receipt.runtime_instance_id, receipt.execution_id));
  }

  void ForgetResult(
      const std::string& runtime_instance_id, uint64_t execution_id) {
    unacknowledged_attempts_.erase(
        PendingKey(runtime_instance_id, execution_id));
  }

  size_t unacknowledged_count() const {
    return unacknowledged_attempts_.size();
  }

  void SetLastPythonReadyMonotonicNs(int64_t timestamp_ns) {
    last_python_ready_monotonic_ns_ = timestamp_ns;
  }

  bool Relay(const result_broker::StoredResult& stored) override {
    const int64_t relay_started_monotonic_ns = WallTimeNs();
    int zmq_events = 0;
    size_t zmq_events_size = sizeof(zmq_events);
    const int zmq_events_rc = router_socket_ == nullptr
        ? -1
        : zmq_getsockopt(
              router_socket_, ZMQ_EVENTS, &zmq_events, &zmq_events_size);
    const int zmq_events_errno = zmq_events_rc < 0 ? zmq_errno() : 0;
    const bool peer_available =
        router_socket_ != nullptr && peer_.state == PeerState::READY &&
        !peer_.identity.empty();
    if (!peer_available) {
      WritePythonDiagnostic(
          diagnostics_stream_,
          "\"event\":\"relay_skip\",\"execution_id\":" +
              std::to_string(stored.result.execution_id) +
              ",\"result_payload_hash\":\"" +
              HexBytes(stored.result.result_payload_hash.data(),
                       stored.result.result_payload_hash.size()) +
              "\",\"peer_available\":false,\"target_router_identity_hex\":\"" +
              HexBytes(peer_.identity) + "\",\"target_router_identity_length\":" +
              std::to_string(peer_.identity.size()) +
              ",\"relay_started_monotonic_ns\":" +
              std::to_string(relay_started_monotonic_ns) +
              ",\"zmq_events\":" + std::to_string(zmq_events) +
              ",\"zmq_events_rc\":" + std::to_string(zmq_events_rc) +
              ",\"zmq_events_errno\":" + std::to_string(zmq_events_errno) +
              ",\"last_python_ready_monotonic_ns\":" +
              std::to_string(last_python_ready_monotonic_ns_) + "," +
              SocketThreadDiagnosticFields(socket_owner_thread_id_));
      return false;
    }
    const std::vector<uint8_t> payload = result_wire::SerializeResult(stored.result);
    int router_mandatory = -1;
    size_t router_mandatory_size = sizeof(router_mandatory);
    const int router_mandatory_rc = zmq_getsockopt(
        router_socket_, ZMQ_ROUTER_MANDATORY, &router_mandatory,
        &router_mandatory_size);
    const int router_mandatory_errno =
        router_mandatory_rc < 0 ? zmq_errno() : 0;
    const bool router_mandatory_available = router_mandatory_rc == 0;
    int socket_send_hwm = -1;
    size_t socket_send_hwm_size = sizeof(socket_send_hwm);
    const int socket_send_hwm_rc = zmq_getsockopt(
        router_socket_, ZMQ_SNDHWM, &socket_send_hwm, &socket_send_hwm_size);
    const int socket_send_hwm_errno =
        socket_send_hwm_rc < 0 ? zmq_errno() : 0;
    const int identity_rc = zmq_send(
        router_socket_, peer_.identity.data(), peer_.identity.size(),
        ZMQ_SNDMORE | ZMQ_DONTWAIT);
    const int identity_errno = identity_rc < 0 ? zmq_errno() : 0;
    int payload_rc = -1;
    int payload_errno = identity_errno;
    if (identity_rc >= 0) {
      payload_rc = zmq_send(
          router_socket_, payload.data(), payload.size(), ZMQ_DONTWAIT);
      payload_errno = payload_rc < 0 ? zmq_errno() : 0;
    }
    const bool router_send_api_accepted =
        identity_rc == static_cast<int>(peer_.identity.size()) &&
        payload_rc == static_cast<int>(payload.size());
    WritePythonDiagnostic(
        diagnostics_stream_,
        "\"event\":\"relay_send\",\"execution_id\":" +
            std::to_string(stored.result.execution_id) +
            ",\"result_payload_hash\":\"" +
            HexBytes(stored.result.result_payload_hash.data(),
                     stored.result.result_payload_hash.size()) +
            "\",\"peer_available\":true,\"peer_identity_hex\":\"" +
            HexBytes(peer_.identity) + "\",\"target_router_identity_hex\":\"" +
            HexBytes(peer_.identity) + "\",\"target_router_identity_length\":" +
            std::to_string(peer_.identity.size()) +
            ",\"relay_started_monotonic_ns\":" +
            std::to_string(relay_started_monotonic_ns) +
            ",\"zmq_events\":" + std::to_string(zmq_events) +
            ",\"zmq_events_rc\":" + std::to_string(zmq_events_rc) +
            ",\"zmq_events_errno\":" + std::to_string(zmq_events_errno) +
            ",\"last_python_ready_monotonic_ns\":" +
            std::to_string(last_python_ready_monotonic_ns_) +
            ",\"router_send_frame_count\":2"
            ",\"router_send_frame_sizes\":[" +
            std::to_string(peer_.identity.size()) + "," +
            std::to_string(payload.size()) + "]"
            ",\"router_send_frame_0_identity_hex\":\"" +
            HexBytes(peer_.identity) +
            "\",\"router_send_frame_1_payload_hex\":\"" +
            HexBytes(payload) +
            "\",\"router_send_frame_0_more\":true"
            ",\"router_send_frame_1_more\":false"
            ",\"router_send_api_accepted\":" +
            std::string(router_send_api_accepted ? "true" : "false") +
            ",\"router_mandatory\":" + std::to_string(router_mandatory) +
            ",\"router_mandatory_available\":" +
            std::string(router_mandatory_available ? "true" : "false") +
            ",\"router_mandatory_rc\":" +
            std::to_string(router_mandatory_rc) +
            ",\"router_mandatory_errno\":" +
            std::to_string(router_mandatory_errno) +
            ",\"socket_send_hwm\":" + std::to_string(socket_send_hwm) +
            ",\"socket_send_hwm_rc\":" +
            std::to_string(socket_send_hwm_rc) +
            ",\"socket_send_hwm_errno\":" +
            std::to_string(socket_send_hwm_errno) +
            ",\"send_queue_depth_available\":false"
            ",\"identity_send_rc\":" +
            std::to_string(identity_rc) + ",\"identity_send_errno\":" +
            std::to_string(identity_errno) + ",\"payload_send_rc\":" +
            std::to_string(payload_rc) + ",\"payload_send_errno\":" +
            std::to_string(payload_errno) + "," +
            SocketThreadDiagnosticFields(socket_owner_thread_id_));
    if (identity_rc < 0) return false;
    if (payload_rc < 0) return false;
    if (relay_count_ != nullptr) ++(*relay_count_);
    int& attempts = unacknowledged_attempts_[
        PendingKey(stored.result.runtime_instance_id, stored.result.execution_id)];
    ++attempts;
    // The first delivery is not a retry.  A peer becomes stale only after
    // the configured number of retry deliveries remain unreceipted.
    if (attempts > stale_retry_threshold_) MarkStale();
    return true;
  }

 private:
  static std::string PendingKey(
      const std::string& runtime_instance_id, uint64_t execution_id) {
    return runtime_instance_id + "\n" + std::to_string(execution_id);
  }

  void* router_socket_ = nullptr;
  uint64_t* relay_count_ = nullptr;
  std::ofstream* diagnostics_stream_ = nullptr;
  PeerRegistration peer_;
  uint64_t next_connection_epoch_ = 0;
  std::vector<uint8_t> observed_identity_;
  std::map<std::string, int> unacknowledged_attempts_;
  int stale_retry_threshold_ = 3;
  int64_t last_python_ready_monotonic_ns_ = 0;
  std::string socket_owner_thread_id_;
};

class ZmqUnityCommandRelay : public command_broker::IUnityCommandRelay {
 public:
  void Configure(void* router_socket, uint64_t* send_count) {
    router_socket_ = router_socket;
    send_count_ = send_count;
  }

  void RegisterReady(
      const std::string& runtime_instance_id,
      const std::vector<uint8_t>& identity) {
    peers_[runtime_instance_id] = identity;
  }

  bool IdentityMatches(
      const std::string& runtime_instance_id,
      const std::vector<uint8_t>& identity) const {
    const auto found = peers_.find(runtime_instance_id);
    return found != peers_.end() && found->second == identity;
  }

  bool Send(const command_broker::StoredCommand& stored) override {
    const auto found = peers_.find(stored.command.runtime_instance_id);
    if (router_socket_ == nullptr || found == peers_.end() || found->second.empty())
      return false;
    const std::vector<uint8_t> payload =
        command_wire::SerializeCommand(stored.command);
    return SendRaw(stored.command.runtime_instance_id, payload);
  }

  bool SendRaw(
      const std::string& runtime_instance_id,
      const std::vector<uint8_t>& payload) {
    const auto found = peers_.find(runtime_instance_id);
    if (router_socket_ == nullptr || found == peers_.end() || found->second.empty())
      return false;
    const int identity_rc = zmq_send(
        router_socket_, found->second.data(), found->second.size(),
        ZMQ_SNDMORE | ZMQ_DONTWAIT);
    if (identity_rc != static_cast<int>(found->second.size())) return false;
    const int payload_rc = zmq_send(
        router_socket_, payload.data(), payload.size(), ZMQ_DONTWAIT);
    if (payload_rc != static_cast<int>(payload.size())) return false;
    if (send_count_ != nullptr) ++(*send_count_);
    return true;
  }

 private:
  void* router_socket_ = nullptr;
  uint64_t* send_count_ = nullptr;
  std::map<std::string, std::vector<uint8_t>> peers_;
};

class ZmqPythonCommandAckSink : public command_broker::IPythonCommandAckSink {
 public:
  void Configure(
      void* router_socket,
      uint64_t* ack_count,
      std::ofstream* diagnostics_stream,
      const std::string& socket_owner_thread_id) {
    router_socket_ = router_socket;
    ack_count_ = ack_count;
    diagnostics_stream_ = diagnostics_stream;
    socket_owner_thread_id_ = socket_owner_thread_id;
  }

  void SetIdentity(const std::vector<uint8_t>& identity) {
    identity_ = identity;
  }

  void SetConnectionEpoch(uint64_t connection_epoch) {
    connection_epoch_ = connection_epoch;
  }

  bool Send(const xp::v4::PrimitiveExecutionCommandReceiptAck& ack) override {
    const std::vector<uint8_t> payload = command_wire::SerializeReceipt(ack);
    std::array<uint8_t, SHA256_DIGEST_LENGTH> payload_digest{{}};
    SHA256(payload.data(), payload.size(), payload_digest.data());
    const std::string common =
        "\"message_type\":\"PrimitiveExecutionCommandReceiptAck\","
        "\"runtime_instance_id\":\"" + ack.runtime_instance_id +
        "\",\"execution_id\":" + std::to_string(ack.execution_id) +
        ",\"ack_status\":\"" + ack.ack_status +
        "\",\"command_sequence_hash\":\"" +
        HexBytes(ack.command_sequence_hash.data(), ack.command_sequence_hash.size()) +
        "\",\"payload_hash\":\"" +
        HexBytes(payload_digest.data(), payload_digest.size()) +
        "\",\"payload_size\":" + std::to_string(payload.size()) +
        ",\"peer_identity_hex\":\"" + HexBytes(identity_) +
        "\",\"peer_identity_length\":" + std::to_string(identity_.size()) +
        ",\"connection_epoch\":" + std::to_string(connection_epoch_) + ",";
    WritePythonDiagnostic(
        diagnostics_stream_, "\"event\":\"COMMAND_RECEIPT_SEND_BEGIN\"," +
            common + SocketThreadDiagnosticFields(socket_owner_thread_id_));
    if (router_socket_ == nullptr || identity_.empty()) {
      WritePythonDiagnostic(
          diagnostics_stream_, "\"event\":\"COMMAND_RECEIPT_SEND_FAILED\"," +
              common + "\"failure\":\"missing_socket_or_identity\"," +
              SocketThreadDiagnosticFields(socket_owner_thread_id_));
      return false;
    }
    errno = 0;
    const int identity_rc = zmq_send(
        router_socket_, identity_.data(), identity_.size(),
        ZMQ_SNDMORE | ZMQ_DONTWAIT);
    const int identity_errno = identity_rc < 0 ? errno : 0;
    if (identity_rc != static_cast<int>(identity_.size())) {
      WritePythonDiagnostic(
          diagnostics_stream_, "\"event\":\"COMMAND_RECEIPT_SEND_FAILED\"," +
              common + "\"failure\":\"identity_frame\",\"identity_send_rc\":" +
              std::to_string(identity_rc) +
              ",\"identity_send_errno\":" + std::to_string(identity_errno) +
              ",\"zmq_events\":" +
              std::to_string(ZmqSocketEvents(router_socket_)) + "," +
              SocketThreadDiagnosticFields(socket_owner_thread_id_));
      return false;
    }
    errno = 0;
    const int payload_rc = zmq_send(
        router_socket_, payload.data(), payload.size(), ZMQ_DONTWAIT);
    const int payload_errno = payload_rc < 0 ? errno : 0;
    const bool sent = payload_rc == static_cast<int>(payload.size());
    WritePythonDiagnostic(
        diagnostics_stream_,
        std::string("\"event\":\"") +
            (sent ? "COMMAND_RECEIPT_SEND_OK" : "COMMAND_RECEIPT_SEND_FAILED") +
            "\"," + common + "\"identity_send_rc\":" +
            std::to_string(identity_rc) + ",\"identity_send_errno\":" +
            std::to_string(identity_errno) + ",\"payload_send_rc\":" +
            std::to_string(payload_rc) + ",\"payload_send_errno\":" +
            std::to_string(payload_errno) + ",\"zmq_events\":" +
            std::to_string(ZmqSocketEvents(router_socket_)) + "," +
            SocketThreadDiagnosticFields(socket_owner_thread_id_));
    if (!sent) return false;
    if (ack_count_ != nullptr) ++(*ack_count_);
    return true;
  }

  bool SendRaw(const std::vector<uint8_t>& payload) {
    if (router_socket_ == nullptr || identity_.empty()) return false;
    const int identity_rc = zmq_send(
        router_socket_, identity_.data(), identity_.size(),
        ZMQ_SNDMORE | ZMQ_DONTWAIT);
    if (identity_rc != static_cast<int>(identity_.size())) return false;
    return zmq_send(router_socket_, payload.data(), payload.size(),
                    ZMQ_DONTWAIT) == static_cast<int>(payload.size());
  }

 private:
  void* router_socket_ = nullptr;
  uint64_t* ack_count_ = nullptr;
  std::vector<uint8_t> identity_;
  std::ofstream* diagnostics_stream_ = nullptr;
  std::string socket_owner_thread_id_;
  uint64_t connection_epoch_ = 0;
};

}  // namespace planning::transport
