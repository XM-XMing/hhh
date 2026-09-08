#include <planning/bridge/result_gateway.hpp>
#include <planning/bridge/snapshot_gateway.hpp>
#include <planning/transport/result_transport.hpp>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <msgpack.hpp>
#include <openssl/sha.h>
#include <zmq.h>

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <ros/serialization.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <tf2/LinearMath/Quaternion.h>

#include <planning/bridge/bridge_util.hpp>
#include <planning/protocol/endpoint_observation_snapshot_wire.hpp>
#include <planning/protocol/primitive_execution_command_wire.hpp>
#include <planning/protocol/telemetry_wire.hpp>

namespace xp = planning::xm_protocol;
namespace result_broker = planning::primitive_execution_result_broker;
namespace result_wire = planning::primitive_execution_result_wire;
namespace command_wire = planning::primitive_execution_command_wire;
namespace snapshot_wire = planning::endpoint_observation_snapshot_wire;

using planning::bridge_util::CurrentThreadId;
using planning::bridge_util::HexBytes;
using planning::bridge_util::RuntimeInstanceIdFromDealerIdentity;
using planning::bridge_util::SocketThreadDiagnosticFields;
using planning::bridge_util::WallTimeNs;
using planning::bridge_util::WritePythonDiagnostic;
using planning::bridge_util::ZmqMonitorEventName;
using planning::bridge_util::ZmqSocketEvents;
using planning::bridge_util::PrimitiveCommandSequenceHashHex;
using planning::telemetry_wire::ParseDepthEndpointStateIdForAudit;
using planning::telemetry_wire::ParseDepthMeta;
using planning::telemetry_wire::ParseDynamicsState;

namespace planning::bridge {

ResultGateway::ResultGateway(
    planning::transport::ResultTransport& transport,
    SnapshotGateway& snapshot_gateway,
    ResultGatewayConfig config)
    : transport_(transport),
      snapshot_gateway_(snapshot_gateway),
      config_(std::move(config)),
      result_broker_(result_store_, result_ack_sink_, python_result_relay_) {
  python_result_router_owner_thread_id_ = CurrentThreadId();
  if (!config_.python_result_diagnostics_path.empty()) {
    python_result_diagnostics_stream_.open(
        config_.python_result_diagnostics_path, std::ios::out | std::ios::trunc);
    if (!python_result_diagnostics_stream_) {
      ROS_WARN("cannot open Python result diagnostics path: %s",
               config_.python_result_diagnostics_path.c_str());
    }
  }
  WritePythonDiagnostic(
      &python_result_diagnostics_stream_,
      "\"event\":\"python_result_router_socket_created\",\"socket_type\":\"ROUTER\",\"socket_creation_thread_id\":\"" +
          python_result_router_owner_thread_id_ + "\"," +
          SocketThreadDiagnosticFields(python_result_router_owner_thread_id_));
  if (!transport_.monitor_enabled()) {
    ROS_WARN("cannot enable Python result ROUTER monitor");
  }
  next_python_retry_ns_ = WallTimeNs() + PythonRetryIntervalNs();
  result_ack_sink_.Configure(transport_.result_router(), &result_ack_count_);
  python_result_relay_.Configure(
      transport_.python_result_router(), &python_relay_count_,
      &python_result_diagnostics_stream_, python_result_router_owner_thread_id_,
      config_.python_result_stale_retry_threshold);
}

ResultGateway::~ResultGateway() {
  if (python_result_diagnostics_stream_.is_open()) {
    python_result_diagnostics_stream_.flush();
    python_result_diagnostics_stream_.close();
  }
}

void ResultGateway::SetMetricsFlushCallback(std::function<void()> callback) {
  metrics_flush_callback_ = std::move(callback);
}

void ResultGateway::SetExecutionLifecycleFinalizer(
    ExecutionLifecycleFinalizer callback) {
  execution_lifecycle_finalizer_ = std::move(callback);
}

int64_t ResultGateway::PythonRetryIntervalNs() const {
  if (config_.python_result_retry_interval_s <= 0.0) return 0;
  return static_cast<int64_t>(config_.python_result_retry_interval_s * 1e9);
}

void ResultGateway::ScheduleNextPythonRetry() {
  next_python_retry_ns_ = WallTimeNs() + PythonRetryIntervalNs();
}

void ResultGateway::RetryPendingPythonResultsIfDue() {
  if (!python_connected_ || config_.python_result_retry_interval_s <= 0.0) return;
  const int64_t now_ns = WallTimeNs();
  if (now_ns < next_python_retry_ns_) return;
  result_broker_.RetryPendingPythonRelays();
  if (!python_result_relay_.is_ready()) python_connected_ = false;
  ScheduleNextPythonRetry();
}

void ResultGateway::SendPythonCommitAck(
    const result_wire::CommitMessage& commit,
    const std::vector<uint8_t>& identity,
    result_broker::CommitOutcome outcome) {
  const char* status = "PROTOCOL_ERROR";
  if (outcome == result_broker::CommitOutcome::COMMITTED) status = "COMMITTED";
  else if (outcome == result_broker::CommitOutcome::DUPLICATE) status = "DUPLICATE";
  const std::vector<uint8_t> payload = result_wire::SerializeCommitAck(commit, status);
  if (identity.empty()) return;
  const int identity_rc = zmq_send(
      transport_.python_result_router(), identity.data(), identity.size(),
      ZMQ_SNDMORE | ZMQ_DONTWAIT);
  int payload_rc = -1;
  if (identity_rc >= 0) {
    payload_rc = zmq_send(
        transport_.python_result_router(), payload.data(), payload.size(),
        ZMQ_DONTWAIT);
  }
  WritePythonDiagnostic(
      &python_result_diagnostics_stream_,
      "\"event\":\"commit_ack_send\",\"execution_id\":" +
          std::to_string(commit.key.execution_id) +
          ",\"peer_identity_hex\":\"" + HexBytes(identity) +
          "\",\"identity_send_rc\":" + std::to_string(identity_rc) +
          ",\"payload_send_rc\":" + std::to_string(payload_rc) + "," +
          SocketThreadDiagnosticFields(python_result_router_owner_thread_id_));
}

void ResultGateway::PollPythonTransportMonitor() {
  if (!transport_.python_result_monitor()) return;
  while (true) {
    zmq_msg_t message;
    zmq_msg_init(&message);
    const int rc = zmq_msg_recv(
        &message, transport_.python_result_monitor(), ZMQ_DONTWAIT);
    if (rc < 0) {
      zmq_msg_close(&message);
      if (errno != EAGAIN) {
        ROS_WARN_THROTTLE(1.0, "Python ROUTER monitor recv failed: %s",
                          zmq_strerror(errno));
      }
      return;
    }
    const size_t size = zmq_msg_size(&message);
    std::string endpoint;
    if (size >= 6) {
      const uint8_t* data = static_cast<const uint8_t*>(zmq_msg_data(&message));
      uint16_t event = 0;
      uint32_t value = 0;
      std::memcpy(&event, data, sizeof(event));
      std::memcpy(&value, data + sizeof(event), sizeof(value));

      int more = 0;
      size_t more_size = sizeof(more);
      if (zmq_getsockopt(
              transport_.python_result_monitor(), ZMQ_RCVMORE, &more, &more_size) == 0 &&
          more) {
        zmq_msg_t endpoint_message;
        zmq_msg_init(&endpoint_message);
        const int endpoint_rc = zmq_msg_recv(
            &endpoint_message, transport_.python_result_monitor(), ZMQ_DONTWAIT);
        if (endpoint_rc >= 0) {
          endpoint.assign(
              static_cast<const char*>(zmq_msg_data(&endpoint_message)),
              zmq_msg_size(&endpoint_message));
        }
        zmq_msg_close(&endpoint_message);
      }
      if (event == ZMQ_EVENT_DISCONNECTED) {
        python_connected_ = false;
        python_result_relay_.MarkStale();
      }
      WritePythonDiagnostic(
          &python_result_diagnostics_stream_,
          "\"event\":\"python_router_monitor\",\"monitor_event\":\"" +
              std::string(ZmqMonitorEventName(event)) +
              "\",\"monitor_event_id\":" + std::to_string(event) +
              ",\"monitor_value\":" + std::to_string(value) +
              ",\"endpoint\":\"" + endpoint + "\"," +
              SocketThreadDiagnosticFields(python_result_router_owner_thread_id_));
    }
    zmq_msg_close(&message);
  }
}

void ResultGateway::PollExecutionResults() {
  while (true) {
    zmq_msg_t identity_msg;
    zmq_msg_init(&identity_msg);
    const int identity_rc = zmq_msg_recv(
        &identity_msg, transport_.result_router(), ZMQ_DONTWAIT);
    if (identity_rc < 0) {
      zmq_msg_close(&identity_msg);
      if (errno != EAGAIN) {
        ++result_protocol_error_count_;
        ROS_WARN_THROTTLE(1.0, "execution result identity recv failed: %s",
                          zmq_strerror(errno));
      }
      return;
    }

    int more = 0;
    size_t more_size = sizeof(more);
    zmq_getsockopt(transport_.result_router(), ZMQ_RCVMORE, &more, &more_size);
    if (!more) {
      zmq_msg_close(&identity_msg);
      ++result_protocol_error_count_;
      ROS_ERROR_THROTTLE(1.0, "execution result missing payload frame");
      continue;
    }

    std::vector<uint8_t> identity(
        static_cast<const uint8_t*>(zmq_msg_data(&identity_msg)),
        static_cast<const uint8_t*>(zmq_msg_data(&identity_msg)) +
            zmq_msg_size(&identity_msg));
    zmq_msg_t payload_msg;
    zmq_msg_init(&payload_msg);
    const int payload_rc = zmq_msg_recv(&payload_msg, transport_.result_router(), 0);
    if (payload_rc < 0) {
      zmq_msg_close(&identity_msg);
      zmq_msg_close(&payload_msg);
      ++result_protocol_error_count_;
      ROS_ERROR_THROTTLE(1.0, "execution result payload recv failed: %s",
                         zmq_strerror(errno));
      continue;
    }

    xp::v4::PrimitiveExecutionResult result;
    std::string error;
    const bool parsed = result_wire::ParseResult(
        static_cast<const char*>(zmq_msg_data(&payload_msg)),
        zmq_msg_size(&payload_msg), &result, &error);
    zmq_msg_close(&identity_msg);
    zmq_msg_close(&payload_msg);
    if (!parsed) {
      ++result_protocol_error_count_;
      ROS_ERROR_THROTTLE(1.0, "execution result rejected: %s", error.c_str());
      continue;
    }

    result_ack_sink_.SetIdentity(identity);
    const size_t conflicts_before = result_broker_.conflict_count();
    const result_broker::ReceiveOutcome outcome = result_broker_.Receive(result);
    if (outcome == result_broker::ReceiveOutcome::ACCEPTED) {
      ++result_accepted_count_;
      if (result.has_endpoint_observation_ref &&
          !snapshot_gateway_.ReceiveResult(
              result, static_cast<uint64_t>(WallTimeNs() / 1000000LL))) {
        ++result_protocol_error_count_;
        ROS_ERROR_THROTTLE(1.0, "terminal observation retrieval request rejected");
      }
    } else if (outcome == result_broker::ReceiveOutcome::DUPLICATE) {
      ++result_duplicate_count_;
      if (result.has_endpoint_observation_ref &&
          !snapshot_gateway_.ReceiveResult(
              result, static_cast<uint64_t>(WallTimeNs() / 1000000LL))) {
        ++result_protocol_error_count_;
        ROS_ERROR_THROTTLE(1.0, "duplicate terminal observation retrieval rejected");
      }
    } else {
      if (result_broker_.conflict_count() > conflicts_before) {
        ++result_conflict_count_;
      }
      ++result_protocol_error_count_;
      ROS_ERROR_THROTTLE(1.0, "execution result protocol error");
    }
  }
}

void ResultGateway::PollPythonResults() {
  while (true) {
    zmq_msg_t identity_msg;
    zmq_msg_init(&identity_msg);
    const int identity_rc = zmq_msg_recv(
        &identity_msg, transport_.python_result_router(), ZMQ_DONTWAIT);
    if (identity_rc < 0) {
      zmq_msg_close(&identity_msg);
      if (errno != EAGAIN) {
        ++result_protocol_error_count_;
        ROS_WARN_THROTTLE(1.0, "Python result identity recv failed: %s",
                          zmq_strerror(errno));
      }
      return;
    }

    int more = 0;
    size_t more_size = sizeof(more);
    zmq_getsockopt(transport_.python_result_router(), ZMQ_RCVMORE, &more, &more_size);
    if (!more) {
      zmq_msg_close(&identity_msg);
      ++result_protocol_error_count_;
      ROS_ERROR_THROTTLE(1.0, "Python result missing payload frame");
      continue;
    }

    std::vector<uint8_t> identity(
        static_cast<const uint8_t*>(zmq_msg_data(&identity_msg)),
        static_cast<const uint8_t*>(zmq_msg_data(&identity_msg)) +
            zmq_msg_size(&identity_msg));
    zmq_msg_t payload_msg;
    zmq_msg_init(&payload_msg);
    const int rc = zmq_msg_recv(
        &payload_msg, transport_.python_result_router(), 0);
    if (rc < 0) {
      zmq_msg_close(&identity_msg);
      zmq_msg_close(&payload_msg);
      if (errno != EAGAIN) {
        ++result_protocol_error_count_;
        ROS_WARN_THROTTLE(1.0, "Python result recv failed: %s",
                          zmq_strerror(errno));
      }
      return;
    }

    const char* data = static_cast<const char*>(zmq_msg_data(&payload_msg));
    const size_t size = zmq_msg_size(&payload_msg);
    WritePythonDiagnostic(
        &python_result_diagnostics_stream_,
        "\"event\":\"python_router_recv\",\"peer_identity_hex\":\"" +
            HexBytes(identity) + "\",\"payload_size\":" +
            std::to_string(size) + "," +
            SocketThreadDiagnosticFields(python_result_router_owner_thread_id_));
    std::string error;
    const bool identity_changed = python_result_relay_.ObserveIdentity(identity);
    if (identity_changed) {
      WritePythonDiagnostic(
          &python_result_diagnostics_stream_,
          "\"event\":\"python_peer_identity_updated\",\"peer_identity_hex\":\"" +
              HexBytes(identity) + "\",\"peer_identity_length\":" +
              std::to_string(identity.size()) +
              ",\"runtime_instance_id_from_dealer_identity\":\"" +
              RuntimeInstanceIdFromDealerIdentity(identity) + "\"");
    }
    if (result_wire::ParseReady(data, size, &error)) {
      const int64_t ready_monotonic_ns = WallTimeNs();
      python_result_relay_.SetLastPythonReadyMonotonicNs(ready_monotonic_ns);
      const bool replaced_existing_identity =
          python_result_relay_.RegisterReady(identity);
      const planning::transport::ZmqPythonResultRelay::PeerRegistration& peer =
          python_result_relay_.peer();
      WritePythonDiagnostic(
          &python_result_diagnostics_stream_,
          "\"event\":\"python_peer_ready\",\"peer_identity_hex\":\"" +
              HexBytes(identity) + "\",\"peer_identity_length\":" +
              std::to_string(identity.size()) +
              ",\"runtime_instance_id_from_dealer_identity\":\"" +
              peer.runtime_instance_id +
              "\",\"ready_monotonic_ns\":" +
              std::to_string(ready_monotonic_ns) +
              ",\"connection_epoch\":" +
              std::to_string(peer.connection_epoch) +
              ",\"replaced_existing_identity\":" +
              std::string(replaced_existing_identity ? "true" : "false") +
              ",\"peer_available\":true," +
              SocketThreadDiagnosticFields(python_result_router_owner_thread_id_));
      python_connected_ = true;
      result_broker_.RetryPendingPythonRelays();
      if (!python_result_relay_.is_ready()) python_connected_ = false;
      ScheduleNextPythonRetry();
      zmq_msg_close(&identity_msg);
      zmq_msg_close(&payload_msg);
      continue;
    }

    result_wire::CommitMessage commit;
    xp::v4::PrimitiveExecutionResultReceiptAck receipt;
    if (result_wire::ParseReceiptAck(data, size, &receipt, &error)) {
      WritePythonDiagnostic(
          &python_result_diagnostics_stream_,
          "\"event\":\"python_receipt_ack_received\",\"execution_id\":" +
              std::to_string(receipt.execution_id) + "," +
              SocketThreadDiagnosticFields(python_result_router_owner_thread_id_));
      const result_broker::ReceiptOutcome outcome =
          result_broker_.AcknowledgePythonReceipt(receipt);
      if (outcome == result_broker::ReceiptOutcome::RECEIVED ||
          outcome == result_broker::ReceiptOutcome::DUPLICATE) {
        python_result_relay_.AcknowledgeReceipt(receipt);
      }
      if (outcome == result_broker::ReceiptOutcome::PROTOCOL_ERROR ||
          outcome == result_broker::ReceiptOutcome::NOT_FOUND) {
        ++result_protocol_error_count_;
        ROS_ERROR_THROTTLE(1.0, "Python result receipt ACK rejected");
      }
      zmq_msg_close(&identity_msg);
      zmq_msg_close(&payload_msg);
      continue;
    }
    if (!result_wire::ParseCommit(data, size, &commit, &error)) {
      ++result_protocol_error_count_;
      ROS_ERROR_THROTTLE(1.0, "Python result message rejected: %s", error.c_str());
      zmq_msg_close(&identity_msg);
      zmq_msg_close(&payload_msg);
      continue;
    }

    result_broker::StoredResult committed_result;
    const bool had_active_result = result_store_.get(commit.key, &committed_result);
    const result_broker::CommitOutcome outcome = result_broker_.CommitPython(
        commit.key, commit.result_payload_hash, &commit.command_sequence_hash);
    WritePythonDiagnostic(
        &python_result_diagnostics_stream_,
        "\"event\":\"python_commit_received\",\"execution_id\":" +
            std::to_string(commit.key.execution_id) +
            ",\"peer_identity_hex\":\"" + HexBytes(identity) + "\"," +
            SocketThreadDiagnosticFields(python_result_router_owner_thread_id_));
    SendPythonCommitAck(commit, identity, outcome);
    if (outcome == result_broker::CommitOutcome::COMMITTED && had_active_result) {
      python_result_relay_.ForgetResult(
          committed_result.key.runtime_instance_id,
          committed_result.key.execution_id);
    }
    if (outcome == result_broker::CommitOutcome::COMMITTED &&
        had_active_result && execution_lifecycle_finalizer_) {
      execution_lifecycle_finalizer_(committed_result.key, committed_result.result);
    }
    if (metrics_flush_callback_) metrics_flush_callback_();
    if (outcome == result_broker::CommitOutcome::PROTOCOL_ERROR ||
        outcome == result_broker::CommitOutcome::NOT_FOUND) {
      ++result_protocol_error_count_;
      ROS_ERROR_THROTTLE(1.0, "Python result commit rejected");
    }
    zmq_msg_close(&identity_msg);
    zmq_msg_close(&payload_msg);
  }
}

}
