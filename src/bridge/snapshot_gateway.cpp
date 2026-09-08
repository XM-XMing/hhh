#include <planning/bridge/snapshot_gateway.hpp>
#include <planning/transport/snapshot_transport.hpp>

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
namespace observation_broker = planning::observation_retrieval_broker;
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

SnapshotGateway::SnapshotGateway(planning::transport::SnapshotTransport& transport)
    : transport_(transport),
      observation_retrieval_broker_(
          observation_snapshot_cache_, observation_snapshot_requester_,
          observation_snapshot_ack_sink_, 100ULL) {
  observation_snapshot_requester_.Configure(transport_.observation_snapshot_router());
  observation_snapshot_ack_sink_.Configure(&observation_snapshot_requester_);
}

void SnapshotGateway::SetMetricsFlushCallback(std::function<void()> callback) {
  metrics_flush_callback_ = std::move(callback);
}

bool SnapshotGateway::ReceiveResult(
    const planning::xm_protocol::v4::PrimitiveExecutionResult& result,
    uint64_t now_ms) {
  if (!result.has_endpoint_observation_ref) return true;
  const auto outcome = observation_retrieval_broker_.ReceiveResult(result, now_ms);
  if (outcome == observation_broker::RequestOutcome::PROTOCOL_ERROR) {
    return false;
  }
  return true;
}

bool SnapshotGateway::FinalizeExecution(
    const planning::xm_protocol::v4::PrimitiveExecutionResult& result) {
  if (!result.has_endpoint_observation_ref) return true;
  xp::v4::SnapshotRequest request;
  request.observation_ref = observation_broker::ObservationRefFor(result);
  request.execution_id = result.execution_id;
  request.result_payload_hash = result.result_payload_hash;
  request.command_sequence_hash = result.command_sequence_hash;
  const bool finalized = observation_retrieval_broker_.FinalizeRequest(request);
  if (finalized) {
    pending_python_snapshot_deliveries_.erase(
        xp::v4::canonical_snapshot_request_bytes(request));
  }
  return finalized;
}

bool SnapshotGateway::FinalizeResetObservation(
    const planning::xm_protocol::v4::ObservationRef& observation_ref) {
  xp::v4::SnapshotRequest request;
  request.observation_ref = observation_ref;
  request.execution_id = 0;
  request.result_payload_hash.fill(0);
  request.command_sequence_hash.fill(0);
  const bool finalized = observation_retrieval_broker_.FinalizeRequest(request);
  if (finalized) {
    pending_python_snapshot_deliveries_.erase(
        xp::v4::canonical_snapshot_request_bytes(request));
  }
  return finalized;
}

void SnapshotGateway::Poll(uint64_t now_ms) {
  observation_retrieval_broker_.Poll(now_ms);
}

void SnapshotGateway::PollObservationSnapshots() {
  while (true) {
    zmq_msg_t identity_msg;
    zmq_msg_init(&identity_msg);
    const int identity_rc = zmq_msg_recv(
        &identity_msg, transport_.observation_snapshot_router(), ZMQ_DONTWAIT);
    if (identity_rc < 0) {
      zmq_msg_close(&identity_msg);
      if (errno != EAGAIN) {
        ++protocol_error_count_;
        ROS_WARN_THROTTLE(1.0, "snapshot identity recv failed: %s", zmq_strerror(errno));
      }
      return;
    }

    int more = 0;
    size_t more_size = sizeof(more);
    zmq_getsockopt(transport_.observation_snapshot_router(), ZMQ_RCVMORE, &more, &more_size);
    if (!more) {
      zmq_msg_close(&identity_msg);
      ++protocol_error_count_;
      ROS_ERROR_THROTTLE(1.0, "snapshot message missing payload frame");
      continue;
    }

    std::vector<uint8_t> identity(
        static_cast<const uint8_t*>(zmq_msg_data(&identity_msg)),
        static_cast<const uint8_t*>(zmq_msg_data(&identity_msg)) +
            zmq_msg_size(&identity_msg));
    zmq_msg_t payload_msg;
    zmq_msg_init(&payload_msg);
    const int payload_rc = zmq_msg_recv(
        &payload_msg, transport_.observation_snapshot_router(), 0);
    if (payload_rc < 0) {
      zmq_msg_close(&identity_msg);
      zmq_msg_close(&payload_msg);
      ++protocol_error_count_;
      ROS_WARN_THROTTLE(1.0, "snapshot payload recv failed: %s", zmq_strerror(errno));
      continue;
    }
    const char* data = static_cast<const char*>(zmq_msg_data(&payload_msg));
    const size_t size = zmq_msg_size(&payload_msg);
    std::string error;
    std::string runtime_instance_id;
    if (snapshot_wire::ParseReady(data, size, &runtime_instance_id, &error)) {
      observation_snapshot_requester_.RegisterReady(runtime_instance_id, identity);
      observation_retrieval_broker_.OnUnityReconnected(
          static_cast<uint64_t>(WallTimeNs() / 1000000LL));
      zmq_msg_close(&identity_msg);
      zmq_msg_close(&payload_msg);
      continue;
    }

    xp::v4::SnapshotRequest request;
    xp::v4::EndpointObservationSnapshot snapshot;
    if (snapshot_wire::ParseSnapshotResponse(data, size, &request, &snapshot, &error)) {
      ++snapshot_response_count_;
      if (!observation_snapshot_requester_.IdentityMatches(
              request.observation_ref.runtime_instance_id, identity)) {
        ++protocol_error_count_;
        ROS_ERROR_THROTTLE(1.0, "snapshot response peer identity mismatch");
      } else {
        const observation_broker::SnapshotOutcome outcome =
            observation_retrieval_broker_.ReceiveSnapshot(request, snapshot);
        if (outcome == observation_broker::SnapshotOutcome::PROTOCOL_ERROR) {
          ++protocol_error_count_;
          ROS_ERROR_THROTTLE(1.0, "snapshot response protocol error");
        } else if (outcome == observation_broker::SnapshotOutcome::STORED ||
                   outcome == observation_broker::SnapshotOutcome::DUPLICATE) {
          ++snapshot_hash_match_count_;
          RelayPendingPythonSnapshot(request);
        }
      }
      if (metrics_flush_callback_) metrics_flush_callback_();
      zmq_msg_close(&identity_msg);
      zmq_msg_close(&payload_msg);
      continue;
    }

    if (snapshot_wire::ParseMissing(data, size, &request, &error)) {
      if (!observation_snapshot_requester_.IdentityMatches(
              request.observation_ref.runtime_instance_id, identity) ||
          observation_retrieval_broker_.ReportMissingSnapshot(request) ==
              observation_broker::SnapshotOutcome::PROTOCOL_ERROR) {
        ++protocol_error_count_;
        ROS_ERROR_THROTTLE(1.0, "snapshot missing response protocol error");
      } else {
        RelayPendingPythonSnapshotMissing(request);
      }
      zmq_msg_close(&identity_msg);
      zmq_msg_close(&payload_msg);
      continue;
    }

    ++protocol_error_count_;
    ROS_ERROR_THROTTLE(1.0, "snapshot message rejected: %s", error.c_str());
    zmq_msg_close(&identity_msg);
    zmq_msg_close(&payload_msg);
  }
}

void SnapshotGateway::PollPythonSnapshotRequests() {
  while (true) {
    zmq_msg_t identity_msg;
    zmq_msg_init(&identity_msg);
    const int identity_rc = zmq_msg_recv(
        &identity_msg, transport_.python_snapshot_router(), ZMQ_DONTWAIT);
    if (identity_rc < 0) {
      zmq_msg_close(&identity_msg);
      if (errno != EAGAIN) ++protocol_error_count_;
      return;
    }
    int more = 0;
    size_t more_size = sizeof(more);
    zmq_getsockopt(transport_.python_snapshot_router(), ZMQ_RCVMORE, &more, &more_size);
    if (!more) {
      zmq_msg_close(&identity_msg);
      ++protocol_error_count_;
      continue;
    }
    std::vector<uint8_t> identity(
        static_cast<const uint8_t*>(zmq_msg_data(&identity_msg)),
        static_cast<const uint8_t*>(zmq_msg_data(&identity_msg)) +
            zmq_msg_size(&identity_msg));
    zmq_msg_t payload_msg;
    zmq_msg_init(&payload_msg);
    const int payload_rc = zmq_msg_recv(&payload_msg, transport_.python_snapshot_router(), 0);
    if (payload_rc < 0) {
      zmq_msg_close(&identity_msg);
      zmq_msg_close(&payload_msg);
      if (errno != EAGAIN) ++protocol_error_count_;
      continue;
    }
    xp::v4::SnapshotRequest request;
    std::string error;
    const bool valid = snapshot_wire::ParsePythonRequest(
        static_cast<const char*>(zmq_msg_data(&payload_msg)),
        zmq_msg_size(&payload_msg), &request, &error);
    std::vector<uint8_t> response;
    if (!valid) {
      ++protocol_error_count_;
    } else {
      const observation_broker::RequestOutcome outcome =
          observation_retrieval_broker_.ReceiveRequest(
              request, static_cast<uint64_t>(WallTimeNs() / 1000000LL));
      const bool terminal_duplicate =
          outcome == observation_broker::RequestOutcome::DUPLICATE &&
          observation_retrieval_broker_.IsTerminalRequest(request);
      if (outcome != observation_broker::RequestOutcome::PROTOCOL_ERROR &&
          !terminal_duplicate) {
        pending_python_snapshot_deliveries_[
            xp::v4::canonical_snapshot_request_bytes(request)] =
            PendingPythonSnapshotDelivery{request, identity};
      }
      xp::v4::EndpointObservationSnapshot snapshot;
      if (observation_retrieval_broker_.GetSnapshot(request, &snapshot)) {
        response = snapshot_wire::SerializePythonSnapshot(request, snapshot);
      }
    }
    if (!response.empty() && !identity.empty()) {
      const bool sent = SendPythonSnapshotWire(identity, response);
      if (sent) {
        pending_python_snapshot_deliveries_.erase(
            xp::v4::canonical_snapshot_request_bytes(request));
        FinalizeResetIfDelivered(request);
      }
    }
    zmq_msg_close(&identity_msg);
    zmq_msg_close(&payload_msg);
  }
}

void SnapshotGateway::RelayPendingPythonSnapshot(
    const xp::v4::SnapshotRequest& request) {
  const std::vector<uint8_t> key =
      xp::v4::canonical_snapshot_request_bytes(request);
  const std::map<std::vector<uint8_t>, PendingPythonSnapshotDelivery>::iterator
      found = pending_python_snapshot_deliveries_.find(key);
  if (found == pending_python_snapshot_deliveries_.end()) return;

  xp::v4::EndpointObservationSnapshot snapshot;
  if (!observation_retrieval_broker_.GetSnapshot(request, &snapshot)) return;
  const std::vector<uint8_t> response =
      snapshot_wire::SerializePythonSnapshot(request, snapshot);
  const bool sent = SendPythonSnapshotWire(found->second.identity, response);
  if (sent) {
    pending_python_snapshot_deliveries_.erase(found);
    FinalizeResetIfDelivered(request);
  }
}

void SnapshotGateway::RelayPendingPythonSnapshotMissing(
    const xp::v4::SnapshotRequest& request) {
  const std::vector<uint8_t> key =
      xp::v4::canonical_snapshot_request_bytes(request);
  const std::map<std::vector<uint8_t>, PendingPythonSnapshotDelivery>::iterator
      found = pending_python_snapshot_deliveries_.find(key);
  if (found == pending_python_snapshot_deliveries_.end()) return;
  const std::vector<uint8_t> response =
      snapshot_wire::SerializePythonMissing(request);
  const bool sent = SendPythonSnapshotWire(found->second.identity, response);
  if (sent) {
    pending_python_snapshot_deliveries_.erase(found);
    if (observation_broker::IsReservedResetRequest(request))
      FinalizeResetIfDelivered(request);
  }
}

bool SnapshotGateway::FinalizeResetIfDelivered(
    const planning::xm_protocol::v4::SnapshotRequest& request) {
  if (!observation_broker::IsReservedResetRequest(request)) return true;
  return FinalizeResetObservation(request.observation_ref);
}

bool SnapshotGateway::SendPythonSnapshotWire(
    const std::vector<uint8_t>& identity,
    const std::vector<uint8_t>& payload) {
  if (transport_.python_snapshot_router() == nullptr || identity.empty() ||
      payload.empty()) return false;
  const int identity_rc = zmq_send(
      transport_.python_snapshot_router(), identity.data(), identity.size(),
      ZMQ_SNDMORE | ZMQ_DONTWAIT);
  if (identity_rc != static_cast<int>(identity.size())) return false;
  return zmq_send(
             transport_.python_snapshot_router(), payload.data(), payload.size(),
             ZMQ_DONTWAIT) == static_cast<int>(payload.size());
}

}
