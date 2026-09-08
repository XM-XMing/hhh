#include <planning/bridge/command_gateway.hpp>
#include <planning/bridge/reset_gateway.hpp>
#include <planning/bridge/telemetry_bridge.hpp>
#include <planning/transport/command_transport.hpp>

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
#include <ros/serialization.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <tf2/LinearMath/Quaternion.h>

#include <planning/bridge/bridge_util.hpp>
#include <planning/protocol/endpoint_observation_snapshot_wire.hpp>
#include <planning/protocol/primitive_execution_command_wire.hpp>
#include <planning/protocol/telemetry_wire.hpp>

namespace xp = planning::xm_protocol;
namespace command_broker = planning::primitive_execution_command_broker;
namespace primitive_reset_wire = planning::primitive_reset_wire;
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

CommandGateway::CommandGateway(
    ros::NodeHandle& node,
    planning::transport::CommandTransport& transport,
    ResetGateway& reset_gateway,
    const TelemetryBridge& telemetry_bridge,
    CommandGatewayConfig config)
    : transport_(transport),
      reset_gateway_(reset_gateway),
      telemetry_bridge_(telemetry_bridge),
      config_(std::move(config)),
      command_broker_(
          command_store_, unity_command_relay_, python_command_ack_sink_,
          config_.runtime_instance_id, this) {
  python_command_router_owner_thread_id_ = CurrentThreadId();
  if (!config_.audit_path.empty()) {
    command_audit_stream_.open(
        config_.audit_path, std::ios::out | std::ios::trunc);
    if (!command_audit_stream_) {
      ROS_WARN("[DEBUG-CMD-TICK-c83e] cannot open command audit path: %s",
               config_.audit_path.c_str());
    }
  }
  unity_command_relay_.Configure(
      transport_.unity_command_router(), &command_forward_count_);
  python_command_ack_sink_.Configure(
      transport_.python_command_router(), &command_receipt_ack_count_,
      &command_audit_stream_, python_command_router_owner_thread_id_);
  next_command_retry_ns_ = WallTimeNs();

  cmd_sub_ = node.subscribe(
      "/xm/cmd_vel", 10, &CommandGateway::CmdVelCallback, this);
  primitive_execution_sub_ = node.subscribe(
      "/xm/primitive_execution", 1,
      &CommandGateway::PrimitiveExecutionCallback, this);
  reset_sub_ = node.subscribe(
      "/xm/reset_pose", 2, &CommandGateway::ResetPoseCallback, this);
  stop_sub_ = node.subscribe(
      "/xm/stop", 2, &CommandGateway::StopCallback, this);
}

CommandGateway::~CommandGateway() {
  if (command_audit_stream_.is_open()) {
    command_audit_stream_.flush();
    command_audit_stream_.close();
  }
}

void CommandGateway::CmdVelCallback(const geometry_msgs::TwistConstPtr& msg) {
  const int64_t callback_monotonic_ns = WallTimeNs();
  std::array<float, 4> action{{
      static_cast<float>(msg->linear.x),
      static_cast<float>(msg->linear.y),
      static_cast<float>(msg->linear.z),
      static_cast<float>(msg->angular.z)}};
  SendCommand(xp::kModeTrajectory, action, nullptr, callback_monotonic_ns);
}

void CommandGateway::ResetPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  std::array<float, 4> action{{0.0f, 0.0f, 0.0f, YawFromQuat(msg->pose.orientation)}};
  std::array<float, 3> pos{{
      static_cast<float>(msg->pose.position.x),
      static_cast<float>(msg->pose.position.y),
      static_cast<float>(msg->pose.position.z)}};
  SendCommand(xp::kModeTeleport, action, &pos);
}

void CommandGateway::StopCallback(const std_msgs::BoolConstPtr& msg) {
  if (!msg->data) return;
  std::array<float, 4> action{{0.0f, 0.0f, 0.0f, 0.0f}};
  SendCommand(xp::kModeTrajectory, action, nullptr);
}

void CommandGateway::WritePrimitiveCommandRelayAudit(
    const char* event,
    const planning::PrimitiveExecution& message,
    const std::string& command_hash,
    size_t payload_size,
    bool has_send_result,
    int send_rc,
    int send_errno,
    int socket_events) {
  if (!command_audit_stream_) return;
  command_audit_stream_ << std::setprecision(9)
      << "{\"contract_id\":\"[DEBUG-CMD-RELAY-4a2f]bridge_command_relay\""
      << ",\"event\":\"" << event << '"'
      << ",\"execution_id\":" << message.execution_id
      << ",\"command_sequence_hash\":\"" << command_hash << '"'
      << ",\"payload_size\":" << payload_size
      << ",\"monotonic_ns\":" << WallTimeNs()
      << ",\"thread_id\":\"" << CurrentThreadId() << '"'
      << ",\"python_peer_identity\":\"\""
      << ",\"unity_command_socket_events\":" << socket_events
      << ",\"zmq_send_return\":";
  if (has_send_result) command_audit_stream_ << send_rc;
  else command_audit_stream_ << "null";
  command_audit_stream_
      << ",\"errno\":" << (has_send_result ? send_errno : 0)
      << "}\n";
  command_audit_stream_.flush();
}

void CommandGateway::PrimitiveExecutionCallback(const planning::PrimitiveExecutionConstPtr& msg) {
  if (!transport_.command_pub()) return;
  if (msg->execution_id < 0 || msg->frame_count <= 0 ||
      msg->command_ids.size() != static_cast<size_t>(msg->frame_count) ||
      msg->commands.size() != static_cast<size_t>(msg->frame_count)) {
    ROS_ERROR("rejecting invalid primitive execution: execution_id=%lld frame_count=%d command_ids=%zu commands=%zu",
              static_cast<long long>(msg->execution_id), msg->frame_count,
              msg->command_ids.size(), msg->commands.size());
    return;
  }

  const std::string command_hash = PrimitiveCommandSequenceHashHex(*msg);
  const size_t received_payload_size = ros::serialization::serializationLength(*msg);
  WritePrimitiveCommandRelayAudit(
      "BRIDGE_COMMAND_RECEIVED", *msg, command_hash, received_payload_size,
      false, 0, 0, ZmqSocketEvents(transport_.command_pub()));

  const int64_t envelope_command_id = ++cmd_id_;
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> pk(&buffer);
  pk.pack_array(xp::kCommandFieldCount);
  pk.pack(xp::kSchemaVersion);
  pk.pack(xp::kModePrimitiveExecution);
  pk.pack_nil();
  pk.pack_nil();
  pk.pack(WallTimeNs());
  pk.pack(envelope_command_id);
  pk.pack(msg->execution_id);
  pk.pack(-1);
  pk.pack(msg->frame_count);
  pk.pack_array(static_cast<uint32_t>(msg->frame_count));
  for (int32_t i = 0; i < msg->frame_count; ++i) {
    const auto& command = msg->commands[static_cast<size_t>(i)];
    pk.pack_array(3);
    pk.pack(i);
    pk.pack(msg->command_ids[static_cast<size_t>(i)]);
    pk.pack_array(4);
    pk.pack(static_cast<float>(command.linear.x));
    pk.pack(static_cast<float>(command.linear.y));
    pk.pack(static_cast<float>(command.linear.z));
    pk.pack(static_cast<float>(command.angular.z));
  }

  WritePrimitiveCommandRelayAudit(
      "BRIDGE_COMMAND_FORWARD_ATTEMPT", *msg, command_hash, buffer.size(),
      false, 0, 0, ZmqSocketEvents(transport_.command_pub()));
  errno = 0;
  const int rc = zmq_send(transport_.command_pub(), buffer.data(), buffer.size(), ZMQ_DONTWAIT);
  const int send_errno = rc >= 0 ? 0 : errno;
  const int socket_events = ZmqSocketEvents(transport_.command_pub());
  if (rc >= 0) {
    ++cmd_count_;
    WritePrimitiveCommandRelayAudit(
        "BRIDGE_COMMAND_FORWARD_OK", *msg, command_hash, buffer.size(),
        true, rc, send_errno, socket_events);
  } else {
    WritePrimitiveCommandRelayAudit(
        "BRIDGE_COMMAND_FORWARD_FAILED", *msg, command_hash, buffer.size(),
        true, rc, send_errno, socket_events);
    ROS_ERROR("primitive execution send failed: execution_id=%lld error=%s",
              static_cast<long long>(msg->execution_id), zmq_strerror(errno));
  }
}

float CommandGateway::YawFromQuat(const geometry_msgs::Quaternion& q) {
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return static_cast<float>(std::atan2(siny_cosp, cosy_cosp));
}

void CommandGateway::SendCommand(int mode, const std::array<float, 4>& action,
                 const std::array<float, 3>* position,
                 int64_t callback_monotonic_ns) {
  if (!transport_.command_pub()) return;

  const int64_t command_id = ++cmd_id_;

  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> pk(&buffer);
  pk.pack_array(xp::kCommandFieldCount);
  pk.pack(xp::kSchemaVersion);
  pk.pack(mode);
  pk.pack_array(4);
  for (float v : action) pk.pack(v);
  if (position) {
    pk.pack_array(3);
    for (float v : *position) pk.pack(v);
  } else {
    pk.pack_nil();
  }
  pk.pack(WallTimeNs());
  pk.pack(command_id);
  pk.pack(-1);
  pk.pack(-1);
  pk.pack(0);
  pk.pack_nil();

  const int64_t send_started_monotonic_ns = WallTimeNs();
  const int rc = zmq_send(transport_.command_pub(), buffer.data(), buffer.size(), ZMQ_DONTWAIT);
  const int64_t send_completed_monotonic_ns = WallTimeNs();
  if (rc >= 0) {
    ++cmd_count_;
  } else if (errno != EAGAIN) {
    ROS_WARN_THROTTLE(1.0, "Unity command send failed: %s", zmq_strerror(errno));
  }
  if (command_audit_stream_) {
    command_audit_stream_ << std::setprecision(9)
        << "{\"contract_id\":\"[DEBUG-CMD-TICK-c83e]bridge_command_audit\""
        << ",\"command_id\":" << command_id
        << ",\"mode\":" << mode
        << ",\"callback_monotonic_ns\":" << callback_monotonic_ns
        << ",\"send_started_monotonic_ns\":" << send_started_monotonic_ns
        << ",\"send_completed_monotonic_ns\":" << send_completed_monotonic_ns
        << ",\"latest_unity_state_id\":" << telemetry_bridge_.latest_unity_state_id()
        << ",\"latest_unity_sim_time_ns\":" << telemetry_bridge_.latest_unity_sim_time_ns()
        << ",\"send_rc\":" << rc
        << ",\"action\":[" << action[0] << ',' << action[1] << ','
        << action[2] << ',' << action[3] << "]}\n";
    command_audit_stream_.flush();
  }
}

bool CommandGateway::ReceiveReliableMultipart(
    void* socket,
    std::vector<uint8_t>* identity,
    std::vector<uint8_t>* payload) {
  zmq_msg_t identity_msg;
  zmq_msg_init(&identity_msg);
  const int identity_rc = zmq_msg_recv(&identity_msg, socket, ZMQ_DONTWAIT);
  if (identity_rc < 0) {
    zmq_msg_close(&identity_msg);
    if (errno == EAGAIN) return false;
    ++command_protocol_error_count_;
    return false;
  }
  int more = 0;
  size_t more_size = sizeof(more);
  if (zmq_getsockopt(socket, ZMQ_RCVMORE, &more, &more_size) != 0 || !more) {
    zmq_msg_close(&identity_msg);
    ++command_protocol_error_count_;
    return true;
  }
  identity->assign(
      static_cast<const uint8_t*>(zmq_msg_data(&identity_msg)),
      static_cast<const uint8_t*>(zmq_msg_data(&identity_msg)) +
          zmq_msg_size(&identity_msg));
  zmq_msg_t payload_msg;
  zmq_msg_init(&payload_msg);
  const int payload_rc = zmq_msg_recv(&payload_msg, socket, 0);
  if (payload_rc < 0) {
    zmq_msg_close(&identity_msg);
    zmq_msg_close(&payload_msg);
    ++command_protocol_error_count_;
    return true;
  }
  payload->assign(
      static_cast<const uint8_t*>(zmq_msg_data(&payload_msg)),
      static_cast<const uint8_t*>(zmq_msg_data(&payload_msg)) +
          zmq_msg_size(&payload_msg));
  zmq_msg_close(&identity_msg);
  zmq_msg_close(&payload_msg);
  return true;
}

void CommandGateway::PollReliablePrimitiveCommands() {
  while (true) {
    std::vector<uint8_t> identity;
    std::vector<uint8_t> payload;
    if (!ReceiveReliableMultipart(
            transport_.python_command_router(), &identity, &payload)) break;
    python_command_ack_sink_.SetIdentity(identity);
    std::string ready_runtime;
    std::string error;
    if (command_wire::ParseReady(
            reinterpret_cast<const char*>(payload.data()), payload.size(),
            &ready_runtime, &error)) {
      python_command_ready_identity_ = identity;
      ++python_command_connection_epoch_;
      python_command_ack_sink_.SetConnectionEpoch(
          python_command_connection_epoch_);
      WritePythonDiagnostic(
          &command_audit_stream_,
          "\"event\":\"COMMAND_PYTHON_PEER_READY\","
          "\"message_type\":\"PrimitiveExecutionCommandReady\","
          "\"runtime_instance_id\":\"" + ready_runtime +
              "\",\"peer_identity_hex\":\"" + HexBytes(identity) +
              "\",\"peer_identity_length\":" +
              std::to_string(identity.size()) +
              ",\"connection_epoch\":" +
              std::to_string(python_command_connection_epoch_) + "," +
              SocketThreadDiagnosticFields(
                  python_command_router_owner_thread_id_));
      reset_gateway_.OnPythonReady(identity);
      continue;
    }
    primitive_reset_wire::ResetReceivedAck reset_ack;
    if (primitive_reset_wire::ParseReceivedAck(
            reinterpret_cast<const char*>(payload.data()), payload.size(),
            &reset_ack, &error)) {
      reset_gateway_.ReceivePythonResetAck(payload);
      continue;
    }
    primitive_reset_wire::ResetRequest reset_request;
    if (primitive_reset_wire::ParseRequest(
            reinterpret_cast<const char*>(payload.data()), payload.size(),
            &reset_request, &error)) {
      reset_gateway_.ReceivePythonReset(identity, payload);
      continue;
    }
    xp::v4::PrimitiveExecutionCommand command;
    if (!command_wire::ParseCommand(
            reinterpret_cast<const char*>(payload.data()), payload.size(),
            &command, &error)) {
      ++command_protocol_error_count_;
      continue;
    }
    const command_broker::CommandReceiveOutcome outcome =
        command_broker_.Receive(command);
    if (outcome == command_broker::CommandReceiveOutcome::PROTOCOL_ERROR ||
        outcome == command_broker::CommandReceiveOutcome::REJECTED) {
      xp::v4::PrimitiveExecutionCommandReceiptAck rejection =
          command_broker::ReceiptFor(
              command,
              outcome == command_broker::CommandReceiveOutcome::REJECTED
                  ? "REJECTED"
                  : "PROTOCOL_ERROR");
      python_command_ack_sink_.Send(rejection);
    }
  }

  while (true) {
    std::vector<uint8_t> identity;
    std::vector<uint8_t> payload;
    if (!ReceiveReliableMultipart(
            transport_.unity_command_router(), &identity, &payload)) break;
    std::string runtime_id;
    std::string error;
    if (command_wire::ParseReady(
            reinterpret_cast<const char*>(payload.data()), payload.size(),
            &runtime_id, &error)) {
      unity_command_relay_.RegisterReady(runtime_id, identity);
      reset_gateway_.OnUnityReady(runtime_id, identity);
      continue;
    }
    primitive_reset_wire::ResetComplete reset_complete;
    if (primitive_reset_wire::ParseComplete(
            reinterpret_cast<const char*>(payload.data()), payload.size(),
            &reset_complete, &error)) {
      reset_gateway_.ReceiveUnityResetComplete(identity, payload);
      continue;
    }
    xp::v4::PrimitiveExecutionCommandReceiptAck receipt;
    if (!command_wire::ParseReceipt(
            reinterpret_cast<const char*>(payload.data()), payload.size(),
            &receipt, &error)) {
      ++command_protocol_error_count_;
      continue;
    }
    const command_broker::CommandReceiptOutcome outcome =
        command_broker_.HandleUnityReceipt(receipt);
    if (outcome == command_broker::CommandReceiptOutcome::PROTOCOL_ERROR)
      ++command_protocol_error_count_;
  }
}

void CommandGateway::RetryPendingPrimitiveCommandsIfDue() {
  const int64_t now_ns = WallTimeNs();
  if (now_ns < next_command_retry_ns_) return;
  command_broker_.RetryPendingCommands();
  command_broker_.RetryPythonAcks();
  reset_gateway_.RetryPendingResets();
  next_command_retry_ns_ = now_ns +
      static_cast<int64_t>(config_.retry_interval_s * 1000000000.0);
}

bool CommandGateway::FinalizeExecution(
    const std::string& runtime_instance_id,
    uint64_t execution_id,
    const std::array<uint8_t, 32>& command_sequence_hash) {
  const command_broker::CommandKey key{runtime_instance_id, execution_id};
  return command_broker_.FinalizeExecution(key, command_sequence_hash);
}

void CommandGateway::Record(const command_broker::CommandAuditRecord& record) {
  if (!command_audit_stream_.is_open()) return;
  command_audit_stream_ << "{\"event\":\"" << record.event
                        << "\",\"runtime_instance_id\":\""
                        << record.runtime_instance_id
                        << "\",\"execution_id\":" << record.execution_id
                        << ",\"command_sequence_hash\":\""
                        << HexBytes(record.command_sequence_hash.data(),
                                    record.command_sequence_hash.size())
                        << "\",\"monotonic_ns\":" << record.monotonic_ns
                        << "}\n";
  command_audit_stream_.flush();
}

}
