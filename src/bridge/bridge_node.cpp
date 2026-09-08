#include <planning/bridge/unity_bridge_node.hpp>

#include <cstdlib>
#include <array>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <planning/bridge/bridge_util.hpp>
#include <planning/protocol/xm_protocol.hpp>

using planning::bridge_util::WallTimeNs;

namespace {

std::string JsonQuote(const std::string& value) {
  std::string quoted = "\"";
  for (const char character : value) {
    if (character == '\\' || character == '"') quoted += '\\';
    quoted += character;
  }
  quoted += '"';
  return quoted;
}

}

UnityBridgeNode::UnityBridgeNode() : nh_(""), pnh_("~") {
  pnh_.param<std::string>("unity_host", transport_config_.unity_host, "127.0.0.1");
  pnh_.param<int>("cmd_port", transport_config_.command_port, -1);
  pnh_.param<int>("python_command_port", transport_config_.python_command_port, -1);
  pnh_.param<int>("unity_command_port", transport_config_.unity_command_port, -1);
  pnh_.param<std::string>(
      "command_runtime_instance_id", command_config_.runtime_instance_id, "");
  reset_config_.runtime_instance_id = command_config_.runtime_instance_id;
  pnh_.param<double>("command_retry_interval_s", command_config_.retry_interval_s, 0.5);
  pnh_.param<int>("state_port", transport_config_.state_port, -1);
  pnh_.param<int>("depth_port", transport_config_.depth_port, -1);
  pnh_.param<int>("execution_result_port", transport_config_.execution_result_port, -1);
  pnh_.param<int>(
      "observation_snapshot_port", transport_config_.observation_snapshot_port, -1);
  pnh_.param<std::string>(
      "python_result_bind_host", transport_config_.python_result_bind_host,
      "127.0.0.1");
  pnh_.param<int>("python_result_port", transport_config_.python_result_port, -1);
  pnh_.param<std::string>(
      "python_snapshot_bind_host", transport_config_.python_snapshot_bind_host,
      "127.0.0.1");
  pnh_.param<int>("python_snapshot_port", transport_config_.python_snapshot_port, -1);

  if (command_config_.retry_interval_s <= 0.0)
    command_config_.retry_interval_s = 0.5;

  pnh_.param<std::string>("frame_id", telemetry_config_.frame_id, "map");
  pnh_.param<std::string>("base_frame_id", telemetry_config_.base_frame_id, "base_link");
  pnh_.param<std::string>(
      "camera_frame_id", telemetry_config_.camera_frame_id,
      "d435i_depth_optical_frame");
  pnh_.param<bool>("publish_tf", telemetry_config_.publish_tf, true);
  pnh_.param<double>("camera_offset_x", telemetry_config_.camera_offset_x, 0.0);
  pnh_.param<double>("camera_offset_y", telemetry_config_.camera_offset_y, 0.0);
  pnh_.param<double>("camera_offset_z", telemetry_config_.camera_offset_z, 0.0);
  pnh_.param<bool>(
      "flip_depth_vertical", telemetry_config_.flip_depth_vertical, false);
  pnh_.param<bool>(
      "flip_depth_horizontal", telemetry_config_.flip_depth_horizontal, false);
  pnh_.param<int>("recv_hwm", transport_config_.receive_hwm, 4);
  pnh_.param<int>("send_hwm", transport_config_.send_hwm, 4);
  telemetry_config_.receive_hwm = transport_config_.receive_hwm;
  telemetry_config_.send_hwm = transport_config_.send_hwm;
  pnh_.param<double>("spin_hz", spin_hz_, 200.0);
  pnh_.param<double>(
      "python_result_retry_interval_s",
      result_config_.python_result_retry_interval_s, 0.5);
  pnh_.param<int>(
      "python_result_stale_retry_threshold",
      result_config_.python_result_stale_retry_threshold, 3);

  pnh_.param<std::string>("command_audit_path", command_config_.audit_path, "");
  if (command_config_.audit_path.empty()) {
    const char* diagnostics_path = std::getenv("P0_M2_BRIDGE_COMMAND_AUDIT_PATH");
    if (diagnostics_path != nullptr) command_config_.audit_path = diagnostics_path;
  }
  pnh_.param<std::string>(
      "execution_transport_audit_path",
      telemetry_config_.execution_transport_audit_path, "");
  pnh_.param<std::string>(
      "execution_transport_audit_run_id",
      telemetry_config_.execution_transport_audit_run_id, "");
  pnh_.param<int>(
      "execution_transport_audit_episode_id",
      telemetry_config_.execution_transport_audit_episode_id, -1);
  pnh_.param<std::string>(
      "execution_transport_audit_unity_runtime_identity",
      telemetry_config_.execution_transport_audit_unity_runtime_identity, "");
  pnh_.param<std::string>(
      "telemetry_transport_audit_path",
      telemetry_config_.telemetry_transport_audit_path, "");
  pnh_.param<std::string>(
      "python_result_diagnostics_path",
      result_config_.python_result_diagnostics_path, "");
  if (result_config_.python_result_diagnostics_path.empty()) {
    const char* diagnostics_path =
        std::getenv("P0_M2_PYTHON_ROUTER_DIAGNOSTICS_PATH");
    if (diagnostics_path != nullptr) {
      result_config_.python_result_diagnostics_path = diagnostics_path;
    }
  }
  pnh_.param<std::string>(
      "execution_result_metrics_path", execution_result_metrics_path_, "");
}

UnityBridgeNode::~UnityBridgeNode() {
  FlushExecutionResultMetrics();
  if (telemetry_bridge_) telemetry_bridge_->FlushExecutionTransportAudit();

  result_gateway_.reset();
  command_gateway_.reset();
  snapshot_gateway_.reset();
  reset_gateway_.reset();
  telemetry_bridge_.reset();
  transport_.Close();
}

bool UnityBridgeNode::Init() {
  const std::array<int, 9> ports = {
      transport_config_.command_port,
      transport_config_.state_port,
      transport_config_.depth_port,
      transport_config_.python_command_port,
      transport_config_.unity_command_port,
      transport_config_.execution_result_port,
      transport_config_.observation_snapshot_port,
      transport_config_.python_result_port,
      transport_config_.python_snapshot_port,
  };
  std::set<int> unique_ports;
  for (const int port : ports) {
    if (port < 1 || port > 65535) {
      ROS_ERROR(
          "BRIDGE_PORT_CONTRACT_FAIL: every runtime port must be explicit and "
          "within TCP range; got %d",
          port);
      return false;
    }
    unique_ports.insert(port);
  }
  if (unique_ports.size() != ports.size()) {
    ROS_ERROR("BRIDGE_PORT_CONTRACT_FAIL: runtime port collision");
    return false;
  }
  if (command_config_.runtime_instance_id.empty()) {
    ROS_ERROR(
        "BRIDGE_PORT_CONTRACT_FAIL: command_runtime_instance_id is required");
    return false;
  }
  std::string transport_error;
  if (!transport_.Init(transport_config_, &transport_error)) {
    ROS_ERROR("%s", transport_error.c_str());
    return false;
  }
  executable_path_ = planning::bridge_util::ExecutablePath();
  executable_sha256_ = planning::bridge_util::FileSha256(executable_path_);
  for (const std::string& option : transport_.degraded_socket_options()) {
    ROS_WARN("BRIDGE_SOCKET_OPTION_DEGRADED: %s", option.c_str());
  }

  telemetry_bridge_ = std::make_unique<planning::bridge::TelemetryBridge>(
      nh_, transport_.telemetry(), telemetry_config_);
  reset_gateway_ = std::make_unique<planning::bridge::ResetGateway>(
      transport_.command(), reset_config_);
  snapshot_gateway_ = std::make_unique<planning::bridge::SnapshotGateway>(
      transport_.snapshot());
  command_gateway_ = std::make_unique<planning::bridge::CommandGateway>(
      nh_, transport_.command(), *reset_gateway_, *telemetry_bridge_, command_config_);
  result_gateway_ = std::make_unique<planning::bridge::ResultGateway>(
      transport_.result(), *snapshot_gateway_, result_config_);

  snapshot_gateway_->SetMetricsFlushCallback(
      [this]() { FlushExecutionResultMetrics(); });
  result_gateway_->SetMetricsFlushCallback(
      [this]() { FlushExecutionResultMetrics(); });
  result_gateway_->SetExecutionLifecycleFinalizer(
      [this](
          const planning::primitive_execution_result_broker::ResultKey& key,
          const planning::xm_protocol::v4::PrimitiveExecutionResult& result) {
        if (snapshot_gateway_) snapshot_gateway_->FinalizeExecution(result);
        if (command_gateway_) {
          command_gateway_->FinalizeExecution(
              key.runtime_instance_id, key.execution_id,
              result.command_sequence_hash);
        }
      });

  const auto& endpoints = transport_.endpoints();
  ROS_INFO(
      "Unity bridge connected: cmd=%s state=%s depth=%s python_cmd=%s unity_cmd=%s result=%s snapshot=%s python_bind=%s python_snapshot_bind=%s",
      endpoints.command.c_str(), endpoints.state.c_str(), endpoints.depth.c_str(),
      endpoints.python_command.c_str(), endpoints.unity_command.c_str(),
      endpoints.result.c_str(), endpoints.snapshot.c_str(),
      endpoints.python_result.c_str(), endpoints.python_snapshot.c_str());
  ROS_INFO(
      "BRIDGE_RUNTIME_IDENTITY: owner=%s implementation=%s binary_name=%s binary_path=%s bridge_sha256=%s protocol_schema_version=%u runtime_instance_id=%s resolved_ports={cmd:%d,state:%d,depth:%d,python_cmd:%d,unity_cmd:%d,result:%d,snapshot:%d,python_result:%d,python_snapshot:%d}",
      "BridgeNode", "split", "unity_bridge_node", executable_path_.c_str(),
      executable_sha256_.c_str(),
      static_cast<unsigned int>(planning::xm_protocol::v4::kSchemaVersion),
      command_config_.runtime_instance_id.c_str(),
      transport_config_.command_port, transport_config_.state_port,
      transport_config_.depth_port, transport_config_.python_command_port,
      transport_config_.unity_command_port,
      transport_config_.execution_result_port,
      transport_config_.observation_snapshot_port,
      transport_config_.python_result_port,
      transport_config_.python_snapshot_port);
  return true;
}

void UnityBridgeNode::Spin() {
  ros::Rate rate(spin_hz_);
  while (ros::ok()) {
    ros::spinOnce();
    command_gateway_->PollReliablePrimitiveCommands();
    result_gateway_->PollPythonTransportMonitor();
    result_gateway_->PollExecutionResults();
    snapshot_gateway_->PollObservationSnapshots();
    snapshot_gateway_->PollPythonSnapshotRequests();
    snapshot_gateway_->Poll(
        static_cast<uint64_t>(WallTimeNs() / 1000000LL));
    result_gateway_->PollPythonResults();
    result_gateway_->RetryPendingPythonResultsIfDue();
    command_gateway_->RetryPendingPrimitiveCommandsIfDue();
    telemetry_bridge_->PollState();
    telemetry_bridge_->PollDepth();
    ROS_INFO_THROTTLE(
        2.0,
        "Unity bridge counters: state=%llu depth=%llu cmd=%llu state_err=%llu depth_err=%llu",
        static_cast<unsigned long long>(telemetry_bridge_->state_count()),
        static_cast<unsigned long long>(telemetry_bridge_->depth_count()),
        static_cast<unsigned long long>(command_gateway_->command_count()),
        static_cast<unsigned long long>(telemetry_bridge_->state_error_count()),
        static_cast<unsigned long long>(telemetry_bridge_->depth_error_count()));
    rate.sleep();
  }
}

void UnityBridgeNode::FlushExecutionResultMetrics() {
  if (execution_result_metrics_path_.empty() || !result_gateway_ ||
      !command_gateway_ || !snapshot_gateway_ || !reset_gateway_) {
    return;
  }
  std::ofstream stream(
      execution_result_metrics_path_, std::ios::out | std::ios::trunc);
  if (!stream) {
    ROS_WARN(
        "cannot open execution result metrics path: %s",
        execution_result_metrics_path_.c_str());
    return;
  }

  const uint64_t bridge_protocol_errors =
      result_gateway_->protocol_error_count() +
      snapshot_gateway_->bridge_protocol_error_count() +
      reset_gateway_->protocol_error_count();
  stream << "{\"accepted\":" << result_gateway_->accepted_count()
         << ",\"duplicate\":" << result_gateway_->duplicate_count()
         << ",\"conflict\":" << result_gateway_->conflict_count()
         << ",\"pending\":" << result_gateway_->pending_count()
         << ",\"pending_receipt\":" << result_gateway_->pending_receipt_count()
         << ",\"commit\":" << result_gateway_->python_commit_count()
         << ",\"ack\":" << result_gateway_->ack_count()
         << ",\"python_relay\":" << result_gateway_->python_relay_count()
         << ",\"receipt\":" << result_gateway_->receipt_count()
         << ",\"duplicate_receipt\":"
         << result_gateway_->duplicate_receipt_count()
         << ",\"protocol_error\":" << bridge_protocol_errors
         << ",\"command_accepted_total\":" << command_gateway_->accepted_count()
         << ",\"command_forward_total\":" << command_gateway_->forward_count()
         << ",\"unity_command_receipt_total\":" << command_gateway_->receipt_count()
         << ",\"command_duplicate_total\":" << command_gateway_->duplicate_count()
         << ",\"command_receipt_duplicate_total\":"
         << command_gateway_->receipt_duplicate_count()
         << ",\"command_retry_total\":" << command_gateway_->retry_count()
         << ",\"command_conflict_total\":" << command_gateway_->conflict_count()
         << ",\"command_protocol_error_total\":"
         << command_gateway_->broker_protocol_error_count()
         << ",\"pending_command_final\":" << command_gateway_->pending_count()
         << ",\"result_cache_current_entries\":"
         << result_gateway_->current_result_entries()
         << ",\"result_cache_peak_entries\":"
         << result_gateway_->peak_result_entries()
         << ",\"result_cache_terminal_entries\":"
         << result_gateway_->terminal_result_count()
         << ",\"command_cache_current_entries\":"
         << command_gateway_->current_command_entries()
         << ",\"command_cache_peak_entries\":"
         << command_gateway_->peak_command_entries()
         << ",\"command_cache_terminal_entries\":"
         << command_gateway_->terminal_command_count()
         << ",\"reset_cache_current_entries\":"
         << reset_gateway_->current_reset_entries()
         << ",\"reset_cache_peak_entries\":"
         << reset_gateway_->peak_reset_entries()
         << ",\"reset_cache_terminal_entries\":"
         << reset_gateway_->terminal_reset_count();

  stream << ",\"degraded_socket_options\":[";
  const std::vector<std::string>& degraded_options =
      transport_.degraded_socket_options();
  for (size_t index = 0; index < degraded_options.size(); ++index) {
    if (index != 0) stream << ',';
    stream << JsonQuote(degraded_options[index]);
  }
  stream << ']';
  stream << ",\"bridge_runtime_identity\":{";
  stream << "\"owner\":" << JsonQuote("BridgeNode")
         << ",\"implementation\":" << JsonQuote("split")
         << ",\"binary_name\":" << JsonQuote("unity_bridge_node")
         << ",\"binary_path\":" << JsonQuote(executable_path_)
         << ",\"binary_sha256\":" << JsonQuote(executable_sha256_)
         << ",\"protocol_schema_version\":"
         << planning::xm_protocol::v4::kSchemaVersion
         << ",\"runtime_instance_id\":"
         << JsonQuote(command_config_.runtime_instance_id)
         << ",\"resolved_ports\":{";
  stream << "\"cmd_port\":" << transport_config_.command_port
         << ",\"state_port\":" << transport_config_.state_port
         << ",\"depth_port\":" << transport_config_.depth_port
         << ",\"python_command_port\":"
         << transport_config_.python_command_port
         << ",\"unity_command_port\":"
         << transport_config_.unity_command_port
         << ",\"execution_result_port\":"
         << transport_config_.execution_result_port
         << ",\"observation_snapshot_port\":"
         << transport_config_.observation_snapshot_port
         << ",\"python_result_port\":"
         << transport_config_.python_result_port
         << ",\"python_snapshot_port\":"
         << transport_config_.python_snapshot_port << "}}";

  if (snapshot_gateway_->response_count() > 0) {
    stream << ",\"snapshot_request\":" << snapshot_gateway_->request_count()
           << ",\"snapshot_response\":" << snapshot_gateway_->response_count()
           << ",\"snapshot_hash_match\":" << snapshot_gateway_->hash_match_count()
           << ",\"snapshot_cache_hit\":"
           << snapshot_gateway_->stored_snapshot_count()
           << ",\"snapshot_missing\":"
           << snapshot_gateway_->missing_snapshot_count()
           << ",\"pending_snapshot\":" << snapshot_gateway_->pending_count()
           << ",\"snapshot_cache_current_entries\":"
           << snapshot_gateway_->current_snapshot_entries()
           << ",\"snapshot_cache_peak_entries\":"
           << snapshot_gateway_->peak_snapshot_entries()
           << ",\"snapshot_cache_terminal_entries\":"
           << snapshot_gateway_->terminal_snapshot_count()
           << ",\"snapshot_protocol_error\":"
           << snapshot_gateway_->broker_protocol_error_count();
  }
  stream << "}\n";
}
