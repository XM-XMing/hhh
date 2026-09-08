#include <planning/transport/bridge_transport.hpp>

#include <zmq.h>

namespace planning::transport {

BridgeTransport::~BridgeTransport() {
    Close();
}

bool BridgeTransport::Init(
    const BridgeTransportConfig& config,
    std::string* error) {
    Close();
    degraded_socket_options_.clear();
    context_ = zmq_ctx_new();
    if (context_ == nullptr) {
        if (error != nullptr) {
            *error = "zmq_ctx_new failed";
        }
        return false;
    }

    CommandTransportConfig command_config;
    command_config.unity_host = config.unity_host;
    command_config.command_port = config.command_port;
    command_config.python_command_port = config.python_command_port;
    command_config.unity_command_port = config.unity_command_port;
    command_config.receive_hwm = config.receive_hwm;
    command_config.send_hwm = config.send_hwm;

    TelemetryTransportConfig telemetry_config;
    telemetry_config.unity_host = config.unity_host;
    telemetry_config.state_port = config.state_port;
    telemetry_config.depth_port = config.depth_port;
    telemetry_config.receive_hwm = config.receive_hwm;

    ResultTransportConfig result_config;
    result_config.python_result_bind_host = config.python_result_bind_host;
    result_config.execution_result_port = config.execution_result_port;
    result_config.python_result_port = config.python_result_port;
    result_config.receive_hwm = config.receive_hwm;
    result_config.send_hwm = config.send_hwm;

    SnapshotTransportConfig snapshot_config;
    snapshot_config.python_snapshot_bind_host = config.python_snapshot_bind_host;
    snapshot_config.observation_snapshot_port = config.observation_snapshot_port;
    snapshot_config.python_snapshot_port = config.python_snapshot_port;
    snapshot_config.receive_hwm = config.receive_hwm;
    snapshot_config.send_hwm = config.send_hwm;

    if (!command_.Init(context_, command_config, error) ||
        !telemetry_.Init(context_, telemetry_config, error) ||
        !result_.Init(context_, result_config, error) ||
        !snapshot_.Init(context_, snapshot_config, error)) {
        Close();
        return false;
    }

    const auto append_degraded = [this](
        const std::vector<std::string>& options) {
        degraded_socket_options_.insert(
            degraded_socket_options_.end(), options.begin(), options.end());
    };
    const auto command_degraded = command_.degraded_socket_options();
    const auto telemetry_degraded = telemetry_.degraded_socket_options();
    const auto result_degraded = result_.degraded_socket_options();
    const auto snapshot_degraded = snapshot_.degraded_socket_options();
    append_degraded(command_degraded);
    append_degraded(telemetry_degraded);
    append_degraded(result_degraded);
    append_degraded(snapshot_degraded);

    const auto& command_endpoints = command_.endpoints();
    const auto& telemetry_endpoints = telemetry_.endpoints();
    const auto& result_endpoints = result_.endpoints();
    const auto& snapshot_endpoints = snapshot_.endpoints();
    endpoints_.command = command_endpoints.command;
    endpoints_.python_command = command_endpoints.python_command;
    endpoints_.unity_command = command_endpoints.unity_command;
    endpoints_.state = telemetry_endpoints.state;
    endpoints_.depth = telemetry_endpoints.depth;
    endpoints_.result = result_endpoints.result;
    endpoints_.python_result = result_endpoints.python_result;
    endpoints_.snapshot = snapshot_endpoints.snapshot;
    endpoints_.python_snapshot = snapshot_endpoints.python_snapshot;
    return true;
}

void BridgeTransport::Close() {
    snapshot_.Close();
    result_.Close();
    telemetry_.Close();
    command_.Close();
    endpoints_ = {};
    if (context_ != nullptr) {
        zmq_ctx_term(context_);
        context_ = nullptr;
    }
}

}
