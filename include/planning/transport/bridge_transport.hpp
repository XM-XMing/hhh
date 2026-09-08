#pragma once

#include <string>
#include <vector>

#include <planning/transport/command_transport.hpp>
#include <planning/transport/result_transport.hpp>
#include <planning/transport/snapshot_transport.hpp>
#include <planning/transport/telemetry_transport.hpp>

namespace planning::transport {

struct BridgeTransportConfig {
    std::string unity_host;
    std::string python_result_bind_host;
    std::string python_snapshot_bind_host;
    int command_port = 0;
    int state_port = 0;
    int depth_port = 0;
    int python_command_port = 0;
    int unity_command_port = 0;
    int execution_result_port = 0;
    int observation_snapshot_port = 0;
    int python_result_port = 0;
    int python_snapshot_port = 0;
    int receive_hwm = 4;
    int send_hwm = 4;
};

struct BridgeTransportEndpoints {
    std::string command;
    std::string state;
    std::string depth;
    std::string python_command;
    std::string unity_command;
    std::string result;
    std::string snapshot;
    std::string python_result;
    std::string python_snapshot;
};

class BridgeTransport final {
public:
    BridgeTransport() = default;
    ~BridgeTransport();

    BridgeTransport(const BridgeTransport&) = delete;
    BridgeTransport& operator=(const BridgeTransport&) = delete;

    bool Init(const BridgeTransportConfig& config, std::string* error);
    void Close();

    CommandTransport& command() { return command_; }
    const CommandTransport& command() const { return command_; }
    TelemetryTransport& telemetry() { return telemetry_; }
    const TelemetryTransport& telemetry() const { return telemetry_; }
    ResultTransport& result() { return result_; }
    const ResultTransport& result() const { return result_; }
    SnapshotTransport& snapshot() { return snapshot_; }
    const SnapshotTransport& snapshot() const { return snapshot_; }
    const BridgeTransportEndpoints& endpoints() const { return endpoints_; }
    const std::vector<std::string>& degraded_socket_options() const {
        return degraded_socket_options_;
    }

private:
    void* context_ = nullptr;
    CommandTransport command_;
    TelemetryTransport telemetry_;
    ResultTransport result_;
    SnapshotTransport snapshot_;
    BridgeTransportEndpoints endpoints_;
    std::vector<std::string> degraded_socket_options_;
};

}
