#pragma once

#include <memory>
#include <string>

#include <ros/ros.h>

#include <planning/bridge/command_gateway.hpp>
#include <planning/bridge/reset_gateway.hpp>
#include <planning/bridge/result_gateway.hpp>
#include <planning/bridge/snapshot_gateway.hpp>
#include <planning/bridge/telemetry_bridge.hpp>
#include <planning/transport/bridge_transport.hpp>

class UnityBridgeNode {
public:
    UnityBridgeNode();
    ~UnityBridgeNode();

    bool Init();
    void Spin();

private:
    void FlushExecutionResultMetrics();

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    planning::transport::BridgeTransportConfig transport_config_;
    planning::bridge::CommandGatewayConfig command_config_;
    planning::bridge::ResetGatewayConfig reset_config_;
    planning::bridge::ResultGatewayConfig result_config_;
    planning::bridge::TelemetryBridgeConfig telemetry_config_;
    std::string execution_result_metrics_path_;
    std::string executable_path_;
    std::string executable_sha256_;
    double spin_hz_ = 200.0;

    planning::transport::BridgeTransport transport_;
    std::unique_ptr<planning::bridge::TelemetryBridge> telemetry_bridge_;
    std::unique_ptr<planning::bridge::ResetGateway> reset_gateway_;
    std::unique_ptr<planning::bridge::SnapshotGateway> snapshot_gateway_;
    std::unique_ptr<planning::bridge::CommandGateway> command_gateway_;
    std::unique_ptr<planning::bridge::ResultGateway> result_gateway_;
};
