#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Twist.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>

#include <planning/PrimitiveExecution.h>
#include <planning/bridge/primitive_execution_command_broker.hpp>
#include <planning/transport/reliable_zmq_adapters.hpp>

namespace planning::transport {
class CommandTransport;
}
namespace planning::bridge {
class ResetGateway;
class TelemetryBridge;

struct CommandGatewayConfig {
    std::string runtime_instance_id;
    double retry_interval_s = 0.5;
    std::string audit_path;
};

class CommandGateway : public planning::primitive_execution_command_broker::ICommandAuditSink {
public:
    CommandGateway(
        ros::NodeHandle& node,
        planning::transport::CommandTransport& transport,
        ResetGateway& reset_gateway,
        const TelemetryBridge& telemetry_bridge,
        CommandGatewayConfig config);
    ~CommandGateway();

    void PollReliablePrimitiveCommands();
    void RetryPendingPrimitiveCommandsIfDue();

    uint64_t command_count() const { return cmd_count_; }
    uint64_t protocol_error_count() const { return command_protocol_error_count_; }
    size_t accepted_count() const { return command_broker_.accepted_count(); }
    size_t forward_count() const { return command_broker_.forward_count(); }
    size_t receipt_count() const { return command_broker_.receipt_count(); }
    size_t duplicate_count() const { return command_broker_.duplicate_count(); }
    size_t receipt_duplicate_count() const { return command_broker_.receipt_duplicate_count(); }
    size_t retry_count() const { return command_broker_.retry_count(); }
    size_t conflict_count() const { return command_broker_.conflict_count(); }
    size_t broker_protocol_error_count() const { return command_broker_.protocol_error_count(); }
    size_t pending_count() const { return command_broker_.pending_count(); }
    bool FinalizeExecution(
        const std::string& runtime_instance_id,
        uint64_t execution_id,
        const std::array<uint8_t, 32>& command_sequence_hash);
    size_t current_command_entries() const { return command_store_.size(); }
    size_t peak_command_entries() const { return command_store_.peak_size(); }
    size_t terminal_command_count() const { return command_store_.terminal_count(); }

private:
    void CmdVelCallback(const geometry_msgs::TwistConstPtr& msg);
    void ResetPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
    void StopCallback(const std_msgs::BoolConstPtr& msg);
    void PrimitiveExecutionCallback(const planning::PrimitiveExecutionConstPtr& msg);
    void WritePrimitiveCommandRelayAudit(
        const char* event,
        const planning::PrimitiveExecution& message,
        const std::string& command_hash,
        size_t payload_size,
        bool has_send_result,
        int send_rc,
        int send_errno,
        int socket_events);
    static float YawFromQuat(const geometry_msgs::Quaternion& q);
    void SendCommand(
        int mode,
        const std::array<float, 4>& action,
        const std::array<float, 3>* position,
        int64_t callback_monotonic_ns = 0);
    bool ReceiveReliableMultipart(
        void* socket,
        std::vector<uint8_t>* identity,
        std::vector<uint8_t>* payload);
    void Record(
        const planning::primitive_execution_command_broker::CommandAuditRecord& record) override;

    ros::Subscriber cmd_sub_;
    ros::Subscriber primitive_execution_sub_;
    ros::Subscriber reset_sub_;
    ros::Subscriber stop_sub_;
    planning::transport::CommandTransport& transport_;
    ResetGateway& reset_gateway_;
    const TelemetryBridge& telemetry_bridge_;
    CommandGatewayConfig config_;
    std::ofstream command_audit_stream_;
    std::string python_command_router_owner_thread_id_;
    std::vector<uint8_t> python_command_ready_identity_;
    uint64_t python_command_connection_epoch_ = 0;
    int64_t cmd_id_ = 0;
    uint64_t cmd_count_ = 0;
    uint64_t command_forward_count_ = 0;
    uint64_t command_receipt_ack_count_ = 0;
    uint64_t command_protocol_error_count_ = 0;
    int64_t next_command_retry_ns_ = 0;
    planning::primitive_execution_command_broker::InMemoryCommandStore command_store_;
    planning::transport::ZmqUnityCommandRelay unity_command_relay_;
    planning::transport::ZmqPythonCommandAckSink python_command_ack_sink_;
    planning::primitive_execution_command_broker::PrimitiveExecutionCommandBroker command_broker_;
};

}
