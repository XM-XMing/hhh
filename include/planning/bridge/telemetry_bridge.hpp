#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>

#include <planning/XMState.h>
#include <planning/protocol/xm_protocol.hpp>

namespace planning::transport {
class TelemetryTransport;
}

namespace planning::bridge {

struct TelemetryBridgeConfig {
    std::string frame_id = "map";
    std::string base_frame_id = "base_link";
    std::string camera_frame_id = "d435i_depth_optical_frame";
    bool publish_tf = true;
    bool flip_depth_vertical = false;
    bool flip_depth_horizontal = false;
    double camera_offset_x = 0.0;
    double camera_offset_y = 0.0;
    double camera_offset_z = 0.0;
    std::string telemetry_transport_audit_path;
    std::string execution_transport_audit_path;
    std::string execution_transport_audit_run_id;
    int execution_transport_audit_episode_id = -1;
    std::string execution_transport_audit_unity_runtime_identity;
    int receive_hwm = 4;
    int send_hwm = 4;
};

class TelemetryBridge {
public:
    TelemetryBridge(
        ros::NodeHandle& node,
        planning::transport::TelemetryTransport& transport,
        TelemetryBridgeConfig config);
    ~TelemetryBridge();

    void PollState();
    void PollDepth();
    void FlushExecutionTransportAudit();

    int64_t latest_unity_state_id() const { return latest_unity_state_id_; }
    int64_t latest_unity_sim_time_ns() const { return latest_unity_sim_time_ns_; }
    uint64_t state_count() const { return state_count_; }
    uint64_t depth_count() const { return depth_count_; }
    uint64_t state_error_count() const { return state_error_count_; }
    uint64_t depth_error_count() const { return depth_error_count_; }

private:
    struct ExecutionTransportAuditRecord {
        int64_t execution_id = -1;
        int32_t frame_index = -1;
        int64_t command_id = -1;
        int64_t state_id = -1;
        int64_t sim_time_ns = 0;
        int32_t execution_status = 0;
        int state_zmq_recv_rc = 0;
        uint64_t receive_sequence = 0;
        int64_t receive_monotonic_ns = 0;
        bool received = false;
        bool forward_attempted = false;
        bool forwarded = false;
        uint64_t forward_sequence = 0;
        int64_t forward_monotonic_ns = 0;
    };

    void RecordTelemetryReceive(
        const char* stream,
        uint64_t sequence,
        int64_t state_id,
        const std::string& depth_id,
        int64_t execution_id,
        int32_t frame_index,
        int meta_recv_rc,
        int payload_recv_rc,
        int64_t receive_monotonic_ns,
        bool decode_ok);
    void PublishState(const planning::xm_protocol::DynamicsState& state);
    void PublishTransforms(
        const ros::Time& stamp,
        const geometry_msgs::Point& position,
        const geometry_msgs::Quaternion& orientation);
    void PublishDepth(
        const planning::xm_protocol::DepthFrameMeta& meta,
        const uint8_t* payload,
        size_t payload_size);

    ros::Publisher state_pub_;
    ros::Publisher odom_pub_;
    ros::Publisher depth_pub_;
    ros::Publisher camera_info_pub_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    planning::transport::TelemetryTransport& transport_;
    TelemetryBridgeConfig config_;
    std::ofstream telemetry_transport_audit_stream_;

    int64_t latest_unity_state_id_ = -1;
    int64_t latest_unity_sim_time_ns_ = 0;
    uint64_t state_count_ = 0;
    uint64_t depth_count_ = 0;
    uint64_t state_error_count_ = 0;
    uint64_t depth_error_count_ = 0;
    uint64_t execution_transport_receive_sequence_ = 0;
    uint64_t telemetry_transport_receive_sequence_ = 0;
    uint64_t execution_transport_forward_sequence_ = 0;
    bool execution_transport_audit_overflow_ = false;
    std::vector<ExecutionTransportAuditRecord> execution_transport_audits_;
    static constexpr size_t kExecutionTransportAuditCapacity = 262144;
};

}
