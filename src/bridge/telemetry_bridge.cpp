#include <planning/bridge/telemetry_bridge.hpp>
#include <planning/transport/telemetry_transport.hpp>

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

namespace {
constexpr double kPi = 3.14159265358979323846;
}

namespace planning::bridge {

TelemetryBridge::TelemetryBridge(
    ros::NodeHandle& node,
    planning::transport::TelemetryTransport& transport,
    TelemetryBridgeConfig config)
    : transport_(transport), config_(std::move(config)) {
  state_pub_ = node.advertise<planning::XMState>("/xm/state", 10);
  odom_pub_ = node.advertise<nav_msgs::Odometry>("/xm/odom", 10);
  depth_pub_ = node.advertise<sensor_msgs::Image>("/xm/depth/image_raw", 2);
  camera_info_pub_ = node.advertise<sensor_msgs::CameraInfo>("/xm/depth/camera_info", 2);
  if (!config_.execution_transport_audit_path.empty()) {
    execution_transport_audits_.reserve(kExecutionTransportAuditCapacity);
  }
  if (!config_.telemetry_transport_audit_path.empty()) {
    telemetry_transport_audit_stream_.open(
        config_.telemetry_transport_audit_path, std::ios::out | std::ios::trunc);
    if (!telemetry_transport_audit_stream_) {
      ROS_WARN("[DEBUG-P0M2-TELEMETRY-e542] cannot open telemetry audit: %s",
               config_.telemetry_transport_audit_path.c_str());
    }
  }
}

TelemetryBridge::~TelemetryBridge() {
  if (telemetry_transport_audit_stream_.is_open()) {
    telemetry_transport_audit_stream_.flush();
    telemetry_transport_audit_stream_.close();
  }
}

void TelemetryBridge::PollState() {
  while (true) {
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    const int rc = zmq_msg_recv(&msg, transport_.state_sub(), ZMQ_DONTWAIT);
    if (rc < 0) {
      zmq_msg_close(&msg);
      if (errno != EAGAIN) ROS_WARN_THROTTLE(1.0, "state recv failed: %s", zmq_strerror(errno));
      return;
    }

    xp::DynamicsState state;
    std::string error;
    const bool ok = ParseDynamicsState(static_cast<const char*>(zmq_msg_data(&msg)), zmq_msg_size(&msg), &state, &error);
    const int64_t receive_monotonic_ns = WallTimeNs();
    zmq_msg_close(&msg);

    RecordTelemetryReceive(
        "state", ++telemetry_transport_receive_sequence_,
        ok ? state.state_id : -1, "", ok ? state.applied_execution_id : -1,
        ok ? state.applied_execution_frame_index : -1, rc, -1,
        receive_monotonic_ns, ok);

    if (!ok) {
      ++state_error_count_;
      ROS_WARN_THROTTLE(1.0, "state decode failed: %s", error.c_str());
      continue;
    }

    latest_unity_state_id_ = state.state_id;
    latest_unity_sim_time_ns_ = state.sim_time_ns;
    if (!config_.execution_transport_audit_path.empty() &&
        state.applied_execution_id >= 0) {
      ExecutionTransportAuditRecord audit;
      audit.execution_id = state.applied_execution_id;
      audit.frame_index = state.applied_execution_frame_index;
      audit.command_id = state.applied_command_id;
      audit.state_id = state.state_id;
      audit.sim_time_ns = state.sim_time_ns;
      audit.execution_status = state.execution_status;
      audit.state_zmq_recv_rc = rc;
      audit.receive_sequence = ++execution_transport_receive_sequence_;
      audit.receive_monotonic_ns = WallTimeNs();
      audit.received = true;
      audit.forward_attempted = true;
      PublishState(state);
      audit.forwarded = true;
      audit.forward_sequence = ++execution_transport_forward_sequence_;
      audit.forward_monotonic_ns = WallTimeNs();
      if (execution_transport_audits_.size() < kExecutionTransportAuditCapacity) {
        execution_transport_audits_.push_back(audit);
      } else {
        execution_transport_audit_overflow_ = true;
      }
    } else {
      PublishState(state);
    }
    ++state_count_;
  }
}

void TelemetryBridge::PollDepth() {
  while (true) {
    zmq_msg_t meta_msg;
    zmq_msg_init(&meta_msg);
    const int rc = zmq_msg_recv(&meta_msg, transport_.depth_sub(), ZMQ_DONTWAIT);
    if (rc < 0) {
      zmq_msg_close(&meta_msg);
      if (errno != EAGAIN) ROS_WARN_THROTTLE(1.0, "depth meta recv failed: %s", zmq_strerror(errno));
      return;
    }

    int more = 0;
    size_t more_size = sizeof(more);
    zmq_getsockopt(transport_.depth_sub(), ZMQ_RCVMORE, &more, &more_size);
    if (!more) {
      zmq_msg_close(&meta_msg);
      ++depth_error_count_;
      ROS_WARN_THROTTLE(1.0, "depth multipart missing payload");
      continue;
    }

    zmq_msg_t payload_msg;
    zmq_msg_init(&payload_msg);
    const int payload_rc = zmq_msg_recv(&payload_msg, transport_.depth_sub(), 0);
    if (payload_rc < 0) {
      zmq_msg_close(&meta_msg);
      zmq_msg_close(&payload_msg);
      ++depth_error_count_;
      ROS_WARN_THROTTLE(1.0, "depth payload recv failed: %s", zmq_strerror(errno));
      continue;
    }

    xp::DepthFrameMeta meta;
    std::string error;
    const bool ok = ParseDepthMeta(static_cast<const char*>(zmq_msg_data(&meta_msg)), zmq_msg_size(&meta_msg), &meta, &error);
    int64_t endpoint_state_id = -1;
    const bool endpoint_state_id_present = ok && ParseDepthEndpointStateIdForAudit(
        static_cast<const char*>(zmq_msg_data(&meta_msg)), zmq_msg_size(&meta_msg),
        &endpoint_state_id);
    const int64_t receive_monotonic_ns = WallTimeNs();
    RecordTelemetryReceive(
        "depth", ++telemetry_transport_receive_sequence_,
        endpoint_state_id_present ? endpoint_state_id : -1,
        ok ? "depth-" + std::to_string(meta.capture_id) : "",
        -1, -1, rc, payload_rc, receive_monotonic_ns, ok);
    if (!ok) {
      zmq_msg_close(&meta_msg);
      zmq_msg_close(&payload_msg);
      ++depth_error_count_;
      ROS_WARN_THROTTLE(1.0, "depth meta decode failed: %s", error.c_str());
      continue;
    }

    const size_t expected = static_cast<size_t>(meta.row_step) * static_cast<size_t>(meta.height);
    if (zmq_msg_size(&payload_msg) < expected) {
      ++depth_error_count_;
      ROS_WARN_THROTTLE(1.0, "depth payload too short: got=%zu expected=%zu", zmq_msg_size(&payload_msg), expected);
    } else {
      PublishDepth(meta, static_cast<const uint8_t*>(zmq_msg_data(&payload_msg)), expected);
      ++depth_count_;
    }

    zmq_msg_close(&meta_msg);
    zmq_msg_close(&payload_msg);
  }
}

void TelemetryBridge::RecordTelemetryReceive(
    const char* stream,
    uint64_t sequence,
    int64_t state_id,
    const std::string& depth_id,
    int64_t execution_id,
    int32_t frame_index,
    int meta_recv_rc,
    int payload_recv_rc,
    int64_t receive_monotonic_ns,
    bool decode_ok) {
  if (!telemetry_transport_audit_stream_) return;
  telemetry_transport_audit_stream_
      << "{\"contract_id\":\"[DEBUG-P0M2-TELEMETRY-e542]\""
      << ",\"side\":\"bridge_sub\""
      << ",\"stream\":\"" << stream << '\"'
      << ",\"sequence\":" << sequence
      << ",\"state_id\":" << state_id
      << ",\"depth_id\":";
  if (depth_id.empty()) telemetry_transport_audit_stream_ << "null";
  else telemetry_transport_audit_stream_ << '\"' << depth_id << '\"';
  telemetry_transport_audit_stream_
      << ",\"execution_id\":" << execution_id
      << ",\"frame_index\":" << frame_index
      << ",\"meta_zmq_recv_rc\":" << meta_recv_rc
      << ",\"payload_zmq_recv_rc\":" << payload_recv_rc
      << ",\"receive_monotonic_ns\":" << receive_monotonic_ns
      << ",\"decode_ok\":" << (decode_ok ? "true" : "false")
      << "}\n";
  telemetry_transport_audit_stream_.flush();
}

void TelemetryBridge::PublishState(const xp::DynamicsState& state) {
  ros::Time stamp = ros::Time::now();

  planning::XMState s;
  s.header.stamp = stamp;
  s.header.frame_id = config_.frame_id;
  s.state_id = state.state_id;
  s.sim_time_ns = state.sim_time_ns;
  s.applied_execution_id = state.applied_execution_id;
  s.applied_execution_frame_index = state.applied_execution_frame_index;
  s.applied_command_id = state.applied_command_id;
  s.execution_status = state.execution_status;
  s.flags = state.flags;
  s.collided = (state.flags & xp::kFlagCollision) != 0;
  s.altitude_violation = (state.flags & xp::kFlagAltitudeViolation) != 0;
  s.min_clearance = state.min_clearance;
  for (size_t i = 0; i < 3; ++i) s.front_clearances[i] = state.front_clearances[i];
  s.position.x = state.pos[0];
  s.position.y = state.pos[1];
  s.position.z = state.pos[2];
  s.orientation.x = state.rot[0];
  s.orientation.y = state.rot[1];
  s.orientation.z = state.rot[2];
  s.orientation.w = state.rot[3];
  s.velocity.x = state.vel[0];
  s.velocity.y = state.vel[1];
  s.velocity.z = state.vel[2];
  s.acceleration.x = state.acc[0];
  s.acceleration.y = state.acc[1];
  s.acceleration.z = state.acc[2];
  state_pub_.publish(s);

  nav_msgs::Odometry odom;
  odom.header = s.header;
  odom.child_frame_id = config_.base_frame_id;
  odom.pose.pose.position = s.position;
  odom.pose.pose.orientation = s.orientation;
  odom.twist.twist.linear = s.velocity;
  odom_pub_.publish(odom);

  PublishTransforms(stamp, s.position, s.orientation);
}

void TelemetryBridge::PublishTransforms(const ros::Time& stamp,
                       const geometry_msgs::Point& position,
                       const geometry_msgs::Quaternion& orientation) {
  if (!config_.publish_tf) return;

  geometry_msgs::TransformStamped map_to_base;
  map_to_base.header.stamp = stamp;
  map_to_base.header.frame_id = config_.frame_id;
  map_to_base.child_frame_id = config_.base_frame_id;
  map_to_base.transform.translation.x = position.x;
  map_to_base.transform.translation.y = position.y;
  map_to_base.transform.translation.z = position.z;
  map_to_base.transform.rotation = orientation;
  tf_broadcaster_.sendTransform(map_to_base);

  geometry_msgs::TransformStamped base_to_camera;
  base_to_camera.header.stamp = stamp;
  base_to_camera.header.frame_id = config_.base_frame_id;
  base_to_camera.child_frame_id = config_.camera_frame_id;
  base_to_camera.transform.translation.x = config_.camera_offset_x;
  base_to_camera.transform.translation.y = config_.camera_offset_y;
  base_to_camera.transform.translation.z = config_.camera_offset_z;

  tf2::Quaternion q;
  // base_link: x forward, y left, z up. optical: z forward, x right, y down.
  q.setRPY(-kPi * 0.5, 0.0, -kPi * 0.5);
  base_to_camera.transform.rotation.x = q.x();
  base_to_camera.transform.rotation.y = q.y();
  base_to_camera.transform.rotation.z = q.z();
  base_to_camera.transform.rotation.w = q.w();
  tf_broadcaster_.sendTransform(base_to_camera);
}

void TelemetryBridge::PublishDepth(const xp::DepthFrameMeta& meta, const uint8_t* payload, size_t payload_size) {
  // Preserve Unity capture time so independent transport queues cannot make
  // an older depth frame appear newer than a physics endpoint.
  ros::Time stamp;
  stamp.fromNSec(static_cast<uint64_t>(meta.sim_time_ns));

  sensor_msgs::Image image;
  image.header.stamp = stamp;
  image.header.frame_id = config_.camera_frame_id;
  image.height = static_cast<uint32_t>(meta.height);
  image.width = static_cast<uint32_t>(meta.width);
  image.encoding = "16UC1";
  image.is_bigendian = static_cast<uint8_t>(meta.byte_order != xp::kByteOrderLittleEndian);
  image.step = static_cast<uint32_t>(meta.row_step);
  image.data.resize(payload_size);

  const size_t row_bytes = static_cast<size_t>(meta.row_step);
  const size_t pixel_bytes = 2;  // 16UC1
  const size_t width_bytes = static_cast<size_t>(meta.width) * pixel_bytes;

  if (!config_.flip_depth_vertical && !config_.flip_depth_horizontal) {
    std::memcpy(image.data.data(), payload, payload_size);
  } else if (config_.flip_depth_vertical && !config_.flip_depth_horizontal) {
    for (int y = 0; y < meta.height; ++y) {
      const uint8_t* src = payload + static_cast<size_t>(meta.height - 1 - y) * row_bytes;
      uint8_t* dst = image.data.data() + static_cast<size_t>(y) * row_bytes;
      std::memcpy(dst, src, row_bytes);
    }
  } else {
    for (int y = 0; y < meta.height; ++y) {
      const int src_y = config_.flip_depth_vertical ? (meta.height - 1 - y) : y;
      const uint8_t* src_row = payload + static_cast<size_t>(src_y) * row_bytes;
      uint8_t* dst_row = image.data.data() + static_cast<size_t>(y) * row_bytes;
      for (int x = 0; x < meta.width; ++x) {
        const int src_x = config_.flip_depth_horizontal ? (meta.width - 1 - x) : x;
        std::memcpy(dst_row + static_cast<size_t>(x) * pixel_bytes,
                    src_row + static_cast<size_t>(src_x) * pixel_bytes,
                    pixel_bytes);
      }
      if (row_bytes > width_bytes) {
        std::memset(dst_row + width_bytes, 0, row_bytes - width_bytes);
      }
    }
  }
  depth_pub_.publish(image);

  sensor_msgs::CameraInfo ci;
  ci.header = image.header;
  ci.height = image.height;
  ci.width = image.width;
  ci.distortion_model = "plumb_bob";
  ci.D.assign(5, 0.0);
  ci.K[0] = meta.fx;
  ci.K[2] = config_.flip_depth_horizontal ? static_cast<double>(meta.width - 1) - meta.cx : meta.cx;
  ci.K[4] = meta.fy;
  ci.K[5] = config_.flip_depth_vertical ? static_cast<double>(meta.height - 1) - meta.cy : meta.cy;
  ci.K[8] = 1.0;
  ci.P[0] = meta.fx;
  ci.P[2] = config_.flip_depth_horizontal ? static_cast<double>(meta.width - 1) - meta.cx : meta.cx;
  ci.P[5] = meta.fy;
  ci.P[6] = ci.K[5];
  ci.P[10] = 1.0;
  camera_info_pub_.publish(ci);
}

void TelemetryBridge::FlushExecutionTransportAudit() {
  if (config_.execution_transport_audit_path.empty()) return;
  std::ofstream stream(
      config_.execution_transport_audit_path, std::ios::out | std::ios::trunc);
  if (!stream) {
    ROS_WARN("[DEBUG-EXEC-TRANSPORT-81660] cannot open bridge execution audit: %s",
             config_.execution_transport_audit_path.c_str());
    return;
  }
  stream << "{\"contract_id\":\"[DEBUG-EXEC-TRANSPORT-81660]bridge_state_audit\""
         << ",\"audit_schema_version\":2"
         << ",\"audit_run_id\":\"" << config_.execution_transport_audit_run_id << '\"'
         << ",\"episode_id\":" << config_.execution_transport_audit_episode_id
         << ",\"unity_runtime_identity\":\""
         << config_.execution_transport_audit_unity_runtime_identity << '\"'
         << ",\"capture_overflow\":"
         << (execution_transport_audit_overflow_ ? "true" : "false")
         << ",\"recv_hwm\":" << config_.receive_hwm
         << ",\"send_hwm\":" << config_.send_hwm
         << ",\"records\":[";
  for (size_t index = 0; index < execution_transport_audits_.size(); ++index) {
    const auto& record = execution_transport_audits_[index];
    if (index > 0) stream << ',';
    stream << "{\"execution_id\":" << record.execution_id
           << ",\"frame_index\":" << record.frame_index
           << ",\"command_id\":" << record.command_id
           << ",\"state_id\":" << record.state_id
           << ",\"sim_time_ns\":" << record.sim_time_ns
           << ",\"execution_status\":" << record.execution_status
           << ",\"state_zmq_recv_rc\":" << record.state_zmq_recv_rc
           << ",\"receive_sequence\":" << record.receive_sequence
           << ",\"receive_monotonic_ns\":" << record.receive_monotonic_ns
           << ",\"received\":" << (record.received ? "true" : "false")
           << ",\"forward_attempted\":"
           << (record.forward_attempted ? "true" : "false")
           << ",\"forwarded\":" << (record.forwarded ? "true" : "false")
           << ",\"forward_sequence\":" << record.forward_sequence
           << ",\"forward_monotonic_ns\":" << record.forward_monotonic_ns
           << '}';
  }
  stream << "]}\n";
}

}
