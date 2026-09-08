#include <planning/bridge/reset_gateway.hpp>
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

bool ResetGateway::ResetKey::operator<(const ResetKey& other) const {
  if (runtime_instance_id != other.runtime_instance_id)
    return runtime_instance_id < other.runtime_instance_id;
  if (episode_id != other.episode_id) return episode_id < other.episode_id;
  return reset_id < other.reset_id;
}

ResetGateway::ResetGateway(
    planning::transport::CommandTransport& transport,
    ResetGatewayConfig config)
    : config_(std::move(config)) {
  unity_command_relay_.Configure(transport.unity_command_router(), nullptr);
  python_command_ack_sink_.Configure(
      transport.python_command_router(), nullptr, nullptr, "");
}

void ResetGateway::OnPythonReady(const std::vector<uint8_t>& identity) {
  for (auto& entry : pending_resets_) {
    entry.second.python_identity = identity;
    if (entry.second.unity_complete && !entry.second.python_ack) {
      SendResetToPython(&entry.second);
    }
  }
}

void ResetGateway::OnUnityReady(
    const std::string& runtime_instance_id,
    const std::vector<uint8_t>& identity) {
  unity_command_relay_.RegisterReady(runtime_instance_id, identity);
}

ResetGateway::ResetKey ResetGateway::ResetKeyFor(const primitive_reset_wire::ResetRequest& request) {
  return ResetKey{request.runtime_instance_id, request.episode_id, request.reset_id};
}

bool ResetGateway::SendResetToPython(PendingReset* pending) {
  if (pending == nullptr || pending->python_identity.empty() ||
      pending->complete_payload.empty()) return false;
  python_command_ack_sink_.SetIdentity(pending->python_identity);
  return python_command_ack_sink_.SendRaw(pending->complete_payload);
}

bool ResetGateway::ForwardResetToUnity(PendingReset* pending) {
  if (pending == nullptr) return false;
  return unity_command_relay_.SendRaw(
      pending->key.runtime_instance_id, pending->request_payload);
}

void ResetGateway::ReceivePythonReset(
    const std::vector<uint8_t>& identity,
    const std::vector<uint8_t>& payload) {
  primitive_reset_wire::ResetRequest request;
  std::string error;
  if (!primitive_reset_wire::ParseRequest(
          reinterpret_cast<const char*>(payload.data()), payload.size(),
          &request, &error)) return;
  if (!config_.runtime_instance_id.empty() &&
      request.runtime_instance_id != config_.runtime_instance_id) return;

  const ResetKey key = ResetKeyFor(request);
  const std::map<ResetKey, TerminalReset>::const_iterator terminal =
      terminal_resets_.find(key);
  if (terminal != terminal_resets_.end()) {
    if (terminal->second.request_payload != payload) {
      ++protocol_error_count_;
      return;
    }
    PendingReset replay;
    replay.key = key;
    replay.request_payload = terminal->second.request_payload;
    replay.complete_payload = terminal->second.complete_payload;
    replay.python_identity = identity;
    replay.unity_complete = true;
    replay.python_ack = false;
    SendResetToPython(&replay);
    return;
  }
  std::map<ResetKey, PendingReset>::iterator found = pending_resets_.find(key);
  if (found != pending_resets_.end()) {
    if (found->second.request_payload != payload) {
      ++protocol_error_count_;
      return;
    }
    found->second.python_identity = identity;
    if (found->second.unity_complete) SendResetToPython(&found->second);
    else ForwardResetToUnity(&found->second);
    return;
  }

  PendingReset pending;
  pending.key = key;
  pending.request_payload = payload;
  pending.python_identity = identity;
  pending_resets_.insert(std::make_pair(key, pending));
  if (pending_resets_.size() > peak_reset_entries_)
    peak_reset_entries_ = pending_resets_.size();
  ForwardResetToUnity(&pending_resets_.find(key)->second);
}

void ResetGateway::ReceiveUnityResetComplete(
    const std::vector<uint8_t>& identity,
    const std::vector<uint8_t>& payload) {
  primitive_reset_wire::ResetComplete complete;
  std::string error;
  if (!primitive_reset_wire::ParseComplete(
          reinterpret_cast<const char*>(payload.data()), payload.size(),
          &complete, &error)) return;
  const ResetKey key{
      complete.runtime_instance_id, complete.episode_id, complete.reset_id};
  std::map<ResetKey, PendingReset>::iterator found = pending_resets_.find(key);
  if (found == pending_resets_.end()) {
    const std::map<ResetKey, TerminalReset>::const_iterator terminal =
        terminal_resets_.find(key);
    if (terminal != terminal_resets_.end() &&
        terminal->second.complete_payload == payload) {
      return;
    }
    ++protocol_error_count_;
    return;
  }
  if (!identity.empty() &&
      !unity_command_relay_.IdentityMatches(complete.runtime_instance_id, identity)) {
    ++protocol_error_count_;
    return;
  }
  if (complete.observation_ref.runtime_instance_id != complete.runtime_instance_id ||
      complete.observation_ref.episode_id != complete.episode_id ||
      complete.observation_ref.reset_id != complete.reset_id) {
    ++protocol_error_count_;
    return;
  }
  if (found->second.unity_complete &&
      found->second.complete_payload != payload) {
    ++protocol_error_count_;
    return;
  }
  found->second.complete_payload = payload;
  found->second.unity_complete = true;
  SendResetToPython(&found->second);
}

void ResetGateway::ReceivePythonResetAck(const std::vector<uint8_t>& payload) {
  primitive_reset_wire::ResetReceivedAck ack;
  std::string error;
  if (!primitive_reset_wire::ParseReceivedAck(
          reinterpret_cast<const char*>(payload.data()), payload.size(),
          &ack, &error)) return;
  const ResetKey key{ack.runtime_instance_id, ack.episode_id, ack.reset_id};
  std::map<ResetKey, PendingReset>::iterator found = pending_resets_.find(key);
  if (found == pending_resets_.end() || !found->second.unity_complete) {
    if (found == pending_resets_.end() && terminal_resets_.find(key) != terminal_resets_.end())
      return;
    ++protocol_error_count_;
    return;
  }
  found->second.python_ack = true;
  FinalizeResetLifecycle(key);
}

void ResetGateway::RetryPendingResets() {
  for (std::map<ResetKey, PendingReset>::iterator entry = pending_resets_.begin();
       entry != pending_resets_.end(); ++entry) {
    if (entry->second.python_ack) continue;
    if (entry->second.unity_complete) SendResetToPython(&entry->second);
    else ForwardResetToUnity(&entry->second);
  }
}

void ResetGateway::FinalizeResetLifecycle(const ResetKey& key) {
  const std::map<ResetKey, PendingReset>::iterator found =
      pending_resets_.find(key);
  if (found == pending_resets_.end() || !found->second.unity_complete ||
      !found->second.python_ack) {
    return;
  }
  TerminalReset terminal;
  terminal.request_payload = found->second.request_payload;
  terminal.complete_payload = found->second.complete_payload;
  RememberTerminal(key, terminal);
  pending_resets_.erase(found);
}

void ResetGateway::RememberTerminal(
    const ResetKey& key, const TerminalReset& reset) {
  const std::map<ResetKey, TerminalReset>::iterator found =
      terminal_resets_.find(key);
  if (found != terminal_resets_.end()) {
    found->second = reset;
    return;
  }
  terminal_resets_.insert(std::make_pair(key, reset));
  terminal_reset_order_.push_back(key);
  while (terminal_reset_order_.size() > kTerminalTombstoneCapacity) {
    terminal_resets_.erase(terminal_reset_order_.front());
    terminal_reset_order_.pop_front();
  }
}

}
