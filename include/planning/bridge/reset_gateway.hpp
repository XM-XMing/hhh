#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include <planning/protocol/primitive_reset_wire.hpp>
#include <planning/transport/reliable_zmq_adapters.hpp>

namespace planning::transport {
class CommandTransport;
}

namespace planning::bridge {

struct ResetGatewayConfig {
    std::string runtime_instance_id;
};

class ResetGateway {
public:
    ResetGateway(
        planning::transport::CommandTransport& transport,
        ResetGatewayConfig config);

    void OnPythonReady(const std::vector<uint8_t>& identity);
    void OnUnityReady(
        const std::string& runtime_instance_id,
        const std::vector<uint8_t>& identity);
    void ReceivePythonReset(
        const std::vector<uint8_t>& identity,
        const std::vector<uint8_t>& payload);
    void ReceiveUnityResetComplete(
        const std::vector<uint8_t>& identity,
        const std::vector<uint8_t>& payload);
    void ReceivePythonResetAck(const std::vector<uint8_t>& payload);
    void RetryPendingResets();

    uint64_t protocol_error_count() const { return protocol_error_count_; }
    size_t pending_count() const { return pending_resets_.size(); }
    size_t current_reset_entries() const { return pending_resets_.size(); }
    size_t peak_reset_entries() const { return peak_reset_entries_; }
    size_t terminal_reset_count() const { return terminal_resets_.size(); }

private:
    struct ResetKey {
        std::string runtime_instance_id;
        std::string episode_id;
        std::string reset_id;

        bool operator<(const ResetKey& other) const;
    };

    struct PendingReset {
        ResetKey key;
        std::vector<uint8_t> request_payload;
        std::vector<uint8_t> complete_payload;
        std::vector<uint8_t> python_identity;
        bool unity_complete = false;
        bool python_ack = false;
    };

    struct TerminalReset {
        std::vector<uint8_t> request_payload;
        std::vector<uint8_t> complete_payload;
    };

    static ResetKey ResetKeyFor(
        const planning::primitive_reset_wire::ResetRequest& request);
    bool SendResetToPython(PendingReset* pending);
    bool ForwardResetToUnity(PendingReset* pending);
    void FinalizeResetLifecycle(const ResetKey& key);
    void RememberTerminal(const ResetKey& key, const TerminalReset& reset);

    planning::transport::ZmqUnityCommandRelay unity_command_relay_;
    planning::transport::ZmqPythonCommandAckSink python_command_ack_sink_;
    ResetGatewayConfig config_;
    std::map<ResetKey, PendingReset> pending_resets_;
    std::map<ResetKey, TerminalReset> terminal_resets_;
    std::deque<ResetKey> terminal_reset_order_;
    size_t peak_reset_entries_ = 0;
    uint64_t protocol_error_count_ = 0;

    static constexpr size_t kTerminalTombstoneCapacity = 1024;
};

}
