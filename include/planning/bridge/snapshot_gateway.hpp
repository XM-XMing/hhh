#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include <planning/bridge/observation_retrieval_broker.hpp>
#include <planning/protocol/xm_protocol.hpp>
#include <planning/transport/reliable_zmq_adapters.hpp>

namespace planning::transport {
class SnapshotTransport;
}

namespace planning::bridge {

class SnapshotGateway {
public:
    explicit SnapshotGateway(planning::transport::SnapshotTransport& transport);

    void SetMetricsFlushCallback(std::function<void()> callback);
    bool ReceiveResult(
        const planning::xm_protocol::v4::PrimitiveExecutionResult& result,
        uint64_t now_ms);
    bool FinalizeExecution(
        const planning::xm_protocol::v4::PrimitiveExecutionResult& result);
    bool FinalizeResetObservation(
        const planning::xm_protocol::v4::ObservationRef& observation_ref);
    void PollObservationSnapshots();
    void PollPythonSnapshotRequests();
    void Poll(uint64_t now_ms);

    uint64_t response_count() const { return snapshot_response_count_; }
    uint64_t hash_match_count() const { return snapshot_hash_match_count_; }
    uint64_t request_count() const { return observation_retrieval_broker_.request_count(); }
    uint64_t stored_snapshot_count() const { return observation_retrieval_broker_.stored_snapshot_count(); }
    uint64_t missing_snapshot_count() const { return observation_retrieval_broker_.missing_snapshot_count(); }
    uint64_t pending_count() const { return observation_retrieval_broker_.pending_count(); }
    size_t current_snapshot_entries() const {
        return observation_retrieval_broker_.current_snapshot_entries();
    }
    size_t peak_snapshot_entries() const {
        return observation_retrieval_broker_.peak_snapshot_entries();
    }
    size_t terminal_snapshot_count() const {
        return observation_retrieval_broker_.terminal_snapshot_count();
    }
    uint64_t bridge_protocol_error_count() const { return protocol_error_count_; }
    uint64_t broker_protocol_error_count() const { return observation_retrieval_broker_.protocol_error_count(); }

private:
    struct PendingPythonSnapshotDelivery {
        planning::xm_protocol::v4::SnapshotRequest request;
        std::vector<uint8_t> identity;
    };

    void RelayPendingPythonSnapshot(
        const planning::xm_protocol::v4::SnapshotRequest& request);
    void RelayPendingPythonSnapshotMissing(
        const planning::xm_protocol::v4::SnapshotRequest& request);
    bool SendPythonSnapshotWire(
        const std::vector<uint8_t>& identity,
        const std::vector<uint8_t>& payload);
    bool FinalizeResetIfDelivered(
        const planning::xm_protocol::v4::SnapshotRequest& request);

    planning::transport::SnapshotTransport& transport_;
    planning::observation_retrieval_broker::InMemorySnapshotCache observation_snapshot_cache_;
    planning::transport::ZmqUnitySnapshotRequester observation_snapshot_requester_;
    planning::transport::ZmqUnitySnapshotAckSink observation_snapshot_ack_sink_;
    planning::observation_retrieval_broker::ObservationRetrievalBroker observation_retrieval_broker_;
    std::map<std::vector<uint8_t>, PendingPythonSnapshotDelivery> pending_python_snapshot_deliveries_;
    std::function<void()> metrics_flush_callback_;
    uint64_t snapshot_response_count_ = 0;
    uint64_t snapshot_hash_match_count_ = 0;
    uint64_t protocol_error_count_ = 0;
};

}
