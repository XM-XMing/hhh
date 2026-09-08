#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <planning/bridge/primitive_execution_result_broker.hpp>
#include <planning/protocol/primitive_execution_result_wire.hpp>
#include <planning/transport/reliable_zmq_adapters.hpp>

namespace planning::transport {
class ResultTransport;
}
namespace planning::bridge {
class SnapshotGateway;

struct ResultGatewayConfig {
    double python_result_retry_interval_s = 0.5;
    int python_result_stale_retry_threshold = 3;
    std::string python_result_diagnostics_path;
};

class ResultGateway {
public:
    using ExecutionLifecycleFinalizer = std::function<void(
        const planning::primitive_execution_result_broker::ResultKey&,
        const planning::xm_protocol::v4::PrimitiveExecutionResult&)>;

    ResultGateway(
        planning::transport::ResultTransport& transport,
        SnapshotGateway& snapshot_gateway,
        ResultGatewayConfig config);
    ~ResultGateway();

    void SetMetricsFlushCallback(std::function<void()> callback);
    void SetExecutionLifecycleFinalizer(ExecutionLifecycleFinalizer callback);
    void PollPythonTransportMonitor();
    void PollExecutionResults();
    void PollPythonResults();
    void RetryPendingPythonResultsIfDue();

    uint64_t accepted_count() const { return result_accepted_count_; }
    uint64_t duplicate_count() const { return result_duplicate_count_; }
    uint64_t conflict_count() const { return result_conflict_count_; }
    uint64_t protocol_error_count() const { return result_protocol_error_count_; }
    uint64_t ack_count() const { return result_ack_count_; }
    uint64_t python_relay_count() const { return python_relay_count_; }
    size_t pending_count() const { return result_store_.pending_count(); }
    size_t pending_receipt_count() const { return result_broker_.receipt_pending_count(); }
    size_t python_commit_count() const { return result_broker_.python_commit_count(); }
    size_t receipt_count() const { return result_broker_.receipt_count(); }
    size_t duplicate_receipt_count() const { return result_broker_.duplicate_receipt_count(); }
    size_t current_result_entries() const { return result_store_.size(); }
    size_t peak_result_entries() const { return result_store_.peak_size(); }
    size_t terminal_result_count() const { return result_store_.terminal_count(); }

private:
    int64_t PythonRetryIntervalNs() const;
    void ScheduleNextPythonRetry();
    void SendPythonCommitAck(
        const planning::primitive_execution_result_wire::CommitMessage& commit,
        const std::vector<uint8_t>& identity,
        planning::primitive_execution_result_broker::CommitOutcome outcome);

    planning::transport::ResultTransport& transport_;
    SnapshotGateway& snapshot_gateway_;
    ResultGatewayConfig config_;
    std::ofstream python_result_diagnostics_stream_;
    std::string python_result_router_owner_thread_id_;
    bool python_connected_ = false;
    int64_t next_python_retry_ns_ = 0;
    uint64_t result_accepted_count_ = 0;
    uint64_t result_duplicate_count_ = 0;
    uint64_t result_conflict_count_ = 0;
    uint64_t result_protocol_error_count_ = 0;
    uint64_t result_ack_count_ = 0;
    uint64_t python_relay_count_ = 0;
    planning::primitive_execution_result_broker::InMemoryResultStore result_store_;
    planning::transport::ZmqUnityResultAckSink result_ack_sink_;
    planning::transport::ZmqPythonResultRelay python_result_relay_;
    planning::primitive_execution_result_broker::PrimitiveExecutionResultBroker result_broker_;
    std::function<void()> metrics_flush_callback_;
    ExecutionLifecycleFinalizer execution_lifecycle_finalizer_;
};

}
