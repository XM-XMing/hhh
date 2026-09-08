#pragma once

#include <string>
#include <vector>

#include <planning/transport/zmq_socket.hpp>

namespace planning::transport {

struct SnapshotTransportConfig {
    std::string python_snapshot_bind_host;
    int observation_snapshot_port = 0;
    int python_snapshot_port = 0;
    int receive_hwm = 4;
    int send_hwm = 4;
};

struct SnapshotTransportEndpoints {
    std::string snapshot;
    std::string python_snapshot;
};

class SnapshotTransport final {
public:
    bool Init(void* context, const SnapshotTransportConfig& config, std::string* error);
    void Close();

    void* observation_snapshot_router() const { return observation_snapshot_router_.get(); }
    void* python_snapshot_router() const { return python_snapshot_router_.get(); }
    const SnapshotTransportEndpoints& endpoints() const { return endpoints_; }
    std::vector<std::string> degraded_socket_options() const {
        std::vector<std::string> result;
        const auto& observation_options =
            observation_snapshot_router_.degraded_socket_options();
        const auto& python_options =
            python_snapshot_router_.degraded_socket_options();
        result.insert(
            result.end(), observation_options.begin(), observation_options.end());
        result.insert(
            result.end(), python_options.begin(), python_options.end());
        return result;
    }

private:
    ZmqSocket observation_snapshot_router_;
    ZmqSocket python_snapshot_router_;
    SnapshotTransportEndpoints endpoints_;
};

}
