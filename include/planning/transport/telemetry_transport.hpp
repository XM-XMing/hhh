#pragma once

#include <string>
#include <vector>

#include <planning/transport/zmq_socket.hpp>

namespace planning::transport {

struct TelemetryTransportConfig {
    std::string unity_host;
    int state_port = 0;
    int depth_port = 0;
    int receive_hwm = 4;
};

struct TelemetryTransportEndpoints {
    std::string state;
    std::string depth;
};

class TelemetryTransport final {
public:
    bool Init(void* context, const TelemetryTransportConfig& config, std::string* error);
    void Close();

    void* state_sub() const { return state_sub_.get(); }
    void* depth_sub() const { return depth_sub_.get(); }
    const TelemetryTransportEndpoints& endpoints() const { return endpoints_; }
    std::vector<std::string> degraded_socket_options() const {
        std::vector<std::string> result;
        const auto& state_options = state_sub_.degraded_socket_options();
        const auto& depth_options = depth_sub_.degraded_socket_options();
        result.insert(
            result.end(), state_options.begin(), state_options.end());
        result.insert(
            result.end(), depth_options.begin(), depth_options.end());
        return result;
    }

private:
    ZmqSocket state_sub_;
    ZmqSocket depth_sub_;
    TelemetryTransportEndpoints endpoints_;
};

}
