#pragma once

#include <string>
#include <vector>

#include <planning/transport/zmq_socket.hpp>

namespace planning::transport {

struct ResultTransportConfig {
    std::string python_result_bind_host;
    int execution_result_port = 0;
    int python_result_port = 0;
    int receive_hwm = 4;
    int send_hwm = 4;
};

struct ResultTransportEndpoints {
    std::string result;
    std::string python_result;
};

class ResultTransport final {
public:
    bool Init(void* context, const ResultTransportConfig& config, std::string* error);
    void Close();

    void* result_router() const { return result_router_.get(); }
    void* python_result_router() const { return python_result_router_.get(); }
    void* python_result_monitor() const { return python_result_monitor_.get(); }
    bool monitor_enabled() const { return static_cast<bool>(python_result_monitor_); }
    const ResultTransportEndpoints& endpoints() const { return endpoints_; }
    std::vector<std::string> degraded_socket_options() const {
        std::vector<std::string> result;
        const auto& result_options = result_router_.degraded_socket_options();
        const auto& python_options =
            python_result_router_.degraded_socket_options();
        result.insert(
            result.end(), result_options.begin(), result_options.end());
        result.insert(
            result.end(), python_options.begin(), python_options.end());
        return result;
    }

private:
    ZmqSocket result_router_;
    ZmqSocket python_result_router_;
    ZmqSocket python_result_monitor_;
    ResultTransportEndpoints endpoints_;
};

}
