#pragma once

#include <string>
#include <vector>

#include <planning/transport/zmq_socket.hpp>

namespace planning::transport {

struct CommandTransportConfig {
    std::string unity_host;
    int command_port = 0;
    int python_command_port = 0;
    int unity_command_port = 0;
    int receive_hwm = 4;
    int send_hwm = 4;
};

struct CommandTransportEndpoints {
    std::string command;
    std::string python_command;
    std::string unity_command;
};

class CommandTransport final {
public:
    bool Init(void* context, const CommandTransportConfig& config, std::string* error);
    void Close();

    void* command_pub() const { return command_pub_.get(); }
    void* python_command_router() const { return python_command_router_.get(); }
    void* unity_command_router() const { return unity_command_router_.get(); }
    const CommandTransportEndpoints& endpoints() const { return endpoints_; }
    std::vector<std::string> degraded_socket_options() const {
        std::vector<std::string> result;
        const auto& command_options = command_pub_.degraded_socket_options();
        const auto& python_options =
            python_command_router_.degraded_socket_options();
        const auto& unity_options = unity_command_router_.degraded_socket_options();
        result.insert(
            result.end(), command_options.begin(), command_options.end());
        result.insert(
            result.end(), python_options.begin(), python_options.end());
        result.insert(
            result.end(), unity_options.begin(), unity_options.end());
        return result;
    }

private:
    ZmqSocket command_pub_;
    ZmqSocket python_command_router_;
    ZmqSocket unity_command_router_;
    CommandTransportEndpoints endpoints_;
};

}
