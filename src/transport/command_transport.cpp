#include <planning/transport/command_transport.hpp>

#include <zmq.h>

#include <planning/transport/zmq_endpoint.hpp>

namespace planning::transport {

bool CommandTransport::Init(
    void* context,
    const CommandTransportConfig& config,
    std::string* error) {
    Close();
    if (!command_pub_.Open(context, ZMQ_PUB, error, "command_pub") ||
        !python_command_router_.Open(
            context, ZMQ_ROUTER, error, "python_command_router") ||
        !unity_command_router_.Open(
            context, ZMQ_ROUTER, error, "unity_command_router")) {
        Close();
        return false;
    }

    if (!command_pub_.ConfigureHwm(
            0, config.send_hwm, false, true, error) ||
        !python_command_router_.ConfigureHwm(
            config.receive_hwm, config.send_hwm, true, true, error) ||
        !unity_command_router_.ConfigureHwm(
            config.receive_hwm, config.send_hwm, true, true, error) ||
        !python_command_router_.SetInt(ZMQ_ROUTER_HANDOVER, 1, error) ||
        !unity_command_router_.SetInt(ZMQ_ROUTER_HANDOVER, 1, error)) {
        Close();
        return false;
    }

    endpoints_.command = MakeTcpEndpoint(config.unity_host, config.command_port);
    endpoints_.python_command = MakeTcpEndpoint("*", config.python_command_port);
    endpoints_.unity_command = MakeTcpEndpoint("*", config.unity_command_port);

    if (!command_pub_.Connect(endpoints_.command, error) ||
        !python_command_router_.Bind(endpoints_.python_command, error) ||
        !unity_command_router_.Bind(endpoints_.unity_command, error)) {
        Close();
        return false;
    }
    return true;
}

void CommandTransport::Close() {
    command_pub_.Close();
    python_command_router_.Close();
    unity_command_router_.Close();
    endpoints_ = {};
}

}
