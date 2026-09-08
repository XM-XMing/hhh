#include <planning/transport/result_transport.hpp>

#include <zmq.h>

#include <planning/transport/zmq_endpoint.hpp>

namespace planning::transport {

bool ResultTransport::Init(
    void* context,
    const ResultTransportConfig& config,
    std::string* error) {
    Close();
    if (!result_router_.Open(context, ZMQ_ROUTER, error, "result_router") ||
        !python_result_router_.Open(
            context, ZMQ_ROUTER, error, "python_result_router")) {
        Close();
        return false;
    }
    if (!result_router_.ConfigureHwm(
            config.receive_hwm, config.send_hwm, true, true, error) ||
        !python_result_router_.ConfigureHwm(
            config.receive_hwm, config.send_hwm, true, true, error) ||
        !python_result_router_.SetInt(ZMQ_ROUTER_HANDOVER, 1, error)) {
        Close();
        return false;
    }

    endpoints_.result = MakeTcpEndpoint("*", config.execution_result_port);
    endpoints_.python_result = MakeTcpEndpoint(
        config.python_result_bind_host, config.python_result_port);
    if (!result_router_.Bind(endpoints_.result, error) ||
        !python_result_router_.Bind(endpoints_.python_result, error)) {
        Close();
        return false;
    }

    const std::string monitor_endpoint = "inproc://python-result-router-monitor";
    std::string ignored_error;
    if (!python_result_monitor_.Open(context, ZMQ_PAIR, &ignored_error) ||
        zmq_socket_monitor(
            python_result_router_.get(), monitor_endpoint.c_str(), ZMQ_EVENT_ALL) != 0 ||
        !python_result_monitor_.Connect(monitor_endpoint, &ignored_error)) {
        python_result_monitor_.Close();
    }
    return true;
}

void ResultTransport::Close() {
    result_router_.Close();
    python_result_router_.Close();
    python_result_monitor_.Close();
    endpoints_ = {};
}

}
