#include <planning/transport/telemetry_transport.hpp>

#include <zmq.h>

#include <planning/transport/zmq_endpoint.hpp>

namespace planning::transport {

bool TelemetryTransport::Init(
    void* context,
    const TelemetryTransportConfig& config,
    std::string* error) {
    Close();
    if (!state_sub_.Open(context, ZMQ_SUB, error, "state_sub") ||
        !depth_sub_.Open(context, ZMQ_SUB, error, "depth_sub")) {
        Close();
        return false;
    }

    const char empty_filter[] = "";
    if (!state_sub_.ConfigureHwm(
            config.receive_hwm, 0, true, false, error) ||
        !state_sub_.SetBytes(ZMQ_SUBSCRIBE, empty_filter, 0, error) ||
        !depth_sub_.ConfigureHwm(
            config.receive_hwm, 0, true, false, error) ||
        !depth_sub_.SetBytes(ZMQ_SUBSCRIBE, empty_filter, 0, error)) {
        Close();
        return false;
    }

    endpoints_.state = MakeTcpEndpoint(config.unity_host, config.state_port);
    endpoints_.depth = MakeTcpEndpoint(config.unity_host, config.depth_port);
    if (!state_sub_.Connect(endpoints_.state, error) ||
        !depth_sub_.Connect(endpoints_.depth, error)) {
        Close();
        return false;
    }
    return true;
}

void TelemetryTransport::Close() {
    state_sub_.Close();
    depth_sub_.Close();
    endpoints_ = {};
}

}
