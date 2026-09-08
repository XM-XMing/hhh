#include <planning/transport/snapshot_transport.hpp>

#include <zmq.h>

#include <planning/transport/zmq_endpoint.hpp>

namespace planning::transport {

bool SnapshotTransport::Init(
    void* context,
    const SnapshotTransportConfig& config,
    std::string* error) {
    Close();
    if (!observation_snapshot_router_.Open(
            context, ZMQ_ROUTER, error, "observation_snapshot_router") ||
        !python_snapshot_router_.Open(
            context, ZMQ_ROUTER, error, "python_snapshot_router")) {
        Close();
        return false;
    }
    if (!observation_snapshot_router_.ConfigureHwm(
            config.receive_hwm, config.send_hwm, true, true, error) ||
        !python_snapshot_router_.ConfigureHwm(
            config.receive_hwm, config.send_hwm, true, true, error) ||
        !observation_snapshot_router_.SetInt(ZMQ_ROUTER_HANDOVER, 1, error)) {
        Close();
        return false;
    }

    endpoints_.snapshot = MakeTcpEndpoint("*", config.observation_snapshot_port);
    endpoints_.python_snapshot = MakeTcpEndpoint(
        config.python_snapshot_bind_host, config.python_snapshot_port);
    if (!observation_snapshot_router_.Bind(endpoints_.snapshot, error) ||
        !python_snapshot_router_.Bind(endpoints_.python_snapshot, error)) {
        Close();
        return false;
    }
    return true;
}

void SnapshotTransport::Close() {
    observation_snapshot_router_.Close();
    python_snapshot_router_.Close();
    endpoints_ = {};
}

}
