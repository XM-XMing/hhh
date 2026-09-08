#include <planning/transport/zmq_socket.hpp>

#include <cerrno>
#include <string>

#include <zmq.h>

namespace planning::transport {
namespace {

void SetError(const char* operation, std::string* error) {
    if (error != nullptr) {
        *error = std::string(operation) + ": " + zmq_strerror(zmq_errno());
    }
}

}

ZmqSocket::~ZmqSocket() {
    Close();
}

bool ZmqSocket::Open(
    void* context,
    int type,
    std::string* error,
    const char* socket_purpose) {
    Close();
    socket_purpose_ = socket_purpose == nullptr ? "unknown_socket" : socket_purpose;
    socket_ = zmq_socket(context, type);
    if (socket_ == nullptr) {
        SetError("zmq_socket failed", error);
        return false;
    }
    return true;
}

bool ZmqSocket::SetInt(int option, int value, std::string* error) {
    if (socket_ != nullptr &&
        zmq_setsockopt(socket_, option, &value, sizeof(value)) == 0) {
        return true;
    }
    const int error_number = socket_ == nullptr ? ENOTSOCK : zmq_errno();
    return HandleOptionFailure(option, error_number, error);
}

bool ZmqSocket::ConfigureHwm(
    int receive_hwm,
    int send_hwm,
    bool configure_receive,
    bool configure_send,
    std::string* error) {
    if (!SetInt(ZMQ_LINGER, 0, error)) return false;
    if (configure_receive && !SetInt(ZMQ_RCVHWM, receive_hwm, error)) {
        return false;
    }
    if (configure_send && !SetInt(ZMQ_SNDHWM, send_hwm, error)) {
        return false;
    }
    return true;
}

bool ZmqSocket::SetBytes(
    int option,
    const void* data,
    size_t size,
    std::string* error) {
    if (socket_ != nullptr &&
        zmq_setsockopt(socket_, option, data, size) == 0) {
        return true;
    }
    const int error_number = socket_ == nullptr ? ENOTSOCK : zmq_errno();
    return HandleOptionFailure(option, error_number, error);
}

bool ZmqSocket::Connect(const std::string& endpoint, std::string* error) {
    if (socket_ == nullptr || zmq_connect(socket_, endpoint.c_str()) != 0) {
        SetError("zmq_connect failed", error);
        return false;
    }
    return true;
}

bool ZmqSocket::Bind(const std::string& endpoint, std::string* error) {
    if (socket_ == nullptr || zmq_bind(socket_, endpoint.c_str()) != 0) {
        SetError("zmq_bind failed", error);
        return false;
    }
    return true;
}

void ZmqSocket::Close() {
    if (socket_ != nullptr) {
        zmq_close(socket_);
        socket_ = nullptr;
    }
    socket_purpose_.clear();
    degraded_socket_options_.clear();
}

bool ZmqSocket::HandleOptionFailure(
    int option,
    int error_number,
    std::string* error) {
    const ZmqSocketOptionCriticality criticality =
        ZmqSocketOptionCriticalityFor(option);
    if (socket_ != nullptr &&
        (criticality == ZmqSocketOptionCriticality::OPTIONAL_PERFORMANCE ||
         criticality == ZmqSocketOptionCriticality::OPTIONAL_RECOVERY_TUNING)) {
        degraded_socket_options_.push_back(
            socket_purpose_ + "/" + ZmqSocketOptionName(option));
        return true;
    }
    // A newly introduced option is never allowed to silently degrade the
    // transport until its lifecycle/protocol criticality is reviewed.
    if (criticality == ZmqSocketOptionCriticality::UNKNOWN) {
        if (error != nullptr) {
            *error = std::string("zmq_setsockopt failed: socket=") +
                socket_purpose_ + ", option=" + ZmqSocketOptionName(option) +
                ", criticality=UNKNOWN: " + zmq_strerror(error_number);
        }
        return false;
    }
    if (error != nullptr) {
        *error = std::string("zmq_setsockopt failed: socket=") +
            socket_purpose_ + ", option=" + ZmqSocketOptionName(option) +
            ", criticality=" +
            ZmqSocketOptionCriticalityName(criticality) + ": " +
            zmq_strerror(error_number);
    }
    return false;
}

}
