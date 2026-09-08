#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <planning/transport/zmq_option_contract.hpp>

namespace planning::transport {

class ZmqSocket final {
public:
    ZmqSocket() = default;
    ~ZmqSocket();

    ZmqSocket(const ZmqSocket&) = delete;
    ZmqSocket& operator=(const ZmqSocket&) = delete;

    bool Open(
        void* context,
        int type,
        std::string* error,
        const char* socket_purpose = nullptr);
    bool ConfigureHwm(
        int receive_hwm,
        int send_hwm,
        bool configure_receive,
        bool configure_send,
        std::string* error);
    bool SetInt(int option, int value, std::string* error);
    bool SetBytes(int option, const void* data, size_t size, std::string* error);
    bool Connect(const std::string& endpoint, std::string* error);
    bool Bind(const std::string& endpoint, std::string* error);
    void Close();

    void* get() const { return socket_; }
    explicit operator bool() const { return socket_ != nullptr; }
    const std::vector<std::string>& degraded_socket_options() const {
        return degraded_socket_options_;
    }

private:
    bool HandleOptionFailure(int option, int error_number, std::string* error);

    void* socket_ = nullptr;
    std::string socket_purpose_;
    std::vector<std::string> degraded_socket_options_;
};

}
