#include <iostream>
#include <string>

#include <zmq.h>

#include <planning/transport/zmq_option_contract.hpp>
#include <planning/transport/zmq_socket.hpp>

int main() {
    void* context = zmq_ctx_new();
    if (context == nullptr) return 1;

    planning::transport::ZmqSocket socket;
    std::string error;
    if (!socket.Open(context, ZMQ_PAIR, &error, "contract_test")) return 2;

    int value = 1;
    if (socket.SetInt(999, value, &error)) return 3;
    if (error.find("criticality=UNKNOWN") == std::string::npos) return 4;
    if (planning::transport::ZmqSocketOptionCriticalityFor(ZMQ_SNDHWM) !=
        planning::transport::ZmqSocketOptionCriticality::OPTIONAL_PERFORMANCE) {
        return 5;
    }

    socket.Close();
    zmq_ctx_term(context);
    std::cout << "C++ ZMQ option contract tests: 2 passed\n";
    return 0;
}
