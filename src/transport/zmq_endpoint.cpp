#include <planning/transport/zmq_endpoint.hpp>

namespace planning::transport {

std::string MakeTcpEndpoint(const std::string& host, int port) {
  return "tcp://" + host + ":" + std::to_string(port);
}

}  // namespace planning::transport
