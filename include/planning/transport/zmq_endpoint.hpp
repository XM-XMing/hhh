#pragma once

#include <string>

namespace planning::transport {

std::string MakeTcpEndpoint(const std::string& host, int port);

}  // namespace planning::transport
