#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <planning/protocol/xm_protocol.hpp>

namespace planning::telemetry_wire {

bool ParseDynamicsState(
    const char* data,
    size_t size,
    xm_protocol::DynamicsState* state,
    std::string* error);
bool ParseDepthMeta(
    const char* data,
    size_t size,
    xm_protocol::DepthFrameMeta* meta,
    std::string* error);
bool ParseDepthEndpointStateIdForAudit(
    const char* data,
    size_t size,
    int64_t* endpoint_state_id);

}  // namespace planning::telemetry_wire
