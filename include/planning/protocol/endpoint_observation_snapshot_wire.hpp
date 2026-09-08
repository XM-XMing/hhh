#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <planning/protocol/xm_protocol.hpp>

namespace planning::endpoint_observation_snapshot_wire {

bool ParseReady(const char* data, size_t size, std::string* runtime_id, std::string* error);
bool ParseSnapshotResponse(
    const char* data,
    size_t size,
    xm_protocol::v4::SnapshotRequest* request,
    xm_protocol::v4::EndpointObservationSnapshot* snapshot,
    std::string* error);
bool ParseMissing(
    const char* data,
    size_t size,
    xm_protocol::v4::SnapshotRequest* request,
    std::string* error);
bool ParsePythonRequest(
    const char* data,
    size_t size,
    xm_protocol::v4::SnapshotRequest* request,
    std::string* error);
std::vector<uint8_t> SerializeRequest(const xm_protocol::v4::SnapshotRequest& request);
std::vector<uint8_t> SerializeAck(const xm_protocol::v4::SnapshotAck& ack);
std::vector<uint8_t> SerializePythonSnapshot(
    const xm_protocol::v4::SnapshotRequest& request,
    const xm_protocol::v4::EndpointObservationSnapshot& snapshot);
std::vector<uint8_t> SerializePythonMissing(const xm_protocol::v4::SnapshotRequest& request);

}  // namespace planning::endpoint_observation_snapshot_wire
