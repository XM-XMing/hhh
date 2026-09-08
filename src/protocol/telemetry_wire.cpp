#include <planning/protocol/telemetry_wire.hpp>

#include <array>
#include <cstdint>
#include <exception>

#include <msgpack.hpp>

#include <planning/protocol/msgpack_value.hpp>

namespace xp = planning::xm_protocol;

namespace planning::telemetry_wire {

namespace {

}  // namespace

bool ParseDynamicsState(const char* data, size_t size, xp::DynamicsState* state, std::string* error) {
  try {
    msgpack::object_handle oh = msgpack::unpack(data, size);
    const msgpack::object& obj = oh.get();
    if (obj.type != msgpack::type::ARRAY || obj.via.array.size < xp::kDynamicsStateFieldCount) {
      *error = "state root is not valid array";
      return false;
    }

    const auto* a = obj.via.array.ptr;
    int32_t schema = 0;
    if (!msgpack_value::AsInt32(a[xp::DynamicsStateIndex::SchemaVersion], &schema) || schema != xp::kSchemaVersion) {
      *error = "state schema mismatch";
      return false;
    }

    int64_t state_id = 0;
    int64_t sim_ns = 0;
    int32_t flags = 0;
    float min_clearance = 0.0f;

    if (!msgpack_value::AsInt64(a[xp::DynamicsStateIndex::StateId], &state_id) ||
        !msgpack_value::AsInt64(a[xp::DynamicsStateIndex::SimTimeNs], &sim_ns) ||
        !msgpack_value::AsInt32(a[xp::DynamicsStateIndex::Flags], &flags) ||
        !msgpack_value::AsFloat(a[xp::DynamicsStateIndex::MinClearance], &min_clearance) ||
        !msgpack_value::AsFloatArray(a[xp::DynamicsStateIndex::CurrPos], &state->pos) ||
        !msgpack_value::AsFloatArray(a[xp::DynamicsStateIndex::CurrRot], &state->rot) ||
        !msgpack_value::AsFloatArray(a[xp::DynamicsStateIndex::CurrVel], &state->vel) ||
        !msgpack_value::AsFloatArray(a[xp::DynamicsStateIndex::CurrAcc], &state->acc) ||
        !msgpack_value::AsFloatArray(a[xp::DynamicsStateIndex::FrontClearances], &state->front_clearances) ||
        !msgpack_value::AsInt64(a[xp::DynamicsStateIndex::AppliedExecutionId], &state->applied_execution_id) ||
        !msgpack_value::AsInt32(a[xp::DynamicsStateIndex::AppliedExecutionFrameIndex], &state->applied_execution_frame_index) ||
        !msgpack_value::AsInt64(a[xp::DynamicsStateIndex::AppliedCommandId], &state->applied_command_id) ||
        !msgpack_value::AsInt32(a[xp::DynamicsStateIndex::ExecutionStatus], &state->execution_status)) {
      *error = "state field decode failed";
      return false;
    }

    state->state_id = state_id;
    state->sim_time_ns = sim_ns;
    state->flags = flags;
    state->min_clearance = min_clearance;
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

bool ParseDepthMeta(const char* data, size_t size, xp::DepthFrameMeta* meta, std::string* error) {
  try {
    msgpack::object_handle oh = msgpack::unpack(data, size);
    const msgpack::object& obj = oh.get();
    if (obj.type != msgpack::type::ARRAY || obj.via.array.size < xp::kDepthFrameFieldCount) {
      *error = "depth meta root is not valid array";
      return false;
    }

    const auto* a = obj.via.array.ptr;
    int32_t schema = 0;
    if (!msgpack_value::AsInt32(a[xp::DepthFrameIndex::SchemaVersion], &schema) || schema != xp::kSchemaVersion) {
      *error = "depth meta schema mismatch";
      return false;
    }

    if (!msgpack_value::AsInt64(a[xp::DepthFrameIndex::CaptureId], &meta->capture_id) ||
        !msgpack_value::AsInt64(a[xp::DepthFrameIndex::SimTimeNs], &meta->sim_time_ns) ||
        !msgpack_value::AsInt32(a[xp::DepthFrameIndex::Flags], &meta->flags) ||
        !msgpack_value::AsFloatArray(a[xp::DepthFrameIndex::CapturePos], &meta->pos) ||
        !msgpack_value::AsFloatArray(a[xp::DepthFrameIndex::CaptureRot], &meta->rot) ||
        !msgpack_value::AsFloatArray(a[xp::DepthFrameIndex::CaptureVel], &meta->vel) ||
        !msgpack_value::AsFloatArray(a[xp::DepthFrameIndex::CaptureAcc], &meta->acc) ||
        !msgpack_value::AsFloatArray(a[xp::DepthFrameIndex::CaptureForward], &meta->forward) ||
        !msgpack_value::AsFloat(a[xp::DepthFrameIndex::MinClearance], &meta->min_clearance) ||
        !msgpack_value::AsFloatArray(a[xp::DepthFrameIndex::FrontClearances], &meta->front_clearances)) {
      *error = "depth meta field decode failed";
      return false;
    }

    const msgpack::object& cam = a[xp::DepthFrameIndex::Camera];
    const msgpack::object& dm = a[xp::DepthFrameIndex::DepthMeta];
    if (cam.type != msgpack::type::ARRAY || cam.via.array.size < xp::kCameraFieldCount ||
        dm.type != msgpack::type::ARRAY || dm.via.array.size < xp::kDepthMetaFieldCount) {
      *error = "depth camera/meta arrays invalid";
      return false;
    }

    float width_f = 0.0f;
    float height_f = 0.0f;
    float encoding_f = 0.0f;
    float row_step_f = 0.0f;
    float byte_order_f = 0.0f;

    if (!msgpack_value::AsFloat(cam.via.array.ptr[0], &meta->fx) ||
        !msgpack_value::AsFloat(cam.via.array.ptr[1], &meta->fy) ||
        !msgpack_value::AsFloat(cam.via.array.ptr[2], &meta->cx) ||
        !msgpack_value::AsFloat(cam.via.array.ptr[3], &meta->cy) ||
        !msgpack_value::AsFloat(cam.via.array.ptr[4], &width_f) ||
        !msgpack_value::AsFloat(cam.via.array.ptr[5], &height_f) ||
        !msgpack_value::AsFloat(cam.via.array.ptr[6], &meta->min_depth_m) ||
        !msgpack_value::AsFloat(cam.via.array.ptr[7], &meta->max_depth_m) ||
        !msgpack_value::AsFloat(dm.via.array.ptr[2], &encoding_f) ||
        !msgpack_value::AsFloat(dm.via.array.ptr[3], &row_step_f) ||
        !msgpack_value::AsFloat(dm.via.array.ptr[4], &byte_order_f)) {
      *error = "depth camera/meta scalar decode failed";
      return false;
    }

    meta->width = static_cast<int>(width_f);
    meta->height = static_cast<int>(height_f);
    meta->encoding = static_cast<int>(encoding_f);
    meta->row_step = static_cast<int>(row_step_f);
    meta->byte_order = static_cast<int>(byte_order_f);
    if (meta->width <= 0 || meta->height <= 0 || meta->row_step <= 0) {
      *error = "depth dimensions invalid";
      return false;
    }
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

// The existing bridge depth model intentionally ignores the v4 endpoint
// extension.  This read-only parser is used only by the P0-M2 telemetry audit;
// it neither changes validation nor forwards the value into bridge state.
bool ParseDepthEndpointStateIdForAudit(
    const char* data, size_t size, int64_t* endpoint_state_id) {
  try {
    msgpack::object_handle oh = msgpack::unpack(data, size);
    const msgpack::object& obj = oh.get();
    if (obj.type != msgpack::type::ARRAY || obj.via.array.size <= 13) return false;
    return msgpack_value::AsInt64(obj.via.array.ptr[13], endpoint_state_id);
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace planning::telemetry_wire
