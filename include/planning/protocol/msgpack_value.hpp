#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <msgpack.hpp>

namespace planning::msgpack_value {

inline bool AsInt64(
    const msgpack::object& object,
    int64_t* value,
    std::string* error = nullptr) {
  if (object.type == msgpack::type::POSITIVE_INTEGER) {
    if (error != nullptr && object.via.u64 >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      *error = "unsigned integer exceeds int64";
      return false;
    }
    *value = static_cast<int64_t>(object.via.u64);
    return true;
  }
  if (object.type == msgpack::type::NEGATIVE_INTEGER) {
    *value = object.via.i64;
    return true;
  }
  if (error != nullptr) *error = "expected signed integer";
  return false;
}

inline bool AsInt32(const msgpack::object& object, int32_t* value) {
  int64_t wide = 0;
  if (!AsInt64(object, &wide)) return false;
  *value = static_cast<int32_t>(wide);
  return true;
}

inline bool AsFloat(
    const msgpack::object& object,
    float* value,
    std::string* error = nullptr) {
  if (object.type == msgpack::type::FLOAT32 ||
      object.type == msgpack::type::FLOAT64) {
    *value = static_cast<float>(object.via.f64);
    return true;
  }
  int64_t integer = 0;
  if (!AsInt64(object, &integer, error)) {
    if (error != nullptr) *error = "expected float";
    return false;
  }
  *value = static_cast<float>(integer);
  return true;
}

inline bool AsBytes(
    const msgpack::object& object, std::vector<uint8_t>* value) {
  if (object.type != msgpack::type::BIN) return false;
  const uint8_t* data = reinterpret_cast<const uint8_t*>(object.via.bin.ptr);
  value->assign(data, data + object.via.bin.size);
  return true;
}

template <size_t N>
inline bool AsFloatArray(
    const msgpack::object& object, std::array<float, N>* value) {
  if (object.type != msgpack::type::ARRAY || object.via.array.size < N) {
    return false;
  }
  for (size_t index = 0; index < N; ++index) {
    if (!AsFloat(object.via.array.ptr[index], &((*value)[index]))) return false;
  }
  return true;
}

}  // namespace planning::msgpack_value
