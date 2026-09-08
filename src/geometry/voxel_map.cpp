#include <planning/geometry/voxel_map.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace planning::geometry {

struct VoxelMapHandle {
    std::shared_ptr<const VoxelMap> map;
};

std::shared_ptr<const VoxelMap> VoxelMap::Create(
    const int64_t* occupied_keys,
    int64_t num_keys,
    const int64_t* origin_ijk,
    const int64_t* grid_shape,
    double voxel_size) {
    std::shared_ptr<VoxelMap> map(new VoxelMap());
    if (!map->Build(
            occupied_keys, num_keys, origin_ijk, grid_shape, voxel_size)) {
        return nullptr;
    }
    return map;
}

bool VoxelMap::Build(
    const int64_t* occupied_keys,
    int64_t num_keys,
    const int64_t* origin_ijk,
    const int64_t* grid_shape,
    double voxel_size) {
    if (num_keys < 0 || (num_keys > 0 && occupied_keys == nullptr) ||
        origin_ijk == nullptr ||
        grid_shape == nullptr || voxel_size <= 0.0) {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (grid_shape[axis] <= 0) {
            return false;
        }
        origin_ijk_[axis] = origin_ijk[axis];
        grid_shape_[axis] = grid_shape[axis];
    }
    voxel_size_ = voxel_size;
    stride_y_ = grid_shape_[2];
    if (grid_shape_[1] > std::numeric_limits<int64_t>::max() / grid_shape_[2]) {
        return false;
    }
    stride_x_ = grid_shape_[1] * grid_shape_[2];

    const uint64_t sx = static_cast<uint64_t>(grid_shape_[0]);
    const uint64_t sy = static_cast<uint64_t>(grid_shape_[1]);
    const uint64_t sz = static_cast<uint64_t>(grid_shape_[2]);
    if (sx > std::numeric_limits<uint64_t>::max() / sy ||
        sx * sy > std::numeric_limits<uint64_t>::max() / sz) {
        return false;
    }
    key_count_ = sx * sy * sz;
    if (key_count_ == 0 || key_count_ > std::numeric_limits<uint64_t>::max() - 63) {
        return false;
    }
    const uint64_t word_count = (key_count_ + 63) / 64;
    if (word_count > std::numeric_limits<size_t>::max()) {
        return false;
    }

    occupied_bits_.assign(static_cast<size_t>(word_count), uint64_t{0});
    occupied_keys_.clear();
    if (num_keys > 0) {
        occupied_keys_.assign(occupied_keys, occupied_keys + num_keys);
        for (int64_t key : occupied_keys_) {
            if (key < 0 || static_cast<uint64_t>(key) >= key_count_) {
                return false;
            }
            const uint64_t value = static_cast<uint64_t>(key);
            occupied_bits_[value >> 6] |= uint64_t{1} << (value & 63);
        }
    }
    valid_ = true;
    return true;
}

bool VoxelMap::HasKey(int64_t key) const {
    if (key < 0 || static_cast<uint64_t>(key) >= key_count_) {
        return false;
    }
    const uint64_t value = static_cast<uint64_t>(key);
    return (occupied_bits_[value >> 6] & (uint64_t{1} << (value & 63))) != 0;
}

bool VoxelMap::IsOccupiedVoxel(int64_t ix, int64_t iy, int64_t iz) const {
    const int64_t rx = ix - origin_ijk_[0];
    const int64_t ry = iy - origin_ijk_[1];
    const int64_t rz = iz - origin_ijk_[2];
    if (rx < 0 || ry < 0 || rz < 0 ||
        rx >= grid_shape_[0] || ry >= grid_shape_[1] || rz >= grid_shape_[2]) {
        return false;
    }
    return HasKey(rx * stride_x_ + ry * stride_y_ + rz);
}

std::array<int64_t, 3> VoxelMap::WorldToVoxel(
    double x,
    double y,
    double z) const {
    return {{
        static_cast<int64_t>(std::floor(x / voxel_size_)),
        static_cast<int64_t>(std::floor(y / voxel_size_)),
        static_cast<int64_t>(std::floor(z / voxel_size_)),
    }};
}

std::array<double, 3> VoxelMap::VoxelCenterForKey(int64_t key) const {
    if (key < 0 || static_cast<uint64_t>(key) >= key_count_) {
        return {{
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
        }};
    }
    const int64_t rel_z = key % grid_shape_[2];
    const int64_t rel_x = key / stride_x_;
    const int64_t rel_y = (key % stride_x_) / grid_shape_[2];
    return {{
        (static_cast<double>(rel_x + origin_ijk_[0]) + 0.5) * voxel_size_,
        (static_cast<double>(rel_y + origin_ijk_[1]) + 0.5) * voxel_size_,
        (static_cast<double>(rel_z + origin_ijk_[2]) + 0.5) * voxel_size_,
    }};
}

std::shared_ptr<const VoxelMap> SharedVoxelMapFromHandle(void* handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    return static_cast<VoxelMapHandle*>(handle)->map;
}

}

extern "C" {

void* planning_voxel_map_create(
    const int64_t* occupied_keys,
    int64_t num_keys,
    const int64_t* origin_ijk,
    const int64_t* grid_shape,
    double voxel_size) {
    auto map = planning::geometry::VoxelMap::Create(
        occupied_keys, num_keys, origin_ijk, grid_shape, voxel_size);
    if (!map) {
        return nullptr;
    }
    auto* handle = new planning::geometry::VoxelMapHandle();
    handle->map = std::move(map);
    return handle;
}

void planning_voxel_map_destroy(void* handle) {
    delete static_cast<planning::geometry::VoxelMapHandle*>(handle);
}

}
