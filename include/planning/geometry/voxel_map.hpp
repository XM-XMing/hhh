#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace planning::geometry {

class VoxelMap final {
public:
    static std::shared_ptr<const VoxelMap> Create(
        const int64_t* occupied_keys,
        int64_t num_keys,
        const int64_t* origin_ijk,
        const int64_t* grid_shape,
        double voxel_size);

    bool valid() const { return valid_; }
    double voxel_size() const { return voxel_size_; }
    uint64_t key_count() const { return key_count_; }
    const std::array<int64_t, 3>& origin_ijk() const { return origin_ijk_; }
    const std::array<int64_t, 3>& grid_shape() const { return grid_shape_; }
    int64_t stride_x() const { return stride_x_; }
    int64_t stride_y() const { return stride_y_; }
    const std::vector<int64_t>& occupied_keys() const { return occupied_keys_; }
    // Read-only view used by collision consumers. The bitmap remains owned by
    // this immutable VoxelMap; callers must retain their shared owner.
    const uint64_t* occupied_bits_data() const { return occupied_bits_.data(); }

    bool HasKey(int64_t key) const;
    bool IsOccupiedVoxel(int64_t ix, int64_t iy, int64_t iz) const;
    std::array<int64_t, 3> WorldToVoxel(double x, double y, double z) const;
    std::array<double, 3> VoxelCenterForKey(int64_t key) const;

private:
    VoxelMap() = default;
    bool Build(
        const int64_t* occupied_keys,
        int64_t num_keys,
        const int64_t* origin_ijk,
        const int64_t* grid_shape,
        double voxel_size);

    bool valid_ = false;
    std::array<int64_t, 3> origin_ijk_{{0, 0, 0}};
    std::array<int64_t, 3> grid_shape_{{0, 0, 0}};
    int64_t stride_x_ = 0;
    int64_t stride_y_ = 0;
    double voxel_size_ = 0.0;
    uint64_t key_count_ = 0;
    std::vector<int64_t> occupied_keys_;
    std::vector<uint64_t> occupied_bits_;
};

std::shared_ptr<const VoxelMap> SharedVoxelMapFromHandle(void* handle);

}

extern "C" {

void* planning_voxel_map_create(
    const int64_t* occupied_keys,
    int64_t num_keys,
    const int64_t* origin_ijk,
    const int64_t* grid_shape,
    double voxel_size);

void planning_voxel_map_destroy(void* handle);

}
