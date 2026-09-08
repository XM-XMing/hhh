#pragma once

#include <cstdint>

extern "C" int planning_depth_safety_mask(
    const float* image,
    int32_t height,
    int32_t width,
    const int32_t* pixel_x,
    const int32_t* pixel_y,
    const float* point_depth,
    const int32_t* patch_radius,
    const int32_t* unclipped_radius,
    const uint8_t* projection_valid,
    int32_t action_count,
    int32_t sample_count,
    float valid_depth_min,
    float valid_depth_max,
    float blocked_clearance,
    uint8_t* mask,
    int32_t* action_checked,
    int32_t* action_valid_patches,
    int32_t* action_capped_patches,
    int32_t* action_max_unclipped_radius,
    double* action_invalid_fraction_sum,
    float* action_min_ray_clearance,
    int64_t* counters);
