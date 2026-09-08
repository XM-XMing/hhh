#include <planning/geometry/depth_safety.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

extern "C" int planning_depth_safety_mask(
    const float* image,
    const int32_t height,
    const int32_t width,
    const int32_t* pixel_x,
    const int32_t* pixel_y,
    const float* point_depth,
    const int32_t* patch_radius,
    const int32_t* unclipped_radius,
    const uint8_t* projection_valid,
    const int32_t action_count,
    const int32_t sample_count,
    const float valid_depth_min,
    const float valid_depth_max,
    const float blocked_clearance,
    uint8_t* mask,
    int32_t* action_checked,
    int32_t* action_valid_patches,
    int32_t* action_capped_patches,
    int32_t* action_max_unclipped_radius,
    double* action_invalid_fraction_sum,
    float* action_min_ray_clearance,
    int64_t* counters) {
  if (!image || !pixel_x || !pixel_y || !point_depth || !patch_radius ||
      !unclipped_radius || !projection_valid || !mask || !action_checked ||
      !action_valid_patches || !action_capped_patches ||
      !action_max_unclipped_radius || !action_invalid_fraction_sum ||
      !action_min_ray_clearance || !counters || height <= 0 || width <= 0 ||
      action_count <= 0 || sample_count < 0) {
    return -1;
  }

  counters[0] = 0;  // checked samples
  counters[1] = 0;  // visible samples
  counters[2] = 0;  // blocked samples
  for (int32_t action = 0; action < action_count; ++action) {
    mask[action] = 1;
    action_checked[action] = 0;
    action_valid_patches[action] = 0;
    action_capped_patches[action] = 0;
    action_max_unclipped_radius[action] = 0;
    action_invalid_fraction_sum[action] = 0.0;
    action_min_ray_clearance[action] = std::numeric_limits<float>::quiet_NaN();

    for (int32_t sample = 0; sample < sample_count; ++sample) {
      const int64_t table_index =
          static_cast<int64_t>(action) * sample_count + sample;
      if (!projection_valid[table_index]) {
        continue;
      }
      ++counters[0];
      ++action_checked[action];
      const int32_t raw_radius = unclipped_radius[table_index];
      action_max_unclipped_radius[action] =
          std::max(action_max_unclipped_radius[action], raw_radius);
      const int32_t radius = patch_radius[table_index];
      if (radius < raw_radius) {
        ++action_capped_patches[action];
      }
      const int32_t x0 = std::max(0, pixel_x[table_index] - radius);
      const int32_t x1 = std::min(width, pixel_x[table_index] + radius + 1);
      const int32_t y0 = std::max(0, pixel_y[table_index] - radius);
      const int32_t y1 = std::min(height, pixel_y[table_index] + radius + 1);

      int64_t valid_count = 0;
      const int64_t pixel_count =
          static_cast<int64_t>(x1 - x0) * static_cast<int64_t>(y1 - y0);
      float minimum_depth = std::numeric_limits<float>::infinity();
      for (int32_t y = y0; y < y1; ++y) {
        const int64_t row = static_cast<int64_t>(y) * width;
        for (int32_t x = x0; x < x1; ++x) {
          const float value = image[row + x];
          if (std::isfinite(value) && value >= valid_depth_min &&
              value < valid_depth_max) {
            ++valid_count;
            minimum_depth = std::min(minimum_depth, value);
          }
        }
      }
      if (pixel_count <= 0) {
        return -2;
      }
      action_invalid_fraction_sum[action] +=
          1.0 - static_cast<double>(valid_count) /
                    static_cast<double>(pixel_count);
      if (valid_count == 0) {
        continue;
      }
      ++counters[1];
      ++action_valid_patches[action];
      const float clearance = minimum_depth - point_depth[table_index];
      if (!std::isfinite(action_min_ray_clearance[action])) {
        action_min_ray_clearance[action] = clearance;
      } else {
        action_min_ray_clearance[action] =
            std::min(action_min_ray_clearance[action], clearance);
      }
      if (clearance <= blocked_clearance) {
        mask[action] = 0;
        ++counters[2];
        break;
      }
    }
  }
  return 0;
}
