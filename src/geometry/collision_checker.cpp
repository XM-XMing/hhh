#include <planning/geometry/collision_checker.hpp>
#include <planning/geometry/voxel_map.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct CollisionChecker {
  std::shared_ptr<const planning::geometry::VoxelMap> map;
  const uint64_t* occupied_bits = nullptr;
  uint64_t key_count = 0;
  int64_t origin[3] = {0, 0, 0};
  int64_t grid_shape[3] = {0, 0, 0};
  int64_t stride_x = 0;
  int64_t stride_y = 0;
  double voxel_size = 0.0;
  double inflate_radius = 0.0;
  double collision_radius = 0.0;
  int thread_count = 1;
  std::vector<int64_t> offsets;

  bool has_key(const int64_t key) const {
    if (key < 0 || static_cast<uint64_t>(key) >= key_count) {
      return false;
    }
    const uint64_t value = static_cast<uint64_t>(key);
    return (occupied_bits[value >> 6] & (uint64_t{1} << (value & 63))) != 0;
  }

  void build_neighbor_offsets() {
    offsets.clear();
    const int radius_voxels = static_cast<int>(std::ceil(collision_radius / voxel_size));
    const int width = 2 * radius_voxels + 1;
    offsets.reserve(static_cast<size_t>(width * width * width * 3));
    for (int x = -radius_voxels; x <= radius_voxels; ++x) {
      for (int y = -radius_voxels; y <= radius_voxels; ++y) {
        for (int z = -radius_voxels; z <= radius_voxels; ++z) {
          const double dx = std::max(0.0, std::abs(static_cast<double>(x)) - 0.5);
          const double dy = std::max(0.0, std::abs(static_cast<double>(y)) - 0.5);
          const double dz = std::max(0.0, std::abs(static_cast<double>(z)) - 0.5);
          const double minimum_possible_distance =
              voxel_size * std::sqrt(dx * dx + dy * dy + dz * dz);
          if (minimum_possible_distance > collision_radius) {
            continue;
          }
          offsets.push_back(static_cast<int64_t>(x));
          offsets.push_back(static_cast<int64_t>(y));
          offsets.push_back(static_cast<int64_t>(z));
        }
      }
    }
  }
};

CollisionChecker* CreateCollisionChecker(
    std::shared_ptr<const planning::geometry::VoxelMap> map,
    double inflate_radius) {
  // Collision's historical contract rejects an empty occupancy map. Route
  // planning may represent an empty map, but that must not widen the C2
  // collision input contract.
  if (!map || !map->valid() || map->occupied_keys().empty() || inflate_radius < 0.0) {
    return nullptr;
  }
  auto* checker = new CollisionChecker();
  checker->map = std::move(map);
  checker->occupied_bits = checker->map->occupied_bits_data();
  checker->key_count = checker->map->key_count();
  checker->origin[0] = checker->map->origin_ijk()[0];
  checker->origin[1] = checker->map->origin_ijk()[1];
  checker->origin[2] = checker->map->origin_ijk()[2];
  checker->grid_shape[0] = checker->map->grid_shape()[0];
  checker->grid_shape[1] = checker->map->grid_shape()[1];
  checker->grid_shape[2] = checker->map->grid_shape()[2];
  checker->stride_x = checker->map->stride_x();
  checker->stride_y = checker->map->stride_y();
  checker->voxel_size = checker->map->voxel_size();
  checker->inflate_radius = inflate_radius;
  checker->collision_radius =
      inflate_radius + 0.5 * std::sqrt(3.0) * checker->voxel_size;
  checker->build_neighbor_offsets();
  return checker;
}

inline bool check_point(
    const CollisionChecker* checker,
    const double px,
    const double py,
    const double pz,
    double* minimum_distance) {
  const int64_t qx = static_cast<int64_t>(std::floor(px / checker->voxel_size));
  const int64_t qy = static_cast<int64_t>(std::floor(py / checker->voxel_size));
  const int64_t qz = static_cast<int64_t>(std::floor(pz / checker->voxel_size));

  double local_minimum = std::numeric_limits<double>::infinity();
  bool found = false;
  for (size_t offset = 0; offset < checker->offsets.size(); offset += 3) {
    const int64_t ix = qx + checker->offsets[offset];
    const int64_t iy = qy + checker->offsets[offset + 1];
    const int64_t iz = qz + checker->offsets[offset + 2];
    const int64_t rx = ix - checker->origin[0];
    const int64_t ry = iy - checker->origin[1];
    const int64_t rz = iz - checker->origin[2];
    if (rx < 0 || ry < 0 || rz < 0 ||
        rx >= checker->grid_shape[0] || ry >= checker->grid_shape[1] ||
        rz >= checker->grid_shape[2]) {
      continue;
    }
    const int64_t key = rx * checker->stride_x + ry * checker->stride_y + rz;
    if (!checker->has_key(key)) {
      continue;
    }
    found = true;
    const double dx = (static_cast<double>(ix) + 0.5) * checker->voxel_size - px;
    const double dy = (static_cast<double>(iy) + 0.5) * checker->voxel_size - py;
    const double dz = (static_cast<double>(iz) + 0.5) * checker->voxel_size - pz;
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    local_minimum = std::min(local_minimum, distance);
  }
  if (found) {
    *minimum_distance = std::min(*minimum_distance, local_minimum);
    return local_minimum <= checker->collision_radius;
  }
  return false;
}

inline int check_transformed_action(
    const CollisionChecker* checker,
    const float* action_path,
    const int64_t point_count,
    const float* position,
    const double cosine,
    const double sine,
    const int check_step,
    int* collision,
    double* minimum_distance,
    int32_t* first_collision_index,
    float* endpoint) {
  *collision = 0;
  *minimum_distance = std::numeric_limits<double>::infinity();
  *first_collision_index = -1;
  const int step = std::max(1, check_step);
  for (int64_t index = 0; index < point_count; index += step) {
    const float* local = action_path + 3 * index;
    const double x = static_cast<double>(position[0]) + cosine * local[0] - sine * local[1];
    const double y = static_cast<double>(position[1]) + sine * local[0] + cosine * local[1];
    const double z = static_cast<double>(position[2]) + local[2];
    if (check_point(checker, x, y, z, minimum_distance)) {
      *collision = 1;
      *first_collision_index = static_cast<int32_t>(index);
      break;
    }
  }
  if (endpoint) {
    const float* local = action_path + 3 * (point_count - 1);
    endpoint[0] = static_cast<float>(static_cast<double>(position[0]) + cosine * local[0] - sine * local[1]);
    endpoint[1] = static_cast<float>(static_cast<double>(position[1]) + sine * local[0] + cosine * local[1]);
    endpoint[2] = static_cast<float>(static_cast<double>(position[2]) + local[2]);
  }
  return 0;
}


}  // namespace

extern "C" {

void* planning_collision_create(
    const int64_t* occupied_keys,
    const int64_t num_keys,
    const int64_t* origin_ijk,
    const int64_t* grid_shape,
    const double voxel_size,
    const double inflate_radius) {
  auto map = planning::geometry::VoxelMap::Create(
      occupied_keys, num_keys, origin_ijk, grid_shape, voxel_size);
  return CreateCollisionChecker(std::move(map), inflate_radius);
}

void* planning_collision_create_from_voxel_map(
    void* voxel_map_handle,
    const double inflate_radius) {
  return CreateCollisionChecker(
      planning::geometry::SharedVoxelMapFromHandle(voxel_map_handle),
      inflate_radius);
}

void planning_collision_destroy(void* handle) {
  delete static_cast<CollisionChecker*>(handle);
}

void planning_collision_set_threads(void* handle, const int thread_count) {
  CollisionChecker* checker = static_cast<CollisionChecker*>(handle);
  if (checker) {
    checker->thread_count = std::max(1, thread_count);
  }
}

double planning_collision_radius(void* handle) {
  const CollisionChecker* checker = static_cast<CollisionChecker*>(handle);
  return checker ? checker->collision_radius : std::numeric_limits<double>::quiet_NaN();
}

int planning_collision_check_path(
    void* handle,
    const float* path_xyz,
    const int64_t point_count,
    const int check_step,
    int* collision,
    double* minimum_distance,
    int32_t* first_collision_index,
    float* first_collision_point_xyz) {
  CollisionChecker* checker = static_cast<CollisionChecker*>(handle);
  if (!checker || !path_xyz || point_count <= 0 || !collision || !minimum_distance ||
      !first_collision_index) {
    return -1;
  }
  *collision = 0;
  *minimum_distance = std::numeric_limits<double>::infinity();
  *first_collision_index = -1;
  if (first_collision_point_xyz) {
    std::fill(first_collision_point_xyz, first_collision_point_xyz + 3, 0.0f);
  }
  const int step = std::max(1, check_step);
  for (int64_t index = 0; index < point_count; index += step) {
    const float* point = path_xyz + 3 * index;
    if (check_point(checker, point[0], point[1], point[2], minimum_distance)) {
      *collision = 1;
      *first_collision_index = static_cast<int32_t>(index);
      if (first_collision_point_xyz) {
        std::copy(point, point + 3, first_collision_point_xyz);
      }
      break;
    }
  }
  return 0;
}

int planning_collision_check_action(
    void* handle,
    const float* pos_ref,
    const int64_t num_actions,
    const int64_t point_count,
    const int action_id,
    const float* position,
    const double yaw,
    const int check_step,
    int* collision,
    double* minimum_distance,
    int32_t* first_collision_index,
    float* first_collision_point_xyz,
    float* endpoint) {
  CollisionChecker* checker = static_cast<CollisionChecker*>(handle);
  if (!checker || !pos_ref || !position || action_id < 0 || action_id >= num_actions ||
      point_count <= 0 || !collision || !minimum_distance || !first_collision_index) {
    return -1;
  }
  if (first_collision_point_xyz) {
    std::fill(first_collision_point_xyz, first_collision_point_xyz + 3, 0.0f);
  }
  const float* action_path = pos_ref + static_cast<int64_t>(action_id) * point_count * 3;
  const int result = check_transformed_action(
      checker, action_path, point_count, position, std::cos(yaw), std::sin(yaw), check_step,
      collision, minimum_distance, first_collision_index, endpoint);
  if (result == 0 && *collision && first_collision_point_xyz) {
    const float* local = action_path + 3 * (*first_collision_index);
    first_collision_point_xyz[0] = static_cast<float>(position[0] + std::cos(yaw) * local[0] - std::sin(yaw) * local[1]);
    first_collision_point_xyz[1] = static_cast<float>(position[1] + std::sin(yaw) * local[0] + std::cos(yaw) * local[1]);
    first_collision_point_xyz[2] = position[2] + local[2];
  }
  return result;
}

int planning_collision_check_actions_batch(
    void* handle,
    const float* pos_ref,
    const int64_t num_actions,
    const int64_t point_count,
    const int32_t* action_ids,
    const int64_t query_count,
    const float* position,
    const double yaw,
    const int check_step,
    int* collisions,
    double* minimum_distances,
    int32_t* first_collision_indices,
    float* endpoints) {
  CollisionChecker* checker = static_cast<CollisionChecker*>(handle);
  if (!checker || !pos_ref || !action_ids || !position || query_count < 0 ||
      !collisions || !minimum_distances || !first_collision_indices || !endpoints) {
    return -1;
  }
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(checker->thread_count) if(query_count > 1)
#endif
  for (int64_t query = 0; query < query_count; ++query) {
    const int action_id = action_ids[query];
    if (action_id < 0 || action_id >= num_actions) {
      collisions[query] = -1;
      continue;
    }
    const float* action_path = pos_ref + static_cast<int64_t>(action_id) * point_count * 3;
    check_transformed_action(
        checker, action_path, point_count, position, cosine, sine, check_step,
        collisions + query, minimum_distances + query, first_collision_indices + query,
        endpoints + 3 * query);
  }
  for (int64_t query = 0; query < query_count; ++query) {
    if (collisions[query] < 0) {
      return -2;
    }
  }
  return 0;
}


// Check the same action set from multiple predicted poses in one C call.
// Output arrays are row-major [pose_count, query_count].
int planning_collision_check_pose_actions_batch(
    void* handle,
    const float* pos_ref,
    const int64_t num_actions,
    const int64_t point_count,
    const int32_t* action_ids,
    const int64_t query_count,
    const float* positions,
    const double* yaws,
    const int64_t pose_count,
    const int check_step,
    int* collisions,
    double* minimum_distances,
    int32_t* first_collision_indices,
    float* endpoints) {
  CollisionChecker* checker = static_cast<CollisionChecker*>(handle);
  if (!checker || !pos_ref || !action_ids || !positions || !yaws || pose_count < 0 ||
      query_count < 0 || !collisions || !minimum_distances || !first_collision_indices ||
      !endpoints) {
    return -1;
  }

  // Parallelize by pose rather than by pose-action pair. Each worker reuses the
  // same position, yaw and action table for a contiguous output row. This avoids
  // repeated trigonometry and improves cache locality for teacher lookahead.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(checker->thread_count) if(pose_count > 1)
#endif
  for (int64_t pose_id = 0; pose_id < pose_count; ++pose_id) {
    const float* position = positions + 3 * pose_id;
    const double cosine = std::cos(yaws[pose_id]);
    const double sine = std::sin(yaws[pose_id]);
    const int64_t row_offset = pose_id * query_count;
    for (int64_t query_id = 0; query_id < query_count; ++query_id) {
      const int64_t flat = row_offset + query_id;
      const int action_id = action_ids[query_id];
      if (action_id < 0 || action_id >= num_actions) {
        collisions[flat] = -1;
        continue;
      }
      const float* action_path = pos_ref + static_cast<int64_t>(action_id) * point_count * 3;
      check_transformed_action(
          checker, action_path, point_count, position, cosine, sine, check_step,
          collisions + flat, minimum_distances + flat,
          first_collision_indices + flat, endpoints + 3 * flat);
    }
  }
  for (int64_t flat = 0; flat < pose_count * query_count; ++flat) {
    if (collisions[flat] < 0) {
      return -2;
    }
  }
  return 0;
}


}  // extern "C"
