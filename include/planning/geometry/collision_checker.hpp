#pragma once

#include <cstdint>

extern "C" {

void* planning_collision_create(
    const int64_t* occupied_keys,
    int64_t num_keys,
    const int64_t* origin_ijk,
    const int64_t* grid_shape,
    double voxel_size,
    double inflate_radius);

void* planning_collision_create_from_voxel_map(
    void* voxel_map_handle,
    double inflate_radius);

void planning_collision_destroy(void* handle);
void planning_collision_set_threads(void* handle, int thread_count);
double planning_collision_radius(void* handle);

int planning_collision_check_path(
    void* handle,
    const float* path_xyz,
    int64_t point_count,
    int check_step,
    int* collision,
    double* minimum_distance,
    int32_t* first_collision_index,
    float* first_collision_point_xyz);

int planning_collision_check_action(
    void* handle,
    const float* pos_ref,
    int64_t num_actions,
    int64_t point_count,
    int action_id,
    const float* position,
    double yaw,
    int check_step,
    int* collision,
    double* minimum_distance,
    int32_t* first_collision_index,
    float* first_collision_point_xyz,
    float* endpoint);

int planning_collision_check_actions_batch(
    void* handle,
    const float* pos_ref,
    int64_t num_actions,
    int64_t point_count,
    const int32_t* action_ids,
    int64_t query_count,
    const float* position,
    double yaw,
    int check_step,
    int* collisions,
    double* minimum_distances,
    int32_t* first_collision_indices,
    float* endpoints);

int planning_collision_check_pose_actions_batch(
    void* handle,
    const float* pos_ref,
    int64_t num_actions,
    int64_t point_count,
    const int32_t* action_ids,
    int64_t query_count,
    const float* positions,
    const double* yaws,
    int64_t pose_count,
    int check_step,
    int* collisions,
    double* minimum_distances,
    int32_t* first_collision_indices,
    float* endpoints);

}
