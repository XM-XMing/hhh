#pragma once

#include <cstdint>

extern "C" {

void* planning_global_route_create(
    const int64_t* occupied_keys,
    int64_t num_keys,
    const int64_t* origin_ijk,
    const int64_t* grid_shape,
    double voxel_size,
    double collision_radius,
    double resolution_m,
    double flight_z_min_m,
    double flight_z_max_m,
    double tracking_margin_m,
    int nearest_free_radius_cells);

void* planning_global_route_create_from_voxel_map(
    void* voxel_map_handle,
    double collision_radius,
    double resolution_m,
    double flight_z_min_m,
    double flight_z_max_m,
    double tracking_margin_m,
    int nearest_free_radius_cells);

void planning_global_route_destroy(void* handle);

int planning_global_route_shape(
    void* handle,
    int32_t* shape_xy,
    double* origin_xy,
    double* resolution_m);

int planning_global_route_copy_blocked(
    void* handle,
    uint8_t* blocked,
    int64_t blocked_count);

int planning_global_route_plan(
    void* handle,
    const float* start_xyz,
    const float* goal_xyz);

int64_t planning_global_route_last_point_count(void* handle);

int planning_global_route_copy_last_route(
    void* handle,
    float* route_xyz,
    int64_t point_capacity);

}
