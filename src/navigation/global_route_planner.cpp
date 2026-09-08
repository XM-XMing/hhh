#include <planning/navigation/global_route_planner.hpp>
#include <planning/geometry/voxel_map.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

namespace {

struct QueueEntry {
    double priority = 0.0;
    int32_t cell_id = -1;
};

struct QueueEntryGreater {
    bool operator()(const QueueEntry& lhs, const QueueEntry& rhs) const {
        if (lhs.priority != rhs.priority) {
            return lhs.priority > rhs.priority;
        }
        return lhs.cell_id > rhs.cell_id;
    }
};

class GlobalRoutePlanner {
public:
    GlobalRoutePlanner(
        std::shared_ptr<const planning::geometry::VoxelMap> map,
        double collision_radius,
        double resolution_m,
        double flight_z_min_m,
        double flight_z_max_m,
        double tracking_margin_m,
        int nearest_free_radius_cells)
        : map_(std::move(map)),
          collision_radius_(collision_radius),
          resolution_m_(resolution_m),
          flight_z_min_m_(flight_z_min_m),
          flight_z_max_m_(flight_z_max_m),
          tracking_margin_m_(tracking_margin_m),
          nearest_free_radius_cells_(std::max(0, nearest_free_radius_cells)) {
        if (!map_ || !map_->valid() || resolution_m_ <= 0.0) {
            return;
        }
        const auto& origin_ijk = map_->origin_ijk();
        const auto& grid_shape = map_->grid_shape();
        const double voxel_size = map_->voxel_size();
        const double lower_x = static_cast<double>(origin_ijk[0]) * voxel_size;
        const double lower_y = static_cast<double>(origin_ijk[1]) * voxel_size;
        const double upper_x = static_cast<double>(origin_ijk[0] + grid_shape[0]) * voxel_size;
        const double upper_y = static_cast<double>(origin_ijk[1] + grid_shape[1]) * voxel_size;
        origin_xy_[0] = std::floor(lower_x / resolution_m_) * resolution_m_;
        origin_xy_[1] = std::floor(lower_y / resolution_m_) * resolution_m_;
        shape_[0] = static_cast<int32_t>(std::ceil((upper_x - origin_xy_[0]) / resolution_m_));
        shape_[1] = static_cast<int32_t>(std::ceil((upper_y - origin_xy_[1]) / resolution_m_));
        const int64_t cell_count = static_cast<int64_t>(shape_[0]) * shape_[1];
        if (cell_count <= 0) {
            return;
        }
        blocked_.assign(static_cast<size_t>(cell_count), 0);
        build_blocked_grid();
        costs_.assign(static_cast<size_t>(cell_count), std::numeric_limits<float>::infinity());
        parents_.assign(static_cast<size_t>(cell_count), -1);
        visit_generation_.assign(static_cast<size_t>(cell_count), 0);
        cost_generation_.assign(static_cast<size_t>(cell_count), 0);
    }

    bool valid() const {
        return map_ && map_->valid() && resolution_m_ > 0.0 &&
               shape_[0] > 0 && shape_[1] > 0;
    }

    int copy_shape(int32_t* shape_xy, double* origin_xy, double* resolution_m) const {
        if (!shape_xy || !origin_xy || !resolution_m) {
            return -1;
        }
        shape_xy[0] = shape_[0];
        shape_xy[1] = shape_[1];
        origin_xy[0] = origin_xy_[0];
        origin_xy[1] = origin_xy_[1];
        *resolution_m = resolution_m_;
        return 0;
    }

    int copy_blocked(uint8_t* blocked, int64_t blocked_count) const {
        if (!blocked || blocked_count != static_cast<int64_t>(blocked_.size())) {
            return -1;
        }
        std::copy(blocked_.begin(), blocked_.end(), blocked);
        return 0;
    }

    int plan(const float* start_xyz, const float* goal_xyz) {
        if (!start_xyz || !goal_xyz) {
            return -1;
        }
        last_route_.clear();
        const auto start_world_cell = world_to_cell(start_xyz[0], start_xyz[1]);
        const auto goal_world_cell = world_to_cell(goal_xyz[0], goal_xyz[1]);
        std::array<int32_t, 2> start_cell{};
        std::array<int32_t, 2> goal_cell{};
        if (!nearest_free(start_world_cell, &start_cell) ||
            !nearest_free(goal_world_cell, &goal_cell)) {
            return 1;
        }

        advance_generation();
        const int32_t ny = shape_[1];
        const int32_t start_id = start_cell[0] * ny + start_cell[1];
        const int32_t goal_id = goal_cell[0] * ny + goal_cell[1];
        set_cost(start_id, 0.0f, -1);

        std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater> queue;
        queue.push(QueueEntry{0.0, start_id});
        static constexpr std::array<int, 8> kDx{{1, -1, 0, 0, 1, 1, -1, -1}};
        static constexpr std::array<int, 8> kDy{{0, 0, 1, -1, 1, -1, 1, -1}};
        const double sqrt_two = std::sqrt(2.0);

        while (!queue.empty()) {
            const QueueEntry entry = queue.top();
            queue.pop();
            const int32_t current_id = entry.cell_id;
            if (visited(current_id)) {
                continue;
            }
            mark_visited(current_id);
            if (current_id == goal_id) {
                break;
            }
            const int32_t x = current_id / ny;
            const int32_t y = current_id % ny;
            for (size_t index = 0; index < kDx.size(); ++index) {
                const int32_t dx = kDx[index];
                const int32_t dy = kDy[index];
                const int32_t xx = x + dx;
                const int32_t yy = y + dy;
                if (!inside(xx, yy) || is_blocked(xx, yy)) {
                    continue;
                }
                if (dx != 0 && dy != 0 &&
                    (is_blocked(x + dx, y) || is_blocked(x, y + dy))) {
                    continue;
                }
                const int32_t neighbor_id = xx * ny + yy;
                const double step_cost = (dx != 0 && dy != 0) ? sqrt_two : 1.0;
                const double candidate = static_cast<double>(cost(current_id)) + step_cost;
                if (candidate >= static_cast<double>(cost(neighbor_id))) {
                    continue;
                }
                set_cost(neighbor_id, static_cast<float>(candidate), current_id);
                const double heuristic = std::hypot(
                    static_cast<double>(goal_cell[0] - xx),
                    static_cast<double>(goal_cell[1] - yy));
                queue.push(QueueEntry{candidate + heuristic, neighbor_id});
            }
        }

        if (!visited(goal_id)) {
            return 1;
        }

        std::vector<int32_t> cells;
        int32_t current = goal_id;
        while (current >= 0) {
            cells.push_back(current);
            if (current == start_id) {
                break;
            }
            current = parent(current);
        }
        if (cells.empty() || cells.back() != start_id) {
            return -2;
        }
        std::reverse(cells.begin(), cells.end());
        build_route(cells, start_xyz, goal_xyz);
        return 0;
    }

    int64_t last_point_count() const {
        return static_cast<int64_t>(last_route_.size() / 3);
    }

    int copy_last_route(float* route_xyz, int64_t point_capacity) const {
        if (!route_xyz || point_capacity < last_point_count()) {
            return -1;
        }
        std::copy(last_route_.begin(), last_route_.end(), route_xyz);
        return 0;
    }

private:
    std::shared_ptr<const planning::geometry::VoxelMap> map_;
    std::array<double, 2> origin_xy_{{0.0, 0.0}};
    std::array<int32_t, 2> shape_{{0, 0}};
    double collision_radius_ = 0.0;
    double resolution_m_ = 0.0;
    double flight_z_min_m_ = 0.0;
    double flight_z_max_m_ = 0.0;
    double tracking_margin_m_ = 0.0;
    int nearest_free_radius_cells_ = 0;
    std::vector<uint8_t> blocked_;
    std::vector<float> costs_;
    std::vector<int32_t> parents_;
    std::vector<uint32_t> visit_generation_;
    std::vector<uint32_t> cost_generation_;
    uint32_t generation_ = 0;
    std::vector<float> last_route_;

    int64_t index(int32_t x, int32_t y) const {
        return static_cast<int64_t>(x) * shape_[1] + y;
    }

    bool inside(int32_t x, int32_t y) const {
        return x >= 0 && y >= 0 && x < shape_[0] && y < shape_[1];
    }

    bool is_blocked(int32_t x, int32_t y) const {
        if (!inside(x, y)) {
            return true;
        }
        return blocked_[static_cast<size_t>(index(x, y))] != 0;
    }

    std::array<int32_t, 2> world_to_cell(double x, double y) const {
        return {{
            static_cast<int32_t>(std::floor((x - origin_xy_[0]) / resolution_m_)),
            static_cast<int32_t>(std::floor((y - origin_xy_[1]) / resolution_m_)),
        }};
    }

    bool nearest_free(
        const std::array<int32_t, 2>& cell,
        std::array<int32_t, 2>* output) const {
        if (!output) {
            return false;
        }
        bool found = false;
        int best_distance = std::numeric_limits<int>::max();
        for (int dx = -nearest_free_radius_cells_; dx <= nearest_free_radius_cells_; ++dx) {
            for (int dy = -nearest_free_radius_cells_; dy <= nearest_free_radius_cells_; ++dy) {
                const int32_t x = cell[0] + dx;
                const int32_t y = cell[1] + dy;
                if (!inside(x, y) || is_blocked(x, y)) {
                    continue;
                }
                const int distance = dx * dx + dy * dy;
                if (distance < best_distance) {
                    (*output)[0] = x;
                    (*output)[1] = y;
                    best_distance = distance;
                    found = true;
                }
            }
        }
        return found;
    }

    void build_blocked_grid() {
        std::vector<uint8_t> occupied(blocked_.size(), 0);
        for (int64_t key : map_->occupied_keys()) {
            const auto center = map_->VoxelCenterForKey(key);
            const double world_x = center[0];
            const double world_y = center[1];
            const double world_z = center[2];
            if (world_z < flight_z_min_m_ - collision_radius_ ||
                world_z > flight_z_max_m_ + collision_radius_) {
                continue;
            }
            const int32_t coarse_x = static_cast<int32_t>(
                std::floor((world_x - origin_xy_[0]) / resolution_m_));
            const int32_t coarse_y = static_cast<int32_t>(
                std::floor((world_y - origin_xy_[1]) / resolution_m_));
            if (inside(coarse_x, coarse_y)) {
                occupied[static_cast<size_t>(index(coarse_x, coarse_y))] = 1;
            }
        }

        blocked_ = occupied;
        const double inflation_m = collision_radius_ + tracking_margin_m_;
        const int radius_cells = static_cast<int>(std::ceil(inflation_m / resolution_m_));
        if (radius_cells <= 0) {
            return;
        }
        for (int32_t x = 0; x < shape_[0]; ++x) {
            for (int32_t y = 0; y < shape_[1]; ++y) {
                if (occupied[static_cast<size_t>(index(x, y))] == 0) {
                    continue;
                }
                for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
                    for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
                        if (dx * dx + dy * dy > radius_cells * radius_cells) {
                            continue;
                        }
                        const int32_t xx = x + dx;
                        const int32_t yy = y + dy;
                        if (inside(xx, yy)) {
                            blocked_[static_cast<size_t>(index(xx, yy))] = 1;
                        }
                    }
                }
            }
        }
    }

    void advance_generation() {
        ++generation_;
        if (generation_ == 0) {
            std::fill(visit_generation_.begin(), visit_generation_.end(), 0);
            std::fill(cost_generation_.begin(), cost_generation_.end(), 0);
            generation_ = 1;
        }
    }

    bool visited(int32_t cell_id) const {
        return visit_generation_[static_cast<size_t>(cell_id)] == generation_;
    }

    void mark_visited(int32_t cell_id) {
        visit_generation_[static_cast<size_t>(cell_id)] = generation_;
    }

    float cost(int32_t cell_id) const {
        if (cost_generation_[static_cast<size_t>(cell_id)] != generation_) {
            return std::numeric_limits<float>::infinity();
        }
        return costs_[static_cast<size_t>(cell_id)];
    }

    int32_t parent(int32_t cell_id) const {
        if (cost_generation_[static_cast<size_t>(cell_id)] != generation_) {
            return -1;
        }
        return parents_[static_cast<size_t>(cell_id)];
    }

    void set_cost(int32_t cell_id, float value, int32_t parent_id) {
        const size_t offset = static_cast<size_t>(cell_id);
        cost_generation_[offset] = generation_;
        costs_[offset] = value;
        parents_[offset] = parent_id;
    }

    void build_route(
        const std::vector<int32_t>& cells,
        const float* start_xyz,
        const float* goal_xyz) {
        const size_t intermediate_count = cells.size();
        last_route_.resize((intermediate_count + 2) * 3);
        last_route_[0] = start_xyz[0];
        last_route_[1] = start_xyz[1];
        last_route_[2] = start_xyz[2];
        for (size_t i = 0; i < intermediate_count; ++i) {
            const int32_t cell_id = cells[i];
            const int32_t x = cell_id / shape_[1];
            const int32_t y = cell_id % shape_[1];
            const float world_x = static_cast<float>(
                origin_xy_[0] + (static_cast<double>(x) + 0.5) * resolution_m_);
            const float world_y = static_cast<float>(
                origin_xy_[1] + (static_cast<double>(y) + 0.5) * resolution_m_);
            float z = start_xyz[2];
            if (intermediate_count > 1) {
                const double ratio = static_cast<double>(i) /
                    static_cast<double>(intermediate_count - 1);
                z = static_cast<float>(
                    static_cast<double>(start_xyz[2]) +
                    (static_cast<double>(goal_xyz[2]) - start_xyz[2]) * ratio);
            }
            const size_t offset = (i + 1) * 3;
            last_route_[offset] = world_x;
            last_route_[offset + 1] = world_y;
            last_route_[offset + 2] = z;
        }
        const size_t goal_offset = (intermediate_count + 1) * 3;
        last_route_[goal_offset] = goal_xyz[0];
        last_route_[goal_offset + 1] = goal_xyz[1];
        last_route_[goal_offset + 2] = goal_xyz[2];
    }
};

void* CreateGlobalRoutePlanner(
    std::shared_ptr<const planning::geometry::VoxelMap> map,
    const double collision_radius,
    const double resolution_m,
    const double flight_z_min_m,
    const double flight_z_max_m,
    const double tracking_margin_m,
    const int nearest_free_radius_cells) {
    if (!map || collision_radius < 0.0 || resolution_m <= 0.0) {
        return nullptr;
    }
    auto* planner = new GlobalRoutePlanner(
        std::move(map),
        collision_radius,
        resolution_m,
        flight_z_min_m,
        flight_z_max_m,
        tracking_margin_m,
        nearest_free_radius_cells);
    if (!planner->valid()) {
        delete planner;
        return nullptr;
    }
    return planner;
}

}

extern "C" {

void* planning_global_route_create(
    const int64_t* occupied_keys,
    const int64_t num_keys,
    const int64_t* origin_ijk,
    const int64_t* grid_shape,
    const double voxel_size,
    const double collision_radius,
    const double resolution_m,
    const double flight_z_min_m,
    const double flight_z_max_m,
    const double tracking_margin_m,
    const int nearest_free_radius_cells) {
    auto map = planning::geometry::VoxelMap::Create(
        occupied_keys, num_keys, origin_ijk, grid_shape, voxel_size);
    return CreateGlobalRoutePlanner(
        std::move(map), collision_radius, resolution_m, flight_z_min_m,
        flight_z_max_m, tracking_margin_m, nearest_free_radius_cells);
}

void* planning_global_route_create_from_voxel_map(
    void* voxel_map_handle,
    const double collision_radius,
    const double resolution_m,
    const double flight_z_min_m,
    const double flight_z_max_m,
    const double tracking_margin_m,
    const int nearest_free_radius_cells) {
    return CreateGlobalRoutePlanner(
        planning::geometry::SharedVoxelMapFromHandle(voxel_map_handle),
        collision_radius, resolution_m, flight_z_min_m, flight_z_max_m,
        tracking_margin_m, nearest_free_radius_cells);
}

void planning_global_route_destroy(void* handle) {
    delete static_cast<GlobalRoutePlanner*>(handle);
}

int planning_global_route_shape(
    void* handle,
    int32_t* shape_xy,
    double* origin_xy,
    double* resolution_m) {
    auto* planner = static_cast<GlobalRoutePlanner*>(handle);
    return planner ? planner->copy_shape(shape_xy, origin_xy, resolution_m) : -1;
}

int planning_global_route_copy_blocked(
    void* handle,
    uint8_t* blocked,
    const int64_t blocked_count) {
    auto* planner = static_cast<GlobalRoutePlanner*>(handle);
    return planner ? planner->copy_blocked(blocked, blocked_count) : -1;
}

int planning_global_route_plan(
    void* handle,
    const float* start_xyz,
    const float* goal_xyz) {
    auto* planner = static_cast<GlobalRoutePlanner*>(handle);
    return planner ? planner->plan(start_xyz, goal_xyz) : -1;
}

int64_t planning_global_route_last_point_count(void* handle) {
    auto* planner = static_cast<GlobalRoutePlanner*>(handle);
    return planner ? planner->last_point_count() : -1;
}

int planning_global_route_copy_last_route(
    void* handle,
    float* route_xyz,
    const int64_t point_capacity) {
    auto* planner = static_cast<GlobalRoutePlanner*>(handle);
    return planner ? planner->copy_last_route(route_xyz, point_capacity) : -1;
}

}
