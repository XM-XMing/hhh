#include <planning/geometry/collision_checker.hpp>
#include <planning/geometry/voxel_map.hpp>
#include <planning/navigation/global_route_planner.hpp>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using CollisionCreateFn = void* (*) (
    const int64_t*, int64_t, const int64_t*, const int64_t*, double, double);
using CollisionDestroyFn = void (*) (void*);
using CollisionCheckPathFn = int (*) (
    void*, const float*, int64_t, int, int*, double*, int32_t*, float*);

struct OriginalCollisionApi {
    void* library = nullptr;
    CollisionCreateFn create = nullptr;
    CollisionDestroyFn destroy = nullptr;
    CollisionCheckPathFn check_path = nullptr;
};

struct Fixture {
    static constexpr double kVoxelSize = 0.25;
    static constexpr int64_t kOrigin[3] = {-6, -4, 1};
    static constexpr int64_t kShape[3] = {18, 16, 9};

    std::vector<int64_t> occupied;

    Fixture() {
        const int64_t key_count = kShape[0] * kShape[1] * kShape[2];
        for (int64_t key = 0; key < key_count; ++key) {
            // Deterministic sparse occupancy with several domain boundaries.
            const uint64_t mixed =
                static_cast<uint64_t>(key) * 0x9e3779b97f4a7c15ULL +
                0x243f6a8885a308d3ULL;
            if ((mixed ^ (mixed >> 29)) % 23U < 2U) {
                occupied.push_back(key);
            }
        }
        occupied.push_back(0);
        occupied.push_back(key_count - 1);
        occupied.push_back(kShape[1] * kShape[2]);
        occupied.push_back(kShape[2] - 1);
        std::sort(occupied.begin(), occupied.end());
        occupied.erase(std::unique(occupied.begin(), occupied.end()), occupied.end());
    }
};

constexpr double Fixture::kVoxelSize;
constexpr int64_t Fixture::kOrigin[3];
constexpr int64_t Fixture::kShape[3];

std::array<int64_t, 3> OriginalWorldToVoxel(
    double x, double y, double z, double voxel_size) {
    return {{
        static_cast<int64_t>(std::floor(x / voxel_size)),
        static_cast<int64_t>(std::floor(y / voxel_size)),
        static_cast<int64_t>(std::floor(z / voxel_size)),
    }};
}

bool OriginalOccupied(
    const std::vector<int64_t>& occupied,
    const int64_t origin[3],
    const int64_t shape[3],
    int64_t ix,
    int64_t iy,
    int64_t iz) {
    const int64_t rx = ix - origin[0];
    const int64_t ry = iy - origin[1];
    const int64_t rz = iz - origin[2];
    if (rx < 0 || ry < 0 || rz < 0 || rx >= shape[0] || ry >= shape[1] ||
        rz >= shape[2]) {
        return false;
    }
    const int64_t key = rx * shape[1] * shape[2] + ry * shape[2] + rz;
    return std::binary_search(occupied.begin(), occupied.end(), key);
}

std::array<double, 3> OriginalCenterForKey(
    int64_t key,
    const int64_t origin[3],
    const int64_t shape[3],
    double voxel_size) {
    const int64_t stride_x = shape[1] * shape[2];
    const int64_t rel_z = key % shape[2];
    const int64_t rel_x = key / stride_x;
    const int64_t rel_y = (key % stride_x) / shape[2];
    return {{
        (static_cast<double>(rel_x + origin[0]) + 0.5) * voxel_size,
        (static_cast<double>(rel_y + origin[1]) + 0.5) * voxel_size,
        (static_cast<double>(rel_z + origin[2]) + 0.5) * voxel_size,
    }};
}

uint64_t Fnv1aKeys(const std::vector<int64_t>& keys) {
    uint64_t hash = 1469598103934665603ULL;
    for (const int64_t key : keys) {
        uint64_t value = static_cast<uint64_t>(key);
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= value & 0xffU;
            hash *= 1099511628211ULL;
            value >>= 8;
        }
    }
    return hash;
}

bool CloseEnough(double lhs, double rhs) {
    if (std::isinf(lhs) || std::isinf(rhs)) {
        return std::isinf(lhs) && std::isinf(rhs) &&
               std::signbit(lhs) == std::signbit(rhs);
    }
    return std::abs(lhs - rhs) <= 1.0e-12 * std::max(1.0, std::abs(rhs));
}

OriginalCollisionApi LoadOriginalCollisionApi() {
    OriginalCollisionApi api;
    const char* path = std::getenv("XM_ORIGINAL_COLLISION_LIB");
    if (!path || *path == '\0') {
        return api;
    }
    api.library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!api.library) {
        return api;
    }
    api.create = reinterpret_cast<CollisionCreateFn>(
        dlsym(api.library, "planning_collision_create"));
    api.destroy = reinterpret_cast<CollisionDestroyFn>(
        dlsym(api.library, "planning_collision_destroy"));
    api.check_path = reinterpret_cast<CollisionCheckPathFn>(
        dlsym(api.library, "planning_collision_check_path"));
    if (!api.create || !api.destroy || !api.check_path) {
        dlclose(api.library);
        api = OriginalCollisionApi{};
    }
    return api;
}

void UnloadOriginalCollisionApi(OriginalCollisionApi* api) {
    if (api && api->library) {
        dlclose(api->library);
        *api = OriginalCollisionApi{};
    }
}

bool CheckVoxelMapParity(
    const Fixture& fixture,
    const std::shared_ptr<const planning::geometry::VoxelMap>& map) {
    bool metadata_ok = map && map->valid();
    metadata_ok = metadata_ok && map->voxel_size() == Fixture::kVoxelSize;
    metadata_ok = metadata_ok && map->key_count() ==
        static_cast<uint64_t>(Fixture::kShape[0] * Fixture::kShape[1] * Fixture::kShape[2]);
    metadata_ok = metadata_ok && map->origin_ijk() ==
        std::array<int64_t, 3>{{Fixture::kOrigin[0], Fixture::kOrigin[1], Fixture::kOrigin[2]}};
    metadata_ok = metadata_ok && map->grid_shape() ==
        std::array<int64_t, 3>{{Fixture::kShape[0], Fixture::kShape[1], Fixture::kShape[2]}};
    metadata_ok = metadata_ok && map->stride_y() == Fixture::kShape[2];
    metadata_ok = metadata_ok && map->stride_x() == Fixture::kShape[1] * Fixture::kShape[2];
    metadata_ok = metadata_ok && map->occupied_keys() == fixture.occupied;

    const uint64_t expected_hash = Fnv1aKeys(fixture.occupied);
    const bool occupancy_hash_ok = Fnv1aKeys(map->occupied_keys()) == expected_hash;

    bool occupancy_ok = metadata_ok && occupancy_hash_ok;
    bool coordinate_ok = metadata_ok;
    std::mt19937_64 rng(0x584d464c49474854ULL);
    std::uniform_real_distribution<double> x_distribution(
        (Fixture::kOrigin[0] - 2) * Fixture::kVoxelSize,
        (Fixture::kOrigin[0] + Fixture::kShape[0] + 2) * Fixture::kVoxelSize);
    std::uniform_real_distribution<double> y_distribution(
        (Fixture::kOrigin[1] - 2) * Fixture::kVoxelSize,
        (Fixture::kOrigin[1] + Fixture::kShape[1] + 2) * Fixture::kVoxelSize);
    std::uniform_real_distribution<double> z_distribution(
        (Fixture::kOrigin[2] - 2) * Fixture::kVoxelSize,
        (Fixture::kOrigin[2] + Fixture::kShape[2] + 2) * Fixture::kVoxelSize);
    for (int sample = 0; sample < 1000; ++sample) {
        const double x = x_distribution(rng);
        const double y = y_distribution(rng);
        const double z = z_distribution(rng);
        const auto expected = OriginalWorldToVoxel(x, y, z, Fixture::kVoxelSize);
        const auto actual = map->WorldToVoxel(x, y, z);
        coordinate_ok = coordinate_ok && actual == expected;
        occupancy_ok = occupancy_ok &&
            map->IsOccupiedVoxel(actual[0], actual[1], actual[2]) ==
            OriginalOccupied(
                fixture.occupied, Fixture::kOrigin, Fixture::kShape,
                expected[0], expected[1], expected[2]);
    }

    std::vector<std::array<double, 3>> boundary_points;
    const double lower[3] = {
        Fixture::kOrigin[0] * Fixture::kVoxelSize,
        Fixture::kOrigin[1] * Fixture::kVoxelSize,
        Fixture::kOrigin[2] * Fixture::kVoxelSize,
    };
    const double upper[3] = {
        (Fixture::kOrigin[0] + Fixture::kShape[0]) * Fixture::kVoxelSize,
        (Fixture::kOrigin[1] + Fixture::kShape[1]) * Fixture::kVoxelSize,
        (Fixture::kOrigin[2] + Fixture::kShape[2]) * Fixture::kVoxelSize,
    };
    for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
            auto point = std::array<double, 3>{{lower[0], lower[1], lower[2]}};
            point[axis] = side == 0
                ? std::nextafter(lower[axis], -std::numeric_limits<double>::infinity())
                : std::nextafter(upper[axis], -std::numeric_limits<double>::infinity());
            boundary_points.push_back(point);
            point[axis] = side == 0 ? lower[axis] : upper[axis];
            boundary_points.push_back(point);
        }
    }
    boundary_points.push_back({{upper[0], upper[1], upper[2]}});
    boundary_points.push_back({{
        std::nextafter(upper[0], -std::numeric_limits<double>::infinity()),
        std::nextafter(upper[1], -std::numeric_limits<double>::infinity()),
        std::nextafter(upper[2], -std::numeric_limits<double>::infinity()),
    }});
    for (const auto& point : boundary_points) {
        const auto expected = OriginalWorldToVoxel(
            point[0], point[1], point[2], Fixture::kVoxelSize);
        const auto actual = map->WorldToVoxel(point[0], point[1], point[2]);
        coordinate_ok = coordinate_ok && actual == expected;
        occupancy_ok = occupancy_ok &&
            map->IsOccupiedVoxel(actual[0], actual[1], actual[2]) ==
            OriginalOccupied(
                fixture.occupied, Fixture::kOrigin, Fixture::kShape,
                expected[0], expected[1], expected[2]);
    }

    for (const int64_t key : fixture.occupied) {
        const auto expected = OriginalCenterForKey(
            key, Fixture::kOrigin, Fixture::kShape, Fixture::kVoxelSize);
        const auto actual = map->VoxelCenterForKey(key);
        for (int axis = 0; axis < 3; ++axis) {
            coordinate_ok = coordinate_ok && CloseEnough(actual[axis], expected[axis]);
        }
    }
    const auto invalid_center = map->VoxelCenterForKey(-1);
    coordinate_ok = coordinate_ok && std::isnan(invalid_center[0]) &&
        std::isnan(invalid_center[1]) && std::isnan(invalid_center[2]);

    std::cout << "FIXTURE_OCCUPIED_COUNT=" << fixture.occupied.size() << '\n';
    std::cout << "FIXTURE_OCCUPIED_KEY_HASH=0x" << std::hex << expected_hash
              << std::dec << '\n';
    std::cout << "RANDOM_POINT_CASES=1000\n";
    std::cout << "BOUNDARY_POINT_CASES=" << boundary_points.size() << '\n';
    std::cout << "METADATA_PARITY=" << (metadata_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "OCCUPANCY_PARITY="
              << (occupancy_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "COORDINATE_CONVERSION_PARITY="
              << (coordinate_ok ? "PASS" : "FAIL") << '\n';
    return metadata_ok && occupancy_ok && coordinate_ok;
}

bool CheckOriginalRuntimeParity(
    const Fixture& fixture,
    const std::shared_ptr<const planning::geometry::VoxelMap>& map) {
    OriginalCollisionApi original = LoadOriginalCollisionApi();
    if (!original.library) {
        std::cout << "ORIGINAL_COLLISION_RUNTIME=NOT_RUN\n";
        std::cout << "ORIGINAL_COLLISION_REASON=XM_ORIGINAL_COLLISION_LIB unavailable\n";
        return false;
    }

    void* original_checker = original.create(
        fixture.occupied.data(), static_cast<int64_t>(fixture.occupied.size()),
        Fixture::kOrigin, Fixture::kShape, Fixture::kVoxelSize, 0.0);
    void* optimized_checker = planning_collision_create(
        fixture.occupied.data(), static_cast<int64_t>(fixture.occupied.size()),
        Fixture::kOrigin, Fixture::kShape, Fixture::kVoxelSize, 0.0);
    bool parity = original_checker != nullptr && optimized_checker != nullptr;
    std::mt19937_64 rng(0x584d464c49474854ULL);
    std::uniform_real_distribution<double> x_distribution(
        (Fixture::kOrigin[0] - 2) * Fixture::kVoxelSize,
        (Fixture::kOrigin[0] + Fixture::kShape[0] + 2) * Fixture::kVoxelSize);
    std::uniform_real_distribution<double> y_distribution(
        (Fixture::kOrigin[1] - 2) * Fixture::kVoxelSize,
        (Fixture::kOrigin[1] + Fixture::kShape[1] + 2) * Fixture::kVoxelSize);
    std::uniform_real_distribution<double> z_distribution(
        (Fixture::kOrigin[2] - 2) * Fixture::kVoxelSize,
        (Fixture::kOrigin[2] + Fixture::kShape[2] + 2) * Fixture::kVoxelSize);
    for (int sample = 0; sample < 1000 && parity; ++sample) {
        const float point[3] = {
            static_cast<float>(x_distribution(rng)),
            static_cast<float>(y_distribution(rng)),
            static_cast<float>(z_distribution(rng)),
        };
        int original_collision = 0;
        int optimized_collision = 0;
        double original_distance = std::numeric_limits<double>::infinity();
        double optimized_distance = std::numeric_limits<double>::infinity();
        int32_t original_index = -1;
        int32_t optimized_index = -1;
        float original_collision_point[3] = {0.0f, 0.0f, 0.0f};
        float optimized_collision_point[3] = {0.0f, 0.0f, 0.0f};
        const int original_rc = original.check_path(
            original_checker, point, 1, 1, &original_collision,
            &original_distance, &original_index, original_collision_point);
        const int optimized_rc = planning_collision_check_path(
            optimized_checker, point, 1, 1, &optimized_collision,
            &optimized_distance, &optimized_index, optimized_collision_point);
        parity = parity && original_rc == optimized_rc &&
            original_collision == optimized_collision &&
            original_index == optimized_index &&
            CloseEnough(original_distance, optimized_distance);
    }
    std::cout << "ORIGINAL_COLLISION_RUNTIME="
              << (parity ? "PASS" : "FAIL") << '\n';
    if (original_checker) {
        original.destroy(original_checker);
    }
    if (optimized_checker) {
        planning_collision_destroy(optimized_checker);
    }
    UnloadOriginalCollisionApi(&original);
    (void)map;
    return parity;
}

bool CheckSharedOwnerLifetime(const Fixture& fixture) {
    void* map_handle = planning_voxel_map_create(
        fixture.occupied.data(), static_cast<int64_t>(fixture.occupied.size()),
        Fixture::kOrigin, Fixture::kShape, Fixture::kVoxelSize);
    if (!map_handle) {
        std::cout << "SHARED_OWNER_LIFETIME=FAIL\n";
        return false;
    }
    void* collision = planning_collision_create_from_voxel_map(map_handle, 0.0);
    void* route = planning_global_route_create_from_voxel_map(
        map_handle, 0.0, Fixture::kVoxelSize, 0.0, 10.0, 0.0, 0);
    planning_voxel_map_destroy(map_handle);

    bool collision_ok = collision != nullptr;
    if (collision_ok) {
        const auto center = OriginalCenterForKey(
            fixture.occupied.front(), Fixture::kOrigin, Fixture::kShape,
            Fixture::kVoxelSize);
        const float point[3] = {
            static_cast<float>(center[0]), static_cast<float>(center[1]),
            static_cast<float>(center[2]),
        };
        int collision_value = 0;
        double minimum_distance = std::numeric_limits<double>::infinity();
        int32_t first_index = -1;
        const int rc = planning_collision_check_path(
            collision, point, 1, 1, &collision_value, &minimum_distance,
            &first_index, nullptr);
        collision_ok = rc == 0 && collision_value == 1 && first_index == 0 &&
            std::isfinite(minimum_distance);
    }

    bool route_ok = route != nullptr;
    if (route_ok) {
        int32_t shape_xy[2] = {0, 0};
        double origin_xy[2] = {0.0, 0.0};
        double resolution = 0.0;
        route_ok = planning_global_route_shape(
            route, shape_xy, origin_xy, &resolution) == 0 &&
            shape_xy[0] > 0 && shape_xy[1] > 0 && resolution == Fixture::kVoxelSize;
        const int64_t blocked_count = static_cast<int64_t>(shape_xy[0]) * shape_xy[1];
        std::vector<uint8_t> blocked(
            blocked_count > 0 ? static_cast<size_t>(blocked_count) : 0U, 0);
        route_ok = route_ok && planning_global_route_copy_blocked(
            route, blocked.data(), blocked_count) == 0;
    }
    if (collision) {
        planning_collision_destroy(collision);
    }
    if (route) {
        planning_global_route_destroy(route);
    }
    const bool passed = collision_ok && route_ok;
    std::cout << "SHARED_OWNER_LIFETIME=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main() {
    const Fixture fixture;
    const auto map = planning::geometry::VoxelMap::Create(
        fixture.occupied.data(), static_cast<int64_t>(fixture.occupied.size()),
        Fixture::kOrigin, Fixture::kShape, Fixture::kVoxelSize);
    const bool voxel_parity = CheckVoxelMapParity(fixture, map);
    const bool original_runtime_parity = CheckOriginalRuntimeParity(fixture, map);
    const bool owner_lifetime = CheckSharedOwnerLifetime(fixture);
    const bool passed = voxel_parity && original_runtime_parity && owner_lifetime;
    std::cout << "RESULT=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
