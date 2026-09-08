from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_has_no_self_authored_cuda_sources():
    assert list(ROOT.rglob("*.cu")) == []
    assert list(ROOT.rglob("*.cuh")) == []


def test_collision_backend_is_cpp_only_in_production_configuration():
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    collision = (ROOT / "python/planning/safety/collision_checker.py").read_text(encoding="utf-8")
    assert "find_package(CUDA" not in cmake
    assert "cuda_add_library" not in cmake
    assert "planning_collision_checker_cuda" not in cmake
    assert "_CudaCollisionBackend" not in collision
    assert "PLANNING_COLLISION_CUDA_LIBRARY" not in collision


def test_global_route_has_cpp17_native_backend():
    source = ROOT / "src/navigation/global_route_planner.cpp"
    header = ROOT / "include/planning/navigation/global_route_planner.hpp"
    wrapper = (ROOT / "python/planning/mission/global_route.py").read_text(encoding="utf-8")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert source.is_file()
    assert header.is_file()
    assert "planning_global_route" in cmake
    assert "GLOBAL_ROUTE_BACKEND_ENV" in wrapper
    assert "FORMAL_GLOBAL_ROUTE_BACKEND" in wrapper
