# C++ Refactor Phase 2

## Scope

- Keep Python package layout unchanged.
- Keep Unity/ROS protocol bytes and bridge polling semantics unchanged.
- Remove project-authored CUDA collision source and build targets.
- Use the existing C++17/OpenMP collision backend as the only production collision backend.
- Move production coarse global A* into a C++17 shared library.
- Keep the Python `GlobalRoutePlanner2D` API stable.

## Native compute

```text
src/geometry/collision_checker.cpp
    -> libplanning_collision_checker.so

src/geometry/depth_safety.cpp
    -> libplanning_depth_safety.so

src/navigation/global_route_planner.cpp
    -> libplanning_global_route.so
```

Project-authored CUDA source count:

```text
*.cu  = 0
*.cuh = 0
```

PyTorch CUDA use for BC/AWAC is not changed.

## Global route

Production:

```text
planning.global_route.GlobalRoutePlanner2D
    -> ctypes
    -> libplanning_global_route.so
    -> C++17 A*
```

The Python A* implementation remains only as an explicit reference backend:

```bash
PLANNING_GLOBAL_ROUTE_BACKEND=python
```

Default:

```text
PLANNING_GLOBAL_ROUTE_BACKEND=cpp
```

## Teacher audit collision command

Use:

```bash
--collision-backend cpu --collision-threads 1
```

or benchmark `--collision-threads 2` with fewer processes. `hybrid`, collision `cuda`, and `--cuda-workers` were removed.
## Bridge transport ownership

`BridgeTransport` now owns the ZMQ context and all bridge sockets.

```text
UnityBridgeNode
    -> BridgeTransport
        -> command PUB
        -> state/depth SUB
        -> reliable command ROUTERs
        -> result/snapshot ROUTERs
        -> Python result/snapshot ROUTERs
        -> result monitor
```

All `UnityBridgeNode` methods outside `Init()` and `CloseZmq()` were checked against Phase 1 after normalizing socket access; 41/41 method bodies are unchanged.

