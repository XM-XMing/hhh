# C++ Phase 4: Transport Ownership and Immutable VoxelMap

## Scope

This phase changes C++ structure only. Python, scripts, launch files, and config files are unchanged from Phase 3.

## Bridge transport ownership

`BridgeTransport` now owns only the ZeroMQ context and four transport modules:

- `CommandTransport`: command PUB, Python command ROUTER, Unity command ROUTER
- `TelemetryTransport`: state SUB, depth SUB
- `ResultTransport`: Unity result ROUTER, Python result ROUTER, result monitor PAIR
- `SnapshotTransport`: Unity snapshot ROUTER, Python snapshot ROUTER

Each Gateway receives only its required transport type. No Gateway receives the aggregate `BridgeTransport`.

## Immutable VoxelMap

`planning::geometry::VoxelMap` is the canonical C++ map owner. It owns:

- origin IJK
- grid shape
- voxel size
- occupied key list
- immutable occupancy bitmap
- coordinate conversion

`CollisionChecker` and `GlobalRoutePlanner` both store `std::shared_ptr<const VoxelMap>`.

New C ABI:

- `planning_voxel_map_create`
- `planning_voxel_map_destroy`
- `planning_collision_create_from_voxel_map`
- `planning_global_route_create_from_voxel_map`

The legacy constructors remain available so the Python bindings do not need to change in this phase.

## Validation

- C++17 geometry compile with `-Wall -Wextra -Werror`: PASS
- transport syntax check with ZMQ interface stub: PASS
- BridgeNode + five Gateway syntax checks with ROS/ZMQ/MessagePack interface stubs: PASS
- Collision Phase 3 vs Phase 4 parity: 200 random cases exact
- Global route Phase 3 vs Phase 4 parity: 120 random cases exact
- Existing global-route native/reference parity: 40/40 exact
- Shared VoxelMap lifetime test: PASS
- Python tree diff vs Phase 3: 0 files
- scripts diff vs Phase 3: 0 files
- launch diff vs Phase 3: 0 files
- config diff vs Phase 3: 0 files
