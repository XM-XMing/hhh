# C++ Foundation Controlled Migration

Date: 2026-08-27

This is the C0/C1 audit for the controlled migration from the read-only
original tree (`/home/xm/XM/xm_ws/src/planning`, O) to the optimized candidate
tree (`/home/xm/XM/src`, P). O is the behavior baseline. P is the only tree
modified in this slice. No Unity, Python algorithm, Bridge implementation,
formal data, or commit was changed.

## Inventory and migration verdicts

| Module | ORIGINAL_OWNER | OPTIMIZED_OWNER | Classification | Behavior/contract finding | Test coverage | MIGRATION_VERDICT |
| --- | --- | --- | --- | --- | --- | --- |
| COMMON | `include/planning/*.hpp`; `src/map_loader_node.cpp`; `src/visualization_node.cpp` | compatibility headers plus `include/planning/geometry`, `protocol`, `bridge`, and `transport`; `src/tools/*` | `EXACT_MOVE`, `STRUCTURAL_REFACTOR`, `SAFE_CANDIDATE` | Point-cloud/tool paths preserve the existing target APIs; include ownership is narrowed. | O/P target compile; existing C++ contract files are present in both trees. | `NEEDS_PARITY_TEST` |
| VOXEL_MAP | Occupancy metadata, bitmap, and coordinate logic are private in `src/collision_checker.cpp` | `include/planning/geometry/voxel_map.hpp`, `src/geometry/voxel_map.cpp`, `planning_voxel_map` | `NEW`, `STRUCTURAL_REFACTOR`, `SAFE_CANDIDATE` | P makes map metadata, immutable bitmap, occupied keys, and conversions one `shared_ptr<const VoxelMap>` owner. O has no equivalent public owner. | 1000 random points, 14 boundary points, occupied-key hash, metadata, and owner lifetime fixture. | `SAFE_TO_MIGRATE` |
| COLLISION | `src/collision_checker.cpp`; `src/collision_checker_cuda.cu` | `src/geometry/collision_checker.cpp`, `include/planning/geometry/collision_checker.hpp` | `STRUCTURAL_REFACTOR`, `REMOVED`, `BLOCKED` | CPU query semantics matched O in the fixed runtime comparator. CUDA target/API is absent from P and was not deleted or replaced in this slice. | O/P CPU collision query comparator; C++ target build. | `NEEDS_PARITY_TEST` |
| DEPTH_SAFETY | `src/depth_safety.cpp` | `src/geometry/depth_safety.cpp`, `include/planning/geometry/depth_safety.hpp` | `EXACT_MOVE`, `STRUCTURAL_REFACTOR`, `SAFE_CANDIDATE` | The implementation body is unchanged apart from the ownership include; the exported depth symbol SHA is identical. | Release compile/link only in C0/C1. | `NEEDS_PARITY_TEST` |
| GLOBAL_ROUTE | No O C++ global-route owner | `src/navigation/global_route_planner.cpp`, `include/planning/navigation/global_route_planner.hpp`, `planning_global_route` | `NEW`, `BLOCKED` | P contains a native route candidate and shared-map entry point. Native A* is not enabled and no O behavioral parity claim is made here. | Target compile; no C2 route behavior test run. | `REJECT` |
| BRIDGE | Monolithic `src/unity_bridge_node.cpp` | `src/unity_bridge_main.cpp`, `src/bridge/*` | `STRUCTURAL_REFACTOR`, `BLOCKED` | P split ownership is not build-closed: existing `ROS_*` macro failures block the executable. No Bridge edits are allowed in C0/C1. | Full P Release build identifies the compile failures. | `BLOCKED` |
| PROTOCOL | Protocol/broker implementation in the monolithic bridge and root headers | compatibility root headers plus `include/planning/protocol/*`, `src/protocol/*` | `STRUCTURAL_REFACTOR`, `NEW`, `BLOCKED` | P has split wire ownership, but existing `AsInt32` compile failures prevent contract closure. No protocol behavior was changed here. | Existing C++ contract inventory; full P build failure evidence. | `BLOCKED` |
| TRANSPORT | Transport implementation embedded in the monolithic bridge | `include/planning/transport/*`, `src/transport/*` | `NEW`, `STRUCTURAL_REFACTOR`, `BLOCKED` | P has separate transport modules, but they cannot be promoted while the enclosing Bridge target is not compile-closed. | Full P build reaches the Bridge target and fails outside scope. | `BLOCKED` |

`package.xml` is byte-identical between O and P. `CMakeLists.txt` differs in
C++ standard, CUDA target presence, split target ownership, install target
list, test fixture target, and the guarded generated-primitives install.

The CMake/install guard added in this slice only skips the absent generated
`data/motion_primitives` directory. It does not create data and does not make
the P package installable while the unrelated conda/catkin Python install
failure and Bridge/protocol compile failures remain.

## Build baseline

All commands used `source /home/xm/anaconda3/etc/profile.d/conda.sh && conda
activate xm` and ROS Noetic setup. Release CMake configuration used the xm
environment Python executable.

### O

- C++ compile/link: `PASS` (CMake all target exit 0).
- Targets reached: `planning_depth_safety`, `planning_collision_checker`,
  `planning_collision_checker_cuda`, `map_loader_node`,
  `visualization_node`, `unity_bridge_node`, and generated message targets.
- The catkin `--install` wrapper then failed in `setup.py` with
  `error: option --install-layout not recognized`; this is an environment
  install failure after all C++ targets linked, not a C++ compile failure.
- Libraries:

  | Library | SHA256 |
  | --- | --- |
  | `libplanning_collision_checker.so` | `c9ffc42895a7845ada624cb5a386b7bc46364e4e9f89132761dda1a070b84131` |
  | `libplanning_collision_checker_cuda.so` | `a6ebd3d9b8e1afe9a3284a9ec196bd242f306f1a98c1211f88bf571c2e2c2132` |
  | `libplanning_depth_safety.so` | `179e165cc2ef0f47232b61c0ccdb4ad1561047aab29893b841f4f8566cf27fab` |

### P

- CMake configure: `PASS`; it reports
  `Skipping generated data/motion_primitives install: directory is absent`.
- Geometry/fixture subset: `PASS` for `planning_voxel_map`,
  `planning_collision_checker`, `planning_global_route`,
  `planning_depth_safety`, and `voxel_map_parity_fixture`.
- Full CMake all target: `FAIL` (exit 2) in the existing Bridge/protocol
  sources: `ROS_WARN`, `ROS_WARN_THROTTLE`, and `ROS_ERROR_THROTTLE` are not
  declared in `result_gateway.cpp`/`snapshot_gateway.cpp`, and `AsInt32` is
  not declared in `telemetry_wire.cpp`. The existing unused `AsInt32` warning
  in `endpoint_observation_snapshot_wire.cpp` is also recorded. These files
  were not modified.
- Libraries:

  | Library | SHA256 |
  | --- | --- |
  | `libplanning_voxel_map.so` | `edb5fb89c14bc81d91d3c2c10fd45be4d2d9102b9924710b1d3c0fd528104785` |
  | `libplanning_collision_checker.so` | `8140348f7df5d13976b2f7dc05b1590815852764d640ccbb9569165ed7333f8c` |
  | `libplanning_depth_safety.so` | `179e165cc2ef0f47232b61c0ccdb4ad1561047aab29893b841f4f8566cf27fab` |
  | `libplanning_global_route.so` | `83f53683afb2b658f43964616507a3e44416a2361fcb3060c81f6a5b26a2e97d` |

P's collision library retains all O CPU C ABI entry points and adds
`planning_collision_create_from_voxel_map`. P adds
`planning_voxel_map_create/destroy` and the global-route C ABI. The O CUDA
library has no P counterpart; CUDA removal remains blocked and out of scope.

## C1 VoxelMap parity

Neither O nor P contains the referenced
`data/map_data/forest_voxels_10cm.npz`. O contains only the raw
`data/map_data/forest_point_cloud.bin`; it was not converted or copied. The
fixture therefore uses a deterministic compact forest-like cache in memory,
not formal project data. Its parameters are:

- voxel size `0.25`
- origin IJK `[-6, -4, 1]`
- grid shape `[18, 16, 9]`
- occupied count `227`
- occupied-key FNV-1a hash `0xf2adfb813cc3d10c`
- 1000 deterministic random points and 14 boundary points

The O reference is the current O `collision_checker.cpp` geometry: floor
world-to-voxel, origin-relative bounds, row-major key
`rx * shape_y * shape_z + ry * shape_z + rz`, and voxel-center conversion.
The fixture also loads O's built `libplanning_collision_checker.so` and
compares the actual O/P CPU path result for the same 1000 random points.

Results:

```text
METADATA_PARITY=PASS
OCCUPANCY_PARITY=PASS
COORDINATE_CONVERSION_PARITY=PASS
ORIGINAL_COLLISION_RUNTIME=PASS
SHARED_OWNER_LIFETIME=PASS
RESULT=PASS
```

The owner test creates one P map handle, creates both
`CollisionChecker` and `GlobalRoutePlanner` through their shared-map C ABI,
destroys the map handle, and successfully queries both consumers. The
consumers store `shared_ptr<const VoxelMap>`. GlobalRoutePlanner's local
`blocked_` array is a derived 2-D coarse route grid, not a second complete
3-D occupancy copy.

The real forest-cache parity remains `NOT_RUN` until the cache is supplied by
the data-producing workflow. No formal data was generated in C0/C1.

## Tests and modified files

The existing eight O/P C++ contract fixture names remain present in both
trees. P adds only `tests/cpp/voxel_map_parity_fixture.cpp` and a CMake test
executable target. The focused fixture build and execution passed under
`conda activate xm`.

Modified P files in this slice:

- `CMakeLists.txt`
- `tests/cpp/voxel_map_parity_fixture.cpp`
- `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`
- `docs/CXX_FOUNDATION_MIGRATION.md`

No O files were modified. No P VoxelMap, CollisionChecker, GlobalRoute, or
Bridge implementation was changed during C1; the existing shared-owner seam
was validated and fixed parity coverage was added around it.

## Explicitly blocked behavior changes

- No collision semantic change or CUDA deletion.
- No native A* enablement or route promotion.
- No corner-cutting, depth-safety, Bridge, protocol, transport, or Python
  algorithm change.
- No generated motion-primitives data fabrication.
- No migration of Unity or BC/AWAC artifacts.

## Follow-up order

1. C2 — Collision Backend Parity
2. C3 — Depth Safety Parity
3. C4 — Global Route Owner/API Parity; decide native-route promotion only
   after parity evidence
4. C5 — Bridge, Protocol, and Transport Compile/Runtime Parity
5. C6 — Catkin install, package, and end-to-end runtime closure

Current next phase: **C2 COLLISION BACKEND PARITY**.

## C2 Collision backend parity (2026-08-27)

C2 compared the read-only O CPU collision library with P's C++17/OpenMP
collision library through the exported C ABI.  No collision implementation,
CUDA source, Python wrapper, Bridge, protocol, A*, or formal project data was
changed.

### Backend inventory and availability

| Backend | OWNER | Available in this run | API/role | Classification | Current migration verdict |
| --- | --- | --- | --- | --- | --- |
| O CPU | O `src/collision_checker.cpp` | `YES` | path, single action, actions batch, pose x actions batch | `COMMON`, `SAFE_CANDIDATE` | `NEEDS_BENCHMARK` |
| O CUDA | O `src/collision_checker_cuda.cu` | `NO` at runtime; library built | actions batch and pose x actions batch | `COMMON`, `BLOCKED` | `REJECT` |
| O hybrid | O `cli/audit_teacher_missions.py` CPU process pool plus CUDA thread pool | `NO` | ordered Teacher audit scheduler | `STRUCTURAL_REFACTOR`, `BLOCKED` | `REJECT` |
| P C++ | P `src/geometry/collision_checker.cpp` | `YES` | O CPU C ABI plus shared-VoxelMap constructor | `STRUCTURAL_REFACTOR`, `SAFE_CANDIDATE` | `REJECT` (throughput gate not met) |
| P hybrid | No P hybrid backend | `NO` | no CUDA/hybrid wrapper | `REMOVED`, `BLOCKED` | `REJECT` |

The O CUDA target compiled and linked, but `planning_collision_cuda_create`
returned null because this machine has no usable NVIDIA driver/device.  The O
hybrid scheduler therefore was not executed and is reported as
`NOT_AVAILABLE`; no CUDA result was fabricated.  Neither O nor P exposes a
separate reachability or minimum-clearance C API.  `minimum_distance` is an
output of the available collision APIs and was compared.

### Fixtures and exact results

The comparator is `tests/compare_collision_backend_parity.py`.  It exercised
the C1 deterministic layouts (sparse, boundary, shell, dense, and empty),
start/middle/end/no-collision paths, boundary cases, check steps 1/2/3,
permuted action IDs, three poses, inflate radii `0.35` and `0.40`, and the
temporary production-like layout.  The O/P CPU comparison covered 121 cases.

The production-like fixture was generated only under `/tmp` using O's formal
motion-primitive generator and voxel builder from the raw O point cloud.  It
used 24,818,315 source points, raw-point-cloud SHA256
`dfd87a5db1ab98276eda57e79de677f711da56f308a373b93e72484fa5876d40`, voxel
size `0.10`, origin `[-1023, -1020, -1]`, shape `[2046, 2045, 42]`, and
1,740,947 occupied keys.  The temporary cache SHA256 is
`a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691` and
its occupied-key byte SHA256 is
`033c43a606a7f9de60d044845b76ceeb24918ef0be7b70a57bdbacd6feff5b28`.
`PRODUCTION_ASSET_PARITY=PASS`; no cache or generated primitive was written
to either project tree.

Discrete outputs and endpoint/minimum-distance values were compared first by
exact bytes.  The O/P CPU result was:

```text
PATH_COLLISION_PARITY=PASS
SINGLE_ACTION_PARITY=PASS
ACTIONS_BATCH_PARITY=PASS
POSE_ACTIONS_BATCH_PARITY=PASS
FIRST_COLLISION_INDEX_PARITY=PASS
ENDPOINT_PARITY=PASS
MIN_DISTANCE_PARITY=PASS
BATCH_ORDER_PARITY=PASS
FIRST_COLLISION_POINT_PARITY=PASS
CPU_MAX_ABS_DELTA=0
CPU_MAX_RELATIVE_DELTA=0
```

### Shared VoxelMap runtime ownership

The C1 lifetime fixture creates one P `VoxelMap`, then both
`CollisionChecker` and `GlobalRoutePlanner` hold
`shared_ptr<const VoxelMap>` after the external map handle is destroyed.  C2
also created two collision checkers with radii `0.35` and `0.40` from one map.
The observed ownership evidence is one map load, one logical occupied-buffer
owner, two checker consumers, successful post-handle-destruction queries, and
zero complete occupancy copies in consumers:

```text
SHARED_VOXELMAP_RUNTIME_OWNERSHIP=PASS
map_load_count=1
occupied_buffer_allocation_count=1
checker_count=2
complete_occupancy_copies_in_consumers=0
```

The count is fixture-level ownership evidence, not malloc tracing.  P's
GlobalRoutePlanner still has only its derived 2-D route grid; it does not copy
the complete 3-D occupancy bitmap.

### Release benchmark

The fixed `c1_sparse` input used three runs at each OpenMP setting (1, 2, 4)
on the same machine.  Each cell below is
`mean / median / min / max`; throughput units are operations per second.

| OMP threads | Backend | paths | single actions | poses x actions | batch p50 / p95 (ms, median) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | O CPU | 137962 / 141851 / 128972 / 143063 | 89674 / 90573 / 87286 / 91163 | 500328 / 503187 / 493009 / 504787 | 0.022773 / 0.023608 |
| 1 | P C++ shared | 137102 / 138092 / 134696 / 138518 | 89640 / 89419 / 88834 / 90666 | 443081 / 442518 / 440716 / 446009 | 0.026020 / 0.026787 |
| 2 | O CPU | 144331 / 144752 / 143249 / 144992 | 91412 / 91587 / 90614 / 92034 | 529323 / 529478 / 528425 / 530067 | 0.021191 / 0.024095 |
| 2 | P C++ shared | 138723 / 138576 / 138510 / 139082 | 91082 / 91012 / 90928 / 91304 | 490475 / 491280 / 488628 / 491516 | 0.023285 / 0.023944 |
| 4 | O CPU | 142139 / 141171 / 140850 / 144397 | 90671 / 90379 / 89794 / 91839 | 439599 / 446073 / 416134 / 456589 | 0.025344 / 0.029123 |
| 4 | P C++ shared | 135380 / 136519 / 132673 / 136947 | 87653 / 87449 / 86663 / 88845 | 399777 / 397903 / 372670 / 428757 | 0.027268 / 0.031576 |

Peak process RSS was `103.312 MB` for all rows.  Process-level CPU
utilization means were O/P `100.006%/100.001%` at one thread,
`116.099%/122.075%` at two, and `127.356%/135.655%` at four; this is not a
fabricated per-core measurement.  O's legacy constructor combines map load
and checker initialization, while P's shared path reports them separately;
the raw JSON retains both timings.  P's pose x actions throughput is below O
at all three settings (median ratios `0.879`, `0.928`, `0.892`), so the
production throughput gate is not met.

Benchmark artifact:
`/tmp/xmflight_collision_c2/collision_backend_benchmark.json`, SHA256
`b0f1999766a15b6533f10ece191aa3a3deeb46c8f04efb9be3e7ebb5c1e785ed`.

### C2 result and blockers

```text
C2_COLLISION_PARITY=PARTIAL
O_CPU_AVAILABLE=YES
O_CUDA_AVAILABLE=NO
O_HYBRID_AVAILABLE=NO
P_CPP_AVAILABLE=YES
PRODUCTION_ASSET_PARITY=PASS
P_CPP_COLLISION_PRODUCTION_CANDIDATE=NO
ALGORITHM_CHANGED=NO
```

The CPU C ABI is behaviorally exact on the fixed fixtures, but C2 cannot be a
full backend parity closure while O CUDA/hybrid are unavailable, P has no
CUDA/hybrid implementation, and the P runtime throughput gate is below O.
No cutover is authorized.  The next action is **C2 COLLISION FIX /
INVESTIGATION**; C2.1 cutover is not entered.

## C2.1 Collision hot-path regression investigation (2026-08-27)

C2.1 is the controlled follow-up to the historical C2 result above. O
(`/home/xm/XM/xm_ws/src/planning`) remained read-only; P
(`/home/xm/XM/src`) was the only implementation tree changed. No Unity,
Bridge, protocol, Global A*, CUDA/hybrid, Python Teacher/Mission algorithm,
BC, or AWAC code was changed.

### Build parity

Official target flags were audited from the generated `flags.make` and
`link.txt` files. Both collision targets use effective Release `-O3
-DNDEBUG -fPIC -fopenmp`, with no LTO, no explicit architecture flags, default
visibility, and default exceptions/RTTI. OpenMP links are `-lgomp -lpthread`.
O's project target is C++14 and P's is C++17. To remove that structural
difference from the performance decision, O's unchanged collision source was
also compiled as a temporary C++17 shared library under
`/tmp/xmflight_collision_c2/o_cxx17/`; the exact compile/link lines are
recorded in `docs/COLLISION_BACKEND_PARITY.md`.

```text
BUILD_FLAGS_PARITY=PASS (controlled O C++17 vs P C++17 Release comparison)
OFFICIAL_O_STANDARD=C++14
OFFICIAL_P_STANDARD=C++17
LTO=OFF
ARCH_FLAGS=NONE
VISIBILITY=DEFAULT
EXCEPTIONS_RTTI=DEFAULT
```

### P-only kernel seam

The prior P lookup path called `VoxelMap::WorldToVoxel` and
`VoxelMap::IsOccupiedVoxel` for every neighbor. C2.1 adds only a const pointer
view to the map-owned bitmap and caches the same origin, shape, stride, key
count, and voxel-size values at checker construction. The checker still holds
`std::shared_ptr<const VoxelMap>`, so the bitmap cannot outlive its owner and
no complete occupancy copy is introduced.

The O verified loop order is now used for P: floor quantization, cached
neighbor offsets, bounds, row-major key, bitmap bit test, distance, minimum,
first-collision break, endpoint write, and batch/OpenMP scheduling. The public
P C ABI is unchanged apart from the already-existing shared-map constructor.

```text
HOT_LOOP_DIFFERENCE=EXTRA_WORK_IN_P before fix; removed after fix
DIFFERENT_ALGORITHM=0
STEADY_STATE_ALLOCATIONS_O=0
STEADY_STATE_ALLOCATIONS_P=0
```

### Gates and evidence

The post-fix O/P CPU comparator passed 121 exact cases with zero maximum
absolute/relative delta and passed the production-like temporary cache. The
updated C++ owner fixture passed 1000 random points, 14 boundary points,
occupied-key hash, coordinate conversion, and shared-owner lifetime. The new
isolated Python subprocess fixture passed 100 candidate missions and 100
Teacher rows with 105-action masks per row:

```text
VOXEL_METADATA_PARITY=PASS
VOXEL_OCCUPANCY_PARITY=PASS
COORDINATE_CONVERSION_PARITY=PASS
SHARED_VOXELMAP_RUNTIME_OWNERSHIP=PASS
CPU_PARITY_CASES=121
CPU_MAX_ABS_DELTA=0
CPU_MAX_RELATIVE_DELTA=0
CANDIDATE_ACCEPT_VECTOR_PARITY=PASS (100/100)
TEACHER_VALID_ACTION_MASK_PARITY=PASS (100/100)
```

The kernel benchmark used three runs at OpenMP 1/2/4 under the controlled
C++17 Release condition. Best median pose x actions throughput was O
`516287` and P `531508` operations/sec at OMP2, giving P `102.95%` of O. A
fixed 1000-candidate C++ end-to-end probe, pinned to CPUs 4-7, gave best
median total throughput O `7835947` and P `9204120` candidates/sec at OMP2;
P's delta was `+17.46%`. P's median initialization was `0.004175 ms` (map
`0.000921 ms` plus checker `0.003152 ms`) versus O's combined raw init
`0.004204 ms`. Both map-load counts were one. The fixed-probe peak RSS was
P `3.145 MB` versus O `3.094 MB`; the Python-wrapper benchmark reported
`103.578 MB` for both.

`perf` and `valgrind/callgrind` are absent, so hardware counters and
function-level profiles were recorded as `NOT_AVAILABLE`, not inferred.

```text
C2_1_COLLISION_REGRESSION=PASS
P_CPP_COLLISION_PRODUCTION_CANDIDATE=YES
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
```

### C2.1 files and non-actions

P files modified in C2.1:

- `include/planning/geometry/voxel_map.hpp`
- `src/geometry/collision_checker.cpp`
- `tests/compare_collision_teacher_mask_fixture.py`
- the three migration/parity documents

The fixture and benchmark outputs remain under `/tmp/xmflight_collision_c2/`.
No formal data was generated or copied, no CUDA/hybrid code was deleted, and
no production cutover was run.

### Follow-up order

The remaining foundation order is:

1. C2.2 — CPU collision production cutover review (not executed in C2.1)
2. C3 — depth-safety parity
3. C4 — Global A* owner/API parity; native A* remains disabled
4. C5 — Bridge, protocol, and transport compile/runtime parity
5. C6 — catkin install, package, and end-to-end runtime closure

`NEXT_PHASE=C2.2 PRODUCTION CUTOVER`

## C2.2 collision production cutover (2026-08-27)

The earlier C2.1 section is historical. With explicit authorization, P's
self-owned collision production path is now closed over C++17/OpenMP:

```text
C2_2_COLLISION_PRODUCTION_CUTOVER=PASS
FORMAL_COLLISION_BACKEND=C++17_OPENMP_CPU
PRODUCTION_BACKEND_UNIQUE=YES
COLLISION_NUMERIC_PARITY=PASS
CANDIDATE_ACCEPT_VECTOR_PARITY=PASS
TEACHER_VALID_ACTION_MASK_PARITY=PASS
MISSION_COLLISION_INTEGRATION_PARITY=PASS
TEACHER_COLLISION_INTEGRATION_PARITY=PASS
P_COLLISION_TARGET_BUILD=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
CXX_SHARED_VOXELMAP=PASS
PYTHON_SHARED_VOXELMAP_BINDING=NOT_STARTED
CUDA_HYBRID_FORMAL_PATHS=0
GPU_RUNTIME_VALIDATION=DEFERRED / NOT_AVAILABLE
ALGORITHM_CHANGED=NO
```

The Python safety owner resolves all production aliases to the native
`planning_collision_checker` library. Native initialization or operation
failure is fail-closed; Python reference mode is explicit test/debug only.
The cutover did not modify collision output/API shape, row ordering, float
calculation, Teacher logic, Global A*, Bridge/protocol, or BC/AWAC PyTorch
CUDA/AMP/GradScaler behavior. P contained no formal collision CUDA or hybrid
target/source to delete; its absence is recorded as `OUT_OF_FORMAL_SCOPE`,
not `PROVEN_INCORRECT`.

The fixed evidence is 121 exact numeric cases, 100 exact candidate missions,
and 100 exact 105-action Teacher masks. The controlled Release benchmark
recorded P at 531508 versus O at 516287 poses x actions/sec, and P at 9204120
versus O at 7835947 candidates/sec. `perf` counters were not available, so no
hardware-counter claim is made. The full P build still stops at the
pre-existing Bridge/protocol `ROS_WARN*` declarations after the collision
targets have built.

No C3 work was executed. Next phase:
`C3 GLOBAL A* SEMANTICS AND PARITY`.

## C3 — Global A* semantics and parity (2026-08-27)

The C3 controlled comparison is complete. O remained read-only and P was the
only candidate tree inspected for implementation changes:

```text
C3_GLOBAL_ASTAR_PARITY=PASS
STRUCTURE_CHANGED=YES
ROUTE_BEHAVIOR_CHANGED=NO
P_CPP_GLOBAL_ROUTE_PRODUCTION_CANDIDATE=YES
```

O's Python `GlobalRoutePlanner2D` remains the route source of truth. P exposes
the same public contract through `python/planning/mission/global_route.py`
and the `planning_global_route` C++17 target. Origin/shape, flight-band
occupancy projection, circular inflation, coordinate conversions,
nearest-free scan order, neighbor order, diagonal corner guard, queue tie
ordering, strict relaxation, reconstruction, no-route status, and output
shape/row order were exact on 127 synthetic and 1000 production-like cases.
The initial intermediate-z byte mismatch was closed at the P wrapper boundary
to preserve O's scalar-z-to-float32 `linspace` result; A* XY behavior and
costs were not changed.

```text
CORNER_CUTTING_PARITY=PASS
TIE_ROUTE_SEQUENCE_PARITY=PASS
ROUTE_FOUND_PARITY=PASS
ROUTE_POINT_COUNT_PARITY=PASS
ROUTE_GRID_SEQUENCE_PARITY=PASS
ROUTE_WORLD_SEQUENCE_PARITY=PASS
ROUTE_COST_PARITY=PASS
ROUTE_LENGTH_PARITY=PASS
METADATA_PARITY=PASS
WORKSPACE_STATE_LEAK=0
MISSION_ROUTE_ACCEPT_VECTOR_PARITY=PASS (1000/1000)
MISSION_ACCEPTED_ID_ORDER_PARITY=PASS
TEACHER_ROUTE_INPUT_PARITY=PASS (100/100)
TEACHER_ACTION_PARITY=PASS (100/100)
TEACHER_VALID_ACTION_MASK_PARITY=PASS (100/100, 105 actions)
```

The route benchmark used three fresh-process runs at each size. At size 1000,
P achieved `2152.737649 routes/sec` versus O `47.200157 routes/sec`, a
`45.608697x` speedup. Mean initialization was P `0.125439 s` versus O
`0.059851 s`; peak RSS was P `380592 KB` versus O `379456 KB`; both loaded
one map and created one planner. P uses a persistent route workspace estimate
of `11402920` bytes versus O's per-plan estimate of `6036840` bytes. This is
the recorded memory tradeoff, not a route-schema change.

The P route target builds independently and exports the expected C ABI. The
P full build remains blocked by the pre-existing Bridge/protocol undeclared
`ROS_WARN*` symbols; Bridge was not changed. Python shared-VoxelMap binding
is still deferred to C5, and the combined Mission/Teacher comparator does not
claim unified Python ownership. No RouteStore, cross-stage deduplication,
production cutover, formal data collection, or commit was performed.

Evidence and detailed semantic mapping are in
`docs/GLOBAL_ROUTE_PARITY.md`; machine-readable results remain under
`/tmp/xmflight_global_route_c3/`.

`NEXT_PHASE=C3.1 GLOBAL A* PRODUCTION CUTOVER`

## C3.1 — Global A* production cutover (2026-08-27)

```text
C3_1_GLOBAL_ASTAR_PRODUCTION_CUTOVER=PASS
FORMAL_GLOBAL_ROUTE_BACKEND=C++17_NATIVE
PRODUCTION_ROUTE_BACKEND_UNIQUE=YES
PYTHON_PRODUCTION_FALLBACK=NO
NATIVE_ROUTE_FAIL_CLOSED=PASS
GLOBAL_ROUTE_PARITY=PASS
COLLISION_REGRESSION=PASS
P_GLOBAL_ROUTE_TARGET_BUILD=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
CXX_SHARED_VOXELMAP=PASS
PYTHON_SHARED_VOXELMAP_BINDING=NOT_STARTED
ROUTE_STORE=NOT_STARTED
ALGORITHM_CHANGED=NO
```

This P-only cutover makes `planning_global_route` the unique formal C++17
native route backend. Backend aliases `unset/cpp/native/auto/cpp_native` map
to `cpp_native`; `python` is allowed only with explicit
`PLANNING_GLOBAL_ROUTE_REFERENCE=1` test/debug opt-in. Native library,
symbol, initialization, and execution failures fail closed, while native
no-path is preserved as the normal business result. Formal Generate, Audit,
Collection, and Relabel call counts remain one each; no RouteStore was added.

Exact parity remains green for route corner/tie/no-path fixtures, 127
synthetic cases, 1000 production-like cases, 1000 mission acceptance/order
rows, and 100 Teacher route/action/mask rows with 105 actions. The C2
collision gates were rerun and remain exact: 121 numeric cases, 100 candidate
accept rows, and 100 Teacher masks. The route benchmark at size 1000 reports
P `2150.730608 routes/s`, O `47.492813 routes/s`, and `45.285391x` speedup;
init means are `0.065384/0.059457 s` and peak RSS is `380528/379520 KB`
(P/O). The full package build remains blocked by the pre-existing Bridge
`ROS_WARN*` compile errors, after the route target passes.

Route provenance metadata, package exports, explicit CMake source/target
ownership, and native command shape are recorded in
`docs/GLOBAL_ROUTE_PRODUCTION_CUTOVER.md`. No source files were deleted, no
formal data was collected, and C4 was not started. Python shared-VoxelMap
binding remains a C5 item and CUDA collision validation remains deferred.

`NEXT_PHASE=C4 DEPTH SAFETY PARITY AND OWNERSHIP`

## C4 — Depth safety parity and ownership (2026-08-27)

```text
C4_DEPTH_SAFETY_PARITY=PASS
O_DEPTH_SAFETY_AVAILABLE=YES
P_DEPTH_SAFETY_AVAILABLE=YES
DEPTH_SAFETY_OWNER_UNIQUE=YES
MASK_BIT_PARITY=PASS
MIN_CLEARANCE_PARITY=PASS
VALID_COUNT_PARITY=PASS
EMPTY_MASK_PARITY=PASS
ACTION_ORDER_PARITY=PASS
PRODUCTION_LIKE_DEPTH_PARITY=PASS
DEPTH_MASK_GENERATOR_PARITY=PASS
POLICY_MASKED_ACTION_PARITY=PASS
TEACHER_DEPTH_SAFETY_PARITY=PASS
P_DEPTH_SAFETY_PRODUCTION_CANDIDATE=YES
NATIVE_DEPTH_SAFETY_FAIL_CLOSED=PASS
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
DEPTH_MASK_ARTIFACT_PROVENANCE=NOT_HANDLED_IN_C4
ALGORITHM_CHANGED=NO
```

P's numeric depth path uses the native `planning_depth_safety` target as its
formal owner.  Native library, symbol, and execution failures fail closed;
the NumPy implementation is explicit reference/debug only.  O/P projection,
threshold, patch, diagnostic, minimum-clearance, 105-action order, and empty
mask behavior are byte-for-byte equal.  This is a behavior-preserving
structural convergence; no Unity, Bridge, Mission/Teacher scoring,
observation provenance producer, BC/AWAC, or shared Python VoxelMap binding
was changed.

The deterministic comparator covered 18 synthetic edge-case frames and 1000
production-like synthetic frames.  It also passed 100 transition mask
generation, 100 policy masked-action, and 100 Teacher safety fixtures.  The
three-run depth benchmark was P `272496.888020` versus O `259712.147181`
actions/sec (`+4.922658%`), with equal maximum peak RSS `150844 KB` and 1000
native crossings per run.  Allocation count was not instrumented.  Exact
fixture hashes and field-level evidence are in
`docs/DEPTH_SAFETY_PARITY.md`; machine-readable results are in
`/tmp/xmflight_depth_safety_c4/depth_safety_parity.json`.

The independent depth, collision, and Global A* targets build.  The full P
build remains blocked only at the unchanged Bridge `ROS_WARN*`/
`ROS_ERROR*` declarations.  No formal data was collected and no commit was
created.

`NEXT_PHASE=C4.1 DEPTH SAFETY PRODUCTION CUTOVER`

## C4.1 — Depth Safety production cutover (2026-08-27)

The authorized P-only cutover is complete. O remained read-only. The single
formal production owner is
planning.safety.depth_safety.local_depth_action_mask, backed by the
planning_depth_safety C++17 shared library. Native library, symbol, input,
and execution failures remain fail-closed; the NumPy implementation requires
the explicit reference/debug opt-in and is not a production fallback.

~~~
C4_1_DEPTH_SAFETY_PRODUCTION_CUTOVER=PASS
FORMAL_DEPTH_SAFETY_OWNER=planning.safety.depth_safety.local_depth_action_mask
PRODUCTION_IMPLEMENTATION_COUNT=1
PYTHON_PRODUCTION_FALLBACK=NO
NATIVE_DEPTH_SAFETY_FAIL_CLOSED=PASS
DEPTH_SAFETY_OWNER_UNIQUE=YES
DEPTH_CONFIG_OWNER_UNIQUE=YES
MASK_BIT_PARITY=PASS
DEPTH_MASK_GENERATOR_PARITY=PASS
POLICY_MASKED_ACTION_PARITY=PASS
TEACHER_DEPTH_SAFETY_PARITY=PASS
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
P_DEPTH_SAFETY_TARGET_BUILD=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
DEPTH_MASK_ARTIFACT_PROVENANCE=NOT_HANDLED
ALGORITHM_CHANGED=NO
~~~

The production callers use the canonical Python seam and do not know native
paths, ctypes handles, C ABI pointers, or buffer layout. The apparent
duplicates are storage, diagnostics, checkpoint/config carriers, or the
explicit reference implementation; no source file was deleted. CMake and
package.xml now record the depth native target, contract, library, and source
identity. The target install rule is present. A full install command in
conda activate xm was blocked before target installation by the active
setuptools rejecting setup.py --install-layout=deb; the full package build
continues to be blocked by the unchanged Bridge protocol macros.

The exact depth, collision, and Global A* rerun evidence and deletion ledger
are recorded in docs/DEPTH_SAFETY_PRODUCTION_CUTOVER.md. Depth-mask artifact
provenance was intentionally not handled. No C5 work was started.

NEXT_PHASE=C5 PYTHON SHARED VOXELMAP AND NATIVE BINDING

## C5 — Python shared VoxelMap and native binding (2026-08-27)

C5 is complete on P as a controlled Python ownership migration. O remained
read-only. `planning.native.geometry.NativeGeometryContext` now owns one
immutable native VoxelMap per process and supplies stable
`VoxelCollisionChecker` and `GlobalRoutePlanner2D` consumers. Different
collision radii share the map; route configuration and voxel artifact identity
are validated before attachment. The context is non-pickleable, spawn workers
load their own map, and fork reuse fails closed.

~~~
C5_PYTHON_NATIVE_BINDING=PASS
BINDING_IMPLEMENTATION=CTYPES
PYBIND11_ADOPTED=NO
PYTHON_SHARED_VOXELMAP_OWNERSHIP=PASS
PYTHON_VOXELMAP_LOAD_COUNT_PER_PROCESS=1
PYTHON_OCCUPANCY_BUFFER_COUNT_PER_PROCESS=1
COLLISION_ROUTE_SHARED_MAP_ID=PASS
NATIVE_FAIL_CLOSED=PASS
PYTHON_PRODUCTION_FALLBACK_COUNT=0
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
DEPTH_SAFETY_REGRESSION=PASS
P_NATIVE_TARGETS_BUILD=PASS
P_PYTHON_NATIVE_IMPORT=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
ALGORITHM_CHANGED=NO
NEXT_PHASE=C6 BRIDGE PROTOCOL TRANSPORT
~~~

The direct-to-shared paired benchmark used the same 1000-row candidate
fixture. Collision throughput was `48258.601195` to `48083.597432`
poses x actions/sec (`-0.362637%`); route throughput was `2130.787380` to
`2124.313410` routes/sec (`-0.303830%`). Map and occupancy allocations both
fell from 2 to 1. Max RSS was `382216 KB` to `382608 KB` (`+392 KB`). The
one-time geometry initialization increased from `0.133446906 s` to
`0.169326813 s` because the shared owner performs strict cache identity/SHA
work; this absolute cost is recorded explicitly. The synthetic mission
generation fixture remained within `+0.483474%` wall time and had identical
direct/shared results per seed.

The exact owner, copy audit, ABI, parity, benchmark, installation, and
wrapper-line limitations are recorded in `docs/PYTHON_NATIVE_BINDING.md`.
Focused owner/API tests passed 35 cases. Collision covered 121 exact numeric
cases; route covered 127 synthetic and 1000 production-like cases; mission
accept/order covered 1000 rows; Teacher covered 100 rows; depth, mask
generator, policy, and C++ VoxelMap fixture gates all passed. No CUDA
collision parity is claimed: GPU validation remains `DEFERRED / NOT_AVAILABLE`.

No C++ algorithm source, Unity source, CLI, formal data, BC/AWAC code, or
Bridge/protocol code was changed before C6.

## C6 — Bridge, protocol, and transport parity (2026-08-28)

C6 compile closure and normal-path parity are complete on P, but production
promotion is intentionally not complete. O and Unity stayed read-only.

~~~text
C6_BRIDGE_PROTOCOL_TRANSPORT=PARTIAL
COMPILE_REPAIR_SCOPE_EXPANDED=NO
P_FULL_BUILD=PASS
BRIDGE_COMPILE=PASS
MISSING_BEHAVIOR_COUNT=0
DUPLICATE_BEHAVIOR_COUNT=0
PROTOCOL_CONSTANT_PARITY=PASS
CANONICAL_BYTES_PARITY=PASS
HASH_PARITY=PASS
WIRE_CODEC_PARITY=PASS
TRANSPORT_ONLY_PARITY=PASS
COMMAND_GATEWAY_PARITY=PASS
RESULT_GATEWAY_PARITY=PASS
SNAPSHOT_GATEWAY_PARITY=PASS
RESET_GATEWAY_PARITY=PASS
TELEMETRY_PARITY=PASS
REAL_UNITY_RUNTIME_PARITY=PASS
PHYSICS_PARITY=PASS
DEPTH_PARITY=PASS
COLLISION_PARITY=PASS
RUNTIME_IDENTITY_PARITY=PASS
PROCESS_CLEANUP=PASS
MANAGED_PORT_PARITY=PASS
DIRECT_DEFAULT_PARITY=FAIL
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE=NO
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=C6.1 BRIDGE PRODUCTION CUTOVER
~~~

The initial P red build was limited to missing ROS logging declarations in
result_gateway.cpp/snapshot_gateway.cpp and the split telemetry AsInt32
helper. P added only the two ROS includes and the exact O-local helper;
business logic, wire fields, retry logic, sockets, and threads were not
changed. The complete O and P Release builds then exited zero. P's only
remaining warning is an unused AsInt32 helper in the endpoint snapshot
translation unit.

The C6 split inventory is recorded in docs/BRIDGE_PROTOCOL_TRANSPORT_PARITY.md.
P has one formal source owner per Bridge target, one BridgeTransport ZMQ
context owner, RAII socket closure, and no duplicate monolithic Bridge source.
The canonical protocol header and all 27 language-neutral fixtures are
byte-identical O/P. O/P C++ golden contracts, the P 99-case focused suite,
and the P 24-case real-process Bridge suite pass.

The local transport stress sent 1000 interleaved transactions across four
Unity identities and one Python identity. O/P metrics matched exactly:
1000 accepted/ACK/relay/receipt/commit, zero pending/cross-talk/conflict,
and one intentional malformed-packet protocol error. The same frozen Player
(SHA256
61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365) ran through O and P
for 20 reliable primitives; both passed reset, 20/20, 25/25, ACK/commit,
snapshot, depth, collision, physics, and identity checks. With equal
explicit retry windows the metrics files were byte-identical. Runtime state
IDs and simulation times differ only because they are generated at launch.

Two blockers remain: the managed/config depth port is 12254 while each
Bridge's direct constructor default is 11254, and a default 0.5-second live
run showed one timing-sensitive extra P command retry/forward. Equal-budget
runs removed the latter. P ZMQ option-failure handling also needs explicit
fault injection against O before cutover. These are recorded as parity gates,
not silently normalized.

No C6.1 cutover, C7, CUDA collision claim, formal data collection, or commit
was performed. Detailed commands, target hashes, endpoint mapping, runtime
artifact paths, and the C6 modified-file list are in
docs/BRIDGE_PROTOCOL_TRANSPORT_PARITY.md.

## C6.0.1 — Bridge contract closure (2026-08-28)

The C6.0.1 follow-up closed the endpoint ownership seam without changing
normal Bridge message semantics. The unique P production owner is
`python/planning/runtime/ports.py`; `WorkerRuntimeSpec` is the managed owner
and `DirectRuntimePortProfile` is the explicit direct owner. The old O C++
reliable-port derivations are not present in P. P rejects missing, unknown,
mixed, duplicate, or colliding profiles before child startup.

~~~text
C6_0_1_BRIDGE_CONTRACT_CLOSURE=FAIL
PORT_OWNER_UNIQUE=YES
MANAGED_PORT_PARITY=PASS
DIRECT_DEFAULT_PARITY=PASS (explicit profile)
TWELVE_WORKER_PORT_COLLISION_COUNT=0
O/P_TRANSPORT_TRANSACTIONS=10000/10000
O/P_REAL_UNITY_PRIMITIVES=400/400
O/P_RETRY_COUNT=754/767 (transport fixture)
RETRY_BEHAVIOR=ACCEPTABLE_TRANSPORT_RECOVERY
ZMQ_OPTION_FAILURE_PARITY=FAIL
ZMQ_BIND_FAILURE_PARITY=PASS
FAULT_PROCESS_CLEANUP=PASS
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE=NO
NORMAL_RUNTIME_BEHAVIOR_CHANGED=NO
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=C6.0.1 CONTINUE INVESTIGATION
~~~

The same frozen Unity Player passed 20 successful independent O runs and 20
successful independent P runs, each with reset completion, exact reset
snapshot retrieval, and 20 reliable primitives. All selected primitives were
`COMPLETE`; result receipt, Unity ACK, commit, identity, snapshot, and
physical-execution counts were exact, with zero business duplicate,
conflict, pending, or protocol-error outcomes. The fixed 10,000-transaction
fixture also passed exact accounting. Full latency percentiles, the complete
port matrix, and artifact hashes are recorded in
`docs/BRIDGE_CONTRACT_CLOSURE.md`.

The remaining blocker is fault-contract parity for a first ZMQ socket-option
failure. O continues because that ordinary option return is ignored; P fails
closed with exit code 1. Bind/address-in-use cleanup is equivalent and passed.
This does not change normal socket types, HWM, linger, timeout, poll, retry,
thread ownership, or message order. C6.1 production cutover was not run.

## C6.0.2 — ZMQ socket-option criticality contract

This phase is complete in P and is documented in
[`ZMQ_SOCKET_OPTION_CONTRACT.md`](ZMQ_SOCKET_OPTION_CONTRACT.md). The formal
production inventory contains five option kinds: three required and two
optional performance options. P's centralized policy is fail-closed for
required/unknown faults and warning-plus-degraded-record for optional HWM
faults.

```text
C6_0_2_ZMQ_OPTION_CONTRACT=PASS
PRODUCTION_SOCKET_OPTION_COUNT=5
REQUIRED_OPTION_COUNT=3
OPTIONAL_OPTION_COUNT=2
UNKNOWN_OPTION_COUNT=0
ZMQ_OPTION_FAILURE_CONTRACT=PASS
O_MATCHES_CANONICAL=MIXED
P_MATCHES_CANONICAL=YES
O_HISTORICAL_FAULT_POLICY_DEFECT=YES
REQUIRED_FAULT_FAIL_CLOSED=PASS
OPTIONAL_DEGRADED_MODE=PASS
FAULT_PROCESS_CLEANUP=PASS
FAULT_PORT_RELEASE=PASS
NORMAL_TRANSPORT_REGRESSION=PASS
NORMAL_UNITY_RUNTIME_REGRESSION=PASS
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE=YES
NORMAL_RUNTIME_BEHAVIOR_CHANGED=NO
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=C6.1 BRIDGE PRODUCTION CUTOVER
```

Required/optional fault injection, unknown-option C++ coverage, normal
10,000-transaction transport regression, and 400-primitive Unity regression
passed. The O ignored ordinary option failure is recorded as historical
fault-policy behavior, not as proven incorrect normal behavior. C6.1 was not
executed.

## C6.1 — Bridge production cutover (2026-08-28)

The authorized P-only Bridge cutover is complete.  `unity_bridge_node` keeps
the same executable and ROS node contract while its formal implementation is
the split `planning::bridge::UnityBridgeNode` owner tree.  O and Unity were
read-only; no collision, route, depth-safety, Python algorithm, formal data,
or training change was made.

```text
C6_1_BRIDGE_PRODUCTION_CUTOVER=PASS
FORMAL_BRIDGE_OWNER=planning::bridge::UnityBridgeNode (BridgeNode)
FORMAL_BRIDGE_IMPLEMENTATION_COUNT=1
OLD_BRIDGE_FORMAL_CALLER_COUNT=0
MISSING_BEHAVIOR_COUNT=0
DUPLICATE_BEHAVIOR_COUNT=0
P_FULL_BUILD=PASS
BRIDGE_COMPILE=PASS
PROTOCOL_PARITY=PASS
COMMAND_GATEWAY_PARITY=PASS
RESULT_GATEWAY_PARITY=PASS
SNAPSHOT_GATEWAY_PARITY=PASS
RESET_GATEWAY_PARITY=PASS
TELEMETRY_PARITY=PASS
PORT_CONTRACT=PASS
ZMQ_OPTION_CONTRACT=PASS
FAULT_PROCESS_CLEANUP=PASS
REAL_UNITY_RUNTIME_PARITY=PASS
PHYSICS_PARITY=PASS
DEPTH_PARITY=PASS
COLLISION_PARITY=PASS
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
DEPTH_SAFETY_REGRESSION=PASS
DUPLICATE_PHYSICAL_EXECUTION_COUNT=0
COMMAND_CONFLICT_COUNT=0
PENDING_FINAL_COUNT=0
PROTOCOL_ERROR_COUNT=0
BRIDGE_BINARY_SHA256=63b00f6a0b31aa72ace3e1c437477e7ee8b670934386f3bb1d645745c302d9ed
RUNTIME_ASSEMBLY_SHA256=7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc
UNITY_PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=C7 FINAL C++ BUILD AND FREEZE
```

The source-level scan found one owner per gateway/protocol/transport
behavior and no formal P caller of the old flat path.  Canonical payload
bytes, normalized business accounting, 10,000 transport transactions per
binary, O/P frozen Unity runs, all five option fault rows, and Collision,
Global Route, and Depth Safety regression fixtures passed.  The deletion and
retention ledger is in [`docs/BRIDGE_PRODUCTION_CUTOVER.md`](BRIDGE_PRODUCTION_CUTOVER.md).

The O ROS test source still has six exact-metrics assertions that reject
diagnostic fields emitted by O itself; this read-only baseline mismatch was
not changed or hidden.  The normalized runtime comparator and P runtime suite
passed.  At the time this C6.1 record was written, C7 was not started.

## C7 — Final C++ build, ABI, install, and runtime freeze (2026-08-28)

C7 is complete for P.  O (`/home/xm/XM/xm_ws/src/planning`) and the frozen
Unity checkout were read-only.  No C++/Python source, CMake, package metadata,
BC/AWAC code, formal data, or training artifact was changed during C7; only
the migration documents were updated.

### Unity identity reconciliation

The historical Unity freeze report labels the editor assembly hash as the
runtime assembly.  The actual Player-loaded assembly is the Player build
artifact and matches the canonical Player:

```text
CANONICAL_UNITY_PLAYER=/home/xm/XM/xm_ws/src/unity/XMflight.x86_64
CANONICAL_UNITY_PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
CANONICAL_UNITY_RUNTIME_ASSEMBLY=/home/xm/XM/xm_ws/src/unity/XMflight_Data/Managed/Assembly-CSharp.dll
CANONICAL_UNITY_RUNTIME_ASSEMBLY_SHA256=7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc
EDITOR_ASSEMBLY_SHA256=11f843ba787566ba1838d47d205978646b6d5eee0126f63260a6352a82551986
```

`11f843...` is the Unity editor/compile artifact (it contains editor-only
strings such as `UnityEditor` and `AssetDatabase`); `7ffb...` is the
PlayerScriptAssemblies/Player-loaded artifact.  This is an artifact-label
correction, not a Unity source change.  The Player and Player runtime assembly
remain frozen.

### Reproducible build and install

Two independent clean CMake Release builds compiled and linked all P targets.
The conda `xm` environment was used for configure, compile, and devel-space
runtime import.  Its ROS `python_distutils_install.sh` rejects the legacy
`--install-layout=deb` option, so a separate `/usr/bin/python3` Catkin install
build was used for the install-space gate.  That split is an environment
toolchain conflict, not a source or link failure.

```text
CLEAN_BUILD=PASS
DEVEL_SPACE_BUILD=PASS
INSTALL_SPACE_BUILD=PASS
P_FULL_BUILD=PASS
CONDA_INSTALL_TOOLCHAIN=ENVIRONMENT_TOOLCHAIN_CONFLICT
P_TARGET_SET=planning_voxel_map,planning_collision_checker,planning_global_route,planning_depth_safety,unity_bridge_node,map_loader_node,visualization_node,voxel_map_parity_fixture
```

The Release compile contract was C++17, `-DNDEBUG`, `-fPIC` for libraries,
`-O2 -Wall -Wextra`, `-O3` on the geometry targets, and OpenMP on Collision.
The Bridge links ROS Noetic/ZMQ/OpenSSL/msgpack and has no CUDA dependency.
Missing `data/motion_primitives` is guarded by CMake and was not fabricated.
Repeated-build target/dependency/ABI semantics passed; differing whole-file
hashes for Collision/Route are only temporary build-root RUNPATH differences.

### ABI, implementation uniqueness, and stale-source gates

The required public C ABI contains 20 symbols.  The exact manifest and hashes
are recorded in [`docs/CXX_ABI_MANIFEST.md`](CXX_ABI_MANIFEST.md).

```text
ABI_SYMBOL_GATE=PASS
REQUIRED_SYMBOL_COUNT=20
MISSING_REQUIRED_SYMBOL_COUNT=0
UNEXPECTED_LEGACY_SYMBOL_COUNT=0
STALE_SOURCE_PATH_COUNT=0
FORMAL_COLLISION_IMPLEMENTATION_COUNT=1
FORMAL_ROUTE_IMPLEMENTATION_COUNT=1
FORMAL_DEPTH_SAFETY_IMPLEMENTATION_COUNT=1
FORMAL_BRIDGE_IMPLEMENTATION_COUNT=1
```

The formal P backends are one C++17/OpenMP Collision implementation, one
native C++17 Global Route implementation, one native C++17 Depth Safety
implementation, and one split C++17/ROS/ZMQ Bridge implementation.  No
formal CUDA/hybrid collision library or automatic Python fallback remains.
The Python reference paths remain available only for explicit test/debug use.
The C++ source manifest is
`2e940edac81555bf0f125b5059a93db951c355ece075cc18e22cf37edae88a77`.
It was rechecked after the documentation-only C7 edit using 60 sorted
relative C++ source/header paths plus `CMakeLists.txt` and `package.xml`
(62 files total), with NUL separators around each filename and file payload.

### Runtime, correctness, and performance gates

The native ctypes binding imported from both devel and system install spaces,
loaded the shared VoxelMap once, and passed fork protection and map identity
checks.  Collision CPU numeric parity was exact for 121 cases.  Candidate
accept and Teacher 105-action-mask parity were exact for 100 missions.  The
Global Route and Depth Safety production-like fixtures also passed.

```text
PYTHON_NATIVE_IMPORT=PASS
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
DEPTH_SAFETY_REGRESSION=PASS
COLLISION_PERFORMANCE_RATIO=0.983331 (filter); 0.992835 (total)
ROUTE_PERFORMANCE_RATIO=0.991387
DEPTH_SAFETY_PERFORMANCE_RATIO=1.009191
PERF_COUNTERS=NOT_AVAILABLE
GPU_RUNTIME_VALIDATION_DEFERRED=YES
```

The Collision ratio is the median of 30 fixed-fixture, affinity-pinned
end-to-end filter samples, P/O; the total-wall-time ratio is shown separately.
The Route ratio compares P with the frozen P benchmark baseline.  `perf` was
not installed, so no hardware-counter claim is made.  CUDA collision runtime
was unavailable and is explicitly `DEFERRED / NOT_AVAILABLE`; removal from the
formal P path is `OUT_OF_FORMAL_SCOPE`, not a claim that CUDA is incorrect.
PyTorch CUDA/AMP/GradScaler support used by BC/AWAC was not touched.

### Protocol and real-runtime gates

The 75-case protocol/golden/C++ contract suite and the 24-case real P Bridge
integration suite passed.  The 10,000-transaction O/P transport comparison
had exact accepted commands, ACKs, receipts, commits, snapshots, and hashes;
duplicate/conflict/pending/protocol-error business counts were zero.  The
canonical Unity Player completed the one-worker 100-primitive smoke.  The
bounded twelve-worker smoke completed 240 primitives with independent runtime
identities and port profiles.  An intentional failed Bridge on one worker
failed closed while the other 11 workers completed and all processes/ports
were cleaned up.

```text
PROTOCOL_BRIDGE_REGRESSION=PASS
ONE_WORKER_RUNTIME_SMOKE=PASS
TWELVE_WORKER_RUNTIME_SMOKE=PASS
SINGLE_WORKER_FAULT_ISOLATION=PASS
DUPLICATE_PHYSICAL_EXECUTION_COUNT=0
COMMAND_CONFLICT_COUNT=0
PENDING_FINAL_COUNT=0
PROTOCOL_ERROR_COUNT=0
PROCESS_CLEANUP=PASS
```

The twelve-worker result validates the bounded runtime/Bridge smoke only.  It
does not promote the separate reliable Teacher multi-worker collector or its
formal-data readiness; those remain Python-pipeline blockers.

### Build hygiene and freeze decision

`compileall` passed with its bytecode cache redirected to `/tmp`.  The clean
rebuild emitted one low-severity existing `-Wunused-function` warning for
`AsInt32` in `src/protocol/endpoint_observation_snapshot_wire.cpp`; there
were no high-severity warnings.  `clang-format`, `clang-tidy`, and `perf` were
not available.  Generated caches and bytecode were moved, not deleted, to
`/tmp/xm-c7-source-tree-garbage.KYx1El`.

```text
COMPILER_WARNING_COUNT=1
HIGH_SEVERITY_WARNING_COUNT=0
C7_FINAL_CXX_FREEZE=PASS
CXX_FINAL_V1=PASS
CXX_FROZEN_FOR_PLANNING=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=PLANNING PYTHON PIPELINE FINALIZATION
```

Earlier C++ notes that called native Global Route `REJECT`, treated split
Bridge evidence as static-only, or left the full build unverified are
historical and superseded by C3.1, C6.1, and C7 respectively.  The remaining
valid blockers are outside this C++ freeze: reliable-exact Teacher collection
and merge/provenance closure, BC checkpoint/training/evaluation handoff, and
AWAC/RL pipeline finalization.  C7 stopped without starting those phases.
