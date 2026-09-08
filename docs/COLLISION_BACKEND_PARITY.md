# Collision Backend Parity — C2

Date: 2026-08-27

## Scope

O (`/home/xm/XM/xm_ws/src/planning`) is the read-only behavior baseline.  P
(`/home/xm/XM/src`) is the candidate.  C2 compared only the collision C ABI
and shared `VoxelMap` runtime.  It did not modify O, Unity, Python algorithms,
Bridge/protocol, Global A*, CUDA source, or formal project data, and it did
not perform a cutover.

## Availability

```text
O_CPU_AVAILABLE=YES
O_CUDA_AVAILABLE=NO
O_HYBRID_AVAILABLE=NO
P_CPP_AVAILABLE=YES
```

O CPU and P C++ expose path, single-action, actions-batch, and
pose-actions-batch entry points.  O CUDA exposes actions-batch and
pose-actions-batch only.  No collision library in either tree exposes a
separate reachability or minimum-clearance function; minimum distance is an
output tested through the available entry points.  O CUDA compiled, but its
runtime constructor returned null because the host has no usable NVIDIA
driver/device.  O hybrid is the Teacher CLI's CPU-process plus CUDA-thread
scheduler and was consequently `NOT_AVAILABLE`.  P has no CUDA or hybrid
backend.

## Fixtures

The parity script is
`tests/compare_collision_backend_parity.py`.  It tests:

- C1 sparse, single-boundary, boundary-shell, dense, and empty layouts;
- no-collision, start/middle/end collision, boundary, and multiple check-step
  paths;
- single action, actions batch, and pose x actions batch with permuted order;
- inflate radii `0.35` and `0.40`;
- a temporary production-like cache built from O's raw forest point cloud.

The temporary cache is at `/tmp/xmflight_collision_c2/` only.  The source
point-cloud SHA256 is
`dfd87a5db1ab98276eda57e79de677f711da56f308a373b93e72484fa5876d40`; the
cache SHA256 is
`a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691`.
The cache has voxel size `0.10`, origin `[-1023,-1020,-1]`, shape
`[2046,2045,42]`, and `1,740,947` occupied keys.  It was generated with O's
formal generator/builder and was never copied into `data/`.

## Parity gates

The CPU comparator covered 121 cases.  All required discrete outputs were
exact.  Float outputs were tested bitwise first and had zero observed delta:

```text
PATH_COLLISION_PARITY=PASS
SINGLE_ACTION_PARITY=PASS
ACTIONS_BATCH_PARITY=PASS
POSE_ACTIONS_BATCH_PARITY=PASS
FIRST_COLLISION_INDEX_PARITY=PASS
ENDPOINT_PARITY=PASS
MIN_DISTANCE_PARITY=PASS
BATCH_ORDER_PARITY=PASS
CPU_MAX_ABS_DELTA=0
CPU_MAX_RELATIVE_DELTA=0
PRODUCTION_ASSET_PARITY=PASS
```

The O CUDA gates are `NOT_AVAILABLE`, not pass-by-assumption.  Candidate
accept-vector and Teacher valid-action-mask parity were not run because this
phase is limited to the collision C ABI and the formal Teacher path was not
changed.

## Ownership gate

The shared-owner fixture creates one P `VoxelMap`, two collision checkers at
radii `0.35` and `0.40`, destroys the external map handle, and queries both
checkers successfully.  The C1 fixture also proves the same lifetime for one
`GlobalRoutePlanner` from the same map handle.

```text
SHARED_VOXELMAP_RUNTIME_OWNERSHIP=PASS
map_load_count=1
occupied_buffer_allocation_count=1
checker_count=2
complete_occupancy_copies_in_consumers=0
```

The allocation count is logical fixture evidence for one map-owned bitmap,
not malloc instrumentation.  The planner's 2-D blocked grid is a derived
route representation, not a complete occupancy copy.

## Benchmark and decision

Release benchmark artifact:
`/tmp/xmflight_collision_c2/collision_backend_benchmark.json` (18 rows: 3
runs each for O CPU and P C++ at OpenMP 1/2/4).  P's median pose x actions
throughput was `442,518`, `491,280`, and `397,903` per second versus O's
`503,187`, `529,478`, and `446,073`.  Peak process RSS was `103.312 MB`.
P was below O at every tested thread count, so the formal production
throughput gate is not met.

```text
C2_COLLISION_PARITY=PARTIAL
P_CPP_COLLISION_PRODUCTION_CANDIDATE=NO
ALGORITHM_CHANGED=NO
NEXT_PHASE=C2 COLLISION FIX / INVESTIGATION
```

Remaining blockers are CUDA device/runtime evidence, hybrid scheduler parity,
and P C++ throughput parity.  No C2.1 cutover or CUDA deletion is permitted.

## C2.1 Collision hot-path regression investigation (2026-08-27)

The previous C2 performance result is retained above as historical evidence.
C2.1 investigated that regression with O still read-only and P as the only
modified tree. The only implementation change was a P collision-kernel lookup
seam; collision semantics, the public C ABI, output buffers, CUDA source,
hybrid behavior, Global A*, Bridge, and Python algorithms were not changed.

### Build and flag audit

The official CMake collision targets were both Release builds with effective
`-O3 -DNDEBUG -fPIC -fopenmp`, no LTO, no explicit `-march`/`-mtune`, default
visibility, and default exception/RTTI settings. Both link with OpenMP
(`libgomp` and `pthread`). O's configured language standard is C++14; P's is
C++17 because P's project contract requires C++17. That standard difference
is structural rather than a hot-loop behavior change, so a controlled O C++17
temporary build under `/tmp` was used for the apples-to-apples benchmark.

The complete CMake-generated compile commands were:

```text
/usr/bin/c++ -DROSCONSOLE_BACKEND_LOG4CXX -DROS_BUILD_SHARED_LIBS=1 -DROS_PACKAGE_NAME=\"planning\" -Dplanning_collision_checker_EXPORTS -I/tmp/xm-cxx-c0-o-devel/planning/include -I/home/xm/XM/xm_ws/src/planning/include -I/opt/ros/noetic/include -I/opt/ros/noetic/share/xmlrpcpp/cmake/../../../include/xmlrpcpp -I/usr/local/cuda-11.8/include -O3 -DNDEBUG -fPIC -std=c++14 -O2 -Wall -Wextra -O3 -fopenmp -o CMakeFiles/planning_collision_checker.dir/src/collision_checker.cpp.o -c /home/xm/XM/xm_ws/src/planning/src/collision_checker.cpp
/usr/bin/c++ -DROSCONSOLE_BACKEND_LOG4CXX -DROS_BUILD_SHARED_LIBS=1 -DROS_PACKAGE_NAME=\"planning\" -Dplanning_collision_checker_EXPORTS -I/tmp/xm-cxx-c0-p-devel/planning/include -I/home/xm/XM/src/include -I/opt/ros/noetic/include -I/opt/ros/noetic/share/xmlrpcpp/cmake/../../../include/xmlrpcpp -O3 -DNDEBUG -fPIC -O2 -Wall -Wextra -O3 -fopenmp -std=c++17 -o CMakeFiles/planning_collision_checker.dir/src/geometry/collision_checker.cpp.o -c /home/xm/XM/src/src/geometry/collision_checker.cpp
/usr/bin/c++ -DROSCONSOLE_BACKEND_LOG4CXX -DROS_BUILD_SHARED_LIBS=1 -DROS_PACKAGE_NAME=\"planning\" -Dplanning_collision_checker_EXPORTS -O3 -DNDEBUG -fPIC -std=c++17 -O2 -Wall -Wextra -O3 -fopenmp -shared -o /tmp/xmflight_collision_c2/o_cxx17/libplanning_collision_checker.so /home/xm/XM/xm_ws/src/planning/src/collision_checker.cpp
```

The official link commands were:

```text
/usr/bin/c++ -fPIC -O3 -DNDEBUG  -shared -Wl,-soname,libplanning_collision_checker.so -o /tmp/xm-cxx-c0-o-devel/planning/lib/libplanning_collision_checker.so CMakeFiles/planning_collision_checker.dir/src/collision_checker.cpp.o  -lgomp -lpthread
/usr/bin/c++ -fPIC -O3 -DNDEBUG  -shared -Wl,-soname,libplanning_collision_checker.so -o /tmp/xm-cxx-c0-p-devel/planning/lib/libplanning_collision_checker.so CMakeFiles/planning_collision_checker.dir/src/geometry/collision_checker.cpp.o  -Wl,-rpath,/tmp/xm-cxx-c0-p-devel/planning/lib: /tmp/xm-cxx-c0-p-devel/planning/lib/libplanning_voxel_map.so -lgomp -lpthread
```

The controlled O C++17 library SHA256 is
`2aa36530f466f78e890e0f668ab0edc36e61b614ea76f6b65272b83991487634` and the
post-fix P collision library SHA256 is
`411b41f5992993f2b9ba725457aa61ea72e33fa449cc52c4b7d588e07ece6680`.
`BUILD_FLAGS_PARITY=PASS` for the controlled performance comparison; the
official O C++14/P C++17 distinction remains recorded as a structural build
difference.

### Hot-loop and allocation classification

| Stage | O CPU | P after C2.1 | Classification |
| --- | --- | --- | --- |
| voxel lookup | direct cached checker fields and bitmap | direct cached fields and a read-only pointer into the immutable map bitmap | `EXACT` after fix; `EXTRA_WORK_IN_P` before fix |
| neighbor iteration | cached offsets, same loop order | same loop order and offsets | `EXACT` |
| inflation offsets | constructor-time calculation | constructor-time calculation | `EXACT` |
| path sampling | same `check_step` loop | same | `EXACT` |
| distance calculation | same double-precision expression/order | same | `EXACT` |
| first collision | same early break and index | same | `EXACT` |
| endpoint | same final-point write | same | `EXACT` |
| batch rows/output | same row-major writes and validation | same public ABI/output | `EXACT` |
| OpenMP | `schedule(static)`, `if(query_count > 1)` / `if(pose_count > 1)` | same pragmas and schedule | `EXACT` |
| ownership | checker owns bitmap | checker retains `shared_ptr<const VoxelMap>` and caches a view | `STRUCTURAL_ONLY` |

Before C2.1, P called `WorldToVoxel` and `IsOccupiedVoxel` for every
neighbor. Those calls repeated map bounds/bitmap lookup across the shared-map
module. P now reads the same immutable bitmap with the O loop's cached
metadata; the map remains the sole complete occupancy owner. No algorithmic
difference was found: `DIFFERENT_ALGORITHM=0`.

The allocation probe warmed both libraries and counted allocations over 1000
pose x action calls. Both returned `STEADY_STATE_ALLOCATIONS=0`. Output
vectors are caller-owned and preallocated; neither kernel resizes vectors,
zeros a temporary, copies input arrays, creates an output temporary, or copies
the shared pointer per batch. P performs one map metadata read and one
`shared_ptr` ownership transfer at checker construction. Each batch creates
the same single conditional OpenMP region as O.

### Numeric and integration parity

The post-fix C ABI comparator passed all 121 O/P CPU cases, including the
temporary production-like cache:

```text
CPU_PARITY_RESULT=PASS
CPU_MAX_ABS_DELTA=0
CPU_MAX_RELATIVE_DELTA=0
PRODUCTION_ASSET_PARITY=PASS
SHARED_VOXELMAP_RUNTIME_OWNERSHIP=PASS
```

The fixed integration fixture runs the same 100 deterministic mission rows
in isolated O/P subprocesses. Candidate acceptance rows and Teacher current
valid-action masks were both exact:

```text
CANDIDATE_ACCEPT_VECTOR_MISSIONS=100
CANDIDATE_ACCEPT_VECTOR_PARITY=PASS
TEACHER_VALID_ACTION_MASK_MISSIONS=100
TEACHER_VALID_ACTION_MASK_PARITY=PASS
```

The fixture result SHA256 is
`e1cc853c91c07501b93a8f32e1a0720d5272272ae02b73de54a99f98089d6d8f`.
Both O and P candidate-vector SHA256 values are
`78ff9e8e47e42d908103aa3cab2f53c4955e2bb2c72a324cb57d84001d1dfacb`; both
Teacher valid-mask SHA256 values are
`a02a3cad5ea0510c95cfac9dbc0ee8f711c378afc2f3d5da81cf5c213e44ee48`.

### Controlled Release performance

The kernel benchmark used three runs at OMP 1/2/4 with the temporary O C++17
library and P's post-fix C++17 library. Values below are median throughput for
each three-run group, in operations per second:

| OMP | O paths | P paths | O single actions | P single actions | O poses x actions | P poses x actions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 135389 | 137663 | 86461 | 86220 | 490532 | 493611 |
| 2 | 136448 | 137286 | 86627 | 87763 | 516287 | 531508 |
| 4 | 138707 | 139063 | 88829 | 88736 | 453884 | 456812 |

The best median kernel result is P `531508 poses x actions/sec` at OMP2
versus O `516287` at OMP2, a P delta of `+2.95%` (`102.95%` of O). The
kernel gate therefore exceeds the required 95%.

The fixed 1000-candidate end-to-end probe used the same CPU affinity
(`taskset -c 4-7`) and three runs at each OMP setting. Values are medians;
times are milliseconds and RSS is process-level MB. O's raw C ABI combines
map construction with checker construction, so its separable map time is
`N/A`; P reports one shared-map construction and a checker view construction.

| OMP | backend | map init | checker init | init total | filter 1000 | total | peak RSS | map loads |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | O raw | N/A | 0.003913 | 0.004058 | 0.160107 | 0.171593 | 3.184 | 1 |
| 1 | P shared | 0.000990 | 0.003250 | 0.004308 | 0.149909 | 0.162568 | 3.074 | 1 |
| 2 | O raw | N/A | 0.004054 | 0.004204 | 0.114459 | 0.127617 | 3.094 | 1 |
| 2 | P shared | 0.000921 | 0.003152 | 0.004175 | 0.096253 | 0.108647 | 3.145 | 1 |
| 4 | O raw | N/A | 0.003779 | 0.003916 | 0.131238 | 0.142148 | 3.039 | 1 |
| 4 | P shared | 0.000910 | 0.003091 | 0.004193 | 0.129337 | 0.141100 | 3.145 | 1 |

The best total-throughput medians are P `9,204,120 candidates/sec` at OMP2
and O `7,835,947 candidates/sec` at OMP2, a P delta of `+17.46%`. The
Python-wrapper benchmark reported the same `103.578 MB` median RSS for O and
P; the smaller C++ probe RSS above excludes Python import/cache overhead.

`perf` and `valgrind/callgrind` are not installed in this environment, so
`perf stat` counters (cycles, instructions, IPC, branches, branch misses,
cache references/misses, and context switches) and `perf record` function
profiles are `NOT_AVAILABLE`; no counter or function share was fabricated.
The direct source audit and allocation probe are the available hot-path
evidence.

### C2.1 gate

```text
C2_1_COLLISION_REGRESSION=PASS
BUILD_FLAGS_PARITY=PASS
DIFFERENT_ALGORITHM=0
EXTRA_ALLOCATIONS_P=0 (same as O in steady state)
CANDIDATE_ACCEPT_VECTOR_PARITY=PASS (100/100)
TEACHER_VALID_ACTION_MASK_PARITY=PASS (100/100, 105 actions each)
P_CPP_COLLISION_PRODUCTION_CANDIDATE=YES
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
```

O CUDA still compiles but cannot initialize without a usable NVIDIA
driver/device; O hybrid remains unavailable and P CUDA/hybrid code was not
deleted. C2.1 therefore closes the CPU collision candidate gate only; it does
not claim CUDA/hybrid parity and does not authorize cutover in this slice.

Modified P files for C2.1:

- `include/planning/geometry/voxel_map.hpp`
- `src/geometry/collision_checker.cpp`
- `tests/compare_collision_teacher_mask_fixture.py`
- `docs/COLLISION_BACKEND_PARITY.md`
- `docs/CXX_FOUNDATION_MIGRATION.md`
- `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`

Evidence artifacts remain under `/tmp/xmflight_collision_c2/`; no formal
project data was generated or copied. The next authorized phase is **C2.2
PRODUCTION CUTOVER**, which was not executed here.

## C2.2 production cutover (2026-08-27)

The historical C2.1 review above is retained as evidence of the pre-cutover
state. The explicit C2.2 authorization has now been applied to P only.

```text
C2_2_PRODUCTION_CHANGE_AUTHORIZED=YES
C2_2_COLLISION_PRODUCTION_CUTOVER=PASS
FORMAL_COLLISION_BACKEND=C++17_OPENMP_CPU
PRODUCTION_BACKEND_UNIQUE=YES
PYTHON_PRODUCTION_FALLBACK=NO
CUDA_HYBRID_FORMAL_PATHS=0
GPU_RUNTIME_VALIDATION=DEFERRED / NOT_AVAILABLE
ALGORITHM_CHANGED=NO
```

`planning.safety.collision_checker.VoxelCollisionChecker` is the stable
Python owner. `cpu`, `cpp`, `auto`, and `cpp_cpu` resolve to the one native
backend. The NumPy reference requires the explicit debug environment opt-in;
native initialization and operation errors fail closed. Production CLI
parsing rejects `python`, `cuda`, and `hybrid`.

The cutover does not remove or alter PyTorch CUDA, AMP, or GradScaler used by
BC/AWAC. P had no formal self-owned collision CUDA/hybrid source or target to
delete; the absence and rejection ledger records `OUT_OF_FORMAL_SCOPE`, not
`PROVEN_INCORRECT`. O CUDA comparator code remains test-only and GPU runtime
validation remains unavailable.

Final C2.2 gates:

```text
COLLISION_NUMERIC_PARITY=PASS (121/121, max abs/relative delta 0)
CANDIDATE_ACCEPT_VECTOR_PARITY=PASS (100/100)
TEACHER_VALID_ACTION_MASK_PARITY=PASS (100/100, 105 actions)
MISSION_COLLISION_INTEGRATION_PARITY=PASS
TEACHER_COLLISION_INTEGRATION_PARITY=PASS
P_COLLISION_TARGET_BUILD=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
CXX_SHARED_VOXELMAP=PASS
PYTHON_SHARED_VOXELMAP_BINDING=NOT_STARTED
```

The controlled Release OMP benchmark retained the C2.1 gate result:

```text
P_KERNEL_THROUGHPUT=531508 poses x actions/sec
O_KERNEL_THROUGHPUT=516287 poses x actions/sec
KERNEL_DELTA_PERCENT=+2.95%
P_END_TO_END_THROUGHPUT=9204120 candidates/sec
O_END_TO_END_THROUGHPUT=7835947 candidates/sec
END_TO_END_DELTA_PERCENT=+17.46%
```

No Bridge/protocol repair, Global A*, Python shared-VoxelMap binding, formal
data collection, or C3 work was started. The next phase is
`C3 GLOBAL A* SEMANTICS AND PARITY`.
