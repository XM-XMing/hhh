# C5 Python shared VoxelMap and native binding

Date: 2026-08-27

This document records the controlled C5 migration for the optimized Planning
tree.

```text
O=/home/xm/XM/xm_ws/src/planning        (read-only behavior source)
P=/home/xm/XM/src                      (only modified tree)
C5_PYTHON_NATIVE_BINDING=PASS
```

The change is limited to the Python ownership seam. O, the Unity checkout,
the C++ collision/route/depth algorithms, Mission and Teacher scoring, BC,
AWAC, Bridge/protocol code, CLI arguments, and formal data were not changed.

## Owner and process boundary

`planning.native.geometry.NativeGeometryContext` is the single Python owner
for one immutable native VoxelMap in one process. It reads an existing voxel
cache, validates its metadata and SHA256, creates exactly one
`planning_voxel_map_create` handle, and caches public-domain consumers:

```text
NativeGeometryContext
├── VoxelCollisionChecker(radius=0.35, ...)
├── VoxelCollisionChecker(radius=0.40, ...)
└── GlobalRoutePlanner2D(route-config, ...)
```

Different collision radii create different checker handles but retain the same
map handle. Route configuration is part of the context identity; a planner
with a different configuration cannot be attached to that context. The
identity includes:

| Field | C5 contract |
|---|---|
| voxel artifact path | resolved absolute path |
| voxel artifact SHA256 | exact file digest; optional expected digest is fail-closed |
| voxel size | positive finite value, matching the cache |
| origin | int64 shape `(3,)` |
| grid shape | positive int64 shape `(3,)` |
| route configuration | resolution, flight bounds, lookahead, tracking margin, nearest-free radius |

The context is deliberately non-pickleable. `spawn` workers load the
read-only artifact themselves, and a `fork` child cannot reuse the parent's
native handle. `close()` closes route/checker consumers before the map owner;
the C++ shared-pointer contract still permits a consumer to retain the map
after an independent external map handle is destroyed.

The production Python call paths now use this owner for Mission generation,
Mission route validation/audit, Teacher relabeling, and Teacher collection.
The collection worker disables only the legacy `UnityForestEnv` constructor
load and attaches the already-created shared checker before readiness or any
transition. `runtime/unity_env.py` and the Unity project were not modified.

## Binding choice

```text
BINDING_IMPLEMENTATION=CTYPES
PYBIND11_ADOPTED=NO
PYBIND11_IMPORT=NO (pybind11 is not installed in conda activate xm)
```

The existing C ABI wrapper remains the implementation detail of the native
owner and stable domain APIs remain `planning.safety` and `planning.mission`.
No scripts call `ctypes`, `dlopen`, a C ABI symbol, a shared-library path, or a
raw native pointer. A pybind11 spike was not adopted: Python 3.8.20 import and
CMake discovery were unavailable, so the required throughput/install/import
comparison and the required 200-line wrapper reduction cannot be claimed.

The pre-C5 source snapshot was not versioned in P, so an exact historical
line count cannot be reconstructed without inventing evidence. The current
manual ctypes binding ranges (library discovery plus the collision and route
backend classes) total 667 source lines. No wrapper reduction was performed:

```text
WRAPPER_LOC_BEFORE=NOT_RECORDED_EXACTLY
WRAPPER_LOC_AFTER=667
WRAPPER_LOC_REDUCTION=0 (no pybind11 adoption)
```

## C ABI ownership audit

The audited symbols are:

```text
planning_voxel_map_create / planning_voxel_map_destroy
planning_collision_create_from_voxel_map / planning_collision_destroy
planning_global_route_create_from_voxel_map / planning_global_route_destroy
```

`VoxelMapHandle` owns a `shared_ptr<const VoxelMap>`. CollisionChecker and
GlobalRoutePlanner acquire their own shared pointer from the map handle; they
do not copy the complete occupancy bitmap. Invalid C ABI input returns an
error/null handle and never crosses the boundary as a C++ exception.

The map owner is immutable after creation. Collision queries are read-only;
the collision thread-count setter is configuration and must not race with
queries. A route planner owns mutable last-route/workspace state, so concurrent
`plan` calls on one planner are not permitted; production workers use their
planner serially. Separate processes own separate contexts and handles.

## Current call-chain instrumentation

The direct/pre-C5 seam constructed separate native consumers from the same
cache. The C5 context benchmark measured the same process and fixture:

| Path | map/native constructions before | after | occupancy-buffer allocations before | after |
|---|---:|---:|---:|---:|
| context collision + route pair | 2 | 1 | 2 | 1 |
| Mission generation, single worker | 1 | 1 | 1 | 1 |
| Mission route validation/audit worker | 1 | 1 | 1 | 1 |
| Teacher relabel worker | 1 | 1 | 1 | 1 |
| Teacher collection worker, after collection wiring | 3 | 1 | 3 | 1 |

The collection pre-C5 count includes behavior/prefetch consumers and the
legacy `UnityForestEnv` checker. The intermediate C5 context wiring reduced
that to two; the final constructor flag/injection removes the remaining
legacy load. For multi-process Mission sampling, the parent preflight context
is opened and closed once, and every child worker creates its own context with
one map load. These counts are process-local; they are not a cross-process
pointer-sharing claim.

The focused owner tests verify that collision and route map IDs are equal,
different radii retain the same map with different checker IDs, cache decoding
is not repeated, and two spawned children each create a distinct one-load
context. The fork-reuse test is explicitly fail-closed.

## NumPy copy audit

The shared binding validates caller-provided NumPy dtype and shape. Wrong
dtype/shape is rejected; a correctly typed non-contiguous array is copied once
for the C ABI. Python sequences are accepted for the existing public API and
are counted as dtype conversions when conversion is required.

On the final 3-run, 1000-candidate context fixture:

```text
NUMPY_INPUT_COPY_COUNT=0
NUMPY_CONTIGUITY_CONVERSION_COUNT=0
NUMPY_DTYPE_CONVERSION_COUNT=2000
NUMPY_OUTPUT_ARRAY_ALLOCATION_COUNT=2009 per run
```

The 2000 conversions are the two sequence-form route endpoints for 1000
routes. The 2009 output-array count includes the collision warm-up, 200 timed
batch calls, route-owner initialization buffers, and 1000 returned route
arrays; it is expected output/ABI-buffer allocation, not an input occupancy
copy. Collision and route output digests are unchanged. Depth Safety retains
the C4.1 wrapper and its established input handling; a separate strict depth
copy audit was not widened into C5 because changing that wrapper is outside
this phase.

## Parity and benchmark evidence

The shared context benchmark used the read-only cache
`/tmp/xmflight_collision_c2/forest_voxels_10cm.npz` and the fixed 1000-row
candidate fixture. Pair fixture SHA256:

```text
225bb2f8d8e5af9acd7b0c2bae05db69d082c8759242d5866f923208ef67acd4
```

Final three-run direct-versus-shared results:

| Metric | direct/pre-C5 | shared/C5 | delta |
|---|---:|---:|---:|
| best collision poses x actions/sec | 48258.601195 | 48083.597432 | -0.362637% |
| best route/sec | 2130.787380 | 2124.313410 | -0.303830% |
| mean geometry init/sec | 0.133446906 | 0.169326813 | +26.88% |
| max peak RSS | 382216 KB | 382608 KB | +392 KB |
| map constructions | 2 | 1 | -1 |
| occupancy allocations | 2 | 1 | -1 |

The initialization increase is a one-time absolute increase of about 36 ms,
primarily the required artifact SHA/strict-cache identity work; it is called
out explicitly rather than hidden by the throughput gate. The synthetic
mission-generation fixture was also run three times per mode: all runs
accepted four missions with identical direct/shared digest per seed, with
mean wall time `0.018192552 s` direct versus `0.018280509 s` shared
(`+0.483474%`). A separate fair three-run synthetic Mission-audit
initialization benchmark (fresh process, with no pre-created route planner)
measured mean `0.046008488 s` direct versus `0.038229823 s` shared
(`-16.907022%`). The artifacts are
`/tmp/xmflight_python_binding_c5_mission_benchmark.json` and
`/tmp/xmflight_python_binding_c5_audit_benchmark.json`.

The independent current O/P native collision comparator covered 121 exact
cases and reported zero max absolute and relative delta, with shared-owner
lifetime PASS. The current OMP fixture best values were O `454909.382171`
and P `495683.424206` poses x actions/sec; this is a separate C++ backend
measurement, not substituted for the paired Python context benchmark.

The final parity artifacts are:

| Contract | Result/evidence |
|---|---|
| Collision | 121 numeric cases exact; production asset and shared-owner parity PASS |
| Global Route | 127 synthetic and 1000 production-like cases exact; no workspace leak |
| Mission accept/order | 1000 candidate rows exact |
| Teacher action/route | 100 rows exact |
| Depth Safety | synthetic, 1000 production-like, generator, policy, and Teacher masks PASS |
| C++ VoxelMap fixture | 1000 random + 14 boundary points; metadata, occupancy, coordinate, and lifetime PASS |

Machine-readable artifacts:

```text
/tmp/xmflight_python_binding_c5_context_benchmark_final.json
/tmp/xmflight_python_binding_c5_mission_benchmark.json
/tmp/xmflight_python_binding_c5_collision_final.json
/tmp/xmflight_python_binding_c5_route_final/c3_global_route_parity_summary.json
/tmp/xmflight_python_binding_c5_mission_final/
/tmp/xmflight_python_binding_c5_depth_final/depth_safety_parity.json
```

The three-run depth result remained P `270290.229650` versus O
`259294.787743` actions/sec (`+4.240518%`), with equal maximum RSS
`150976 KB`; C5 did not modify Depth Safety. The current O/P route benchmark
reported P `2167.969470` versus O `47.547953` routes/sec. These are parity and
regression evidence only; no CUDA collision parity is claimed.

## Fail-closed and fallback gate

The focused tests cover missing library/symbol, missing cache, SHA mismatch,
cache metadata/dtype/shape mismatch, closed context/consumer, and forked
handle reuse. Production native initialization/execution does not fall back to
Python collision, Python A*, or Python Depth Safety. The Python reference
implementations remain explicit test/debug choices.

```text
PYTHON_SHARED_VOXELMAP_OWNERSHIP=PASS
PYTHON_VOXELMAP_LOAD_COUNT_PER_PROCESS=1 (context and audited production workers)
PYTHON_OCCUPANCY_BUFFER_COUNT_PER_PROCESS=1 (context and audited production workers)
NATIVE_FAIL_CLOSED=PASS
PYTHON_PRODUCTION_FALLBACK_COUNT=0
```

## Build and installation boundary

The P native targets `planning_voxel_map`, `planning_collision_checker`,
`planning_global_route`, `planning_depth_safety`, and the VoxelMap parity
fixture build successfully from the existing Release build tree. CMake keeps
the absent generated `data/motion_primitives` install directory conditional;
no formal data was generated or copied.

`PYTHONPYCACHEPREFIX=/tmp/xmflight_c5_pycache python -m compileall` passed.
The source-tree and installed-system-Python imports passed, including a real
installed `NativeGeometryContext` creating and closing a native map. The
conda catkin install remains blocked by its pre-existing setuptools error:
`option --install-layout not recognized`. A system Python isolated install
completed successfully under `/tmp/xmflight_c5_system_install`.

The full P build remains:

```text
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
```

The unchanged Bridge sources still reference undeclared `ROS_WARN`,
`ROS_WARN_THROTTLE`, and `ROS_ERROR_THROTTLE`. This C5 phase neither repairs
nor hides that blocker.

## Files and non-actions

Modified P files are the new `python/planning/native` owner, the stable
collision/route adapters and their Mission/Teacher wiring, the shared-owner
focused test and benchmark, and the three C5 documents. No P C++ algorithm
source was changed. No files were deleted. O and Unity were not modified, no
formal data was collected, and no commit was created.

The remaining non-C5 items are the conda/catkin install compatibility issue,
the Bridge full-build blocker, lack of pybind11/200-line wrapper reduction,
the one-time strict-identity initialization cost, and the separate depth
strict-copy audit. The next authorized phase is C6 Bridge protocol transport;
it was not executed here.

```text
ALGORITHM_CHANGED=NO
NEXT_PHASE=C6 BRIDGE PROTOCOL TRANSPORT
```

## Native library path resolution finalization V1 — 2026-09-04

The current devel-space runtime has one native shared-library path owner:
`planning.native.loader`.  The normal environment is:

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
```

No native-library export is required after this setup.  With no explicit
override, the resolver derives the workspace from the Planning package root
and uses only:

```text
/home/xm/XM/xm_ws/devel/lib/libplanning_voxel_map.so
/home/xm/XM/xm_ws/devel/lib/libplanning_collision_checker.so
/home/xm/XM/xm_ws/devel/lib/libplanning_global_route.so
/home/xm/XM/xm_ws/devel/lib/libplanning_depth_safety.so
```

The formal precedence is explicit override first, then the project-derived
`devel/lib` path.  A missing explicit override fails fast and is never
replaced by the devel path.  There is no install-tree fallback, package-local
fallback, filesystem search, or `ctypes.util.find_library` discovery.  The
four formal path variables remain supported only as optional advanced
overrides; `PLANNING_DEPTH_SAFETY_TEST_LIBRARY` is likewise optional for an
isolated depth test.

```text
NATIVE_LIBRARY_PATH_OWNER=planning.native.loader
NATIVE_LIBRARY_PATH_OWNER_COUNT=1
WORKSPACE_AUTO_DETECTION=PASS
INSTALL_TREE_NATIVE_FALLBACK=NO
ENV_OVERRIDE_SUPPORTED=YES
ENV_OVERRIDE_REQUIRED=NO
DEPTH_SAFETY_TEST_ENV_REQUIRED=NO
```

The focused path-resolution, native geometry, collision, route, and depth
tests passed with all five path variables unset.  Full evidence is recorded
in [`NATIVE_LIBRARY_PATH_RESOLUTION_FINALIZATION_V1.md`](NATIVE_LIBRARY_PATH_RESOLUTION_FINALIZATION_V1.md)
and its machine-readable companion.
