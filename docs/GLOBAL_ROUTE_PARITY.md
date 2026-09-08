# C3 Global A* Semantics and Parity

Date: 2026-08-27

## Decision

```text
C3_GLOBAL_ASTAR_PARITY=PASS
STRUCTURE_CHANGED=YES
ROUTE_BEHAVIOR_CHANGED=NO
P_CPP_GLOBAL_ROUTE_PRODUCTION_CANDIDATE=YES
NEXT_PHASE=C3.1 GLOBAL A* PRODUCTION CUTOVER
```

This is a controlled comparison, not a production cutover. The original tree
is the behavior and algorithm source of truth:

```text
O=/home/xm/XM/xm_ws/src/planning
P=/home/xm/XM/src
```

O was read-only. No formal data was generated, collected, or copied. No
RouteStore, cross-stage A* deduplication, Mission/Teacher algorithm change,
Bridge change, or commit was made. The environment for the executed Python
commands was `conda activate xm` with `PYTHONPATH` set to the selected tree.

## Owner and call graph

O owns the public Python implementation at
`python/planning/global_route.py` (`GlobalRoutePlanner2D`). Its route callers
are mission route validation/generation, mission audit, Teacher setup,
teacher rollout collection, and Teacher relabeling.

P keeps the O implementation as an explicit Python reference backend and adds
the candidate native path:

```text
python/planning/mission/global_route.py  ->  ctypes C ABI
src/navigation/global_route_planner.cpp ->  libplanning_global_route.so
```

Only `python/planning/mission/global_route.py` calls the native C ABI. No
business caller calls ctypes directly. The native planner receives a
`std::shared_ptr<const VoxelMap>` and keeps no complete duplicate of the
three-dimensional occupied-key array; its derived two-dimensional blocked
grid is the route representation required by the O algorithm.

The static call count per mission is:

```text
Generate:   1 A* plan
Audit:       1 A* plan
Collection:  1 A* plan; prefetch set_mission_route reuses that route
Relabel:     1 A* plan
```

There is no cross-stage route cache in this phase. In the route-only native
benchmark, one immutable map load and one planner instance were observed per
fresh process. The combined Python Mission/Teacher comparator intentionally
still has separate route and collision Python owners; unifying that binding
is deferred to C5.

## O/P semantic matrix

| Contract dimension | O reference behavior | P candidate behavior | Classification | Verdict |
| --- | --- | --- | --- | --- |
| Contract/defaults | `coarse_xy_astar_025m_lookahead3m_flight_band`; resolution 0.25 m; flight band 1–3 m; lookahead 3 m; margin 0; nearest-free radius 6 | Same public defaults and explicit config | STRUCTURAL_ONLY | SAFE_CANDIDATE |
| XY origin | `floor(lower_xy / resolution) * resolution` | Same floor-based origin | EXACT | SAFE_CANDIDATE |
| XY shape | `ceil((upper_xy - origin_xy) / resolution)` | Same ceil-based shape | EXACT | SAFE_CANDIDATE |
| Occupancy projection | Project occupied voxels whose z center is in the configured flight band expanded by collision radius | Same z-band projection | EXACT | SAFE_CANDIDATE |
| Inflation | Circular coarse-cell dilation with `ceil((collision_radius + tracking_margin) / resolution)` | Same radius and dilation | EXACT | SAFE_CANDIDATE |
| World to cell | `floor((xy - origin_xy) / resolution)` | Same floor conversion and bounds behavior | EXACT | SAFE_CANDIDATE |
| Cell to world | `origin_xy + (float32(cell) + 0.5) * resolution` | Same cell-center conversion | EXACT | SAFE_CANDIDATE |
| Nearest free | dx outer loop, dy inner loop, strict radius, first scan-order tie winner | Same scan order, strict radius, and tie behavior | EXACT | SAFE_CANDIDATE |
| Neighbor order | axial `(+x,-x,+y,-y)`, then diagonals `(+,+),(+,-),(-,+),(-,-)` | Same order | EXACT | SAFE_CANDIDATE |
| Diagonal corner guard | A diagonal is rejected when either orthogonal side cell is blocked | Same guard; runtime O corner fixture is `no_route` | EXACT | SAFE_CANDIDATE |
| Queue ordering | `(priority, cell_id)` min-heap; goal closes on first pop | Same priority then cell-id ordering | EXACT | SAFE_CANDIDATE |
| Relaxation | strict candidate `<` existing cost; equal-cost parent is not replaced | Same strict comparison | EXACT | SAFE_CANDIDATE |
| Search/reconstruction | closed-on-pop A*, reverse parent chain, then forward route | Same search/reconstruction | EXACT | SAFE_CANDIDATE |
| Route output | actual start, cell-center path including start/goal cells, actual goal; `float32[point,3]` | Same row order, shape, and API output | EXACT | SAFE_CANDIDATE |
| Intermediate z | `np.linspace(float(start_z), float(goal_z), n, dtype=float32)` | Native XY route plus wrapper seam restoring the O scalar-z/float32 sequence | STRUCTURAL_ONLY | SAFE_CANDIDATE |
| No route | `GlobalRouteUnavailableError` with no-route contract | Same Python exception contract from native status | EXACT | SAFE_CANDIDATE |
| Empty route map | O route planner accepts an empty occupied-key set | P VoxelMap now permits zero occupied keys for route ownership; collision creation remains fail-closed for empty maps | STRUCTURAL_ONLY | NEEDS_PARITY_TEST |

The empty-map adjustment is a narrow P-only contract repair required by the
route fixture. It does not alter the collision backend's existing empty-map
rejection. The intermediate-z wrapper adjustment fixes the only initial
production-like byte mismatch; XY cells, A* ordering, costs, and route point
selection were unchanged.

The tie fixture `tie-000` produced the same grid sequence
`[[0,0],[1,1],[2,1]]` and cost `2.414213562373095` in both trees. The
corner fixture is intentionally recorded from O runtime: the two blocked
orthogonal neighbors make the diagonal unavailable, so both trees return
`no_route`. This is not an assumption that the candidate may change the O
corner policy.

## Fixed parity fixtures

The comparator runs O and P in separate subprocesses and observes O outputs;
it does not reimplement A* as an oracle.

### Synthetic

`tests/compare_global_route_parity.py` covers 127 deterministic cases:

```text
empty=15                 single_obstacle=10       narrow_corridor=10
diagonal_corridor=10     corner_cutting=1         blocked_start=4
blocked_goal=4           no_path=8                tie=10
boundary=10              long_route=5             dense_obstacle=20
sparse_extra=20
```

Route availability, point count, grid sequence, world sequence, cost,
metadata, and conversion outputs are exact. O/P workspace state-leak lists
are empty.

### Production-like

The same comparator covers 1000 fixed pairs using the existing cache
`/tmp/xmflight_collision_c2/forest_voxels_10cm.npz` and the first 1000 rows
of the historical candidate fixture
`/home/xm/XM/xm_ws/src/planning/data_back_20260826/teach/flight_20260717/mission_candidates.csv`.
The generated pair fixture is temporary under `/tmp/xmflight_global_route_c3/`
and is not formal training or mission data.

```text
production_candidate=100   short=180       medium=180
long=180                   blocked=120     near_obstacle=120
boundary=120
```

The route-grid fixture has shape `[820, 818]`, resolution `0.25`, origin
`[-102.5, -102.0]`, blocked count `182118`, and blocked-grid SHA256
`bf8d788ea35e64d9139227a50e9ae06396304d06ccc140d865d7dde6468d3d93`.
All route and metadata gates passed; workspace state leak is `0`.

Comparator result:

```text
SYNTHETIC_CASES=127
SYNTHETIC_ROUTE_PARITY=PASS
SYNTHETIC_METADATA_PARITY=PASS
SYNTHETIC_WORKSPACE_STATE_LEAK=0
PRODUCTION_CASES=1000
PRODUCTION_ROUTE_PARITY=PASS
PRODUCTION_METADATA_PARITY=PASS
PRODUCTION_WORKSPACE_STATE_LEAK=0
RESULT=PASS
```

## Mission and Teacher integration

`tests/compare_global_route_mission_teacher.py` exercised the same 1000
candidate rows and the same cache in separate O/P runners. Route validation
used the route length/stretch acceptance rule and compared the accepted
mission IDs in order. The first 100 route-bearing missions were then injected
into Teacher through `set_mission_route`; Teacher configuration and scoring
code were not changed.

```text
MISSION_COUNT=1000
MISSION_ROUTE_ACCEPT_VECTOR_PARITY=PASS
MISSION_ACCEPTED_ID_ORDER_PARITY=PASS
TEACHER_COUNT=100
TEACHER_ROUTE_INPUT_PARITY=PASS
TEACHER_ACTION_PARITY=PASS
TEACHER_VALID_ACTION_MASK_PARITY=PASS (100/100, 105 actions)
RESULT=PASS
```

The detailed artifact also reports exact route-found, route-length,
route-stretch, and route-validation parity. The temporary MPL and metadata
fixtures are under `/tmp/xmflight_global_route_c3/`.

## Build and API evidence

P target `planning_global_route` built successfully in the controlled P
Release build. The exported C ABI was present for create/destroy, shape,
blocked-grid copy, plan, point count, and route copy. O has no C++ global-route
target; its route owner is Python.

The P full `all` build remains blocked by the pre-existing Bridge/protocol
compile errors for undeclared `ROS_WARN`, `ROS_WARN_THROTTLE`, and
`ROS_ERROR_THROTTLE` in `src/bridge/result_gateway.cpp` and
`src/bridge/snapshot_gateway.cpp`. Bridge repair is outside C3 and was not
attempted. This does not invalidate the independently built route target, but
it blocks whole-package build closure.

## Performance evidence

`tests/benchmark_global_route.py` ran three fresh-process repetitions for each
size and backend against fixed pairs. Route throughput is the route batch
throughput; initialization reports the cache/planner initialization path.

| Fixture size | O best routes/s | P best routes/s | P/O | P speedup |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 49.001299 | 2208.455789 | 45.069x | +4406.93% |
| 1000 | 47.200157 | 2152.737649 | 45.609x | +4460.87% |

For the 1000-case run, mean initialization was O `0.059851 s` and P
`0.125439 s`; peak RSS was O `379456 KB` and P `380592 KB`. Both reported one
map load and one planner instance. The source-derived planner workspace
estimate is O `6036840` bytes per plan and P `11402920` bytes persistent;
this is an explicit memory tradeoff for avoiding repeated route workspace
allocation and does not change route output. A 10000-case run was not needed
for the gate and was not executed.

## Files and artifacts

P-side C3 implementation/validation files:

- `python/planning/mission/global_route.py`
- `include/planning/navigation/global_route_planner.hpp`
- `src/navigation/global_route_planner.cpp`
- `src/geometry/voxel_map.cpp` (zero-occupancy route fixture seam)
- `src/geometry/collision_checker.cpp` (preserve collision empty-map fail-closed behavior)
- `tests/test_global_route_empty_map.py`
- `tests/compare_global_route_parity.py`
- `tests/compare_global_route_mission_teacher.py`
- `tests/benchmark_global_route.py`
- `docs/GLOBAL_ROUTE_PARITY.md`
- this C3 section in `docs/CXX_FOUNDATION_MIGRATION.md` and
  `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`

Fixed evidence artifacts:

- `/tmp/xmflight_global_route_c3/c3_global_route_parity_summary.json`
- `/tmp/xmflight_global_route_c3/c3_mission_teacher_parity.json`
- `/tmp/xmflight_global_route_c3/global_route_benchmark.json`
- `/tmp/xmflight_global_route_c3/c3_mpl.npz` and `c3_mpl.json` for Teacher consumption only

## Remaining blockers and handoff

The route-specific candidate gate is closed as YES, but production cutover is
not executed. Before C3.1, retain the Bridge full-build blocker and the
deferred Python shared-VoxelMap binding. Do not infer whole-package readiness
from the route target alone. CUDA collision runtime was not part of C3 and no
CUDA parity claim is made.

`NEXT_PHASE=C3.1 GLOBAL A* PRODUCTION CUTOVER`

## C3.1 — Global A* production cutover (2026-08-27)

The explicitly authorized P-only cutover is complete. O remained read-only,
and the formal route backend is now the C++17 native target:

```text
C3_1_GLOBAL_ASTAR_PRODUCTION_CUTOVER=PASS
FORMAL_GLOBAL_ROUTE_BACKEND=cpp_native
PRODUCTION_ROUTE_BACKEND_UNIQUE=YES
PYTHON_PRODUCTION_FALLBACK=NO
NATIVE_ROUTE_FAIL_CLOSED=PASS
P_GLOBAL_ROUTE_TARGET_BUILD=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
ALGORITHM_CHANGED=NO
```

Unset, `cpp`, `native`, `auto`, and `cpp_native` all resolve to native. The
Python reference is rejected unless the explicit
`PLANNING_GLOBAL_ROUTE_REFERENCE=1` test/debug seam is present. Missing native
library/symbol, initialization failure, and execution failure are explicit
failures; no Python fallback is attempted. Native no-path remains the normal
`GlobalRouteUnavailableError` result.

The formal Generate, Audit, Collection, and Relabel callers still make one
route call per stage. No RouteStore or cross-stage deduplication was added.
Route identity, native source identity, resolved planner configuration, and
VoxelMap/cache identity are recorded in the existing metadata/manifests; no
Mission CSV business field was removed or changed.

The cutover gates are exact: 127 synthetic and 1000 production-like route
fixtures, 1000 mission acceptance/order rows, 100 Teacher route/action/mask
rows with 105 actions, and the C2 collision rerun (121 numeric cases, 100
candidate rows, 100 Teacher masks). The size-1000 three-run benchmark is
`P=2150.730608 routes/s`, `O=47.492813 routes/s`, `45.285391x`; P init mean is
`0.065384 s` versus O `0.059457 s`, and P/O peak RSS is `380528/379520 KB`.
The detailed ledger is `docs/GLOBAL_ROUTE_PRODUCTION_CUTOVER.md` and the
machine-readable benchmark is
`/tmp/xmflight_global_route_c3/global_route_cutover_benchmark.json`.

The independently built route target and shared C++ VoxelMap ownership pass.
The full P build is still blocked by pre-existing Bridge/protocol `ROS_WARN*`
declarations. Python shared-VoxelMap binding remains `NOT_STARTED`; CUDA
collision runtime remains `DEFERRED / NOT_AVAILABLE`. C4 was not executed.

`NEXT_PHASE=C4 DEPTH SAFETY PARITY AND OWNERSHIP`
