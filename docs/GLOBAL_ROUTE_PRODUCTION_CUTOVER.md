# Global Route Production Cutover

## C3.1 result

Date: 2026-08-27

```text
C3_1_GLOBAL_ASTAR_PRODUCTION_CUTOVER=PASS
FORMAL_GLOBAL_ROUTE_BACKEND=C++17_NATIVE
PRODUCTION_ROUTE_BACKEND_UNIQUE=YES
PYTHON_PRODUCTION_FALLBACK=NO
NATIVE_ROUTE_FAIL_CLOSED=PASS
GLOBAL_ROUTE_PARITY=PASS
CORNER_CUTTING_PARITY=PASS
TIE_ROUTE_SEQUENCE_PARITY=PASS
MISSION_ROUTE_ACCEPT_VECTOR_PARITY=PASS
MISSION_ACCEPTED_ID_ORDER_PARITY=PASS
TEACHER_ROUTE_INPUT_PARITY=PASS
TEACHER_ACTION_PARITY=PASS
TEACHER_VALID_ACTION_MASK_PARITY=PASS
COLLISION_REGRESSION=PASS
P_GLOBAL_ROUTE_TARGET_BUILD=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
P_CPP_ROUTE_THROUGHPUT=2150.730608 routes/sec
O_PYTHON_ROUTE_THROUGHPUT=47.492813 routes/sec
ROUTE_SPEEDUP=45.285391x
CXX_SHARED_VOXELMAP=PASS
PYTHON_SHARED_VOXELMAP_BINDING=NOT_STARTED
ROUTE_STORE=NOT_STARTED
ALGORITHM_CHANGED=NO
NEXT_PHASE=C4 DEPTH SAFETY PARITY AND OWNERSHIP
```

This was a P-only production behavior change. O at
`/home/xm/XM/xm_ws/src/planning` remained read-only. Unity, Bridge/protocol,
Mission and Teacher algorithms, BC/AWAC math, and formal data collection were
not changed or run. No commit was made.

## Backend contract

The formal route owner is the P `planning_global_route` C++17 target, reached
through `python/planning/mission/global_route.py`. Production aliases resolve
to the single native backend:

| Requested backend | Resolution |
| --- | --- |
| unset, `cpp`, `native`, `auto`, `cpp_native` | `cpp_native` |
| `python` without `PLANNING_GLOBAL_ROUTE_REFERENCE=1` | reject |
| `python` with explicit reference opt-in | Python reference, test/debug only |
| unknown value | `UNSUPPORTED_GLOBAL_ROUTE_BACKEND` |

An explicit missing native library, missing symbol, initialization failure, or
execution failure raises an explicit failure. The wrapper never changes to the
Python reference after a native failure. Native return code 1 remains the
normal no-path business result; other native failures are fail-closed.

The formal route callers remain one route call per stage and keep their
existing call sites:

```text
Generate:   1
Audit:      1
Collection: 1
Relabel:    1
```

There is no RouteStore or cross-stage route deduplication in C3.1. The Python
reference remains available only through the explicit test/debug seam and the
O/P parity fixtures.

## Fail-closed tests

`tests/test_global_route_production_cutover.py` covers:

```text
A native library exists -> planner creation succeeds
B missing native library -> explicit failure
C missing native symbol -> explicit failure
D python without explicit test/debug seam -> reject
E auto with native missing -> explicit failure, no Python fallback
F no-path -> normal GlobalRouteUnavailableError business result
native execution failure -> explicit failure, no Python fallback
```

The focused cutover, empty-map, and native architecture tests pass: 16 tests.

## Parity and regression evidence

Route parity was rerun against the fixed O/P fixtures:

```text
127 synthetic cases: PASS
1000 production-like cases: PASS
mission route acceptance/order: PASS (1000/1000)
Teacher route inputs/actions/masks: PASS (100/100, 105 actions)
```

The C2 collision gates were rerun after the native route cutover changes:

```text
numeric collision parity: PASS (121/121, max abs delta 0, max relative delta 0)
candidate accept vector parity: PASS (100/100)
Teacher valid action mask parity: PASS (100/100, 105 actions)
shared VoxelMap runtime ownership: PASS (one map load, zero complete consumer copies)
```

The three-run size-1000 route benchmark is recorded in
`/tmp/xmflight_global_route_c3/global_route_cutover_benchmark.json`:

| Measure | O Python | P C++17 native |
| --- | ---: | ---: |
| best throughput (routes/s) | 47.492813 | 2150.730608 |
| mean initialization (s) | 0.059457 | 0.065384 |
| p50 latency (ms) | 19.253185 | 0.406781 |
| p95 latency (ms) | 40.343377 | 0.944529 |
| peak RSS (KB) | 379520 | 380528 |
| map loads / planner instances | 1 / 1 | 1 / 1 |

P is 45.285391x the O best throughput, exceeding the C3.1 10x threshold.

## Provenance and installation

Formal metadata now records `global_route_backend=cpp_native`, the route
contract identity, the native source identity/list, the resolved planner
configuration, and the VoxelMap/cache identity where the stage already emits
metadata. The Mission CSV business schema is unchanged. Collection manifests,
mission generation/audit metadata, relabel metadata, and rollout episode
metadata retain their existing fields and add route provenance.

`planning_global_route` is explicitly listed in P's CMake source/target and
install path, with C++17 and the existing VoxelMap dependency. `package.xml`
exports the native backend, contract, library, and source identity. The route
target builds successfully. The complete P package remains blocked only by
pre-existing Bridge/protocol `ROS_WARN`, `ROS_WARN_THROTTLE`, and
`ROS_ERROR_THROTTLE` declarations; Bridge was not repaired in this phase.

Formal command shape is native by default and does not set a Python backend,
for example:

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source <catkin-workspace>/devel/setup.bash
rosrun planning generate_missions.py --collision-cache <cache> --out-index <index>
rosrun planning audit_teacher_missions.py --candidate-index <index> --out-index <audit>
```

These are command-shape records only; no formal missions, labels, rollouts,
or training data were generated during C3.1. The source-only CMake setup
continues to skip absent generated `data/motion_primitives` inputs rather than
fabricating them.

## Files and remaining blockers

P files modified or added for this cutover are:

```text
CMakeLists.txt
package.xml
python/planning/mission/global_route.py
python/planning/mission/generator.py
python/planning/mission/auditor.py
python/planning/contracts/collection.py
python/planning/teacher/labeling.py
python/planning/teacher/rollout_collector.py
tests/test_global_route_production_cutover.py
tests/test_global_route_empty_map.py
tests/test_native_compute_architecture.py
tests/contracts/test_teacher_collection_contract.py
docs/GLOBAL_ROUTE_PARITY.md
docs/CXX_FOUNDATION_MIGRATION.md
docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md
docs/GLOBAL_ROUTE_PRODUCTION_CUTOVER.md
```

The key contract seam is `python/planning/mission/global_route.py`; the
focused fail-closed tests are in `tests/test_global_route_production_cutover.py`.
The existing C++ VoxelMap and route implementation remain the built native
owners; no C++ algorithm rewrite was needed for C3.1. No files were deleted.

Remaining blockers are deliberately outside C3.1:

- full P build closure requires a separately authorized Bridge/protocol repair;
- Python shared-VoxelMap binding remains `NOT_STARTED` and belongs to C5;
- CUDA collision runtime validation remains `DEFERRED / NOT_AVAILABLE` and no
  CUDA parity claim is made;
- C4 depth-safety parity and ownership was not started.

`NEXT_PHASE=C4 DEPTH SAFETY PARITY AND OWNERSHIP`
