# Collision production cutover

Date: 2026-08-27

This is the P-only C2.2 cutover record.  The original tree
(`/home/xm/XM/xm_ws/src/planning`) is read-only and remains the historical
behavior baseline.  The explicit authorization was limited to the self-owned
collision backend; PyTorch CUDA/AMP/GradScaler used by BC/AWAC was not part of
this change.

## Formal production contract

```text
FORMAL_COLLISION_BACKEND=cpp_cpu
FORMAL_COLLISION_BACKEND_DESCRIPTION=C++17_OPENMP_CPU
COLLISION_BACKEND_CONTRACT_ID=cpp_bitmap_openmp
COLLISION_SOURCE_ID=planning_collision_checker_cpp17_openmp
PRODUCTION_BACKEND_UNIQUE=YES
PYTHON_PRODUCTION_FALLBACK=NO
```

The stable Python owner remains
`planning.safety.collision_checker.VoxelCollisionChecker`.  Existing public
method shapes and result arrays are unchanged.  `cpu`, `cpp`, and `auto` are
compatibility spellings that all resolve to `cpp_cpu`; the public backend
diagnostic value remains `cpp` for API compatibility.  Python reference mode
requires both an explicit `PLANNING_COLLISION_BACKEND=python` selection and
`PLANNING_COLLISION_REFERENCE=1`.

Native library initialization failures and native operation failures raise a
runtime error.  They do not clear the native backend or continue through the
NumPy implementation.  `cuda` and `hybrid` requests fail with
`UNSUPPORTED_COLLISION_BACKEND`.

## Production callers and source identity

All formal collision consumers use the same owner:

- `planning.mission.sampling`
- `planning.mission.generator`
- `planning.mission.route_validation`
- `planning.mission.auditor` and `audit_worker`
- `planning.teacher.label_worker` and `rollout_collector`
- `planning.runtime.unity_env`

The depth-only worker is not a collision consumer and was intentionally left
unchanged in C2.2.

The native source manifest is:

```text
include/planning/geometry/voxel_map.hpp
src/geometry/voxel_map.cpp
include/planning/geometry/collision_checker.hpp
src/geometry/collision_checker.cpp
python/planning/safety/collision_checker.py  # stable binding + explicit reference
```

P CMake requires C++17 and OpenMP and builds/installs
`planning_voxel_map` and `planning_collision_checker` for this cutover.  No
collision CUDA language, NVCC flags, CUDA target, CUDA install target, or
hybrid scheduler is in P's formal path.

## Removal ledger

| P item | Old role | Current callers | Replacement | Evidence/tests | Removal reason |
| --- | --- | --- | --- | --- | --- |
| `src/geometry/collision_checker_cuda.cu` | CUDA collision source | None found in P | `src/geometry/collision_checker.cpp` | P source scan, CMake scan, numeric parity | Pre-existing absence; no P file deletion in C2.2 |
| `planning_collision_checker_cuda` CMake target | CUDA collision build/install target | None found in P | `planning_collision_checker` | fresh P CMake configure/build and source contract tests | Pre-existing absence; `OUT_OF_FORMAL_SCOPE`, not proven incorrect |
| P collision `cuda`/`hybrid` CLI path | alternate collision selection | None found in formal P CLI | `cpp_cpu` with `cpu/cpp/auto` aliases | backend resolver tests and CLI type rejection | No formal path existed; unsupported requests now fail-fast |
| `PLANNING_COLLISION_CUDA_LIBRARY` in P collision binding | CUDA library lookup | None in formal P binding | `libplanning_collision_checker.so` discovery | source scan and native target build | No formal path existed; O CUDA comparator reference is test-only |
| native-exception NumPy auto fallback in `collision_checker.py` | silent runtime downgrade | Mission/Teacher/safety consumers could reach it | fail-closed native operation error | missing-library and forced-runtime-failure tests | Production behavior change authorized by C2.2 |

No P files were deleted in C2.2.  O CUDA source and O hybrid behavior remain
untouched.  The O CUDA comparator references and generic PyTorch CUDA code for
BC/AWAC are test/learner concerns, not collision production paths, and remain
present.

## P files modified in C2.2

- `CMakeLists.txt`
- `package.xml`
- `python/planning/safety/collision_checker.py`
- `python/planning/mission/auditor.py`
- `python/planning/teacher/labeling.py`
- `tests/compare_collision_teacher_mask_fixture.py`
- `tests/test_collision_production_backend.py`
- `docs/COLLISION_PRODUCTION_CUTOVER.md`
- `docs/COLLISION_BACKEND_PARITY.md`
- `docs/CXX_FOUNDATION_MIGRATION.md`
- `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`

No file under O was modified.

## Verification

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
CUDA_HYBRID_FORMAL_PATHS=0
GPU_RUNTIME_VALIDATION=DEFERRED / NOT_AVAILABLE
ALGORITHM_CHANGED=NO
```

The fixed C++17 Release benchmark used OMP 1/2/4 and three runs per setting:

```text
P_KERNEL_THROUGHPUT=531508 poses x actions/sec
O_KERNEL_THROUGHPUT=516287 poses x actions/sec
KERNEL_DELTA_PERCENT=+2.95%
P_END_TO_END_THROUGHPUT=9204120 candidates/sec
O_END_TO_END_THROUGHPUT=7835947 candidates/sec
END_TO_END_DELTA_PERCENT=+17.46%
```

The C2.2 artifacts are under `/tmp/xmflight_collision_c2/`.  The fresh P
collision build was configured under `/tmp/xm-cxx-c2-2-p-build` with devel
output under `/tmp/xm-cxx-c2-2-p-devel/planning`.  `perf` was unavailable, so
hardware-counter parity is not claimed.  The final comparator emitted
`FIXTURE_RESULT_SHA256=e1cc853c91c07501b93a8f32e1a0720d5272272ae02b73de54a99f98089d6d8f`;
the current benchmark artifact SHA256 is
`250cba9a17117c3d3a862d74ac25f8cc6b2912cc436f8153af421953490c6199`.

## Boundaries and next phase

The full P build remains blocked only after the collision targets, at the
pre-existing Bridge/protocol `ROS_WARN*` errors.  Bridge repair is out of
scope.  Python shared VoxelMap binding remains a later C5 task and was not
started.  Global A* semantics/parity is the next phase; no C3 work was
executed in this cutover.

`C2_2_COLLISION_PRODUCTION_CUTOVER=PASS`
`NEXT_PHASE=C3 GLOBAL A* SEMANTICS AND PARITY`
