# Native Library Path Resolution Finalization V1

Date: 2026-09-04  
Project root: `/home/xm/XM/xm_ws/src/planning`  
Workspace: `/home/xm/XM/xm_ws`

## Result

```text
NATIVE_LIBRARY_PATH_RESOLUTION_FINALIZATION_V1=FAIL_ENVIRONMENT_BLOCKED
IMPLEMENTATION_AND_NATIVE_FOCUSED_TESTS=PASS
```

The resolver and all native-path focused checks pass. The strict full pytest
gate remains blocked by eight existing socket-dependent tests that fail at
socket creation with `PermissionError: [Errno 1] Operation not permitted` in
the managed execution environment. They fail before their behavioral
assertions. No test was weakened and no fallback was added.

## Canonical owner

```text
CANONICAL_NATIVE_PATH_OWNER=planning.native.loader
NATIVE_LIBRARY_PATH_OWNER_COUNT=1
DIRECT_NATIVE_PATH_RESOLVER_DUPLICATE_COUNT=0
```

The owner exposes typed specifications and public resolver seams for
`voxel_map`, `collision`, `global_route`, and `depth_safety`. Geometry,
collision, route, depth safety, Mission/Teacher consumers, and tests no longer
implement independent native path discovery. `planning.common.paths` remains
the owner of project/workspace root detection only; it no longer owns native
library candidate lists.

## Formal artifacts and precedence

The only automatic candidates are the four files below under the workspace
derived from the Planning package root:

```text
/home/xm/XM/xm_ws/devel/lib/libplanning_voxel_map.so
/home/xm/XM/xm_ws/devel/lib/libplanning_collision_checker.so
/home/xm/XM/xm_ws/devel/lib/libplanning_global_route.so
/home/xm/XM/xm_ws/devel/lib/libplanning_depth_safety.so
```

Resolution precedence is deterministic:

1. A non-empty explicit override is used first. A path override must be an
   existing file; a missing override raises the component-specific
   `*_NATIVE_LIBRARY_MISSING` error and never falls back.
2. With no override, the project-derived workspace `devel/lib` path is used.
3. Install trees, package-local libraries, arbitrary filesystem searches, and
   `ctypes.util.find_library` are not part of the formal resolver.

The error includes `library_kind`, `expected_path`, `workspace`, and the
instruction `build workspace with catkin_make`. The resolver does not build,
silently skip, or switch to a Python backend.

The supported advanced overrides are:

```text
PLANNING_VOXEL_MAP_LIBRARY
PLANNING_COLLISION_LIBRARY
PLANNING_GLOBAL_ROUTE_LIBRARY
PLANNING_DEPTH_SAFETY_LIB
```

`PLANNING_DEPTH_SAFETY_TEST_LIBRARY` remains an optional test-only override.

```text
ENV_OVERRIDE_SUPPORTED=YES
ENV_OVERRIDE_REQUIRED=NO
DEPTH_SAFETY_TEST_ENV_REQUIRED=NO
INSTALL_TREE_NATIVE_FALLBACK=NO
```

## Verification matrix

All Python commands were run after `conda activate xm`, with ROS and the
current devel space sourced. All five path variables were unset for the
automatic-resolution checks.

```text
WORKSPACE_AUTO_DETECTION=PASS
AUTO_RESOLVE_VOXEL_MAP=PASS
AUTO_RESOLVE_COLLISION=PASS
AUTO_RESOLVE_GLOBAL_ROUTE=PASS
AUTO_RESOLVE_DEPTH_SAFETY=PASS
```

Override cases:

```text
A unset override -> project-derived devel/lib=PASS
B existing explicit override -> override path=PASS
C missing explicit override -> fail-fast without fallback=PASS
D competing workspace -> current project-derived workspace=PASS
E install exists while devel is missing -> no install fallback=PASS
```

All four current devel libraries were loaded with no path overrides. The
formal CLI help checks also passed without path exports:

```text
prepare_teacher_missions.py=PASS
collect_rollouts_parallel.py=PASS
label_teacher_rollouts.py=PASS
generate_depth_action_masks.py=PASS
train_awac.py=PASS
evaluate_policy_unity.py=PASS
```

Native focused suite:

```text
FOCUSED_NATIVE_TEST_PASS_COUNT=58
FOCUSED_NATIVE_TEST_FAILURE_COUNT=0
VOXEL_MAP_NATIVE_LIBRARY_MISSING_FAILURES=0
DEPTH_SAFETY_NATIVE_UNAVAILABLE_FAILURES=0
COMPILEALL=PASS
```

The unfiltered full suite was run with all path overrides unset:

```text
FULL_TEST_PASS_COUNT=697
FULL_TEST_SKIP_COUNT=42
FULL_TEST_FAILURE_COUNT=8
```

The eight failures are:

```text
tests/test_p0_m2_runner_startup.py (3)
tests/test_p0_m2_single_worker_runner.py (5)
```

Each failed with `PermissionError: [Errno 1] Operation not permitted` from
`socket.socket(...)`. The code-eligible result, excluding only these
environment-blocked modules, is `697 passed, 42 skipped, 0 failed` for this
run.

## Scope and changes

Changed files are limited to:

```text
python/planning/native/loader.py
python/planning/common/paths.py
python/planning/native/geometry.py
python/planning/native/collision_backend.py
python/planning/native/route_backend.py
python/planning/safety/collision_checker.py
python/planning/mission/global_route.py
python/planning/safety/depth_safety.py
tests/test_native_library_path_resolution.py
tests/test_python_native_geometry_context.py
tests/test_depth_safety_production_cutover.py
docs/PYTHON_NATIVE_BINDING.md
docs/FORMAL_TEACHER_TO_BC_END_TO_END_COMMANDS_V3.md
docs/NATIVE_LIBRARY_PATH_RESOLUTION_FINALIZATION_V1.md
docs/native_library_path_resolution_finalization_v1.json
docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md
```

No C++ source, native ABI, Unity, Bridge, AWAC, BC, Teacher, MPL, task,
observation, reward, or trajectory semantics changed. No install build,
formal collection, training, or commit was performed.

## Next action

Rerun the unfiltered full pytest in an environment that permits loopback
socket creation. After that gate is closed, certify the already implemented
RL Phase 0 before entering RL Phase 1.
