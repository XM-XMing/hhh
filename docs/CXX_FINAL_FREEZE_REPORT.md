# C7 Final C++ Freeze Report

Date: 2026-08-28
Formal baseline: `/home/xm/XM/xm_ws/src/planning` (read-only)
Formal candidate: `/home/xm/XM/src` (P-only changes)
Environment: `conda activate xm`

## Decision

```text
C7_FINAL_CXX_FREEZE=PASS
CXX_FINAL_V1=PASS
CXX_FROZEN_FOR_PLANNING=YES
ALGORITHM_CHANGED=NO
```

P is frozen as the formal C++ foundation for Planning.  This decision is
limited to the native geometry, shared-map, protocol/transport, Bridge, and
runtime/build contracts below.  It does not freeze or promote the Python
Pre-RL pipeline.

## Unity identity

The frozen Player is:

```text
CANONICAL_UNITY_PLAYER=/home/xm/XM/xm_ws/src/unity/XMflight.x86_64
CANONICAL_UNITY_PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
CANONICAL_UNITY_RUNTIME_ASSEMBLY=/home/xm/XM/xm_ws/src/unity/XMflight_Data/Managed/Assembly-CSharp.dll
CANONICAL_UNITY_RUNTIME_ASSEMBLY_SHA256=7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc
```

The `11f843ba...` hash in an older Unity freeze document is the editor
`Library/ScriptAssemblies` artifact.  It contains editor-only metadata and is
not the Player-loaded assembly.  The actual Player artifact is `7ffb...`.
No Unity file was modified during C7.

## Build, install, and ABI

```text
CLEAN_BUILD=PASS
DEVEL_SPACE_BUILD=PASS
INSTALL_SPACE_BUILD=PASS
P_FULL_BUILD=PASS
PYTHON_NATIVE_IMPORT=PASS
ABI_SYMBOL_GATE=PASS
MISSING_REQUIRED_SYMBOL_COUNT=0
UNEXPECTED_LEGACY_SYMBOL_COUNT=0
STALE_SOURCE_PATH_COUNT=0
FORMAL_COLLISION_IMPLEMENTATION_COUNT=1
FORMAL_ROUTE_IMPLEMENTATION_COUNT=1
FORMAL_DEPTH_SAFETY_IMPLEMENTATION_COUNT=1
FORMAL_BRIDGE_IMPLEMENTATION_COUNT=1
```

Two clean CMake Release builds compiled and linked the same target set.  The
conda environment passed source/devel checks and installed-package native
imports.  Catkin installation with conda Python failed only at the obsolete
`--install-layout=deb` distutils option; the independent system-Python
install-space build passed.  The absent motion-primitives source directory
was handled by the existing guarded CMake install rule; no data was created.

The exact required symbols, source manifest, public ABI hashes, devel/install
artifact hashes, dependencies, and contract hashes are in
[`docs/CXX_ABI_MANIFEST.md`](CXX_ABI_MANIFEST.md).

```text
BRIDGE_BINARY_SHA256=63b00f6a0b31aa72ace3e1c437477e7ee8b670934386f3bb1d645745c302d9ed
VOXEL_LIBRARY_SHA256=745ce457ccbca218e06cf3650ee1f7abffe42036ef90eb57c03a0497a8aaba7f
COLLISION_LIBRARY_SHA256=fa9f82c6916bf4750994dbb707c80d35926804726bc63d293a3b611b8620cb94
GLOBAL_ROUTE_LIBRARY_SHA256=8547cda1c049964861f25e531a94017508993a7e06844087a1bd7c823cb5d5e0
DEPTH_SAFETY_LIBRARY_SHA256=179e165cc2ef0f47232b61c0ccdb4ad1561047aab29893b841f4f8566cf27fab
```

## Correctness and performance

All native numerical and integration gates passed:

```text
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
DEPTH_SAFETY_REGRESSION=PASS
COLLISION_PERFORMANCE_RATIO=0.983331
ROUTE_PERFORMANCE_RATIO=0.991387
DEPTH_SAFETY_PERFORMANCE_RATIO=1.009191
```

Collision `0.983331` is the robust 30-sample fixed-fixture filter ratio
(P/O); corresponding total-wall-time ratio was `0.992835`.  The Route ratio
was measured against the frozen P route baseline, and Depth Safety against
the frozen P depth baseline.  `perf` was unavailable, so cycles, instructions,
IPC, branches, and cache-counter results are not asserted.

The Collision numeric fixture was 121 exact cases with zero absolute and
relative delta.  Candidate-accept and Teacher 105-action-mask comparators
covered 100 missions each and were exact.  No output field, row order, float
calculation, or action mask was removed or reordered.

## Protocol, Bridge, and runtime

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

Evidence includes 75 protocol/golden/C++ contract tests, 24 real P Bridge
integration tests, 10,000 O/P transport transactions, a real-Player
one-worker 100-primitive smoke, and a bounded 12-worker/240-primitive smoke.
The fault run intentionally replaced one worker's Bridge with a failing
executable: that worker failed closed, the other 11 completed 220 primitives,
and no Unity/Bridge/roscore process or test port remained.  The twelve-worker
test is a runtime isolation smoke, not a claim that the separate reliable
Teacher collector is production-ready.

## Warnings, tools, and cleanup

```text
COMPILER_WARNING_COUNT=1
HIGH_SEVERITY_WARNING_COUNT=0
CLANG_FORMAT=NOT_AVAILABLE
CLANG_TIDY=NOT_AVAILABLE
PERF_COUNTERS=NOT_AVAILABLE
GPU_RUNTIME_VALIDATION_DEFERRED=YES
```

The one warning is the existing unused `AsInt32` helper in
`src/protocol/endpoint_observation_snapshot_wire.cpp`; no high-severity
warning was found.  Generated caches and bytecode were moved recoverably to
`/tmp/xm-c7-source-tree-garbage.KYx1El`.  No file was deleted.

## Modified/deleted files and remaining blockers

```text
MODIFIED_FILES=
docs/CXX_FOUNDATION_MIGRATION.md
docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md
docs/CXX_FINAL_V1.md
docs/CXX_ABI_MANIFEST.md
docs/CXX_FINAL_FREEZE_REPORT.md
DELETED_FILES=NONE
```

The first two files are append-only progress records.  The last three are
the final contract, ABI manifest, and freeze report.  O and Unity were not
modified.  The P source/CMake/package behavior was not changed during C7.

Remaining non-C++ blockers are reliable-exact Teacher collection and
merge/provenance closure, BC checkpoint/training/evaluation handoff, and
AWAC/RL pipeline finalization.  Python reference paths, PyTorch CUDA/AMP, and
the Unity project remain outside this freeze change.

```text
NEXT_PHASE=PLANNING PYTHON PIPELINE FINALIZATION
```

No C3 work, formal data collection, training, or commit was performed after
the C7 gates passed.
