# Pre-Collection Runtime Final V3

Date: 2026-08-29

This is the current pre-collection freeze for the canonical Planning source
tree.  It is a readiness record, not a record of formal data collection.

## Status

```text
FINAL_PRE_COLLECTION_CONSOLIDATION=PASS
FULL_COLLECTION_READY=YES
FORMAL_COMMANDS_EMITTED=YES
FORMAL_COMMANDS_EXECUTED_BY_CODEX=NO
FORMAL_2M_EXECUTED_BY_CODEX=NO
FORMAL_60K_EXECUTED_BY_CODEX=NO
BC_TRAINING_EXECUTED=NO
AWAC_EXECUTED=NO
COMMIT_EXECUTED=NO
```

## Frozen paths and identities

```text
WORKSPACE=/home/xm/XM/xm_ws
PLANNING_ROOT=/home/xm/XM/xm_ws/src/planning
UNITY_PLAYER=/home/xm/XM/xm_ws/src/unity/XMflight.x86_64
UNITY_DATA=/home/xm/XM/xm_ws/src/unity/XMflight_Data
BUILD_SYSTEM=catkin_make
BUILD_PREFIX=/home/xm/XM/xm_ws/devel
INSTALL_USED=NO
PYTHON_ENV=xm
OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
TASK_CONTRACT_SCHEMA_VERSION=2
TASK_CONTRACT_SHA256=2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df
BASE_SEED=2026
```

```text
UNITY_PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
UNITY_ASSEMBLY_SHA256=319f9e764cf51078d3b5e439214361f54aa9d5726449e26e4c0a1ad6232f21f0
BRIDGE_PATH=/home/xm/XM/xm_ws/devel/lib/planning/unity_bridge_node
BRIDGE_SHA256=667c266e96211aabb5a0cd25b8b35b303903d35cbe2d95be8f9d2bbe5e63d789
FOREST_POINT_CLOUD_SHA256=dfd87a5db1ab98276eda57e79de677f711da56f308a373b93e72484fa5876d40
VOXEL_CACHE=/home/xm/XM/xm_ws/src/planning/data/map_data/forest_voxels_10cm.npz
VOXEL_CACHE_SHA256=a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691
MPL_NPZ_SHA256=22ad22fe66a88633de6effff11c03aa4ca4e8df362f8b410e17effbd91ff7309
MPL_JSON_SHA256=c1b795e4737a12034b0d59a1b457f559e7ad138527df2c90ed35db1326e42ea8
MPL_CONTRACT_SHA256=f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d
```

The current Bridge identity is the SHA256 value above.  The previous
2026-08-29 Bridge identity `462e9f7bc25d5bef63ce17b863daa256726e17de870bda04a1bcb72e79e80f2b`
is retained as `historical_runtime_identity` in the machine-readable
manifest and is not a current runtime requirement.

The cache was generated from the current point cloud with the canonical
voxel-cache owner.  Its metadata is `voxel_size=0.10`, `file_frame=unity`,
`data_offset_bytes=4` after `-1` auto-detection, `origin_ijk=[-1023,-1020,-1]`,
`grid_shape=[2046,2045,42]`, and `occupied_count=1740947`.  Occupied keys,
metadata, coordinate conversion, and the fixed collision fixture matched the
read-only historical oracle exactly.  The historical cache is not a runtime
dependency.

The devel native libraries loaded successfully:

```text
libplanning_voxel_map.so
libplanning_collision_checker.so
libplanning_global_route.so
libplanning_depth_safety.so
```

The shared native geometry context loads one immutable VoxelMap per process;
CollisionChecker and GlobalRoutePlanner share that map.  No CUDA/hybrid
collision claim is made here; the formal backend is C++ CPU/native route and
native depth safety.

## Pipeline consolidation

```text
MISSION_ROUTE_STORE_IMPLEMENTATION_COUNT=1
MISSION_ROUTE_STORE_COMPAT_WRAPPER_COUNT=0
FORMAL_MISSION_PREPARATION_ENTRY_COUNT=1
PRODUCTION_SCRIPT_COUNT_BEFORE=28
PRODUCTION_SCRIPT_COUNT_AFTER=21
DELETED_FILES=scripts/generate_missions.py,scripts/audit_teacher_missions.py
MOVED_DIAGNOSTIC_FILE_COUNT=13
ZERO_CALLER_PRODUCTION_WRAPPER_COUNT=0
GENERATION_ASTAR_PER_CANDIDATE=1
AUDIT_ASTAR_CALL_COUNT=0
COLLECTION_ASTAR_CALL_COUNT=0
RELABEL_ASTAR_CALL_COUNT=0
MAX_INFLIGHT_RESULTS=24
MAX_REORDER_BUFFER_ROUTES=24
ROUTE_STORE_MEMORY_COMPLEXITY=BOUNDED
ROUTE_STORE_MEMORY_GATE=PASS
FORMAL_2M_MEMORY_SAFE=YES
```

`prepare_teacher_missions.py` streams candidates, stored routes, and audit
results through bounded journals and a bounded deterministic reorder buffer.
The scale-safe global sampling budget is `max(200000, 4 * count)` when
`--max-sampling-attempts 0`; therefore the formal 2M candidate bound is
8,000,000 attempts.  The base seed is 2026 and the production default scan
found zero non-2026 defaults.  The historical/test/artifact scan found 87
non-2026 literal references; those are retained as historical evidence and
are not production defaults.

The small memory run produced 13 candidates, 12 passing missions, a 27,405
byte route store, and peak RSS 105,192 kB (102.726562 MiB).  The larger run
produced 472 candidates, 400 passing missions, a 943,147 byte route store,
and peak RSS 105,380 kB (102.910156 MiB).  Both used 12 route workers and the
same formal cache.

## Runtime and Pre-BC evidence

The final bounded smoke used the current Player, current devel Bridge, current
native libraries, current formal cache, V2 task contract, and
`reliable_exact_endpoint_snapshot`:

```text
WORKER_READY=12/12
WORKER_FINISHED=12/12
ATTEMPTED=24
ACCEPTED=22
RELIABLE_ROLLOUT_ROWS=657
PERSISTED_TRANSITIONS=632
LEGACY_ROWS=0
TELEMETRY_LOOKUP_COUNT=0
SNAPSHOT_MISSING_COUNT=0
STATE_DEPTH_SKEW_MAX_NS=0
FRAME_CONTRACT_FAILURES=0
COLLECTOR_ERROR_TOTAL=0
ENDPOINT_IDENTITY_CHAIN_VALID=true
RUNTIME_IDENTITY_UNIQUE=12
PORT_COLLISION=0
CROSS_TALK=0
ORPHAN_PROCESS=0
TASK_CONTRACT_V2_SHA_PARITY=PASS
OBSERVATION_CONTRACT_PARITY=PASS
PRE_BC_E2E=PASS
```

The bounded chain completed relabel, dataset audit, depth-mask generation,
and BC mmap construction.  The current artifact hashes are:

```text
ROLLOUT_INDEX_SHA256=58f93ee7afda87de8803fefd04cc72d17989112d1cae1894aeb6913d7cea989c
ROLLOUT_MANIFEST_SHA256=7c7ffe7f2b1e3d6d191cf10fafa7e4ae41bdb812eb103b6b5d733bba44594c35
TEACHER_LABELS_NPZ_SHA256=f060ce3515b1d11fd54b62965140038ed3ea37fd3a690a8f9363870c89cee634
DEPTH_MASKS_NPZ_SHA256=56884ebf83f3deb5ad2ed0ac3f78fc811e28aa5742be9e57c6cb4c1a3f01ed43
DATASET_AUDIT_SHA256=0a4a579762fbe73b5c6fd83a3d1167c577325ce60d67bd012005f1f70c5c8e06
BC_MMAP_MANIFEST_CONTRACT=bc_mmap_dataset
BC_MMAP_TRANSITIONS=632
BC_MMAP_ACTIONS=105
```

All producer artifacts carry the same observation contract and source.  The
collection root contains worker provenance directories, so relabel and audit
were run against the canonical merged root.  A merge path-rebasing defect and
a preparation-to-collector path-contract publication defect were exposed by
bounded smoke tests and fixed at their ownership boundaries; no scoring,
sampling criterion, route geometry, wire behavior, or array layout changed.

## Scope boundary

The formal collection commands are a user handoff and were not executed by
Codex.  The next user action is to execute the final command block.  BC
training/checkpoint production, policy evaluation, and AWAC/SAC remain
explicitly deferred.

## FINAL PRE-COLLECTION CODE CLEANUP + RELEASE GATE — HISTORICAL RESULT (2026-08-29)

This section is a historical release result.  Earlier sections are retained as
historical evidence and are not rewritten.  The current cleanup changed only
the canonical Planning tree; Unity, formal data, BC/AWAC, and the historical
baseline were not modified.

```text
FINAL_PRE_COLLECTION_CODE_CLEANUP=PASS
OLD_MISSION_GENERATOR_DELETED=YES
OLD_MISSION_AUDITOR_DELETED=YES
OLD_ROUTE_VALIDATION_DELETED=YES
OLD_MISSION_PIPELINE_FILE_COUNT=0
FORMAL_MISSION_PREPARATION_ENTRY_COUNT=1
MISSION_PREPARATION_IMPLEMENTATION_COUNT=1
FORMAL_ROLLOUT_COLLECTOR_IMPLEMENTATION_COUNT=1
FORMAL_LEGACY_ASYNC_CALLER_COUNT=0
LEGACY_ASYNC_CODE_IN_TEACHER_ROLLOUT_COLLECTOR=0
LEGACY_DIAGNOSTIC_MOVED=NOT_NEEDED
COLLECTION_SUMMARY_VALIDATOR_IMPLEMENTATION_COUNT=1
COLLECTION_ROW_VALIDATOR_IMPLEMENTATION_COUNT=1
ROLLOUT_MERGE_CONTRACT_RULE_IMPLEMENTATION_COUNT=0
BOOL_PARSER_IMPLEMENTATION_COUNT=1
PRODUCTION_LOCAL_AS_BOOL_COUNT=0
THIN_CANONICAL_HASH_WRAPPER_COUNT=0
ORDINARY_FILE_SHA_IMPLEMENTATION_COUNT=1
GENERIC_CANONICAL_JSON_SHA_IMPLEMENTATION_COUNT=1
UNEXPLAINED_LOCAL_ATOMIC_WRITE_COUNT=0
PROTOCOL_VALIDATION_PRIMITIVE_OWNER_COUNT=1
Z1_3_OCCURRENCE_COUNT_BEFORE=7
Z1_3_OCCURRENCE_COUNT_AFTER=2
REMOVE_NONSEMANTIC_Z1_3_COUNT=5
KEEP_FROZEN_CONTRACT_ID_COUNT=1
KEEP_HISTORICAL_FIXTURE_COUNT=1
HARDCODED_DUPLICATE_TASK_CONTRACT_ID_COUNT=0
UNKNOWN_Z1_3_COUNT=0
FORMAL_BASE_SEED=2026
PRODUCTION_NON_2026_SEED_DEFAULT_COUNT=0
ROUTE_STORE_MEMORY_GATE=PASS
MAX_INFLIGHT_RESULTS=24
MAX_REORDER_BUFFER_ROUTES=24
FORMAL_2M_MEMORY_SAFE=YES
PRODUCTION_SCRIPT_COUNT_BEFORE=28
PRODUCTION_SCRIPT_COUNT_AFTER=21
MOVED_DIAGNOSTIC_FILE_COUNT=13
DELETED_FILE_COUNT=3
DELETED_FILES=python/planning/mission/generator.py,python/planning/mission/auditor.py,python/planning/mission/route_validation.py
ZERO_CALLER_PRODUCTION_WRAPPER_COUNT=0
SHELL_BUSINESS_LOGIC_LOC=0
NEW_PRODUCTION_FILE_COUNT=0
DELETED_PRODUCTION_FILE_COUNT=3
SCOPE_GROWTH_WARNING=NO
CATKIN_MAKE=PASS
COMPILER_WARNING_COUNT=0
FULL_TESTS=PASS
FULL_TEST_PASS_COUNT=815
FULL_TEST_SKIP_COUNT=42
FULL_TEST_FAILURE_COUNT=0
TASK_V1_SHA_PARITY=PASS
TASK_V2_SHA_PARITY=PASS
WIRE_GOLDEN_BYTES_PARITY=PASS
PROTOCOL_ACCEPT_REJECT_PARITY=PASS
UNITY_PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
UNITY_ASSEMBLY_SHA256=319f9e764cf51078d3b5e439214361f54aa9d5726449e26e4c0a1ad6232f21f0
BRIDGE_SHA256=462e9f7bc25d5bef63ce17b863daa256726e17de870bda04a1bcb72e79e80f2b
FOREST_POINT_CLOUD_SHA256=dfd87a5db1ab98276eda57e79de677f711da56f308a373b93e72484fa5876d40
VOXEL_CACHE_SHA256=a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691
PRE_BC_E2E=PASS
TWELVE_WORKER_SMOKE=PASS
ALGORITHM_CHANGED=NO
TEACHER_SCORING_CHANGED=NO
WIRE_BEHAVIOR_CHANGED=NO
TASK_SEMANTICS_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
ROUTE_STORE_SCHEMA_CHANGED=NO
FULL_COLLECTION_READY=YES
CLI_NORMALIZATION=PASS
INLINE_PYTHON_BLOCK_COUNT_IN_FORMAL_COMMANDS=0
RUNTIME_VALIDATOR_ENTRY=scripts/validate_pre_collection_runtime.py
MISSION_VALIDATOR_ENTRY=scripts/validate_teacher_missions.py
COLLECTION_MONITOR_ENTRY=scripts/monitor_teacher_collection.py
COLLECTION_VALIDATOR_ENTRY=scripts/validate_teacher_collection.py
BC_DATASET_VALIDATOR_ENTRY=scripts/validate_bc_dataset.py
THIN_SCRIPT_BUSINESS_LOGIC_LOC=0
FORMAL_COMMANDS_EMITTED=YES
FORMAL_MISSION_PREPARATION_EXECUTED_BY_CODEX=NO
FORMAL_60K_EXECUTED_BY_CODEX=NO
BC_TRAINING_EXECUTED=NO
AWAC_EXECUTED=NO
COMMIT_EXECUTED=NO
```

The three deleted mission modules had no production, dynamic, RL, public API,
checkpoint, or install caller.  Their reusable preparation helper now has one
canonical owner in `planning.mission.preparation`.  The formal collector is
Reliable Exact only; the legacy observation contract remains only as an
explicit compatibility/audit identity and is not a formal collector path.
Collection summary/row validation is owned by
`planning.contracts.pipeline_provenance`; `rollout_merge` performs merge,
ordering, aggregation, and I/O only.  Boolean parsing, mechanical hashes,
ordinary atomic I/O, and shared protocol validation primitives likewise have
one common owner.  The frozen task-contract identifier and one historical
fixture reference remain intentional.

The clean devel build produced eight expected targets with zero compiler
warnings, and all four native libraries loaded from `$WS/devel`.  The fixed
20-primitive runtime alignment passed 20/20 physical executions, ACKs,
commits, receipts, and endpoint snapshots with 25 frames.  The current
12-worker smoke passed 12/12 ready and finished, 24 attempts, 22 accepted,
657 reliable rollout rows, and 632 persisted transitions.  Legacy rows,
telemetry lookups, snapshot misses, skew, frame failures, collector errors,
port collisions, cross-talk, and orphan processes were all zero.

The bounded Pre-BC chain passed collection merge, relabel, depth masks,
dataset audit, and BC mmap construction.  Direct cross-artifact validation
also passed for the exact observation contract, row count, episode order,
transition offsets, and action count.  This is a preprocessing gate only;
the explicitly forbidden BC training/evaluation and AWAC/SAC were not run.
The smoke artifact evidence is kept under `/tmp`; no formal `$ROOT/data/teach`
run was created or populated.

```text
SMOKE_ROLLOUT_INDEX_SHA256=b9a12fb30cf7de159bdf522df6fd2af877e8198331ffb13ca1941bd669cf6f3d
SMOKE_ROLLOUT_MANIFEST_SHA256=6663798bdc139b77cc034880b6ee712fb9ed5f6f8f73d4d68f9eb7617f4bf5c7
SMOKE_TEACHER_LABELS_NPZ_SHA256=56c1eaea0c27946affd6353b13e70434a3144fbfab12cc4383bf2aa6ae19c2e9
SMOKE_DEPTH_MASKS_NPZ_SHA256=8b183a99b1212beb8ad2bebb42d86fe7be13a21b49f01f92860208f713818d1f
SMOKE_DATASET_AUDIT_SHA256=fb413e99132942b69c8cdccf906d238dc524474c507dc39a64b37dbb4da0d6da
SMOKE_BC_MMAP_MANIFEST_SHA256=c5e00f8cb3816e82a00c5c27284a5eacf0df3f927909384e2e739960d8e00c80
SMOKE_OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
SMOKE_TASK_CONTRACT_SCHEMA_VERSION=2
SMOKE_MAX_STEPS=45
SMOKE_CROSS_ARTIFACT_VALIDATION=PASS
```

The formal handoff uses the fresh run name
`flight_reliable_exact_teacher_seed2026_20260829_v3`; its commands are
emitted below in the final response but were not executed by Codex.

## CLI normalization — current result (2026-08-29)

The five validation/monitor entries are thin public adapters to canonical
Planning owners.  The parallel collector accepts the normalized flag form and
retains its established positional compatibility seam.  No contract, schema,
algorithm, or collection behavior changed.  The full suite was rerun with
local TCP permissions required by runtime lifecycle tests.

```text
CLI_NORMALIZATION=PASS
INLINE_PYTHON_BLOCK_COUNT_IN_FORMAL_COMMANDS=0
RUNTIME_VALIDATOR_ENTRY=scripts/validate_pre_collection_runtime.py
MISSION_VALIDATOR_ENTRY=scripts/validate_teacher_missions.py
COLLECTION_MONITOR_ENTRY=scripts/monitor_teacher_collection.py
COLLECTION_VALIDATOR_ENTRY=scripts/validate_teacher_collection.py
BC_DATASET_VALIDATOR_ENTRY=scripts/validate_bc_dataset.py
THIN_SCRIPT_BUSINESS_LOGIC_LOC=0
FULL_TESTS=PASS
FULL_TEST_PASS_COUNT=815
FULL_TEST_SKIP_COUNT=42
FULL_TEST_FAILURE_COUNT=0
FORMAL_COMMANDS_EXECUTED_BY_CODEX=NO
```

## Collection CLI + logging normalization — current result (2026-08-29)

This is an append-only operational normalization record.  It changes only
typed CLI/config resolution and human-facing log routing.  Mission
preparation, collection/reliable-exact behavior, protocol, task and
observation contracts, artifact schema, Unity, C++, BC, and AWAC/SAC are
unchanged.  The machine-readable progress and summary files remain the
quality-gate owners; human logs are not parsed as a quality gate.

```text
COLLECTION_CLI_NORMALIZATION=PASS
FORMAL_COLLECTION_REQUIRED_ENV_VAR_COUNT=0
COLLECTOR_EXTRA_ARGS_REQUIRED=NO
MAIN_COLLECTION_LOG=logs/collection.log
WORKER_LOG_PATTERN=logs/workers/worker_%02d.log
UNITY_LOG_PATTERN=logs/unity/worker_%02d.log (+ .resume_%02d.log and .launcher.log)
BRIDGE_LOG_PATTERN=logs/bridge/worker_%02d.log
PREPARATION_LOGGING=PASS
COLLECTION_LOGGING=PASS
RESUME_LOGGING=PASS
CLI_RESOLVED_CONFIG_PARITY=PASS
FULL_TESTS=PASS
FULL_TEST_PASS_COUNT=825
FULL_TEST_FAILURE_COUNT=0
TWELVE_WORKER_SMOKE=PASS
SMOKE_PATH=/tmp/xm-cli-logging-12-worker-smoke-20260829-r1
SMOKE_ATTEMPTED=24
SMOKE_ACCEPTED=22
SMOKE_RELIABLE_ROWS=657
SMOKE_LEGACY_ROWS=0
SMOKE_TELEMETRY_LOOKUP=0
SMOKE_SNAPSHOT_MISSING=0
SMOKE_SKEW_MAX_NS=0
SMOKE_FRAME_FAILURES=0
SMOKE_COLLECTOR_ERRORS=0
ALGORITHM_CHANGED=NO
WIRE_BEHAVIOR_CHANGED=NO
ARTIFACT_SCHEMA_CHANGED=NO
FULL_COLLECTION_READY=YES
FORMAL_PREPARATION_EXECUTED_BY_CODEX=NO
FORMAL_COLLECTION_EXECUTED_BY_CODEX=NO
BC_TRAINING_EXECUTED=NO
AWAC_EXECUTED=NO
COMMIT_EXECUTED=NO
```

The normalized collection CLI owns all formal collection values that were
previously supplied through collection-specific environment variables.
Deprecated environment compatibility is fail-closed for
`COLLECTOR_EXTRA_ARGS`; typed Teacher and route flags are emitted directly
from the resolved configuration.  Preparation creates the same common log
directory layout and mirrors its existing console output to
`logs/preparation.log` without changing its business flow.
