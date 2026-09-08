# PY5 clean build and cutover preparation report

Date: 2026-08-28

## Scope and environment

```text
O=/home/xm/XM/xm_ws/src/planning       # read-only baseline
P=/home/xm/XM/src                      # development candidate
CONDA_ENV=xm
SYSTEM_PYTHON=/usr/bin/python3        # catkin install-space
CONDA_PYTHON=/home/xm/anaconda3/envs/xm/bin/python3
UNITY_FINAL_V1=PASS
CXX_FINAL_V1=PASS
FORMAL_60000_COLLECTION=NOT_RUN
AWAC_RUN=NOT_RUN
```

All build and Python runtime commands used:

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
```

No O, Unity, C++, algorithm, formal data, or commit was changed by PY5.

## Generated-file cleanup

The exact P source cleanup removed only confirmed generated Python bytecode
and cache directories. Source tests, fixtures, golden vectors, docs, and
benchmark source files were retained.

```text
GENERATED_FILE_COUNT_BEFORE=56
GENERATED_FILE_COUNT_AFTER=0
P_SOURCE_PYC_COUNT=0
P_SOURCE_CACHE_DIR_COUNT=0
P_SOURCE_BUILD_DEVEL_INSTALL=ABSENT
P_TEMP_BENCHMARK_OUTPUTS=0
```

The old PY4-A `56 -> 56` cache observation was a historical pre-cleanup
observation and is superseded by the PY5 `56 -> 0` result.

## Clean P build

Fresh temporary catkin roots were used; P was symlinked into each temporary
workspace and no existing P build cache was reused.

```text
DEVEL_ROOT=/tmp/xm-py5-devel-rfsn9K
INSTALL_ROOT=/tmp/xm-py5-install-MXchWt
PY5_DEVEL_BUILD_RC=0
PY5_INSTALL_BUILD_RC=0
CMAKE_CONFIGURE=PASS
CLEAN_BUILD=PASS
DEVEL_SPACE_BUILD=PASS
INSTALL_SPACE_BUILD=PASS
```

The relevant built targets were `planning_voxel_map`,
`planning_collision_checker`, `planning_depth_safety`,
`planning_global_route`, `voxel_map_parity_fixture`, `map_loader_node`,
`visualization_node`, and `unity_bridge_node`, plus generated ROS message
targets. Installed library SHA256 values were:

```text
libplanning_collision_checker.so a50a78c4e74d68af55a382c90ae0b13cde1cf7272d07343305bd714cd9e5e638
libplanning_depth_safety.so     179e165cc2ef0f47232b61c0ccdb4ad1561047aab29893b841f4f8566cf27fab
libplanning_global_route.so     ee7375e31bc8eba08c458974a4523b59e92e908b89df3d4cd97e809051c01435
libplanning_voxel_map.so        745ce457ccbca218e06cf3650ee1f7abffe42036ef90eb57c03a0497a8aaba7f
```

The only compiler warning in both builds was the existing unused anonymous
`AsInt32` function in
`src/protocol/endpoint_observation_snapshot_wire.cpp`. There were no compile
errors. The collision target compile flags include `-O3 -DNDEBUG -fPIC
-fopenmp -std=c++17`; link flags include `-O3 -DNDEBUG`, `-lgomp`, and
`-lpthread`. No LTO, explicit architecture flag, visibility override,
`-fno-exceptions`, or `-fno-rtti` flag was present.

## Installed surface and import

```text
INSTALLED_TEST_FILE_COUNT=0
INSTALLED_FIXTURE_FILE_COUNT=0
INSTALLED_DIAGNOSTIC_FILE_COUNT=26
INSTALLED_DIAGNOSTIC_REQUIRED_COUNT=4
INSTALLED_DIAGNOSTIC_OPERATOR_COUNT=22
INSTALLED_DIAGNOSTIC_QUALIFICATION_COUNT=0
QUALIFICATION_ONLY_INSTALLED_COUNT=0
```

The four required diagnostic package files are the package initializer plus
`action_distribution.py`, `observability.py`, and
`pairing_counterfactual.py`. The remaining 16 diagnostic modules and six
named debug/reference CLIs are operator/reference tools. They are retained;
none is classified as a qualification-only installed artifact.

The clean install imported the staged P package and generated ROS messages:

```text
PYTHON_IMPORT=PASS
PYTHON_IMPORT_ROOT=/tmp/xm-py5-install-MXchWt/install/lib/python3/dist-packages/planning/__init__.py
PLANNING_MSG_ROOT=/tmp/xm-py5-install-MXchWt/install/lib/python3/dist-packages/planning/msg/__init__.py
```

## Compileall and test gates

```text
PY5_COMPILEALL=PASS
CANONICAL_CLI=11/11
FROZEN_CXX_PROTOCOL_FOCUSED=55 passed, 1 skipped
```

The full P pytest command imported the staged P install and produced:

```text
PYTHON_FULL_TESTS=FAIL
TEST_FAILURE_COUNT=39
717 passed, 42 skipped, 39 failed in 28.62s
```

Failure categories are explicit and were not masked:

| Count | Blocker |
|---:|---|
| 10 | Frozen Unity C# contract tests refer to old flat `Assets/Scripts` paths; current Unity files are under `Runtime/Observation`, `Runtime/Protocol`, and `Runtime/Primitive`. Unity is frozen. |
| 4 | Existing `test_p0_m2_single_worker_runner.py` fake runtime lacks `port_profile` required by the current P `run_real()` seam. |
| 24 | Frozen SAC/AWAC guarded-runner tests source a hard-coded `/home/xm/devel/setup.bash` in their test environment. AWAC/SAC is frozen. |
| 1 | Existing contract-naming test flags versioned frozen `_v1/_v2/_v4` names. |

The full log is `/tmp/xm-py5-full-tests-escalated.log`. A focused native C++
and protocol run passed 55 tests with one intentional ROS-gated skip; its log
is `/tmp/xm-py5-frozen-cxx-protocol.log`.

## Formal MPL gate

The P formal config was used as the input. Its relative output paths were
resolved by a P-first staged ROS package root, so no P or O formal data path
was written.

```bash
python3 /home/xm/XM/src/scripts/generate_motion_primitives.py \
  --config /home/xm/XM/src/config/motion_primitives.yaml
```

Staged output:

```text
STAGED_MPL_READY=YES
MPL_DETERMINISTIC=PASS
MPL_ACTION_COUNT=105
MPL_FRAME_COUNT=25 command frames; 26 reference frames
MPL_NPZ_SHA256=22ad22fe66a88633de6effff11c03aa4ca4e8df362f8b410e17effbd91ff7309
MPL_JSON_SHA256=c1b795e4737a12034b0d59a1b457f559e7ad138527df2c90ed35db1326e42ea8
MPL_CONTRACT_SHA256=f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d
```

The pair is at `/tmp/xm-planning-cutover-stage/data/motion_primitives/` and
was generated twice with identical hashes. It was not copied to P or O.

## Staged cutover rehearsal

The staged tree `/tmp/xm-planning-cutover-stage/` contains 479 P source files,
the staged MPL pair, and the O voxel cache. It contains no P bytecode and no
O history archive.

```text
STAGE_FILE_COUNT=482
STAGE_PYTHON_MODULE_COUNT=132
STAGE_PYC_COUNT=0
P_SOURCE_MANIFEST_SHA256=1da2e39d74f154205346daee314e02653f6fe378a972d3e4331bc36ceb9bdb18
STAGE_MANIFEST_SHA256=c4c93570d85c411bd1781a828bccddb836a9e0c5f6c007e15e2954f3c0411ada
```

The clean staged mini-chain passed:

```text
STAGED_PRE_BC_E2E=PASS
collection: 2 workers, 8 attempted, 4 accepted, 150 reliable rows
relabel/audit/masks/mmap: 4 episodes, 114 transitions
legacy_rows=0
telemetry_lookup_count=0
snapshot_missing_count=0
state_depth_skew_max_ns=0
frame_contract_failures=0
BC training: 1 epoch, train 85 rows, validation 29 rows
reliable evaluation: 1 audit-only episode, quality_gate_applicable=false
```

The staged artifacts include the audited mission index, collection summary,
labels, masks, dataset audit, BC mmap manifest, one-epoch checkpoint, and
evaluation summary. The BC mmap manifest SHA is
`ff45bf96abd6176d6a68245b54ac57d148d792f02ad6fd01fabd22c0c9e5d4e1`; the
checkpoint SHA is
`995dbab57f7e67d7eccf1205b791dee6cc0af1f5735389228fe9a8b52db83205`.

The 12-worker bounded infrastructure smoke used `MAX_STEPS=3` and
`TARGET_ACCEPTED=0`; it intentionally did not assert a collection quality
gate or collect formal data. It passed startup, identity, port, reliability,
and cleanup checks:

```text
STAGED_TWELVE_WORKER_SMOKE=PASS
WORKER_SPEC_COUNT=12
WORKER_ID_CONTIGUOUS=True
RUNTIME_IDS_UNIQUE=True
PORT_UNIQUE=True
ALL_WORKERS_COMPLETE=True
ALL_RELIABLE_ROWS_POSITIVE=True
ALL_LEGACY_ZERO=True
ALL_TELEMETRY_LOOKUP_ZERO=True
ALL_SNAPSHOT_MISSING_ZERO=True
ALL_SKEW_ZERO=True
ALL_FRAME_FAILURES_ZERO=True
ALL_COLLECTOR_ERRORS_ZERO=True
ORPHAN_PROCESS_COUNT=0
```

The parent collector returned a quality-gate nonzero status only because the
bounded run accepted zero episodes; that is expected for this infrastructure
smoke and does not change the startup/cleanup result.

## Cutover and rollback gates

```text
CUTOVER_DRY_RUN=PASS
CUTOVER_REPLACE_COUNT=381
CUTOVER_DELETE_COUNT=0
CUTOVER_PRESERVE_COUNT=120179
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
ROLLBACK_PLAN=PASS
P_TO_O_CUTOVER_READY=NO
```

The dry-run output is `/tmp/xm-py5-cutover-rsync-dry-run.txt`. The exact
exclude rules protect O data/history/build/devel/install/cache paths. The
future backup, cutover, and rollback commands are in
`P_TO_O_CUTOVER_PLAN.md`.

## PY5 result

```text
PY5_P_CLEAN_BUILD_AND_CUTOVER_PREP=FAIL
P_CLEAN_BUILD=PASS
P_INSTALL_BUILD=PASS
PYTHON_FULL_TESTS=FAIL
STAGED_MPL_READY=YES
STAGED_PRE_BC_E2E=PASS
STAGED_TWELVE_WORKER_SMOKE=PASS
O_PRESERVATION_MANIFEST=PASS
CUTOVER_DRY_RUN=PASS
ROLLBACK_PLAN=PASS
NEXT_PHASE=PY5 FIX CUTOVER BLOCKERS
```

## PY5.1 — Full test contract closure (current)

PY5.1 supersedes the failing full-suite gate above. O, Unity, and C++
remained read-only/frozen; no formal 60,000-row collection, RL run, or
cutover was executed.

```text
PY5_1_FULL_TEST_CONTRACT_CLOSURE=PASS
INITIAL_TEST_FAILURE_COUNT=39
CLASSIFIED_FAILURE_COUNT=39
UNKNOWN_FAILURE_COUNT=0
TRUE_RUNTIME_REGRESSION_COUNT=0
TRUE_ALGORITHM_REGRESSION_COUNT=0
PYTHON_FULL_TESTS=PASS
FINAL_TEST_FAILURE_COUNT=0
FULL_SUITE=756 passed, 42 skipped
SKIP_COUNT_BEFORE=42
SKIP_COUNT_AFTER=42
XFAIL_COUNT_BEFORE=0
XFAIL_COUNT_AFTER=0
CANONICAL_CLI=11/11
PY5_COMPILEALL=PASS
```

The complete 39-row matrix, classifications, and required actions is in
`docs/PY5_FULL_TEST_CONTRACT_CLOSURE.md`. The 10 Unity failures were stale
frozen-owner paths, the four P0 failures were a stale fake-runtime field, the
24 AWAC/SAC failures were a stale setup root, and the naming failure was a
stale unbounded text assertion. All retained the original business
assertions; no skip/xfail count increased. AWAC/SAC math, cadence,
checkpoint/resume, and guard behavior were not changed.

The one proven production path seam changed was
`scripts/run_sac_guarded.sh`; therefore the clean-build and runtime gates
were rerun. Fresh roots were `/tmp/xm-py5-1-devel-final` and
`/tmp/xm-py5-1-install-final`, with installed library hashes:

```text
libplanning_collision_checker.so c30fe84daa77af68cf2bf7a0b3882f59a152b66c814535fc67834431a33af803
libplanning_depth_safety.so     179e165cc2ef0f47232b61c0ccdb4ad1561047aab29893b841f4f8566cf27fab
libplanning_global_route.so     c56fcdc2dffbe39b36bdcbfae9d8f18dcab31f031214765b0b7f540765c7b070
libplanning_voxel_map.so        745ce457ccbca218e06cf3650ee1f7abffe42036ef90eb57c03a0497a8aaba7f
```

The fresh staged evidence is `/tmp/xm-py5-1-staged-e2e-final/`:

```text
STAGED_PRE_BC_E2E=PASS
2-worker collection=8 attempted, 4 accepted, 150 reliable rows
relabel/masks/audit/mmap=PASS, 4 episodes, 114 transitions
BC_TRAINING_SMOKE=PASS, 1 epoch, train 85, validation 29
RELIABLE_EVALUATION_SMOKE=PASS, 1 audit-only episode
legacy_rows=0
telemetry_lookup_count=0
snapshot_missing_count=0
state_depth_skew_max_ns=0
frame_contract_failures=0
```

The post-build 12-worker infrastructure smoke used the existing audited
mission fixture, 12 unique workers, isolated ports, `MAX_STEPS=3`, and
`TARGET_ACCEPTED=0`. All workers completed with unique identities and zero
collector/reliability errors. The parent quality status was intentionally
false only because zero accepted episodes fails the separate
`accepted_nonzero` quality gate; infrastructure status was PASS.

```text
STAGED_TWELVE_WORKER_SMOKE=PASS
WORKER_SPEC_COUNT=12
WORKER_ID_CONTIGUOUS=True
RUNTIME_IDS_UNIQUE=True
PORT_UNIQUE=True
ALL_WORKERS_COMPLETE=True
ALL_RELIABLE_ROWS_POSITIVE=True
ALL_LEGACY_ZERO=True
ALL_TELEMETRY_LOOKUP_ZERO=True
ALL_SNAPSHOT_MISSING_ZERO=True
ALL_SKEW_ZERO=True
ALL_FRAME_FAILURES_ZERO=True
ALL_COLLECTOR_ERRORS_ZERO=True
ORPHAN_PROCESS_COUNT=0
```

The final read-only cutover rehearsal is
`/tmp/xm-py5-1-cutover-rsync-dry-run-final.txt`; the final O protection
check is `/tmp/xm-py5-1-o-preservation-final.txt`.

```text
P_SOURCE_FILE_COUNT=482
P_SOURCE_MANIFEST_SHA256=5b6fa89c65011606c556cc92d879173cf0ae1d13ea12804ccddaae12c0f3360d
CUTOVER_DRY_RUN=PASS
CUTOVER_REPLACE_COUNT=396
CUTOVER_DELETE_COUNT=0
O_PRESERVATION_MANIFEST=PASS
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
O_DATA_MANIFEST_SHA256=71e21ea091be6c075651d3bb727497c530c40081a30188af2cb45422450fef7b
O_HISTORY_MANIFEST_SHA256=6b600703263bf3b7f0916fe61a92deaa869278fa99ebb16ccb654bd324e84f41
ROLLBACK_PLAN=PASS
P_TO_O_CUTOVER_READY=YES
NEXT_PHASE=PY6 EXECUTE P TO O CUTOVER
```

This readiness result authorizes only a future, separately executed PY6
cutover; PY5.1 stopped before it.
