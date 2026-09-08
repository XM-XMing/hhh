# PY6 P to O cutover report

Date: 2026-08-28

This report records the authorized `PY6_P_TO_O_CUTOVER`. The historical O
source tree was `/home/xm/XM/xm_ws/src/planning`; the only development source
used for the replacement was `/home/xm/XM/src`. The P tree and Unity were not
modified by this cutover.

## Final gate

```text
PY6_P_TO_O_CUTOVER=PASS
PLANNING_PRE_BC_FINAL_V1=PASS
BACKUP=PASS
BACKUP_PATH=/home/xm/XM/xm_ws/src/planning_backup_pre_py6_20260828T141420Z
MIGRATION_FILE_COUNT=482
MIGRATION_REPLACE_COUNT=396
MIGRATION_ARCHIVE_COUNT=119
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
O_PRESERVED_FILE_MUTATION_COUNT=0
O_CLEAN_BUILD=PASS
O_DEVEL_BUILD=PASS
O_INSTALL_BUILD=PASS
O_PYTHON_IMPORT=PASS
PYTHON_IMPORT_ROOT=/home/xm/XM/xm_ws/src/planning/python
FULL_PYTHON_TESTS=PASS
FULL_TEST_PASS_COUNT=756
FULL_TEST_SKIP_COUNT=42
FULL_TEST_FAILURE_COUNT=0
CANONICAL_CLI=11/11
FORMAL_MPL=PASS
MPL_NPZ_SHA256=22ad22fe66a88633de6effff11c03aa4ca4e8df362f8b410e17effbd91ff7309
MPL_JSON_SHA256=c1b795e4737a12034b0d59a1b457f559e7ad138527df2c90ed35db1326e42ea8
O_PRE_BC_E2E=PASS
O_TWELVE_WORKER_SMOKE=PASS
OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
LEGACY_ROWS=0
TELEMETRY_LOOKUP_COUNT=0
SNAPSHOT_MISSING_COUNT=0
STATE_DEPTH_SKEW_MAX_NS=0
FRAME_CONTRACT_FAILURES=0
RUNTIME_IDENTITY=PASS
PROCESS_CLEANUP=PASS
ROLLBACK_EXECUTED=NO
AWAC_TO_SAC_IMPORT_COUNT=3
ALGORITHM_CHANGED=NO
RUNTIME_BEHAVIOR_CHANGED=NO
NEXT_PHASE=USER EXECUTES FORMAL RELIABLE BC DATA COLLECTION
```

`PASS` here means the controlled source cutover and bounded Pre-BC readiness
smokes passed. It is not a claim that the formal 60,000-row collection,
formal BC training, or AWAC/RL evaluation has been run.

## Backup and controlled replacement

The pre-cutover O code/config/test/document tree is retained at
`/home/xm/XM/xm_ws/src/planning_backup_pre_py6_20260828T141420Z/tree` with
`backup_manifest.json` and `backup_sha256.txt`. It contains 349 regular files;
the backup excludes `data/`, `data_back_20260826/`, generated build roots and
Python caches. The archive metadata is outside the source tree and is not a
second source of truth.

The replacement followed `docs/P_TO_O_MIGRATION_MANIFEST.md` with explicit
excludes for data, history, build, install, and generated caches. It used no
recursive tree deletion, no `--delete`, and no copy of P data. The rsync
source listing contained 482 regular files and transferred 396 regular files.
The 119 stale O-only non-generated source files were moved to the backup
`archive/` and remain recoverable; canonical `planning/awac/`, `planning/rl/`,
`train_sac.py`, `run_sac_guarded.sh`, and the protected AWAC/SAC import edges
remain in O.

The exact replacement log is `/tmp/xm-py6-cutover-rsync.log`. The backup
archive has `archive_manifest.json` and `archive_sha256.txt`.

## Build and import evidence

O was rebuilt from clean generated `build/`, `devel/`, and `install/` roots
using `conda activate xm`, ROS Noetic, CMake Release, and `/usr/bin/python3`
for catkin message generation. The built targets include the generated ROS
messages, `planning_voxel_map`, `planning_depth_safety`,
`planning_global_route`, `planning_collision_checker`, `map_loader_node`,
`visualization_node`, `unity_bridge_node`, and `voxel_map_parity_fixture`.
The build had no errors and one unused-function warning in
`src/protocol/endpoint_observation_snapshot_wire.cpp`.

Installed native library hashes were recorded during the gate:

```text
libplanning_voxel_map.so=745ce457ccbca218e06cf3650ee1f7abffe42036ef90eb57c03a0497a8aaba7f
libplanning_collision_checker.so=a3a9942410f5102b782c38acd1e20616a0b2e632ba13107529c8f9686c39a3b9
libplanning_global_route.so=cefdb007bd4b617b7076063f5807223c8158c39e69fb3b394fcd9e3b9a711dd5
libplanning_depth_safety.so=179e165cc2ef0f47232b61c0ccdb4ad1561047aab29893b841f4f8566cf27fab
```

The O Python import resolved to
`/home/xm/XM/xm_ws/src/planning/python/planning/__init__.py`; no P or backup
Python root was on the validation import path. Compileall passed with 218
temporary bytecode files and zero residual O source-tree bytecode/cache
directories.

CTest reported no registered CTest cases for this catkin build. The full
pytest suite and C6 protocol/Bridge focused suite remained the executable
contract gate.

## Test and CLI evidence

```text
full pytest: 756 passed, 42 skipped, 0 failed in 70.08s
C6 focused: 3 passed, 1 skipped
canonical CLI help: 11/11
```

The 42 skips are the existing opt-in Unity/ROS checks and the documented
missing historical Gate-3 artifact; no skip or xfail was added by PY6.

## Formal MPL

O generated the formal 105-action motion primitive pair after the clean build:

```text
data/motion_primitives/motion_primitives_105.npz
  actions=105, command_frames=25, reference_samples=26
  sha256=22ad22fe66a88633de6effff11c03aa4ca4e8df362f8b410e17effbd91ff7309
data/motion_primitives/motion_primitives_105.json
  sha256=c1b795e4737a12034b0d59a1b457f559e7ad138527df2c90ed35db1326e42ea8
contract_sha256=f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d
```

These are the only formal MPL files added by the cutover. No formal rollout
or training dataset was generated in O.

## Bounded O Pre-BC chain

All bounded artifacts are under `/tmp/xm-py6-o-e2e/` and have a passing
`sha_chain.json`:

```text
missions: generated and audited; audited mission SHA=
  812e2a8838c6ab14930700eaf1bb0dbebaf2200e1ff5e51f3456310f683a9087
2-worker reliable collection: 4 accepted, 176 reliable rows
relabel: 114 transitions, teacher valid rate=1.0
depth masks: 114 transitions, 105 actions
dataset audit: 4/4 episodes, 0 failures
BC mmap: 114 transitions, 105 actions
BC smoke: one CPU epoch, quality_pass=true
evaluation smoke: episode 2, reliable audit-only, quality_gate_applicable=false
```

Every chain artifact carried
`reliable_exact_endpoint_snapshot`; legacy rows, telemetry lookups, snapshot
misses, state/depth skew, and frame-contract failures were zero. Episode IDs
were `[2, 4, 7, 9]` with offsets `[0, 28, 56, 85]` and lengths `[28, 28, 29,
29]`. The SHA-chain checks source index/manifest, labels, masks, mmap, BC
checkpoint, evaluation index, MPL, and the exact array/layout dimensions.

The evaluation smoke ended in a `dead_end` for episode 2. It is audit-only
runtime evidence and is not a formal policy-quality result.

## Twelve-worker infrastructure smoke

The O 12-worker smoke used isolated ports, `MAX_STEPS=3`, and
`TARGET_ACCEPTED=0`. All 12 workers became ready and finished; the parent
observed 25 attempts and 75 reliable rows. It intentionally returned the
parent `accepted_nonzero` quality gate as false because this was an
infrastructure-only smoke. The gate nevertheless passed because all 12
worker reports had unique runtime identities and canonical endpoint ports,
zero collector errors, zero legacy/telemetry/snapshot/skew/frame failures,
and a valid endpoint identity chain.

## Frozen runtime identity

The actual Unity runtime identity used by both runtime smokes was explicitly
validated:

```text
Player=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
Assembly-CSharp=7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc
Bridge=63b00f6a0b31aa72ace3e1c437477e7ee8b670934386f3bb1d645745c302d9ed
Bridge path=/tmp/xm-c7-clean-1.vLxvbk/devel/lib/planning/unity_bridge_node
```

The freshly built O install/devel Bridge binaries have different hashes due
to build-artifact identity/reproducibility (`d50b...` in install and
`29a9...` in devel) and were not silently substituted for the fixed runtime
artifact. The runtime identity gate therefore refers to the explicitly
selected frozen Bridge above; CUDA/hybrid collision validation remains
outside this cutover.

## Preservation and boundaries

The final preservation record is `/tmp/xm-py6-o-preservation-final/summary.json`.
Compared with the post-sync O snapshot, existing `data/` content (excluding
the formal MPL directory) and the complete `data_back_20260826/` tree have:

```text
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
O_PRESERVED_FILE_MUTATION_COUNT=0
FORMAL_MPL_FILES_ADDED=2
point-cloud-sha=dfd87a5db1ab98276eda57e79de677f711da56f308a373b93e72484fa5876d40
voxel-cache-sha=a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691
```

All project runtime processes and managed ports were absent after each smoke;
the final record is `/tmp/xm-py6-o-e2e/process_cleanup.json`. Unity, C++ source
semantics, Bridge source, Python algorithms, BC mathematics, AWAC/SAC, and
historical data were not changed. No commit was created. The next authorized
operation is user-run formal reliable BC data collection; formal BC training
and AWAC/RL remain subsequent, separate gates.
