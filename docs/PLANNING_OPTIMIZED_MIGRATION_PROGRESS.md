# Planning optimized-tree controlled migration progress

Date: 2026-08-27

## M0.1 status

```text
M0_1_BC_MMAP_PROVENANCE=PASS
OBSERVATION_PROVENANCE_RESTORED=YES
BC_MMAP_ARRAY_PARITY=PASS
BC_MMAP_MANIFEST_PARITY=PASS
ALGORITHM_CHANGED=NO
DATA_LAYOUT_CHANGED=NO
```

Scope was limited to the P-side BC mmap provenance seam. O was not modified.
Mission, Teacher, Collection, BC training, Evaluation, AWAC/RL, Unity, and
performance code were not changed.

## Modified files

- `python/planning/contracts/observation.py`
- `python/planning/data/bc_mmap.py`
- `tests/test_observation_contract.py`
- `tests/test_bc_mmap_observation_provenance.py`
- `tests/compare_bc_mmap_fixture.py`
- `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`

The new contract owner accepts only explicitly declared supported values and
fails closed for missing, unknown, or mismatched `observation_source` values.
BC mmap now requires all rollout, label, and depth-mask contracts to agree on
`reliable_exact_endpoint_snapshot`. Legacy-only and mixed legacy/reliable rows
are rejected. Existing mmap arrays, dtypes, shapes, row order, and manifest
business fields were retained; the four provenance fields were restored:
`observation_contract`, `observation_source`, `reliable_rows`, and
`legacy_rows`.

## Tests and comparator

All commands used `conda activate xm`.

```text
compileall: PASS
focused pytest: 14 passed
BC_MMAP_ARRAY_PARITY=PASS
BC_MMAP_MANIFEST_PARITY=PASS
BC_MMAP_ROW_ORDER_PARITY=PASS
OBSERVATION_PROVENANCE_RESTORED=YES
TRAIN_VAL_SPLIT_PARITY=NOT_APPLICABLE_AT_BC_MMAP_SEAM
```

The same two-transition reliable-exact fixture was built by O and P.
Manifest SHA-256 was identical:

```text
O/P manifest: 83b0b3ff4c0f0bf8fa58924238033bbe16908c9b51f106947f705764316c44d5
depths:         712d347f31aee4cf3f6bd464163089bd13fbe71ba06389980f69e3bc8b50426b
continuous:     35ee2b5c446d41f60df99ccef70dc4b519e8770af821ca3fb40174d7909fc83e
prev_actions:   bebf27d0ac6c8707dd957023219bafa9cea972e7e55b9aa0be4e5df8bf27e2da
height_masks:   3a2c872e723eff921dfbc42f433d977f720b369cc5e7cfcb82a8ecd379814b71
local_masks:    3a2c872e723eff921dfbc42f433d977f720b369cc5e7cfcb82a8ecd379814b71
behavior:       f5c0dd07755f49f61eff05ecf63e1b6cb082e28a1e771720ca3e5bf6e9943eab
teacher:        f5c0dd07755f49f61eff05ecf63e1b6cb082e28a1e771720ca3e5bf6e9943eab
soft_targets:   1b29a675a9057e5fda9722d3cf63fd89b0560c81000489fee8ce87e6ac430fc0
history_starts: 7500f15e4319372a86620f1b865dac4901887634e69213f76e1df4927cbd5f51
episode_ids:    7500f15e4319372a86620f1b865dac4901887634e69213f76e1df4927cbd5f51
```

## Remaining observation provenance chain

- Reliable Teacher collection and its managed launcher remain absent in P.
- P depth-mask and relabel producers still do not emit/validate provenance;
  M0.1 only validates their supplied artifact metadata at the BC mmap seam.
- BC checkpoint/trainer provenance is intentionally unchanged and remains the
  next phase.
- Evaluation and AWAC/RL handoff provenance remain intentionally unchanged.
- Train/validation split parity is not applicable to the BC mmap builder and
  must be checked in the next checkpoint/training phase.

## Next phase

`NEXT_PHASE=M0.2 BC CHECKPOINT / TRAINING PROVENANCE`

M0.2 was pending at the time of this earlier progress snapshot; the current
status is recorded below.

## M0.2 status

```text
M0_2_BC_TRAINING_PROVENANCE=PASS
BC_INPUT_CONTRACT_VALIDATION=PASS
BC_CHECKPOINT_PROVENANCE=PASS
BC_SUMMARY_PROVENANCE=PASS
BC_RESUME_PROVENANCE=NOT_APPLICABLE
MODEL_TENSOR_SHA_PARITY=PASS
NORMALIZER_PARITY=PASS
OPTIMIZER_STATE_PARITY=PASS
TRAINING_METRIC_PARITY=PASS
CHECKPOINT_RELOAD_LOGIT_PARITY=PASS
ALGORITHM_CHANGED=NO
DATA_LAYOUT_CHANGED=NO
```

M0.2 was limited to P's BC trainer boundary.  The trainer now reads the BC
mmap manifest before constructing the training dataset and rejects missing or
unknown observation provenance, CLI contract mismatches, legacy rows in a
reliable dataset, and reliable-row count mismatches.  The resolved contract is
printed, written to `resolved_training_config.json`, and carried into the
checkpoint, summary, and structured run events.  Existing model state,
normalizer values, optimizer behavior, split logic, metrics calculation, and
mmap arrays were not changed.

O has no BC optimizer-resume entry point.  Its
`--val-episodes-from-checkpoint` option only reuses a validation split, so
`BC_RESUME_PROVENANCE=NOT_APPLICABLE`; no resume path was added to P.

### M0.2 fixture and parity evidence

The deterministic fixture used four reliable-exact episodes and one shared P
BC mmap cache.  Both trainers ran on CPU with seed `17`, one epoch, batch size
`2`, one CPU thread, `num_workers=0`, no AMP, and the explicit
`reliable_exact_endpoint_snapshot` contract.  The comparator exercised dataset
loading, normalizer fitting, forward, loss, backward, and AdamW update, then
compared the canonical states and business metadata.

```text
INITIAL_MODEL_TENSOR_SHA256=7ed5ccc486a173a0b0b4fd28b8be61aedda9ba23372a2fbc82317f6a215e9335
FINAL_MODEL_TENSOR_SHA256=10b95de583daac7f27a8b990f08acbed504fa4b91ee71f823f56e3a94ac359b2
NORMALIZER_SHA256=e50956c535ff7dfc9559a12398ba5eefe6b743a15e4352843a3c3e300b05142e
OPTIMIZER_STATE_SHA256=70b96a2769f41008f458854f57bd788d131eb0479fa800718875b7a1cdef01c6
P_CHECKPOINT_SHA256=3fa18124dcca50cc0d7097103bd40bc07e58251d0457bb89f436ea607416e16a
P_SUMMARY_SHA256=4cc6ec8238c0a1c8b89e367057161e4645effb1229b9ef9f84bb5e61c6e15a5c
```

### M0.2 modified files and tests

Modified only P-side files:

- `python/planning/bc/trainer.py`
- `tests/test_bc_training_observation_provenance.py`
- `tests/compare_bc_training_fixture.py`
- `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`

The existing P observation contract owner from M0.1 was reused; no upstream
producer was changed.

```text
focused pytest: 22 passed
compileall: PASS
BC CLI --help: PASS
deterministic O/P BC comparator: PASS
```

The comparator also reported exact parity for train/validation row indices and
batch order, checkpoint business metadata, and checkpoint reload logits.  The
checkpoint and summary file SHA values are recorded as provenance evidence;
the whole `.pt` file SHA was not used as a parity condition because runtime
metrics and serialization metadata are not business tensors.

## Remaining observation provenance chain after M0.2

- Reliable Teacher collection and its managed launcher remain absent in P.
- P depth-mask and relabel producers remain outside this migration slice;
  M0.1 validates their supplied metadata at the mmap seam.
- Evaluation provenance is unchanged and is the next controlled phase.
- AWAC/RL handoff provenance remains unchanged.

## Next phase

`NEXT_PHASE=M0.3 EVALUATION PROVENANCE`

## M0.3 status

```text
M0_3_EVALUATION_PROVENANCE=PASS
CHECKPOINT_OBSERVATION_VALIDATION=PASS
RUNTIME_OBSERVATION_VALIDATION=PASS
MANAGED_EVALUATOR_HANDOFF=PASS
EVALUATION_SUMMARY_PROVENANCE=PASS
P_CHECKPOINT_EVALUATOR_LOAD=PASS
LOGIT_PARITY=PASS
RAW_ACTION_PARITY=PASS
MASK_PARITY=PASS
MASKED_ACTION_PARITY=PASS
OUTCOME_PARITY=NOT_APPLICABLE
RETURN_PARITY=NOT_APPLICABLE
EVALUATION_SUMMARY_BUSINESS_PARITY=PASS
POLICY_BEHAVIOR_CHANGED=NO
```

M0.3 was limited to the P-side policy evaluation provenance seam.  The P
evaluator now exposes the O-compatible `--expected-observation-contract`
interface, defaults formal evaluation to
`reliable_exact_endpoint_snapshot`, validates checkpoint source/contract and
runtime exact-endpoint binding before constructing Unity, and revalidates the
runtime counters after evaluation.  Missing/unknown provenance, checkpoint or
runtime mismatch, telemetry fallback, non-zero snapshot-missing count, and
non-zero telemetry lookup count fail closed unless the existing explicit audit
runtime override is supplied.  Policy tensor construction, normalization,
masking, temperature, action selection, episode ordering, and reward logic
were not changed.

The evaluation summary now retains both checkpoint and runtime provenance,
including `observation_contract`, `observation_source`, expected/runtime
contracts and sources, override state, reliable/telemetry mode, and runtime
counters.  `rollout_index.csv`, per-step input traces, first-divergence
traces, pairing reports, and `runtime_observation_manifest.json` carry the
same contract association.  The managed launcher now defaults formal runs to
reliable v4, passes the expected contract and all reliable endpoints, and
rejects reused summaries without matching provenance.

### M0.3 fixture and parity evidence

The comparator created a temporary deterministic checkpoint and temporary
motion-primitive fixture; it did not train or collect data and removed both
after the run.  It compared the O CLI helper and P evaluator helper on the
same 90x160 depth frame, 115-dimensional vector input, previous action `-1`,
105-action mask, deterministic argmax, and temperature `0.0`.

```text
FIXTURE_CHECKPOINT_SHA256=e5de053ed643adcf54aef14a0993a4fcbb3127523e0cb065b76948a2acdefa0a
MODEL_STATE_SHA256=2f3368c8c1f7e2a380b7afa635d2fb058b71375cdda66e0727269c8f66d94392
NORMALIZER_SHA256=e292cbd17232542bb194f4e8902f1fd5cab0c1f099e197a3046ea32161593032
DEPTH_TENSOR_SHA256=87e201869ce5c39897e9e326916319e758e6479a33559500761ff543ea01b5a1
VECTOR_TENSOR_SHA256=3383d53764c2bba471eccd9fdfa14a60c7e2c4204634589808e3fbf927c51a38
MASK_SHA256=b1532e0fb821adffb7def2a187da92ee04495e1f52af574f5e63c0f48d381181
RAW_LOGITS_SHA256=04e9bbab91fc5aba9ce0bcaa5b723fe0855577216b3ce366599612ec9c4eac0c
MASKED_LOGITS_SHA256=d0e067287a202283fda7878563a2648a63dc42af87cb57b8555f4dad34b94cad
RAW_ACTION=6
MASKED_ACTION=6
```

Focused provenance and managed tests passed: `15 passed`; existing evaluator
action/terminal tests passed: `7 passed`; reliable runtime and snapshot
provider tests passed: `25 passed`.  `compileall`, shell syntax, and the
O/P fixed-fixture comparator passed.  No real Unity episode was started, so
outcome and return parity remain `NOT_APPLICABLE`; this is not a closed-loop
success-rate result.

### M0.3 modified files

Modified only P-side files:

- `python/planning/contracts/observation.py`
- `python/planning/evaluation/observation_provenance.py`
- `python/planning/evaluation/policy_evaluator.py`
- `scripts/evaluate_policy_unity_managed.sh`
- `tests/test_evaluation_observation_provenance.py`
- `tests/compare_policy_evaluation_fixture.py`
- `tests/test_managed_policy_eval_contract.py`
- `tests/test_evaluator_terminal_abort_contract.py`
- `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`

## Remaining observation provenance chain after M0.3

- No real Unity managed evaluation was run in this phase; closed-loop outcome
  and return parity remain unverified and are not promoted to metrics.
- Reliable Teacher collection, depth-mask producer, relabel producer, and
  their upstream provenance remain outside this migration slice.
- AWAC/RL BC-to-RL handoff provenance remains unchanged.

## Next phase

`NEXT_PHASE=M0.4 AWAC BC-TO-RL HANDOFF PROVENANCE`

## M0.4 status

```text
M0_4_AWAC_HANDOFF_PROVENANCE=PASS
BC_CHECKPOINT_HANDOFF_VALIDATION=PASS
ACTOR_INITIALIZATION_TENSOR_PARITY=PASS
ACTOR_INITIALIZATION_LOGIT_PARITY=PASS
CRITIC_INITIALIZATION_PARITY=PASS
REPLAY_OBSERVATION_PROVENANCE=PASS
AWAC_RUN_CONTRACT_PROVENANCE=PASS
AWAC_CHECKPOINT_PROVENANCE=PASS
AWAC_RESUME_PROVENANCE=PASS
FIRST_UPDATE_METRIC_PARITY=PASS
POST_UPDATE_TENSOR_PARITY=PASS
AWAC_MATH_CHANGED=NO
REPLAY_LAYOUT_CHANGED=NO
```

M0.4 was limited to the P-side BC-to-AWAC handoff.  The formal AWAC path now
fails closed unless BC, runtime, and replay all declare the exact
`reliable_exact_endpoint_snapshot` contract and the reliable execution source.
The handoff validates the feature/policy input contract, task and motion
primitive identities, depth history, previous-action encoding, action count and
ordering, normalizer shape/finite values, strict BC Actor state loading, and
zero legacy rows.  Run contract, checkpoint, summary, replay metadata, and
resume validation carry the BC SHA, observation provenance, feature/task/MPL
identities, resolved-config identity, run identity, and reliable/legacy counts.

The replay arrays and learner math were not changed.  The P learner uses an
explicit strict Actor state loader; the existing SAC/AWAC update equations,
optimizer groups, schedules, target updates, and checkpoint learner/RNG/step
state remain unchanged.  The historical protocol-v4 implementation name is
retained only as an internal compatibility symbol; formal business metadata
uses the canonical exact-endpoint contract.

### M0.4 modified files

Modified only P-side files:

- `python/planning/contracts/awac_handoff.py`
- `python/planning/awac/learner.py`
- `python/planning/rl/sac_model.py`
- `python/planning/rl/sac_replay.py`
- `python/planning/rl/train_sac.py`
- `python/planning/runtime/reliable_training.py`
- `python/planning/runtime/worker.py`
- `tests/test_awac_handoff_provenance.py`
- `tests/test_reliable_v4_awac_training_contract.py`
- `tests/test_sac_core.py`
- `tests/compare_awac_handoff_fixture.py`
- `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`

### M0.4 tests and deterministic fixture evidence

All commands used `conda activate xm`.  No Unity process, formal RL schedule,
or data collection was started.

```text
compileall: PASS
handoff/replay/SAC/AWAC focused regression: 184 passed
interruption contract: 1 passed
observation contracts: 12 passed
reliable worker/lifecycle contracts: 16 passed
evaluation provenance contracts: 15 passed
O/P CPU handoff comparator: PASS
```

The comparator used one temporary BC checkpoint, four deterministic replay
rows, CPU-only initialization with fixed seeds, one learner update, and
checkpoint save/load.  It reported:

```text
BC_FIXTURE_SHA256=fc61d925f9edd2835b814a44654754261692e1c5991bb17bf9935e4374eca799
INITIAL_ACTOR_STATE_SHA256=2f71f7219cb65de2df22b20a14eca41865dcb69287bb0ca7a4302a8e8a58faaf
INITIAL_CRITIC_STATE_SHA256=ed4899bdf2eda536630f34570b1baa4c507e938bc54e2271a62707c4d0ec3a80
INITIAL_LOGITS_SHA256=1153d44870567a0b669df3bb0c9c58016b4f45ce3f10ab055707a5ef8b3b5196
REPLAY_ARRAYS_SHA256=6dfc0132f41cf8b524a10bd14b289b6df2d0caf62b016d2de94c59a6e542c04d
POST_UPDATE_STATE_SHA256=84c7537fa95268472dde5f25ecc7d17a90f8bcf295f622c556273953a033bafc
POST_UPDATE_OPTIMIZER_SHA256=4f7dfa17dc296f99c8e31c045fd963952f7894907ba4dba0c4412096d2f04fdf
BC_CHECKPOINT_HANDOFF_VALIDATION=PASS
ACTOR_INITIALIZATION_TENSOR_PARITY=PASS
ACTOR_INITIALIZATION_LOGIT_PARITY=PASS
ACTOR_INITIALIZATION_ACTION_PARITY=PASS
CRITIC_INITIALIZATION_PARITY=PASS
REPLAY_OBSERVATION_PROVENANCE=PASS
FIRST_UPDATE_METRIC_PARITY=PASS
POST_UPDATE_TENSOR_PARITY=PASS
AWAC_CHECKPOINT_BUSINESS_METADATA_PARITY=PASS
AWAC_RESUME_PROVENANCE=PASS
```

The comparator's replay-array hashes include the existing depth, vector,
previous-action, mask, action, reward, next-state, and candidate provenance
arrays; no fields, dtypes, shapes, or row order were added or removed.

## Remaining blockers after M0.4

- No real Unity managed AWAC run was executed; closed-loop runtime outcome and
  return parity remain `NOT_RUNTIME_VERIFIABLE`.
- P's reliable multi-worker Teacher collection and upstream depth-mask/relabel
  provenance remain outside this controlled handoff slice.
- Environment-gated tests that import `rospy` could not collect because this
  shell has no ROS Python module.  Guarded-runner tests also require the absent
  `/home/xm/devel/setup.bash` and an optimized-tree guard-audit path; these are
  retained as pre-existing optimized-tree blockers and were not changed here.

## Next phase

`NEXT_PHASE=M0.5 RELIABLE-EXACT MULTI-WORKER TEACHER COLLECTION`

## C++ Foundation Controlled Migration: C0 + C1 (2026-08-27)

Old M0.4 is paused.  This is a separate controlled migration with
`/home/xm/XM/xm_ws/src/planning` as the read-only behavior baseline (O) and
`/home/xm/XM/src` as the only candidate modification tree (P).  No Unity,
Python algorithm, BC/AWAC math, Bridge implementation, formal data, or commit
was touched.

### C0 inventory and build

The C++ map is recorded in `docs/CXX_FOUNDATION_MIGRATION.md`.  In summary:

- `package.xml` is byte-identical between O and P.
- P adds the `geometry/VoxelMap`, `navigation/GlobalRoutePlanner`, and split
  Bridge/protocol/transport ownership; O keeps those concerns in the
  collision checker or monolithic bridge.
- P retains root compatibility headers and the existing CPU collision C ABI.
- O has `planning_collision_checker_cuda`; P has no CUDA target.  This remains
  blocked and was not removed or replaced.
- O C++ Release compile/link passed.  Its catkin install wrapper is blocked by
  the xm conda setuptools error `option --install-layout not recognized`.
- P CMake configure and geometry/fixture targets passed.  The full P build is
  blocked by pre-existing Bridge/protocol compile errors (`ROS_*` macros and
  `AsInt32`), which are outside C0/C1 scope.
- P's CMake install of generated `data/motion_primitives` is now guarded by
  an existence check.  The absent directory is reported and skipped; no data
  was created.

### C1 immutable shared VoxelMap

P's existing `std::shared_ptr<const planning::geometry::VoxelMap>` seam was
validated against the current O geometry.  The fixed fixture uses the same
deterministic occupied-key input for O and P, 1000 random points, 14 boundary
points, metadata, occupied-key hash, coordinate conversions, and direct O/P
CPU collision calls.  It reports:

```text
METADATA_PARITY=PASS
OCCUPANCY_PARITY=PASS
COORDINATE_CONVERSION_PARITY=PASS
ORIGINAL_COLLISION_RUNTIME=PASS
SHARED_OWNER_LIFETIME=PASS
RESULT=PASS
```

The ownership test creates CollisionChecker and GlobalRoutePlanner from one
map handle, destroys the map handle, then queries both successfully.  The
planner's derived 2-D blocked grid is not a duplicate complete 3-D
occupancy.  The actual `forest_voxels_10cm.npz` cache is absent from both O
and P; the raw O point cloud was not converted, copied, or treated as formal
data, so real forest-cache parity remains `NOT_RUN`.

### C0/C1 modified files and tests

Modified only P:

- `CMakeLists.txt`
- `tests/cpp/voxel_map_parity_fixture.cpp`
- `docs/CXX_FOUNDATION_MIGRATION.md`
- this progress document

The focused C++ fixture target and direct comparator passed under
`conda activate xm`.  No P VoxelMap/CollisionChecker/GlobalRoute/Bridge
implementation was rewritten; this slice fixed the missing generated-data
install handling and added the minimal parity/ownership fixture.

### Blockers and next order

Still blocked: the real forest voxel cache, P full Bridge/protocol build, O/P
catkin install under the current conda setuptools combination, CUDA parity,
and all unapproved behavior changes.  The subsequent controlled order is:

`C2 COLLISION BACKEND PARITY` → `C3 DEPTH SAFETY PARITY` →
`C4 GLOBAL ROUTE OWNER/API PARITY` → `C5 BRIDGE/PROTOCOL/TRANSPORT PARITY` →
`C6 INSTALL/PACKAGE/RUNTIME CLOSURE`.

`NEXT_PHASE=C2 COLLISION BACKEND PARITY`

## C2 Collision Backend Parity (2026-08-27)

The old M0.4 line remains paused.  C2 used O
(`/home/xm/XM/xm_ws/src/planning`) as a read-only collision behavior baseline
and P (`/home/xm/XM/src`) as the only candidate tree.  No O file, Unity file,
Python algorithm, Bridge/protocol source, Global A*, CUDA source, formal data,
or commit was changed.

### Result

```text
C2_COLLISION_PARITY=PARTIAL
O_CPU_AVAILABLE=YES
O_CUDA_AVAILABLE=NO
O_HYBRID_AVAILABLE=NO
P_CPP_AVAILABLE=YES
PATH_COLLISION_PARITY=PASS
SINGLE_ACTION_PARITY=PASS
ACTIONS_BATCH_PARITY=PASS
POSE_ACTIONS_BATCH_PARITY=PASS
FIRST_COLLISION_INDEX_PARITY=PASS
ENDPOINT_PARITY=PASS
MIN_DISTANCE_PARITY=PASS
BATCH_ORDER_PARITY=PASS
CANDIDATE_ACCEPT_VECTOR_PARITY=NOT_RUN
TEACHER_VALID_ACTION_MASK_PARITY=NOT_RUN
SHARED_VOXELMAP_RUNTIME_OWNERSHIP=PASS
P_CPP_COLLISION_PRODUCTION_CANDIDATE=NO
PRODUCTION_ASSET_PARITY=PASS
ALGORITHM_CHANGED=NO
```

O/P CPU C ABI parity was exact for path, single action, actions batch, and
pose x actions batch.  Discrete outputs and float buffers were compared
bitwise first; the maximum observed absolute and relative deltas were both
zero.  The comparator covered 121 cases, radii `0.35` and `0.40`, check steps
1/2/3, boundary/dense/empty layouts, collision positions, and batch order.

O CUDA compiled but was not runtime-available because the host NVIDIA driver
could not be contacted; O hybrid consequently remained unavailable.  P has no
CUDA or hybrid backend.  These states are reported as unavailable rather than
treated as parity passes.

### Production-like asset and ownership evidence

O's raw forest point cloud was used only to build a temporary cache under
`/tmp/xmflight_collision_c2/` with O's formal generator/builder.  The temporary
cache has voxel size `0.10`, origin `[-1023,-1020,-1]`, shape
`[2046,2045,42]`, and `1,740,947` occupied keys.  Cache SHA256:
`a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691`.
No generated file was written to either project's `data/` directory.

The shared-owner fixture observed one logical map load/occupied-buffer owner,
two P collision checker consumers with different radii, successful queries
after external map-handle destruction, and zero complete occupancy copies in
consumers.  The C1 fixture also verified the same shared map lifetime for
GlobalRoutePlanner.

### Performance gate

The Release benchmark ran three times per backend at OpenMP 1/2/4 on the
fixed `c1_sparse` input.  Median poses x actions throughput was:

```text
threads   O_CPU     P_CPP_SHARED_VOXELMAP
1         503187    442518
2         529478    491280
4         446073    397903
```

P was below O at all tested settings, so
`P_CPP_COLLISION_PRODUCTION_CANDIDATE=NO`.  Full mean/median/min/max
throughput, p50/p95 batch latency, process CPU utilization, peak RSS, and
initialization timings are in
`/tmp/xmflight_collision_c2/collision_backend_benchmark.json`.

### Modified files and tests

Modified in C2:

- `tests/compare_collision_backend_parity.py`
- `docs/COLLISION_BACKEND_PARITY.md`
- `docs/CXX_FOUNDATION_MIGRATION.md`
- `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`

Tests and artifacts:

- O/P C++ collision comparator: PASS; production-like cache parity: PASS.
- C1 `voxel_map_parity_fixture`: PASS with 1000 random and 14 boundary
  points; shared owner lifetime: PASS.
- C2 parity script `compileall` with `PYTHONPYCACHEPREFIX` under `/tmp`: PASS.
- O CUDA runtime probe: `NOT_AVAILABLE` (no driver/device).
- Benchmark: PASS, 18 raw rows, 3 runs x 2 backends x 3 thread counts.
- Parity script SHA256:
  `4da74f44bb21ca65d58986467b5efb8eddb187d43325f78d3952964420da0823`.

### Remaining blockers and next phase

Blocked: O CUDA runtime parity, O hybrid scheduler parity, and P C++ runtime
throughput parity.  Candidate accept-vector parity and Teacher valid-action
mask parity were intentionally not run because C2 is restricted to the
collision backend foundation.  No cutover, CUDA deletion, or C2.1 action was
started.

## Native library path resolution finalization V1 — 2026-09-04

Native shared-library path resolution is now owned by
`planning.native.loader`. The formal automatic search is limited to the
project-derived workspace `devel/lib`; explicit library environment variables
remain optional advanced overrides. The detailed report and machine-readable
manifest are in
[`NATIVE_LIBRARY_PATH_RESOLUTION_FINALIZATION_V1.md`](NATIVE_LIBRARY_PATH_RESOLUTION_FINALIZATION_V1.md)
and
[`native_library_path_resolution_finalization_v1.json`](native_library_path_resolution_finalization_v1.json).

```text
NATIVE_LIBRARY_PATH_RESOLUTION_FINALIZATION_V1=FAIL_ENVIRONMENT_BLOCKED
NATIVE_LIBRARY_PATH_IMPLEMENTATION=PASS
CANONICAL_NATIVE_PATH_OWNER=planning.native.loader
NATIVE_LIBRARY_PATH_OWNER_COUNT=1
WORKSPACE_AUTO_DETECTION=PASS
AUTO_RESOLVE_VOXEL_MAP=PASS
AUTO_RESOLVE_COLLISION=PASS
AUTO_RESOLVE_GLOBAL_ROUTE=PASS
AUTO_RESOLVE_DEPTH_SAFETY=PASS
ENV_OVERRIDE_SUPPORTED=YES
ENV_OVERRIDE_REQUIRED=NO
DEPTH_SAFETY_TEST_ENV_REQUIRED=NO
INSTALL_TREE_NATIVE_FALLBACK=NO
DIRECT_NATIVE_PATH_RESOLVER_DUPLICATE_COUNT=0
FOCUSED_NATIVE_TESTS=58 passed, 0 failed
FULL_TESTS=697 passed, 42 skipped, 8 failed
FULL_TEST_FAILURE_CLASS=ENVIRONMENT_SOCKET_PERMISSION
VOXEL_MAP_NATIVE_LIBRARY_MISSING_FAILURES=0
DEPTH_SAFETY_NATIVE_UNAVAILABLE_FAILURES=0
AWAC_CHANGED=NO
BC_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
REWARD_CONTRACT_CHANGED=NO
COMMIT=NO
NEXT_ACTION=RERUN_UNFILTERED_FULL_PYTEST_WITH_LOOPBACK_PERMISSION
```

The eight full-suite failures are the pre-existing startup and single-worker
socket tests, each blocked by the managed environment before behavior
assertions. The focused native suite and all six formal CLI help checks pass
with every `PLANNING_*_LIBRARY` variable unset. No C++, native ABI, Unity,
Bridge, AWAC, BC, Teacher, MPL, task, observation, reward, or trajectory
semantics changed.

`NEXT_PHASE=C2 COLLISION FIX / INVESTIGATION`

## C2.1 Collision hot-path regression investigation (2026-08-27)

The historical C2 section above is retained as the pre-investigation record.
C2.1 is the current controlled-migration result and supersedes its CPU
throughput status.  M0.4 remains paused; no C2.2 production cutover has been
performed.

### Status

```text
C2_1_COLLISION_REGRESSION=PASS
BUILD_FLAGS_PARITY=PASS
HOT_LOOP_DIFFERENCE=EXTRA_WORK_IN_P before fix; removed after fix
DIFFERENT_ALGORITHM=0
EXTRA_ALLOCATIONS_P=0 (steady state; same as O)
P_CPP_COLLISION_PRODUCTION_CANDIDATE=YES
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
```

The official O and P builds both use Release `-O3 -DNDEBUG -fPIC -fopenmp`;
the controlled benchmark rebuilt the unchanged O collision source with C++17
to remove the official C++14/C++17 standard mismatch from the performance
comparison.  P retains shared immutable `VoxelMap` ownership while using the
O-verified direct bitmap hot loop.  Numeric parity remained exact for 121
cases (`max_abs_delta=0`, `max_relative_delta=0`).

### Controlled evidence

- Kernel benchmark: OMP 1/2/4, three runs per backend, controlled C++17
  Release.  Best median was P `531508` versus O `516287` poses x actions/sec
  (`+2.95%`).
- Fixed 1000-candidate end-to-end benchmark: best median was P `9204120`
  versus O `7835947` candidates/sec (`+17.46%`); P init `0.004175 ms`, O
  init `0.004204 ms`; P/O map load count was `1`.
- C++ probe peak RSS at the best setting was P `3.145 MB` and O `3.094 MB`.
  The Python-wrapper measurement was `103.578 MB` for both.
- Candidate accept-vector parity: `100/100` exact.  Teacher valid-action-mask
  parity: `100/100` exact, all 105 actions.
- C1 metadata, occupancy, coordinate-conversion, and shared-owner lifetime
  fixture: PASS.  Focused collision/action-mask/Teacher tests and O/P
  `compileall`: PASS.
- `perf stat`/`perf record` were unavailable in this environment; cycles,
  instructions, IPC, branch/cache counters, and function-level samples are
  therefore not claimed.

Artifacts are kept under `/tmp/xmflight_collision_c2/`, including the final
numeric parity log, controlled C++17 kernel benchmark, fixed end-to-end logs,
allocation probe, and integration fixture.  The integration fixture result
SHA256 is `e1cc853c91c07501b93a8f32e1a0720d5272272ae02b73de54a99f98089d6d8f`.

### Modified files

- `include/planning/geometry/voxel_map.hpp`
- `src/geometry/collision_checker.cpp`
- `tests/compare_collision_teacher_mask_fixture.py`
- `docs/COLLISION_BACKEND_PARITY.md`
- `docs/CXX_FOUNDATION_MIGRATION.md`
- `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`

No O files, Unity files, Bridge, Global A*, Teacher/Mission algorithms,
protocol, BC/AWAC code, or CUDA/hybrid code were changed.  No formal data was
collected and no commit was created.

### Remaining blockers and next phase

GPU/hybrid runtime validation remains unavailable and is explicitly deferred;
the CUDA/hybrid code is retained.  The prior full P Bridge/protocol build
blocker remains out of scope for C2.1.  C2.2 may be considered only as the
next authorized review step; it was not executed here.

`NEXT_PHASE=C2.2 PRODUCTION CUTOVER`

### C2.2 — Collision production cutover (2026-08-27)

The prior C2.1 candidate gate is historical. The explicitly authorized
P-only production change is complete:

```text
C2_2_PRODUCTION_CHANGE_AUTHORIZED=YES
C2_2_COLLISION_PRODUCTION_CUTOVER=PASS
FORMAL_COLLISION_BACKEND=C++17_OPENMP_CPU
PRODUCTION_BACKEND_UNIQUE=YES
COLLISION_NUMERIC_PARITY=PASS
CANDIDATE_ACCEPT_VECTOR_PARITY=PASS
TEACHER_VALID_ACTION_MASK_PARITY=PASS
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

Scope was limited to the self-owned collision backend. Production aliases
resolve to the C++17/OpenMP library; native initialization and operation
failures are fail-closed; Python reference execution requires explicit
test/debug opt-in. No P files were deleted. P's pre-existing absence of
formal collision CUDA/hybrid sources and targets is recorded as
`OUT_OF_FORMAL_SCOPE`, not `PROVEN_INCORRECT`. BC/AWAC PyTorch CUDA, AMP, and
GradScaler paths remain untouched.

Parity evidence is 121/121 exact numeric cases, 100/100 exact candidate
accept rows, and 100/100 exact 105-action Teacher masks. The controlled
Release benchmark recorded:

```text
P_KERNEL_THROUGHPUT=531508 poses x actions/sec
O_KERNEL_THROUGHPUT=516287 poses x actions/sec
KERNEL_DELTA_PERCENT=+2.95%
P_END_TO_END_THROUGHPUT=9204120 candidates/sec
O_END_TO_END_THROUGHPUT=7835947 candidates/sec
END_TO_END_DELTA_PERCENT=+17.46%
```

The collision target builds successfully. The full P build remains blocked
by the pre-existing Bridge/protocol `ROS_WARN*` declarations after collision
targets; Bridge repair is out of scope. `perf` was unavailable, so hardware
counter results are unknown. No C3 work, Python shared-VoxelMap binding,
formal data collection, or commit was performed.

`NEXT_PHASE=C3 GLOBAL A* SEMANTICS AND PARITY`

## C3 — Global A* semantics and parity (2026-08-27)

```text
C3_GLOBAL_ASTAR_PARITY=PASS
STRUCTURE_CHANGED=YES
ROUTE_BEHAVIOR_CHANGED=NO
O_PYTHON_ROUTE_AVAILABLE=YES
P_CPP_ROUTE_AVAILABLE=YES
CORNER_CUTTING_PARITY=PASS
TIE_ROUTE_SEQUENCE_PARITY=PASS
ROUTE_FOUND_PARITY=PASS
ROUTE_POINT_COUNT_PARITY=PASS
ROUTE_GRID_SEQUENCE_PARITY=PASS
ROUTE_WORLD_SEQUENCE_PARITY=PASS
ROUTE_COST_PARITY=PASS
ROUTE_LENGTH_PARITY=PASS
WORKSPACE_STATE_LEAK=0
MISSION_ROUTE_ACCEPT_VECTOR_PARITY=PASS
MISSION_ACCEPTED_ID_ORDER_PARITY=PASS
TEACHER_ROUTE_INPUT_PARITY=PASS
TEACHER_ACTION_PARITY=PASS
TEACHER_VALID_ACTION_MASK_PARITY=PASS
P_CPP_GLOBAL_ROUTE_PRODUCTION_CANDIDATE=YES
ALGORITHM_CHANGED=NO
```

O was the read-only behavior source of truth. P's native C++17 route target
and Python wrapper matched the O route contract on 127 synthetic cases and
1000 production-like cases, including metadata/conversions, corner blocking,
tie sequence, no-route behavior, point count, grid/world sequence, cost, and
length. The initial P intermediate-z representation mismatch was fixed only
at the Python ABI output seam; no A* algorithm, XY calculation, or route
ordering was changed.

The mission/Teacher comparator reported 1000/1000 exact route acceptance and
accepted-ID order, followed by 100/100 exact route inputs, actions, and
105-action valid masks. Cross-stage A* call counts remain Generate 1, Audit 1,
Collection 1 (collection prefetch reuses the route), and Relabel 1. No
RouteStore or cross-stage dedupe was introduced. Python shared-VoxelMap owner
unification remains a C5 item.

The three-run route benchmark at size 1000 recorded:

```text
P_CPP_ROUTE_THROUGHPUT=2152.737649 routes/sec
O_PYTHON_ROUTE_THROUGHPUT=47.200157 routes/sec
ROUTE_SPEEDUP=45.608697x
P_INIT_TIME=0.125439 s mean
O_INIT_TIME=0.059851 s mean
P_PEAK_RSS=380592 KB
O_PEAK_RSS=379456 KB
```

P's route target build passed. The whole P build remains blocked by the
pre-existing Bridge/protocol `ROS_WARN*` compile errors; no Bridge repair was
attempted. The route-specific candidate gate is YES, but C3.1 production
cutover was not run.

Modified/added P-side C3 evidence includes the native route implementation
and header, the explicit native/reference Python wrapper seam, the empty-map
route fixture, route parity/mission-Teacher comparator/benchmark tests, and
this progress record. Detailed files, fixture categories/hashes, build
status, workspace ownership, and temporary artifact paths are recorded in
`docs/GLOBAL_ROUTE_PARITY.md` and the CXX foundation log.

`NEXT_PHASE=C3.1 GLOBAL A* PRODUCTION CUTOVER`

## C3.1 — Global A* production cutover (2026-08-27)

```text
C3_1_GLOBAL_ASTAR_PRODUCTION_CUTOVER=PASS
FORMAL_GLOBAL_ROUTE_BACKEND=cpp_native
PRODUCTION_ROUTE_BACKEND_UNIQUE=YES
PYTHON_PRODUCTION_FALLBACK=NO
NATIVE_ROUTE_FAIL_CLOSED=PASS
GLOBAL_ROUTE_PARITY=PASS
MISSION_ROUTE_ACCEPT_VECTOR_PARITY=PASS
MISSION_ACCEPTED_ID_ORDER_PARITY=PASS
TEACHER_VALID_ACTION_MASK_PARITY=PASS
COLLISION_REGRESSION=PASS
P_GLOBAL_ROUTE_TARGET_BUILD=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
ALGORITHM_CHANGED=NO
```

O remained the read-only route behavior baseline and P was the only modified
tree. The formal route pipeline now resolves to the C++17 native backend for
Generate, Audit, Collection, and Relabel. Python reference use requires an
explicit test/debug opt-in; production native failures do not fall back to
Python. Existing route output, row order, no-path semantics, and Mission CSV
schema are unchanged. Route contract/config/source and VoxelMap/cache
provenance are retained in stage metadata/manifests.

Rerun evidence is exact: 127 synthetic routes, 1000 production-like routes,
1000 mission acceptance/order rows, 100 Teacher route/action/mask rows with
105 actions, and C2 collision gates at 121 numeric cases plus 100 candidate
and Teacher-mask rows. The three-run size-1000 benchmark is P
`2150.730608 routes/s` versus O `47.492813 routes/s` (`45.285391x`); P/O
initialization means are `0.065384/0.059457 s`, and peak RSS is
`380528/379520 KB`.

The native route target builds and shared C++ VoxelMap ownership passes. The
full P build remains blocked by the pre-existing Bridge/protocol `ROS_WARN*`
errors; Bridge was not changed. Python shared-VoxelMap binding is
`NOT_STARTED`, CUDA collision validation is `DEFERRED / NOT_AVAILABLE`, and
RouteStore remains `NOT_STARTED`. No formal data was collected and C4 was not
executed. Full evidence is in
`docs/GLOBAL_ROUTE_PRODUCTION_CUTOVER.md`.

`NEXT_PHASE=C4 DEPTH SAFETY PARITY AND OWNERSHIP`

## C4 — Depth safety parity and ownership (2026-08-27)

```text
C4_DEPTH_SAFETY_PARITY=PASS
O_DEPTH_SAFETY_AVAILABLE=YES
P_DEPTH_SAFETY_AVAILABLE=YES
DEPTH_SAFETY_OWNER_UNIQUE=YES
MASK_BIT_PARITY=PASS
MIN_CLEARANCE_PARITY=PASS
VALID_COUNT_PARITY=PASS
EMPTY_MASK_PARITY=PASS
ACTION_ORDER_PARITY=PASS
PRODUCTION_LIKE_DEPTH_PARITY=PASS
DEPTH_MASK_GENERATOR_PARITY=PASS
POLICY_MASKED_ACTION_PARITY=PASS
TEACHER_DEPTH_SAFETY_PARITY=PASS
P_DEPTH_SAFETY_PRODUCTION_CANDIDATE=YES
NATIVE_DEPTH_SAFETY_FAIL_CLOSED=PASS
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
DEPTH_MASK_ARTIFACT_PROVENANCE=NOT_HANDLED_IN_C4
ALGORITHM_CHANGED=NO
```

C4 closed the depth-observation-to-action-safety numeric seam.  O remained
read-only.  P's native C++ depth implementation is the formal owner and
Python reference mode is explicit debug/test only; missing native library,
symbol, or execution failure is fail-closed.  No numeric geometry, output
schema, action order, Unity behavior, Teacher scoring, Mission semantics,
Bridge, BC/AWAC, or observation provenance producer was changed.

Evidence is exact across 18 synthetic edge cases, 1000 deterministic
production-like frames, 100 mask-generator transitions, 100 policy mask
rows, and 100 Teacher safety observations.  P/O three-run depth throughput is
`272496.888020/259712.147181 actions/sec` (`+4.922658%`), with equal peak RSS
`150844 KB`.  Full evidence is in `docs/DEPTH_SAFETY_PARITY.md` and
`/tmp/xmflight_depth_safety_c4/depth_safety_parity.json`.  Independent depth,
collision, and route targets pass; full P build remains blocked by the
pre-existing Bridge `ROS_WARN*` declarations.  No formal data was collected
and no commit was created.

`NEXT_PHASE=C4.1 DEPTH SAFETY PRODUCTION CUTOVER`

## C4.1 — Depth Safety production cutover (2026-08-27)

This phase is complete as an authorized P-only production cutover. O remained
read-only; no Unity, Bridge/protocol, Mission, Teacher scoring, BC/AWAC,
formal data, or depth-mask artifact provenance producer was changed.

~~~
C4_1_DEPTH_SAFETY_PRODUCTION_CUTOVER=PASS
FORMAL_DEPTH_SAFETY_OWNER=planning.safety.depth_safety.local_depth_action_mask
PRODUCTION_IMPLEMENTATION_COUNT=1
PYTHON_PRODUCTION_FALLBACK=NO
NATIVE_DEPTH_SAFETY_FAIL_CLOSED=PASS
DEPTH_SAFETY_OWNER_UNIQUE=YES
DEPTH_CONFIG_OWNER_UNIQUE=YES
MASK_BIT_PARITY=PASS
DEPTH_MASK_GENERATOR_PARITY=PASS
POLICY_MASKED_ACTION_PARITY=PASS
TEACHER_DEPTH_SAFETY_PARITY=PASS
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
P_DEPTH_SAFETY_TARGET_BUILD=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
DEPTH_MASK_ARTIFACT_PROVENANCE=NOT_HANDLED
ALGORITHM_CHANGED=NO
NEXT_PHASE=C5 PYTHON SHARED VOXELMAP AND NATIVE BINDING
~~~

The native planning_depth_safety target is the sole production algorithm
implementation behind the canonical Python seam. Python reference execution
requires explicit debug/test opt-in and native failure never falls back to
it. The exact 18-frame and 1000-frame depth fixtures, 100-transition
generator fixture, 100-row policy/Teacher fixture, collision/route frozen
regressions, three-run benchmark, package identity, caller audit, and
deletion ledger are recorded in
docs/DEPTH_SAFETY_PRODUCTION_CUTOVER.md.

P depth throughput is 270011.951375 actions/sec versus O 259187.179987
(+4.176430%). The native target build passes. The full P build remains blocked
at unchanged Bridge/protocol ROS_WARN* declarations; full install is
separately blocked by the conda setup.py --install-layout=deb compatibility
failure. Artifact provenance remains NOT_HANDLED. C5 was not started.

## C5 — Python shared VoxelMap and native binding (2026-08-27)

The C5 Python ownership seam is complete on P. O and Unity remained
unchanged. `NativeGeometryContext` reads and validates one existing voxel
artifact, owns one `planning_voxel_map_create` handle, and caches the public
collision and route consumers over that handle. It includes artifact path and
SHA256, voxel size/origin/shape, and route configuration in its identity.
Different worker processes create independent contexts; raw handles are never
sent through multiprocessing. Closed contexts, missing libraries/symbols,
invalid cache identity, invalid dtype/shape, and fork reuse fail closed.

~~~
C5_PYTHON_NATIVE_BINDING=PASS
BINDING_IMPLEMENTATION=CTYPES
PYBIND11_ADOPTED=NO
PYTHON_SHARED_VOXELMAP_OWNERSHIP=PASS
PYTHON_VOXELMAP_LOADS_BEFORE=2 (paired collision+route context fixture)
PYTHON_VOXELMAP_LOADS_AFTER=1 (paired collision+route context fixture)
PYTHON_OCCUPANCY_ALLOCATIONS_BEFORE=2
PYTHON_OCCUPANCY_ALLOCATIONS_AFTER=1
COLLISION_ROUTE_SHARED_MAP_ID=PASS
NATIVE_FAIL_CLOSED=PASS
PYTHON_PRODUCTION_FALLBACK_COUNT=0
NUMPY_INPUT_COPY_COUNT=0
NUMPY_CONTIGUITY_CONVERSION_COUNT=0
NUMPY_DTYPE_CONVERSION_COUNT=2000 (1000 route pairs, sequence inputs)
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
DEPTH_SAFETY_REGRESSION=PASS
P_NATIVE_TARGETS_BUILD=PASS
P_PYTHON_NATIVE_IMPORT=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
ALGORITHM_CHANGED=NO
NEXT_PHASE=C6 BRIDGE PROTOCOL TRANSPORT
~~~

The final three-run paired benchmark used the fixed cache
`/tmp/xmflight_collision_c2/forest_voxels_10cm.npz`, 1000 candidate rows, and
pair SHA256
`225bb2f8d8e5af9acd7b0c2bae05db69d082c8759242d5866f923208ef67acd4`. Direct
versus shared best collision throughput was `48258.601195` versus
`48083.597432` poses x actions/sec (`-0.362637%`); route throughput was
`2130.787380` versus `2124.313410` routes/sec (`-0.303830%`). Max RSS was
`382216 KB` versus `382608 KB`. Shared context output allocation audit was
`2009` arrays/run, all expected native output/ABI buffers; no large input copy
or contiguity conversion occurred. Synthetic mission generation ran three
times per mode with identical result digests and mean wall times
`0.018192552 s` direct versus `0.018280509 s` shared.
A fair fresh-process synthetic Mission-audit initialization benchmark measured
mean `0.046008488 s` direct versus `0.038229823 s` shared (`-16.907022%`).
The benchmark artifacts are
`/tmp/xmflight_python_binding_c5_mission_benchmark.json` and
`/tmp/xmflight_python_binding_c5_audit_benchmark.json`.

Current parity evidence remains exact: collision 121 cases and production
asset, Global A* 127 synthetic plus 1000 production-like, mission
accept/order 1000, Teacher 100, depth synthetic/production-like/generator/
policy/Teacher, and C++ VoxelMap metadata/occupancy/coordinate/lifetime
fixtures all pass. The installed system-Python import and a real installed
context pass; conda `setup.py --install-layout=deb` remains an environment
install blocker. pybind11 was not available, so no extension adoption or
200-line wrapper reduction is claimed. The C++ current OMP benchmark remains
separate evidence; no CUDA collision parity is claimed.

Modified P files are the new native owner, collision/route adapter and
Mission/Teacher wiring, owner tests/benchmark, and this documentation. No
files were deleted, no formal data was generated, and no commit was created.
The unchanged Bridge full-build blocker and the one-time strict-identity
initialization cost remain recorded. C6 Bridge protocol transport was not
executed before this controlled slice.


## C6 status — Bridge, protocol, and transport (2026-08-28)

C6 is a P-only controlled parity audit. O and Unity remained read-only. The
Bridge split is compile-closed and normal-path parity is green, but cutover is
blocked by a known direct-port default drift and two unclosed fault/timing
characterizations.

~~~text
C6_BRIDGE_PROTOCOL_TRANSPORT=PARTIAL
COMPILE_REPAIR_SCOPE_EXPANDED=NO
P_FULL_BUILD=PASS
BRIDGE_COMPILE=PASS
MISSING_BEHAVIOR_COUNT=0
DUPLICATE_BEHAVIOR_COUNT=0
PROTOCOL_CONSTANT_PARITY=PASS
CANONICAL_BYTES_PARITY=PASS
HASH_PARITY=PASS
WIRE_CODEC_PARITY=PASS
TRANSPORT_ONLY_PARITY=PASS
COMMAND_GATEWAY_PARITY=PASS
RESULT_GATEWAY_PARITY=PASS
SNAPSHOT_GATEWAY_PARITY=PASS
RESET_GATEWAY_PARITY=PASS
TELEMETRY_PARITY=PASS
REAL_UNITY_RUNTIME_PARITY=PASS
PHYSICS_PARITY=PASS
DEPTH_PARITY=PASS
COLLISION_PARITY=PASS
RUNTIME_IDENTITY_PARITY=PASS
PROCESS_CLEANUP=PASS
MANAGED_PORT_PARITY=PASS
DIRECT_DEFAULT_PARITY=FAIL
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE=NO
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=C6.1 BRIDGE PRODUCTION CUTOVER
~~~

### C6 modified files

- src/bridge/result_gateway.cpp
- src/bridge/snapshot_gateway.cpp
- src/protocol/telemetry_wire.cpp
- tests/test_primitive_execution_protocol_contract.py
- tests/test_unity_bridge_result_runtime_integration.py
- tests/test_primitive_execution_v4_csharp_contract.py
- tests/test_primitive_reset_v4_csharp_contract.py
- tests/test_c6_bridge_protocol_transport.py
- docs/BRIDGE_PROTOCOL_TRANSPORT_PARITY.md
- docs/CXX_FOUNDATION_MIGRATION.md
- docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md

The three C++ source changes restore ROS logging/type declarations only. The
test changes point at P's canonical protocol headers, permit shared historical
metric fields, point C# contracts at the existing read-only Unity
Runtime/Protocol files, and add the minimum C6 ownership/port/stress
characterization. No Unity file, C++ algorithm, wire schema, Python
algorithm, formal data, or training artifact changed.

### C6 evidence

- O and P full Release CMake builds passed under conda activate xm.
- P focused protocol/gateway/transport suite: 99 passed, 1 deselected;
  compileall passed.
- P real-process Bridge integration: 24 passed.
- O/P 1000-transaction transport-only stress: exact counters, zero
  cross-talk, zero pending, one intentional malformed-packet error.
- The same frozen Player SHA256
  61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365 passed
  O/P live 20-primitive, 25/25-frame reset/ACK/commit/snapshot/depth/physics
  checks with equal retry windows and equal metrics.

The detailed inventory, ownership table, build flags, target hashes, endpoint
map, golden-vector hashes, runtime artifact paths, and remaining blockers are
in docs/BRIDGE_PROTOCOL_TRANSPORT_PARITY.md. No C6.1 cutover or C7 work was
started.

## C6.0.1 status — Bridge contract closure (2026-08-28)

This controlled slice made `python/planning/runtime/ports.py` the sole P
production endpoint owner. Managed workers use `WorkerRuntimeSpec`; direct
smokes require `DirectRuntimePortProfile`. The explicit direct frozen profile
matches the frozen Unity/O defaults, while P Bridge has no internal fallback
ports. Resolved managed values and normal transport behavior remain unchanged.

~~~text
C6_0_1_BRIDGE_CONTRACT_CLOSURE=FAIL
PORT_OWNER=python/planning/runtime/ports.py::WorkerRuntimeSpec+DirectRuntimePortProfile
PORT_OWNER_UNIQUE=YES
MANAGED_PORT_PARITY=PASS
DIRECT_DEFAULT_PARITY=PASS
TWELVE_WORKER_PORT_COLLISION_COUNT=0
TRANSPORT_TRANSACTIONS=10000 per O/P
REAL_UNITY_PRIMITIVES=O=400,P=400
O_RETRY_COUNT=754
P_RETRY_COUNT=767
RETRY_BEHAVIOR=ACCEPTABLE_TRANSPORT_RECOVERY
DUPLICATE_PHYSICAL_EXECUTION_COUNT=0
COMMAND_CONFLICT_COUNT=0
PENDING_FINAL_COUNT=0
PROTOCOL_ERROR_COUNT=0
ZMQ_OPTION_FAILURE_PARITY=FAIL
ZMQ_BIND_FAILURE_PARITY=PASS
FAULT_PROCESS_CLEANUP=PASS
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE=NO
NORMAL_RUNTIME_BEHAVIOR_CHANGED=NO
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=C6.0.1 CONTINUE INVESTIGATION
~~~

Evidence includes 33 focused unit passes, 24 O Bridge ROS regression passes,
24 P Bridge ROS regression passes, compileall, 10,000 exact transport
transactions per binary, and 20 successful independent Unity runs per binary
with reset plus 20 reliable primitives. One O dynamic-port startup bind
failure was retained as a failed attempt and excluded from the selected
400-primitive success aggregate. The fixed transport fixture had zero
business duplicates, conflicts, pending final state, or protocol errors.

ZMQ fault injection is the remaining blocker: bind/address-in-use exits and
cleanup match, but O continues after a first ordinary `zmq_setsockopt` failure
while P exits 1 fail-closed. This is documented as a fault-contract
difference, not a normal runtime semantic change. See
`docs/BRIDGE_CONTRACT_CLOSURE.md` for the full matrix, percentiles, commands,
artifact hashes, and modified-file evidence. C6.1 was not executed.

## C6.0.2 — ZMQ socket-option criticality contract

P's socket-option provenance and criticality seam is now closed. The detailed
inventory and runtime artifacts are in
[`docs/ZMQ_SOCKET_OPTION_CONTRACT.md`](ZMQ_SOCKET_OPTION_CONTRACT.md).

```text
C6_0_2_ZMQ_OPTION_CONTRACT = PASS
PRODUCTION_SOCKET_OPTION_COUNT = 5
REQUIRED_OPTION_COUNT = 3
OPTIONAL_OPTION_COUNT = 2
UNKNOWN_OPTION_COUNT = 0
ZMQ_OPTION_FAILURE_CONTRACT = PASS
O_MATCHES_CANONICAL = MIXED
P_MATCHES_CANONICAL = YES
O_HISTORICAL_FAULT_POLICY_DEFECT = YES
REQUIRED_FAULT_FAIL_CLOSED = PASS
OPTIONAL_DEGRADED_MODE = PASS
FAULT_PROCESS_CLEANUP = PASS
FAULT_PORT_RELEASE = PASS
NORMAL_TRANSPORT_REGRESSION = PASS
NORMAL_UNITY_RUNTIME_REGRESSION = PASS
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE = YES
NORMAL_RUNTIME_BEHAVIOR_CHANGED = NO
GPU_RUNTIME_VALIDATION_DEFERRED = YES
ALGORITHM_CHANGED = NO
NEXT_PHASE = C6.1 BRIDGE PRODUCTION CUTOVER
```

Evidence: the option-fault matrix passed for required, optional, and unknown
paths; the C++ unknown-option harness passed; focused unit tests passed; and
normal P transport/Unity regressions remained exact. No O or Unity source was
modified, and C6.1 was not executed.

## C6.1 status — Bridge production cutover (2026-08-28)

The authorized P-only Bridge production cutover is complete.  O remained the
read-only behavior baseline and Unity remained frozen.  The split
`planning::bridge::UnityBridgeNode` owner tree is the only formal P Bridge
implementation behind `unity_bridge_node`; no old P flat implementation or
formal caller remains.  Compatibility forwarding headers were retained as
public/test references, and no O file was deleted.

```text
C6_1_BRIDGE_PRODUCTION_CUTOVER=PASS
FORMAL_BRIDGE_OWNER=planning::bridge::UnityBridgeNode (BridgeNode)
FORMAL_BRIDGE_IMPLEMENTATION_COUNT=1
OLD_BRIDGE_FORMAL_CALLER_COUNT=0
MISSING_BEHAVIOR_COUNT=0
DUPLICATE_BEHAVIOR_COUNT=0
P_FULL_BUILD=PASS
BRIDGE_COMPILE=PASS
PROTOCOL_PARITY=PASS
COMMAND_GATEWAY_PARITY=PASS
RESULT_GATEWAY_PARITY=PASS
SNAPSHOT_GATEWAY_PARITY=PASS
RESET_GATEWAY_PARITY=PASS
TELEMETRY_PARITY=PASS
PORT_CONTRACT=PASS
ZMQ_OPTION_CONTRACT=PASS
FAULT_PROCESS_CLEANUP=PASS
REAL_UNITY_RUNTIME_PARITY=PASS
PHYSICS_PARITY=PASS
DEPTH_PARITY=PASS
COLLISION_PARITY=PASS
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
DEPTH_SAFETY_REGRESSION=PASS
DUPLICATE_PHYSICAL_EXECUTION_COUNT=0
COMMAND_CONFLICT_COUNT=0
PENDING_FINAL_COUNT=0
PROTOCOL_ERROR_COUNT=0
BRIDGE_BINARY_SHA256=63b00f6a0b31aa72ace3e1c437477e7ee8b670934386f3bb1d645745c302d9ed
RUNTIME_ASSEMBLY_SHA256=7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc
UNITY_PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=C7 FINAL C++ BUILD AND FREEZE
```

Evidence: 10,000 transport transactions per binary; O/P frozen Unity runs
with 400 physical primitives per side; canonical payload/hash parity; five
socket-option fault rows plus bind/startup/shutdown cleanup; and the native
Collision, Global Route, and Depth Safety regression suites.  The detailed
deletion ledger, modified-file list, and artifact hashes are in
[`docs/BRIDGE_PRODUCTION_CUTOVER.md`](BRIDGE_PRODUCTION_CUTOVER.md).

The O legacy ROS integration test has 18 passing cases and six exact-metrics
assertion failures against fields emitted by O's own binary.  The normalized
O/P runtime comparator and P's corresponding 24-test suite pass; O was not
modified.  At the time this C6.1 record was written, no C7 work, Python
pipeline refactor, formal data collection, or commit had been performed.

## C7 — Final C++ build / ABI / install / runtime freeze (2026-08-28)

C7 closed the P-only C++ foundation freeze.  O remained read-only, Unity
remained frozen, and no source, algorithm, BC/AWAC, formal-data, or training
change was made.  The four C7 documents are the durable handoff; the detailed
ABI list is in [`docs/CXX_ABI_MANIFEST.md`](CXX_ABI_MANIFEST.md).

```text
C7_FINAL_CXX_FREEZE=PASS
CXX_FINAL_V1=PASS
CXX_FROZEN_FOR_PLANNING=YES
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

The conda `xm` environment passed Release configure/compile/devel runtime
checks.  Its Catkin install helper is incompatible with the conda distutils
implementation (`--install-layout=deb`); an independent `/usr/bin/python3`
install-space build passed.  This is recorded as a toolchain split, not as a
P source failure.  Missing `data/motion_primitives` remained guarded and no
formal data was generated.

Unity identity was reconciled: the canonical Player SHA is
`61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365`, and the
Player-loaded runtime `Assembly-CSharp.dll` SHA is
`7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc`.
The historical `11f843...` value is the editor artifact, not a second Unity
runtime revision.

Correctness and runtime gates remained exact: Collision 121-case CPU parity,
100-mission candidate/Teacher parity, Global Route and Depth Safety fixtures,
75 protocol/golden cases, 24 real Bridge integration cases, 10,000 transport
transactions per binary, one-worker 100-primitive smoke, and bounded
twelve-worker 240-primitive smoke all passed.  The intentional one-worker
Bridge failure was fail-closed; 11 workers completed and no process/port was
left behind.

```text
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
DEPTH_SAFETY_REGRESSION=PASS
COLLISION_PERFORMANCE_RATIO=0.983331 filter / 0.992835 total
ROUTE_PERFORMANCE_RATIO=0.991387
DEPTH_SAFETY_PERFORMANCE_RATIO=1.009191
PROTOCOL_BRIDGE_REGRESSION=PASS
ONE_WORKER_RUNTIME_SMOKE=PASS
TWELVE_WORKER_RUNTIME_SMOKE=PASS
SINGLE_WORKER_FAULT_ISOLATION=PASS
DUPLICATE_PHYSICAL_EXECUTION_COUNT=0
COMMAND_CONFLICT_COUNT=0
PENDING_FINAL_COUNT=0
PROTOCOL_ERROR_COUNT=0
PROCESS_CLEANUP=PASS
COMPILER_WARNING_COUNT=1
HIGH_SEVERITY_WARNING_COUNT=0
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=PLANNING PYTHON PIPELINE FINALIZATION
```

`perf`, `clang-format`, and `clang-tidy` were unavailable, so hardware-counter
and formatter/linter claims are explicitly not made.  Generated caches and
bytecode were moved recoverably to `/tmp/xm-c7-source-tree-garbage.KYx1El`.
The remaining blockers are reliable-exact Teacher collection and merge/
provenance closure, BC checkpoint/training/evaluation handoff, and AWAC/RL
pipeline finalization.  C7 stopped here.

## PY2 — Relabel and depth-mask producer provenance (2026-08-28)

PY2 closed the P-only provenance seam for offline Teacher relabeling and
depth-mask production. O remained read-only. Unity, C++, Teacher scoring,
depth-safety numeric behavior, BC training math, and AWAC/SAC were not
changed. No formal 60,000-row collection was started.

```text
PY2_RELABEL_MASK_PROVENANCE=PASS
COUNT_DIFFERENCE_EXPLAINED=YES
LABEL_PRODUCER_PROVENANCE=PASS
DEPTH_MASK_PRODUCER_PROVENANCE=PASS
CROSS_ARTIFACT_CONSISTENCY=PASS
TEACHER_LABEL_ARRAY_PARITY=PASS
DEPTH_MASK_ARRAY_PARITY=PASS
DATASET_AUDIT_PROVENANCE=PASS
RELIABLE_ROLLOUT_TO_BC_MMAP_SMOKE=PASS
OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
LEGACY_ROWS=0
TELEMETRY_LOOKUP_COUNT=0
SNAPSHOT_MISSING_COUNT=0
STATE_DEPTH_SKEW_MAX_NS=0
FRAME_CONTRACT_FAILURES=0
ALGORITHM_CHANGED=NO
DATA_LAYOUT_CHANGED=NO
```

### PY0 count semantics

`RELIABLE_ROWS=696` is the root two-worker collection report's sum of exact
endpoint observations across all 24 attempts, including the two rejected
attempts. The accepted NPZ transition count is 624: 22 accepted report rows
persisted 624 action-level transitions. The rejected attempts contributed 72
reliable observations that were not persisted as accepted NPZ transitions.
Thus the difference is expected accounting, not a producer mismatch.

The bounded PY2 fixture selected the first two accepted episodes: 56
transitions, with 28 transitions per episode. Its immutable input identities
were:

```text
ROLLOUT_INDEX_SHA256=bdf6f6f6900ac41428de03671f32cfb893d8edeedcbe045ee708b275b205e287
ROLLOUT_MANIFEST_SHA256=182e10a8486733e5aef2713b2f27fb352fb45f4111efae0247fa35ad2c1d84c1
MISSION_INDEX_SHA256=a55699151bf59f7bcea3ed934846a9db26061341e09dd27e07854c728301f729
MPL_CONTRACT_SHA256=f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d
```

### Producer and audit contract

`python/planning/contracts/pipeline_provenance.py` is the minimal PY2
provenance owner. It validates root, worker, index-row, and episode metadata
before either producer runs. Missing or unknown observation contracts,
legacy rows, mixed sources, non-zero telemetry lookup, snapshot-missing,
state/depth skew, frame failures, or invalid endpoint identity fail closed.
The label producer and depth-mask producer validate the same contract again
before writing their NPZ artifacts. Their metadata carries the input index and
manifest SHA values, root/accepted reliable counts, episode identity and
offset mapping, action count, mission/MPL/resolved-config identity, Teacher
contract identity, and the depth-mask contract and numeric configuration.

`TeacherLabelStore`, `DepthActionMaskStore`, dataset audit, and BC mmap reject
invalid PY2 metadata. Dataset audit now checks rollout, labels, and masks as
one row/episode/contract domain. BC mmap retains all existing fields and adds
PY2 provenance fields when strict producer artifacts are present. Its
`--max-episodes` option only bounds a validated fixture selection; the default
remains the full accepted index.

### O/P parity and artifacts

O's historical relabel/mask artifacts do not carry the new producer schema,
so O is used as the numerical and ordering oracle while the P provenance
schema is canonical for new artifacts. On the same PY0 fixture:

```text
TEACHER_SOFT_TARGETS_SHA256=8686a1a5224695e0b22d4695b52337698b280ffbf7812754423ed7df930a287
TEACHER_GLOBAL_MASKS_SHA256=c8b1ee2f74b81931bbf03f0739c44a90790462514e1bd14b7203255ff7e5d3a3
TEACHER_ARGMAX_SHA256=3ecf501d7ddebb2a98cd06dc37d9f79f77badd6c3d114dafb55082b8463e0cb1
BEHAVIOR_ACTIONS_SHA256=3ecf501d7ddebb2a98cd06dc37d9f79f77badd6c3d114dafb55082b8463e0cb1
VALID_COUNTS_SHA256=da22508279b7d2b7c13d03d3232210dde1654bf372a3e0089d51794439775d50
TEACHER_ENTROPY_SHA256=4342027c5322e0616425db25cdee998b9b39700fe47011ee237e13f4f8798274
DEPTH_ACTION_MASK_SHA256=7a702c609b4bbed73ef2b1cc231cd7ec0245e8f3a75e8cf36166d80bb598dfd4
BC_MMAP_LOCAL_DEPTH_MASK_SHA256=7a702c609b4bbed73ef2b1cc231cd7ec0245e8cf36166d80bb598dfd4
```

The P fixture artifacts were written under `/tmp/xm-py2-chain`:

```text
P_TEACHER_LABELS_SHA256=4608f80e86f49ab3b4b718f404903879aa009faf418ce73eb34032c5ad39b65a
P_DEPTH_MASKS_SHA256=e3ca5ea4cc88170bc64e45e2fcdbc5a2fba063892964903e2d076dc079fa9ee6
P_DATASET_AUDIT_SHA256=ea957f5ef4b3d8424306645227c338e01f27f6763ac9a724f517025c73dbc183
P_BC_MMAP_MANIFEST_SHA256=8bdbf5b313b945f6ef13d3495680e5a67b73467ff72a332140e48b1826f88589
```

The O/P BC mmap comparison also passed for all ten mmap arrays. Source
artifact path spellings and their file hashes differ only because P adds the
new provenance metadata; the business manifest fields, row order, and array
payloads remain exact. The historical M0.1 BC mmap comparator remains PASS.

### PY2 modified files and tests

Modified only P-side files:

- `python/planning/contracts/pipeline_provenance.py`
- `python/planning/teacher/label_worker.py`
- `python/planning/teacher/labeling.py`
- `python/planning/teacher/dataset_audit.py`
- `python/planning/safety/depth_action_mask_worker.py`
- `python/planning/safety/depth_action_masks.py`
- `python/planning/safety/depth_mask.py`
- `python/planning/data/rollout.py`
- `python/planning/data/bc_mmap.py`
- `tests/test_py2_relabel_mask_provenance.py`
- `tests/compare_py2_relabel_mask_fixture.py`
- `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`

```text
compileall=PASS
PY2_FOCUSED_TESTS=33 passed
PY2_CLI_HELP=PASS
O/P_RELABEL_COMPARATOR=PASS
O/P_DEPTH_MASK_COMPARATOR=PASS
O/P_BC_MMAP_ARRAY_PARITY=PASS
RELIABLE_ROLLOUT_TO_AUDIT_TO_BC_MMAP=PASS
```

The remaining observation chain is BC checkpoint/training provenance,
evaluation handoff, and the frozen AWAC/RL boundary already recorded in the
M0.x sections. PY2 does not execute any of those stages or formal data
collection.

```text
NEXT_PHASE=PY3 PRE-BC PIPELINE CUTOVER AND E2E SMOKE
```

## PY0 — Reliable-exact Teacher collection final ownership (2026-08-28)

PY0 closed the P-only formal Teacher collection owner. O remained read-only,
Unity remained frozen, and the P C++ foundation, Collision, Global Route,
Depth Safety, Bridge/Protocol/Transport, BC/AWAC math, and formal data scale
were not changed.

```text
PY0_RELIABLE_EXACT_COLLECTION=PASS
FORMAL_COLLECTION_IMPLEMENTATION_COUNT=1
LEGACY_FORMAL_COLLECTION_CALLER_COUNT=0
SINGLE_WORKER_RELIABLE_COLLECTION=PASS
MULTIWORKER_RELIABLE_COLLECTION=PASS
AGGREGATE_TARGET_SEMANTICS=PASS
MERGE_COMPATIBILITY=PASS
RESUME=PASS
OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
RELIABLE_ROWS=696
LEGACY_ROWS=0
TELEMETRY_LOOKUP_COUNT=0
SNAPSHOT_MISSING_COUNT=0
STATE_DEPTH_SKEW_MAX_NS=0
FRAME_CONTRACT_FAILURES=0
ENDPOINT_IDENTITY_CHAIN=PASS
ONE_WORKER_SMOKE=PASS
TWO_WORKER_SMOKE=PASS
TWELVE_WORKER_PREFLIGHT=PASS
PROCESS_CLEANUP=PASS
ALGORITHM_CHANGED=NO
```

The formal path is owned by `scripts/collect_teacher_rollouts.py` and
`scripts/collect_rollouts_parallel.py`, with the Teacher, runtime, shard, and
merge seams under `planning/teacher`, `planning/runtime`, and `planning/data`.
The default path is reliable exact: one reliable command, physical primitive,
result/ACK/commit, and authoritative co-timestamped endpoint snapshot. The
legacy asynchronous path remains explicit diagnostic-only and has no formal
caller.

Evidence included a one-worker exact smoke (2 accepted, 56 reliable rows), a
two-worker aggregate smoke (22 accepted, 696 reliable rows), and a real resume
smoke (3 accepted before stop, resumed to 5 accepted, 141 unique reliable
rows). The two-worker root manifest passed its aggregate target of 20. The
resume manifest passed after Unity execution-counter reset by allocating a
transition-ID offset above the persisted shard; episode, mission, and
transition identities remained unique. A separate 50-target boundary run was
stopped at 46 accepted and is recorded as an unreached-target diagnostic, not
as passing evidence.

The primary two-worker smoke artifacts are under
`/tmp/xm-py0-two-worker-VTl5tw/rollouts` (root summary SHA256
`182e10a8486733e5aef2713b2f27fb352fb45f4111efae0247fa35ad2c1d84c1`; merged
index SHA256
`bdf6f6f6900ac41428de03671f32cfb893d8edeedcbe045ee708b275b205e287`). The
resume evidence is under `/tmp/xm-py0-resume-real-PLynpF/rollouts` (root
summary SHA256
`61b00d24d37b12fcd121804b4b725fdc51a578c1503a61f0b3c508873e771c17`). The
shared mission-index SHA256 is
`a55699151bf59f7bcea3ed934846a9db26061341e09dd27e07854c728301f729`, the
forest-cache SHA256 is
`a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691`, the
MPL contract SHA256 is
`f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d`, and
the P collection source SHA256 recorded by the launcher is
`41017590ebc6e90d841a80c2eede2d03a4faefcaf62c5cf974dcaee7a3fbfe0a`.
The two-worker manifest's 696 reliable-row counter includes rejected attempts;
the 22 accepted NPZ shards contain 624 action-level transition rows.

Runtime artifact identities used for the passing smokes were:

```text
UNITY_PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
RUNTIME_ASSEMBLY_SHA256=7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc
BRIDGE_SHA256=63b00f6a0b31aa72ace3e1c437477e7ee8b670934386f3bb1d645745c302d9ed
```

The Player SHA is authoritative in the runtime manifest and detailed PY0
document; the progress record above is a handoff copy and must match that
manifest before it is used for a later production run.

Focused PY0 plus protected provenance/runtime tests passed `101 passed`; P
Python and script `compileall` passed. Modified files were limited to the P
Python collection/contracts/runtime/data seams, the two collection shell/script
entry points, focused PY0 tests, and the two Python collection documents. No
files were deleted. No O, Unity, C++, BC/AWAC, or formal 60,000-row artifact
was modified or generated.

```text
PY2_STATUS=COMPLETED_IN_THE_SECTION_ABOVE
NEXT_PHASE=PY3 PRE-BC PIPELINE CUTOVER AND E2E SMOKE
```

## PY3 — Pre-BC canonical pipeline cutover and bounded E2E smoke (2026-08-28)

PY3 completed the isolated P pre-BC chain with O remaining read-only and all
generated outputs confined to `/tmp/xm-pre-bc-py3/`. No formal 60,000-episode
collection was performed. Unity, C++, Mission/Teacher algorithms, BC math,
and AWAC/SAC were not changed.

```text
PY3_PRE_BC_PIPELINE=PASS
PRE_BC_E2E_SMOKE=PASS
MOTION_PRIMITIVE=PASS
MISSION_GENERATION=PASS
MISSION_AUDIT=PASS
RELIABLE_COLLECTION=PASS
RELABEL=PASS
DATASET_AUDIT=PASS
DEPTH_MASK=PASS
BC_MMAP=PASS
BC_TRAINING_SMOKE=PASS
RELIABLE_EVALUATION_SMOKE=PASS
ARTIFACT_SHA_CHAIN=PASS
OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
OLD_BACKUP_REQUIRED_BY_PRODUCTION=NO
OLD_BACKUP_TESTS_FIXED=YES
COMPILEALL=PASS
TARGETED_TESTS=35 passed, 0 failed
FULL_TESTS=870 passed, 42 skipped, 0 failed
```

The newly generated MPL used 105 actions and 25 command frames. Mission
generation published 200 candidates after deterministic sampling and mission
audit selected 60 passing rows from 74 audited rows. The reliable exact
collection attempted 57 missions, accepted 21, and merged 21 NPZ shards.

The count semantics are explicit and exact:

```text
ATTEMPT_EXACT_OBSERVATION_ROWS=1018
ACCEPTED_PERSISTED_TRANSITION_ROWS=598
REJECTED_ATTEMPT_OBSERVATION_ROWS=420
1018 = 598 + 420
```

All formal observation counters were zero for legacy rows, telemetry lookup,
snapshot missing, state/depth skew, and frame-contract failures. Relabel,
dataset audit, and depth masks each passed 21 episodes / 598 transitions with
the exact observation contract. BC mmap and the one-epoch CPU mmap training
smoke passed; the checkpoint carries the mmap, source labels/masks/index,
MPL, row-count, and observation provenance hashes.

The reliable evaluation smoke ran ten separate audit-only episodes because the
ordinary managed wrapper's formal mode requires at least 100 episodes. All
ten runtime summaries passed checkpoint/index identity and reliable-v4
provenance checks with no telemetry fallback or snapshot misses. The quality
gate was intentionally not applicable, so these runs are runtime smoke
evidence and not a success-rate conclusion.

O/P relabel and depth-mask numerical parity remains exact on the fixed PY0
fixture: teacher label arrays, depth-mask arrays, episode/transition order,
and business metadata all passed. O does not expose the current reliable
producer metadata schema; P's full smoke therefore uses the canonical strict
provenance chain, while the O/P fixture remains the numerical oracle.

The machine-readable lineage and inventory records are:

```text
/tmp/xm-pre-bc-py3/sha_chain.json
/tmp/xm-pre-bc-py3/evaluation/smoke_summary.json
/tmp/xm-pre-bc-py3/inventory_classification.json
```

Documentation added:

```text
docs/PRE_BC_CANONICAL_PIPELINE.md
docs/PRE_BC_CANONICAL_COMMANDS.md
docs/PRE_BC_KEEP_SET.md
docs/PRE_BC_DELETION_LEDGER.md
```

Focused verification passed `83 passed in 4.13s`; compileall passed and all ten
canonical CLI help commands passed. No deletion was executed. The collection
shell wrapper remains retained because its workspace bootstrap can select O
when invoked with the wrong workspace; PY4 must close that operational
ambiguity before formal cutover. P's default formal MPL files remain absent;
the bounded smoke used a newly generated `/tmp` MPL.

```text
ALGORITHM_CHANGED=NO
DATA_LAYOUT_CHANGED=NO
NEXT_PHASE=PY4 EXECUTE SAFE DELETION AND CANONICAL CODE CLEANUP
```

## PY4-A — Canonical production surface and symbol-level dedup audit (2026-08-28)

PY4-A audited the P install surface, canonical pre-BC entrypoints, AST/static
symbol graph, common mechanical capabilities, configuration defaults,
test-only production symbols, frozen RL imports, formal MPL ownership, and
generated caches. O remained read-only. No Unity/C++, algorithm, artifact
schema, formal 60,000-row collection, AWAC/SAC, or O cutover was executed.

```text
PY4_A_PRODUCTION_SURFACE_AUDIT=PASS
CANONICAL_PRE_BC_ENTRY_COUNT=11
DUPLICATE_CANONICAL_ENTRY_COUNT=0
PRODUCTION_PYTHON_MODULE_COUNT=132
PRODUCTION_SCRIPT_COUNT=28
INSTALLED_TEST_FILE_COUNT=0
INSTALLED_DIAGNOSTIC_FILE_COUNT=26
DUPLICATE_COMMON_CAPABILITY_COUNT=3
DUPLICATE_CONFIG_OWNER_COUNT=13
PURE_REEXPORT_UNUSED_COUNT=0
TEST_ONLY_PRODUCTION_SYMBOL_COUNT=212
AWAC_TO_SAC_IMPORT_COUNT=3
RL_FORMAL_ENTRY_COUNT=10
RL_TEMPORARY_DEPENDENCY_COUNT=3
FORMAL_MPL_READY=NO
WORKSPACE_WRAPPER_RESOLUTION=PASS
GENERATED_FILE_COUNT_BEFORE=56
GENERATED_FILE_COUNT_AFTER=56
SAFE_SYMBOL_DELETION_COUNT=0
SAFE_FILE_DELETION_COUNT=0
SAFE_DELETION_BATCH=[]
SCOPE_EXPANDED=NO
PRE_BC_BEHAVIOR_CHANGED=NO
ARTIFACT_SCHEMA_CHANGED=NO
ALGORITHM_CHANGED=NO
```

The only source modification was
`scripts/collect_rollouts_parallel.sh`. It now preserves CLI arguments while
sourcing setup, handles source/devel/install script locations, and prepends P
source package/ROS paths. The wrapper remained bootstrap plus exec only: no
collection policy, port derivation, process management, or merge logic was
added. Source, devel, install, arbitrary-cwd, and explicit-workspace help
checks passed.

The production surface contains 132 Python module files in 16 discovered
packages, 24 installed Python scripts, 4 installed shell scripts, 2 configs,
8 launch files, and 3 RViz files. Tests/docs are not installed. The 26-file
diagnostic count is the 20-module `planning.diagnostics` package plus six
explicit diagnostic/reference CLIs; these remain named qualification seams.

The AST graph `/tmp/xm-py4-a-symbol-graph.json` contains 1,197 symbols. It
found 28 same-name groups but only one exact body duplicate, the protected
`_update_bytes` checkpoint helper shared by a contract module and RL evidence
module. It found 212 static test/diagnostic-only candidates and 134 dynamic or
public-use candidates; neither count is a deletion authorization. The
compatibility `planning.common.atomic` re-export has frozen RL/SAC callers,
so `PURE_REEXPORT_UNUSED=0`.

Three common capability families remain parity-gated rather than merged:
file SHA-256, canonical JSON hashing, and atomic JSON replacement. Thirteen
requested configuration families have multiple independent default
declarations, including stage-specific worker pools; no resolved business
value was changed. AWAC/SAC imports and all three temporary RL dependencies
remain frozen.

P still lacks the formal MPL NPZ/JSON pair under `data/motion_primitives`.
CMake installs that directory only when it already exists, and runtime loading
fails closed when the pair is absent or inconsistent. The recorded generation
command was not executed; the PY3 `/tmp` MPL was not copied into P.

Compileall passed, the PY3 focused plus collection-architecture suite passed
`87`, all 11 canonical CLI help commands passed, the O/P relabel-mask
comparator remained exact, and the existing artifact SHA chain remained
`PASS`. The auxiliary script-contract invocation had two unrelated existing
or environment-bound failures and was not used as the PY4-A gate.

```text
NEXT_PHASE=PY5 P CLEAN BUILD AND O CUTOVER PREPARATION
```

## PY5 — P clean build and O cutover preparation (2026-08-28)

PY5 completed the non-destructive build, install, staged MPL, staged Pre-BC,
bounded 12-worker infrastructure, preservation inventory, and cutover dry-run
evidence. O remained read-only. No formal 60,000-row collection, AWAC run,
Unity/C++ change, source deletion, or cutover was executed.

The old optimized-tree assessment is explicitly historical and superseded:

```text
OPTIMIZED_TREE_ASSESSMENT=HISTORICAL_PRE_MIGRATION_BASELINE
STATUS=SUPERSEDED_BY_CURRENT_GATES
```

This invalidates the old current-status claims that native route was rejected,
CUDA/hybrid removal was rejected, Bridge was static-only, and reliable
collection, BC mmap provenance, BC trainer provenance, evaluation provenance,
or relabel/mask provenance were missing. The old report is retained as
history; it was not edited or deleted.

```text
PY5_P_CLEAN_BUILD_AND_CUTOVER_PREP=FAIL
P_CLEAN_BUILD=PASS
P_INSTALL_BUILD=PASS
PYTHON_FULL_TESTS=FAIL
TEST_FAILURE_COUNT=39
CANONICAL_CLI=11/11
STAGED_MPL_READY=YES
MPL_ACTION_COUNT=105
MPL_FRAME_COUNT=25 command / 26 reference
MPL_NPZ_SHA256=22ad22fe66a88633de6effff11c03aa4ca4e8df362f8b410e17effbd91ff7309
MPL_JSON_SHA256=c1b795e4737a12034b0d59a1b457f559e7ad138527df2c90ed35db1326e42ea8
STAGED_PRE_BC_E2E=PASS
STAGED_TWELVE_WORKER_SMOKE=PASS
O_PRESERVATION_MANIFEST=PASS
CUTOVER_DRY_RUN=PASS
CUTOVER_REPLACE_COUNT=381
CUTOVER_DELETE_COUNT=0
CUTOVER_PRESERVE_COUNT=120179
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
ROLLBACK_PLAN=PASS
P_TO_O_CUTOVER_READY=NO
DUPLICATE_COMMON_CAPABILITIES_DEFERRED=3
DUPLICATE_CONFIG_OWNERS_DEFERRED=13
AWAC_TO_SAC_IMPORT_COUNT=3
ALGORITHM_CHANGED=NO
DATA_LAYOUT_CHANGED=NO
NEXT_PHASE=PY5 FIX CUTOVER BLOCKERS
```

P source caches were cleaned from 56 confirmed `.pyc` files to zero. Fresh
temporary devel and install roots built all P C++ targets with only the
existing unused `AsInt32` warning. The staged install imported P from
`/tmp/xm-py5-install-MXchWt/install/lib/python3/dist-packages/planning` and
installed zero tests or fixtures. Of 26 installed diagnostic/reference files,
4 are required shared diagnostics, 22 are operator/reference tools, and 0 are
qualification-only files.

The full Python suite is the remaining direct gate: `717 passed, 42 skipped,
39 failed`. The failures are 10 frozen Unity C# path contracts, 4 existing P0
fake-runtime contracts, 24 frozen AWAC/SAC guarded-runner environment-path
contracts, and 1 existing versioned-contract naming check. They are recorded
without changing their protected owners. The focused native C++/protocol suite
passed 55 tests with one ROS-gated skip.

The staged P config generated the 105-action MPL twice with identical hashes.
The staged chain passed motion primitives, mission generation/audit, reliable
2-worker collection, relabel, dataset audit, depth masks, BC mmap, one-epoch
BC, and an audit-only reliable evaluation smoke. All observation provenance
counters were zero. The 12-worker run passed identity/port/startup/cleanup
checks with 12 unique workers; its zero accepted episodes were intentional for
the `MAX_STEPS=3` infrastructure-only smoke.

The O preservation and migration details are in:

```text
docs/O_PRESERVATION_MANIFEST.md
docs/P_TO_O_MIGRATION_MANIFEST.md
docs/P_TO_O_CUTOVER_PLAN.md
docs/PY5_CLEAN_BUILD_REPORT.md
```

The dry-run wrote 381 regular P files, deleted nothing, and excluded all
120,179 O data/history files. No cutover was performed. The next permitted
action is to fix or explicitly resolve the full-test blockers; do not execute
PY6 from this state.

## PY5.1 — Full test contract closure (current, 2026-08-28)

PY5.1 is the current phase record and supersedes the preceding PY5 failure
gate. O remained read-only; Unity and C++ remained frozen. No formal 60,000
row collection, RL run, or P-to-O cutover was executed.

The old optimized-tree conclusion remains explicitly historical:

```text
OPTIMIZED_TREE_ASSESSMENT=HISTORICAL_PRE_MIGRATION_BASELINE
STATUS=SUPERSEDED_BY_CURRENT_GATES
```

Its obsolete claims were that native Global Route was unverified,
CUDA/hybrid removal was unverified, Bridge split was unverified, and reliable
collection, BC mmap provenance, BC trainer provenance, evaluation provenance,
or relabel/mask provenance were missing. Those claims are superseded by the
C7, M0.1, M0.2, M0.3, PY0, PY2, PY3, PY4-A, and PY5 evidence.

```text
PY5_1_FULL_TEST_CONTRACT_CLOSURE=PASS
INITIAL_TEST_FAILURE_COUNT=39
CLASSIFIED_FAILURE_COUNT=39
UNKNOWN_FAILURE_COUNT=0
STALE_PATH_CONTRACT_COUNT=10
STALE_SOURCE_MANIFEST_COUNT=0
STALE_FIXTURE_ROOT_COUNT=28
STALE_IMPORT_PATH_COUNT=0
STALE_TEXT_ASSERTION_COUNT=1
INTENTIONAL_DEFERRED_SCOPE_COUNT=0
TRUE_RUNTIME_REGRESSION_COUNT=0
TRUE_ALGORITHM_REGRESSION_COUNT=0
UNITY_PATH_CONTRACT_TESTS=PASS
FAKE_RUNTIME_CONTRACT_TESTS=PASS
AWAC_SAC_RUNNER_CONTRACT_TESTS=PASS
VERSION_NAMING_TEST=PASS
UNEXPLAINED_VERSIONED_BUSINESS_NAME_COUNT=0
FULL_PYTHON_TESTS=PASS
FINAL_TEST_FAILURE_COUNT=0
SKIP_COUNT_BEFORE=42
SKIP_COUNT_AFTER=42
XFAIL_COUNT_BEFORE=0
XFAIL_COUNT_AFTER=0
CANONICAL_CLI=11/11
PY5_COMPILEALL=PASS
```

The 39-row failure matrix and protected behavior record is
`docs/PY5_FULL_TEST_CONTRACT_CLOSURE.md`. All 39 were stale path, fixture,
or text contracts; no algorithm file, business assertion, skip, or xfail was
weakened. The only production path change was
`scripts/run_sac_guarded.sh`, and its guarded AWAC/SAC behavior remains
unchanged. The versioned-name allowlist has 36 exact entries with reasons and
removal phases; AWAC/SAC cleanup remains deferred until new BC evaluation.

Post-production validation used the fresh P build/install roots
`/tmp/xm-py5-1-devel-final` and `/tmp/xm-py5-1-install-final`:

```text
P_CLEAN_BUILD=PASS
P_INSTALL_BUILD=PASS
STAGED_PRE_BC_E2E=PASS
STAGED_TWELVE_WORKER_SMOKE=PASS
O_PRESERVATION_MANIFEST=PASS
CUTOVER_DRY_RUN=PASS
CUTOVER_REPLACE_COUNT=396
CUTOVER_DELETE_COUNT=0
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
ROLLBACK_PLAN=PASS
AWAC_TO_SAC_IMPORT_COUNT=3
ALGORITHM_CHANGED=NO
RUNTIME_BEHAVIOR_CHANGED=NO
P_TO_O_CUTOVER_READY=YES
NEXT_PHASE=PY6 EXECUTE P TO O CUTOVER
```

The fresh bounded chain is under `/tmp/xm-py5-1-staged-e2e-final/`:
2-worker collection produced 150 reliable rows and four accepted episodes;
relabel, masks, dataset audit, and BC mmap passed for 114 transitions; the
one-epoch BC and audit-only reliable evaluation passed. The 12-worker
infrastructure smoke completed all 12 unique workers/ports with zero
collector or reliability errors; its zero-accepted quality gate was
intentionally not applicable to infrastructure readiness. All observation
provenance counters were zero.

The final dry-run and O check are recorded at:

```text
/tmp/xm-py5-1-cutover-rsync-dry-run-final.txt
/tmp/xm-py5-1-o-preservation-final.txt
```

This section records readiness only. The next phase must execute the future
cutover procedure under its own authorization; PY5.1 stops here.

## PY6 — Formal P to O cutover (completed, 2026-08-28)

The preceding PY5/PY5.1 readiness sections are historical pre-cutover
records. They are superseded by the authorized execution below and are
retained for audit history.

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

The detailed record is `docs/PY6_CUTOVER_REPORT.md`; the final Pre-BC
contract is `docs/PLANNING_PRE_BC_FINAL_V1.md`. The controlled replacement
used 482 P regular files as its source listing, transferred 396 regular files,
and archived 119 stale O-only non-generated files. It preserved O `data/` and
the complete `data_back_20260826/` history; the only new O data files are the
formal 105-action MPL pair.

The final bounded chain is under `/tmp/xm-py6-o-e2e/`. Its 2-worker reliable
collection produced four accepted episodes and 176 reliable rows, with 114
accepted downstream transitions. Relabel, depth masks, dataset audit, BC mmap,
one-epoch CPU BC, and one-episode reliable evaluation audit all passed. The
12-worker infrastructure smoke completed all 12 workers with unique runtime
identities and canonical ports; `TARGET_ACCEPTED=0` intentionally made its
parent accepted-nonzero quality gate false.

The actual runtime used the exact frozen Player, Assembly-CSharp, and C7
Bridge hashes. O clean/devel/install builds passed independently; the newly
built O Bridge hash was not substituted for the frozen runtime artifact. No
formal 60,000-row collection, formal BC training, AWAC/RL run, Unity change,
or commit was performed. The next phase is user execution of formal reliable
BC data collection.

## PY7-A / PY7-B0 — common/config ownership and naming inventory (2026-08-28)

This phase continued only in canonical O at
`/home/xm/XM/xm_ws/src/planning`; P and Unity remained read-only.  PY7-A
centralized the targeted generic file-hash, canonical-JSON, atomic-JSON,
protocol-constant, domain-default, and Pre-BC YAML seams.  The explicit
newline option on the shared JSON writer preserves the previous byte contract
for both newline and no-newline callers.  Protocol aliases remain public, but
the Python numeric owner is now `python/planning/protocol/constants.py`.

The owner audit reports:

```text
DUPLICATE_COMMON_CAPABILITY_COUNT_BEFORE=3
DUPLICATE_COMMON_CAPABILITY_COUNT_AFTER=0
DUPLICATE_CONFIG_OWNER_COUNT_BEFORE=13
DUPLICATE_CONFIG_OWNER_COUNT_AFTER=0
PORT_OWNER_UNIQUE=YES
TASK_MAX_STEPS_OWNER_UNIQUE=YES
PRIMITIVE_FRAME_COUNT_OWNER_UNIQUE=YES
```

The task default is still `DEFAULT_MAX_PRIMITIVE_STEPS=45`, but the frozen
`task_contract_metadata()` and task SHA intentionally remain unchanged.  The
required inclusion of `max_steps` in that metadata/SHA is a documented blocker
because it would invalidate the protected P3/AWAC/SAC task identity.  No
second hash or compatibility fiction was introduced.

PY7-B0 scanned 509 files and 312 unique `(scope,path,identifier)` rows.  The
formal V1/V2/V3/V4 marker count is 283, with 0 unknowns.  The full category
counts and all bounded rename candidates are in
`docs/CROSS_LANGUAGE_NAMING_INVENTORY.md` and
`docs/CROSS_LANGUAGE_RENAME_MANIFEST.md`; no rename was executed.  Frozen
Unity partial filenames remain `KEEP` after confirming all eight files declare
the same partial class and no ownership leak was introduced.  RL names remain
deferred and `AWAC_TO_SAC_IMPORT_COUNT=3`.

Validation evidence:

```text
PYTHON_IMPORT_ROOT=/home/xm/XM/xm_ws/src/planning/python
COMPILEALL=PASS
FULL_TESTS=763 passed, 42 skipped, 0 failed
CANONICAL_CLI=11/11
PROTOCOL_GOLDEN_FOCUSED=13 passed
O_CLEAN_CXX_BUILD=PASS
PRE_BC_E2E=PASS (existing PY6 chain revalidated by current dataset audit)
TWELVE_WORKER_SMOKE=PASS (12 workers; accepted-zero infrastructure gate)
BUSINESS_RESOLVED_CONFIG_UNCHANGED=YES
PRE_BC_BEHAVIOR_CHANGED=NO
ALGORITHM_CHANGED=NO
WIRE_BEHAVIOR_CHANGED=NO
```

Modified production/test/doc files are listed in the PY7 handoff documents;
no files were deleted, no formal data/training run was started, and no commit
was made.  The next authorized phase is `PY7-B EXECUTE BOUNDED CROSS-LANGUAGE
RENAME`; it must respect the closed V1/V2 task-contract boundary and the
manifest's compatibility gates.

## PY7-A1 — Task Contract V2 closure (2026-08-29)

PY7-A1 was executed only in canonical O at
`/home/xm/XM/xm_ws/src/planning`. P and Unity remained read-only. The prior V1
identity is frozen exactly at:

```text
TASK_CONTRACT_V1_SHA=5862af0f354c408d74cf4904dc7ecf609944553443ae1128097136be979fd30a
```

The formal new Pre-BC identity is:

```text
TASK_CONTRACT_SCHEMA_VERSION=2
TASK_CONTRACT_V2_SHA=2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df
MAX_PRIMITIVE_STEPS=45
FORMAL_PRE_BC_TASK_CONTRACT=V2
```

`python/planning/contracts/task.py` is the unique owner of the canonical V2
serializer/SHA and the explicit V1 historical seam. `mission.max_steps`
resolves to `max_primitive_steps`. Missing/unknown schema, missing max, SHA
mismatch, max/config mismatch, and V1 input to formal V2 fail closed. Frozen
AWAC/SAC uses explicit legacy V1 fixtures and was not changed.

The bounded V2 artifact chain is
`/tmp/xm-py7-a1-v2-chain/sha_chain.json`:

```text
MISSION_CANDIDATES=V2
MISSION_AUDIT=V2
RELIABLE_COLLECTION_ROOT_WORKERS=V2
TEACHER_LABELS=V2
DATASET_AUDIT=V2
DEPTH_MASKS=V2
BC_MMAP=V2
BC_CHECKPOINT=V2
EVALUATION_SUMMARY=V2
TASK_CONTRACT_CHAIN_MISMATCH_COUNT=0
```

The current O producers passed a 4-episode/114-transition bounded chain.
Teacher labels, depth masks, BC mmap arrays, model state/normalizer, fixed
policy logits, and evaluation business metrics are exact parity with the
pre-V2 bounded fixture. Only task-contract identity/provenance and
non-business timing fields changed. The O/P historical V1 relabel-mask
comparator also passed exact arrays, order, and business metadata.

Validation evidence:

```text
COMPILEALL=PASS
FULL_TESTS=772 passed, 42 skipped, 0 failed
CANONICAL_CLI=11/11
TWELVE_WORKER_SMOKE=PASS (existing accepted-zero infrastructure evidence; current focused gate 17 passed)
PRE_BC_E2E=PASS (strict V2 bounded relabel -> audit -> masks -> mmap -> BC -> reliable evaluation smoke)
LEGACY_ROWS=0
TELEMETRY_LOOKUP_COUNT=0
SNAPSHOT_MISSING_COUNT=0
STATE_DEPTH_SKEW_MAX_NS=0
FRAME_CONTRACT_FAILURES=0
ALGORITHM_CHANGED=NO
RUNTIME_BEHAVIOR_CHANGED=NO
ARTIFACT_IDENTITY_CHANGED=YES
```

Modified production files were limited to the task-contract seam and the
Pre-BC producer/validator propagation; tests were updated for explicit V2
fixtures and frozen V1 RL fixtures. Documentation was updated in the task,
config, Pre-BC, naming inventory, rename manifest, and this progress file.
No formal 60,000-row collection, production BC training, AWAC/SAC run,
Unity/C++ change, PY7-B rename, or commit was performed.

The next phase is `PY7-B EXECUTE BOUNDED CROSS-LANGUAGE RENAME`; it was not
started.

## PY7-B0 — Naming inventory normalization (2026-08-29)

PY7-B0 normalized the existing cross-language naming inventory only. No
production rename, business-code change, formal data collection, BC training,
AWAC/SAC run, Unity/C++ change, or commit was performed.

The current accounting separates raw occurrences from unique rename targets:

```text
VERSIONED_NAME_OCCURRENCE_TOTAL=1032
VERSIONED_NAME_UNIQUE_TARGET_TOTAL=319
FILE_TEXT_OCCURRENCE_TOTAL=995
PATH_SURFACE_OCCURRENCE_TOTAL=37
KEEP_PROTOCOL_VERSION_UNIQUE_COUNT=86
KEEP_SCHEMA_VERSION_UNIQUE_COUNT=8
KEEP_ARTIFACT_VERSION_UNIQUE_COUNT=8
RENAME_PRE_BC_BUSINESS_NAME_UNIQUE_COUNT=26
DEFER_RL_NAME_UNIQUE_COUNT=50
TEST_FIXTURE_NAME_UNIQUE_COUNT=140
HISTORICAL_DOC_REFERENCE_UNIQUE_COUNT=1
UNKNOWN_UNIQUE_COUNT=0
UNIQUE_COUNT_CLOSED=YES
OCCURRENCE_COUNT_CLOSED=YES
UNITY_PARTIAL_NAMING=KEEP
RENAME_MANIFEST=PASS
PRODUCTION_CODE_CHANGED=NO
```

The row-level source of truth is
`docs/cross_language_rename_manifest.json`. Every target has one mutually
exclusive category; only the 26 Pre-BC business-name rows have
`rename_phase=PY7-B`. Protocol/schema/artifact identities, test fixtures,
historical documentation, and all RL identities remain preserved or deferred.

The previous `OPTIMIZED_TREE_ASSESSMENT` remains explicitly marked
`HISTORICAL_PRE_MIGRATION_BASELINE` and `SUPERSEDED`. Its former next-step
field is explicitly retired:

```text
NEXT_SINGLE_MIGRATION=SUPERSEDED
```

The next authorized phase is
`PY7-B EXECUTE BOUNDED PRE-BC BUSINESS RENAME`, and execution stops here.

## FINAL PRE-COLLECTION OPTIMIZATION + NEW RUNTIME ALIGNMENT (2026-08-29)

This bounded optimization and alignment pass used the live canonical Planning
tree at `/home/xm/XM/xm_ws/src/planning`.  Unity source was not modified.  No
formal 2M mission generation, 100K audit, 60K collection, BC training,
AWAC/SAC run, or commit was performed.

The route, protocol, native-binding, journal, common-IO, and C++ helper seams
are closed by the following evidence:

```text
GLOBAL_ROUTE_SINGLE_COMPUTE=PASS
MISSION_GENERATION_ASTAR_COUNT_PER_MISSION=1
AUDIT_ASTAR_CALL_COUNT=0
COLLECTION_ASTAR_CALL_COUNT=0
RELABEL_ASTAR_CALL_COUNT=0
MISSION_ROUTE_STORE=PASS
ROUTE_EXACT_PARITY=PASS
PROTOCOL_DEDUP=PASS
WIRE_GOLDEN_BYTES_PARITY=PASS
NATIVE_BINDING_DEDUP=PASS
DOMAIN_CTYPES_CDLL_COUNT=0
NATIVE_LIBRARY_DISCOVERY_IMPLEMENTATION_COUNT=1
VOXEL_MAP_INSTANCE_PER_PROCESS=1
COLLISION_ROUTE_SHARED_MAP=YES
AUDIT_JOURNAL=PASS
COLLECTION_JOURNAL=PASS
AUDIT_FULL_HISTORY_REWRITE_COUNT=0
COLLECTION_FULL_HISTORY_REWRITE_COUNT=0
HIGH_VOLUME_PER_ROW_FSYNC_COUNT=0
COMMON_IO_HASH_DEDUP=PASS
CXX_HELPER_DEDUP=PASS
TCP_ENDPOINT_IMPLEMENTATION_COUNT=1
COMPILER_WARNING_COUNT=0
```

The corrected O/P route comparator passed 127 synthetic cases and 1000
production-like cases with exact route status, point/world sequence, grid
sequence, cost/length bits, map metadata, and zero workspace state leaks.
The mission/Teacher comparator passed 1000 candidate rows and 100 Teacher
states with exact acceptance order, route bytes, actions, and masks.  The
route-store and journal focused tests passed, and formal downstream workers
use `set_mission_route` with the read-only store rather than replanning.

The clean devel-only build used `catkin_make` with the `xm` conda Python,
`Release`, C++17, OpenMP, and zero compiler warnings.  The current aligned
runtime identities are:

```text
BUILD_SYSTEM=catkin_make
INSTALL_USED=NO
NEW_UNITY_PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
NEW_UNITY_ASSEMBLY_SHA256=319f9e764cf51078d3b5e439214361f54aa9d5726449e26e4c0a1ad6232f21f0
NEW_BRIDGE_PATH=/home/xm/XM/xm_ws/devel/lib/planning/unity_bridge_node
NEW_BRIDGE_SHA256=462e9f7bc25d5bef63ce17b863daa256726e17de870bda04a1bcb72e79e80f2b
```

`compileall` passed; the latest full suite was `789 passed, 42 skipped,
0 failed`.  The final focused suites passed 45/1 skipped for Python
contracts, 19/0 for C# protocol/runtime contracts, and the C++ voxel fixture
passed metadata, occupancy, coordinate conversion, original collision
runtime, and shared-owner lifetime checks.  The current Player plus current
devel Bridge completed a 20-primitive single-worker run with 20/20 ACK,
COMMIT, RECEIPT, 25-frame executions, snapshot hash matches, zero telemetry
lookups, zero protocol errors, and zero leftover processes.

The bounded reliable-exact V2 chain is under
`/tmp/xm-final-12-worker-reliable-exact-smoke-20260829-r6/`:

```text
PRE_BC_E2E=PASS
TWELVE_WORKER_SMOKE=PASS
WORKER_READY=12/12
WORKER_FINISHED=12/12
SMOKE_ACCEPTED=15
SMOKE_RELIABLE_ROWS=451
SMOKE_PERSISTED_TRANSITIONS=430
LEGACY_ROWS=0
TELEMETRY_LOOKUP_COUNT=0
SNAPSHOT_MISSING_COUNT=0
STATE_DEPTH_SKEW_MAX_NS=0
FRAME_CONTRACT_FAILURES=0
COLLECTOR_ERROR_TOTAL=0
ENDPOINT_IDENTITY_CHAIN_VALID=true
TASK_CONTRACT_V2_SHA_PARITY=PASS
OBSERVATION_CONTRACT_PARITY=PASS
```

The smoke deliberately used the existing read-only historical cache
`data_back_20260826/map_data/forest_voxels_10cm.npz`, SHA256
`a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691`,
because the formal current path
`data/map_data/forest_voxels_10cm.npz` is absent.  This is the single
remaining disk-preflight blocker.  The raw point cloud was not converted or
copied, and no formal runtime manifest was frozen from the historical cache.

Therefore the optimization seams themselves are green, but the final
pre-collection gate remains closed until the formal cache is supplied or
created at the canonical current path and its identity gate is rerun:

```text
FINAL_PRE_COLLECTION_OPTIMIZATION=FAIL
CROSS_LANGUAGE_ALIGNMENT=PASS
UNITY_RUNTIME_ALIGNMENT=PASS
PRE_BC_E2E=PASS
TWELVE_WORKER_SMOKE=PASS
TASK_CONTRACT_V2_SHA_PARITY=PASS
OBSERVATION_CONTRACT_PARITY=PASS
ALGORITHM_CHANGED=NO
TEACHER_SCORING_CHANGED=NO
WIRE_BEHAVIOR_CHANGED=NO
TASK_SEMANTICS_CHANGED=NO
MISSION_ARTIFACT_SCHEMA_CHANGED=YES
VOXEL_CACHE_MMAP=DEFERRED_POST_NEW_BC
FULL_COLLECTION_READY=NO
FORMAL_COMMANDS_EMITTED=NO
FORMAL_2M_EXECUTED_BY_CODEX=NO
FORMAL_100K_EXECUTED_BY_CODEX=NO
FORMAL_60K_EXECUTED_BY_CODEX=NO
BC_TRAINING_EXECUTED=NO
AWAC_EXECUTED=NO
COMMIT_EXECUTED=NO
NEXT_USER_ACTION=FIX ONLY THE REPORTED ALIGNMENT BLOCKER
```

The new final runtime document and machine-readable runtime manifest are
intentionally not created while this disk gate is red; historical freeze
documents remain unchanged.

## FINAL PRE-COLLECTION CONSOLIDATION — CURRENT RESULT (2026-08-29)

This section supersedes the immediately preceding red-cache assessment.  The
formal cache was deterministically generated from the current formal point
cloud by `scripts/build_voxel_cache.py`; no cache was copied from `data_back`
or `/tmp`.  Earlier sections remain historical evidence and are not rewritten.

```text
FINAL_PRE_COLLECTION_CONSOLIDATION=PASS
FOREST_POINT_CLOUD_READY=YES
FOREST_POINT_CLOUD_SHA256=dfd87a5db1ab98276eda57e79de677f711da56f308a373b93e72484fa5876d40
VOXEL_CACHE_READY=YES
VOXEL_CACHE_SHA256=a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691
VOXEL_CACHE_SEMANTIC_PARITY=PASS
OCCUPIED_VOXEL_COUNT=1740947
GRID_SHAPE=[2046,2045,42]
ORIGIN_IJK=[-1023,-1020,-1]
MISSION_ROUTE_STORE_IMPLEMENTATION_COUNT=1
MISSION_ROUTE_STORE_COMPAT_WRAPPER_COUNT=0
PRODUCTION_SCRIPT_COUNT_BEFORE=28
PRODUCTION_SCRIPT_COUNT_AFTER=21
DELETED_FILE_COUNT=2
DELETED_FILES=scripts/generate_missions.py,scripts/audit_teacher_missions.py
MOVED_DIAGNOSTIC_FILE_COUNT=13
ZERO_CALLER_PRODUCTION_WRAPPER_COUNT=0
FORMAL_MISSION_PREPARATION_ENTRY_COUNT=1
MISSION_PREPARATION_INTEGRATION=PASS
PREPARATION_BUSINESS_PARITY=PASS
GENERATION_ASTAR_PER_CANDIDATE=1
AUDIT_ASTAR_CALL_COUNT=0
COLLECTION_ASTAR_CALL_COUNT=0
RELABEL_ASTAR_CALL_COUNT=0
ROUTE_STORE_MEMORY_GATE=PASS
MAX_INFLIGHT_RESULTS=24
MAX_REORDER_BUFFER_ROUTES=24
SMALL_PEAK_RSS_MB=102.726562
LARGE_PEAK_RSS_MB=102.910156
ROUTE_STORE_MEMORY_COMPLEXITY=BOUNDED
FORMAL_2M_MEMORY_SAFE=YES
FORMAL_BASE_SEED=2026
PRODUCTION_NON_2026_SEED_DEFAULT_COUNT=0
HISTORICAL_NON_2026_SEED_REFERENCE_COUNT=87
SEED_REPRODUCIBILITY=PASS
VOXEL_MAP_INSTANCE_PER_PROCESS=1
COLLISION_ROUTE_SHARED_MAP=YES
CATKIN_MAKE=PASS
COMPILER_WARNING_COUNT=0
FULL_TESTS=PASS
FULL_TEST_PASS_COUNT=800
FULL_TEST_SKIP_COUNT=42
FULL_TEST_FAILURE_COUNT=0
UNITY_PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
UNITY_ASSEMBLY_SHA256=319f9e764cf51078d3b5e439214361f54aa9d5726449e26e4c0a1ad6232f21f0
BRIDGE_SHA256=462e9f7bc25d5bef63ce17b863daa256726e17de870bda04a1bcb72e79e80f2b
PRE_BC_E2E=PASS
TWELVE_WORKER_SMOKE=PASS
TASK_CONTRACT_V2_SHA_PARITY=PASS
OBSERVATION_CONTRACT_PARITY=PASS
ALGORITHM_CHANGED=NO
TEACHER_SCORING_CHANGED=NO
WIRE_BEHAVIOR_CHANGED=NO
TASK_SEMANTICS_CHANGED=NO
MISSION_ARTIFACT_SCHEMA_CHANGED=YES
FULL_COLLECTION_READY=YES
FORMAL_COMMANDS_EMITTED=YES
FORMAL_MISSION_PREPARATION_EXECUTED_BY_CODEX=NO
FORMAL_60K_EXECUTED_BY_CODEX=NO
BC_TRAINING_EXECUTED=NO
AWAC_EXECUTED=NO
COMMIT_EXECUTED=NO
```

## Final preparation parallelism and ETA logging

Date: 2026-08-29

```text
PREPARATION_FULL_PIPELINE_PARALLELISM=PASS
RAW_SAMPLING_OWNER=parent deterministic raw-attempt generator
SAFETY_CHECK_OWNER=worker initializer state
GLOBAL_ROUTE_OWNER=worker initializer state
TEACHER_AUDIT_OWNER=worker initializer state
JOURNAL_COMMIT_OWNER=parent ordered publisher
PARENT_MISSION_SAFETY_CHECK_CALL_COUNT=0 (workers=12 path)
PARENT_GLOBAL_ROUTE_PLAN_CALL_COUNT=0 (workers=12 path)
PARENT_TEACHER_AUDIT_CALL_COUNT=0 (workers=12 path)
WORKER_MPL_INIT_COUNT_PER_PROCESS=1
WORKER_NATIVE_GEOMETRY_INIT_COUNT_PER_PROCESS=1
WORKER_VOXEL_MAP_INSTANCE_COUNT=1
WORKER_COLLISION_ROUTE_SHARED_MAP=YES
WORKER_TEACHER_INIT_COUNT_PER_PROCESS=1
COLLISION_THREADS_PER_WORKER=1
MAX_INFLIGHT_RESULTS=24
MAX_REORDER_BUFFER_RESULTS=24
PREPARATION_MEMORY_COMPLEXITY=BOUNDED
ARTIFACT_SCHEMA_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
```

The parallel preparation path now advances the same deterministic raw RNG
stream for every worker count.  Worker processes initialize the MPL, one
`NativeGeometryContext`, the shared-map collision checker/route planner, and
the Teacher once.  Safety, one route plan, route gates, and Teacher audit stay
inside the worker; the parent only reorders by attempt and publishes the
existing journals/route store.  The existing CSV journal and route-store
schemas were not changed.

The common progress owner now supports rolling and EWMA rates, calibration,
explicit ETA completion counters/targets, and duration formatting beyond 24
hours.  Preparation progress uses passing as its ETA counter and collection
progress uses accepted rows.  `collection_monitor.py` aggregates worker
structured progress JSON and reports accepted/sec, target, percentage, elapsed,
and ETA.  Unknown rates are serialized as `null`; `NaN`/`inf` are not emitted
to progress artifacts.

### Bounded parity and performance evidence

All runs used `conda activate xm`, the devel native libraries, the formal
voxel cache, seed `2026`, max inflight `24`, collision threads `1`, and wrote
only to `/tmp`.

```text
W1: 8 passing / 13 attempts, 20.15 s, peak RSS 104480 KB
W3: 8 passing / 13 attempts, 15.46 s, peak RSS 101492 KB
W6: 8 passing / 13 attempts,  8.20 s, peak RSS 101728 KB
W12: 8 passing / 13 attempts, 7.15 s, peak RSS 101696 KB
W1_PASSING_PER_SEC=0.397
W3_PASSING_PER_SEC=0.517
W6_PASSING_PER_SEC=0.976
W12_PASSING_PER_SEC=1.119
W12_SPEEDUP_VS_W1=2.82
W12_SPEEDUP_VS_W3=2.16
PREPARATION_CPU_SCALING=PASS (bounded fixture gate)
MEAN_ACTIVE_CPU_EQUIVALENT=NOT_MEASURED
PEAK_ACTIVE_CPU_EQUIVALENT=NOT_MEASURED
```

The four runs produced exact hashes for the business outputs:

```text
mission_candidates.csv       a9473a4c10f771bd2c30e102032da98540097981b16897cd5ec891f516f730a9
missions.csv                 af3185ce58eba17002e60b8b0d468d9ff8ea394f5b590ceb7291a8ef148eed0c
mission_routes.meta.json     f4281a3bf3a948e07d870c10c0cf0c4eb6d23e7ad07f8a39b32e4861f2047733
mission_routes.points.bin    b1c10842f908992d629e92f8fa8a51645e1fff40b17826aeab5af8c792c92953
mission_routes.offsets.bin   7a0066eeaa383f712133acd625cba1191ddfccec7de620c82faab77729fa65fd
```

Candidate, routed-candidate, audit, and passing journals were also bytewise
identical across W1/W3/W6/W12.  The parallel resume test interrupted after a
worker failure and resumed only the uncommitted attempt; committed routes were
not replanned.

### Tests and scope

```text
compileall=PASS
focused preparation/progress/monitor tests=15 passed
full pytest=835 passed, 42 skipped, 0 failed
formal 100K preparation=NO
formal 60K collection=NO
BC training=NO
AWAC=NO
commit=NO
```

Modified in this phase:

- `python/planning/common/progress.py`
- `python/planning/common/__init__.py`
- `python/planning/mission/sampling.py`
- `python/planning/mission/preparation.py`
- `python/planning/teacher/parallel_collection.py`
- `python/planning/teacher/rollout_collector.py`
- `python/planning/teacher/collection_monitor.py`
- `tests/test_preparation_parallelism_eta.py`
- `tests/test_progress_logging.py`
- `tests/test_collection_eta_monitor.py`
- `docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md`

No Unity, C++, Mission sampling predicate, collision semantics, A*, Teacher
scoring, route-store schema, CSV journal schema, task/observation contract,
BC, or AWAC changes were made.  A real Unity collection smoke was not started
in this phase; the evidence above is preparation-only bounded native smoke.

`NEXT_PHASE=FORMAL PREPARATION / COLLECTION ONLY AFTER USER EXPLICITLY STARTS IT`

`PRODUCTION_SCRIPT_COUNT_*` is the CMake install surface: 24 Python plus 4
shell entries before cleanup, and 17 Python plus 4 shell entries after
cleanup.  The source tree still contains RL entries and source-only internal
workers.  The two deleted files were zero-caller legacy formal wrappers; the
13 diagnostic/reference files are source-only under `tools/diagnostics`.

The formal cache metadata is `voxel_size=0.1`, `file_frame=unity`,
`data_offset_bytes=4` (the requested `-1` auto-detect mode), with 24,818,315
kept points.  Its occupied keys, origin, shape, voxel size, and metadata were
exactly equal to the read-only historical oracle.  The current native build
targets are `planning_depth_safety`, `planning_voxel_map`,
`planning_collision_checker`, `planning_global_route`, `map_loader_node`,
`visualization_node`, `voxel_map_parity_fixture`, and `unity_bridge_node`.

The route-memory gate used the same bounded preparation path.  The small run
produced 13 candidates and 12 passing missions with a 27,405-byte route store;
the large run produced 472 candidates and 400 passing missions with a
943,147-byte route store.  Peak RSS was 105,192 kB and 105,380 kB,
respectively.  `max_sampling_attempts=0` is a global scale-safe bound and
resolves to `max(200000, 4 * max_candidates)`; the formal 2M bound is
8,000,000.

The final bounded 12-worker V2 smoke used the current formal cache, Player,
and devel Bridge.  It reached 22 accepted episodes from 24 attempts, 657
reliable rollout rows, and 632 persisted transitions.  All reliable-exact
gates were zero: legacy rows, telemetry lookups, snapshot misses, state/depth
skew, frame failures, collector errors, port collisions, cross-talk, and
orphan processes.  Relabel, dataset audit, depth masks, and BC mmap all
passed; BC training, evaluation, and AWAC/SAC were intentionally not run.

The full post-fix suite is `800 passed, 42 skipped, 0 failed`; compileall,
native loading, voxel fixture, route parity, protocol checks, and the bounded
Pre-BC artifact chain are green.  The current machine-readable freeze is
`docs/pre_collection_runtime_final_v3.json`, with the human-readable freeze
in `docs/PRE_COLLECTION_RUNTIME_FINAL_V3.md`.

## Final handoff

```text
NEXT_USER_ACTION=EXECUTE FINAL FORMAL COMMANDS
```

The formal command handoff is emitted by the final response only.  No formal
2M preparation, 100K collection/audit run, 60K collection, BC training,
AWAC/SAC run, Unity modification, O modification, or commit was performed.

## Final preparation parallelism and ETA logging — verification closure

Date: 2026-08-29

The follow-up verification was run against the current canonical Planning
tree after the monitor completion-state fix.  The full suite and a fresh
bounded Unity collection smoke were executed; no formal dataset path was
written.

```text
PREPARATION_FULL_PIPELINE_PARALLELISM=PASS
RAW_SAMPLING_OWNER=parent deterministic raw-attempt generator
SAFETY_CHECK_OWNER=worker initializer state
GLOBAL_ROUTE_OWNER=worker initializer state
TEACHER_AUDIT_OWNER=worker initializer state
JOURNAL_COMMIT_OWNER=parent ordered publisher
PARENT_MISSION_SAFETY_CHECK_CALL_COUNT=0
PARENT_GLOBAL_ROUTE_PLAN_CALL_COUNT=0
PARENT_TEACHER_AUDIT_CALL_COUNT=0
WORKER_MPL_INIT_COUNT_PER_PROCESS=1
WORKER_NATIVE_GEOMETRY_INIT_COUNT_PER_PROCESS=1
WORKER_VOXEL_MAP_INSTANCE_COUNT=1
WORKER_COLLISION_ROUTE_SHARED_MAP=YES
WORKER_TEACHER_INIT_COUNT_PER_PROCESS=1
COLLISION_THREADS_PER_WORKER=1
MAX_INFLIGHT_RESULTS=24
MAX_REORDER_BUFFER_RESULTS=24
WORKER_COUNT_DETERMINISM=PASS
BUSINESS_OUTPUT_PARITY=PASS
ROUTE_EXACT_PARITY=PASS
PREPARATION_RESUME=PASS
ROUTE_STORE_MEMORY_GATE=PASS
PREPARATION_MEMORY_COMPLEXITY=BOUNDED
PREPARATION_CPU_SCALING=PASS
W1_PASSING_PER_SEC=0.397
W3_PASSING_PER_SEC=0.517
W6_PASSING_PER_SEC=0.976
W12_PASSING_PER_SEC=1.119
W12_SPEEDUP_VS_W1=2.82
W12_SPEEDUP_VS_W3=2.16
MEAN_ACTIVE_CPU_EQUIVALENT=NOT_MEASURED
PREPARATION_PROGRESS_LOGGING=PASS
PREPARATION_ETA=PASS
COLLECTION_PROGRESS_LOGGING=PASS
COLLECTION_ETA=PASS
MONITOR_ETA=PASS
ETA_ZERO_RATE_HANDLING=PASS
ETA_RESUME_HANDLING=PASS
PREPARATION_LOG_PATTERN=PROGRESS component=teacher_mission_preparation ... passing_per_s=... elapsed=... eta_h=...
COLLECTION_LOG_PATTERN=PROGRESS component=teacher_parallel_collection ... accepted_per_s=... elapsed=... eta_h=...
```

The fresh bounded collection smoke used
`/tmp/xm-preparation-collection-eta-smoke-20260829-2vjpXM`, the current
Player, the current devel Bridge, the formal voxel cache, and 12 workers with
one collision thread each.  It passed 12/12 worker readiness and completion,
24 attempts, 22 accepted, 657 reliable rows, and 632 persisted transitions.
Legacy rows, telemetry lookups, snapshot misses, state/depth skew, frame
failures, collector errors, port collisions, cross-talk, and orphan processes
were all zero.  The collection log contained calibration (`eta_h=unknown`)
and completion (`eta_h=0.00`) states; the structured monitor reported
`eta_status=READY` and `eta_h=0.0` after the target was reached.  Worker,
Unity, and Bridge logs were present (12, 24, and 12 respectively), and all
JSON artifacts contained finite numeric values.

```text
SMOKE_ATTEMPTED=24
SMOKE_ACCEPTED=22
SMOKE_RELIABLE_ROWS=657
SMOKE_PERSISTED_TRANSITIONS=632
SMOKE_LEGACY_ROWS=0
SMOKE_TELEMETRY_LOOKUP=0
SMOKE_SNAPSHOT_MISSING=0
SMOKE_SKEW_MAX_NS=0
SMOKE_FRAME_FAILURES=0
SMOKE_COLLECTOR_ERRORS=0
SMOKE_WORKER_READY=12/12
SMOKE_WORKER_FINISHED=12/12
SMOKE_ENDPOINT_IDENTITY_CHAIN=PASS
SMOKE_TASK_CONTRACT_SCHEMA=2
SMOKE_TASK_CONTRACT_SHA256=2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df
SMOKE_OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
SMOKE_UNITY_PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
SMOKE_BRIDGE_SHA256=462e9f7bc25d5bef63ce17b863daa256726e17de870bda04a1bcb72e79e80f2b
SMOKE_MONITOR_ETA_STATUS=READY
SMOKE_MONITOR_ETA_H=0.0
```

The final preparation smoke used
`/tmp/xm-preparation-eta-final-20260829-fLNbbF` with the same seed, native
cache, 12 workers, 24-result bound, and one collision thread.  It produced
9 candidates, 8 routed/audited/passing missions from 13 attempts.  Its
progress artifact reported `TARGET_AT_RISK=NO`,
`ATTEMPT_BUDGET_AT_RISK=NO`, `eta_status=READY`, and `eta_h=0.0`; the business
candidate/mission/journal outputs remained byte-identical to the worker-count
parity fixture.  The follow-up logging changes cover monitor completion-state
precedence (an aggregate target already reached reports `READY` even when a
stale worker file says `CALIBRATING`), budget-risk telemetry, monotonic
elapsed-time measurement, and finite-value hardening.  No mission, route,
collision, Teacher scoring, protocol, Unity, C++, BC, AWAC/SAC, or artifact
schema behavior changed.

Preparation progress additionally projects the remaining passing target
against the observed candidate and attempt pass rates.  It records remaining
budgets, estimated totals, and `TARGET_AT_RISK` /
`ATTEMPT_BUDGET_AT_RISK` as runtime telemetry; insufficient observations are
reported as `UNKNOWN`, and the fields do not enter any artifact identity.

```text
compileall=PASS
FOCUSED_TESTS=17 passed
FULL_TESTS=PASS
FULL_TEST_PASS_COUNT=838
FULL_TEST_SKIP_COUNT=42
FULL_TEST_FAILURE_COUNT=0
PRE_BC_E2E=PASS
TWELVE_WORKER_SMOKE=PASS
FORMAL_100K_EXECUTED=NO
FORMAL_60K_EXECUTED=NO
BC_TRAINING_EXECUTED=NO
AWAC_EXECUTED=NO
ARTIFACT_SCHEMA_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
COMMIT_EXECUTED=NO
```

`NEXT_USER_ACTION=EXECUTE NEW V4 FORMAL PREPARATION COMMAND`

---

## MISSION_PREPARATION_VALIDATION_COUNT_SEMANTICS — 2026-08-30

The existing formal preparation artifact was validated without rerunning
preparation and without modifying any data artifact. The initial validator
failure was reproducible as:

```text
MISSION_VALIDATION=FAIL
ERROR=preparation candidate count mismatch
```

The writer and artifact establish these distinct owners:

```text
sampling_attempts       = 223353  raw sampling attempts
candidate_count         = 133584  safe candidates sent to route/A* and audit
routed_count            = 104108  routed candidates persisted to CSV/RouteStore
audited_count           = 104108  routed candidates audited by Teacher feasibility gate
passing_count           = 100000  final missions persisted to missions.csv
rejected_route_count    = 29476   candidate_count - routed_count
```

`mission_candidates.csv` therefore has 104,108 data rows, while
`mission_candidates.preparation.json` has `candidate_count=133584`. The old
validator incorrectly compared those two levels and also required the
RouteStore count to equal the total safe-candidate count. The corrected
invariants are:

```text
sampling_attempts >= candidate_count >= routed_count >= passing_count
routed_count == audited_count
candidate_count - routed_count == rejected_route_count
len(routed candidate CSV) == routed_count
RouteStore.mission_count == routed_count
len(final missions CSV) == passing_count == expected_passing
generation_astar_calls == candidate_count
all final/candidate route references are in-range and provenance-matched
```

The canonical validator now reads both preparation metadata and its progress
metadata, reports the count levels separately, and retains fail-closed checks
for route references, RouteStore hashes/contracts, mission SHA, candidate SHA,
seed, and A* call accounting.

Validation result:

```text
CANDIDATE_CSV_ROW_COUNT=104108
MISSION_CSV_ROW_COUNT=100000
ROUTE_STORE_COUNT=104108
MISSION_VALIDATION=PASS
PREPARATION_RERUN=NO
ARTIFACTS_MODIFIED=NO
ALGORITHM_CHANGED=NO
ARTIFACT_SCHEMA_CHANGED=NO
```

Regression coverage in
`tests/test_mission_validation_count_semantics.py` covers candidate > routed >
passing success, RouteStore/routed mismatch, out-of-range final route index,
passing-count mismatch, and candidate SHA mismatch. The focused suite passed
45 tests; compileall passed; the full Planning suite passed 846 tests with 42
gated skips.

The implementation and regression test are the only code/test changes for this
fix. No formal collection was started.

---

## COLLECTION_SCALING_AND_TEACHER_RUNTIME_V1 — 2026-08-29

This bounded scaling audit is complete. It did not execute formal 100K mission
preparation, formal 60K collection, BC training, AWAC/SAC, Unity changes, or a
commit. The formal project remains `/home/xm/XM/xm_ws/src/planning`.

### Phase accounting

`PRIMITIVE_EXECUTION_CONTAINS_RESULT_WAIT=YES`. `primitive_execution_ms` is
inclusive of reliable command/result waiting and endpoint/depth/commit work.
The only additive outer decomposition is:

```text
reset + teacher_decision + primitive_execution(inclusive)
+ python_postprocess + episode_finalize + npz_write + uninstrumented_idle
```

Nested `result_wait`, `command_wait`, `snapshot_wait`, `depth_wait`, `commit`,
and crossed-boundary action-mask timings are diagnostic only and were not added
to wall time.

### Worker contract and scaling gate

The audited managed collection bound was a historical `[1,12]` safety check,
not a protocol/ABI/Unity/Bridge limit. It is now explicitly `[1,24]` in
`python/planning/runtime/ports.py`; `config/pre_bc.yaml` still defaults to 12,
and RL/SAC bounds are unchanged. The per-worker port layout, identity, nonce,
protocol, and artifact schema were not changed.

```text
WORKER_CONTRACT_BEFORE=1-12
WORKER_CONTRACT_AFTER=1-24
WORKER_24_PORT_PREFLIGHT=PASS
PORT_COUNT=240
PORT_COLLISION_COUNT=0
```

Bounded startup passed for W12/W16/W20/W24 with unique runtime bindings,
cross-talk=0, wrong-worker ACK/snapshot=0, and orphan cleanup=0. No OOM,
Unity/Bridge crash, or GPU failure occurred. W24 reached approximately
50.01 GiB peak tree RSS and 8.95 GiB minimum available memory; this is a
long-run headroom warning, not a failed bounded resource gate.

| workers | accepted/attempted | wall (s) | accepted/s | reliable rows/s | transitions/s |
|---:|---:|---:|---:|---:|---:|
| 12 | 111/126 | 227.561 | 0.487781 | 14.721327 | 13.943514 |
| 16 | 112/127 | 184.994 | 0.605425 | 18.265457 | 17.308669 |
| 20 | 111/126 | 172.854 | 0.642160 | 19.380518 | 18.356532 |
| 24 | 113/128 | 168.524 | 0.670528 | 20.216705 | 19.166410 |

### Teacher and parity

The 1,000-decision Teacher benchmark showed inclusive `score_state` as the
dominant component (72.914 ms mean); pose reachability (23.700 ms) and
candidate evaluation (15.847 ms) followed. A vectorized finite-min trial
preserved exact output hashes but improved mean runtime only 0.13%, so it was
reverted. No Teacher scoring, beam, sorting, tie-break, collision, or float
semantics were retained from that trial.

```text
TEACHER_RUNTIME_EXACT_PARITY=PASS
CANDIDATE_ACCEPT_VECTOR_PARITY=PASS
TEACHER_VALID_ACTION_MASK_PARITY=PASS
```

The final correctly configured W24 bounded benchmark completed 522 accepted
from 583 attempts in 619.823 s: 0.842175912 accepted/s, 15,845 reliable rows,
and 14,920 persisted transitions. All reliability counters were zero and the
endpoint identity chain and process cleanup passed. Its output and resource
artifacts are under `/tmp/xm-collection-scaling-teacher-runtime-v1-20260829/`.

### Current gate

```text
COLLECTION_SCALING_AND_TEACHER_RUNTIME_V1=PASS
BEST_COLLECTION_WORKERS=24
BEST_COLLECTION_ACCEPTED_PER_SEC=0.842175912
ETA_60K_OPTIMISTIC_HOURS=19.7900
ETA_60K_NORMAL_HOURS=22.0356
ETA_60K_PESSIMISTIC_HOURS=24.8560
ETA_100K_PREPARATION_HOURS=10.1558
ESTIMATED_TOTAL_TO_BC_TRAINING_READY_HOURS=UNKNOWN
compileall=PASS
FULL_TESTS=841 passed, 42 skipped, 0 failed
PRE_BC_E2E=PASS (frozen existing evidence)
SELECTED_WORKER_SMOKE=PASS
UNITY_CHANGED=NO
FORMAL_100K_EXECUTED=NO
FORMAL_60K_EXECUTED=NO
BC_TRAINING_EXECUTED=NO
AWAC_EXECUTED=NO
COMMIT_EXECUTED=NO
NEXT_USER_ACTION=EXECUTE UPDATED FORMAL DATA PIPELINE
```

Detailed evidence is recorded in
`docs/COLLECTION_SCALING_AND_TEACHER_RUNTIME_V1.md`.

---

## COLLECTION_LONG_RUN_STABILITY_W12_W16 — 2026-08-30

W8 was skipped as explicitly requested.  The completed W4 diagnostic ran for
approximately 1.19 hours and finished naturally with 803 accepted / 907
attempted, 24,524 reliable rows, all collection quality counters zero, and
complete runtime cleanup.  Its resource evidence is under
`/tmp/xm-collection-runtime-lifecycle-memory-fix-v1-w4-final-retry-20260830/`.

The W12 diagnostic used the formal 100K mission pool and MissionRouteStore as
read-only inputs, current devel Bridge, formal Unity Player, formal Voxel
cache, formal MPL, Task V2, and
`reliable_exact_endpoint_snapshot`.  It wrote only to
`/tmp/xm-w12-long-stability-20260830/`.  Per the user's latest instruction it
was stopped after one hour through the existing stop-file lifecycle, rather
than being extended to the original two-hour gate.  It finished with:

```text
COLLECTION_LONG_RUN_STABILITY_W12_W16=FAIL (W16 not run; original two-hour W12 gate not completed)
W4_EVIDENCE_SUFFICIENT=YES
W12_LONG_RUN_STABLE=YES (one-hour user-bounded diagnostic)
W12_TEST_HOURS=1.010984
W12_ATTEMPTED=2249
W12_ACCEPTED=1989
W12_ACCEPTED_PER_SEC=0.546497
W12_ACCEPTANCE_RATE=0.884393
W12_RELIABLE_ROWS=61075
W12_RELIABLE_ROWS_PER_SEC=16.780953
W12_BRIDGE_RSS_START_GB=0.281
W12_BRIDGE_RSS_END_GB=0.280
W12_BRIDGE_RSS_GROWTH_MB_PER_HOUR_PER_WORKER=0.17
W12_COLLECTOR_RSS_START_GB=10.718
W12_COLLECTOR_RSS_END_GB=10.735
W12_COLLECTOR_RSS_GROWTH_MB_PER_HOUR_PER_WORKER=1.76
W12_UNITY_RSS_START_GB=3.436
W12_UNITY_RSS_END_GB=3.582
W12_MAX_SWAP_GB=0
W12_RAM_HEADROOM_MIN_GB=42.91
W12_SYSTEM_RESPONSIVENESS=PASS
W12_QUALITY_COUNTERS=ALL_ZERO
W12_WORKERS_READY=12/12
W12_WORKERS_FINISHED=12/12
W12_ENDPOINT_IDENTITY_CHAIN=PASS
W12_ETA_60K_HOURS=30.497 (diagnostic rate only; not a release estimate)
W16_TESTED=NO
W16_LONG_RUN_STABLE=NO
FORMAL_SAFE_COLLECTION_WORKERS=NOT_PROMOTED
FORMAL_SAFE_ACCEPTED_PER_SEC=NOT_PROMOTED
FORMAL_60K_ETA_HOURS=NOT_AVAILABLE
CURRENT_BRIDGE_SHA256=667c266e96211aabb5a0cd25b8b35b303903d35cbe2d95be8f9d2bbe5e63d789
FULL_TESTS=FAIL (862 passed, 42 skipped, 1 stale Bridge identity fixture failure)
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
UNITY_CHANGED=NO
ALGORITHM_CHANGED=NO
FORMAL_60K_EXECUTED=NO
NEXT_USER_ACTION=DECIDE WHETHER TO AUTHORIZE THE ORIGINAL TWO-HOUR W12 GATE OR STOP AND RECONCILE THE STALE BRIDGE FIXTURE
```

The W12 final `collection_summary.json` reports endpoint identity valid,
legacy rows 0, telemetry lookup 0, snapshot missing 0, state/depth skew 0 ns,
frame failures 0, collector errors 0, and all twelve worker progress files
completed.  The monitor recorded process count 85, FD count 2,258, swap 0,
minimum available RAM 42.91 GiB, load/I/O responsiveness `PASS`, and no
port-collision, cross-talk, or orphan indicators.

The final full-suite rerun was made with the current devel native libraries
explicitly bound: 862 passed and 42 skipped.  The only failure is the
read-only historical runtime fixture expecting Bridge SHA
`462e9f7bc25d5bef63ce17b863daa256726e17de870bda04a1bcb72e79e80f2b`; the
current W4/W12 devel Bridge SHA is
`667c266e96211aabb5a0cd25b8b35b303903d35cbe2d95be8f9d2bbe5e63d789`.  No
fixture, formal artifact, Unity tree, task contract, observation contract, or
algorithm was changed.

---

## W12_ACCEPTANCE_AND_W16_FORMAL_RELEASE — 2026-08-30

The user accepted the one-hour W12 gate; W12 was not rerun and W8 was not
added.  The stale Bridge identity was classified correctly as a current
runtime fixture issue: `docs/pre_collection_runtime_final_v3.json` now stores
the current devel Bridge SHA in `runtime_identity`, while retaining the old
`462e9f7bc25d5bef63ce17b863daa256726e17de870bda04a1bcb72e79e80f2b` under an
explicit `HISTORICAL_RUNTIME_FIXTURE` identity record.  Unity, Task V2,
Observation Contract, MPL, route store, and algorithms were unchanged.

```text
W12_USER_ACCEPTANCE_GATE=PASS
STALE_BRIDGE_FIXTURE_FIXED=YES
CURRENT_BRIDGE_SHA256=667c266e96211aabb5a0cd25b8b35b303903d35cbe2d95be8f9d2bbe5e63d789
FULL_TESTS=PASS
FULL_TEST_PASS_COUNT=864
FULL_TEST_SKIP_COUNT=42
FULL_TEST_FAILURE_COUNT=0
```

The W16 estimate is based on the W12 stable resource samples rather than a
new W16 run.  W12 minimum available RAM was 42.908756 GiB and its maximum
process-tree RSS was 16.256531 GiB.  Scaling that observed tree from 12 to 16
workers adds 5.418844 GiB, leaving an estimated 37.489913 GiB minimum
headroom.  This exceeds the 15 GiB gate; W16 remains subject to the user's
first-hour canary rules.

```text
W16_ESTIMATED_RAM_HEADROOM_GB=37.489913
W16_FORMAL_READY=YES
FORMAL_COLLECTION_WORKERS=16
FORMAL_COLLECTION_RUN=flight_reliable_exact_teacher_seed2026_20260830_v7
FORMAL_60K_EXECUTED_BY_CODEX=NO
NEXT_USER_ACTION=START W16 FORMAL COLLECTION AND USE FIRST HOUR AS CANARY
```

---

## XMFLIGHT_FULL_PIPELINE_PERFORMANCE_OPTIMIZATION_AND_FORMAL_FREEZE — 2026-09-01

The bounded performance-freeze task completed two independent fresh 3K
baseline/optimized pipelines. The optimized pipeline did not reuse baseline
trajectories or derived artifacts. Direct mmap relabel/mask production,
single-pass BC mmap construction, fast/full BC validation, bounded Teacher
selection, and disk-safe collection lifecycle logging are retained. A measured
minimal-clearance collision experiment was reverted after a 4.95% regression.

~~~text
OPTIMIZATION_FREEZE=PASS
ALGORITHM_CHANGED=NO
DATA_LAYOUT_CHANGED=NO
UNITY_CHANGED=NO
FORMAL_SCALE_EXECUTED=NO
FORMAL_COLLECTION_EXECUTED=NO
FORMAL_BC_TRAINING_EXECUTED=NO
AWAC_SAC_EXECUTED=NO
CATKIN_MAKE_RELEASE=PASS
COMPILEALL=PASS
FULL_PYTEST=869 passed, 42 skipped, 0 failed
FRESH_BASELINE_3K=PASS
FRESH_OPTIMIZED_3K=PASS
ARRAY_AND_LAYOUT_PARITY=PASS
FORMAL_SAFE_COLLECTION_WORKERS=12
FORMAL_DISK_GATE=BLOCKED
~~~

The current measured free space is approximately 207.55 GiB. The conservative
formal output estimate is 210.039 GiB, requiring 225.039 GiB with the
operational margin; no deletion was performed. See
docs/FULL_PIPELINE_PERFORMANCE_OPTIMIZATION_FINAL_V3.md,
docs/full_pipeline_performance_optimization_final_v3.json, and
docs/FORMAL_TEACHER_TO_BC_END_TO_END_COMMANDS_V3.md for the complete evidence,
gate, and user handoff. Formal collection and later policy/RL stages remain
separate, explicitly gated operations.

---

## FORMAL_COLLECTION_W20_PROMOTION_V3_1 — 2026-09-01

The V3.1 task evaluated a Collection-only promotion from 12 to 20 workers.
V3 remains the historical 12-worker freeze. No Unity, Bridge protocol,
Teacher, trajectory, task, observation, or downstream pipeline behavior was
changed.

~~~text
FORMAL_COLLECTION_W20_PROMOTION_V3_1=FAIL
FORMAL_SAFE_COLLECTION_WORKERS=12
OTHER_PIPELINE_WORKER_CONFIG_CHANGED=NO
COLLECTION_TRAJECTORY_SEMANTICS_CHANGED=NO
COLLECTION_ALGORITHM_CHANGED=NO
UNITY_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
DATA_LAYOUT_CHANGED=NO
FORMAL_60K_EXECUTED=NO
~~~

The bounded run used the validated V4 missions and current frozen runtime at
`/tmp/xm-formal-w20-promotion-v31`: 20/20 workers ready and finished, 420
accepted from 469 attempts, 0.798889164 accepted/s by total wall time, and
0.895522388 acceptance rate. All 420 accepted NPZ files passed
`load_rollout_episode(validate=True)`. Collection validation passed with
zero legacy rows, telemetry lookups, snapshot misses, skew, frame failures,
collector errors, port collisions, cross-talk, duplicate identities, and
orphan processes. Graceful cleanup and the disk watchdog passed.

The hard promotion gate remains closed because 0.697266 GiB swap usage was
observed, while the requirement is zero. The resource monitor also omitted
its available-memory and swap columns under the host locale; a spot sample
showed 36.399048 GiB available, but no complete-run minimum can be claimed.
The V3 commandbook and all non-Collection worker settings therefore remain
unchanged. See
`docs/FULL_PIPELINE_PERFORMANCE_OPTIMIZATION_FINAL_V3_1.md`,
`docs/full_pipeline_performance_optimization_final_v3_1.json`, and
`docs/FORMAL_TEACHER_TO_BC_END_TO_END_COMMANDS_V3_1.md` for the complete
gate and evidence.

---

## OFFLINE_WORKER_12_VS_20_PROMOTION_500_SMOKE_V1 — 2026-09-01

This bounded offline benchmark evaluated Mission Preparation, Teacher
relabel, and depth-mask worker promotion independently. The benchmark used
fresh, identical 500-passing Mission inputs under
`/tmp/xm-worker-12-vs-20-500-v1`. The existing V3.1 Collection decision is
unaffected: formal Collection remains at 12 workers.

~~~text
OFFLINE_WORKER_12_VS_20_PROMOTION_500_SMOKE_V1=FAIL
MISSION_BENCHMARK=PASS
MISSION_W12_WALL_SEC=145.024
MISSION_W20_WALL_SEC=119.826
MISSION_SPEEDUP_20_VS_12=1.210288251
MISSION_THROUGHPUT_GAIN_PERCENT=21.028825
MISSION_PARITY=PASS
MISSION_WORKERS_PROMOTED_TO_20=YES
MISSION_DEFAULT_WORKERS=20
MISSION_DEFAULT_MAX_INFLIGHT=40
LABEL_MASK_BENCHMARK=BLOCKED
LABEL_MASK_BLOCKER=canonical accepted-only 500 rollout fixture is absent
LABEL_WORKERS_PROMOTED_TO_20=NO
MASK_WORKERS_PROMOTED_TO_20=NO
COLLECTION_DEFAULT_WORKERS=12
BC_DATALOADER_WORKERS=8
FORMAL_60K_EXECUTED=NO
FORMAL_100K_MISSIONS_EXECUTED=NO
BC_TRAINING_EXECUTED=NO
AWAC_SAC_EXECUTED=NO
COMPILEALL=PASS
TARGETED_TESTS=27 passed, 0 failed
FULL_TESTS=868 passed, 42 skipped, 2 failed (historical backup absent)
ALGORITHM_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
DATA_LAYOUT_CHANGED=NO
~~~

Mission candidates, missions, and RouteStore metadata/offset/point artifacts
were byte-for-byte identical between 12 and 20 workers. Both runs reached 500
passing missions from 1,111 sampling attempts with no worker crash, no swap,
and no I/O wait. Minimum available memory was 55.286102 GiB for 12 workers
and 54.697879 GiB for 20 workers. The active process pool matched the request:
12 and 20 workers, with in-flight windows 24 and 40 respectively.

Label and depth-mask stages were not run because the canonical
`flight_reliable_exact_teacher_seed2026_20260830_v8_bc59930` accepted-only
rollout index/episodes are absent from this checkout. No old checkout or
partial mission run was substituted. Their worker defaults and formal
overrides therefore remain unchanged pending a valid fixture. See
`docs/FULL_PIPELINE_PERFORMANCE_OPTIMIZATION_FINAL_V3_2.md` and
`docs/FORMAL_TEACHER_TO_BC_END_TO_END_COMMANDS_V3_2.md` for the current
stage-specific profile.

---

## OFFLINE_LABEL_MASK_WORKER_PROMOTION_500_V2 — 2026-09-01

The bounded offline Label/Mask worker promotion gate is closed independently
of Collection. The documented validated Optimized 3K artifact was absent, so
the permitted fresh 500-accepted reliable-exact smoke was used as the source
for a fixed benchmark fixture. The source collection was not modified.

```text
OFFLINE_LABEL_MASK_WORKER_PROMOTION_500_V2=PASS
FIXTURE_SOURCE=/tmp/xm-label-mask-worker-benchmark-v2/collection500
FIXTURE_PATH=/tmp/xm-label-mask-worker-benchmark-v2/fixture500
FIXTURE_INDEX_SHA256=b3936eb1528e907b289e7d110d46e06085ecee591cee044dfae314682d7f45a7
FIXTURE_MANIFEST_SHA256=d1d335137207927500d909e64d4f335ccd74a3a3cea25238dd537bdd70a8cbcb
FIXTURE_EPISODES=500
FIXTURE_TRANSITIONS=14290
FIXTURE_FIRST_EPISODE_ID=0
FIXTURE_LAST_EPISODE_ID=555
FIXTURE_INVALID_NPZ=0
FIXTURE_SOURCE_MODIFIED=NO
FIXTURE_SOURCE_NPZ_BYTES_DUPLICATED=0
OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
```

The fixture contains 500 hard-linked episode NPZ files selected by canonical
episode-id ordering. Every episode passed `load_rollout_episode(validate=True)`
and `validate_rollout_provenance`; no NPZ bytes were duplicated.

All Label and Mask runs used identical input episodes, RouteStore, collision
cache, Teacher beam, direct-mmap publication, and safety parameters. Arrays,
episode order, transition mappings, and business metadata were exact across
worker counts. Cross-artifact Label/Mask provenance validation passed.

```text
LABEL_W8_TRANSITIONS_PER_SEC=85.076625
LABEL_W12_TRANSITIONS_PER_SEC=105.973999
LABEL_W20_TRANSITIONS_PER_SEC=123.461033
LABEL_SPEEDUP_12_VS_8=24.563003%
LABEL_SPEEDUP_20_VS_8=45.117455%
LABEL_PARITY=PASS
LABEL_DEFAULT_WORKERS=20

MASK_W8_TRANSITIONS_PER_SEC=7043.503790 (three-run median)
MASK_W12_TRANSITIONS_PER_SEC=8210.814082 (three-run median)
MASK_W20_TRANSITIONS_PER_SEC=8976.467885 (three-run median)
MASK_SPEEDUP_12_VS_8=16.572864%
MASK_SPEEDUP_20_VS_8=27.443218%
MASK_PARITY=PASS
MASK_DEFAULT_WORKERS=20

MISSION_DEFAULT_WORKERS=20
MISSION_DEFAULT_MAX_INFLIGHT=40
COLLECTION_DEFAULT_WORKERS=12
BC_DATALOADER_WORKERS=8
```

The selected Label 20-worker run had peak process-tree RSS 2.035 GiB and
minimum available memory 53.883 GiB; the selected Mask 20-worker recheck had
peak RSS 1.040 GiB and minimum available memory 55.229 GiB. Maximum swap was
zero for every run. Resource recheck CPU utilization was 793/1183/1954% for
Label 8/12/20 and 599/759/1098% for Mask 8/12/20. Filesystem I/O remained
small and non-saturating, and every producer exited with `RESULT=PASS`.

The formal owners now are `config/pre_bc.yaml::label.workers=20` and
`config/pre_bc.yaml::depth_mask.workers=20`; corresponding argparse defaults
read those owners. Collection remains 12 and BC DataLoader remains workers 8,
batch 256, prefetch 4. No formal 60K Collection, 100K Mission, full Label or
Mask run, BC training, AWAC/SAC run, Unity change, Bridge change, Task or
Observation contract change, or commit was performed.

---

## Collection canonical/default worker promotion to 20 — 2026-09-01

The previous V3.1 W20 bounded promotion record remains historical and
fail-closed: it observed 0.697266 GiB swap, so it did not pass that zero-swap
gate. The user has now explicitly accepted W20 as the formal Collection
profile based on the separate historical near-20-hour W20 long-run success
evidence. This is a Collection-only profile change; no collection algorithm,
trajectory semantics, Unity, Bridge, Task Contract, or Observation Contract
changed.

```text
COLLECTION_W20_FORMAL_PROFILE_PROMOTION=PASS
COLLECTION_W20_USER_ACCEPTANCE=YES
COLLECTION_W20_ACCEPTANCE_BASIS=HISTORICAL_NEAR_20_HOUR_LONG_RUN_SUCCESS
HISTORICAL_W20_FORMAL_PROMOTION=FAIL_CLOSED
HISTORICAL_W20_SWAP_USED_GIB=0.697266
COLLECTION_CANONICAL_DEFAULT_WORKERS=20
COLLECTION_RESOLVED_DEFAULT_WORKERS=20
MISSION_WORKERS=20
LABEL_WORKERS=20
MASK_WORKERS=20
BC_DATALOADER_WORKERS=8
COLLECTION_ALGORITHM_CHANGED=NO
COLLECTION_TRAJECTORY_SEMANTICS_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
FORMAL_60K_EXECUTED=NO
```

The canonical owner is `config/pre_bc.yaml::collection.workers=20`; the
parallel collector resolves an omitted worker flag through that owner, and
`validate_teacher_collection` formal invocations now use
`--expected-workers 20`. The V3.2 commandbook, owner matrix, and JSON release
manifest were updated. Full verification in the `xm` conda environment with
the devel native libraries reported `870 passed, 42 skipped, 0 failed`; the
initial restricted-sandbox run's socket failures were not code failures and
were cleared by the allowed loopback test rerun. No formal 60K collection,
BC training, AWAC/SAC run, or commit was performed.

---

## AWAC-only architecture migration V1 — 2026-09-03

The current RL architecture is now owned by `planning.awac`, with
`scripts/train_awac.py` as the only formal training entrypoint. The old
`python/planning/rl/` package and SAC-only production surfaces were removed
after a zero-reference check. The historical sections above remain evidence
and are not current execution instructions; the detailed current result is in
[`AWAC_ONLY_ARCHITECTURE_MIGRATION_V1.md`](AWAC_ONLY_ARCHITECTURE_MIGRATION_V1.md).

```text
AWAC_ONLY_ARCHITECTURE_MIGRATION_V1=PASS
RL_DIRECTORY_EXISTS=NO
AWAC_TO_RL_IMPORT_COUNT=0
AWAC_TO_SAC_IMPORT_COUNT=0
SAC_PRODUCTION_FILE_COUNT=0
AWAC_DEV_SET_STATUS=MISSING
FULL_TESTS=680 passed, 42 skipped, 0 failed
ALGORITHM_CHANGED=NO
BC_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
COMMIT=NO
NEXT_ACTION=RL_PHASE_0_AWAC_CONTRACT_AND_CRITIC_CALIBRATION_IMPLEMENTATION
```

## Shell entrypoint cleanup and test-suite integrity V1 — 2026-09-03

The current shell surface and the post-AWAC test-count audit are recorded in
[`SHELL_ENTRYPOINT_CLEANUP_AND_TEST_SUITE_INTEGRITY_V1.md`](SHELL_ENTRYPOINT_CLEANUP_AND_TEST_SUITE_INTEGRITY_V1.md)
and its machine-readable companion. Earlier sections that mention removed
SAC/AWAC/collection wrappers are historical evidence and are not current
execution instructions. The current formal AWAC, Collection, and evaluation
entrypoints are Python scripts; the managed evaluation shell remains only for
ROS/Unity/Bridge lifecycle management.

## RL Phase 0 — AWAC contract and critic calibration implementation V1 — 2026-09-03

Phase 0 implementation components passed: the formal owner is
`planning.awac`, the entrypoint is `scripts/train_awac.py`, the frozen BC
actor is used for critic initialization, the actor remains frozen during
calibration, and Twin Critic updates/checkpoint-resume identity checks are
implemented. The detailed report and machine-readable evidence are in
[`RL_PHASE_0_AWAC_CONTRACT_AND_CRITIC_CALIBRATION_IMPLEMENTATION_V1.md`](RL_PHASE_0_AWAC_CONTRACT_AND_CRITIC_CALIBRATION_IMPLEMENTATION_V1.md)
and
[`rl_phase_0_awac_contract_and_critic_calibration_implementation_v1.json`](rl_phase_0_awac_contract_and_critic_calibration_implementation_v1.json).

```text
RL_PHASE_0_IMPLEMENTATION_COMPONENTS=PASS
RL_PHASE_0_RELEASE_GATE=BLOCKED_ENVIRONMENT
AWAC_TRAINING_CONTRACT_SHA256=ac6edadadcdf5a7966c0499aacccb25054268dfd6e2a56348369d5a781509469
REWARD_CONTRACT_SHA256=1eacab3788e9051e8c77d7c6aeafa2c50bd93dc09434b3adf4985850dd2b62d0
REWARD_SCALE_APPLIED_EXACTLY_ONCE=YES
REPLAY_CONTRACT_SHA256=2b6198b0d295a51687542e475505f85c0d84180a8e529b0c3ea4f84d99c2244f
AWAC_DEV_SET=data/test/awac_dev_seed4026_100
AWAC_DEV_SET_MISSIONS=100
AWAC_DEV_SET_VALIDATION=PASS
CALIBRATION_SMOKE=PASS
CALIBRATION_REPLAY_TRANSITIONS=160
CALIBRATION_EPISODES=20
CALIBRATION_CRITIC_UPDATES=8
CALIBRATION_GATE_STATE=PENDING
FULL_TESTS=685 passed, 42 skipped, 8 failed
CODE_ELIGIBLE_FULL_TESTS=669 passed, 42 skipped, 0 failed
FULL_TEST_FAILURE_CLASS=ENVIRONMENT_SOCKET_PERMISSION
CONFIDENCE_ALGORITHM_IMPLEMENTED=NO
ADAPTIVE_BC_KL_IMPLEMENTED=NO
PRIMITIVE_NEIGHBOR_ALGORITHM_IMPLEMENTED=NO
FORMAL_AWAC_LONG_TRAINING_EXECUTED=NO
FINAL_300_EVALUATION_EXECUTED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
BC_CHANGED=NO
PHASE1_STARTED=NO
COMMIT=NO
NEXT_ACTION=RERUN_UNFILTERED_FULL_PYTEST_WITH_LOOPBACK_PERMISSION
```

The eight full-suite failures are the existing socket-dependent startup and
single-worker tests, all failing before behavior assertions because this
managed execution environment denies socket creation. No production test was
altered to bypass that condition. Phase 1 remains blocked and was not
started.
