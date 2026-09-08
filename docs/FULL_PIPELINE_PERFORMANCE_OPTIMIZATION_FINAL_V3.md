# XMflight Full Pipeline Performance Optimization and Formal Freeze V3

Date: 2026-09-01

This is the final bounded performance-fix and release-freeze record for the
canonical Planning tree. It is based on two independent fresh 3K pipelines:

~~~text
BASELINE = data/benchmark/full_pipeline_3k_baseline
OPTIMIZED = data/benchmark/full_pipeline_3k_optimized
EVIDENCE = /tmp/xm-full-pipeline-opt-20260831/
~~~

The optimized run did not reuse baseline trajectories, labels, masks, mmap
arrays, or checkpoints. Formal-scale collection, formal BC training, and
AWAC/SAC were not executed.

## Release status

~~~text
XMFLIGHT_FULL_PIPELINE_PERFORMANCE_OPTIMIZATION = PASS
OPTIMIZATION_FREEZE = PASS
ALGORITHM_CHANGED = NO
DATA_LAYOUT_CHANGED = NO
CONTRACTS_CHANGED = NO
UNITY_CHANGED = NO
CPP_COLLISION_SEMANTICS_CHANGED = NO
FORMAL_SCALE_EXECUTED = NO
FORMAL_COLLECTION_EXECUTED = NO
FORMAL_BC_TRAINING_EXECUTED = NO
AWAC_SAC_EXECUTED = NO
COMMIT_EXECUTED = NO
~~~

The performance freeze is complete. The formal disk-capacity gate is currently
blocked until additional free space is available; the command handoff is
therefore a gated user handoff, not an authorization to start a formal run.

## Frozen identities and runtime boundary

| Item | Frozen value |
|---|---|
| Python environment | conda activate xm |
| Build system | catkin_make, Release, devel/ only |
| Unity Player | /home/xm/XM/xm_ws/src/unity/XMflight.x86_64 |
| Unity Player SHA256 | 61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365 |
| Unity Assembly SHA256 | 319f9e764cf51078d3b5e439214361f54aa9d5726449e26e4c0a1ad6232f21f0 |
| Bridge | /home/xm/XM/xm_ws/devel/lib/planning/unity_bridge_node |
| Bridge SHA256 | 667c266e96211aabb5a0cd25b8b35b303903d35cbe2d95be8f9d2bbe5e63d789 |
| Observation contract | reliable_exact_endpoint_snapshot |
| Task contract | V2, max primitive steps 45, SHA256 2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df |
| MPL NPZ SHA256 | 22ad22fe66a88633de6effff11c03aa4ca4e8df362f8b410e17effbd91ff7309 |
| MPL JSON SHA256 | c1b795e4737a12034b0d59a1b457f559e7ad138527df2c90ed35db1326e42ea8 |
| MPL contract SHA256 | f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d |
| Voxel cache SHA256 | a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691 |

Formal collision is C++17/OpenMP CPU, global route and depth safety are native,
and Python production fallback is disabled. CUDA/hybrid collision remains
outside formal scope and was not claimed as validated. PyTorch CUDA/AMP remains
available for BC training.

## Measured stage wall time

Times are from /usr/bin/time -v around the same bounded stages. The collection
rows differ by one due to normal bounded in-flight overshoot; throughput is
reported with the observed accepted count.

| Stage | Baseline | Optimized | Delta | Interpretation |
|---|---:|---:|---:|---|
| MPL generation | 0.12 s | 0.12 s | 0.00% | unchanged |
| Voxel cache | 3.14 s | 3.28 s | +4.46% | deterministic cache build; no algorithm change |
| Mission preparation | 1454.94 s | 1437.35 s | -1.21% | bounded scheduler retained; fast output-only Teacher seam |
| 3K reliable collection | 5466.00 s | 5457.00 s | -0.16% | runtime behavior and quality counters unchanged |
| Collection validation | 3.47 s | 4.40 s | +26.80% | provenance validation, not collection hot path |
| Teacher relabel | 786.64 s | 798.50 s | +1.51% | direct mmap reduces parent copies; exact labels retained |
| Depth masks | 9.12 s | 9.26 s | +1.54% | direct field loader/direct mmap; exact masks retained |
| Dataset audit | 17.63 s | 22.12 s | +25.47% | full provenance audit remains fail-closed |
| BC mmap | 30.46 s | 22.37 s | -26.56% | metadata prepass removed; single-pass block materialization |
| BC validation | 3.52 s | 4.41 s | +25.28% | optimized full validation includes explicit sampled/header checks |
| BC one-epoch benchmark | 6.58 s | 5.22 s | -20.67% | CUDA AMP/DataLoader path; same loss contract |
| **sum of measured stages** | **7781.62 s** | **7764.03 s** | **-0.226%** | collection dominates the end-to-end wall time |

Derived throughput:

~~~text
MISSION_PREP_PASSING_PER_SEC       3.436568 -> 3.478624
COLLECTION_ACCEPTED_PER_SEC        0.551226 -> 0.551952
RELABEL_TRANSITIONS_PER_SEC        109.722  -> 108.056
DEPTH_MASK_TRANSITIONS_PER_SEC     9464.035 -> 9317.819
BC_MMAP_TRANSITIONS_PER_SEC        2833.618 -> 3857.085
BC_TRAIN_TRANSITIONS_PER_SEC       13117.325 -> 16529.310
~~~

## Optimization ledger

| Area | Change | Contract status | Decision |
|---|---|---|---|
| Mission scheduler | Existing bounded process pool, bounded in-flight results, deterministic reorder and journals were retained | exact mission validation passed | KEEP |
| Teacher selection | Added score_state_fast for mission audit/preparation; detailed relabel/collection path remains unchanged | Teacher beam contract tests passed; fallback is fail-safe when diagnostics are missing | KEEP |
| Collision | A minimal-clearance ABI experiment was measured and reverted because detailed throughput regressed 4.95% | numeric parity was exact during the experiment | REJECT EXPERIMENT |
| Relabel | Field-selective episode loading and disjoint direct-mmap worker slices | label arrays exact on 3011 common episodes | KEEP |
| Depth masks | Field-selective depth/goal loading and disjoint direct-mmap worker slices | local action masks exact on 3011 common episodes | KEEP |
| BC mmap | Uses label offsets/lengths as the size prepass and writes final arrays in one block pass | shapes, dtypes, row order, split and manifest contract passed | KEEP |
| BC validation | Added fast headers/manifest/sample validation while retaining full semantic validation | fast and full modes passed | KEEP |
| Collection disk guard | Preflight estimate, warning/stop thresholds, atomic stop marker, cleanup-safe logging | disk/lifecycle tests and fresh collection passed | KEEP |
| BC DataLoader | Existing workers/prefetch/persistent/pin-memory path benchmarked; best bounded profile was batch 256, workers 8, prefetch 4 | CUDA quality gate passed; no OOM in sweep | RECOMMEND PROFILE |

No output field, action count, float computation order, observation content,
episode ordering contract, or BC loss formula was removed or changed.

## Independent pipeline gates

Both pipelines used fresh motion primitives, a fresh deterministic voxel cache,
fresh mission artifacts, a fresh 12-worker reliable-exact collection, relabel,
depth masks, dataset audit, BC mmap, BC validation, and a bounded CUDA BC
benchmark.

~~~text
BASELINE_COLLECTION_VALIDATION = PASS
OPTIMIZED_COLLECTION_VALIDATION = PASS
BASELINE_RELIABLE_ROWS = 92174
OPTIMIZED_RELIABLE_ROWS = 92145
BASELINE_COLLECTION_ERRORS = 0
OPTIMIZED_COLLECTION_ERRORS = 0
BASELINE_LEGACY_ROWS = 0
OPTIMIZED_LEGACY_ROWS = 0
BASELINE_TELEMETRY_LOOKUP = 0
OPTIMIZED_TELEMETRY_LOOKUP = 0
BASELINE_SNAPSHOT_MISSING = 0
OPTIMIZED_SNAPSHOT_MISSING = 0
BASELINE_STATE_DEPTH_SKEW_MAX_NS = 0
OPTIMIZED_STATE_DEPTH_SKEW_MAX_NS = 0
BASELINE_FRAME_FAILURES = 0
OPTIMIZED_FRAME_FAILURES = 0
BASELINE_ENDPOINT_IDENTITY_CHAIN = PASS
OPTIMIZED_ENDPOINT_IDENTITY_CHAIN = PASS
~~~

Common-episode parity covered 3011 episodes. Baseline-only and optimized-only
episode IDs were caused by independent fresh collections; common mission IDs
had zero mismatch. The following were exact on every common episode:

~~~text
soft_targets
global_action_masks
teacher_argmax
behavior_actions
valid_counts
teacher_entropies
local_action_masks
~~~

MPL byte parity, voxel metadata/occupancy/coordinate parity, mission validation,
BC artifact validation, and runtime identity validation all passed.

## Release gates

~~~text
CATKIN_MAKE_RELEASE = PASS
INSTALL_USED = NO
COMPILEALL = PASS
FULL_PYTEST = 869 passed, 42 skipped, 0 failed
SKIP_COUNT_INCREASED = NO
CLI_HELP_SNAPSHOTS = PASS (14 canonical scripts)
RUNTIME_VALIDATION = PASS
FRESH_BASELINE_3K = PASS
FRESH_OPTIMIZED_3K = PASS
ARRAY_AND_LAYOUT_PARITY = PASS
~~~

The final C++ build produced all Planning targets, including the four native
libraries, bridge, and parity fixture, without compiler warnings or errors.

## Resource gate and formal worker recommendation

Current free space on the Planning filesystem was 222901706752 bytes, or
approximately 207.55 GiB.

The conservative formal estimate uses the measured 60K-scale rollout footprint
from the existing reliable collection, scales the optimized label/mask/mmap
outputs from the fresh 3K run, and includes the existing formal mission/route
footprint for a full-clean scenario:

~~~text
ESTIMATED_FORMAL_OUTPUT = 225.527 GB decimal = 210.039 GiB
REQUIRED_FREE_WITH_15_GiB_OPERATIONAL_MARGIN = 225.039 GiB
RECOMMENDED_FREE_WITH_30_PERCENT_AND_30_GiB_MARGIN = 303.050 GiB
CURRENT_FREE = 207.55 GiB
FORMAL_DISK_GATE = BLOCKED
~~~

No deletion was performed. The formal commandbook contains a disk preflight and
must not be started until the gate is true. The only worker count independently
measured in the fresh optimized pipeline is 12; therefore:

~~~text
FORMAL_SAFE_COLLECTION_WORKERS = 12
W16_OR_HIGHER_PROMOTION = NOT_SUPPORTED_BY_THIS_BENCHMARK
~~~

## Evidence and handoff

Machine-readable metrics are in
docs/full_pipeline_performance_optimization_final_v3.json.

The actual CLI-derived handoff is in
docs/FORMAL_TEACHER_TO_BC_END_TO_END_COMMANDS_V3.md. It contains a full-clean
path and a current-fast path that reuses the validated mission/RouteStore. Both
paths stop at BC training, use devel/, require conda activate xm, and have
explicit identity and disk gates. Neither path was executed at formal scale.

The remaining next stage after the bounded BC handoff is policy evaluation. RL
and AWAC/SAC remain frozen and are not part of this release gate.

