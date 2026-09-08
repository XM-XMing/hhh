# Collection Scaling and Teacher Runtime V1

Date: 2026-08-29  
Scope: bounded runtime and collection scaling only.

## Scope and freeze

The formal project for this audit is `/home/xm/XM/xm_ws/src/planning`.
Unity, the reliable protocol, Teacher scoring/beam semantics, collision and
depth algorithms, BC/AWAC, and formal data collection remain unchanged. No
formal 60K collection, BC training, AWAC run, or commit was performed.

The only retained source change is the explicit managed-collection worker
bound in `python/planning/runtime/ports.py`: `[1,24]`. The default in
`config/pre_bc.yaml` remains 12. RL/SAC worker bounds remain separate and
unchanged. A focused test covers 24-worker port uniqueness and fail-closed
rejection of 25 workers.

## Profiler phase contract

`primitive_execution_ms` is an inclusive outer phase. It contains the
reliable backend command wait, result wait, endpoint snapshot lookup, commit,
and the endpoint/depth work performed by `step_primitive`.

```text
PRIMITIVE_EXECUTION_CONTAINS_RESULT_WAIT=YES
```

The disjoint per-episode wall decomposition is:

```text
reset
+ teacher_decision
+ primitive_execution (inclusive)
+ python_postprocess
+ episode_finalize
+ npz_write
+ uninstrumented_idle
```

`result_wait`, `command_wait`, `snapshot_wait`, `depth_wait`, and `commit` are
diagnostic nested phases and are not added to that decomposition. Action-mask
timings also cross the primitive boundary and are not independently additive.
The wall clock/report elapsed time remains authoritative.

## Worker contract and port gate

The original `[1,12]` check was a managed collection safety bound in
`ports.py`; port derivation, worker identity, runtime nonce, Unity/Bridge
allocation, reliable protocol, and artifact schema impose no separate 12-worker
limit in the audited collection path. The bound is now `[1,24]` without
changing the per-worker layout.

The 24-worker preflight generated 240 ports (10 per worker):

```text
WORKER_24_PORT_PREFLIGHT=PASS
PORT_COUNT=240
UNIQUE_PORT_COUNT=240
PORT_COLLISION_COUNT=0
CURRENT_LISTENER_CONFLICTS=0
```

The bounded startup gate passed 12/12, 16/16, 20/20, and 24/24 workers. Each
run had unique worker/runtime bindings, no cross-talk or wrong-worker
ACK/snapshot, and all child processes were cleaned up. No OOM, Unity crash,
Bridge crash, or GPU failure occurred. The 24-worker run reached approximately
50.01 GiB peak tree RSS and 8.95 GiB minimum available memory, so it is usable
for the bounded benchmark but retains a clear memory-headroom warning for
longer production runs.

## Fixed-fixture collection scaling

All rows below use the same V2 mission fixture, seed 2026, reliable-exact
observation contract, Unity Player, devel Bridge, voxel cache, MPL, and native
backend configuration. Runs with startup races or an intentionally incorrect
ROS/PYTHONPATH setup were excluded from the scaling evidence.

| workers | accepted/attempted | wall (s) | accepted/s | reliable rows/s | transitions/s | primitive mean/p95 (ms) | teacher mean (ms) | max RSS (GiB) | min avail (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 111/126 | 227.561 | 0.487781 | 14.721327 | 13.943514 | 596.555/611.360 | 111.696 | 14.469 | 42.92 |
| 16 | 112/127 | 184.994 | 0.605425 | 18.265457 | 17.308669 | 601.163/616.869 | 135.852 | 16.103 | 41.89 |
| 20 | 111/126 | 172.854 | 0.642160 | 19.380518 | 18.356532 | 614.710/638.726 | 167.366 | 18.700 | 39.87 |
| 24 | 113/128 | 168.524 | 0.670528 | 20.216705 | 19.166410 | 642.878/685.097 | 225.908 | 20.466 | 38.61 |

The fixed-fixture throughput winner is 24 workers, but teacher and primitive
latencies rise with concurrency and CPU/load saturation is visible. The final
500-accepted benchmark below is the stronger ETA measurement.

## Teacher runtime profile and parity

The 1,000-decision exact benchmark used depth 3, width 8, branching 4, and
candidate top-k 12. The main inclusive component costs, ranked by the native
production path, were:

```text
score_state                 mean 72.914 ms (inclusive)
evaluate_pose_reachability  mean 23.700 ms (inclusive)
evaluate_candidates         mean 15.847 ms (inclusive)
score_state_one_step        mean 3.437 ms (inclusive)
collision_info              mean 2.157 ms (inclusive)
predict_action              mean 0.011 ms
guidance_goal               mean 0.037 ms
```

The vectorized finite-min trial preserved all output hashes but improved the
1,000-call mean from 69.2956 ms to 69.2051 ms (about 0.13%). It was reverted
because it did not meet the 5% retention threshold. The retained production
measurement is 69.2594 ms mean, p95 95.9634 ms, with no Teacher scoring or
beam change.

```text
TEACHER_RUNTIME_EXACT_PARITY=PASS
candidate trace/action/score/target/mask/rank hashes=exact
```

The common O/P fixture also passed candidate-accept and Teacher-valid-mask
parity with zero mismatches. The O comparator used its explicit Python
reference because the temporary historical O C0 library lacked the shared-map
factory symbol; this is not a claim of native O library parity. The current
formal O devel native library contains the shared-map symbol.

## Final bounded benchmark

The final correctly configured W24 run used a `/tmp` preparation fixture with
650 passing missions and collected 500 as the bounded target. It completed
with:

```text
accepted=522
attempted=583
wall=619.823 s
accepted_per_sec=0.842175912
reliable_rows=15845
reliable_rows_per_sec=25.563750
transitions=14920
transitions_per_sec=24.071388
legacy_rows=0
telemetry_lookup=0
snapshot_missing=0
state_depth_skew_max_ns=0
frame_failures=0
collector_errors=0
endpoint_identity_chain_valid=true
process_cleanup=PASS
```

The final run used Unity Player SHA
`61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365`, current
devel Bridge SHA
`462e9f7bc25d5bef63ce17b863daa256726e17de870bda04a1bcb72e79e80f2b`, MPL
contract SHA
`f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d`, and
Task V2 SHA
`2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df`.

## ETA and validation

The normal 60K ETA uses the midpoint rate of the two measured W24 rates (the
fixed fixture and the final 500-accepted run), because their mission subsets
have different acceptance composition. The lower and higher measured rates
are reported explicitly as pessimistic and optimistic bounds:

```text
ETA_60K_OPTIMISTIC_HOURS=19.7900  # final 500-accepted rate
ETA_60K_NORMAL_HOURS=22.0356      # midpoint of the two measured rates
ETA_60K_PESSIMISTIC_HOURS=24.8560 # lower measured W24 rate
ETA_100K_PREPARATION_HOURS=10.1558
```

The post-collection relabel/mask/mmap and BC training duration was not run or
benchmarked, therefore total time to BC-training-ready remains `UNKNOWN`.

Current validation:

```text
compileall=PASS
focused worker/port tests=26 passed
full tests=841 passed, 42 skipped, 0 failed
PRE_BC_E2E=PASS (existing frozen evidence; not rerun in this benchmark)
SELECTED_WORKER_SMOKE=PASS
UNITY_CHANGED=NO
FORMAL_100K_EXECUTED=NO
FORMAL_60K_EXECUTED=NO
BC_TRAINING_EXECUTED=NO
AWAC_EXECUTED=NO
```

Evidence artifacts are under
`/tmp/xm-collection-scaling-teacher-runtime-v1-20260829/`, including
`collection_scaling_report.json`, the W12/W16/W20/W24 summaries, the final
`final_benchmark/collection_w24_500_retry/collection_summary.json`, the
resource summary, the 24-worker port preflight, and the exact Teacher/parity
reports.

## Decision

```text
COLLECTION_SCALING_AND_TEACHER_RUNTIME_V1=PASS
BEST_COLLECTION_WORKERS=24
BEST_COLLECTION_ACCEPTED_PER_SEC=0.842175912 (final bounded benchmark)
NEXT_USER_ACTION=EXECUTE UPDATED FORMAL DATA PIPELINE
```

The next action is user-controlled formal pipeline execution. This audit did
not execute that pipeline.
