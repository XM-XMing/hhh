# XMflight Full Pipeline Performance Optimization and Formal Freeze V3.2

Date: 2026-09-01

This is the current stage-specific worker decision following the historical
V3.1 Collection-only gate and the V3.2 offline Label/Mask benchmark. V3 and
V3.1 remain historical records and are not rewritten. The current Collection
profile is 20 workers by explicit user acceptance of the historical
near-20-hour W20 long-run success evidence.

## Release decision

~~~text
OFFLINE_WORKER_12_VS_20_PROMOTION_500_SMOKE_V1=FAIL (HISTORICAL BASELINE)
OFFLINE_LABEL_MASK_WORKER_PROMOTION_500_V2=PASS
COLLECTION_W20_FORMAL_PROFILE_PROMOTION=PASS
MISSION_PREPARATION_PROMOTION=PASS
LABEL_PROMOTION=PASS
DEPTH_MASK_PROMOTION=PASS
MISSION_WORKERS_PROMOTED_TO_20=YES
MISSION_DEFAULT_WORKERS=20
MISSION_DEFAULT_MAX_INFLIGHT=40
LABEL_DEFAULT_WORKERS=20
MASK_DEFAULT_WORKERS=20
COLLECTION_DEFAULT_WORKERS=20
FORMAL_SAFE_COLLECTION_WORKERS=20
COLLECTION_W20_USER_ACCEPTANCE=YES
HISTORICAL_W20_SWAP_FAIL_CLOSED_GIB=0.697266
OTHER_PIPELINE_WORKER_CONFIG_CHANGED=NO
COMPILEALL=PASS
TARGETED_TESTS=35 passed, 0 failed
FULL_TESTS=870 passed, 42 skipped, 0 failed
FORMAL_60K_EXECUTED=NO
FORMAL_100K_MISSIONS_EXECUTED=NO
BC_TRAINING_EXECUTED=NO
AWAC_SAC_EXECUTED=NO
COMMIT_EXECUTED=NO
~~~

The historical V3.1 W20 bounded promotion remains a fail-closed record because
the run observed 0.697266 GiB swap. It is retained as evidence and was not
used to claim zero-swap runtime validation. The new formal Collection=20
profile is accepted by the user from the separate historical near-20-hour W20
long-run success evidence. No formal 60K Collection was executed here.

The earlier V1 result below is retained as historical evidence. The V2
follow-up closes the missing-fixture blocker with an independent bounded
accepted-rollout fixture and promotes Label and depth-mask workers
independently.

## Historical V1 blocked baseline

The remainder of this historical section records why the V1 decision was
blocked. It is superseded by the V2 controlled follow-up below and is kept to
preserve the earlier evidence.

## Frozen scope

The benchmark kept Task Contract V2, `max_steps=45`,
`reliable_exact_endpoint_snapshot`, seed 2026, MPL, voxel/collision
semantics, Teacher beam depth 3, width 8, branching 4, discount 0.95, Unity,
Bridge, trajectory layout, Collection worker 12, and BC DataLoader workers 8.
No formal data, Unity runtime, Collection, BC training, or AWAC/SAC run was
executed.

## Mission Preparation evidence

Both runs used the same empty output shape, source configuration, voxel cache,
sampling seed, collision thread count, and preparation semantics. The only
benchmark variables were workers and the matching in-flight window.

| Metric | 12 workers | 20 workers |
|---|---:|---:|
| Wall time (s) | 145.024 | 119.826 |
| Passing missions | 500 | 500 |
| Sampling attempts | 1,111 | 1,111 |
| Passing missions/s | 3.447705207 | 4.172717106 |
| Attempts/s | 7.660800971 | 9.271777411 |
| Aggregate process CPU | 1011% | 1647% |
| Peak sampled process-tree RSS (GiB) | 1.205200 | 1.978275 |
| Minimum available memory (GiB) | 55.286102 | 54.697879 |
| Maximum swap used | 0 | 0 |
| Maximum process count | 15 | 23 |
| Maximum I/O wait | 0% | 0% |

~~~text
MISSION_SPEEDUP_20_VS_12=1.210288251
MISSION_THROUGHPUT_GAIN_PERCENT=21.028825
MISSION_PARITY=PASS
MISSION_WORKERS_PROMOTED_TO_20=YES
ACTIVE_PROCESS_POOL_W12=12
ACTIVE_PROCESS_POOL_W20=20
INFLIGHT_WINDOW_W12=24
INFLIGHT_WINDOW_W20=40
~~~

The following artifacts were byte-for-byte equal between both runs:

~~~text
mission_candidates.csv
missions.csv
mission_routes.meta.json
mission_routes.offsets.bin
mission_routes.points.bin
~~~

Their common SHA256 values were respectively:

~~~text
7cb85f9f98f8578ad810a2e0754e3f3a00038a3d47d3a3bb169f8487177230d7
ee65bcafced3f11dd9120fc7d3d9ed22bc3e3ac61ab084ae09109bde2676dec9
27cdd655c3d843c963b4f614c1bc89b44347e198ed42d6973f83071ff1b70e06
5e807b6389adfb961ef127f9ccbf41ce179906c586e7573968169a756fb9db19
b897e21da09489838f5f39566da71818c2f61f1d62df011bcc7f93f59f9a4598
~~~

The exact artifact equality closes candidate ordering, mission IDs, route
ordering, Teacher action sequence, and dead-end/result parity for this fixture.

## Label and depth-mask status

The canonical accepted-only input required by this task is not present:

~~~text
MISSING_LABEL_MASK_FIXTURE=data/teach/flight_reliable_exact_teacher_seed2026_20260830_v8_bc59930
MISSING_VALIDATED_V4_MISSION_FIXTURE=data/teach/flight_reliable_exact_teacher_seed2026_20260829_v4
LABEL_W12_WALL_SEC=NOT_RUN
LABEL_W20_WALL_SEC=NOT_RUN
MASK_W12_WALL_SEC=NOT_RUN
MASK_W20_WALL_SEC=NOT_RUN
LABEL_PARITY=NOT_RUN
MASK_PARITY=NOT_RUN
~~~

Using the old `z_xm_ws111` checkout, partial artifacts, or the interrupted
`data/teach/2026_6w` mission preparation would violate the fixed-input and
accepted-only contract. Consequently no Label or Mask default was changed;
their source CLI defaults remain 8 and the existing formal commandbook
overrides remain 12 until a canonical fixture is restored.

## Stage-specific current profile

~~~text
MISSION: workers=20, max_inflight_results=40, collision_threads=1
LABEL: source_cli_default_num_workers=8, formal_command_override=12
MASK: source_cli_default_num_workers=8, formal_command_override=12
COLLECTION: workers=12
BC_DATALOADER: workers=8, batch_size=256, prefetch_factor=4
~~~

The Mission default is owned by `config/pre_bc.yaml::mission.route_workers`;
the Python parser and dataclass resolve that value, and the in-flight bound is
derived as twice the configured Mission worker count. Collection and BC
settings are independent owners and were not changed.

## Verification status

`compileall`, the Mission parallelism/config tests, JSON validation, and all
three CLI help checks passed. The corrected full pytest invocation used the
current devel native libraries and loopback-capable host environment. It
reported 868 passed and 42 skipped. The only two failures were the historical
comparison tests
`test_protocol_header_and_language_neutral_fixture_bytes_match_original` and
`test_managed_ports_match_and_direct_profile_matches_frozen_defaults`; both
refer to the absent
`/home/xm/XM/xm_ws/src/planning_backup_pre_py6_20260828T141420Z/tree` backup.
That missing evidence was not recreated or substituted.

## OFFLINE_LABEL_MASK_WORKER_PROMOTION_500_V2 — 2026-09-01

This controlled follow-up was run after the historical V1 blocker was closed.
The referenced validated Optimized 3K artifact was absent, so the permitted
independent bounded reliable-exact smoke was used to create a fixed fixture.
No formal Collection, full relabel/mask run, BC training, or RL run was
executed.

~~~text
OFFLINE_LABEL_MASK_WORKER_PROMOTION_500_V2=PASS
FIXTURE_SOURCE=/tmp/xm-label-mask-worker-benchmark-v2/collection500
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
MISSION_DEFAULT_WORKERS=20
MISSION_DEFAULT_MAX_INFLIGHT=40
LABEL_DEFAULT_WORKERS=20
MASK_DEFAULT_WORKERS=20
COLLECTION_DEFAULT_WORKERS=12
BC_DATALOADER_WORKERS=8
~~

The fixture used the same RouteStore, collision cache, Teacher beam
(`depth=3`, `width=8`, `branching=4`, `discount=0.95`), C++ CPU collision
backend, one collision thread, and direct-mmap publication for every worker
configuration. All six producer runs returned `RESULT=PASS`; all numerical
arrays, mapping arrays, resolved episode order, and business metadata were
exactly equal across worker counts. Cross-artifact provenance validation also
passed for the Label/Mask pair.

| Stage | Workers | Median/primary wall (s) | Transitions/s | Peak tree RSS (GiB) | Minimum available (GiB) | Max swap (GiB) | CPU recheck |
|---|---:|---:|---:|---:|---:|---:|---:|
| Label | 8 | 167.966 | 85.076625 | 0.860 | 54.806 | 0 | 793% |
| Label | 12 | 134.844 | 105.973999 | 1.252 | 54.528 | 0 | 1183% |
| Label | 20 | 115.745 | 123.461033 | 2.035 | 53.883 | 0 | 1954% |
| Mask | 8 | 2.0288 median | 7043.503790 median | 0.455* | 55.229* | 0 | 599% |
| Mask | 12 | 1.7404 median | 8210.814082 median | 0.651* | 55.324* | 0 | 759% |
| Mask | 20 | 1.5919 median | 8976.467885 median | 1.040* | 55.229* | 0 | 1098% |

`*` Mask resource values are from the resource recheck; the primary run also
passed with peak tree RSS 1.018 GiB at 20 workers and minimum available memory
55.246 GiB. Mask repeated runs (three per worker count) were used for the
worker decision because the stage wall time is short; their output arrays were
also exact against the primary result.

~~~text
LABEL_SPEEDUP_12_VS_8=24.563003%
LABEL_SPEEDUP_20_VS_8=45.117455%
LABEL_PARITY=PASS
LABEL_PROMOTION=PASS
LABEL_RESOLVED_DEFAULT=20
MASK_SPEEDUP_12_VS_8=16.572864% (three-run median)
MASK_SPEEDUP_20_VS_8=27.443218% (three-run median)
MASK_PARITY=PASS
MASK_PROMOTION=PASS
MASK_RESOLVED_DEFAULT=20
SWAP_GATE=PASS
RAM_HEADROOM_GATE=PASS
IO_RESPONSIVENESS_GATE=PASS
WORKER_CRASH_GATE=PASS
~~

The offline promotion is stage-specific: the measured Label and depth-mask
stages select 20, while the then-current Collection baseline in this
historical section was 12. The current Collection profile is updated by the
promotion amendment above. BC DataLoader remains workers 8, batch 256,
prefetch 4. The default owners are now
`config/pre_bc.yaml::label.workers=20` and
`config/pre_bc.yaml::depth_mask.workers=20`; explicit CLI values remain valid
overrides.
