# XMflight Full Pipeline Performance Optimization and Formal Freeze V3.1

Date: 2026-09-01

This is the bounded Collection-worker promotion audit following V3. V3 is
preserved as the historical 12-worker freeze. V3.1 does not change the
trajectory, observation, task, Unity, Bridge, Teacher, or downstream pipeline
contracts.

## Release decision

~~~text
FORMAL_COLLECTION_W20_PROMOTION_V3_1 = FAIL
FORMAL_SAFE_COLLECTION_WORKERS = 12
OTHER_PIPELINE_WORKER_CONFIG_CHANGED = NO
COLLECTION_TRAJECTORY_SEMANTICS_CHANGED = NO
COLLECTION_ALGORITHM_CHANGED = NO
UNITY_CHANGED = NO
TASK_CONTRACT_CHANGED = NO
OBSERVATION_CONTRACT_CHANGED = NO
DATA_LAYOUT_CHANGED = NO
FORMAL_60K_EXECUTED = NO
FORMAL_100K_MISSIONS_EXECUTED = NO
BC_TRAINING_EXECUTED = NO
AWAC_SAC_EXECUTED = NO
COMMIT_EXECUTED = NO
~~~

The W20 bounded run passed the collection quality and lifecycle checks, but
the promotion gate is fail-closed. Swap usage was observed at `0.697266 GiB`,
while the hard gate requires zero. In addition, the original resource monitor
ran under a Chinese locale, so its `available_bytes` and `swap_used_bytes`
columns were blank. A spot sample showed 36.399048 GiB available, but that is
not a complete-run minimum and cannot close the gate.

## Scope and behavior audit

The historical W20 source snapshot is not present in the workspace, so the
source comparison is based on the historical scaling record, the V3
optimization ledger, the current source seams, and the current runtime
artifact's `code_version_sha256`.

| Area | V3.1 result | Evidence/interpretation |
|---|---|---|
| Collection trajectory semantics | unchanged | Current run used the same reliable-exact endpoint session and accepted-row contract |
| Collection algorithm and Teacher beam | unchanged | Current resolved beam is depth 3, width 8, branching 4, discount 0.95 |
| Observation contract | unchanged | `reliable_exact_endpoint_snapshot`; telemetry false; skew 0 |
| Task contract | unchanged | schema 2, SHA `2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df`, max steps 45 |
| Unity | unchanged | Current frozen Player identity was used |
| Bridge protocol/lifecycle semantics | unchanged | Current devel Bridge identity was used; lifecycle additions are cleanup/observability only |
| Allowed V3 changes | retained | disk preflight/watchdog, graceful cleanup, logging robustness, resume diagnostics, monitoring/provenance |
| Downstream worker profiles | unchanged | Mission, label, mask, audit, BC mmap, and BC DataLoader remain on their V3 settings |

Current collection source files were audited without modification:

~~~text
python/planning/teacher/parallel_collection.py = d75408352170abfdb237c1c8bf74bdcbfbbe9874b034a0d25b6af3dc9cdc7082
python/planning/runtime/collection_worker.py = dd56982930b9d2df502237e649122b370e9ed49a396e49d51d6a35ce3d44e588
python/planning/teacher/rollout_collector.py = b20cd5289978c766c01e37aaa5b11000e6ab509b6f29ba5b90c2210aea74b940
python/planning/teacher/rollout_session.py = 0d5893c072963e4545d91912f4cd5d21a7f9d16a108e762838865b3c1876bcf0
python/planning/teacher/collection_config.py = 87f11e006e46e571b115f46ddcdec9e7c1b480dc553403ecb1507b8475dbbd00
python/planning/teacher/collection_disk.py = 3a71bb916e30064490fd2642a26482b9868a68e50a2f13a217704ea92dcc230e
python/planning/common/logging.py = ed645fb7a32151257ad325438205cfd78dadef409eaf60c6adff5a436b41f11b
~~~

The smoke artifact recorded `code_version_sha256` as
`1e253e3bbc7fb8914b92eaa9317743437183281676b9013a40cbff3459c0b2d0`.

## W20 bounded certification

Output: `/tmp/xm-formal-w20-promotion-v31`

Input was the already validated V4 mission/RouteStore fixture. The run used
the current devel native libraries, current Unity Player, current devel
Bridge, and the reliable-exact contract.

~~~text
workers requested/active       = 20
workers ready                  = 20/20
workers finished               = 20/20
target accepted               = 400
accepted                      = 420
attempted                     = 469
accepted/sec (525.73 s wall)  = 0.798889164
acceptance rate               = 0.895522388
root reliable rows            = 12729
accepted NPZ transition rows  = 12000
valid NPZ                     = 420/420
invalid/orphan NPZ            = 0/0
duplicate episode IDs         = 0
duplicate transition IDs      = 0
physical ports                = 200 unique, collision 0
runtime identities            = 20 unique
cross-talk markers            = 0
orphan processes after exit   = 0
temporary/partial files       = 0
collector exit                = 0
collection quality result     = PASS
~~~

The root `reliable_rows` intentionally counts collection attempts, while the
validated merged index reports the 12,000 transitions belonging to the 420
accepted NPZ episodes. This is accounting, not a behavior change.

All 420 accepted episodes passed
`planning.data.rollout.load_rollout_episode(..., validate=True)`. The
collection validator also passed with 420 accepted rows, 20 workers, zero
legacy rows, zero telemetry lookups, zero snapshot misses, zero skew, zero
frame failures, and zero collector errors.

## Resource and lifecycle evidence

The monitor captured 100 samples of process/RSS/FD data. Its correctly
recorded columns were:

~~~text
process count       = 530..681
Unity RSS           = 21772..6059752 KiB
Bridge RSS          = 22816..1408792 KiB
collector RSS       = 21772..56636 KiB
matched FD count    = 16..2636
wall time           = 525.73 s
~~~

After worker startup, Unity and Bridge RSS remained in a bounded plateau with
no crash or clear linear growth. Cleanup logged all 20 collectors finished,
`runtime cleanup complete {"workers": 20}`, `SESSION END {"status": "PASS"}`,
and left no matching Unity, Bridge, ROS launch, or collector process.

Disk watchdog thresholds were enabled at 30 GiB warning and 15 GiB stop. The
run reached its accepted target normally, exited zero, and did not emit a disk
stop event; post-run `/home` free space was 208 GiB. The watchdog therefore
passed this bounded run, but it does not override the separate swap gate.

The lowest available-memory value observed in external spot samples was
36.399048 GiB, above the 15 GiB threshold. It is recorded as an observed spot
value rather than a full-run minimum because the locale-sensitive monitor did
not persist that column. Swap was 0.697266 GiB in the same observation window
and remained nonzero post-run.

## Gate table

| Gate | Result |
|---|---|
| 20/20 ready and finished | PASS |
| Reliable-exact quality counters | PASS, all zero |
| Runtime identity uniqueness | PASS |
| Port collision/cross-talk/orphan | PASS, all zero |
| Bridge stability | PASS for bounded plateau |
| Collector stability | PASS for bounded plateau |
| Unity stability | PASS for bounded plateau |
| Graceful cleanup | PASS |
| Disk watchdog | PASS |
| RAM headroom | OBSERVED PASS, full minimum not captured |
| Swap equals zero | FAIL, 0.697266 GiB observed |
| W20 promotion | NOT GRANTED |

Because a hard gate failed and the memory evidence is incomplete, the formal
Collection worker freeze remains 12. V3's commandbook is not rewritten to 20;
the V3.1 gated commandbook records this decision.

## Frozen runtime identities

~~~text
Unity Player = /home/xm/XM/xm_ws/src/unity/XMflight.x86_64
Unity Player SHA256 = 61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
Unity Assembly SHA256 = 319f9e764cf51078d3b5e439214361f54aa9d5726449e26e4c0a1ad6232f21f0
Bridge = /home/xm/XM/xm_ws/devel/lib/planning/unity_bridge_node
Bridge SHA256 = 667c266e96211aabb5a0cd25b8b35b303903d35cbe2d95be8f9d2bbe5e63d789
Voxel cache SHA256 = a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691
MPL contract SHA256 = f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d
Observation contract = reliable_exact_endpoint_snapshot
~~~

