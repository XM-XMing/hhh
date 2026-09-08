# Collection Runtime Lifecycle Memory Fix V1

Status: lifecycle fix validated; W4 and user-bounded one-hour W12 diagnostics complete

Date: 2026-08-30

Scope: the formal Planning runtime at `/home/xm/XM/xm_ws/src/planning`.  The
historical Unity checkout, task contract, observation contract, wire schema,
Teacher algorithm, BC/AWAC code, and existing formal artifacts are outside this
change.  No formal collection, BC training, AWAC/SAC run, or commit is part of
this work.

## Root-cause evidence

The pre-fix 24-worker long run exhausted RAM and swap.  The v6 12-worker run
was stopped before producing a complete merged artifact.  The pre-fix bounded
W4 diagnostic used `/tmp/xm-collection-long-run-memory-leak-20260830/w4-60m`.
At approximately 30 to 270 seconds its resource samples showed:

| Component | RSS at ~30 s | RSS at ~270 s | Approximate aggregate growth |
|---|---:|---:|---:|
| parent | 39.684 MiB | 39.684 MiB | 0 MiB/h |
| collector | 3,753.027 MiB | 4,872.727 MiB | +16,795 MiB/h |
| bridge | 306.543 MiB | 2,552.348 MiB | +33,687 MiB/h |
| Unity | approximately flat | approximately flat | approximately 0 MiB/h |

Process count and file-descriptor count were stable in that run.  The source
ownership audit identified the primary Bridge retention as
`InMemorySnapshotCache::snapshots_`: it retained the complete endpoint state,
depth, and canonical payload after the result lifecycle had become terminal.
The active result and command stores also retained complete payloads.  The
Python collector retained accepted report rows in memory through
`CsvJournal.committed_rows`/`all_reports`, and the Python runtime retained
completed transaction maps.  These are independent retention paths; a short
benchmark cannot certify a long run.

## Lifecycle ownership matrix

The invariant is: a full payload belongs to one active transaction, is erased
at its protocol terminal event, and may leave only a bounded identity/hash
tombstone needed for exact duplicate/conflict handling.

| Container | Key | Full value retained | Insert | Terminal/read event | Erase | Bound after terminal |
|---|---|---|---|---|---|---|
| `InMemorySnapshotCache` | `ObservationRef` | endpoint state, depth, canonical snapshot bytes | validated Unity snapshot | Python endpoint result commit; reset snapshot reply send boundary | `SnapshotGateway::FinalizeExecution` or `FinalizeResetObservation` | active payloads only; broker identity/hash tombstones <= 1024 |
| `ObservationRetrievalBroker::pending_` | runtime/execution, plus reset ref | request and canonical request bytes | validated result/reset snapshot request | result commit or reset snapshot terminal delivery | `FinalizeRequest` | active requests only |
| `SnapshotGateway::pending_python_snapshot_deliveries_` | canonical snapshot request | Python identity and request | Python snapshot request | successful Python response send | response send or execution finalizer | active snapshot transactions; no history |
| `InMemoryResultStore::values_` | runtime/execution | complete result and canonical result bytes | Unity result receive | Python commit | `CommitPython` → `erase` | active payloads only; hash tombstones <= 1024 |
| `InMemoryCommandStore::values_` | runtime/execution | complete command and canonical command bytes | Python command receive | execution result commit | `CommandGateway::FinalizeExecution` → `erase` | active payloads only; hash tombstones <= 1024 |
| `ResetGateway::pending_resets_` | runtime/episode/reset | request, completion, identity, ACK flags | Python reset request | Python reset ACK after Unity completion | `FinalizeResetLifecycle` | active payloads only; request/completion tombstones <= 1024 |
| `ZmqPythonResultRelay::unacknowledged_attempts_` | runtime/execution | retry bookkeeping | result relay | Python receipt/commit lifecycle | `ForgetResult` on commit | active relay state only |
| Python backend `_completed` | execution id | one completed execution result | successful execution | exact duplicate return or next execution | clear/replace and `close()` | one transaction |
| Python backend `_completed_resets`/`_reset_payloads` | reset identity | one reset completion/retry payload | successful/failed reset transaction | exact duplicate return or next reset | clear/replace and `close()` | one transaction |
| formal collector `CsvJournal` | journal path | disk journal; no full row list in formal mode | report append | recovery/compact boundary | stream/close; disk artifact retained | `retain_rows=False`: counters and last row only |
| formal collector accepted accounting | scalar counter | integer accepted count | accepted outcome | progress/finalization | not a row container | O(1) runtime state |

There is no TTL-based deletion.  Exact retry bytes remain available while a
transaction is active.  Terminal tombstones retain only identity and hashes;
they are diagnostic/idempotency state, not formal lookup payloads.  Diagnostic
counters and peak values do not alter the wire or artifact schema.

## Earliest safe erase boundaries

For a primitive endpoint snapshot, the earliest safe boundary is the accepted
Python `PrimitiveExecutionResultCommit`.  Python cannot commit a valid
transition until it has read and validated the exact snapshot, so the full
snapshot remains available through Unity response retry and Python lookup.
The commit callback then finalizes the snapshot, command, result, and relay
payloads together.

For a reset snapshot there is no primitive result commit.  The current v4
snapshot request/reply protocol has no separate Python snapshot receipt.  The
existing delivery boundary is therefore a successful bridge ROUTER send of
the exact response; failed sends leave the active request available for retry.
The reset request itself remains active until its normal Python reset ACK, then
only its bounded request/completion tombstone remains.

## Changes made

- Added terminal cleanup and bounded identity/hash tombstones to the snapshot,
  result, command, and reset lifecycle stores.
- Connected accepted Python result commit to snapshot and command finalizers.
- Removed committed result retry bookkeeping from the ZMQ Python relay.
- Added active/peak/terminal cache metrics for diagnosis only.
- Changed formal collection report journaling to streaming retention and made
  progress/target accounting scalar rather than a retained report list.
- Bounded the Python reliable backend's completed transaction caches and made
  the result transport discard a repeated result at its commit terminal event;
  only genuinely out-of-order active results remain in its cache, and
  `close()` releases them.  No artificial hard cap changes the transport
  capacity semantics.
- Added C++ lifecycle stress fixtures and Python regression coverage.

No Unity source or binary, Python planning algorithm, task/observation
contract, protocol schema, output row, transition field, or formal artifact
was changed.

## Validation

Completed checks:

```text
Release planning C++ build: PASS
Python compileall: PASS
Python/C++ lifecycle focused tests: 28 passed
C++ bridge lifecycle stress: 10,000 snapshot + 10,000 result + 10,000 command transactions: PASS
C++ ResetGateway lifecycle stress: 10,000 reset transactions: PASS
Real bridge snapshot/result commit cache cleanup test: PASS
Real bridge result/runtime integration suite: 23 passed; one initial startup-race failure was reproduced as a single-test PASS on rerun
Bounded 1,000-row bridge transport stress: PASS
Real Unity/Bridge 4-worker short regression after the transport fix: PASS
  (27 accepted / 31 attempted, collector errors 0)
```

The post-fix W4 and W12 bounded runtime diagnostics are written only under
`/tmp` and were evaluated from their final resource samples and collection
summaries.  W4 completed naturally at 803 accepted / 907 attempted after
approximately 70 minutes with all quality counters zero and complete cleanup.
The user then changed the W12 stop condition to one hour; W12 completed an
orderly stop at 1.01 hours with 1,989 accepted / 2,249 attempted and 61,075
reliable rows.  This is sufficient one-hour evidence for the requested
diagnostic, but it is not the original two-hour W12 certification gate, and no
W16 run was started.

## W4/W12 stability evidence — 2026-08-30

W4 output: `/tmp/xm-collection-runtime-lifecycle-memory-fix-v1-w4-final-retry-20260830/`.
It ended with `collection_summary.json` status `PASS`, 803 accepted, 907
attempted, 24,524 reliable rows, legacy/telemetry/snapshot/skew/frame/errors
all zero, and no remaining runtime processes.

W12 output: `/tmp/xm-w12-long-stability-20260830/`.  It used the formal
100,000-mission pool and MissionRouteStore as read-only inputs, the current
devel Bridge, formal Unity Player, Voxel cache, MPL, Task V2, and
`reliable_exact_endpoint_snapshot`.  It ran with 12 workers from
`2026-08-30T12:38:04.779Z` to `2026-08-30T13:38:44.322Z` and stopped through
the collector's existing `.collection_stop` path.  Final summary quality was
`PASS`: 2,249 attempted, 1,989 accepted, 61,075 reliable rows, acceptance
rate 0.884393, legacy rows 0, telemetry lookups 0, snapshot missing 0, state /
depth skew 0 ns, frame failures 0, collector errors 0, and endpoint identity
chain valid.

After a five-minute warm-up, aggregate RSS slopes were approximately +0.17
MiB/h/worker for Bridge, +1.76 MiB/h/worker for collectors, and +8.24
MiB/h/worker for Unity.  The minimum available RAM was 42.91 GiB, swap was
zero, process count was 85, FD count was 2,258, and the automated system
responsiveness gate passed for every sample.  W8 was not run per user
instruction.  W16 was not run after the user's one-hour stop instruction;
the original two-hour W12/W16 certification remains open.

`tracemalloc` top-allocation evidence was not obtained in the earlier run; the
previous attempt used a misnamed `sitecustomize` file and is invalid.  This is
an optional diagnostic gap, not a current release blocker.  The stale unit
fixture was reconciled after the user accepted the one-hour W12 gate:
`docs/pre_collection_runtime_final_v3.json` now uses the current devel Bridge
SHA `667c266e96211aabb5a0cd25b8b35b303903d35cbe2d95be8f9d2bbe5e63d789` in
`runtime_identity`; the previous
`462e9f7bc25d5bef63ce17b863daa256726e17de870bda04a1bcb72e79e80f2b` remains
explicitly classified as `HISTORICAL_RUNTIME_FIXTURE` and is not validated as
the current runtime identity.

```text
ALGORITHM_CHANGED=NO
UNITY_BEHAVIOR_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
FORMAL_60K_EXECUTED=NO
BC_TRAINING_EXECUTED=NO
AWAC_EXECUTED=NO
COMMIT_EXECUTED=NO
W4_EVIDENCE_SUFFICIENT=YES
W8_TESTED=NO
W12_ONE_HOUR_STABILITY=PASS
W12_TWO_HOUR_CERTIFICATION=NOT_RUN
W16_TESTED=NO
HISTORICAL_FULL_TESTS_BEFORE_BRIDGE_FIX=862 passed, 42 skipped, 1 stale Bridge identity fixture failure
```

## W12 acceptance and W16 formal release gate — 2026-08-30

The user-accepted one-hour W12 gate is final; W12 was not rerun and W8 was
not inserted.  The post-fix full suite, run with the current devel native
libraries and local socket creation enabled, completed with:

```text
W12_USER_ACCEPTANCE_GATE=PASS
STALE_BRIDGE_FIXTURE_FIXED=YES
CURRENT_BRIDGE_SHA256=667c266e96211aabb5a0cd25b8b35b303903d35cbe2d95be8f9d2bbe5e63d789
FULL_TESTS=864 passed, 42 skipped, 0 failed
```

The W16 estimate uses the W12 stable resource samples: minimum available RAM
42.908756 GiB, maximum observed W12 process-tree RSS 16.256531 GiB, and a
linear four-worker increment of 5.418844 GiB.  The resulting conservative
estimate is 37.489913 GiB minimum available headroom, above the 15 GiB gate.
This is an estimate, not a W16 runtime measurement.

```text
W16_ESTIMATED_RAM_HEADROOM_GB=37.489913
W16_FORMAL_READY=YES
FORMAL_COLLECTION_WORKERS=16
FORMAL_COLLECTION_RUN=flight_reliable_exact_teacher_seed2026_20260830_v7
FORMAL_60K_EXECUTED_BY_CODEX=NO
```
