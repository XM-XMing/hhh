# PY0 reliable-exact Teacher collection

Date: 2026-08-28

P (`/home/xm/XM/src`) owns the formal collection path. O
(`/home/xm/XM/xm_ws/src/planning`) is the read-only behavior baseline and
Unity (`/home/xm/XM/XMflight`) is frozen.

## Formal owner and path

The only default formal collector is `scripts/collect_teacher_rollouts.py`,
invoked by `scripts/collect_rollouts_parallel.py`. The shell wrapper only
bootstraps ROS and executes Python. The implementation is split across:

- `planning/teacher/collection_config.py`, `collection_run.py`,
  `rollout_collector.py`, `rollout_session.py`, and `parallel_collection.py`;
- `planning/runtime/ports.py`, `worker.py`, `supervisor.py`, `process.py`,
  `health.py`, `lifecycle.py`, and `identity.py`;
- `planning/data/rollout.py`, `rollout_merge.py`, and `rollout_shard.py`.

The old asynchronous collector remains source-visible only behind the explicit
`--diagnostic-legacy-async` option. It is not the default and is not a formal
BC-data caller.

## Exact observation contract

Formal rows use only:

```text
observation_contract = reliable_exact_endpoint_snapshot
observation_source   = reliable_exact_endpoint_snapshot
protocol_version     = 4
reliable_execution   = true
telemetry_observation = false
telemetry_lookup_count = 0
state_depth_skew_ns = 0
snapshot_missing_count = 0
legacy_rows = 0
endpoint_identity_chain_valid = true
```

Each action is submitted as one reliable command, checked against its result
and ACK/commit receipt, and paired with the authoritative endpoint state and
depth snapshot returned by that transaction. `execute_primitive_stream`,
`/xm/state`, and `/xm/depth/image_raw` are excluded from the formal function.
The existing feature arrays, masks, previous action, row order, and NPZ
schema are unchanged.

## Shards, aggregate target, and resume

`WorkerRuntimeSpec` is the only process endpoint owner. Runtime identity,
ports, ROS home, logs, stop file, and worker shard paths are deterministic.
`target_accepted` is interpreted by the parent as an aggregate target; workers
do not receive a per-worker target quota. The parent creates a shared stop
file after the aggregate target is reached.

The post-collection merge has two deliberately separate quality results:

- `runtime_quality_pass` describes the complete collection history. It remains
  false when a failed attempt is recorded, even when that attempt has no
  accepted artifact.
- `accepted_dataset_quality_pass` describes the published accepted-only
  selection. It requires the target count, an existing and
  `load_rollout_episode(..., validate=True)`-valid NPZ for every selected row,
  exact observation provenance, and unique episode/transition identities.

`quality_pass` is retained as a compatibility field and is now an explicit
legacy alias of `runtime_quality_pass`; it must not be used to block a clean
accepted-only dataset. Non-accepted NPZ/temp/partial artifacts are removed
only after the accepted dataset gate passes. Raw reports, journals, and logs
remain run evidence. Consumers that need the original all-attempt runtime
gate can invoke `validate_teacher_collection.py --require-runtime-quality`.

Each worker report is an idempotency ledger keyed by run ID, runtime instance,
mission ID, episode ID, and exact transition IDs. Resume rejects mixed runtime
identities, duplicate identities, missing transition identities, and accepted
rows without a transition. A restarted Unity process may reset its protocol
execution counter, so the resumed Python session allocates transition IDs
above the largest persisted numeric suffix; protocol execution IDs themselves
are unchanged. The ledger is validated before a report row is appended. Merge
is deterministic, retains source shards, and rejects cross-worker
mission/episode/transition collisions or mixed legacy/exact provenance.

## Manifest identity

Worker resolved configuration, worker summary, report rows, and the merged
summary carry the exact observation counters plus:

- `unity_player_sha256`;
- `runtime_assembly_sha256` / `assembly_csharp_sha256`;
- `bridge_sha256`;
- `WorkerRuntimeSpec` identity and endpoint projection;
- Teacher, mission-index, MPL, collision-cache, source, and protocol identity.

The runtime artifact hashes are resolved by `planning.runtime.identity` and a
missing Player, Assembly, or Bridge artifact fails closed before reset.

## Verification record

All Python commands use `conda activate xm` and `PYTHONPATH=python` (plus the
ROS generated-message path when ROS tests are run). The focused PY0 seam suite
covers exact field mapping, missing snapshot/telemetry/identity/frame failure,
lost-result/retry contracts, aggregate targets, merge compatibility,
collisions, resume idempotency, twelve-worker port isolation, and process
cleanup.

The valid real bounded smokes used the existing read-only O mission/cache and
the frozen Unity Player. The one-worker exact smoke produced two accepted
missions and 56 exact transitions. The two-worker aggregate smoke produced 22
accepted missions and 696 exact transitions, with merge quality gates passing
and zero legacy rows, telemetry lookups, snapshot misses, skew, or frame
failures. A separate resume smoke stopped after three accepted missions and
then resumed to five accepted missions; all 141 transitions remained unique
and the final merge passed. The resumed process used the same worker identity
and a new transition-ID offset after Unity execution IDs reset. A 50-target
one-worker boundary run was intentionally stopped at 46 accepted missions;
its target gate was therefore not used as a passing artifact. No formal
60,000-row collection is permitted in PY0.

The final runtime artifact identities were Player SHA
`61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365`,
Assembly-CSharp SHA
`7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc`, and P
Bridge SHA
`63b00f6a0b31aa72ace3e1c437477e7ee8b670934386f3bb1d645745c302d9ed`.

## Remaining chain

PY0 closes the formal collection owner, exact endpoint provenance seam,
aggregate merge, and resume idempotency seam. The next controlled phase is
PY2 relabel and depth-mask producer provenance. Neither phase is executed by
this document.
