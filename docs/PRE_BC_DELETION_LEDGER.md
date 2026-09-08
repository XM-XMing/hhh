# PY3 pre-BC deletion ledger

Date: 2026-08-28

This ledger is intentionally a no-delete record. PY3 only established the
canonical bounded pipeline and its keep set. It did not perform PY4 cleanup.

```text
DELETE_DUPLICATE=0
DELETE_LEGACY_FORMAL=0
DELETE_DEAD=0
DEFER_UNKNOWN=0
```

## Safe first deletion batch

```text
SAFE_FIRST_DELETION_BATCH = []
```

No files are listed because no candidate has yet passed all of these gates:

1. no import, script, CMake/install, launch, or test reachability;
2. no protected RL/AWAC/checkpoint/resume dependency;
3. no O/P parity or historical evidence role;
4. no formal CLI or runtime bootstrap ownership;
5. deletion reviewed against the canonical pre-BC keep set.

The explicit diagnostic legacy async path is retained as
`KEEP_TEST_REFERENCE`; it is not a second formal owner. The shell collection
wrapper is retained because it is a public bootstrap entrypoint, although PY3
identified its workspace-resolution hazard. Neither fact justifies deletion.

Confirmed generated outputs were kept under `/tmp/xm-pre-bc-py3/` and no P
`data/` artifact was populated. Existing tests, fixtures, evidence, historical
artifacts, and RL files were not removed.

## PY4 handoff

PY4 may propose a non-empty deletion batch only after rebuilding the inventory,
checking the exact file paths, running import/build/CLI tests, and recording a
recoverable review. The first action must remain a proposal/listing step; it
must not silently delete a directory or a broad glob.

```text
NEXT_PHASE=PY4 EXECUTE SAFE DELETION AND CANONICAL CODE CLEANUP
```

## PY4-A audit result

PY4-A performed the production install, canonical-entry, symbol, wrapper,
configuration, RL-boundary, MPL, and generated-cache audit. It did not delete
production files or symbols. The only source change was the path/bootstrap
fix in `scripts/collect_rollouts_parallel.sh`; no source cache was removed.

```text
PY4_A_PRODUCTION_SURFACE_AUDIT=PASS
SAFE_SYMBOL_DELETION_COUNT=0
SAFE_FILE_DELETION_COUNT=0
SAFE_DELETION_BATCH=[]
GENERATED_FILE_COUNT_BEFORE=56
GENERATED_FILE_COUNT_AFTER=56
```

The one exact symbol-body duplicate is a protected RL/checkpoint helper. The
compatibility `planning.common.atomic` re-export is used by frozen RL/SAC
callers. Same-name local helpers, diagnostic/reference symbols, tests, golden
vectors, and versioned contract names were not deletion candidates.

```text
NEXT_PHASE=PY5 P CLEAN BUILD AND O CUTOVER PREPARATION
```
