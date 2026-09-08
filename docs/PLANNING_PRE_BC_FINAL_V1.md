# Planning Pre-BC Final V1

Date: 2026-08-28

Status: `PASS` for the controlled P to O cutover and bounded Pre-BC
qualification chain.

## Canonical owner

The canonical source is now:

```text
PLANNING_ROOT=/home/xm/XM/xm_ws/src/planning
PYTHON_ROOT=/home/xm/XM/xm_ws/src/planning/python
MOTION_PRIMITIVES=/home/xm/XM/xm_ws/src/planning/data/motion_primitives
```

The historical optimized source `/home/xm/XM/src` remains unchanged and is
not an import or runtime fallback. Validation uses `conda activate xm`; ROS
message imports come from the O workspace devel tree.

## Stage contract

The bounded final chain is:

```text
formal 105-action MPL
  -> mission generation
  -> mission audit
  -> reliable exact endpoint collection
  -> Teacher relabel
  -> dataset provenance audit
  -> depth action masks
  -> BC mmap dataset
  -> one-epoch CPU BC smoke
  -> one-episode reliable evaluation audit
```

The O result is:

```text
MPL=PASS
MISSION_GENERATION=PASS
MISSION_AUDIT=PASS
RELIABLE_COLLECTION=PASS
RELABEL=PASS
DATASET_AUDIT=PASS
DEPTH_MASKS=PASS
BC_MMAP=PASS
BC_SMOKE=PASS
EVALUATION_AUDIT_SMOKE=PASS
```

The observation contract is
`reliable_exact_endpoint_snapshot`. For the 2-worker chain, the collection
manifest recorded 176 reliable rows and four accepted episodes; downstream
artifacts contain 114 accepted transitions. All downstream artifacts agree
on episode IDs `[2, 4, 7, 9]`, offsets `[0, 28, 56, 85]`, lengths `[28, 28,
29, 29]`, and action count 105.

## Provenance gates

The collection, labels, masks, dataset audit, BC mmap, BC smoke, and
evaluation audit all validated the same source chain. The required counters
were:

```text
LEGACY_ROWS=0
TELEMETRY_LOOKUP_COUNT=0
SNAPSHOT_MISSING_COUNT=0
STATE_DEPTH_SKEW_MAX_NS=0
FRAME_CONTRACT_FAILURES=0
ENDPOINT_IDENTITY_CHAIN_VALID=true
```

The BC mmap, BC checkpoint, and evaluation summaries retain their source
index/manifest hashes, MPL contract hash, observation contract, row counts,
episode identity, and mask/label contract identities. The complete bounded
hash evidence is `/tmp/xm-py6-o-e2e/sha_chain.json`.

## Runtime and test gates

```text
O clean/devel/install build=PASS
O Python import root=PASS
full Python suite=756 passed, 42 skipped, 0 failed
canonical CLI help=11/11
12-worker startup/short-transition smoke=PASS
process cleanup=PASS
data/history preservation=PASS
```

The runtime smoke used the explicitly frozen C7 Bridge artifact together
with the fixed Player and Assembly-CSharp hashes. The 12-worker smoke used
`MAX_STEPS=3` and `TARGET_ACCEPTED=0`; its parent accepted-zero quality gate
is intentional infrastructure evidence, not collection evidence.

## Explicit non-claims

This document does not authorize or report:

- formal 60,000-row collection;
- formal BC training or quality promotion;
- AWAC/SAC/RL execution;
- CUDA collision runtime parity;
- changes to Unity, C++, Bridge, BC/AWAC mathematics, or historical data.

The next phase is user execution of formal reliable BC data collection under
the preserved runtime identity and provenance gates.

## PY7-A common/config hygiene checkpoint

PY7-A centralized the targeted generic common seams and experiment defaults in
the canonical O tree.  The bounded 12-worker infrastructure smoke was rerun
with `MAX_STEPS=3` and `TARGET_ACCEPTED=0`: all 12 workers started and stopped,
with unique identities/ports and zero collector, legacy, telemetry, snapshot,
skew, and frame failures.  Its accepted-zero quality gate is intentionally not
a data-quality claim.  The existing `/tmp/xm-py6-o-e2e/` SHA chain remains
`PASS`, and the current dataset audit still reports 4/4 episodes and 114
transitions with the exact reliable observation contract.

The Python full suite after PY7 returned `763 passed, 42 skipped, 0 failed`;
the skip count is unchanged.  Protocol golden tests returned `13 passed`, and
canonical CLI help returned `11/11`.  No formal 60,000-row collection, BC
training, AWAC/SAC run, Unity change, C++ source change, wire change, or
artifact schema change was performed.

The generic owner counts are zero, and `max_steps` has one Python default
owner (`contracts/task.py`, value 45).  The preceding text records the V1
checkpoint; PY7-A1 closes the task-contract issue with an explicit V2
identity. Cross-language naming is still inventory-only; no rename was
executed.

## PY7-A1 task-contract V2 closure (2026-08-29)

The historical V1 identity is frozen exactly at:

```text
5862af0f354c408d74cf4904dc7ecf609944553443ae1128097136be979fd30a
```

The formal Pre-BC contract is now V2, owned by
`python/planning/contracts/task.py`:

```text
TASK_CONTRACT_SCHEMA_VERSION=2
MAX_PRIMITIVE_STEPS_IN_METADATA=YES
TASK_CONTRACT_V2_SHA=2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df
MAX_STEPS_CHANGE_CHANGES_SHA=PASS
FORMAL_PRE_BC_TASK_CONTRACT=V2
```

Mission candidates/audit, reliable collection root and workers, Teacher
labels, dataset audit, depth masks, BC mmap/checkpoint, and evaluation
summary each carry the V2 identity fields. Explicit V1 historical mode is
marked `legacy_v1`; missing schema is not auto-upgraded and V1 cannot enter
the formal V2 path. AWAC/SAC remains frozen on explicit legacy fixtures.

The current bounded chain is recorded at
`/tmp/xm-py7-a1-v2-chain/sha_chain.json` and has
`TASK_CONTRACT_CHAIN_MISMATCH_COUNT=0`. Teacher labels, depth masks, BC mmap
arrays, BC model/normalizer/logits, and evaluation business fields are exact
parity with the pre-V2 bounded fixture; only contract identity/provenance and
non-business timing fields changed. The runtime still uses max steps 45 and
all reliable observation counters remain zero.

PY7-A1 validation returned `772 passed, 42 skipped, 0 failed`, canonical CLI
`11/11`, and the bounded reliable evaluation smoke passed. No formal 60,000
row collection, BC production training, AWAC/SAC run, Unity/C++ change, or
rename was performed. The next phase is `PY7-B EXECUTE BOUNDED
CROSS-LANGUAGE RENAME`.
