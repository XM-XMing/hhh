# PY3 pre-BC canonical pipeline

Date: 2026-08-28

```text
O=/home/xm/XM/xm_ws/src/planning   (read-only behavior baseline)
P=/home/xm/XM/src                 (development mainline)
OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
```

This document records the bounded PY3 pre-BC smoke. It is not a formal
60,000-episode collection, a BC quality evaluation, or an AWAC/SAC run. O,
Unity, C++, collision semantics, Teacher math, BC math, and AWAC/SAC were not
modified.

## Stage result

| Stage | Result | Evidence |
|---|---|---|
| Motion primitives | PASS | Newly generated 105-action, 25-frame MPL under `/tmp/xm-pre-bc-py3/motion_primitives/`; contract SHA `f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d` |
| Mission generation | PASS | 200 published deterministic candidates; internal route sampling attempted 2000 candidates; candidate index SHA `2e92f311a409056a85ad4ef437e121276fe3add92c45a97884f170ea853afc1f` |
| Mission audit | PASS | 60 passing missions selected from 74 audited rows; audited index SHA `d809a5ee364461c93b740189602856e923cab918e18a895d25f528ac382e4755` |
| Reliable exact collection | PASS | 57 attempts, 21 accepted episodes; all exact provenance counters zero |
| Merge | PASS | merged `rollout_index.csv` and root `collection_summary.json` |
| Relabel | PASS | 21 episodes / 598 transitions; behavior and teacher validity 100% |
| Dataset audit | PASS | 21 episodes / 598 transitions; provenance, episode identity, and offsets checked |
| Depth masks | PASS | 21 episodes / 598 rows / 105 actions; strict rollout provenance checked |
| BC mmap | PASS | 21 episodes / 598 rows; mmap manifest contract and source hashes checked |
| BC training smoke | PASS | mmap mode, one CPU epoch, 17 train episodes / 4 validation episodes, `quality_pass=true` |
| Reliable evaluation smoke | PASS | 10 separate reliable-v4 audit-only episodes; runtime contract passed; quality gate intentionally not applicable |

The aggregate bounded result is:

```text
PY3_PRE_BC_PIPELINE=PASS
PRE_BC_E2E_SMOKE=PASS
```

## Exact observation-row accounting

The collection root report contains one row for every attempted mission,
including rejected attempts. The accepted merged index contains one NPZ for
each accepted mission.

```text
ATTEMPT_EXACT_OBSERVATION_ROWS=1018
ACCEPTED_PERSISTED_TRANSITION_ROWS=598
REJECTED_ATTEMPT_OBSERVATION_ROWS=420

1018 = 598 + 420
```

`ATTEMPT_EXACT_OBSERVATION_ROWS` is the sum of `reliable_rows` over all 57
attempt report rows. `ACCEPTED_PERSISTED_TRANSITION_ROWS` is the sum of the
21 accepted NPZ transition arrays. `REJECTED_ATTEMPT_OBSERVATION_ROWS` is the
same exact-observation row count for the 36 rejected attempts; those rows are
not persisted as accepted training transitions.

```text
LEGACY_ROWS=0
TELEMETRY_LOOKUP_COUNT=0
SNAPSHOT_MISSING_COUNT=0
STATE_DEPTH_SKEW_MAX_NS=0
FRAME_CONTRACT_FAILURES=0
```

All accepted rows use `reliable_exact_endpoint_snapshot`, with endpoint
identity validation enabled, `reliable_execution=true`, and
`telemetry_observation=false`.

## SHA lineage

The machine-readable verification is
`/tmp/xm-pre-bc-py3/sha_chain.json` and reports `ARTIFACT_SHA_CHAIN=PASS`.
The principal file hashes are:

| Artifact | SHA256 |
|---|---|
| MPL NPZ | `22ad22fe66a88633de6effff11c03aa4ca4e8df362f8b410e17effbd91ff7309` |
| MPL JSON | `c1b795e4737a12034b0d59a1b457f559e7ad138527df2c90ed35db1326e42ea8` |
| Candidate index | `2e92f311a409056a85ad4ef437e121276fe3add92c45a97884f170ea853afc1f` |
| Audited index | `d809a5ee364461c93b740189602856e923cab918e18a895d25f528ac382e4755` |
| Rollout index | `e6ac0a7e1a29bcf895d89e628081bd02b14bf5fea9c1c0183d79726ac18696f6` |
| Rollout manifest | `7a67cba2a65c3fd03a095240312c54dce7636300c562e9ea2c31b9aa98370803` |
| Teacher labels | `b2e152ab4d1cec590e92afe0c0e9fd3e3e1deb2d7d027d37c56ec89150f4e459` |
| Depth masks | `0515c50b1cce64ee3b950b015c6d710dba2e9f9963bc349b069d8baf8c44c237` |
| BC mmap manifest | `7b518f6eb8dae5283dfd3f099c963b462100482111d47559c1dce7e75712404a` |
| BC checkpoint | `721bf83d9c99f2ee6ef099abeb29d3a937a5553a34f86309b10c4a2da918597d` |
| Evaluation smoke aggregate | `630d81a3375fdf6c3dc1e72e558e23aa12894348d99e610265005eaa2f1bed48` |

The verified links are:

```text
MPL contract
  -> candidate index
  -> audited index
  -> rollout manifest/index
  -> teacher labels and depth masks
  -> BC mmap manifest
  -> BC checkpoint
  -> 10 evaluation summaries
```

The checkpoint payload itself repeats the mmap manifest, rollout index,
labels, masks, MPL, observation contract, and row-count identities. Every
evaluation summary repeats the checkpoint SHA, audited mission-index SHA, and
reliable exact runtime contract.

## O/P algorithm parity

The existing PY0 reliable fixture was run through the O/P relabel and depth
mask comparator:

```text
TEACHER_LABEL_ARRAY_PARITY=PASS
DEPTH_MASK_ARRAY_PARITY=PASS
EPISODE_ORDER_PARITY=PASS
TRANSITION_ORDER_PARITY=PASS
METADATA_BUSINESS_PARITY=PASS
```

The comparator covered 56 transitions and exact soft targets, global action
masks, teacher argmax, behavior actions, valid counts, entropy, local depth
masks, episode offsets, lengths, and business metadata. O does not expose the
current reliable producer metadata schema, so the current full smoke uses P's
canonical provenance validation while the numerical producer result remains
anchored to the O/P fixture oracle.

## Evaluation boundary

The evaluation smoke ran episode IDs `2,4,7,13,15,17,19,22,23,24` as ten
separate reliable-v4 audit-only runs. All ten had:

```text
reliable_execution=true
observation_contract=reliable_exact_endpoint_snapshot
observation_source=reliable_exact_endpoint_snapshot
snapshot_missing_count=0
telemetry_lookup_count=0
quality_gate_applicable=false
```

The observed terminal outcomes are runtime smoke evidence only (eight
`dead_end`, two `timeout`). They are not a success-rate conclusion and do not
promote this checkpoint to a formal evaluation result. The formal managed
holdout gate remains 100 episodes.

## Remaining blockers

- No formal 60,000-episode collection was performed by design.
- P has no formal MPL files under `P/data`; an explicit `/tmp` MPL override is
  required for this isolated smoke and must be resolved by the future formal
  launcher without writing historical data.
- At PY3 time `scripts/collect_rollouts_parallel.sh` could re-source an O
  workspace when invoked with the wrong `WORKSPACE`; the passing smoke used the
  direct P Python entrypoint plus a P-first ROS overlay. PY4-A closed this
  bootstrap ambiguity and verified the wrapper from source, devel, install,
  arbitrary-cwd, and explicit-workspace cases.
- O does not provide the current reliable producer metadata schema, so a
  same-full-fixture O provenance artifact cannot be compared; the bounded O/P
  numerical fixture parity is already exact.
- BC was trained for one bounded epoch and evaluation was audit-only; neither
  is a quality or generalization claim.
- AWAC/SAC remains frozen and was not run.

## PY4-A production surface closure

The PY4-A audit found 11 unique canonical pre-BC entries. Each is a thin
7-line adapter to one domain owner; no duplicate canonical entry was proven.
The P collection shell wrapper is now a bootstrap-only path resolver and was
verified from source-tree, devel-space, install-space, arbitrary cwd, and an
explicit catkin workspace. It does not change collection arguments, worker
ports, process lifecycle, merge behavior, or artifact schemas.

The O/P relabel-mask comparator remains exact. The existing bounded artifact
chain at `/tmp/xm-pre-bc-py3/sha_chain.json` remains `PASS`. P still has no
formal MPL pair under `data/motion_primitives`; the PY3 MPL remains a `/tmp`
fixture and was not copied.

```text
PY4_A_PRODUCTION_SURFACE_AUDIT=PASS
PRE_BC_BEHAVIOR_CHANGED=NO
ARTIFACT_SCHEMA_CHANGED=NO
ALGORITHM_CHANGED=NO
SAFE_DELETION_BATCH=[]
NEXT_PHASE=PY5 P CLEAN BUILD AND O CUTOVER PREPARATION
```

```text
ALGORITHM_CHANGED=NO
DATA_LAYOUT_CHANGED=NO
NEXT_PHASE=PY4 EXECUTE SAFE DELETION AND CANONICAL CODE CLEANUP
```

## PY6 canonical O pipeline

PY6 completed the controlled source cutover. The canonical pipeline root is
now `/home/xm/XM/xm_ws/src/planning`; the optimized tree remains a historical
candidate and is not on the validation import path.

```text
O formal MPL (105 actions)
  -> O mission generation
  -> O mission audit
  -> O reliable exact-endpoint collection
  -> O Teacher relabel
  -> O dataset provenance audit
  -> O depth action masks
  -> O BC mmap
  -> O one-epoch BC smoke
  -> O reliable evaluation audit smoke
```

The bounded 2-worker chain passed with four accepted episodes, 176 reliable
collection rows, and 114 downstream transitions. It retained exact episode
order, transition offsets, action count, observation contract, source hashes,
and endpoint identity across labels, masks, audit, mmap, checkpoint, and
evaluation. The SHA-chain record is `/tmp/xm-py6-o-e2e/sha_chain.json`.

The 12-worker infrastructure smoke passed startup, unique runtime identity,
canonical port allocation, short transition processing, and cleanup. Its
`TARGET_ACCEPTED=0` parent quality-gate rejection is intentional and does not
constitute a data-quality result.

The final contract is documented in `docs/PLANNING_PRE_BC_FINAL_V1.md` and
the cutover evidence in `docs/PY6_CUTOVER_REPORT.md`. Formal 60,000-row
collection, formal BC training, and AWAC/RL remain outside this completed
phase.
