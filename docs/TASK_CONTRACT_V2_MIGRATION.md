# Task Contract V2 migration

Date: 2026-08-29

Status: `PY7_A1_TASK_CONTRACT_V2=PASS` for the canonical O Planning tree.
Only `/home/xm/XM/xm_ws/src/planning` was modified. P and the Unity Player
remain frozen. No formal collection, formal BC training, AWAC/SAC run, or
rename was performed.

## Contract owner and identities

The single implementation owner is
`python/planning/contracts/task.py`. It owns the canonical metadata
serializer, SHA calculation, formal V2 fields, and the explicit historical
V1 validators. `mission/spec.py` only re-exports that seam.

The V1 metadata bytes remain unchanged:

```text
TASK_CONTRACT_V1_SCHEMA=1
TASK_CONTRACT_V1_SHA256=5862af0f354c408d74cf4904dc7ecf609944553443ae1128097136be979fd30a
```

Formal Pre-BC artifacts now use:

```text
task_contract_id=xm_3d_flight_z1_3
task_contract_schema_version=2
max_primitive_steps=45
task_contract_sha256=2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df
task_contract_mode=formal_v2
```

`mission.max_steps` in `config/pre_bc.yaml` is the experiment input and is
resolved by the task-contract owner as `max_primitive_steps`. Changing 45 to
40 produces
`964e1a970369eb11431ced9db88f6d3599713fdb8b55d4e1870276997842b32d`.

## Legacy boundary

`legacy_task_contract_v1_metadata()` and
`legacy_task_contract_v1_sha256()` are an explicit historical seam. V1 is
accepted only by callers that select historical/diagnostic validation. A
missing schema is not upgraded, V1 and V2 are not mutually accepted, and a
V1 artifact cannot enter a formal V2 Pre-BC producer. Historical summaries
carry `task_contract_mode=legacy_v1`.

The frozen AWAC/SAC path remains on its explicit V1 fixture/checkpoint
identity. Its mathematics, replay, cadence, and checkpoint selection were
not changed. It does not select the new formal Pre-BC default.

## Artifact coverage

Every new formal artifact carries the four identity fields independently:

| Stage | V2 fields | Validation boundary |
|---|---|---|
| Mission candidates | yes | `validate_mission_rows` |
| Mission audit | yes | audit checkpoint, rows, summary |
| Reliable collection root/worker | yes | rollout provenance |
| Teacher labels | yes | label artifact provenance |
| Dataset audit | yes | dataset summary and cross-artifact check |
| Depth masks | yes | depth-mask artifact provenance |
| BC mmap | yes | `validate_bc_mmap_provenance` |
| BC checkpoint | yes | BC checkpoint load validation |
| Evaluation summary | yes | evaluator task-contract validation |

Formal V2 rejects missing or unknown schema, missing `max_primitive_steps`,
SHA mismatch, a max-step mismatch with resolved configuration, and V1 input.
The validation is fail-closed before producer output is accepted.

## TDD and chain evidence

The A-I tests are in
`tests/test_py7_a1_task_contract_v2.py`. They cover the frozen V1 SHA, V2
metadata and deterministic SHA, the 45-to-40 SHA change, V1/V2 separation,
missing/max mismatch rejection, formal V1 rejection, and explicit historical
validation.

The bounded V2 chain evidence is:

```text
/tmp/xm-py7-a1-v2-chain/sha_chain.json
TASK_CONTRACT_CHAIN_MISMATCH_COUNT=0
episodes=4
accepted transitions=114
episode_ids=[2,4,7,9]
transition_offsets=[0,28,56,85]
transition_lengths=[28,28,29,29]
```

The chain was built from a temporary copy of the existing bounded reliable
fixture. Its V2 collection metadata was constructed explicitly for the
fixture; no historical artifact was silently auto-upgraded by production
code. Current O relabel, dataset audit, depth-mask, BC mmap, one-epoch CPU
BC smoke, and one-episode reliable evaluation smoke all passed strict V2
validation. The evaluation was audit-only (`quality_gate_applicable=false`)
and used the frozen Unity runtime.

The O/P historical V1 relabel/mask comparator passed all label arrays, mask
arrays, episode order, transition order, and business metadata. The current
O V2 arrays are byte-identical to the pre-V2 O fixture. BC mmap arrays,
normalizer, model state, fixed policy logits, and evaluation business metrics
also match; only provenance/identity and wall-clock timing fields differ.

```text
V2_CHAIN_VALIDATION=PASS
TEACHER_LABEL_ARRAY_PARITY=PASS
DEPTH_MASK_ARRAY_PARITY=PASS
BC_MMAP_ARRAY_PARITY=PASS
POLICY_LOGIT_PARITY=PASS
EVALUATION_BUSINESS_PARITY=PASS
LEGACY_ROWS=0
TELEMETRY_LOOKUP_COUNT=0
SNAPSHOT_MISSING_COUNT=0
STATE_DEPTH_SKEW_MAX_NS=0
FRAME_CONTRACT_FAILURES=0
```

## Validation gates

```text
COMPILEALL=PASS
FULL_TESTS=772 passed, 42 skipped, 0 failed
CANONICAL_CLI=11/11
TWELVE_WORKER_FOCUSED=17 passed
RELIABLE_EVALUATION_SMOKE=PASS
ALGORITHM_CHANGED=NO
RUNTIME_BEHAVIOR_CHANGED=NO
ARTIFACT_IDENTITY_CHANGED=YES
```

The 772 total includes the nine new A1 tests; the existing skip count stayed
at 42. The twelve-worker result retains the previously validated bounded
accepted-zero infrastructure smoke and current runtime identity/port/cleanup
tests; it is not formal data evidence.

The next authorized phase is `PY7-B EXECUTE BOUNDED CROSS-LANGUAGE RENAME`.
It was not started in this phase.
