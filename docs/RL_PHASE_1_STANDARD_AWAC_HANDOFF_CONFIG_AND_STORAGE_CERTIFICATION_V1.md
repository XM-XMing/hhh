# RL Phase 1 Standard AWAC Handoff, Config, and Storage Certification

Date: 2026-09-04

This report covers the construction-time handoff from the existing Formal
Critic Calibration PASS artifact to Standard AWAC, plus checkpoint retention,
replay storage, and disk-budget policy. No Actor update, AWAC online
interaction, Unity, Bridge, ROS runtime, Final300 evaluation, or training run
was executed.

## Certification result

```text
RL_PHASE_1_STANDARD_AWAC_HANDOFF_CONFIG_AND_STORAGE_CERTIFICATION_V1=PASS
STANDARD_AWAC_HANDOFF_ALLOWED=YES
STANDARD_AWAC_STATE_TRANSFER=PASS
ACTOR_ENABLE_FAIL_CLOSED=PASS
ACTOR_UPDATE_EXECUTED=NO
ACTOR_OPTIMIZER_STEP_COUNT=0
```

The source artifact is:

```text
data/awac/awac_bc60k_formal_critic_calibration_v6/checkpoint_calibration_pass.pt
sha256=bb9433d8d9fb2fde2e2148e84a4058c2b328b9d676608d38c00962f821c3d8de
```

The live, read-only loader verified `calibration_gate_state=PASS`,
`calibration_pass_checkpoint=true`, a `COMMITTED` transaction, exact replay
identity, and matching BC, MPL, Task, Observation, Reward, Replay, and policy
input contracts. The state-transfer check constructed the Actor, twin Critics,
target Critics, optimizers, and replay handoff without taking an optimizer
step. The Actor is enabled only by the explicit validated handoff object.

Pending, diverged, non-pass, uncommitted, contract-mismatched, and replay-
mismatched inputs are fail-closed. Standard training also requires both the
Calibration PASS resume checkpoint and its replay directory before argument
validation succeeds.

## Resolved training-config audit

The V6 persisted resolved configuration has SHA256
`053b3e7aa00616cd2f782f75f16b7a9d786bfd4636d9a7f278ca5d56d80c8f49`.
The current Standard AWAC configuration reconstructed from the real CLI has
SHA256 `2075af1d8a1d175373da668ed0ae80902b498bd862e5432e5e663b4da25d5e14`.
The current Calibration CLI reconstruction has SHA256
`d8f4812ead4aa26a6f5d3710336ac00df873b0ed818e7991d93063a9eff6dc0b`.

The previously reported dry-run SHA
`a8f14ff4c975692dec840f1723f31e034fafe8e6523159d1a892c2c6847a48d3` has no
current workspace artifact or reproducible command/config source. It is kept
as a historical claim, not used as current evidence.

Field-level differences (the calibration value is absent where the field was
introduced by the Standard handoff) are:

| Field | Calibration V6 | Standard handoff | Classification |
| --- | --- | --- | --- |
| `training_contract.phase` | `critic_calibration` | `awac_training` | `EXPECTED_PHASE_CHANGE` |
| `training_contract.actor_update_enabled` | `false` | `true` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.actor_depth_lr` | absent | `1e-6` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.critic_depth_lr` | absent | `1e-5` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.actor_enabled_phase` | absent | `ACTOR_ENABLED_STANDARD_AWAC` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.actor_update_enabled_before_transition` | absent | `false` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.initial_phase` | absent | `CRITIC_CALIBRATION` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.replay_behavior_sources` | absent | `BC_CALIBRATION, AWAC_ONLINE` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.online_transition_cap` | absent | `50000` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.calibration_safety_cap.max_transitions` | absent | `30000` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.calibration_safety_cap.max_episodes` | absent | `1000` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.collapse_margin` | absent | `0.1` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.dev_evaluation_role` | absent | `DEV` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.dev_evaluation_mission_count` | absent | `100` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.milestones` | absent | `10K, 25K, 50K DEV milestones` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.schema_version` | absent | `1` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.contract_id` | absent | `awac_phase1_readiness_v1` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract.contract_sha256` | absent | `6a3ae6e650db8e60b693396fbc3ac15bf6f6030c86dcd650dc1b6534a63c4233` | `EXPECTED_ACTOR_ENABLE_CHANGE` |
| `phase1_training_contract_sha256` | absent | `6a3ae6e650db8e60b693396fbc3ac15bf6f6030c86dcd650dc1b6534a63c4233` | `EXPECTED_ACTOR_ENABLE_CHANGE` |

Field comparison between the V6 persisted configuration and the reconstructed
Standard configuration:

| Classification | Count | Result |
| --- | ---: | --- |
| Total changed fields | 19 | closed |
| `EXPECTED_PHASE_CHANGE` | 1 | `critic_calibration` -> `awac_training` |
| `EXPECTED_ACTOR_ENABLE_CHANGE` | 18 | Phase-1 contract and Actor-enable fields only |
| `INVOCATION_ONLY` | 0 | no difference |
| `IDENTITY` | 0 | no difference |
| `TRAINING_SEMANTIC` unexpected | 0 | PASS |
| `UNKNOWN` | 0 | PASS |

The frozen optimization values remain unchanged, including fixed
`bc_kl_weight=0.05`, `gamma=0.99`, `tau=0.005`, reward scale `0.10`, CQL
weight `0.05`, and the Phase-1 depth learning rates. Confidence, adaptive BC
KL, and primitive-neighbor algorithms remain unimplemented.

## Checkpoint storage audit

All byte values below use GiB (`1024^3`) and were measured read-only from the
V6 directory on 2026-09-04.

| Item | Bytes | GiB |
| --- | ---: | ---: |
| V6 tree, allocated/physical | 8,977,014,784 | 8.360497 |
| V6 tree, apparent | 10,323,774,501 | 9.614764 |
| `checkpoint_last` generations | 177 | — |
| `checkpoint_last` generation files, physical | 8,728,920,064 | 8.129440 |
| Calibration-pass generation files | 1 | — |
| Calibration-pass generation, physical | 91,975,680 | 0.085659 |
| All checkpoint generation files, physical | 8,820,895,744 | 8.215099 |

The root cause of the historical growth is `KEEP_ALL_COMMITTED_GENERATIONS`:
the commit path wrote and validated each immutable generation but had no
post-commit rolling garbage-collection step. The existing V6 directory is not
modified by this task.

### Alias and retention policy

The following inode audit was exact, not inferred:

| Alias pair | Type | Inode |
| --- | --- | ---: |
| `checkpoint_last.pt` / `checkpoint_last.generation-00000177.pt` | hard link | 1,048,887 |
| `checkpoint_calibration_pass.pt` / `checkpoint_calibration_pass.generation-00000001.pt` | hard link | 1,048,694 |

The rolling owner accepts only `checkpoint_last`. Its policy is:

```text
commit N artifact atomically
-> validate artifact and write/fsync COMMITTED marker
-> preserve N and N-1
-> delete only older committed checkpoint_last generations
```

The marker remains the recovery authority. A generation newer than the marker
is preserved as an uncommitted tail. A GC unlink failure produces
`STORAGE_GC_WARNING` but never changes a valid transaction from
`COMMITTED/PASS` to failure. Pinned names, including
`checkpoint_calibration_pass`, `checkpoint_best_dev`, the 10K/25K/50K
milestones, and the final checkpoint, are rejected by the rolling GC owner.

For the current V6 read-only dry-run:

```text
KEEP=checkpoint_last.generation-00000176.pt, checkpoint_last.generation-00000177.pt
SAFE_DELETE_CANDIDATES=checkpoint_last.generation-00000001.pt through checkpoint_last.generation-00000175.pt
SAFE_DELETE_CANDIDATE_COUNT=175
ESTIMATED_RELEASE=7.958214 GiB physical
AUTO_DELETE_EXISTING_V6_GENERATIONS=NO
```

If the listed old V6 generations are later removed by an operator, the
estimated remaining V6 allocation is `0.402283 GiB`. This is an estimate; no
existing V6 generation was deleted here.

The dry-run command is:

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
cd /home/xm/XM/xm_ws/src/planning
python scripts/plan_awac_checkpoint_gc.py \
  --checkpoint-dir data/awac/awac_bc60k_formal_critic_calibration_v6 \
  --retention 2
```

It prints the committed generation, files to keep, uncommitted files to
preserve, safe-delete candidates, and estimated release. It never deletes.

## Replay storage audit

The current replay reports `size=5142`, `capacity=50000`, vector dimension
127, action dimension 105, and depth shape `[1,90,160]`. The full-capacity
arrays are preallocated with `open_memmap(mode="w+")`:

```text
depth, next_depth       uint8       [50000,1,90,160]
vector, next_vector     float32     [50000,127]
action_mask, next_mask  uint8       [50000,105]
action                  int16       [50000]
reward                  float32     [50000]
done, behavior_source   uint8       [50000]
```

The replay directory allocation is 154,517,504 bytes (`0.143906 GiB`) and
its apparent size is 1,501,709,322 bytes (`1.398576 GiB`). Therefore:

```text
REPLAY_PREALLOCATED=YES
REPLAY_SPARSE=YES
ESTIMATED_50K_REPLAY_PHYSICAL_GB=1.399316
ESTIMATED_50K_REPLAY_APPARENT_GB=1.398576
```

The estimate uses the measured allocated bytes per current transition and the
actual 50,000-row capacity; the apparent value is already capacity-sized.
Checkpoint payloads store replay identity, cursor, size, and committed state,
not replay arrays. The replay body remains one separate `<run>/replay`
directory. `env_workers` changes runtime process/log/temp usage but does not
multiply replay storage: `ENV_WORKER_COUNT_MULTIPLIES_REPLAY_STORAGE=NO`.

## Disk safety and historical runs

The filesystem containing Planning had 148,020,129,792 free bytes, or
`137.854488 GiB`. The formal conservative Standard AWAC run budget is:

| Budget component | GiB |
| --- | ---: |
| 50K replay | 1.399 |
| rolling checkpoints, pinned milestones, best/final artifacts | 0.500 |
| logs, runtime artifacts, and allocation margin | 3.101 |
| `ESTIMATED_STANDARD_AWAC_RUN_MAX_GB` | **5.000** |

With the required 20 GiB post-run reserve, the current guard is:

```text
HEADROOM_AFTER_ESTIMATED_RUN_GB=132.854488
MIN_FREE_DISK_AFTER_ESTIMATED_RUN_GB=20
DISK_HEADROOM_FOR_STANDARD_AWAC=PASS
```

The V1–V5 directories are absent from `data/awac`; there is no failed-run
payload to classify or delete. V6 remains required and
`FORMAL_V6_DELETE_ALLOWED=NO`. Existing BC60K, Dev100, and Final300 artifacts
are also not deletion targets in this certification.

## Verification

```text
compileall=PASS
focused AWAC/storage/config/resume tests=38 passed
CODE_ELIGIBLE_TEST_FAILURE_COUNT=0
```

The full suite was run in the requested conda/ROS environment:

```text
786 passed, 43 skipped, 8 failed
```

All 8 failures are pre-existing runtime tests that fail before their fake
runner can start, at socket creation with
`PermissionError: [Errno 1] Operation not permitted`. They are not caused by
the handoff, checkpoint, config-identity, or storage changes. No Unity,
Bridge, ROS master, AWAC online interaction, Actor update, or Final300 run was
started by this certification.

## Immutable boundaries and next phase

```text
BC_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
REWARD_CONTRACT_CHANGED=NO
MPL_CHANGED=NO
ALGORITHM_CHANGED=NO
COMMIT=NO
```

The next phase is:

```text
RL_PHASE_1_STANDARD_AWAC_WORKER_THROUGHPUT_BENCHMARK_V1
```

That phase is not executed by this report.

## Host V6 checkpoint GC dry-run

```text
========================================
HOST_V6_CHECKPOINT_GC_DRY_RUN
========================================
KEEP:
  checkpoint_last.generation-00000176.pt
  checkpoint_last.generation-00000177.pt
  checkpoint_last.pt (hard link to generation 177)
  checkpoint_calibration_pass.generation-00000001.pt
  checkpoint_calibration_pass.pt (hard link to pass generation)
  transaction manifests, reports, replay, and diagnostics

SAFE_DELETE_CANDIDATES:
  checkpoint_last.generation-00000001.pt ... checkpoint_last.generation-00000175.pt

ESTIMATED_RELEASE_GIB=7.958214
ACTUAL_DELETE_PERFORMED=NO
```
