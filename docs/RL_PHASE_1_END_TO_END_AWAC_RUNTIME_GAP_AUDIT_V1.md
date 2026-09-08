# RL Phase 1 End-to-End AWAC Runtime Gap Audit V1

## Audit result

> Historical pre-fix snapshot. Its recovery and V6-readiness findings are
> superseded by
> [`RL_PHASE_1_CRASH_SAFE_CHECKPOINT_REPLAY_AND_EXACT_RESUME_V1.md`](RL_PHASE_1_CRASH_SAFE_CHECKPOINT_REPLAY_AND_EXACT_RESUME_V1.md).
> This retained report still documents the original four blocker observations;
> it must not be read as the current recovery status.

- Audit status: PASS
- Audit mode: integration-readiness audit only
- V6 readiness: NO
- Runtime executed by this audit: NO
- Actor update executed by this audit: NO
- Final300 evaluation executed by this audit: NO

This report answers whether the current AWAC online-calibration path is
structurally ready to begin Formal Critic Calibration V6. It does not approve
V6. The mission-to-replay path has real-runtime evidence, but crash-safe
checkpointing and exact resume do not yet meet the V6 gate.

## Scope and evidence rules

Only the following evidence types were used:

1. Current production source and current tests.
2. Historical migration, architecture, and Phase 1 reports.
3. Existing V2 through V5 calibration artifacts and the BC60K Dev100 artifact.
4. Focused and full test runs performed for this audit.

No Unity, Bridge, ROS master, managed runtime, Formal Critic Calibration,
Actor update, Dev100, or Final300 runtime was started. No pre-refactor source
tree was restored.

The old python/planning/rl tree is absent. Pre-refactor behavior evidence is
still available through the retained migration and ownership reports, so its
absence does not prevent this behavior-and-ownership audit.

## Frozen contracts and artifact identities

| Item | Value | Audit disposition |
| --- | --- | --- |
| BC60K checkpoint | data/teach/2026_6w/bc_training/checkpoint_best_soft.pt | Read-only |
| BC60K SHA256 | ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2 | Unchanged |
| Observation contract | reliable_exact_endpoint_snapshot | Unchanged |
| Task contract | Existing V2 task contract | Unchanged |
| Reward contract SHA256 | 1eacab3788e9051e8c77d7c6aeafa2c50bd93dc09434b3adf4985850dd2b62d0 | Unchanged |
| Formal training missions | data/teach/2026_6w/missions.csv | Read-only |
| Final300 | data/test/bc_closed_loop_eval_300 | Untouched and training-protected |

The retained BC60K Dev100 baseline is real-runtime evidence for the
reliable-exact environment path: 69 successes out of 100, 9 collisions, 22
dead ends, and no timeouts. It is not rerun by this audit.

## Pre-refactor behavior evidence

| Question | Finding |
| --- | --- |
| Pre-refactor source tree available | NO |
| Pre-refactor behavior/ownership evidence available | YES |
| Basis | Retained AWAC-only migration, managed-runtime lifecycle, replay-producer, checkpoint-identity, and Phase 1 reports |
| Old SAC/RL restoration required | NO |

The audit found no evidence that the current AWAC call pattern changes the
ParallelEnv protocol. The current chain remains ManagedRuntimePool followed by
ParallelEnvPool, with worker-local runtime identities and ordered request /
response handling. This is an evidence-limited statement: the deleted old
source cannot provide a fresh textual diff.

## Real runtime evidence already available

The inspected V2 through V5 calibration artifacts establish more than mock
coverage:

- managed ROS, Unity, and Bridge processes reached readiness for two workers;
- isolated worker runtime identities and ports were recorded;
- environment interaction persisted 95 reliable-exact transitions;
- replay metadata recorded 95 reliable-v4 rows and zero legacy rows;
- observation contract was reliable_exact_endpoint_snapshot;
- the phase was critic_calibration and the actor update count was zero;
- V5 had a five-episode run with a deterministic train/holdout split and a
  checkpoint replay size equal to the persisted replay size, 95.

The earlier V2 through V4 artifacts contain checkpoint replay size 66 versus
persisted replay size 95. That is retained historical evidence, not a claim
that the current normal checkpoint order still emits that mismatch. Current
normal checkpoint code explicitly flushes replay before creating replay
identity and checkpoint payload. The V5 95/95 artifact is compatible with
that order. It is not a fault-injection proof of a crash-safe transaction.

## Complete component evidence table

Classification values are intentionally limited to the approved vocabulary.
Risk is scoped to a fresh Formal Critic Calibration V6 run.

| Component | Current Owner | Status | Runtime Evidence | Unit Evidence | Risk | Required Before V6 |
| --- | --- | --- | --- | --- | --- | --- |
| Mission Loading | planning.awac.calibration_runtime.load_calibration_missions | RUNTIME_VERIFIED | V2-V5 record a formal mission-source identity and ordered mission cursor | Duplicate and invalid-row validation | INFO | No |
| Mission Dispatch | CalibrationReplayProducer.run | RUNTIME_VERIFIED | Two workers dispatched real episodes without duplicate replay identities | Ordering and failure seams covered | LOW | No |
| Managed Runtime | planning.awac.trainer.start_managed_calibration_runtime and ManagedRuntimePool | RUNTIME_VERIFIED | V2-V5 manifests show ROS, Unity, and Bridge readiness for two isolated workers | Lifecycle failure and close tests | MEDIUM | No |
| ParallelEnv | planning.runtime.parallel_env.ParallelEnvPool | RUNTIME_VERIFIED | V2-V5 obtained real reset and step responses | Ordering, timeout, and close tests | MEDIUM | No |
| Observation | CalibrationReplayProducer observation validators | RUNTIME_VERIFIED | Persisted replay is reliable-exact with zero legacy rows | Endpoint, depth, vector, and 105-mask contract tests | INFO | No |
| BC Action | CalibrationReplayProducer.bc_calibration_action through policy_evaluator.choose_action | UNIT_VERIFIED_ONLY | Real calibration stored actions, but exact evaluator-owner parity is not independently runtime-compared | Deterministic action-owner parity tests | MEDIUM | No |
| Terminal Resolution | unity_env.apply_post_action_dead_end and calibration terminal normalization | RUNTIME_VERIFIED | Real calibration terminal episodes were persisted | Zero next-mask terminal ordering tests | LOW | No |
| Reward | Environment response to replay transition construction | RUNTIME_VERIFIED | Runtime replay stored raw environment rewards with formal reward-scale metadata | Exact-once scaling path tests | LOW | No |
| Replay Append | planning.awac.replay.AWACReplayBuffer.append | RUNTIME_VERIFIED | 95 persisted reliable-exact replay rows | Shape, dtype, and transition validation tests | INFO | No |
| Replay Flush | AWACReplayBuffer.flush | IMPLEMENTED_NOT_VERIFIED | V5 inspected artifact is 95 checkpoint / 95 persisted replay | Flush and metadata writer tests | HIGH | Yes |
| Train/Holdout Split | calibration_runtime deterministic split owner and CalibrationReplayProducer | RUNTIME_VERIFIED | V5 retains five episodes with separate train and holdout identities | Overlap and identity validation tests | INFO | No |
| Critic Warmup | CalibrationReplayProducer._update_critics | RUNTIME_VERIFIED | V2-V5 reached 95 train replay rows and zero critic updates, as required below warmup | Threshold tests | INFO | No |
| Critic Sampling | AWACReplayBuffer.sample | UNIT_VERIFIED_ONLY | No historical run reached an update | Batch sampling tests | HIGH | Yes |
| Critic Update | AWACLearner.calibration_update | UNIT_VERIFIED_ONLY | No historical run reached an update | Critic-only update tests | HIGH | Yes |
| Target Update | AWACLearner.update | UNIT_VERIFIED_ONLY | No historical run reached an update | Twin target soft-update tests | HIGH | Yes |
| Device Transfer | move_awac_batch_to_device and resolve_torch_device | UNIT_VERIFIED_ONLY | No critic batch was executed in V2-V5 | CPU/CUDA canonical-device and holdout tests | HIGH | Yes |
| Holdout Evaluation | CalibrationReplayProducer holdout-window construction | SEMANTICS_CHANGED_NOT_VERIFIED | V5 exposed the old placeholder classification issue | Current finite-immature and mode-restore tests | HIGH | Yes |
| Gate Maturity | planning.awac.calibration.evaluate_calibration_gate | SEMANTICS_CHANGED_NOT_VERIFIED | V5 historical gate result is not valid proof for corrected semantics | PENDING versus FAIL_DIVERGED maturity tests | HIGH | Yes |
| Gate Window Accounting | CalibrationReplayProducer gate history and progress state | SEMANTICS_CHANGED_NOT_VERIFIED | No current-runtime run after the ordering repair | Window de-duplication and maturity tests | HIGH | Yes |
| Checkpoint Identity | planning.awac.checkpoint.build_awac_checkpoint_identity | SEMANTICS_CHANGED_AND_VERIFIED | V5 checkpoint identity validates against its current replay and source artifacts | Nineteen-field identity validation tests | INFO | No |
| Checkpoint Transaction | AWACTrainer.persist_checkpoint and checkpoint.save_torch_atomic | IMPLEMENTED_NOT_VERIFIED | Normal V5 95/95 alignment only | Save/validate path tests; no interruption transaction test | BLOCKER | Yes |
| Pass Checkpoint | AWACTrainer calibration completion path | UNIT_VERIFIED_ONLY | No calibration gate has passed | PASS-only checkpoint construction tests | MEDIUM | No |
| Resume | Trainer resume validators and CalibrationReplayProducer.restore_progress | MISSING | No successful exact-resume runtime artifact | Identity mismatch is fail-closed in tests | BLOCKER | Yes |
| RNG State | CalibrationReplayProducer RNG and framework RNG owners | MISSING | No checkpointed exact-RNG evidence exists | Static inspection establishes omission | BLOCKER | Yes |
| Failure Recovery | CalibrationReplayProducer and managed runtime start error handling | UNIT_VERIFIED_ONLY | No real Unity or Bridge failure recovery run is retained | Fake-worker failure and startup cleanup tests | HIGH | No |
| Interrupt Cleanup | CalibrationReplayProducer._safe_stop_and_close and trainer finalization | IMPLEMENTED_NOT_VERIFIED | No interrupted calibration artifact proves final checkpoint ordering | Cleanup path tests cover close behavior, not durable checkpoint recovery | BLOCKER | Yes |
| Metrics Persistence | Trainer metrics and checkpoint progress payload | IMPLEMENTED_NOT_VERIFIED | V5 lacks save reason, traceback, and durable runtime-failure diagnostics | Metrics serialization tests | MEDIUM | No |
| Actor Freeze | AWACLearner.calibration_update | RUNTIME_VERIFIED | V2-V5 checkpoints record zero actor updates and zero actor optimizer steps | Frozen-actor assertion tests | INFO | No |
| Phase Transition | planning.awac.phase1 | UNIT_VERIFIED_ONLY | No calibration PASS exists | PENDING/PASS/FAIL state-machine tests | MEDIUM | No |
| Final300 Protection | calibration mission loader and phase1 manifest validation | UNIT_VERIFIED_ONLY | Final300 was not read or changed | Final-test manifest rejection tests | INFO | No |
| AWAC Online Seam | BehaviorSource.AWAC_ONLINE | MISSING | No producer path consumes it | Enum and contract-only coverage | MEDIUM | No, future standard-AWAC work |
| Dev Milestone Seam | planning.awac.phase1 milestone callbacks | IMPLEMENTED_NOT_VERIFIED | No 10K, 25K, or 50K actor milestone has run | Callback plumbing tests | MEDIUM | No, post-calibration work |

## Mission, observation, and terminal-chain findings

### Mission loading and dispatch

The mission loader validates non-empty source rows, row task contracts,
duplicate mission identity, duplicate episode identity, and produces a mission
source identity containing source SHA256, row count, and ordered mission and
episode identity hashes. It retains source order rather than reordering by
worker completion.

V2-V5 provide real two-worker dispatch evidence and no duplicate persisted
replay identities. Worker-crash reassignment remains unit-only, so this is not
being used as a V6 crash-recovery guarantee.

### Reliable-exact observation construction

The producer validates reliable-exact metadata, depth, continuous vector,
previous action, and the 105-action mask. Runtime metadata with telemetry
lookup, missing snapshot, frame failure, sensor skew, missing runtime identity,
or invalid endpoint continuity is rejected before replay append. There is no
telemetry fallback path used for calibration replay.

### Frozen BC action and terminal behavior

The calibration action path delegates to the evaluator action owner with the
current reliable-exact observation, depth mask, previous action, normalizer,
resolved device, and deterministic temperature zero. Unit evidence confirms
the parity seam. Existing runtime artifacts prove actions were produced, but
do not independently compare every real-runtime action to a simultaneous
evaluator trace; the classification therefore remains UNIT_VERIFIED_ONLY.

Terminal aliases normalize altitude_violation to hard_altitude and max_steps to
timeout. Unknown terminal reasons are rejected. A zero next-action mask is
permitted only for a terminal transition; a nonterminal zero mask fails before
replay construction. The environment owner remains
planning.runtime.unity_env.apply_post_action_dead_end.

### Reward path

The transition stores the raw environment reward. The learner applies the
formal reward scale in its consumption path; no second scale is applied by
replay append. Existing tests cover exact-once scaling. Runtime artifacts prove
the reward path and formal scale metadata, while exact-once behavior is
unit-evidenced rather than independently runtime-instrumented.

## Replay, critic schedule, and target path

### Replay transaction

Current normal periodic checkpoint order is:

1. flush replay arrays and replay metadata;
2. construct replay identity from the flushed metadata;
3. construct the checkpoint snapshot;
4. write a temporary Torch checkpoint and replace the destination.

This removes the specific ordinary-order explanation for the historical V2-V4
66-versus-95 mismatch, and the inspected V5 checkpoint is 95 versus 95.
However, the sequence is not a durable cross-artifact transaction:

- replay arrays and metadata have no transaction record jointly committed with
  the checkpoint;
- the checkpoint writer uses temporary write plus rename but does not fsync
  the checkpoint file and parent directory;
- no post-save reload/identity validation occurs on the periodic save path;
- there is no crash recovery or rollback record for a failure between replay
  flush and checkpoint replacement;
- a KeyboardInterrupt or runtime failure does not persist a current last
  checkpoint before cleanup.

Therefore normal alignment is implemented, but fault-safe replay/checkpoint
atomicity is not verified and is a V6 blocker.

### Warmup and update schedule

The owner is CalibrationReplayProducer._update_critics. Under the current
formal defaults:

| Parameter | Value |
| --- | --- |
| batch size | 128 |
| learning starts | 5000 committed train replay transitions |
| updates per committed train transition | 0.50 |
| target critic update timing | After each critic update |

The target update count is:

    floor(max(0, replay_size - learning_starts + 1) * updates_per_step)

Consequently, the first actual critic update requires 5001 committed training
replay rows under the current defaults. The V2-V5 95-row runtime artifacts
correctly show zero critic updates. They cannot provide runtime evidence for
sampling, critic updates, target updates, or device transfer.

Twin critics are separate Q heads with separate optimizer parameter groups.
Each calibration update computes both critic losses, steps the critic
optimizer, and soft-updates target critic 1 and target critic 2 with tau.
These mechanisms are unit-verified, not current-runtime verified.

## Holdout, gate, and phase findings

The deterministic split is episode-based: the split owner derives train versus
holdout membership from the calibration seed and episode identity, then retains
the source order. Holdout episode transitions are not appended to critic
replay. The inspected V5 artifact and split validation tests provide evidence
that no train/holdout leakage was observed.

The V5 additional audit found that immature holdout placeholders were
incorrectly treated as nonfinite metrics. Current semantics separate:

- unavailable or not-ready holdout placeholder: PENDING, excluded from
  canonical finite metric validation;
- finite but immature holdout window: PENDING;
- NaN or Inf diagnostics: immediate FAIL_DIVERGED.

This is a behavior correction. It is covered by current unit tests but has not
been exercised by a fresh formal runtime after the correction. Holdout window,
gate maturity, and gate-window accounting remain
SEMANTICS_CHANGED_NOT_VERIFIED and high risk before V6.

The phase owner permits the standard-AWAC actor phase only after a calibration
PASS and a valid pass checkpoint. This is unit-verified. The AWAC_ONLINE
behavior-source enum exists but no online producer path uses it; that is
missing future work, not a V6 calibration prerequisite. Milestone hooks for
10K, 25K, and 50K are plumbing only and are outside V6.

## Checkpoint and exact-resume audit

### Current checkpoint identity

The checkpoint identity builder uses the shared nineteen-field identity:
contracts, mission source, topology, runtime identity, split identity, replay
identity, and calibration configuration are validated fail-closed on resume.
V5 gives real artifact evidence that the current identity contract validates
for its replay and mission source. It does not prove exact continuation.

### Required versus restored state

The counts below group model and optimizer payloads by logical state owner so
that a resume audit is actionable rather than just a list of serialized keys.
In-flight, uncommitted episode buffers are deliberately not required: only
episode-complete accepted training transitions may become durable replay.

| Logical state required for exact resume | Current state |
| --- | --- |
| Actor parameters | Restored |
| Critic 1 parameters | Restored |
| Critic 2 parameters | Restored |
| Target critic 1 parameters | Restored |
| Target critic 2 parameters | Restored |
| Actor optimizer state | Restored |
| Critic optimizer state | Restored |
| Learner counters and frozen-actor state | Restored |
| Replay arrays, cursor, and metadata identity | Restored |
| Mission-source identity and ordered IDs | Restored |
| Train/holdout split identity | Restored |
| Managed worker and runtime identity | Restored |
| Completed mission cursor and completed IDs | Restored |
| Environment and episode counters | Restored |
| Gate history and gate controller state | Restored |
| Completed holdout episode IDs | Restored |
| Runtime configuration and phase identity | Restored |
| Raw completed holdout transition records | Missing |
| Producer RandomState used by replay sampling | Missing |
| Python global random state | Missing |
| NumPy global RNG state | Missing |
| Torch CPU RNG state | Missing |
| Torch CUDA RNG state | Missing |

Required logical states: 23. Restored logical states: 17. Missing logical
states: 6.

Completed holdout IDs without the corresponding holdout records cannot rebuild
the same holdout window after resume. A fresh producer RNG is initialized from
the seed on resume, but it is not advanced to the saved state. Python, NumPy,
Torch CPU, and Torch CUDA RNG states are not checkpointed. As a result,
replay-sample order, dropout or other framework stochasticity, and gate-window
content cannot be exact after a stop and resume.

The current resume validators correctly fail closed on incompatible replay,
mission, contract, split, topology, or static runtime identity. Fail-closed
identity validation is valuable, but it is not equivalent to exact resume.
RESUME_EXACTNESS_STATUS is therefore FAIL.

## Failure, interrupt, cleanup, and metrics findings

ManagedRuntimePool owns process groups for the current run and closes only
owned processes. No global pkill mechanism was identified. ParallelEnvPool
close behavior and managed-start failure cleanup have focused unit evidence.

Normal completion performs replay flush and resource close. On
KeyboardInterrupt, cleanup runs from finalization, but the path does not save a
fresh last checkpoint first. The producer cleanup helper catches Exception, not
BaseException, and the trainer does not provide a durable interrupt checkpoint
transaction. This makes graceful process cleanup different from a resumable
controlled stop.

The following diagnostics are not durably sufficient for postmortem recovery:

- checkpoint save reason;
- interrupted-save phase or traceback;
- durable runtime failure count and worker-failure summary;
- durable confirmation of final replay flush and close outcome;
- raw holdout records needed to reconstruct the gate window.

These gaps do not alter trajectory, Teacher, BC, Unity, Bridge, task,
observation, reward, or MPL semantics. They are durability and diagnosis gaps.

## Risks and V6 decision

Component-level risk totals:

| Risk | Count |
| --- | --- |
| BLOCKER | 4 |
| HIGH | 9 |
| MEDIUM | 8 |
| LOW | 3 |
| INFO | 8 |

Status totals:

| Status | Count |
| --- | --- |
| RUNTIME_VERIFIED | 11 |
| UNIT_VERIFIED_ONLY | 9 |
| IMPLEMENTED_NOT_VERIFIED | 5 |
| SEMANTICS_CHANGED_AND_VERIFIED | 1 |
| SEMANTICS_CHANGED_NOT_VERIFIED | 3 |
| MISSING | 3 |
| UNKNOWN | 0 |
| NOT_APPLICABLE | 0 |

The combined semantics-changed count is 4. Unknown critical count is 0:
the critical gaps are known rather than unknown.

### V6 blockers

1. Checkpoint/replay transaction and interrupted-stop path are not a
   crash-safe, recoverable transaction. Periodic normal ordering exists, but
   interruption can leave replay and last checkpoint without a recoverable
   joint commit.
2. Raw holdout records are absent from resume state. Resuming after holdout
   activity changes gate-window content and accounting.
3. Exact RNG state is absent for producer replay sampling, Python, NumPy,
   Torch CPU, and Torch CUDA. Exact continuation cannot be claimed.

The component-level blocker count is four because Resume and Interrupt Cleanup
are independently blocked components. The priority list groups their shared
checkpoint-transaction defect under blocker 1.

V6_READY is NO. The next action is:

    FIX_AWAC_RUNTIME_GAP_BLOCKERS

This is intentionally not a request to run V6, train an actor, add confidence,
adaptive BC KL, primitive-neighbor behavior, or change any frozen runtime
contract.

## Test evidence

All Python commands were run after activating conda environment xm and sourcing
the ROS and workspace devel setup.

| Check | Result | Interpretation |
| --- | --- | --- |
| python -m compileall -q python scripts tests | PASS | Current Python source compiled |
| Focused AWAC, managed-runtime, and ParallelEnv tests | 136 passed, 1 skipped, 0 failed | Code-eligible focused evidence passed; one CUDA opt-in test remained skipped |
| Full pytest | 758 passed, 43 skipped, 8 failed | The eight failures are sandbox socket PermissionError failures before product code behavior |
| Full code-eligible result | 758 passed, 43 skipped, 0 failed | No code-eligible test failure |

The eight full-suite environment failures all occur while trying to create a
socket and raise PermissionError: Errno 1, Operation not permitted. They are
not classified as source-code regressions and were not hidden by adding skips.

## Audit invariants

- BC changed: NO
- Unity changed: NO
- Bridge changed: NO
- Task contract changed: NO
- Observation contract changed: NO
- Reward contract changed: NO
- MPL changed: NO
- Commit: NO
- Confidence algorithm implemented: NO
- Adaptive BC KL implemented: NO
- Primitive-neighbor algorithm implemented: NO
