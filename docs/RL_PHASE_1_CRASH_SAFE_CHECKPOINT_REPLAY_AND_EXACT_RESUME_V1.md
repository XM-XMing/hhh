# RL Phase 1 Crash-Safe Checkpoint, Replay, and Exact Resume V1

Status: `PASS` (code and CPU/synthetic certification only)  
Date: 2026-09-04  
Scope: `planning.awac` Critic Calibration persistence and recovery only.

## Scope and non-goals

This change closes the four recovery blockers reported by
`RL_PHASE_1_END_TO_END_AWAC_RUNTIME_GAP_AUDIT_V1`. It does not start a
managed runtime, Unity, Bridge, ROS master, Formal Calibration V6, Actor
training, or Final300 evaluation.

No Critic update rule, learning rate, reward contract, replay transition
schema, replay sampling distribution, calibration-gate threshold, Task
Contract, Observation Contract, MPL, Unity, or Bridge protocol changed.

`BLOCKER_4=Interrupt Cleanup`.

`BLOCKER_4_IN_SCOPE=YES` because a shutdown is recoverable only when the
replay and checkpoint belong to the same fully committed generation.

## Exact-resume state contract

The exact-resume owner is
`planning.awac.checkpoint.validate_exact_calibration_resume_payload`.
It requires all 23 logical state owners below. In-flight episode buffers are
deliberately excluded: they have not entered replay and restart from their
first uncommitted mission boundary.

1. Actor parameters
2. Critic 1 parameters
3. Critic 2 parameters
4. Target Critic 1 parameters
5. Target Critic 2 parameters
6. Actor optimizer state
7. Critic optimizer state
8. Learner counters and frozen-Actor state
9. Replay arrays, cursor, and metadata identity
10. Mission-source identity and ordered IDs
11. Train/holdout split identity
12. Managed-worker and runtime identity
13. Completed mission cursor and IDs
14. Environment and episode counters
15. Gate history and controller state
16. Completed holdout episode IDs
17. Runtime configuration and phase identity
18. Raw completed holdout transition records
19. Producer/replay-sampler `numpy.RandomState`
20. Python global `random` state
21. NumPy global RNG state
22. Torch CPU RNG state
23. Torch CUDA RNG state(s)

`CalibrationReplayProducer` is the producer and replay-sampler RNG owner.
It captures and restores its `numpy.RandomState`; no second sampler RNG was
introduced. The same checkpoint captures Python, NumPy global, Torch CPU, and
all available CUDA device RNG states. CUDA serialization plumbing is covered
with a device-independent fake CUDA owner; a real CUDA API test remains
capability-gated by the host.

Raw holdout records are preserved in their existing transition-diagnostic
form. Every nonempty persisted record must have `episode_id`, `mission_id`,
`episode_transition_index`, `holdout_record_index`, `terminal_reason`, and
the existing observation/action/reward transition fields. The ordered raw
record SHA256 is checked before restore, so gate windows cannot silently use
generation N+1 evidence with generation N model/replay state.

## Transaction protocol

`planning.awac.checkpoint.commit_calibration_checkpoint_transaction` owns the
single recoverable checkpoint protocol. `AWACReplayBuffer` remains the only
replay storage owner.

1. Persist a non-authoritative attempt diagnostic sidecar.
2. Flush replay arrays and `metadata.json`.
3. Capture replay size, position, total-added count, metadata SHA256,
   provenance identity, trainer/progress state, raw holdout evidence, and RNG
   state.
4. Validate the exact-resume payload.
5. Write and fsync an immutable generation file:
   `checkpoint_last.generation-N.pt` or
   `checkpoint_calibration_pass.generation-N.pt`.
6. Reload and validate that immutable file.
7. Atomically publish and fsync the corresponding
   `*.transaction.json` manifest last.
8. Activate the replay pending-generation journal and update the conventional
   `*.pt` alias only after the marker exists.

The manifest includes generation, checkpoint filename/SHA256, checkpoint
kind, save reason, full committed replay metadata, replay identity, and final
commit state. It is the only resume authority. A temporary file, a generation
file without a manifest, and an old direct V1--V5-style checkpoint are all
non-resumable.

After a committed generation, `AWACReplayBuffer` writes only a small
pending-generation journal. It permits rollback of an announced append-only
tail only, and rejects an overwrite of a committed ring-buffer row. Recovery
therefore cannot normalize arbitrary metadata corruption as an uncommitted
tail. Formal V6's configured `replay_capacity=50000` remains above its
`calibration_max_transitions=30000` cap; the runtime also rejects a capacity
that could require overwriting a committed generation before the next safe
checkpoint.

The same protocol is used for `checkpoint_last` and
`checkpoint_calibration_pass`; a PASS artifact is not published by a direct
Torch save path.

## Interrupt and diagnostics behavior

After `CalibrationReplayProducer.run()` leaves its `finally` cleanup, the
trainer handles `KeyboardInterrupt` and other runtime exceptions as follows:

- persist `checkpoint_last.runtime_failure_recovery.json` first;
- include failure summary, traceback, replay final-flush state, and runtime
  close state;
- if the gate is not terminal-diverged, attempt the same crash-safe
  `checkpoint_last` transaction at the last completed boundary;
- retain the original runtime exception after diagnostics/save handling;
- if the transaction cannot commit, persist `checkpoint_final_commit=FAIL`
  rather than claiming a resumable checkpoint.

Each transaction attempt also records
`checkpoint_save_reason`, generation, transaction state, final commit result,
and any interrupted-save traceback in
`*.transaction_attempt.json`. This sidecar is diagnostic evidence only; it
cannot select a resume generation.

Normal shutdown evidence is persisted as:

```text
REPLAY_FINAL_FLUSH=PASS/FAIL
CHECKPOINT_FINAL_COMMIT=PASS/FAIL/NOT_ATTEMPTED
RUNTIME_CLOSE=PASS/FAIL
```

## Fault-injection and parity evidence

The following source tests were run under `conda activate xm`, ROS Noetic,
and the workspace `devel/setup.bash`; no handwritten `PYTHONPATH` was used.

| Check | Evidence | Result |
| --- | --- | --- |
| Crash A | before replay flush | previous committed generation selected |
| Crash B | after replay flush, before checkpoint write | previous committed generation selected |
| Crash C | checkpoint temporary bytes written, before fsync | previous committed generation selected |
| Crash D | temporary checkpoint complete, before rename | previous committed generation selected |
| Crash E | checkpoint rename complete, before manifest | previous committed generation selected |
| Crash F | manifest committed | new generation selected |
| Replay mismatch | metadata beyond durable pending journal | fail closed |
| Last known good | generation N followed by incomplete N+1 | generation N restored |
| Raw holdout | ordered records and SHA256 round trip | exact |
| RNG | producer, Python, NumPy, Torch CPU | exact next sequence |
| CUDA RNG plumbing | serialized fake CUDA state round trip | pass; host CUDA runtime not required |
| Replay sample sequence | producer RNG is restored before next samples | exact |
| Mission cursor | two-worker completed ordering resumes at first uncommitted mission | exact |
| Gate state | gate history/controller reconstruction | exact |
| CPU learner | four updates vs. two + committed checkpoint + two | model, targets, optimizers, counters, and sample indices exact |
| PASS checkpoint | gate-PASS checkpoint uses commit marker protocol | pass |
| Interrupt lifecycle | fake runtime failure flushes/closes and diagnostics persist | pass |

Focused AWAC coverage:

```text
pytest -q tests/test_awac_*.py
135 passed, 1 skipped, 0 failed
```

The one skip is an existing opt-in real CUDA test. It was not added to hide a
failure.

Full suite:

```text
python -m compileall -q python scripts tests
PASS

pytest -q --tb=short
775 passed, 43 skipped, 8 failed
```

All eight failures occur before Planning behavior in sandbox-denied
`socket.socket()` calls and raise `PermissionError: [Errno 1] Operation not
permitted`. They are environment failures, not code-eligible failures.

```text
CODE_ELIGIBLE_TEST_PASS_COUNT=775
CODE_ELIGIBLE_TEST_SKIP_COUNT=43
CODE_ELIGIBLE_TEST_FAILURE_COUNT=0
FULL_TEST_ENVIRONMENT_FAILURE_COUNT=8
```

## Gap-readiness reassessment

The prior gap audit is retained as a historical pre-fix finding. Its recovery
statuses are superseded by this report:

```text
REPLAY_FLUSH_STATUS=PASS
REPLAY_CHECKPOINT_TRANSACTION_STATUS=PASS
CHECKPOINT_ATOMICITY_STATUS=PASS
RESUME_EXACTNESS_STATUS=PASS
RNG_RESTORE_STATUS=PASS
METRICS_PERSISTENCE_STATUS=PASS

BLOCKER_COUNT=0
UNKNOWN_CRITICAL_COUNT=0
V6_READY=YES
```

`V6_READY=YES` means the pre-run persistence/recovery gate is satisfied by
source and deterministic CPU/fault-injection evidence. It does not claim that
Formal Critic Calibration V6 has run, nor that Unity/Bridge/CUDA runtime
parity has been exercised in this Codex sandbox.

## Invariants

```text
CHECKPOINT_SCHEMA_CHANGED=YES (awac_checkpoint_schema_v2 -> awac_checkpoint_schema_v3)
CRITIC_ALGORITHM_CHANGED=NO
CRITIC_UPDATE_SCHEDULE_CHANGED=NO
CALIBRATION_GATE_CHANGED=NO
CALIBRATION_GATE_THRESHOLDS_CHANGED=NO
REPLAY_TRANSITION_SCHEMA_CHANGED=NO
REWARD_SEMANTICS_CHANGED=NO
ACTOR_UPDATE_EXECUTED=NO

RUNTIME_EXECUTED=NO
FORMAL_CALIBRATION_V6_EXECUTED=NO
UNITY_STARTED=NO
BRIDGE_STARTED=NO
BC_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
REWARD_CONTRACT_CHANGED=NO
MPL_CHANGED=NO
COMMIT=NO
```

Next action: `RUN_HOST_FORMAL_CRITIC_CALIBRATION_V6`.
