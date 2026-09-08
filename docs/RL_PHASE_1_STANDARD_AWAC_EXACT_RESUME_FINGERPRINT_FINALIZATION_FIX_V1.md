# RL Phase 1 Standard AWAC Exact-Resume Fingerprint Finalization Fix V1

Date: 2026-09-05

## Result

```text
RL_PHASE_1_STANDARD_AWAC_EXACT_RESUME_FINGERPRINT_FINALIZATION_FIX_V1=PASS
RUNTIME_EXECUTED=NO
UNITY_STARTED=NO
BRIDGE_STARTED=NO
ACTOR_UPDATE_EXECUTED=NO
COMMIT=NO
```

The repair is limited to Standard AWAC checkpoint exact-resume
canonicalization, finalization diagnostics, and the associated regression
tests. No AWAC update rule, replay schema, task/observation contract, Unity,
or Bridge code was changed.

## V2 evidence and root cause

The preserved v2 traceback is in
`data/awac/smoke/standard_awac_online_w2_v2/runtime_failure.json`:

```text
_run_standard_awac_online
  -> StandardAWACOnlineRunner.checkpoint_snapshot
  -> build_standard_online_exact_resume_state
  -> _resume_state_sha256
  -> _hash_resume_value
  -> KeyError: '0'
```

The failing logical payload path is
`learner_state.actor_optimizer_state_dict.state[0]`. PyTorch optimizer state
uses integer parameter IDs. The old all-mapping path sorted `str(key)` values
and then indexed the original mapping with the string, turning integer `0`
into the invalid lookup `'0'`.

The fix iterates original `(key, value)` pairs, encodes key type and value,
sorts the encoded key bytes, and never stringifies a key for lookup. Existing
all-string calibration mappings retain their historical byte encoding so the
V6 fingerprint remains compatible. Unsupported key types still fail closed.

```text
FINGERPRINT_FAILURE_OWNER=planning.awac.checkpoint
FINGERPRINT_FAILURE_FUNCTION=_hash_resume_value
FINGERPRINT_FAILURE_PAYLOAD_PATH=learner_state.actor_optimizer_state_dict.state[0]
FINGERPRINT_FAILURE_KEY_TYPE=int (erroneous lookup was str '0')
EXACT_RESUME_FINGERPRINT_OWNER_COUNT=1
```

## Canonicalization and resume inventory

Focused tests cover integer, string, bool, nested mapping, list, tuple, None,
bytes, NumPy scalar/array, and Torch tensor values. Integer/string key
collisions are distinguished, insertion order is deterministic, and
unsupported mapping keys raise a typed error.

```text
INTEGER_KEY_MAPPING_SUPPORTED=YES
MIXED_TYPE_KEY_COLLISION_PROTECTION=PASS
NESTED_MAPPING_FINGERPRINT=PASS
DETERMINISTIC_MAPPING_ORDER=PASS
STANDARD_ONLINE_RESUME_MISSING_STATE=NONE
```

The Standard exact-resume owner list contains 14 entries: learner parameters
and optimizer state, learner counters, replay identity, producer replay/sampler
RNG state, Python RNG state, NumPy RNG state, Torch CPU RNG state, Torch CUDA
RNG state, mission source and ordered IDs, mission cursor and completed IDs,
environment/online counters, online schedule counters, Phase-1 state, and
runtime identity.

## Finalization and failure evidence

The audited finalization order remains:

```text
runner.run()
-> replay flush + exact-resume snapshot
-> checkpoint transaction commit
-> committed checkpoint load/validation
-> canonical summary write
-> lifecycle cleanup
```

The Standard failure payload now preserves traceback, checkpoint save reason,
transaction state, checkpoint paths, committed replay scalars, cleanup owner,
and Actor before/after fingerprints. The pre-runtime manifest persists the
Actor-before fingerprint before managed runtime startup. A failed final
checkpoint does not create a successful `summary.json`.

```text
STANDARD_ONLINE_CHECKPOINT_ROUNDTRIP=PASS
FINGERPRINT_ROUNDTRIP=PASS
FINAL_CHECKPOINT_TRANSACTION=PASS
SUMMARY_AFTER_COMMITTED_CHECKPOINT=PASS
FAILURE_DIAGNOSTICS_PERSISTENCE=PASS
ACTOR_SHA_BEFORE_PERSISTENCE=PASS
ACTOR_SHA_AFTER_PERSISTENCE=PASS
ROLLING_CHECKPOINT_RETENTION=2
RETENTION_REGRESSION=PASS
```

## 3964/3965 accounting

The v2 runner stopped at 3964/3965 because the next two-worker in-flight wave
would have exceeded the phase-local environment-step budget. This is bounded
in-flight worker-wave semantics, not an off-by-one defect. Environment steps
are the budget owner; valid committed replay transitions remain a separate
counter.

```text
ONLINE_ENV_STEPS_V2=3964
ONLINE_ENV_STEPS_BUDGET_V2=3965
ONLINE_BUDGET_SHORTFALL_REASON=INFLIGHT_WORKER_WAVE_WOULD_EXCEED_BUDGET
ONLINE_BUDGET_OFF_BY_ONE=NO
ONLINE_BUDGET_CONTRACT=PASS
```

The budget tests cover exact single-worker budgets 1, 2, and 5 and the
two-worker bounded shortfall case.

## Regression and immutable input evidence

The V6 Calibration PASS checkpoint remains unchanged:

```text
V6_CHECKPOINT_SHA256=bb9433d8d9fb2fde2e2148e84a4058c2b328b9d676608d38c00962f821c3d8de
CALIBRATION_CHECKPOINT_REGRESSION=PASS
CALIBRATION_GATE=PASS
CALIBRATION_ACTOR_UPDATE_COUNT=0
CALIBRATION_CRITIC_UPDATE_COUNT=71
```

The previous v2 run remains diagnostic only and is not resumable or reused:

```text
V2_SMOKE_REPLAY_ROWS=9083
V2_SMOKE_AWAC_ONLINE_ROWS=3941
V2_SMOKE_ACTOR_UPDATES=10
V2_SMOKE_CRITIC_UPDATES=2042
V2_SMOKE_RESUMABLE=NO
NEXT_SMOKE_DIR=data/awac/smoke/standard_awac_online_w2_v3
```

```text
REPLAY_TRANSITION_SCHEMA_CHANGED=NO
BEHAVIOR_SOURCE_SEMANTICS_CHANGED=NO
REWARD_SEMANTICS_CHANGED=NO
CRITIC_ALGORITHM_CHANGED=NO
ACTOR_ALGORITHM_CHANGED=NO
CALIBRATION_GATE_CHANGED=NO
ACTOR_LR_CHANGED=NO
CRITIC_LR_CHANGED=NO
AWAC_TEMPERATURE_CHANGED=NO
AWAC_WEIGHT_MAX_CHANGED=NO
BC_KL_WEIGHT_CHANGED=NO
CQL_CHANGED=NO
GAMMA_CHANGED=NO
TAU_CHANGED=NO
BC_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
REWARD_CONTRACT_CHANGED=NO
MPL_CHANGED=NO
```

## Tests

All Python commands were run after activating the `xm` conda environment.

```text
FOCUSED_TESTS=59 passed, 1 skipped, 0 failed
STANDARD_ONLINE_RUNTIME_TESTS=20 passed, 0 skipped, 0 failed
COMPILEALL=PASS
FULL_TESTS=815 passed, 43 skipped, 0 failed
```

## Next action

No runtime was retried in this phase. The next host-only action is the new
bounded W2 smoke from the original V6 Calibration PASS checkpoint, using the
commands printed in the handoff response and output directory
`data/awac/smoke/standard_awac_online_w2_v3`.
