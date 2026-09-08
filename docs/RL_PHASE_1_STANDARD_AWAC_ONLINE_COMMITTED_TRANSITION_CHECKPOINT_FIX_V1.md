# RL Phase 1 Standard AWAC Online Committed-Transition Checkpoint Fix V1

Status: PASS (implementation and offline verification only)

## Scope

This fix closes the Standard AWAC online checkpoint/finalization seam exposed
by `standard_awac_online_w2_v3`. It does not change learner mathematics,
replay transition layout, reward semantics, or runtime contracts.

The counter owners remain distinct:

- `online_environment_steps`: `StandardAWACOnlineRunner._environment_step_count`,
  incremented for each executed environment step.
- `online_transitions_committed`:
  `StandardAWACOnlineRunner._online_transitions_committed`, incremented only
  after a validated AWAC_ONLINE episode is appended to Replay.

The legal v3 relationship remains `3964` environment steps and `3941`
committed AWAC_ONLINE rows.

## Fixed seams

- `checkpoint_snapshot()` now exposes the canonical committed counter at the
  top level and in `progress` and exact-resume state.
- The Standard checkpoint builder requires the explicit counter and checks it
  against `progress`; it does not default a missing value to zero.
- Standard checkpoint payloads carry explicit `online_transitions_committed`
  and retain `online_transition_count` as the established compatibility alias.
- The Standard checkpoint transaction cross-checks the canonical counter with
  Replay's committed `behavior_source=AWAC_ONLINE` rows and append accounting.
  Replay is an independent fail-closed audit, not a counter owner.
- Exact-resume validation requires mission progress and environment counters
  to agree on the committed count.
- Standard summaries explicitly carry `awac_online_rows`,
  `online_transitions_committed`, and `online_env_steps`.

Missing or mismatched committed-counter fields are rejected. Validator
strictness was not relaxed.

## Synthetic regression coverage

- `env_steps=100`, `committed=97` checkpoint round-trip.
- Missing `online_transitions_committed` rejected by builder and validator.
- Checkpoint committed `97` versus Replay AWAC_ONLINE `96` rejected.
- Exact resume restores `online_transitions_committed=97`.
- Continuing the runner advances `97 -> 100` exactly.

Commands used, with the `xm` conda environment active:

```bash
python -m compileall -q python/planning
python -m pytest -q tests/test_awac_standard_online_runtime.py
python -m pytest -q
```

Results:

```text
focused Standard suite: 24 passed
full suite: 819 passed, 43 skipped, 0 failed
```

## v3 disposition and next action

`standard_awac_online_w2_v3` remains historical and is not patched:

- `SMOKE_FAILED_FINALIZATION`
- `NOT_FORMAL_MODEL`
- `NOT_RESUMABLE_AS_FORMAL_SOURCE`

No real runtime was executed by this fix. The next action is a fresh bounded
W2 smoke in `data/awac/smoke/standard_awac_online_w2_v4`, sourced from the
existing V6 calibration-pass checkpoint and replay; v3 must not be resumed.

```text
ACTOR_LR_CHANGED=NO
CRITIC_LR_CHANGED=NO
AWAC_MATH_CHANGED=NO
REPLAY_TRANSITION_SCHEMA_CHANGED=NO
REWARD_SEMANTICS_CHANGED=NO
CHECKPOINT_VALIDATOR_STRICTNESS_CHANGED=NO
RUNTIME_EXECUTED=NO
ACTOR_UPDATE_EXECUTED=NO
COMMIT=NO
```
