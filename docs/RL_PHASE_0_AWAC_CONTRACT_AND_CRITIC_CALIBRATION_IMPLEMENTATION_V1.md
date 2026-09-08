# RL Phase 0 — AWAC Contract and Critic Calibration Implementation V1

Date: 2026-09-03  
Project root: `/home/xm/XM/xm_ws/src/planning`

## Gate status

```text
RL_PHASE_0_AWAC_CONTRACT_AND_CRITIC_CALIBRATION_IMPLEMENTATION_V1=FAIL
IMPLEMENTATION_COMPONENTS=PASS
RELEASE_GATE=BLOCKED_ENVIRONMENT
```

The Phase 0 implementation, focused tests, compile check, dev-set checks, and
bounded critic-calibration smoke passed. The unfiltered full pytest gate did
not pass in this execution environment: eight pre-existing socket-dependent
tests fail at `socket.socket(...)` with `PermissionError: [Errno 1]
Operation not permitted`, before their behavior assertions. The same suite
with those two environment-blocked test modules excluded is
`669 passed, 42 skipped, 0 failed`. No test was weakened or changed to hide
the restriction.

## Frozen ownership and contracts

```text
FORMAL_AWAC_OWNER=planning.awac
FORMAL_AWAC_ENTRYPOINT=scripts/train_awac.py
AWAC_PHASE=critic_calibration
OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
TASK_CONTRACT_SCHEMA_VERSION=2
TASK_CONTRACT_SHA256=2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df
FEATURE_CONTRACT_ID=depth_goal_state_prev_action
VECTOR_DIMENSION=127
ACTION_COUNT=105
REWARD_CONTRACT_SHA256=1eacab3788e9051e8c77d7c6aeafa2c50bd93dc09434b3adf4985850dd2b62d0
REWARD_SCALE=0.10
REWARD_SCALE_APPLIED_EXACTLY_ONCE=YES
AWAC_TRAINING_CONTRACT_SHA256=ac6edadadcdf5a7966c0499aacccb25054268dfd6e2a56348369d5a781509469
REPLAY_CONTRACT_SHA256=2b6198b0d295a51687542e475505f85c0d84180a8e529b0c3ea4f84d99c2244f
```

The replay schema retains the ten non-privileged fields:

```text
depth
vector
action_mask
action
reward
next_depth
next_vector
next_action_mask
done
behavior_source
```

`REPLAY_PRIVILEGED_FIELD_COUNT=0`. Global-map leakage checks for both actor
and critics are zero. The terminal contract records success, collision,
dead-end, timeout, and hard-altitude terminals; runtime-aborted attempts are
dropped/diagnosed rather than converted into training transitions.

## Dev set and final-test protection

```text
AWAC_DEV_SET_PATH=data/test/awac_dev_seed4026_100
AWAC_DEV_SET_MISSIONS=100
AWAC_DEV_SET_SEED=4026
AWAC_DEV_SET_VALIDATION=PASS
DEV_FINAL_DUPLICATE_MISSION_COUNT=0
AWAC_CALIBRATION_SPLIT_PATH=data/test/awac_calibration_split_seed4026.json
AWAC_CALIBRATION_SPLIT_VALIDATION=PASS
FROZEN_FINAL_TEST_300_MODIFIED=NO
```

The dev-set builder uses deterministic seed-and-mission identity selection and
rejects overlap with the BC-train and frozen final-test indexes. The
calibration split has explicit train/holdout episode IDs and rejects
transition leakage.

## BC-to-critic initialization

```text
BC60K_CHECKPOINT=data/teach/2026_6w/bc_training/checkpoint_best_soft.pt
BC60K_CHECKPOINT_SHA256=ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2
BC_ACTOR_STATE_SHA256=6ce3d0ff0ef20b6b8a382f0bef7205c86ae9d80d81f5dfd067c86927bdbf88df
BC_AWAC_INITIALIZATION_PARITY=PASS
CRITIC1_INITIALIZED_FROM_BC_ENCODER=YES
CRITIC2_INITIALIZED_FROM_BC_ENCODER=YES
CRITIC1_Q_HEAD_INDEPENDENT=YES
CRITIC2_Q_HEAD_INDEPENDENT=YES
```

The BC actor is frozen for calibration. The two critics share the BC feature
initialization but have independent Q heads and independent optimizer state.
No BC checkpoint, model, objective, or dataset was modified.

## Bounded calibration smoke

The smoke is synthetic/offline and is not formal AWAC training:

```text
SMOKE_PATH=/tmp/xm-awac-critic-calibration-v1-final-current
RUNTIME_MODE=synthetic_offline_smoke
CALIBRATION_REPLAY_TRANSITIONS=160
CALIBRATION_EPISODES=20
CALIBRATION_CRITIC_UPDATES=8
CALIBRATION_ACTOR_FROZEN=YES
CALIBRATION_ACTOR_UNCHANGED=YES
CALIBRATION_ACTOR_UPDATE_COUNT=0
CALIBRATION_CRITIC1_UPDATED=YES
CALIBRATION_CRITIC2_UPDATED=YES
CALIBRATION_ACTION_MISMATCH_COUNT=0
AWAC_REPLAY_AUDIT=PASS
CALIBRATION_GATE_STATE=PENDING
CALIBRATION_GATE_WOULD_OPEN=NO
```

The gate remains `PENDING` because the two-episode holdout slice does not
contain both success and failure classes; success-vs-failure ordering is
therefore not asserted from insufficient data. This is an explicit
fail-closed calibration result, not a fixed absolute-Q threshold.

Observed smoke metrics:

```text
CALIBRATION_HOLDOUT_TD_LOSS=113.28556823730469
CALIBRATION_Q_MEAN=-0.5933074355125427
CALIBRATION_Q_STD=0.2773297429084778
CALIBRATION_Q_P99=-0.3215242874622345
CALIBRATION_TWIN_DISAGREEMENT_MEAN=2.218523472547531
CALIBRATION_TWIN_DISAGREEMENT_P95=3.990880438685417
CALIBRATION_NORMALIZED_TWIN_DISAGREEMENT=0.23937382436193758
HOLDOUT_Q_RETURN_RANK_CORRELATION=0.0
SUCCESS_Q_MEDIAN=UNKNOWN_INSUFFICIENT_DATA
FAILURE_Q_MEDIAN=-0.5933074355125427
SUCCESS_MINUS_FAILURE_Q_MEDIAN=UNKNOWN_INSUFFICIENT_DATA
```

Checkpoint validation persists and checks the run identity, phase, BC actor
fingerprint, replay/training/reward contract identities, replay size and
metadata identity, critic states, freeze state, counters, and calibration
metrics. Empty actor-optimizer state is legal only for the frozen calibration
phase.

## Explicitly not implemented or executed

```text
CONFIDENCE_ALGORITHM_IMPLEMENTED=NO
ADAPTIVE_BC_KL_IMPLEMENTED=NO
PRIMITIVE_NEIGHBOR_ALGORITHM_IMPLEMENTED=NO
ACTOR_UPDATE_ENABLED=NO
FORMAL_AWAC_LONG_TRAINING_EXECUTED=NO
FINAL_300_EVALUATION_EXECUTED=NO
SAC_EXECUTED=NO
UNITY_STARTED=NO
BRIDGE_STARTED=NO
FORMAL_COLLECTION_EXECUTED=NO
```

No A*, MPL, BC mathematics, Unity, Bridge, task/observation contract,
trajectory semantics, or reward numerical semantics were changed.

## Verification

All commands involving Python were run after activating the `xm` conda
environment.

```text
compileall=PASS
Phase0 focused tests=17 passed
script contract test=15 passed
native depth/geometry focused tests=20 passed
latest unfiltered full pytest=685 passed, 42 skipped, 8 failed
code-eligible full pytest=669 passed, 42 skipped, 0 failed
```

The eight raw failures are limited to:

```text
tests/test_p0_m2_runner_startup.py
tests/test_p0_m2_single_worker_runner.py
```

Each is blocked by the managed environment's inability to create the test
socket. An unrestricted loopback-capable test environment is required to
close the user-requested `FULL_TEST_FAILURE_COUNT=0` gate.

## Modified files

The implementation changed only the Planning Python AWAC/dev-set seam,
focused tests, CMake script installation, and this documentation. BC,
Unity, Bridge, C++, and formal data artifacts were not modified. The complete
machine-readable inventory is in
[`rl_phase_0_awac_contract_and_critic_calibration_implementation_v1.json`](rl_phase_0_awac_contract_and_critic_calibration_implementation_v1.json).

## Next action

Rerun the unfiltered full pytest in an environment with loopback/socket
permission, then—only after that gate passes—consider Phase 1 standard AWAC
actor enablement and dev evaluation. Phase 1 was not started.
