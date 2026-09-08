# RL Phase 1 — Standard AWAC Actor Enable and Dev Evaluation V1

Date: 2026-09-04  
Project root: `/home/xm/XM/xm_ws/src/planning`

## Result

```text
RL_PHASE_1_STANDARD_AWAC_ACTOR_ENABLE_AND_DEV_EVALUATION_V1=FAIL
PHASE1_EXECUTED=NO
RELEASE_GATE=BLOCKED_PRECONDITION
COMMIT=NO
```

Phase 1 was stopped before any Actor update, Unity/Bridge runtime, replay
mutation, formal AWAC training, Dev100 evaluation, Final300 evaluation, or
SAC execution. This is a fail-closed preflight result.

## Blocking evidence

1. The current Phase 0 report
   (`docs/rl_phase_0_awac_contract_and_critic_calibration_implementation_v1.json`)
   records `status=FAIL_GATE_ENVIRONMENT_BLOCKED`; its calibration smoke gate
   is `PENDING`, not `PASS`. No `checkpoint_calibration_pass.pt` exists under
   `data/`. The synthetic smoke's `actor_unchanged=true` and eight critic
   updates are not a formal calibration gate pass.
2. The canonical `train_awac.py` default is `--actor-depth-lr 0.0`
   (`python/planning/awac/trainer.py`). Phase 1 requires a positive current
   Actor depth learning rate. No separate AWAC config currently supplies one;
   this value was not changed or tuned in this run.
3. The current non-smoke trainer has no Phase 1 runtime loop: it samples one
   replay batch and performs one calibration or AWAC update before writing a
   checkpoint. It does not implement the required gate transition, bounded
   online collection, milestone Dev100 evaluation, collapse stop, or best
   checkpoint selection.
4. `data/test/awac_dev_seed4026_100` contains the Dev mission index and
   manifest, but no completed BC Dev100 evaluation summary. Therefore the
   required BC Dev baseline is `UNKNOWN`, and no post-Actor comparison is
   valid.
5. Verification in the current managed environment:

```text
compileall=PASS
AWAC_FOCUSED_TESTS=52 passed, 0 failed
FULL_PYTEST=697 passed, 42 skipped, 8 failed
```

All eight full-suite failures are the existing socket-permission failures in
`tests/test_p0_m2_runner_startup.py` and
`tests/test_p0_m2_single_worker_runner.py`, occurring at
`socket.socket(...)` with `PermissionError: [Errno 1] Operation not
permitted`. No test was weakened or skipped to hide them.

## Frozen inputs observed

```text
FORMAL_AWAC_OWNER=planning.awac
FORMAL_AWAC_ENTRYPOINT=scripts/train_awac.py
BC60K_CHECKPOINT=data/teach/2026_6w/bc_training/checkpoint_best_soft.pt
BC60K_CHECKPOINT_SHA256=ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2
TASK_CONTRACT_SHA256=2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df
OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
ACTION_COUNT=105
REWARD_CONTRACT_SHA256=1eacab3788e9051e8c77d7c6aeafa2c50bd93dc09434b3adf4985850dd2b62d0
PHASE0_TRAINING_CONTRACT_SHA256=ac6edadadcdf5a7966c0499aacccb25054268dfd6e2a56348369d5a781509469
REPLAY_CONTRACT_SHA256=2b6198b0d295a51687542e475505f85c0d84180a8e529b0c3ea4f84d99c2244f
ACTOR_DEPTH_LR=0.0
CRITIC_DEPTH_LR=1.0e-5
```

The frozen BC checkpoint, AWAC Dev mission artifact, Final300 artifact,
MPL, native libraries, Unity, Bridge, Task Contract, Observation Contract,
Reward Contract, and BC data were not modified.

## Phase 1 output record

```text
BC_DEV_SUCCESS_COUNT=UNKNOWN_NOT_EXECUTED
BC_DEV_SUCCESS_RATE=UNKNOWN_NOT_EXECUTED
BC_DEV_COLLISION=UNKNOWN_NOT_EXECUTED
BC_DEV_DEAD_END=UNKNOWN_NOT_EXECUTED
BC_DEV_TIMEOUT=UNKNOWN_NOT_EXECUTED
BC_DEV_HARD_ALTITUDE=UNKNOWN_NOT_EXECUTED
BC_DEV_MEAN_STEPS=UNKNOWN_NOT_EXECUTED
BC_DEV_MEAN_RETURN=UNKNOWN_NOT_EXECUTED

CALIBRATION_GATE_STATE=PENDING
CALIBRATION_GATE_PASS_ENV_STEP=NOT_RECORDED
CALIBRATION_GATE_PASS_REPLAY_SIZE=NOT_RECORDED
CALIBRATION_GATE_PASS_CRITIC_UPDATES=NOT_RECORDED
CALIBRATION_ACTOR_UNCHANGED=YES_FOR_PHASE0_SYNTHETIC_SMOKE_ONLY

ACTOR_UPDATE_ENABLED=NO
ACTOR_ENABLE_ENV_STEP=NOT_EXECUTED
ACTOR_UPDATE_COUNT=0
CRITIC1_UPDATE_COUNT=0
CRITIC2_UPDATE_COUNT=0

REPLAY_TOTAL_TRANSITIONS=NOT_EXECUTED
REPLAY_BC_CALIBRATION_COUNT=NOT_EXECUTED
REPLAY_AWAC_ONLINE_COUNT=NOT_EXECUTED

AWAC_WEIGHT_MEAN=UNKNOWN_NOT_EXECUTED
AWAC_WEIGHT_P95=UNKNOWN_NOT_EXECUTED
AWAC_WEIGHT_MAX=UNKNOWN_NOT_EXECUTED
AWAC_WEIGHT_CLIP_FRACTION=UNKNOWN_NOT_EXECUTED
BC_KL_MODE=FIXED
BC_KL_WEIGHT=0.05
BC_AWAC_KL_MEAN=UNKNOWN_NOT_EXECUTED
BC_AWAC_ACTION_DISAGREEMENT_RATE=UNKNOWN_NOT_EXECUTED

BEST_DEV_CHECKPOINT=NOT_EXECUTED
BEST_DEV_CHECKPOINT_SHA=NOT_EXECUTED
BEST_DEV_SUCCESS_RATE=UNKNOWN_NOT_EXECUTED
STANDARD_AWAC_CLASSIFICATION=NOT_EXECUTED
CATASTROPHIC_POLICY_COLLAPSE=NOT_EXECUTED
CRITIC_DIVERGED=NOT_EXECUTED
AWAC_REPLAY_AUDIT=NOT_EXECUTED
RELIABLE_EXACT_QUALITY=NOT_EXECUTED
```

Static policy-input inspection remains clean:

```text
ACTOR_GLOBAL_MAP_LEAK_COUNT=0
CRITIC_GLOBAL_MAP_LEAK_COUNT=0
REPLAY_PRIVILEGED_FIELD_COUNT=0
CONFIDENCE_ALGORITHM_IMPLEMENTED=NO
ADAPTIVE_BC_KL_IMPLEMENTED=NO
PRIMITIVE_NEIGHBOR_ALGORITHM_IMPLEMENTED=NO
```

The frozen-final and protected-component flags remain:

```text
FINAL_300_EVALUATION_EXECUTED=NO
FROZEN_FINAL_TEST_300_MODIFIED=NO
BC_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
REWARD_CONTRACT_CHANGED=NO
```

## Verification and next action

```text
FULL_TEST_PASS_COUNT=697
FULL_TEST_SKIP_COUNT=42
FULL_TEST_FAILURE_COUNT=8
```

No Phase 1 tests were added because the required implementation seams are not
present and the hard preconditions fail. The next action is:

```text
NEXT_ACTION=RESOLVE_FULL_TEST_SOCKET_PERMISSION; PRODUCE_FORMAL_PHASE0_CALIBRATION_PASS_CHECKPOINT; SET_CANONICAL_ACTOR_DEPTH_LR_GT_ZERO; THEN_RESTART_PHASE1_PREFLIGHT
```

Phase 2 was not entered.
