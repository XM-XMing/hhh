# RL Phase 1 Runtime Certification: Dev100 and Formal Critic Calibration V1

Status: `FAIL_BLOCKED_ENVIRONMENT`

Root: `/home/xm/XM/xm_ws/src/planning`  
Python environment: `xm`

## Executive result

The first required Runtime Capability Gate failed before any Unity, Bridge,
Dev100, or formal calibration runtime was started:

```text
RUNTIME_CAPABILITY_GATE=FAIL_SOCKET_ENVIRONMENT
socket_error=PermissionError(1, Operation not permitted)
```

All runtime work was stopped fail-closed. The failure is an environment
capability failure, not a calibration result and not evidence of critic
divergence. No socket workaround, test modification, mock runtime, Unity
launch, Bridge launch, Actor update, or formal calibration run was used.

The bounded synthetic critic-only smoke remains available as implementation
evidence. It is not a BC Dev100 baseline and it is not formal calibration
certification.

## Frozen identity

| Item | Value |
|---|---|
| BC60K checkpoint | `data/teach/2026_6w/bc_training/checkpoint_best_soft.pt` |
| BC60K checkpoint SHA256 | `ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2` |
| Task contract SHA256 | `2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df` |
| Observation contract | `reliable_exact_endpoint_snapshot` |
| Reward contract SHA256 | `1eacab3788e9051e8c77d7c6aeafa2c50bd93dc09434b3adf4985850dd2b62d0` |
| Replay contract SHA256 | `2b6198b0d295a51687542e475505f85c0d84180a8e529b0c3ea4f84d99c2244f` |
| Actions / max steps | `105 / 45` |
| Execution mode | `continuous` |
| Deployment safety mask | `depth` |
| Phase 1 contract SHA256 | `6a3ae6e650db8e60b693396fbc3ac15bf6f6030c86dcd650dc1b6534a63c4233` |
| Calibration split SHA256 | `a0dc1ae4fcca131b8872bf813c40c3fdab6e0b3e68d9681e7474337647c06334` |
| Dev manifest SHA256 | `831c7904c594065fab5d04fd836021f04eb1447ac93b9c693daca42b618949b8` |
| Final300 manifest SHA256 | `eb61056faf7e1a6dd16509768b29f57f8e79d9e6db56291626eb7affec4c69f4` |

`BC_AWAC_INITIALIZATION_PARITY=PASS` is retained as the existing static
checkpoint/contract parity evidence. It does not replace the missing runtime
Dev100 baseline.

## Runtime capability gate

The command was executed after activating `conda` environment `xm`; Python
resolved to `/home/xm/anaconda3/envs/xm/bin/python`. A loopback bind attempt
failed with `PermissionError(1, Operation not permitted)`. Because the gate is
the first step of this task, the following were not attempted:

- ROS master runtime validation;
- Unity Player or Bridge startup;
- BC Dev100 closed-loop evaluation;
- formal critic calibration and replay collection;
- calibration pass-checkpoint creation or resume validation.

## BC Dev100 baseline

The immutable Dev set remains
`data/test/awac_dev_seed4026_100` (`role=DEV`, 100 missions). Its baseline
target is `data/test/awac_dev_seed4026_100/bc60k_baseline/`, but no baseline
was created or redefined:

```text
BC_DEV_ATTEMPTED=0
BC_DEV_RELIABLE_EXACT_QUALITY=NOT_EXECUTED_SOCKET_GATE
BC_DEV_SUCCESS_COUNT=NOT_AVAILABLE
BC_DEV_SUCCESS_RATE=NOT_AVAILABLE
BC_DEV_COLLISION=NOT_AVAILABLE
BC_DEV_DEAD_END=NOT_AVAILABLE
BC_DEV_TIMEOUT=NOT_AVAILABLE
BC_DEV_HARD_ALTITUDE=NOT_AVAILABLE
BC_DEV_OTHER_FAILURE=NOT_AVAILABLE
BC_DEV_MEAN_RETURN=NOT_AVAILABLE
BC_DEV_MEAN_STEPS=NOT_AVAILABLE
BC_DEV_MEAN_SUCCESS_STEPS=NOT_AVAILABLE
BC_DEV_DECISION_TIME_MEAN_MS=NOT_AVAILABLE
BC_DEV_DECISION_TIME_P95_MS=NOT_AVAILABLE
BC_DEV_BASELINE_SHA256=NOT_CREATED
```

The `FINAL_TEST_ONLY` artifact at
`data/test/bc_closed_loop_eval_300` was not read as a policy baseline, run, or
modified.

## Formal critic calibration

Formal calibration run
`awac_bc60k_seed2026_phase1_calibration_v1` was not started. Consequently,
formal calibration metrics, Actor before/after hashes, formal replay size,
and a formal pass checkpoint are unavailable.

The calibration gate state below is inherited from the bounded synthetic
critic-only smoke, not from the formal run:

```text
CALIBRATION_GATE_STATE=PENDING
CALIBRATION_CERTIFICATION=BLOCKED_PENDING
CALIBRATION_CERTIFICATION_DETAIL=FORMAL_RUN_BLOCKED_BY_SOCKET_GATE; SAFETY_CAP_NOT_REACHED
```

The canonical typed safety cap remains 30,000 transitions or 1,000 episodes.
It is a hard safety cap, not a PASS threshold. The synthetic smoke evidence
was:

| Metric | Synthetic smoke value |
|---|---:|
| Replay transitions | 160 |
| Episodes | 20 |
| Critic updates | 8 |
| Holdout TD loss | 109.96343994140625 |
| Q mean / std | -0.7482772469520569 / 0.3176383376121521 |
| Q p99 | -0.43699167609214784 |
| Twin-Q disagreement mean / p95 | 2.214468777179718 / 3.9950104773044584 |
| Normalized twin disagreement | 0.24353810783601798 |
| Holdout Q-return rank correlation | 0.0 |
| Success Q median | unavailable |
| Failure Q median | -0.7482772469520569 |
| Success-minus-failure Q median | unavailable |

Synthetic smoke artifact:
`/tmp/xm-awac-phase1-calibration-smoke-v2-20260904/checkpoint_calibration.pt`  
SHA256: `11dab8d816823eb8fb7036db1f7732df0ecaea64d6339d153ef06aef8e8cce72`

The smoke kept the Actor frozen and updated both Critics. This supports the
implementation seam only:

```text
CALIBRATION_ACTOR_UNCHANGED=YES_FOR_SYNTHETIC_SMOKE_ONLY
ACTOR_UPDATE_COUNT=0
CRITIC1_UPDATED=YES_FOR_SYNTHETIC_SMOKE_ONLY
CRITIC2_UPDATED=YES_FOR_SYNTHETIC_SMOKE_ONLY
```

No `checkpoint_calibration_pass.pt` exists:

```text
CALIBRATION_PASS_CHECKPOINT_CREATED=NO
CALIBRATION_PASS_CHECKPOINT=checkpoint_calibration_pass.pt
CALIBRATION_PASS_CHECKPOINT_SHA256=NOT_CREATED
CALIBRATION_RESUME_DRY_RUN=FAIL_NO_PASS_CHECKPOINT
```

## Phase 1 implementation and safety audit

The previously implemented Phase 1 state machine, milestone loop, Dev hook,
typed calibration cap, and PASS-only checkpoint path remain present. The
explicit Actor-enabling transition remains gated on a genuine calibration
`PASS`; no Actor update was enabled or executed in this certification attempt.

```text
FORMAL_ACTOR_DEPTH_LR=1.0e-6
FORMAL_CRITIC_DEPTH_LR=1.0e-5
PHASE1_STATE_MACHINE_IMPLEMENTED=YES
PHASE1_MILESTONE_LOOP_IMPLEMENTED=YES
DEV_EVALUATION_HOOK_IMPLEMENTED=YES
PHASE1_READINESS_DRY_RUN=PASS
PHASE1_READINESS=FAIL
PHASE1_READINESS_DETAIL=BLOCKED_BY_RUNTIME_CAPABILITY_GATE
```

Static/synthetic replay audit remains clean within its scope:

```text
AWAC_REPLAY_AUDIT=PASS
ACTOR_GLOBAL_MAP_LEAK_COUNT=0
CRITIC_GLOBAL_MAP_LEAK_COUNT=0
REPLAY_PRIVILEGED_FIELD_COUNT=0
```

This is not a formal calibration replay audit because no formal replay was
created.

## Verification evidence

The retained host certification is `705 passed, 42 skipped, 0 failed`. The
current restricted sandbox also produced `706 passed, 42 skipped, 8 failed`;
all eight failures were pre-existing loopback-socket permission failures in
`tests/test_p0_m2_runner_startup.py` and
`tests/test_p0_m2_single_worker_runner.py`. Excluding only those environment
failures, the code-eligible result was `706 passed, 42 skipped, 0 failed`.
No test was deleted, skipped, mocked, or altered to bypass the restriction.

```text
COMPILEALL=PASS
FOCUSED_AWAC_PHASE1_TESTS=76 passed, 0 failed
FULL_TEST_HOST_CERTIFICATION=705 passed, 42 skipped, 0 failed
FULL_TEST_CODE_ELIGIBLE=706 passed, 42 skipped, 0 failed
RUNTIME_SOCKET_BLOCKED_TEST_FAILURES=8 (sandbox-only)
```

## Frozen components and execution boundaries

```text
ACTOR_UPDATE_ENABLED=NO
STANDARD_AWAC_LONG_TRAINING_EXECUTED=NO
CONFIDENCE_ALGORITHM_IMPLEMENTED=NO
ADAPTIVE_BC_KL_IMPLEMENTED=NO
PRIMITIVE_NEIGHBOR_ALGORITHM_IMPLEMENTED=NO
FINAL_300_EVALUATION_EXECUTED=NO
FROZEN_FINAL_TEST_300_MODIFIED=NO
BC_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
REWARD_CONTRACT_CHANGED=NO
MPL_CHANGED=NO
COMMIT=NO
```

Only the two report files for this certification were added. The previous
Phase 1 implementation files and prior readiness report were not overwritten
by this report. No formal data, replay, checkpoint, Unity artifact, or Bridge
artifact was created or modified.

## Final status

```text
RL_PHASE_1_RUNTIME_CERTIFICATION_DEV100_AND_FORMAL_CRITIC_CALIBRATION_V1=FAIL
RUNTIME_CAPABILITY_GATE=FAIL_SOCKET_ENVIRONMENT

BC60K_CHECKPOINT=data/teach/2026_6w/bc_training/checkpoint_best_soft.pt
BC60K_CHECKPOINT_SHA256=ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2

BC_DEV_BASELINE_PATH=data/test/awac_dev_seed4026_100/bc60k_baseline/
BC_DEV_ATTEMPTED=0
BC_DEV_SUCCESS_COUNT=NOT_AVAILABLE
BC_DEV_SUCCESS_RATE=NOT_AVAILABLE
BC_DEV_COLLISION=NOT_AVAILABLE
BC_DEV_DEAD_END=NOT_AVAILABLE
BC_DEV_TIMEOUT=NOT_AVAILABLE
BC_DEV_HARD_ALTITUDE=NOT_AVAILABLE
BC_DEV_OTHER_FAILURE=NOT_AVAILABLE
BC_DEV_MEAN_RETURN=NOT_AVAILABLE
BC_DEV_MEAN_STEPS=NOT_AVAILABLE
BC_DEV_DECISION_TIME_MEAN_MS=NOT_AVAILABLE
BC_DEV_DECISION_TIME_P95_MS=NOT_AVAILABLE
BC_DEV_RELIABLE_EXACT_QUALITY=NOT_EXECUTED_SOCKET_GATE
BC_DEV_BASELINE_SHA256=NOT_CREATED

FORMAL_ACTOR_DEPTH_LR=1.0e-6
FORMAL_CRITIC_DEPTH_LR=1.0e-5
CALIBRATION_GATE_STATE=PENDING
CALIBRATION_CERTIFICATION=BLOCKED_PENDING
CALIBRATION_PASS_ENV_STEP=NOT_AVAILABLE
CALIBRATION_PASS_REPLAY_SIZE=NOT_AVAILABLE
CALIBRATION_PASS_EPISODES=NOT_AVAILABLE
CALIBRATION_PASS_CRITIC_UPDATES=NOT_AVAILABLE
CALIBRATION_REPLAY_TRANSITIONS=NOT_AVAILABLE_FORMAL; 160_SYNTHETIC_SMOKE
CALIBRATION_EPISODES=NOT_AVAILABLE_FORMAL; 20_SYNTHETIC_SMOKE
CALIBRATION_CRITIC_UPDATES=NOT_AVAILABLE_FORMAL; 8_SYNTHETIC_SMOKE
CALIBRATION_HOLDOUT_TD_LOSS=109.96343994140625_SYNTHETIC_SMOKE_ONLY
CALIBRATION_Q_MEAN=-0.7482772469520569_SYNTHETIC_SMOKE_ONLY
CALIBRATION_Q_STD=0.3176383376121521_SYNTHETIC_SMOKE_ONLY
CALIBRATION_Q_P01=NOT_RECORDED
CALIBRATION_Q_P99=-0.43699167609214784_SYNTHETIC_SMOKE_ONLY
CALIBRATION_TWIN_DISAGREEMENT_MEAN=2.214468777179718_SYNTHETIC_SMOKE_ONLY
CALIBRATION_TWIN_DISAGREEMENT_P95=3.9950104773044584_SYNTHETIC_SMOKE_ONLY
CALIBRATION_NORMALIZED_TWIN_DISAGREEMENT=0.24353810783601798_SYNTHETIC_SMOKE_ONLY
HOLDOUT_Q_RETURN_RANK_CORRELATION=0.0_SYNTHETIC_SMOKE_ONLY
SUCCESS_Q_MEDIAN=NOT_AVAILABLE
FAILURE_Q_MEDIAN=-0.7482772469520569_SYNTHETIC_SMOKE_ONLY
SUCCESS_MINUS_FAILURE_Q_MEDIAN=NOT_AVAILABLE
CALIBRATION_ACTOR_UNCHANGED=YES_FOR_SYNTHETIC_SMOKE_ONLY
ACTOR_UPDATE_COUNT=0
CRITIC1_UPDATED=YES_FOR_SYNTHETIC_SMOKE_ONLY
CRITIC2_UPDATED=YES_FOR_SYNTHETIC_SMOKE_ONLY
CALIBRATION_PASS_CHECKPOINT_CREATED=NO
CALIBRATION_PASS_CHECKPOINT=checkpoint_calibration_pass.pt
CALIBRATION_PASS_CHECKPOINT_SHA256=NOT_CREATED
AWAC_REPLAY_AUDIT=PASS_STATIC_AND_SYNTHETIC_ONLY
CALIBRATION_RESUME_DRY_RUN=FAIL_NO_PASS_CHECKPOINT
PHASE1_READINESS_DRY_RUN=PASS
PHASE1_READINESS=FAIL

ACTOR_UPDATE_ENABLED=NO
STANDARD_AWAC_LONG_TRAINING_EXECUTED=NO
CONFIDENCE_ALGORITHM_IMPLEMENTED=NO
ADAPTIVE_BC_KL_IMPLEMENTED=NO
PRIMITIVE_NEIGHBOR_ALGORITHM_IMPLEMENTED=NO
ACTOR_GLOBAL_MAP_LEAK_COUNT=0
CRITIC_GLOBAL_MAP_LEAK_COUNT=0
REPLAY_PRIVILEGED_FIELD_COUNT=0
FINAL_300_EVALUATION_EXECUTED=NO
FROZEN_FINAL_TEST_300_MODIFIED=NO
FULL_TEST_PASS_COUNT=705
FULL_TEST_SKIP_COUNT=42
FULL_TEST_FAILURE_COUNT=0
BC_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
REWARD_CONTRACT_CHANGED=NO
MPL_CHANGED=NO
COMMIT=NO
NEXT_ACTION=RUN_ON_SOCKET_ENABLED_HOST_BC_DEV100_BASELINE_THEN_FORMAL_CRITIC_CALIBRATION
```

The next action is blocked until a socket-enabled runtime is available. Do
not start Actor AWAC training automatically.
