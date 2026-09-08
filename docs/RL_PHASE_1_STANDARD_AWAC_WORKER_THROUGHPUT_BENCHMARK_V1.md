# RL Phase 1 Standard AWAC Worker Throughput Benchmark

Date: 2026-09-04

## Result

```text
RL_PHASE_1_STANDARD_AWAC_WORKER_THROUGHPUT_BENCHMARK_V1=FAIL
RUNTIME_CAPABILITY=FAIL
DIRECT_SOCKET=FAIL
SUBPROCESS_SOCKET=FAIL
BENCHMARK_BLOCKED_BY_RUNTIME_CAPABILITY=YES
```

The required capability gate was run after activating `conda xm`. Both probes
failed at `socket.socket()` with:

```text
PermissionError: [Errno 1] Operation not permitted
```

The subprocess probe returned code 1 with the same error. Per the task's
fail-stop rule, no CLI bypass was attempted and no benchmark runtime was
started. The process check found no `XMflight`, `unity_bridge_node`, `roscore`,
`rosmaster`, or `train_awac` process owned by this attempt.

## Benchmark matrix

No worker configuration was executed, so no throughput or health value is
invented:

| Workers | Startup s | Trans/s | Episodes/min | Actor upd/s | Critic upd/s | GPU mean | GPU mem max | RAM max | Swap growth | Runtime failures | Replay audit | Eligible |
| ------: | --------: | ------: | -----------: | ----------: | -----------: | -------: | ----------: | ------: | ----------: | --------------- | ------------ | -------- |
| 2 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | capability gate | NOT_RUN | NO |
| 4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | capability gate | NOT_RUN | NO |
| 8 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | capability gate | NOT_RUN | NO |
| 12 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | capability gate | NOT_RUN | NO |

`W12_SKIPPED_RESOURCE_SAFETY=NO`: these runs were not skipped for resource
reasons; the complete matrix was blocked before runtime creation.

## Scope protection

```text
CALIBRATION_PASS_CHECKPOINT_SHA256=bb9433d8d9fb2fde2e2148e84a4058c2b328b9d676608d38c00962f821c3d8de
STANDARD_AWAC_RUNTIME_STARTED=NO
DEV100_EVALUATION_EXECUTED=NO
FINAL_300_EVALUATION_EXECUTED=NO
BENCHMARK_MODELS_ELIGIBLE_FOR_FORMAL_SELECTION=NO
```

The V6 Calibration checkpoint and Replay were not opened or modified in this
blocked attempt. No benchmark Replay/checkpoint/log artifact was created, so
there is no benchmark cleanup target. Retention and disk policy were not
retested because the capability gate precedes benchmark setup.

The frozen AWAC algorithm and all contracts remain unchanged:

```text
CONFIDENCE_ALGORITHM_IMPLEMENTED=NO
ADAPTIVE_BC_KL_IMPLEMENTED=NO
PRIMITIVE_NEIGHBOR_ALGORITHM_IMPLEMENTED=NO
BC_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
REWARD_CONTRACT_CHANGED=NO
MPL_CHANGED=NO
COMMIT=NO
```

## Tests

The task requires the full pytest pre-benchmark gate only after the runtime
capability gate passes. Because the capability gate failed, both benchmark
test slots are explicitly not run:

```text
FULL_TEST_PRE_BENCH_PASS_COUNT=NOT_RUN_CAPABILITY_GATE_FAIL
FULL_TEST_PRE_BENCH_SKIP_COUNT=NOT_RUN_CAPABILITY_GATE_FAIL
FULL_TEST_PRE_BENCH_FAILURE_COUNT=NOT_RUN_CAPABILITY_GATE_FAIL
FULL_TEST_POST_BENCH_PASS_COUNT=NOT_RUN_CAPABILITY_GATE_FAIL
FULL_TEST_POST_BENCH_SKIP_COUNT=NOT_RUN_CAPABILITY_GATE_FAIL
FULL_TEST_POST_BENCH_FAILURE_COUNT=NOT_RUN_CAPABILITY_GATE_FAIL
```

## Next action

Run this same benchmark request from a host execution session that permits
loopback socket creation. Once both direct and subprocess probes pass, rerun
the real CLI audit, full pytest pre-gate, and the isolated 2/4/8/12-worker
Standard AWAC matrix. No worker count is selected by this report.
