# RL Phase 1 Standard AWAC Smoke V3 and Fast Worker Benchmark V1

## Result

`RL_PHASE_1_STANDARD_AWAC_SMOKE_V3_AND_FAST_WORKER_BENCHMARK_V1=FAIL`

Stage A ran on the real managed Unity/Bridge runtime. It reached the bounded
online budget with a one-step parallel-wave shortfall, but final checkpoint
validation failed. Stage B was correctly not started.

This report records a failed smoke and must not be interpreted as a formal
AWAC model, a worker selection result, or a training checkpoint.

## Frozen inputs

- Environment: conda environment `xm`, ROS Noetic, catkin `devel` tree.
- BC checkpoint: `data/teach/2026_6w/bc_training/checkpoint_best_soft.pt`
- BC SHA256: `ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2`
- Formal start checkpoint: `data/awac/awac_bc60k_formal_critic_calibration_v6/checkpoint_calibration_pass.pt`
- Formal start checkpoint SHA256: `bb9433d8d9fb2fde2e2148e84a4058c2b328b9d676608d38c00962f821c3d8de`
- MPL contract SHA256: `f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d`
- Task contract SHA256: `2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df`
- Observation contract: `reliable_exact_endpoint_snapshot`
- Unity SHA256: `61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365`
- Bridge SHA256: `667c266e96211aabb5a0cd25b8b35b303903d35cbe2d95be8f9d2bbe5e63d789`

The initial direct and subprocess socket capability probes passed. No source
code, Unity project, Bridge binary, contract, or formal checkpoint was
modified.

## Stage A: W2 bounded smoke

Output directory:

`data/awac/smoke/standard_awac_online_w2_v3`

The executed command used the production `scripts/train_awac.py` entrypoint,
`--phase awac_training`, `--env-workers 2`, `--reliable-v4`,
`--online-env-steps 3965`, `--max-steps 45`, and the V6 calibration-pass
checkpoint as the resume source.

Observed terminal state:

- `online_env_steps=3964` with budget `3965` (bounded parallel-wave shortfall)
- `online_transitions_committed=3941`
- replay rows: `9083 = 5142 BC_CALIBRATION + 3941 AWAC_ONLINE`
- completed episodes: `359`
- critic updates: `2042`
- actor updates: `10`
- actor optimizer steps: `10`
- actor before SHA256: `07b3b1ae86b6f2b1a4d15fcda7684f08961da8856d10dc78ebb948095b1ca71a`
- actor after SHA256: `80d1b3bf39b8515263a9e0d970590467a862c693edd2cf97233ad3d4af479216`
- actor changed: `YES`

Replay audit was independently run after termination and returned `PASS`:

- `row_count=9083`
- `BC_CALIBRATION=5142`
- `AWAC_ONLINE=3941`
- `legacy_replay_transition_count=0`
- `privileged_field_count=0`
- `reliable_v4_transition_count=9083`
- terminal rows: `336`; nonterminal rows: `8747`
- replay float scan: `NaN=0`, `Inf=0`

The V6 source checkpoint SHA remained unchanged. Runtime final flush and owned
runtime cleanup both returned `PASS`; no owned Unity, Bridge, ROS master, or
training process remained, and no owned ports remained listening.

## Stage A failure

The final checkpoint transaction was `FAILED_INCOMPLETE`; no committed
checkpoint alias or `summary.json` was published. The summary validator
therefore failed because the summary does not exist.

Primary error:

```text
ValueError: standard AWAC progress online transition count mismatch
```

The failure is reproducible from the recorded state: the durable runner
progress contains `online_transitions_committed=3941`, while the final payload
builder obtains `online_transition_count` from the top-level
`snapshot["online_transitions_committed"]`. The current
`StandardAWACOnlineRunner.checkpoint_snapshot()` return mapping does not expose
that top-level key, so the builder falls back to `0`; the fail-closed validator
rejects the mismatch. This is an accounting/finalization defect, not evidence
of a numerical replay failure or a runtime quality failure.

The failed run is retained as evidence at:

- `data/awac/smoke/standard_awac_online_w2_v3/runtime_failure.json`
- `data/awac/smoke/standard_awac_online_w2_v3/checkpoint_last.transaction_attempt.json`
- `data/awac/smoke/standard_awac_online_w2_v3/replay_audit.json`

## Stage B decision

Stage B was not executed because Stage A did not satisfy all hard gates.

- W8 executed: `NO`
- W12 executed: `NO`
- benchmark common base: `NOT_ESTABLISHED`
- benchmark models eligible for formal selection: `NO`
- formal 10K training: `NO`
- Dev100 evaluation: `NO`
- Final300 evaluation: `NO`

## Required final fields

```text
RL_PHASE_1_STANDARD_AWAC_SMOKE_V3_AND_FAST_WORKER_BENCHMARK_V1=FAIL
RUNTIME_CAPABILITY=PASS
SMOKE_OUTPUT_DIR=data/awac/smoke/standard_awac_online_w2_v3
SMOKE_ONLINE_ENV_STEPS=3964
SMOKE_AWAC_ONLINE_ROWS=3941
SMOKE_ACTOR_UPDATE_COUNT=10
SMOKE_ACTOR_OPTIMIZER_STEP_COUNT=10
SMOKE_CRITIC_UPDATE_COUNT_END=2042
SMOKE_ACTOR_SHA_BEFORE=07b3b1ae86b6f2b1a4d15fcda7684f08961da8856d10dc78ebb948095b1ca71a
SMOKE_ACTOR_SHA_AFTER=80d1b3bf39b8515263a9e0d970590467a862c693edd2cf97233ad3d4af479216
SMOKE_ACTOR_CHANGED=YES
SMOKE_REPLAY_AUDIT=PASS
SMOKE_SUMMARY_VALIDATION=FAIL
SMOKE_CHECKPOINT_TRANSACTION=FAIL
SMOKE_NAN_COUNT=0
SMOKE_INF_COUNT=0
SMOKE_Q_DIVERGENCE=N/A_NOT_CERTIFIED
SMOKE_ACTOR_DIVERGENCE=N/A_NOT_CERTIFIED
SMOKE_RETENTION=N/A_NO_COMMITTED_CHECKPOINT
SMOKE_RUNTIME_CLEANUP=PASS
SMOKE_PASS=NO
BENCHMARK_BASE_CHECKPOINT=NOT_ESTABLISHED
BENCHMARK_BASE_CHECKPOINT_SHA256=NOT_ESTABLISHED
W8_EXECUTED=NO
W8_ELIGIBLE=NO
W12_EXECUTED=NO
W12_ELIGIBLE=NO
W12_VS_W8_THROUGHPUT_DELTA=NOT_EXECUTED
SELECTED_ENV_WORKERS=NOT_SELECTED
SELECTED_WORKER_REASON=Stage A checkpoint finalization gate failed; Stage B was not authorized.
FORMAL_STANDARD_AWAC_START_CHECKPOINT=data/awac/awac_bc60k_formal_critic_calibration_v6/checkpoint_calibration_pass.pt
FORMAL_STANDARD_AWAC_START_CHECKPOINT_SHA256=bb9433d8d9fb2fde2e2148e84a4058c2b328b9d676608d38c00962f821c3d8de
BENCHMARK_MODELS_ELIGIBLE_FOR_FORMAL_SELECTION=NO
DEV100_EVALUATION_EXECUTED=NO
FINAL_300_EVALUATION_EXECUTED=NO
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
NEXT_ACTION=DIAGNOSE_STANDARD_AWAC_SMOKE
```

