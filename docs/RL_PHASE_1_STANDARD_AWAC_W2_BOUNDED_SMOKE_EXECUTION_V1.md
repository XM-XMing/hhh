# RL Phase 1 Standard AWAC W2 Bounded Smoke Execution V1

Date: 2026-09-05

## Result

```text
RL_PHASE_1_STANDARD_AWAC_W2_BOUNDED_SMOKE_EXECUTION_V1=FAIL
RUNTIME_CAPABILITY=PASS
DIRECT_SOCKET=PASS
SUBPROCESS_SOCKET=PASS
RUNTIME_EXECUTED=YES
DEV100_EVALUATION_EXECUTED=NO
FINAL_300_EVALUATION_EXECUTED=NO
COMMIT=NO
```

The real managed W2 runtime started successfully with two ROS masters, two
Unity Players, and two Bridges. It reached the bounded online budget boundary
and produced both calibration and `AWAC_ONLINE` replay rows. The run is still
classified as `FAIL` because final Standard AWAC checkpoint snapshot creation
failed before `summary.json` and the final checkpoint were written.

## Frozen inputs and output

```text
ROOT=/home/xm/XM/xm_ws/src/planning
WS=/home/xm/XM/xm_ws
SMOKE_OUTPUT_DIR=data/awac/smoke/standard_awac_online_w2_v2
ENV_WORKERS_REQUESTED=2
ENV_WORKERS_ACTUAL=2
STARTING_REPLAY_SIZE=5142
ONLINE_ENV_STEPS_BUDGET=3965
ONLINE_ENV_STEPS_ACTUAL=3964
ENDING_REPLAY_SIZE=9083
BC_CALIBRATION_ROWS=5142
AWAC_ONLINE_ROWS=3941
```

The BC checkpoint SHA was
`ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2`.
The Calibration PASS checkpoint SHA was
`bb9433d8d9fb2fde2e2148e84a4058c2b328b9d676608d38c00962f821c3d8de`.
The V6 replay metadata SHA remained
`0e758ed534d82bd8b508522775e1e9409033873ae952664b38b018fb8cbfd65e`.

## Observed training progress

The durable failure report records:

```text
completed_episode_count=359
online_transitions_committed=3941
actor_update_count=10
actor_optimizer_step_count=10
critic_update_count_start=71
critic_update_count_end=2042
```

Actor updates therefore occurred in memory, but the required
`actor_state_before_sha256 != actor_state_after_sha256` proof was not emitted
because the final checkpoint snapshot failed. Actor identity is consequently
`NOT_VERIFIABLE`, not a success claim.

## Failure and preserved diagnostics

The managed runtime cleanup completed successfully. Replay final flush also
reported `PASS`, and no owned ROS master, Unity Player, Bridge, or environment
worker remained after the process exited.

The terminal exception was:

```text
KeyError: '0'
```

at `StandardAWACOnlineRunner.checkpoint_snapshot()` while
`build_standard_online_exact_resume_state()` was fingerprinting
`learner_state` through `_resume_state_sha256()` / `_hash_resume_value()`.
This is a Standard AWAC exact-resume checkpoint serialization blocker. No
retry, rollback, parameter change, or source-code fix was performed.

Preserved failure evidence:

```text
data/awac/smoke/standard_awac_online_w2_v2/runtime_failure.json
data/awac/smoke/standard_awac_online_w2_v2/replay_audit.json
data/awac/smoke/standard_awac_online_w2_v2/replay/metadata.json
```

`summary.json` and `run_outcome.json` were not created. No rolling checkpoint
file was created, so the observed rolling-checkpoint generation count is 0;
this does not compensate for the missing final checkpoint.

## Replay and contract audit

`audit_awac_replay.py` returned `status=PASS`:

```text
row_count=9083
behavior_source_counts={"0": 5142, "1": 3941}
legacy_replay_transition_count=0
privileged_field_count=0
action_dim=105
vector_dim=127
depth_shape=[1, 90, 160]
```

The replay contains the exact endpoint observation contract
`reliable_exact_endpoint_snapshot`, and the online rows are marked
`AWAC_ONLINE`. The replay data layer is therefore reliable-exact for the
rows that were committed; this is separate from the failed run finalization.

The required summary validator was executed and failed closed because
`summary.json` did not exist. Consequently TD/Q/AWAC-weight/BC-KL/entropy
summary statistics, actor before/after hashes, and formal summary quality
counters are `NOT_WRITTEN`.

## Resource and integrity observations

```text
SWAP_GROWTH_GB=0
SWAP_AT_FAILURE_BYTES=1202978816
RAM_AVAILABLE_DURING_RUN_APPROX_GB=53
OOM=NO
```

No runtime failure storm, Unity/Bridge protocol error, or replay corruption
was observed before the checkpoint-snapshot exception. GPU utilization and GPU
memory were not durably captured by this run. The source V6 checkpoint,
source V6 replay metadata, and BC checkpoint retained their pre-run SHA256
identities. V6 was not modified.

## Frozen boundaries

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
SMOKE_MODEL_ELIGIBLE_FOR_FORMAL_SELECTION=NO
```

The smoke output is benchmark/diagnostic evidence only. It is not a formal
10K checkpoint, best development model, future training start, or final
model. No worker benchmark or later evaluation was started.

## Next action

```text
NEXT_ACTION=FIX_STANDARD_AWAC_RUNTIME_BLOCKER
```

Fix the exact-resume learner-state fingerprint contract, add a focused
regression test for non-string optimizer-state keys, and only then request a
new bounded smoke from the original V6 Calibration PASS checkpoint.
