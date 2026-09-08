# RL Phase 1 Standard AWAC Online Runtime Implementation V1

Date: 2026-09-05

## Result

This report records the implementation and CPU/synthetic verification of the
Standard AWAC online owner. No real online runtime was executed in this
environment: no socket, ROS master, Unity, Bridge, Dev100, Final300, Actor
production update, BC training, AWAC training, or data collection was run.

```text
RL_PHASE_1_STANDARD_AWAC_ONLINE_RUNTIME_IMPLEMENTATION_V1=PASS
STANDARD_AWAC_ONLINE_RUNTIME_IMPLEMENTED=YES
STANDARD_AWAC_ONLINE_OWNER=planning.awac.online_runtime.StandardAWACOnlineRunner
STANDARD_AWAC_ONLINE_CLI_READY=YES
RUNTIME_EXECUTED=NO
ACTOR_UPDATE_EXECUTED=NO
```

The PASS is an implementation/test result, not a claim of host runtime
success. The host commands below are intentionally not executed here.

## Ownership and handoff

`planning.awac.trainer` now selects a distinct `awac_training` path when a
committed Calibration PASS checkpoint and its replay are supplied. The path
uses `StandardAWACOnlineRunner`; it does not copy a second environment
implementation or alter the existing Calibration producer.

| Concern | Owner | Result |
| --- | --- | --- |
| Managed lifecycle | `planning.runtime.managed_runtime.ManagedRuntimePool` | reused |
| Parallel environment | `planning.runtime.parallel_env.ParallelEnvPool` | reused |
| Mission dispatch and cursor | inherited calibration producer plus Standard progress | exact identity checked |
| Observation/endpoint validation | `CalibrationReplayProducer` / `planning.runtime.unity_env` | reliable-exact contract reused |
| Transition encoding | inherited replay transition builder | fields and terminal mask preserved |
| Online behavior | `StandardAWACOnlineRunner._policy_action` | masked stochastic AWAC Actor |
| Replay handoff | `AWACReplayBuffer.clone_from` | independent sparse/preallocated replay |
| Checkpoint transaction | Standard wrapper over existing transaction owner | committed, crash-safe, retention 2 |
| Summary | `build_standard_online_summary` | typed `awac_standard_online_summary_v1` |

The only valid starting state is a checkpoint with Calibration gate `PASS`,
Calibration-pass marker `true`, and a `COMMITTED` transaction. BC, Task,
Observation, Policy-input, MPL, Reward, Replay, mission, and checkpoint
identities are compared before the managed runtime is constructed.

Frozen handoff artifacts:

```text
Calibration PASS checkpoint:
data/awac/awac_bc60k_formal_critic_calibration_v6/checkpoint_calibration_pass.pt
SHA256:
bb9433d8d9fb2fde2e2148e84a4058c2b328b9d676608d38c00962f821c3d8de

Calibration replay:
data/awac/awac_bc60k_formal_critic_calibration_v6/replay
starting size and total_added: 5142
behavior source: BC_CALIBRATION=5142, AWAC_ONLINE=0

BC checkpoint:
data/teach/2026_6w/bc_training/checkpoint_best_soft.pt
SHA256:
ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2

Observation contract: reliable_exact_endpoint_snapshot
Task contract SHA256:
2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df
MPL contract SHA256:
f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d
```

The V6 checkpoint and replay are opened read-only for the handoff. Standard
creates a new run replay with the 5,142 Calibration rows preserved and marks
new rows `AWAC_ONLINE`; it never writes the V6 replay.

## Standard online behavior and transition contract

The behavior action is selected from the current Actor through the existing
masked policy owner, with `deterministic=false` and the configured positive
temperature. The current action mask is validated before the action is
stored. A zero feasible-action mask follows the inherited terminal/dead-end
path and is never passed to the Actor as a selectable distribution.

Every committed online row retains the existing transition layout:

```text
depth, vector, action_mask, action, reward,
next_depth, next_vector, next_action_mask, done,
behavior_source=AWAC_ONLINE
```

Runtime/endpoint failure, timeout, invalid observation, snapshot/skew failure,
invalid identity, or non-finite reward is dropped as a failed attempt and is
not converted into a replay terminal row. Cleanup is fail-safe and owned by
the inherited managed runtime lifecycle.

The loss remains the existing masked-discrete AWAC implementation: minimum
twin Q, valid-action masked policy expectation, clipped exponential advantage
weight, and fixed masked BC KL. No confidence, adaptive KL, primitive-neighbor,
candidate-funnel, SAC alpha, or legacy rollback logic was added.

```text
STANDARD_AWAC_LOSS=MASKED_DISCRETE_AWAC
BC_KL_MODE=FIXED
BC_KL_WEIGHT=0.05
CONFIDENCE_ALGORITHM_IMPLEMENTED=NO
ADAPTIVE_BC_KL_IMPLEMENTED=NO
PRIMITIVE_NEIGHBOR_ALGORITHM_IMPLEMENTED=NO
```

## Warmup and budget semantics

The schedule clocks are persistent replay/learner clocks, not a newly reset
phase clock:

| Setting | Value | Counter owner |
| --- | ---: | --- |
| `learning_starts` | 5,000 | `replay.total_added` |
| `actor_learning_starts` | 8,000 | `replay.total_added` |
| `critic_burnin_updates` | 2,000 | `learner.update_step` |
| `actor_update_interval` | 4 | post-burn-in learner update number |
| `updates_per_step` | 0.50 | persistent replay cursor target |
| `batch_size` | 128 | canonical CLI default |

The Calibration PASS state is `replay.total_added=5142`,
`critic_update_count=71`, and `learner.update_step=71`. The exact schedule
simulation returns:

```text
MINIMUM_ONLINE_TRANSITIONS_FOR_FIRST_ACTOR_UPDATE=3865
```

This value is a derived schedule result, not a lowered threshold. The bounded
W2 host smoke uses `--online-env-steps 3965`. The CLI budget is explicitly
additional, phase-local environment steps. The extra 100 steps cover two
workers, episode-commit granularity (rows are committed on complete episode),
and a short post-Actor window while keeping the run bounded; it is not a
claim that exactly 3,965 replay rows will be produced.

The Standard runner keeps Critic updates continuous against the persistent
replay cursor, enables Actor updates only after both threshold predicates, and
updates targets through the existing learner update owner.

```text
LEARNING_STARTS_COUNTER_OWNER=replay_total_added
ACTOR_LEARNING_STARTS_COUNTER_OWNER=replay_total_added
CRITIC_BURNIN_COUNTER_OWNER=learner.update_step
ONLINE_BUDGET_COUNTER_OWNER=phase_local_online_environment_steps
ONLINE_BUDGET_ACTUALLY_CONSUMED=YES
CONTINUOUS_CRITIC_UPDATE_LOOP=YES
CONTINUOUS_ACTOR_UPDATE_LOOP=YES
TARGET_UPDATE_LOOP=YES
```

## Checkpoint, resume, and summary

Standard checkpoints use a separate exact-resume state contract with 14
required logical state owners:

```text
learner parameters and optimizer state
learner counters
replay identity
producer / Python / NumPy / Torch CPU / Torch CUDA RNG state
mission source identity and ordered IDs
mission cursor and completed IDs
environment and online counters
online schedule counters
Phase-1 state
runtime identity
```

The actual serialized state also validates the learner field inventory, replay
identity, mission order, completed mission order, online budget, Phase-1 state,
runtime identity, and RNG capability. Missing or malformed Standard state is
fail-closed; it is not interpreted as a Calibration checkpoint. Resume uses
the current Standard replay and exact online progress, while a Calibration
handoff starts the phase-local online counter at zero and preserves the source
Calibration replay counters for warmup decisions.

```text
STANDARD_ONLINE_RESUME_STATUS=PASS
STANDARD_ONLINE_RESUME_REQUIRED_STATE=14 logical owners
STANDARD_ONLINE_RESUME_MISSING_STATE=NONE
ROLLING_CHECKPOINT_RETENTION=2
RETENTION_REGRESSION=PASS
```

The typed summary schema is `awac_standard_online_summary_v1` and requires
phase, start checkpoint identity, starting/ending replay size, online steps and
budget, completed episodes, behavior-source counts, Actor before/after SHA,
Critic/Actor diagnostics, runtime failures, runtime identity, checkpoint paths,
cleanup status, stop reason, exact observation contract, and finite-metric
status. `scripts/validate_standard_awac_online_summary.py` is the canonical
summary validator.

Replay audit remains the existing `scripts/audit_awac_replay.py`; it accepts
the mixed `BC_CALIBRATION` plus `AWAC_ONLINE` source contract and reports
numeric source keys `0` and `1`.

## Verification

All Python commands were run only after `conda activate xm`. The implementation
focused suite passed after the final source/test changes:

```text
CODE_ELIGIBLE_FOCUSED_TESTS=70 passed, 8 skipped, 0 failed
```

The final full pytest run was:

```text
795 passed, 43 skipped, 8 failed
```

All eight failures are existing executor restrictions: the test modules
`tests/test_p0_m2_runner_startup.py` and
`tests/test_p0_m2_single_worker_runner.py` attempt socket creation and receive
`PermissionError: [Errno 1] Operation not permitted`. With those two socket
runtime modules excluded, the code-eligible full collection is:

```text
CODE_ELIGIBLE_TEST_PASS_COUNT=779
CODE_ELIGIBLE_TEST_SKIP_COUNT=43
CODE_ELIGIBLE_TEST_FAILURE_COUNT=0
```

Compileall, real CLI help for `train_awac.py` and `audit_awac_replay.py`, the
typed summary CLI, Standard runner tests, handoff tests, replay tests, learner
tests, checkpoint/resume tests, bounded-stop tests, and naming/contract tests
were run. No runtime process or socket was started by these checks.

## Changed files

Implementation and test files in the Planning tree:

```text
python/planning/awac/checkpoint.py
python/planning/awac/learner.py
python/planning/awac/model.py
python/planning/awac/online_runtime.py
python/planning/awac/optimization.py
python/planning/awac/replay.py
python/planning/awac/trainer.py
scripts/validate_standard_awac_online_summary.py
tests/test_awac_standard_online_runtime.py
tests/test_awac_trainer_migration.py
tests/contracts/test_contract_naming.py
docs/RL_PHASE_1_STANDARD_AWAC_ONLINE_RUNTIME_IMPLEMENTATION_V1.md
docs/rl_phase_1_standard_awac_online_runtime_implementation_v1.json
```

Protected BC artifacts, V6 Calibration artifacts, Unity, Bridge, Task,
Observation, Reward, MPL, trajectory, and legacy SAC/RL sources were not
modified by this phase.

## Host commands

These commands are generated from the current CLI and are not executed by
Codex in this environment. The first command creates the Standard run replay
through the implementation's read-only V6 handoff; it does not copy or mutate
the V6 replay.

### HOST_STANDARD_AWAC_W2_SMOKE

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
set -euo pipefail
ROOT=/home/xm/XM/xm_ws/src/planning
V6="$ROOT/data/awac/awac_bc60k_formal_critic_calibration_v6"
BC="$ROOT/data/teach/2026_6w/bc_training/checkpoint_best_soft.pt"
MISSIONS="$ROOT/data/teach/2026_6w/missions.csv"
WORKER_SPEC="$ROOT/data/teach/2026_6w/rollouts/worker_runtime_specs.json"
OUT="$ROOT/data/awac/smoke/standard_awac_online_w2_v1"
REPLAY_SRC="$V6/replay"
PASS_CHECKPOINT="$V6/checkpoint_calibration_pass.pt"
BC_SHA=ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2
PASS_SHA=bb9433d8d9fb2fde2e2148e84a4058c2b328b9d676608d38c00962f821c3d8de
MPL_SHA=f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d
TASK_SHA=2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df
test -f "$BC"
test -f "$MISSIONS"
test -f "$WORKER_SPEC"
test -d "$REPLAY_SRC"
test -f "$PASS_CHECKPOINT"
test "$(sha256sum "$BC" | awk '{print $1}')" = "$BC_SHA"
test "$(sha256sum "$PASS_CHECKPOINT" | awk '{print $1}')" = "$PASS_SHA"
test ! -e "$OUT"
python scripts/train_awac.py \
  --phase awac_training \
  --bc-checkpoint "$BC" \
  --train-index "$MISSIONS" \
  --out-dir "$OUT" \
  --resume-checkpoint "$PASS_CHECKPOINT" \
  --replay-dir "$REPLAY_SRC" \
  --mpl-contract-sha256 "$MPL_SHA" \
  --online-env-steps 3965 \
  --max-steps 45 \
  --env-workers 2 \
  --worker-spec-file "$WORKER_SPEC" \
  --reliable-v4 \
  --device auto \
  --seed 55 \
  --cpu-threads 1
```

### HOST_STANDARD_AWAC_W2_REPLAY_AUDIT

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
set -euo pipefail
ROOT=/home/xm/XM/xm_ws/src/planning
OUT="$ROOT/data/awac/smoke/standard_awac_online_w2_v1"
AUDIT="$OUT/replay_audit.json"
test -d "$OUT/replay"
python scripts/audit_awac_replay.py \
  --replay-dir "$OUT/replay" \
  --output "$AUDIT"
python -c 'import json, pathlib; p = pathlib.Path("'"$AUDIT"'"); r = json.loads(p.read_text(encoding="utf-8")); assert r["status"] == "PASS"; assert int(r["behavior_source_counts"].get("0", 0)) > 0; assert int(r["behavior_source_counts"].get("1", 0)) > 0; assert int(r["legacy_replay_transition_count"]) == 0; assert int(r["privileged_field_count"]) == 0; print("STANDARD_AWAC_REPLAY_AUDIT=PASS")'
```

### HOST_STANDARD_AWAC_W2_SMOKE_SUMMARY

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
set -euo pipefail
ROOT=/home/xm/XM/xm_ws/src/planning
OUT="$ROOT/data/awac/smoke/standard_awac_online_w2_v1"
SUMMARY="$OUT/summary.json"
test -f "$SUMMARY"
python scripts/validate_standard_awac_online_summary.py \
  --summary "$SUMMARY"
python -c 'import json, pathlib; p = pathlib.Path("'"$SUMMARY"'"); s = json.loads(p.read_text(encoding="utf-8")); assert s["phase"] == "awac_training"; assert s["reliable_exact_observation_contract"] == "reliable_exact_endpoint_snapshot"; assert int(s["starting_replay_size"]) == 5142; assert int(s["awac_online_rows"]) > 0; assert int(s["actor_updates"]) > 0; assert s["actor_state_before_sha256"] != s["actor_state_after_sha256"]; assert int(s["critic_updates"]) > 0; assert s["online_budget_counter_owner"] == "phase_local_online_environment_steps"; assert int(s["online_env_steps"]) <= int(s["online_env_steps_budget"]); assert int(s["nan_inf_count"]) == 0; assert int(s["nonfinite_metric_count"]) == 0; print("STANDARD_AWAC_SMOKE_SUMMARY=PASS")'
```

The next action after this implementation phase is
`RUN_HOST_STANDARD_AWAC_W2_BOUNDED_SMOKE`. It is a human-host action. No
Dev100, Final300, production Actor update, long training, or later RL phase is
authorized by this report.
