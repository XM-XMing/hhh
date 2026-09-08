# RL Phase 1: Formal Replay Producer and Calibration Loop V1

## Result and scope

This change adds the formal critic-only online calibration path under
`planning.awac`. It is a capability implementation; no Unity, Bridge, ROS,
socket, formal calibration run, Actor update, or Final300 evaluation was
executed in this environment.

```text
FORMAL_REPLAY_PRODUCER_OWNER=planning.awac.calibration_runtime.CalibrationReplayProducer
PARALLEL_ENV_OWNER=planning.runtime.parallel_env.ParallelEnvPool
FORMAL_ENTRYPOINT=scripts/train_awac.py
FRESH_ONLINE_SELECTOR=phase=critic_calibration + train_index + no resume checkpoint
DEFAULT_ONLINE_REPLAY_DIR=<out-dir>/replay
OFFLINE_REPLAY_SMOKE=RETAINED
ACTOR_UPDATE_ENABLED_DURING_CALIBRATION=NO
RUNTIME_EXECUTED=NO
UNITY_STARTED=NO
BRIDGE_STARTED=NO
```

The producer consumes a validated V2 training mission CSV in source order,
dispatches start/goal resets through the existing pool, and never substitutes
a replay report or a Final Test index. `--env-workers` is applied while
constructing the actual `ParallelEnvPool`; the selected worker specifications
must be contiguous and have unique runtime identities.

## Transition and behavior contract

The persistent replay remains the existing ten-field contract:

```text
depth, vector, action_mask, action, reward,
next_depth, next_vector, next_action_mask, done, behavior_source
```

No episode or route/map fields were added to transitions. Continuous features
are normalized with the BC checkpoint normalizer, the previous-action one-hot
is unchanged, and the first previous action is `-1`. The action path delegates
to `planning.evaluation.policy_evaluator.choose_action` with temperature `0.0`
and the current depth mask. Its tuple result is unwrapped to the deterministic
action only; confidence and top-k diagnostics are not used.

The producer stores the raw environment reward. The learner's existing masked
Bellman target remains the sole owner of the `0.10` reward scale. A runtime or
observation failure is recorded diagnostically and cannot append a replay row
or fabricate a terminal reward. A normal terminal reason is committed only at
episode boundary, with `done=True`, so the existing learner applies no
bootstrap to it. Holdout episodes are retained only in a diagnostic window and
are never added to the optimizer replay.

## Loop and lifecycle

The producer performs:

```text
ready -> reset missions -> frozen BC action -> step wave
      -> validate all worker replies -> terminal episode commit
      -> critic warmup/update -> calibration gate window -> repeat
```

The gate uses the existing `evaluate_calibration_gate` and
`Phase1CalibrationController`. `PASS` stops new mission assignment and allows
safe in-flight completion before replay flush and the pass checkpoint.
`PENDING` at a typed transition/episode cap becomes `BLOCKED_PENDING` and
creates only `checkpoint_last.pt`. `FAIL_DIVERGED` stops collection, flushes
diagnostics, and creates only `checkpoint_last.pt`. All paths close the pool
and replay; the persistent replay is committed atomically at terminal episode
boundaries, so an in-flight episode is not a resume duplicate.

The transition cap is checked before a parallel step wave so a worker wave does
not cross the configured maximum. A pending in-flight episode is discarded at
that boundary. Checkpoints persist the mission source identity, source order
progress, split, worker topology, replay metadata identity, observed runtime
IDs, gate history, environment steps, completed episodes, and critic updates.
Resume requires exact source/split/worker/replay identity and a pending gate;
missing split/replay or changed identity fails closed.

## Replay provenance

The online replay metadata contains the existing contract fields plus the
mission source SHA needed for strict resume identity. Its run contract binds:

```text
BC checkpoint SHA
mission source path/SHA and ordered IDs
Task Contract SHA
Observation Contract
MPL Contract SHA
AWAC training contract SHA
Reward Contract SHA and single scale owner
Replay Contract SHA
selected worker topology and runtime IDs
```

The policy input remains `depth_goal_state_prev_action`; no global map, global
route, waypoint, collision voxel map, teacher cost, or future expert path is
introduced into Actor, Critic, or replay fields. Existing replay audit remains
the owner of the zero-privileged-field check.

## Files changed

```text
python/planning/awac/calibration_runtime.py
python/planning/awac/trainer.py
python/planning/awac/calibration.py
python/planning/awac/checkpoint.py
python/planning/awac/replay.py
tests/test_awac_formal_calibration_runtime.py
docs/RL_PHASE_1_FORMAL_REPLAY_PRODUCER_AND_CALIBRATION_LOOP_V1.md
docs/rl_phase_1_formal_replay_producer_and_calibration_loop_v1.json
```

The frozen BC model/data, Unity, Bridge, Task Contract, Observation Contract,
Reward Contract, MPL, trajectory semantics, and AWAC Actor-enabled phase were
not changed. No commit was created.

## Verification boundary

The focused AWAC contract set is run with the `xm` conda environment. The
focused AWAC set passed with 41 tests, the naming-contract test passed, and
`python -m compileall -q python/planning` passed. The full suite completed with
720 passed, 42 skipped, and 8 failures. All 8 failures are socket creation
`PermissionError: [Errno 1] Operation not permitted` failures in the P0-M2
runner tests; no code-eligible test failure remains. These host capability
failures are not converted into runtime evidence. The recorded result is
maintained in the machine-readable report beside this document.

## Host commands

Run these only on a socket-enabled XM host after the required managed
ROS/Unity/Bridge supervisor is available. Every Python command below is
intended to run after activating `xm`.

### HOST_STEP_1_BC_DEV100

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
scripts/evaluate_policy_unity_managed.sh \
  data/teach/2026_6w/bc_training/checkpoint_best_soft.pt \
  data/test/awac_dev_seed4026_100/missions.csv \
  data/test/awac_dev_seed4026_100/bc60k_baseline \
  100
```

This uses the existing managed evaluator wrapper and does not read the
Final300 artifact.

### HOST_STEP_2_FORMAL_CRITIC_CALIBRATION

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
python scripts/train_awac.py \
  --phase critic_calibration \
  --bc-checkpoint data/teach/2026_6w/bc_training/checkpoint_best_soft.pt \
  --train-index data/teach/2026_6w/missions.csv \
  --out-dir data/awac/awac_bc60k_formal_critic_calibration_v1 \
  --env-workers 2 \
  --worker-spec-file data/teach/2026_6w/rollouts/worker_runtime_specs.json \
  --reliable-v4 \
  --mpl-contract-sha256 f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d \
  --max-steps 45 \
  --total-env-steps 30000 \
  --calibration-max-transitions 30000 \
  --calibration-max-episodes 1000 \
  --actor-depth-lr 1e-6 \
  --critic-depth-lr 1e-5 \
  --device auto
```

The command intentionally omits `--replay-dir`, so the canonical fresh-run
owner creates `data/awac/awac_bc60k_formal_critic_calibration_v1/replay`.
The parser defaults supply the existing batch, optimizer, gamma, tau, CQL,
depth-mask, and gate configuration; no Actor update flag is enabled in this
phase.

### HOST_STEP_3_REPLAY_AUDIT

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
python scripts/audit_awac_replay.py \
  --replay-dir data/awac/awac_bc60k_formal_critic_calibration_v1/replay \
  --output data/awac/awac_bc60k_formal_critic_calibration_v1/replay_audit.json
```

### HOST_STEP_4_CALIBRATION_RESUME_DRY_RUN

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
OUT=data/awac/awac_bc60k_formal_critic_calibration_v1
test -f "$OUT/checkpoint_calibration_pass.pt" || { echo "checkpoint_calibration_pass.pt is required" >&2; exit 1; }
python scripts/train_awac.py \
  --phase critic_calibration \
  --bc-checkpoint data/teach/2026_6w/bc_training/checkpoint_best_soft.pt \
  --train-index data/teach/2026_6w/missions.csv \
  --out-dir "$OUT" \
  --resume-checkpoint "$OUT/checkpoint_calibration_pass.pt" \
  --env-workers 2 \
  --worker-spec-file data/teach/2026_6w/rollouts/worker_runtime_specs.json \
  --reliable-v4 \
  --mpl-contract-sha256 f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d \
  --max-steps 45 \
  --total-env-steps 30000 \
  --calibration-max-transitions 30000 \
  --calibration-max-episodes 1000 \
  --actor-depth-lr 1e-6 \
  --critic-depth-lr 1e-5 \
  --device auto \
  --dry-run
```

The guard and `--dry-run` perform checkpoint/source construction validation
without interaction, training, or Actor updates.

## Next action

```text
RUN_GENERATED_HOST_DEV100_AND_FORMAL_CALIBRATION_COMMANDS
```

This document does not authorize running those commands in the current Codex
environment.
