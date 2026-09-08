# RL Phase 1 Managed Calibration Runtime Lifecycle V1

## Result

The managed lifecycle seam is implemented and verified with fake external
processes.  This Codex environment cannot create sockets, so no ROS master,
Unity Player, Bridge, or formal calibration run was started.

```text
RL_PHASE_1_MANAGED_CALIBRATION_RUNTIME_LIFECYCLE_V1=PASS
MANAGED_CALIBRATION_STARTUP_SMOKE=PASS
MANAGED_RUNTIME_LIFECYCLE_IMPLEMENTED=YES
RUNTIME_EXECUTED=NO
```

The PASS is a code and mock-lifecycle result.  It is not a claim that formal
critic calibration completed.

## Root cause and ownership

The failed v1 host attempt reached `CalibrationReplayProducer` and then
constructed `ParallelEnvPool`, but no owner had started the per-worker ROS
masters, Unity processes, or Bridge processes.  The Python workers therefore
timed out while registering with `127.0.0.1:11621` and `127.0.0.1:11622`.

The old v1 output is retained as `FAILED_STARTUP_ATTEMPT`: its replay metadata
has `size=0`, `total_added=0`, and `reliable_v4_transition_count=0`; no critic
update or checkpoint was produced.  It is not a resumable calibration run.

| Responsibility | Canonical owner |
| --- | --- |
| DEV100 managed runtime | `planning.runtime.managed_runtime.ManagedRuntimePool` |
| ROS master launch | `ManagedRuntimePool._start_worker_runtime` / `_launch` |
| Unity launch | `ManagedRuntimePool._start_worker_runtime` / `_unity_argv` |
| Bridge launch | `ManagedRuntimePool._start_worker_runtime` / `_bridge_argv` |
| Port resolution and validation | `planning.runtime.ports.WorkerRuntimeSpec` and `planning.runtime.ports` |
| Process cleanup | `ManagedRuntimePool.close` / `_terminate_process_group` |
| Python environment workers | `planning.runtime.parallel_env.ParallelEnvPool` |

`evaluate_policy_unity_managed.sh` now creates the canonical one-worker spec
and delegates process lifecycle to `scripts/run_managed_runtime.py`.  The
runner is a thin entrypoint to the same `ManagedRuntimePool`; it does not
duplicate evaluator or AWAC logic.  Formal calibration uses the same owner
directly from `planning.awac.trainer`.

## Worker and startup contract

`data/teach/2026_6w/rollouts/worker_runtime_specs.json` is schema
`p3_worker_runtime_spec_v2` with 20 workers.  The selected calibration workers
are the first two canonical specs:

```text
worker 0: ROS_MASTER_URI=http://127.0.0.1:11621,
          runtime_instance_id=xmflight-2026_6w-2026_6w_r1-w00,
          ROS_HOME=.../rollouts/ros/worker_00
worker 1: ROS_MASTER_URI=http://127.0.0.1:11622,
          runtime_instance_id=xmflight-2026_6w-2026_6w_r1-w01,
          ROS_HOME=.../rollouts/ros/worker_01
```

The canonical port projection supplies distinct command, state, depth,
reliable command/result, and snapshot endpoints.  The current launch
namespace is `/`; isolation is provided by the distinct ROS master, ROS_HOME,
runtime identity, and endpoint set.  No parent-process environment mutation
is used for worker isolation.

For each selected worker the persisted runtime identity includes:

```text
worker_id
runtime_instance_id
ROS_MASTER_URI
Unity executable path and SHA256
Bridge executable path and SHA256
Task Contract SHA256
Observation Contract
MPL Contract SHA256
```

Startup is ordered as:

```text
canonical worker spec validation and port preflight
→ ROS master launch
→ rosparam list readiness
→ Unity launch
→ Bridge launch through roslaunch planning sim_realtime.launch
→ /xm/state and /xm/depth/image_raw readiness
→ ParallelEnvPool construction/startup handshake
```

Startup failure closes every process already owned by the run.  Normal exit,
gate PASS, pending cap, divergence, worker failure, exception, and
`KeyboardInterrupt` all use the same owned-process cleanup boundary.  The
implementation never invokes global `pkill`; default children are launched in
their own process groups and only those groups are terminated.

## Calibration and resume behavior

The formal command selector remains:

```text
phase=critic_calibration + train_index + no resume checkpoint
```

When this selector is used, the trainer starts `ManagedRuntimePool` before
constructing `ParallelEnvPool`.  `--env-workers 2` therefore selects two
managed runtime instances and two Python workers.  Existing reliable-v4
observation, depth-mask, mission, replay, reward, critic, and calibration-gate
contracts remain unchanged.

The managed runtime identity is nested in the existing calibration checkpoint
runtime identity for resume validation.  A missing or changed managed identity
fails closed.  No calibration algorithm, Actor update, BC checkpoint content,
replay transition layout, Task Contract, Observation Contract, Reward Contract,
or MPL was changed.

The next host attempt must use a new output lineage:
`data/awac/awac_bc60k_formal_critic_calibration_v2`.  It must not silently
resume v1.

## Verification

```text
compileall=PASS
managed/AWAC focused tests=27 passed
full pytest=730 passed, 42 skipped, 8 failed
code-eligible failures=0
```

The eight full-suite failures are host-capability failures while creating a
socket (`PermissionError: [Errno 1] Operation not permitted`) in the existing
P0-M2 runner/ZMQ tests.  They are not managed-runtime implementation failures
and are not converted into runtime evidence.

The focused mock tests cover two workers, launch order, isolated environment
variables, per-worker identities, occupied-port fail-closed behavior, partial
startup cleanup, idempotent normal cleanup, caller cleanup, and
`KeyboardInterrupt` cleanup.  The evaluator delegation and trainer startup
ordering tests also pass.

## Files changed

```text
python/planning/runtime/managed_runtime.py
python/planning/awac/trainer.py
scripts/run_managed_runtime.py
scripts/evaluate_policy_unity_managed.sh
tests/test_managed_runtime_lifecycle.py
tests/test_managed_runtime_integration.py
tests/contracts/test_contract_naming.py
docs/RL_PHASE_1_MANAGED_CALIBRATION_RUNTIME_LIFECYCLE_V1.md
docs/rl_phase_1_managed_calibration_runtime_lifecycle_v1.json
```

## Host command

Run only on a normal socket-enabled XM host.  The command below starts the
managed lifecycle automatically; do not manually start `roscore`, Unity, or
Bridge.  It is deliberately recorded but was not executed by Codex.

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
  --out-dir data/awac/awac_bc60k_formal_critic_calibration_v2 \
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

This command performs critic-only calibration.  Actor updates, confidence,
adaptive BC KL, primitive-neighbor logic, AWAC online selection, and subsequent
phases are not enabled.

