# RL Phase 1 calibration checkpoint identity contract fix V1

Status: `PASS` for code and offline contract validation. No formal runtime was
started in this task.

## Scope

The repair shares invariant AWAC checkpoint identity construction between
standard AWAC and Critic Calibration. It does not change the masked-discrete
AWAC objective, Actor/Critic or optimizer state, replay behavior, calibration
gate, runtime lifecycle, Unity, Bridge, Task Contract, Observation Contract,
reward contract, or MPL.

```text
algorithm owner=planning.awac.contract.AWAC_ALGORITHM_ID
algorithm_id=discrete_masked_awac
phase=critic_calibration
```

`phase` remains a training-phase field, not an alternate algorithm identity.

## Root cause and repair

The reported exception proves that the checkpoint producer active at that time
passed a payload without a valid `algorithm_id` to the strict validator. The
surviving V5 artifact cannot identify that historical save call: it has no
`checkpoint_save_reason`, no retained Python traceback, and its surviving
`checkpoint_last.pt` already contains the canonical algorithm ID.

The durable defect was duplicated manual identity assembly in standard and
calibration producers. The shared producer is now:

```text
planning.awac.checkpoint.build_awac_checkpoint_identity
```

Both `trainer.build_checkpoint_payload` and
`calibration.build_calibration_checkpoint_payload` use it. The latter feeds
online periodic/last saves, non-PASS final last checkpoints, synthetic
calibration checkpoints, and the PASS-only builder. Validators remain strict:
missing, `None`, and wrong algorithm IDs fail closed.

## Identity contract

The shared invariant builder contains 19 fields:

```text
awac_checkpoint_schema_id
algorithm_id
awac_checkpoint_contract_id
model_type
training_config_contract_id
software_version
feature_contract_id
policy_input_contract_sha256
policy_runtime_contract_id
task_contract_id
task_contract_sha256
reward_contract_id
reward_contract_sha256
observation_contract
observation_source
vec_dim
num_actions
depth_history_frames
initial_prev_action
```

Calibration additionally has strict phase-bound identity requirements: MPL
SHA, resolved configuration and SHA, training contract and SHA, calibration
split and SHA, BC source SHA/fingerprint, replay identity and contract SHA,
gate status/metrics, reward scale/owner, model-state fingerprints, and, for
online runs, mission/progress/runtime identities. The valid payload has zero
missing required identity fields.

## Checkpoint kinds

| Checkpoint kind | Producer | Identity |
| --- | --- | --- |
| periodic online progress | `persist_checkpoint` to `checkpoint_last.pt` | PASS |
| final non-PASS | normal calibration builder to `checkpoint_last.pt` | PASS |
| PASS-only artifact | pass builder to `checkpoint_calibration_pass.pt` | PASS |
| offline/synthetic | normal calibration builder to `checkpoint_calibration.pt` | PASS |

`checkpoint_calibration_pass.pt` remains `PASS_ONLY`; the phase state machine
is unchanged and the normal checkpoint retains `phase=critic_calibration`.

## V5 read-only audit

```text
run=data/awac/awac_bc60k_formal_critic_calibration_v5
checkpoint=checkpoint_last.pt
checkpoint_validator=PASS
algorithm_id=discrete_masked_awac
replay_transitions=95
checkpoint_replay_size=95
completed_episodes=5
critic_updates=0
holdout_episodes=2
calibration_gate_state=FAIL_DIVERGED
checkpoint_save_reason=ABSENT
failed_checkpoint_kind=UNKNOWN_NO_PERSISTED_SAVE_REASON
first_real_critic_training_reached=NO
resumable=NO
```

V5 is non-resumable because exact resume requires `PENDING`, while V5 is
`FAIL_DIVERGED`. It must not be patched or resumed. The fresh formal target is
`data/awac/awac_bc60k_formal_critic_calibration_v6`.

## Verification

Focused tests cover shared standard/calibration identity, periodic/last
round-trip, PASS-only round-trip, missing/wrong algorithm ID rejection, wrong
immutable contract identity rejection, synthetic resume dry-run, and naming
contract compatibility for the existing schema owner.

```text
compileall=PASS
focused=35 passed, 1 skipped (CUDA capability gate)
full_pytest=752 passed, 43 skipped, 8 environment failures
code_eligible_failures=0
```

All eight full-suite failures are sandbox
`PermissionError: [Errno 1] Operation not permitted` when tests create a TCP
socket. They occur before Unity/Bridge behavior and do not exercise this
checkpoint identity change.

## Fresh V6 host command

Run only in a normal XM host terminal. It deliberately has no
`--resume-checkpoint`, no existing replay path, and no `--calibration-split`;
the fresh path creates the deterministic split inside the V6 directory.

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
  --out-dir data/awac/awac_bc60k_formal_critic_calibration_v6 \
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

The command uses real current CLI flags and starts Critic Calibration only; it
does not enable Actor updates.

