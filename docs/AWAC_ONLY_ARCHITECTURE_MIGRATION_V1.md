# AWAC-only architecture migration V1

Status: `PASS`  
Scope: `/home/xm/XM/xm_ws/src/planning` only  
Date: 2026-09-03

This migration makes `planning.awac` the only formal RL owner. Unity, Bridge,
Task V2, the reliable exact-endpoint observation contract, BC artifacts, and
trajectory semantics were not changed. No Unity process, Bridge process,
formal collection, long AWAC training, or final 300-mission evaluation was
run.

## Current owners

| Responsibility | Owner |
|---|---|
| Actor and Critic construction | `planning.awac.model` |
| AWAC learner and optimization | `planning.awac.learner`, `planning.awac.optimization` |
| Replay | `planning.awac.replay` |
| Replay audit | `planning.awac.replay_audit`, `scripts/audit_awac_replay.py` |
| Runtime worker pool | `planning.runtime.parallel_env` |
| Checkpoint persistence | `planning.common.checkpoint` + `planning.awac.checkpoint` |
| Evaluation | `planning.evaluation.policy_evaluator` |
| Mission split/holdout | `planning.evaluation.mission_split` |
| Reward arithmetic contract | `planning.contracts.reward` |
| Formal trainer | `planning.awac.trainer`, `scripts/train_awac.py` |
| Schedule | `planning.awac.schedule` |
| Future confidence seam | `planning.awac.confidence` (typed placeholder only) |
| Future primitive-neighbor seam | `planning.primitives.neighbors` (typed placeholder only) |

## Structural result

Moved or re-owned production surfaces:

- `rl/sac_model.py` -> `awac/model.py`;
- `rl/sac_parallel_env.py` -> `runtime/parallel_env.py`;
- `rl/sac_mission_split.py` -> `evaluation/mission_split.py`;
- `rl/guarded_schedule.py` -> `awac/schedule.py`;
- `rl/sac_dev_selection.py` -> `awac/dev_selection.py`;
- `diagnostics/sac_run.py` -> `diagnostics/awac_run.py`;
- `common/runtime_accounting.py` -> `diagnostics/runtime_accounting.py`;
- SAC-named mission-split, holdout, and dev-selection scripts -> AWAC/generic
  script names.

Added AWAC-specific model, optimization, interaction, replay, replay-audit,
checkpoint, contract, trainer, and typed future-owner modules. The old
`python/planning/rl/` package and old SAC-only learner/replay/candidate,
rollback, guard, audit, trainer, and script surfaces were removed after the
caller/test/CMake zero-reference gate.

`CMakeLists.txt` installs `train_awac.py`, `audit_awac_replay.py`, the AWAC
holdout/split/select tools, and no AWAC/SAC runner shell. The reliable-v4
behavior is selected directly with the Python `--reliable-v4` option. A
managed evaluation shell remains separately installed because it owns the
ROS/Unity/Bridge lifecycle. `package.xml` describes the package as
BC-initialized discrete AWAC.

## Behavior and contract preservation

- Actor input remains `depth + 127-dimensional policy vector`, with 105
  actions and the existing depth action mask.
- Critic input is the same policy state plus action; no global map, route,
  teacher cost, or future expert path is admitted.
- Twin-Q, masked Bellman expectation, AWAC advantage weighting, CQL helper,
  and existing BC-KL protection remain the current math. Confidence,
  adaptive beta, critic calibration, and primitive-neighbor exploration are
  not implemented in this phase.
- AWAC checkpoint schema is `awac_checkpoint_schema_v2`; legacy entropy state
  is rejected rather than interpreted.
- Replay contains only the ten formal transition fields and rejects the old
  candidate-funnel fields.
- Reward arithmetic is owned by `planning.contracts.reward` and called by
  `runtime.unity_env` without changing numeric values or terminal precedence.
  Canonical artifact: [`reward_contract.json`](reward_contract.json).
- BC handoff strictly validates `reliable_exact_endpoint_snapshot`, Task V2,
  feature contract, 127/105 dimensions, and exact Actor state loading.

## Verification

Static gates:

```text
RL_DIRECTORY_EXISTS=NO
SAC_PRODUCTION_FILE_COUNT=0
RL_DIRECTORY_PRODUCTION_FILE_COUNT=0
AWAC_TO_RL_IMPORT_COUNT=0
AWAC_TO_SAC_IMPORT_COUNT=0
SAC_ALPHA_LOGIC_COUNT=0
SAC_ENTROPY_OBJECTIVE_COUNT=0
LEGACY_CANDIDATE_FUNNEL_EXISTS=NO
LEGACY_TRUST_GATE_EXISTS=NO
LEGACY_ROLLBACK_EXISTS=NO
```

Bounded, offline checks:

```text
BC_AWAC_INITIALIZATION_PARITY=PASS
AWAC_CHECKPOINT_PARSE=PASS
AWAC_SYNTHETIC_SMOKE=PASS
REWARD_NUMERIC_PARITY=PASS
FROZEN_FINAL_TEST_300_MODIFIED=NO
```

The actual checks were run after `conda activate xm` with:

```bash
python -m compileall -q python/planning scripts
python tests/contracts/test_contract_naming.py
python -m pytest -q
```

Full result: `680 passed, 42 skipped, 0 failed`. The optional skips retain
their pre-existing ROS/Unity/data prerequisites; no skip was added to hide a
migration failure. Native library/socket-dependent tests used freshly built
test libraries under `/tmp/xm-awac-cxx-build`; no source C++ file was changed.

## Deferred work and blockers

`AWAC_DEV_SET_STATUS=MISSING`. The frozen BC 300-mission set remains
`FINAL_TEST_ONLY` and was not used for AWAC selection. Phase 0 must separately
define the AWAC contract and critic calibration before any formal AWAC
training. Confidence and primitive-neighbor algorithms remain explicitly
unimplemented. CUDA/AWAC long-run and final evaluation evidence are outside
this migration.

`COMMIT=NO`.

Next action: `RL_PHASE_0_AWAC_CONTRACT_AND_CRITIC_CALIBRATION_IMPLEMENTATION`.
