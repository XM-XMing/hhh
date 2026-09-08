# PY4-A Python production surface and symbol audit

> STATUS: HISTORICAL / SUPERSEDED_BY_SHELL_ENTRYPOINT_CLEANUP_AND_TEST_SUITE_INTEGRITY_V1.
> This report retains the earlier production-surface evidence and shell counts;
> it is not the current command or install manifest.

Date: 2026-08-28

This is a structural audit of the optimized Planning tree. The authoritative
behavior tree `O=/home/xm/XM/xm_ws/src/planning` was read-only. The only source
change in this slice is the P collection shell bootstrap at
`scripts/collect_rollouts_parallel.sh`; no algorithm, artifact schema, Unity,
C++, or formal collection data was changed.

## Gate result

```text
PY4_A_PRODUCTION_SURFACE_AUDIT=PASS
CANONICAL_PRE_BC_ENTRY_COUNT=11
DUPLICATE_CANONICAL_ENTRY_COUNT=0
PRODUCTION_PYTHON_MODULE_COUNT=132
PRODUCTION_SCRIPT_COUNT=28
INSTALLED_TEST_FILE_COUNT=0
INSTALLED_DIAGNOSTIC_FILE_COUNT=26
DUPLICATE_COMMON_CAPABILITY_COUNT=3
DUPLICATE_CONFIG_OWNER_COUNT=13
PURE_REEXPORT_UNUSED_COUNT=0
TEST_ONLY_PRODUCTION_SYMBOL_COUNT=212
AWAC_TO_SAC_IMPORT_COUNT=3
RL_FORMAL_ENTRY_COUNT=10
RL_TEMPORARY_DEPENDENCY_COUNT=3
FORMAL_MPL_READY=NO
WORKSPACE_WRAPPER_RESOLUTION=PASS
GENERATED_FILE_COUNT_BEFORE=56
GENERATED_FILE_COUNT_AFTER=56
SAFE_SYMBOL_DELETION_COUNT=0
SAFE_FILE_DELETION_COUNT=0
SAFE_DELETION_BATCH=[]
SCOPE_EXPANDED=NO
PRE_BC_BEHAVIOR_CHANGED=NO
ARTIFACT_SCHEMA_CHANGED=NO
ALGORITHM_CHANGED=NO
```

`PRODUCTION_PYTHON_MODULE_COUNT` counts the 132 Python files under
`python/planning`, including package `__init__.py` files, discovered by
`setup.py` as 16 packages. `PRODUCTION_SCRIPT_COUNT` is the CMake install
surface: 24 Python scripts plus 4 shell scripts. CMake installs `config/`
(2 files), `launch/` (8 files), and `rviz/` (3 files); it does not install
`tests/` or `docs/`.

`INSTALLED_DIAGNOSTIC_FILE_COUNT=26` is explicitly defined as the 20 modules
under `planning.diagnostics` plus these 6 installed diagnostic/reference CLIs:
`visualize_motion_primitives.py`, `collision_checker.py`,
`check_base_system.py`, `check_depth_correctness.py`,
`check_unity_depth_profile.py`, and `visualize_rollout.py`. They are retained
as named debug/reference seams, not counted as test files. The frozen
`planning.awac`, `planning.rl`, and checkpoint helpers remain installed and
protected by policy.

## Canonical pre-BC entrypoints

Every row below is a 7-line executable adapter with zero business-logic lines.
The domain module is the sole owner of the stage behavior. All are installed
to `${CATKIN_PACKAGE_BIN_DESTINATION}` by `catkin_install_python`.

| Entry | Domain owner | Script LOC | Business LOC | Other wrapper | Status |
|---|---|---:|---:|---|---|
| `generate_motion_primitives.py` | `planning.primitives.generator` | 7 | 0 | none | `THIN_CANONICAL_ENTRY` |
| `generate_missions.py` | `planning.mission.generator` | 7 | 0 | none | `THIN_CANONICAL_ENTRY` |
| `audit_teacher_missions.py` | `planning.mission.auditor` | 7 | 0 | none | `THIN_CANONICAL_ENTRY` |
| `collect_teacher_rollouts.py` | `planning.teacher.rollout_collector` | 7 | 0 | none | `THIN_CANONICAL_ENTRY` |
| `collect_rollouts_parallel.py` | `planning.teacher.parallel_collection` | 7 | 0 | `collect_rollouts_parallel.sh` compatibility bootstrap | `THIN_CANONICAL_ENTRY` |
| `label_teacher_rollouts.py` | `planning.teacher.labeling` | 7 | 0 | none | `THIN_CANONICAL_ENTRY` |
| `audit_teacher_dataset.py` | `planning.teacher.dataset_audit` | 7 | 0 | none | `THIN_CANONICAL_ENTRY` |
| `generate_depth_action_masks.py` | `planning.safety.depth_action_masks` | 7 | 0 | none | `THIN_CANONICAL_ENTRY` |
| `build_bc_mmap_dataset.py` | `planning.data.bc_mmap` | 7 | 0 | none | `THIN_CANONICAL_ENTRY` |
| `train_soft_bc.py` | `planning.bc.trainer` | 7 | 0 | none | `THIN_CANONICAL_ENTRY` |
| `evaluate_policy_unity.py` | `planning.evaluation.policy_evaluator` | 7 | 0 | `evaluate_policy_unity_managed.sh` managed lifecycle | `THIN_CANONICAL_ENTRY` |

The repeated `main` names are domain-local entry functions, not duplicate
canonical entries. The two shell files listed as other wrappers have distinct
bootstrap/managed-lifecycle ownership; they do not duplicate the Python
stage's business logic.

## Collection wrapper resolution

The former wrapper inferred `SCRIPT_DIR/../..` as a workspace and sourced
catkin setup with the caller's CLI arguments. From P this could select O, and
`--help` could be interpreted as a setup option and fail while sourcing a
temporary setup file.

The corrected wrapper remains bootstrap plus `exec` only. It:

- detects a source-tree package only when its adjacent `package.xml` exists;
- selects an explicit `${WORKSPACE}/devel/setup.bash`, then a devel/install
  prefix setup adjacent to the installed script, then the ROS setup;
- clears positional arguments only while sourcing setup and restores them;
- prepends P's source package and ROS package path when running from P;
- contains no port derivation, process management, worker management, merge,
  or collection policy logic.

The wrapper is 9 nonblank lines. The `--help` matrix passed from source P,
temporary catkin-style devel and install layouts, and `/tmp` as arbitrary cwd;
an explicit `WORKSPACE=/home/xm/XM/xm_ws` source-tree run also passed. No
collection process was started.

## Symbol graph and duplicate qualification

The AST/static graph is recorded at `/tmp/xm-py4-a-symbol-graph.json`:

```text
symbols=1197
KEEP_PRODUCTION=526
KEEP_PUBLIC_API=85
KEEP_RL_DEPENDENCY=240
KEEP_TEST_REFERENCE=212
DEFER_DYNAMIC=134
same-name-groups=28
exact-duplicate-body-groups=1
```

The one exact duplicate body is `_update_bytes` in
`planning.contracts.policy_checkpoint_fingerprint` and
`planning.rl.p3_checkpoint_evidence`. It is a protected RL/checkpoint seam
and is not a PY4-A deletion or merge candidate. The other 27 same-name groups
are local helpers, diagnostics, or separate domain `main` functions; names
alone do not establish duplicate behavior.

The 212 test-only static candidates are not a deletion count. They include
formal debug/reference, protocol qualification, and diagnostic seams that are
inside the production package and are still referenced by tests or evidence.
Dynamic/public use, pickle/checkpoint loading, and public exports are
conservatively represented by `DEFER_DYNAMIC=134`.

`planning.common.atomic` is a compatibility re-export of
`planning.common.io.write_json_atomic`, but it is used by frozen RL/SAC code
and guarded shell snippets. Therefore `PURE_REEXPORT_UNUSED_COUNT=0`.

## Common capability audit

The count `DUPLICATE_COMMON_CAPABILITY_COUNT=3` counts only proven
cross-module mechanical patterns, not every textual occurrence:

| Capability | Canonical owner | Other implementation/adapter | Decision |
|---|---|---|---|
| File SHA-256 | `planning.common.hashing.file_sha256` | `runtime.identity._sha256`, diagnostic `_sha256`/`_sha256_file`; `data.bc_mmap._sha256` delegates to the owner | merge candidate, parity required; unchanged |
| Canonical JSON SHA-256 | contract-local owners; no single safe common owner | `contracts/pipeline_provenance.py`, `bc.trainer`, `mission.spec`, `contracts.feature`, and other contract modules | encoding/contract differences require parity; unchanged |
| Atomic JSON replace | `planning.common.io.write_json_atomic` | `runtime.lifecycle._write_json_atomic`, protected RL checkpoint writers, and the compatibility re-export | lifecycle/checkpoint semantics require parity; unchanged |

CSV/NPZ rollout persistence, artifact-relative path resolution, runtime port
profiles, process supervision, timestamps, and manifest/provenance serializers
are domain or lifecycle seams rather than proven duplicate implementations.
The audit did not put business algorithms into `common/`, and no helper was
merged in this slice.

## Configuration owner audit

The following values are currently resolved without changing the PY3 business
values. `DUPLICATE_CONFIG_OWNER_COUNT=13` counts requested parameter families
with more than one independent default declaration. Stage-specific worker
pools are included because they are separate defaults and therefore remain a
configuration-parity risk; they are not silently treated as one global pool.

| Parameter family | Current owners/defaults | YAML value | Runtime value in PY3 | Finding |
|---|---|---|---|---|
| `voxel_size` | mission/teacher CLIs, collision config, environment collection config | none | `0.10` | duplicate default family; parity needed |
| `inflate_radius` | `config/motion_primitives.yaml`, mission/teacher/collision/evaluation defaults | `0.35` | `0.35` | duplicate default family; parity needed |
| `max_steps` | `mission.spec.DEFAULT_MAX_PRIMITIVE_STEPS`, collector/audit/evaluation CLIs, collection environment default | none | `45` for teacher stages | duplicate default family; parity needed |
| `beam_depth` | `TeacherConfig`, shared Teacher CLI argument | none | `3` | duplicate literal; parity needed |
| `beam_width` | `TeacherConfig`, shared Teacher CLI argument | none | `8` | duplicate literal; parity needed |
| `beam_branching` | `TeacherConfig`, shared Teacher CLI argument | none | `4` | duplicate literal; parity needed |
| `beam_discount` | `TeacherConfig`, shared Teacher CLI argument | none | `0.95` | duplicate literal; parity needed |
| `worker_count` | parallel supervisor (`12`), hardware profile (`12`/`8`), mission/audit/label/mask stage defaults | none | stage-specific | separate pools; parity needed |
| depth collision radius | `EnvConfig`, `DepthSafetyConfig`, depth-mask CLI | none | `0.40` | duplicate default family; parity needed |
| depth slack | `EnvConfig`, `DepthSafetyConfig`, depth-mask CLI | none | `0.08` | duplicate default family; parity needed |
| depth sample stride | `EnvConfig`, `DepthSafetyConfig`, depth-mask CLI | none | `4` | duplicate default family; parity needed |
| patch radius | `EnvConfig`, `DepthSafetyConfig`, depth-mask CLI | none | `2` | duplicate default family; parity needed |
| max patch radius | `EnvConfig`, `DepthSafetyConfig`, depth-mask CLI | none | `14` | duplicate default family; parity needed |

BC hyperparameters remain owned by `planning.bc.trainer` for this audit; RL
hyperparameters are a frozen boundary and are not candidates for unification.
No config default was mechanically changed.

## Frozen RL boundary

There are 3 direct AST import edges from `planning.awac` to `planning.rl`:
`sac_candidate_interaction`, `sac_learner`, and `sac_model` from
`planning.awac.learner`. There are 10 formal source entry files in the frozen
RL set, and 3 temporary shared contract dependencies:

```text
python/planning/contracts/awac_handoff.py
python/planning/contracts/offpolicy.py
python/planning/contracts/policy_checkpoint_fingerprint.py
```

No AWAC/SAC import, checkpoint, resume, or runner was modified or executed.

## Formal MPL policy

```text
FORMAL_MPL_ARTIFACT_POLICY=
  config/motion_primitives.yaml is the source contract; the canonical
  generate_motion_primitives.py command is the sole producer; CMake installs
  data/motion_primitives only when it already exists; runtime loading fails
  closed when the NPZ/metadata pair is absent or inconsistent; no build or
  collection step fabricates formal data.
FORMAL_MPL_READY=NO
```

The formal P files are currently absent:
`data/motion_primitives/motion_primitives_105.npz` and
`data/motion_primitives/motion_primitives_105.json`. The reproducible command
is recorded but was not run:

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
PYTHONPATH=/home/xm/XM/src/python \
  python3 /home/xm/XM/src/scripts/generate_motion_primitives.py \
  --config /home/xm/XM/src/config/motion_primitives.yaml
```

The PY3 MPL remains under `/tmp/xm-pre-bc-py3/`; it was not copied into P.

## Generated files and deletion gates

Before and after validation, P contained 56 confirmed `.pyc` files under 13
`__pycache__` directories. No P `.pytest_cache`, `.ruff_cache`, `.mypy_cache`,
`build/`, `devel/`, or `install/` directory was present. Caches were left in
place in this audit; no source, fixture, report, or documentation was removed.

Every deletion gate therefore remains closed:

```text
SAFE_SYMBOL_DELETION_COUNT=0
SAFE_FILE_DELETION_COUNT=0
SAFE_DELETION_BATCH=[]
```

The existing file-level inventory remains authoritative: KEEP_PRE_BC=111,
KEEP_RL=36, KEEP_RL_TEMPORARY_DEPENDENCY=3, KEEP_TEST_REFERENCE=223, with no
proven duplicate, legacy-formal, dead, or unknown deletion candidate.

## Validation evidence

```text
compileall=PASS
PY3 focused + collection architecture=87 passed
canonical CLI help=11/11 passed
collection wrapper source/devel/install/arbitrary-cwd=PASS
O/P relabel-mask comparator=PASS
ARTIFACT_SHA_CHAIN=/tmp/xm-pre-bc-py3/sha_chain.json status=PASS
```

The O/P fixture comparator reported exact array hashes for soft targets,
global masks, teacher argmax, behavior actions, valid counts, entropies, and
depth masks, plus exact episode/transition ordering and business metadata.
The wrapper and structural changes did not alter those artifacts.

An additional non-gating `tests/test_script_contracts.py` invocation reported
two existing/environment-bound failures: the numeric-suffix naming checker
sees frozen versioned contracts, and the standalone global-route contract
needs a native library not present in that invocation. Neither test is part
of the PY3 focused gate, neither failure was caused by the wrapper change, and
no unrelated code was changed to mask them.

## Next phase

```text
NEXT_PHASE=PY5 P CLEAN BUILD AND O CUTOVER PREPARATION
```

This document stops at PY4-A. PY4-B cleanup, formal MPL generation, formal
60,000-row collection, AWAC/SAC, and O cutover were not executed.

## Current live-tree consolidation override (2026-08-29)

The preceding sections are the historical PY4-A inventory and retain their
original counts.  The current live source surface was subsequently
consolidated without changing Planning algorithms, Unity, C++, wire behavior,
task semantics, or RL code:

```text
CURRENT_PRODUCTION_SCRIPT_COUNT_BEFORE=28
CURRENT_PRODUCTION_SCRIPT_COUNT_AFTER=21
CURRENT_INSTALLED_DIAGNOSTIC_SCRIPT_COUNT_AFTER=0
CURRENT_MISSION_ROUTE_STORE_IMPLEMENTATION_COUNT=1
CURRENT_MISSION_ROUTE_STORE_COMPAT_WRAPPER_COUNT=0
CURRENT_ZERO_CALLER_PRODUCTION_WRAPPER_COUNT=0
CURRENT_DELETED_FILE_COUNT=2
CURRENT_MOVED_DIAGNOSTIC_FILE_COUNT=13
CURRENT_FORMAL_MISSION_PREPARATION_ENTRY_COUNT=1
CURRENT_MISSION_PREPARATION_INTEGRATION=PASS
CURRENT_PREPARATION_BUSINESS_PARITY=PASS
CURRENT_ROUTE_STORE_MEMORY_GATE=PASS
CURRENT_FULL_TESTS=800 passed, 42 skipped, 0 failed
CURRENT_ARTIFACT_SCHEMA_CHANGED=YES
CURRENT_ALGORITHM_CHANGED=NO
```

The production-script count is the CMake install surface, not the number of
files physically remaining in `scripts/`.  The current install list has 17
Python entries and 4 shell entries.  RL entries remain protected, while the
source-only internal collector worker is retained for its formal caller.  The
two removed files were `scripts/generate_missions.py` and
`scripts/audit_teacher_missions.py`; their behavior is now orchestrated by
the single `scripts/prepare_teacher_missions.py` entry.  Thirteen diagnostic
and reference modules were moved to `tools/diagnostics` and are not installed.

The current runtime freeze and evidence are recorded in
`PRE_COLLECTION_RUNTIME_FINAL_V3.md` and
`pre_collection_runtime_final_v3.json`.  This current section supersedes the
historical `FORMAL_MPL_READY=NO` statement: the formal MPL and voxel cache
are now present at their canonical `data/` paths and have passed identity and
semantic checks.
